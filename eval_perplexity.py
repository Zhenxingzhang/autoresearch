"""
Evaluate perplexity (and BPB) of the custom GPT model on standard benchmarks.

Usage:
    # Evaluate on WikiText-103 test set (default)
    python eval_perplexity.py

    # Evaluate on a specific dataset
    python eval_perplexity.py --dataset wikitext-2

    # Evaluate on a custom text file
    python eval_perplexity.py --text-file /path/to/file.txt

    # Use random weights as baseline
    python eval_perplexity.py --random-weights

    # Specify device
    python eval_perplexity.py --device mps
"""

import argparse
import math
import os
import pickle
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model definition (inlined from generate.py to avoid side-effects)
# ---------------------------------------------------------------------------

BOS_TOKEN = "<|reserved_0|>"


class Tokenizer:
    def __init__(self, enc):
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(BOS_TOKEN)

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_bos_token_id(self):
        return self.bos_token_id

    def encode(self, text):
        return self.enc.encode_ordinary(text)

    def decode(self, ids):
        return self.enc.decode(ids)


def norm(x):
    return F.rms_norm(x, (x.size(-1),))


def has_ve(layer_idx, n_layer):
    return layer_idx % 2 == (n_layer - 1) % 2


def apply_rotary_emb(x, cos, sin):
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6
    n_kv_head: int = 6
    n_embd: int = 768
    window_pattern: str = "SSSL"


class CausalSelfAttention(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        self.c_q = torch.nn.Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = torch.nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = torch.nn.Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = torch.nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 32
        self.ve_gate = (
            torch.nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        return self.c_proj(F.relu(self.c_fc(x)).square())


class Block(torch.nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin):
        x = x + self.attn(norm(x), ve, cos_sin)
        x = x + self.mlp(norm(x))
        return x


class GPT(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = torch.nn.ModuleDict(
            {
                "wte": torch.nn.Embedding(config.vocab_size, config.n_embd),
                "h": torch.nn.ModuleList(
                    [Block(config, i) for i in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = torch.nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = torch.nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = torch.nn.ModuleDict(
            {
                str(i): torch.nn.Embedding(config.vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        cos, sin = self._precompute_rotary_embeddings(config.sequence_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = "cpu"
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        dtype = (
            torch.bfloat16
            if device == "cpu" or (torch.cuda.is_available() and device != "mps")
            else torch.float16
        )
        cos, sin = cos.to(dtype), sin.to(dtype)
        return cos[None, :, None, :], sin[None, :, None, :]

    def forward(self, idx):
        B, T = idx.size()
        cos_sin = self.cos[:, :T], self.sin[:, :T]
        x = self.transformer.wte(idx)
        x = norm(x)
        x0 = x
        for i, block in enumerate(self.transformer.h):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin)
        x = norm(x)
        return self.lm_head(x)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

DATASETS = {
    "wikitext-103": ("wikitext", "wikitext-103-raw-v1", "test"),
    "wikitext-2": ("wikitext", "wikitext-2-raw-v1", "test"),
    "lambada": ("EleutherAI/lambada_openai", "default", "test"),
    "ptb": ("ptb_text_only", "penn_treebank", "test"),
}


def load_dataset_text(name: str) -> str:
    """Load a standard evaluation dataset and return its text."""
    from datasets import load_dataset

    if name not in DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {list(DATASETS.keys())}")

    path, subset, split = DATASETS[name]
    print(f"Loading {name} ({path}/{subset}, split={split}) ...")
    ds = load_dataset(path, subset, split=split, trust_remote_code=True)

    # Concatenate all text fields
    if "text" in ds.column_names:
        text = "\n".join(row for row in ds["text"] if row.strip())
    elif "content" in ds.column_names:
        text = "\n".join(row for row in ds["content"] if row.strip())
    else:
        raise ValueError(f"Dataset has no 'text' or 'content' column: {ds.column_names}")

    print(f"  Loaded {len(text):,} characters ({len(text.split()):,} words)")
    return text


def load_text_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, tokenizer, text: str, device: str, stride: int | None = None):
    """
    Compute perplexity and BPB using a sliding-window approach.

    For texts longer than the model's context window, we use a stride-based
    method: slide a window of size `sequence_len` across the tokens, only
    counting loss on the new (non-overlapping) tokens in each window. This
    gives an accurate perplexity without truncating the evaluation text.

    Returns dict with: perplexity, bpb, num_tokens, num_bytes, total_time_s
    """
    model.eval()
    seq_len = model.config.sequence_len

    # Tokenize
    token_ids = tokenizer.encode(text)
    num_tokens = len(token_ids)
    if num_tokens < 2:
        return {"perplexity": float("nan"), "bpb": float("nan")}

    # Prepend BOS
    tokens = [tokenizer.get_bos_token_id()] + token_ids
    tokens_t = torch.tensor(tokens, dtype=torch.long, device=device)

    if stride is None:
        stride = seq_len // 2  # 50% overlap by default

    total_nll = 0.0  # negative log-likelihood (nats)
    total_tokens_scored = 0
    total_nats_for_bpb = 0.0
    total_bytes = 0

    t0 = time.time()

    # Sliding window evaluation
    # tokens has length N+1 (BOS + N real tokens). We predict tokens[1:].
    N = len(tokens)
    prev_end = 0

    for begin in range(0, N - 1, stride):
        end = min(begin + seq_len, N - 1)
        # Input:  tokens[begin : end]
        # Target: tokens[begin+1 : end+1]
        input_ids = tokens_t[begin:end].unsqueeze(0)
        target_ids = tokens_t[begin + 1 : end + 1].unsqueeze(0)

        logits = model(input_ids)

        # Only score tokens that haven't been scored yet (avoid double-counting)
        # For the first window, score all; for subsequent, score from stride offset
        score_start = max(0, prev_end - begin)
        loss_all = F.cross_entropy(
            logits[0].view(-1, logits.size(-1)),
            target_ids[0].view(-1),
            reduction="none",
        )
        loss_scored = loss_all[score_start:]

        total_nll += loss_scored.sum().item()
        total_tokens_scored += loss_scored.numel()

        # BPB: weight by byte length of each scored target token
        for i in range(score_start, end - begin):
            tid = tokens[begin + 1 + i]
            token_str = tokenizer.decode([tid])
            nbytes = len(token_str.encode("utf-8"))
            if nbytes > 0:
                total_nats_for_bpb += loss_all[i].item()
                total_bytes += nbytes

        prev_end = end

        if end == N - 1:
            break

    elapsed = time.time() - t0

    avg_nll = total_nll / total_tokens_scored
    perplexity = math.exp(avg_nll)
    bpb = total_nats_for_bpb / (math.log(2) * total_bytes) if total_bytes > 0 else float("nan")

    return {
        "perplexity": perplexity,
        "bpb": bpb,
        "avg_nll_nats": avg_nll,
        "num_tokens_scored": total_tokens_scored,
        "num_tokens_total": num_tokens,
        "num_bytes": total_bytes,
        "total_time_s": elapsed,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of the custom GPT model")
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext-2",
        choices=list(DATASETS.keys()),
        help="Standard benchmark dataset (default: wikitext-2)",
    )
    parser.add_argument("--text-file", type=str, default=None, help="Evaluate on a custom text file instead")
    parser.add_argument("--stride", type=int, default=None, help="Sliding window stride (default: seq_len // 2)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: auto-detect)")
    parser.add_argument("--random-weights", action="store_true", help="Use random weights as baseline")
    parser.add_argument("--max-chars", type=int, default=None, help="Truncate eval text to N characters (for quick testing)")
    args = parser.parse_args()

    # Device selection
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Load checkpoint
    default_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    ckpt_dir = os.environ.get("CKPT_DIR", default_ckpt_dir)

    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.pkl")
    print(f"Loading tokenizer from {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        enc = pickle.load(f)
    tokenizer = Tokenizer(enc)

    if args.random_weights:
        print("Using randomly initialized weights (baseline)")
        config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
        model = GPT(config)
    else:
        ckpt_path = os.path.join(ckpt_dir, "model.pt")
        print(f"Loading model from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config)
        state_dict = ckpt["model"]
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {config.n_layer}L / {config.n_head}H / {config.n_embd}D — {num_params / 1e6:.1f}M params")
    print(f"Context window: {config.sequence_len} tokens")
    print()

    # Load text
    if args.text_file:
        print(f"Loading text from {args.text_file}")
        text = load_text_file(args.text_file)
    else:
        text = load_dataset_text(args.dataset)

    if args.max_chars and len(text) > args.max_chars:
        text = text[: args.max_chars]
        print(f"  Truncated to {args.max_chars:,} characters")

    # Evaluate
    print()
    print("Evaluating ...")
    results = evaluate(model, tokenizer, text, device, stride=args.stride)

    # Report
    print()
    print("=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  Perplexity:       {results['perplexity']:.2f}")
    print(f"  BPB:              {results['bpb']:.4f}")
    print(f"  Avg NLL (nats):   {results['avg_nll_nats']:.4f}")
    print(f"  Tokens scored:    {results['num_tokens_scored']:,} / {results['num_tokens_total']:,}")
    print(f"  Bytes evaluated:  {results['num_bytes']:,}")
    print(f"  Time:             {results['total_time_s']:.1f}s")
    print(f"  Throughput:       {results['num_tokens_scored'] / results['total_time_s']:.0f} tok/s")
    print("=" * 50)


if __name__ == "__main__":
    main()
