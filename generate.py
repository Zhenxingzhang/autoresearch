"""
Text generation script. Loads a checkpoint and generates text.
Usage: python generate.py --prompt "Once upon a time"
       CKPT_DIR=/path/to/checkpoints python generate.py --prompt "Hello"
"""

import argparse
import math
import os
import pickle
from dataclasses import dataclass

import torch
import torch.nn.functional as F


BOS_TOKEN = "<|reserved_0|>"


class Tokenizer:
    """Minimal tokenizer wrapper (self-contained, no prepare.py dependency)."""

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

# Re-import model definition from train.py without running the training loop.
# We inline the minimal model code to avoid side effects from train.py's top-level execution.
# Instead, we reconstruct GPTConfig and GPT from the saved config dict.


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
        self.ve_gate = torch.nn.Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin):
        B, T, C = x.size()
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 2 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))
            v = v + gate.unsqueeze(-1) * ve

        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k)

        # Use scaled_dot_product_attention for inference (no FA3 dependency)
        q = q.transpose(1, 2)  # (B, n_head, T, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Expand KV heads if grouped query attention
        if self.n_kv_head < self.n_head:
            reps = self.n_head // self.n_kv_head
            k = k.repeat_interleave(reps, dim=1)
            v = v.repeat_interleave(reps, dim=1)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = torch.nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = torch.nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


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
        self.transformer = torch.nn.ModuleDict({
            "wte": torch.nn.Embedding(config.vocab_size, config.n_embd),
            "h": torch.nn.ModuleList([Block(config, i) for i in range(config.n_layer)]),
        })
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.resid_lambdas = torch.nn.Parameter(torch.ones(config.n_layer))
        self.x0_lambdas = torch.nn.Parameter(torch.zeros(config.n_layer))
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = torch.nn.ModuleDict({
            str(i): torch.nn.Embedding(config.vocab_size, kv_dim)
            for i in range(config.n_layer) if has_ve(i, config.n_layer)
        })
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
        # bfloat16 not fully supported on MPS; use float16 as fallback
        dtype = torch.bfloat16 if device == "cpu" or (torch.cuda.is_available() and device != "mps") else torch.float16
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


@torch.no_grad()
def evaluate_bpb(model, tokenizer, text, device="cpu"):
    """Compute bits per byte on a given text string."""
    model.eval()
    token_ids = tokenizer.encode(text)
    if len(token_ids) < 2:
        return float("nan")
    tokens = [tokenizer.get_bos_token_id()] + token_ids
    x = torch.tensor([tokens[:-1]], dtype=torch.long, device=device)
    targets = torch.tensor([tokens[1:]], dtype=torch.long, device=device)

    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction="none")

    # Compute byte length per target token
    total_nats = 0.0
    total_bytes = 0
    for i, tid in enumerate(tokens[1:]):
        token_str = tokenizer.decode([tid])
        nbytes = len(token_str.encode("utf-8"))
        if nbytes > 0:
            total_nats += loss[i].item()
            total_bytes += nbytes

    if total_bytes == 0:
        return float("nan")
    return total_nats / (math.log(2) * total_bytes)


@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8,
             top_k=50, top_p=0.95, repetition_penalty=1.2, frequency_penalty=0.0,
             device="cpu"):
    model.eval()
    token_ids = tokenizer.encode(prompt)
    tokens = [tokenizer.get_bos_token_id()] + token_ids
    x = torch.tensor([tokens], dtype=torch.long, device=device)
    seq_len = model.config.sequence_len
    token_counts = torch.zeros(model.config.vocab_size, dtype=torch.long, device=device)
    for t in tokens:
        token_counts[t] += 1

    for _ in range(max_new_tokens):
        # Crop to sequence length if needed
        x_cond = x if x.size(1) <= seq_len else x[:, -seq_len:]
        logits = model(x_cond)
        logits = logits[:, -1, :]

        # Repetition penalty: penalize tokens that appeared in the sequence
        if repetition_penalty != 1.0:
            generated_tokens = x[0].unique()
            penalty_logits = logits[0, generated_tokens]
            # Shrink positive logits, magnify negative logits
            logits[0, generated_tokens] = torch.where(
                penalty_logits > 0,
                penalty_logits / repetition_penalty,
                penalty_logits * repetition_penalty,
            )

        # Frequency penalty: scale by how often each token was used
        if frequency_penalty > 0.0:
            logits[0] -= frequency_penalty * token_counts.float()

        logits = logits / temperature

        # Top-k filtering
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            # Remove tokens with cumulative probability above the threshold
            sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
            sorted_logits[sorted_mask] = -float("inf")
            logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_token], dim=1)
        token_counts[next_token.item()] += 1

    output_ids = x[0].tolist()
    # Skip the prepended BOS token
    return tokenizer.decode(output_ids[1:])


def main():
    parser = argparse.ArgumentParser(description="Generate text from a trained model")
    parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Input prompt")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering (0 to disable)")
    parser.add_argument("--top_p", type=float, default=0.95, help="Top-p nucleus filtering (1.0 to disable)")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="Repetition penalty (1.0 to disable)")
    parser.add_argument("--frequency_penalty", type=float, default=0.0, help="Frequency penalty (0.0 to disable)")
    parser.add_argument("--device", type=str, default=None, help="Device (default: cuda if available)")
    parser.add_argument("--random_weights", action="store_true", help="Use randomly initialized weights instead of loading a checkpoint")
    parser.add_argument("--eval", type=str, default=None, metavar="TEXT", help="Compute BPB on the given text instead of generating")
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Default checkpoint directory
    default_ckpt_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
    ckpt_dir = os.environ.get("CKPT_DIR", default_ckpt_dir)

    # Load tokenizer
    tokenizer_path = os.path.join(ckpt_dir, "tokenizer.pkl")
    print(f"Loading tokenizer from {tokenizer_path}")
    with open(tokenizer_path, "rb") as f:
        enc = pickle.load(f)
    tokenizer = Tokenizer(enc)

    # Load checkpoint or use random weights
    if args.random_weights:
        print("Using randomly initialized weights")
        config = GPTConfig(vocab_size=tokenizer.get_vocab_size())
        model = GPT(config)
    else:
        ckpt_path = os.path.join(ckpt_dir, "model.pt")
        print(f"Loading checkpoint from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        model = GPT(config)
        state_dict = ckpt["model"]
        # Strip _orig_mod. prefix added by torch.compile
        if any(k.startswith("_orig_mod.") for k in state_dict):
            state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
        model.load_state_dict(state_dict)
    model = model.to(device)

    # Evaluate BPB or generate text
    if args.eval:
        bpb = evaluate_bpb(model, tokenizer, args.eval, device)
        print(f"Text: {args.eval[:80]}{'...' if len(args.eval) > 80 else ''}")
        print(f"BPB: {bpb:.4f}")
    else:
        print(f"Prompt: {args.prompt}")
        print("---")
        output = generate(model, tokenizer, args.prompt, args.max_new_tokens,
                          args.temperature, args.top_k, args.top_p,
                          args.repetition_penalty, args.frequency_penalty, device)
        print(output)


if __name__ == "__main__":
    main()
