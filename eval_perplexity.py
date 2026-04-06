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

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT, Tokenizer


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
