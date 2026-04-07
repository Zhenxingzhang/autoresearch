"""Evaluate HuggingFace GPT-2 perplexity on standard benchmarks (same methodology as eval_perplexity.py)."""

import argparse
import math
import time

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from datasets import load_dataset

DATASETS = {
    "wikitext-103": ("Salesforce/wikitext", "wikitext-103-raw-v1", "test"),
    "wikitext-2": ("Salesforce/wikitext", "wikitext-2-raw-v1", "test"),
}


def load_dataset_text(name):
    path, subset, split = DATASETS[name]
    print(f"Loading {name} ({path}/{subset}, split={split}) ...")
    ds = load_dataset(path, subset, split=split)
    text = "\n".join(row for row in ds["text"] if row.strip())
    print(f"  Loaded {len(text):,} characters ({len(text.split()):,} words)")
    return text


@torch.no_grad()
def evaluate(model, tokenizer, text, device, stride=None):
    model.eval()
    seq_len = model.config.n_positions

    tokens = tokenizer.encode(text, return_tensors="pt").squeeze(0).to(device)
    num_tokens = len(tokens)
    if num_tokens < 2:
        return {"perplexity": float("nan")}

    if stride is None:
        stride = seq_len // 2

    total_nll = 0.0
    total_tokens_scored = 0
    prev_end = 0
    N = len(tokens)
    t0 = time.time()

    for begin in range(0, N - 1, stride):
        end = min(begin + seq_len, N - 1)
        input_ids = tokens[begin:end].unsqueeze(0)
        target_ids = tokens[begin + 1:end + 1].unsqueeze(0)

        logits = model(input_ids).logits

        score_start = max(0, prev_end - begin)
        loss_all = F.cross_entropy(
            logits[0].view(-1, logits.size(-1)),
            target_ids[0].view(-1),
            reduction="none",
        )
        loss_scored = loss_all[score_start:]
        total_nll += loss_scored.sum().item()
        total_tokens_scored += loss_scored.numel()
        prev_end = end

        if end == N - 1:
            break

    elapsed = time.time() - t0
    avg_nll = total_nll / total_tokens_scored
    perplexity = math.exp(avg_nll)

    # BPB
    text_bytes = len(text.encode("utf-8"))
    bpb = total_nll / (math.log(2) * text_bytes) if text_bytes > 0 else float("nan")

    return {
        "perplexity": perplexity,
        "bpb": bpb,
        "avg_nll_nats": avg_nll,
        "num_tokens_scored": total_tokens_scored,
        "num_tokens_total": num_tokens,
        "total_time_s": elapsed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="gpt2", help="HF model name (gpt2, gpt2-medium, gpt2-large, gpt2-xl)")
    parser.add_argument("--dataset", type=str, default="wikitext-103", choices=list(DATASETS.keys()))
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-chars", type=int, default=None)
    args = parser.parse_args()

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Device: {device}")

    print(f"Loading {args.model} from HuggingFace ...")
    tokenizer = GPT2TokenizerFast.from_pretrained(args.model)
    model = GPT2LMHeadModel.from_pretrained(args.model).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} — {num_params / 1e6:.1f}M params")
    print(f"Context window: {model.config.n_positions} tokens")
    print()

    text = load_dataset_text(args.dataset)
    if args.max_chars and len(text) > args.max_chars:
        text = text[:args.max_chars]
        print(f"  Truncated to {args.max_chars:,} characters")

    print("\nEvaluating ...")
    results = evaluate(model, tokenizer, text, device, stride=args.stride)

    print()
    print("=" * 50)
    print("  RESULTS")
    print("=" * 50)
    print(f"  Perplexity:       {results['perplexity']:.2f}")
    print(f"  BPB:              {results['bpb']:.4f}")
    print(f"  Avg NLL (nats):   {results['avg_nll_nats']:.4f}")
    print(f"  Tokens scored:    {results['num_tokens_scored']:,} / {results['num_tokens_total']:,}")
    print(f"  Time:             {results['total_time_s']:.1f}s")
    print(f"  Throughput:       {results['num_tokens_scored'] / results['total_time_s']:.0f} tok/s")
    print("=" * 50)


if __name__ == "__main__":
    main()
