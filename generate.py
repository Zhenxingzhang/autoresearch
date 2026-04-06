"""
Text generation script. Loads a checkpoint and generates text.
Usage: python generate.py --prompt "Once upon a time"
       CKPT_DIR=/path/to/checkpoints python generate.py --prompt "Hello"
"""

import argparse
import math
import os
import pickle

import torch
import torch.nn.functional as F

from model import GPTConfig, GPT, Tokenizer


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
