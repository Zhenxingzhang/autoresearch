# Overfitting Analysis: Automated Research Loop

## How the Problem Was Spotted

We evaluated three checkpoints produced by the automated research loop on the **WikiText-2** benchmark using `eval_perplexity.py`. The results contradicted expectations set by the training validation BPB:

### Training Validation BPB (climbmix, in-distribution)

| Checkpoint | val_bpb |
|---|---|
| baseline | 0.995 |
| iter_50 | — |
| iter_100 | 0.972 |

Lower is better — iter_100 appeared to be the best model.

### WikiText-2 Evaluation (out-of-distribution)

| Metric | baseline | iter_50 | iter_100 |
|---|---|---|---|
| **Architecture** | 8L / 4H / 512D | 9L / 5H / 640D | 9L / 5H / 640D |
| **Params** | 50.3M | 80.9M | 80.9M |
| **Perplexity** | **38.77** | 55.00 | 106.65 |
| **BPB** | **1.5244** | 1.6701 | 1.9461 |

The baseline (smaller model) significantly outperformed both iterated checkpoints on the held-out benchmark, despite having a *worse* training validation BPB.

### Text Generation Quality

We generated text with the prompt "The meaning of life is" using `generate.py`:

- **baseline**: Produced coherent, varied text with no repetition loops
- **iter_50**: Fell into repetition loops after a few sentences ("Life is a system of physical and biological processes..." repeated verbatim)
- **iter_100**: Also degenerated into loops ("as simple as a tennis court or as complex as a pizza oven" repeated verbatim)

The repetitive degeneration is a hallmark of models that have memorized narrow patterns rather than learning general language structure.

## Root Cause Analysis

### The search loop optimizes the wrong metric

The research loop (defined in `program.md`) uses `val_bpb` as its sole fitness function for keep/discard decisions. This validation BPB is computed on a held-out shard (`shard_06542.parquet`) from the **same distribution** as the training data (climbmix). A model can score well on this metric by fitting climbmix-specific patterns without learning generalizable language modeling.

### Insufficient training data relative to model size

The setup downloads only **10 shards** out of 6,542 available in the climbmix dataset. With a 5-minute training budget, the search grew the model from 50.3M to 80.9M parameters. A larger model trained on limited data for a fixed time budget is a recipe for overfitting — the model has enough capacity to memorize training-distribution patterns rather than learning general ones.

### No regularization

The training configuration has:
- No dropout anywhere (attention, MLP, embeddings)
- Weight decay only on Muon optimizer params (0.2, decaying to 0)
- No explicit regularization beyond logit softcapping at 15

### Architecture changes may have hurt generalization

The iter_100 checkpoint changed `window_pattern` from `"SSSL"` to `"SSS"`, removing the long-range attention window entirely. This could make the model more prone to local repetition patterns, as it can only attend to nearby tokens.

## Discussion

### Lower BPB does not always mean a better model

This is the central takeaway. BPB (Bits Per Byte) is a valid and useful metric, but **only when measured on data the model has never seen from a different distribution**. When measured on in-distribution validation data, a lower BPB may simply indicate better memorization.

The divergence between in-distribution and out-of-distribution performance is the textbook definition of overfitting:

```
Training val_bpb:  baseline (0.995) > iter_100 (0.972)  — iter_100 "wins"
WikiText-2 BPB:    baseline (1.524) < iter_100 (1.946)  — baseline wins
```

### The search process has a blind spot

The automated research loop runs ~12 experiments per hour, each time modifying `train.py` and keeping changes that lower `val_bpb`. Over 100 iterations, this optimization pressure found configurations that fit climbmix well (bigger model, different attention patterns) but degraded generalization. The search had no signal telling it that generalization was getting worse.

## Proposed Solutions

### 1. Add out-of-distribution evaluation to the fitness function (highest impact)

Incorporate a WikiText-2 (or similar) BPB check into the keep/discard decision. The research loop should compute both:
- `val_bpb`: in-distribution performance (climbmix)
- `ood_bpb`: out-of-distribution performance (WikiText-2)

A combined metric like `score = 0.5 * val_bpb + 0.5 * ood_bpb` or a hard constraint (`reject if ood_bpb > threshold`) would prevent the search from drifting toward overfitting. This could be implemented by adding a quick WikiText-2 eval at the end of `train.py` or as a post-training check in the research loop.

### 2. Increase training data volume (easiest change)

Currently only 10 out of 6,542 available shards are downloaded. Increasing to 100–500 shards makes overfitting in 5 minutes much harder, especially for models in the 50–100M parameter range. This is likely a one-line change in `prepare.py`'s download configuration.

### 3. Add regularization to the model

Introduce dropout to reduce memorization:
- Attention dropout: 0.05–0.1
- MLP dropout: 0.05–0.1
- Embedding dropout: 0.05

These are standard for transformer pretraining and add negligible overhead.

### 4. Penalize model size growth in the search

The search freely grew the model from 50.3M to 80.9M. Adding a size penalty to the fitness function would discourage capacity increases that don't yield proportional generalization gains:

```
score = val_bpb + alpha * max(0, params_M - baseline_params_M)
```

Alternatively, cap the model size explicitly (e.g., max 60M params for 10-shard training).

### 5. Enforce minimum tokens-per-parameter ratio

A rule of thumb: require at least 20 tokens seen per parameter during training. With 10 shards and a 5-minute budget, this naturally constrains model size. This can be enforced in `program.md` as a constraint the research agent must respect.

## Recommended Action Plan

1. **Immediate**: Increase data shards from 10 to 100+ (prevents data-side overfitting)
2. **Short-term**: Add WikiText-2 BPB as a secondary metric in the research loop
3. **Optional**: Add lightweight dropout (0.05) as a default in `train.py`

These three changes together address the root causes: insufficient data diversity, a blind fitness metric, and lack of regularization.
