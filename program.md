# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
6. **Initialize scratchpad.md**: Create `scratchpad.md` with the template below. This is your persistent research notebook — do not commit it.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it simply as: `uv run train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, tokenizer, and training constants (time budget, sequence length, etc).
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.
- Modify the evaluation harness. The `evaluate_bpb` function in `prepare.py` is the ground truth metric.

**The goal is simple: get the lowest fitness score.** Fitness is a composite metric: `fitness = 0.5 * val_bpb + 0.5 * ood_bpb` (where ood_bpb is WikiText-2 BPB). This prevents overfitting to the in-distribution validation set. Since the time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**VRAM** is a soft constraint. Some increase is acceptable for meaningful val_bpb gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20 lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code? Definitely keep. An improvement of ~0 but much simpler code? Keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training script as is.

## Research strategy

You are not just running experiments mechanically — you are a researcher. Think before each experiment, learn from every result, and adapt your strategy over time.

### Phases

Roughly prioritize exploration in this order. You don't need to be rigid about it, but the early experiments should focus on high-leverage knobs before fine-tuning.

1. **Baseline & scaling** (experiments 1–10): Establish the baseline, then explore the highest-leverage knobs — model size (DEPTH, ASPECT_RATIO), learning rates, and batch size. These dominate val_bpb at this stage.
2. **Architecture & optimizer** (experiments 10–30): Try structural changes — activation functions, attention patterns, normalization, optimizer hyperparameters, schedule shapes. Keep what works from phase 1 as your foundation.
3. **Fine-tuning & combinations** (experiments 30+): Combine the best ideas from earlier phases. Sweep LR more finely around the best region. Try small ablations to simplify without hurting.

### Learn from your results

Before choosing each experiment, re-read `scratchpad.md` (your research notebook). Look for patterns:
- Which *categories* of changes tend to help? (e.g., "LR changes helped 3/4 times, activation swaps helped 0/3 times")
- Are you stuck in a local optimum? If the last 5+ experiments all failed to improve, shift strategy — try something structurally different rather than incremental tweaks.
- Did a recent discard *almost* work? Combine it with something else, or try a milder version.

### Isolate variables

When possible, change one thing at a time. If you change the model depth AND the learning rate simultaneously and it helps, you don't know which one mattered. When you do make multi-variable changes (e.g., for speed), follow up with ablations to understand why it worked.

### Avoid wasted experiments

- Don't retry an idea that already failed unless you have a specific reason to believe the context changed (e.g., you changed the model size since then).
- Don't make tiny tweaks (e.g., LR 0.04 → 0.041) unless you're in the fine-tuning phase and have strong signal that the optimum is nearby.
- If an architectural idea crashes with OOM, don't just discard it — estimate how much you need to shrink the model to fit, and try a scaled-down version.

## Scratchpad

Maintain a `scratchpad.md` file as your persistent research notebook. Do NOT commit it (like results.tsv). Update it after every experiment.

Template for the initial scratchpad:

```markdown
# Research Scratchpad

## Current best
- fitness: (filled after baseline)
- val_bpb: (filled after baseline)
- ood_bpb: (filled after baseline)
- commit: (filled after baseline)
- key config: depth=8, aspect_ratio=64, matrix_lr=0.04

## Learnings
(What you've discovered so far. Update after every experiment.)

## Ideas queue
(Prioritized list of things to try next. Add/remove/reorder as you learn.)

## Dead ends
(Ideas that definitively didn't work. Don't retry these without a clear reason.)
```

Consult the scratchpad at the start of every experiment iteration. Update "Learnings" after every result. Reorder "Ideas queue" as your understanding evolves. Move failed ideas to "Dead ends" with a short note on why.

## Research hints

These are optional starting points drawn from recent ML research and the nanochat lineage. Use your judgment — not all of these will help.

**High-leverage hyperparameters:**
- Learning rate is usually the single most impactful knob. Sweep it early (e.g., 2x and 0.5x the default).
- Warmup can stabilize training even with a small budget. Try 5–10% warmup ratio (default is 0%).
- Batch size interacts with LR — if you increase one, consider adjusting the other.

**Model scaling:**
- Try depth 6–12 and aspect ratios 48–96. Wider-but-shallower or deeper-but-narrower may trade off differently under a 5-minute budget.
- Smaller models train more steps in the same time budget, which can help if the model is currently undertrained.

**Architecture ideas:**
- SwiGLU / GeGLU activations (replace ReLU² in the MLP) — widely used in modern LLMs.
- Grouped query attention: try n_kv_head < n_head (e.g., 2 KV heads instead of 4) to save compute and reinvest in model size.
- The sliding window pattern "SSSL" is one option — try "LLLL" (all full attention) or "SSLL" or others. **Constraint: the pattern must contain at least one "L" (global attention layer).** Patterns like "SSSS" with no global attention degrade long-range modeling.

**Optimizer:**
- The Muon optimizer is relatively new. Its momentum schedule, NS steps, and interaction with weight decay may not be optimal.
- The per-parameter-group LR structure (embedding_lr, matrix_lr, scalar_lr, etc.) has many knobs — some may be far from optimal.

**Things that probably won't help:**
- Changing the tokenizer or vocab size (controlled by prepare.py, out of scope).
- Dropout or regularization — the model is undertrained in 5 minutes, not overfitting.

## Output format

Once the script finishes it prints a summary like this:

```
---
fitness:          0.997900
val_bpb:          0.985000
ood_bpb:          1.010800
training_seconds: 300.1
total_seconds:    325.9
peak_vram_mb:     45060.2
mfu_percent:      39.80
total_tokens_M:   499.6
num_steps:        953
num_params_M:     50.3
depth:            8
```

Note that the script is configured to always stop after 5 minutes, so depending on the computing platform of this computer the numbers might look different. You can extract the key metric from the log file:

```
grep "^fitness:" run.log
```

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	fitness	memory_gb	status	description
```

1. git commit hash (short, 7 chars)
2. fitness score achieved (e.g. 1.234567) — use 0.000000 for crashes
3. peak memory in GB, round to .1f (e.g. 12.3 — divide peak_vram_mb by 1024) — use 0.0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	fitness	memory_gb	status	description
a1b2c3d	0.997900	44.0	keep	baseline
b2c3d4e	0.993200	44.2	keep	increase LR to 0.04
c3d4e5f	1.005000	44.0	discard	switch to GeLU activation
d4e5f6g	0.000000	0.0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5` or `autoresearch/mar5-gpu0`).

LOOP FOREVER:

0. **Refresh certs**: Run `source refresh_claude_cert_path.sh` to refresh the cert files before each iteration.
1. **Plan**: Re-read `scratchpad.md`. Based on your learnings and ideas queue, pick the next experiment. Briefly reason about why you expect it to help.
2. Tune `train.py` with the experimental idea by directly hacking the code.
3. git commit
4. Run the experiment: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Read out the results: `grep "^fitness:\|^val_bpb:\|^ood_bpb:\|^peak_vram_mb:" run.log`
6. If the grep output is empty, the run crashed. Run `tail -n 50 run.log` to read the Python stack trace and attempt a fix. If you can't get things to work after more than a few attempts, give up.
7. Record the results in the tsv (NOTE: do not commit the results.tsv file, leave it untracked by git)
8. **Update scratchpad.md**: Record what you learned. Update the ideas queue. Move dead ends. Update "Current best" if applicable.
9. If fitness improved (lower), you "advance" the branch, keeping the git commit
10. If fitness is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind but you should probably do this very very sparingly (if ever).

**Timeout**: Each experiment should take ~5 minutes total (+ a few seconds for startup and eval overhead). If a run exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, or etc.), use your judgment: If it's something dumb and easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the tsv, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
