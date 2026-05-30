# Few-Shot Learning Analysis

Evaluating in-context learning ability of our models on structured few-shot prompts.

## Models Tested

| Checkpoint | Training Budget | Model Config | Params | Wikitext-2 PPL | BPB |
|------------|----------------|-------------|--------|----------------|------|
| 5_mins_baseline | 5 min | 8L / 4H / 512D | 50.3M | 21.59 | 1.2803 |
| iter_200 (5 min) | 5 min | 8L / 4H / 512D | 50.3M | 21.32 | 1.2751 |
| 10_mins_baseline | 10 min | 8L / 4H / 512D | 50.3M | 19.05 | 1.2283 |
| 10_mins_improvement | 10 min | 8L / 6H / 768D | 94.4M | 17.60 | 1.1952 |

Both 5-min models share the same architecture; iter_200 adds sliding window attention and logit softcapping. The 10-min baseline uses the same architecture as the 5-min baseline but trained for twice as long. The 10-min improvement is a larger model (94.4M params, 6 heads, 768D) trained for 10 minutes.

## Generation Settings

- `temperature=0.3`, `top_k=50`, `top_p=0.95`, `repetition_penalty=1.2`
- `max_new_tokens=50`, `device=mps`

## Task 1: Factual Q&A (World Capitals)

**Prompt:**
```
Q: What is the capital of France?
A: Paris

Q: What is the capital of Japan?
A: Tokyo

Q: What is the capital of Germany?
A:
```

| Model | Output | Format Followed | Correct |
|-------|--------|-----------------|---------|
| 5_mins_baseline | "The capital of Germany is the capital of France. It is located in the city of Titu..." | No (long-form) | No |
| iter_200 (5 min) | "Tokyo" | Yes (short Q&A) | No |
| 10_mins_baseline | "Toko" then continued with more Q&A pairs | Yes (short Q&A) | No |
| 10_mins_improvement | "Tokyo" then continued with more Q&A pairs | Yes (short Q&A) | No |

## Task 2: Sentiment Classification

**Prompt:**
```
Sentence: I love this movie, it was fantastic!
Sentiment: Positive

Sentence: The food was terrible and the service was slow.
Sentiment: Negative

Sentence: The weather is nice today and I feel great.
Sentiment:
```

| Model | Output | Format Followed | Correct |
|-------|--------|-----------------|---------|
| 5_mins_baseline | "The weather is cool and there's a lot of rain in the summer..." | No | No |
| iter_200 (5 min) | "It's a good time to get some fresh fruits and vegetables..." | No | No |
| 10_mins_baseline | "Wow! It's a funny thing to watch. I think that's what makes me happy." | No | No |
| 10_mins_improvement | "I think you're right. You are not going to be able to eat that fast food at home." | No | No |

## Task 3: English-to-French Translation

**Prompt:**
```
English: Hello, how are you?
French: Bonjour, comment allez-vous?

English: Thank you very much.
French: Merci beaucoup.

English: Where is the train station?
French:
```

| Model | Output | Format Followed | Correct |
|-------|--------|-----------------|---------|
| 5_mins_baseline | "I'm not sure if it's a train station or a train station..." | No | No |
| iter_200 (5 min) | "Freedom of speech" | No | No |
| 10_mins_baseline | "Galenais, France. The train station is located in the capital of France..." | No | No |
| 10_mins_improvement | "The train station is located in a small town called Santa Barbara..." | No | No |

## Task 4: Arithmetic

**Prompt:**
```
2 + 3 = 5
7 + 8 = 15
12 + 9 = 21
25 + 17 =
```

| Model | Output | Format Followed | Correct |
|-------|--------|-----------------|---------|
| 5_mins_baseline | 20 (then continued: 22, 30, 23, 34, 40, 45) | Yes | No (should be 42) |
| iter_200 (5 min) | 18 (then continued: 19, 20, 22, 24, 30, 31) | Yes | No (should be 42) |
| 10_mins_baseline | 20 (then continued: 10, 11, 30, 40, 14, 42) | Yes | No (should be 42) |
| 10_mins_improvement | 20 (then continued: 24, 29, 33, 40, 44, 45) | Yes | No (should be 42) |

## Summary

| Task | 5min Baseline | iter_200 | 10min Baseline | 10min Improvement |
|------|:---:|:---:|:---:|:---:|
| Q&A (capitals) — format | No | Yes | Yes | Yes |
| Q&A (capitals) — correct | No | No | No | No |
| Sentiment — format | No | No | No | No |
| Sentiment — correct | No | No | No | No |
| Translation — format | No | No | No | No |
| Translation — correct | No | No | No | No |
| Arithmetic — format | Yes | Yes | Yes | Yes |
| Arithmetic — correct | No | No | No | No |
| **Format score** | **1/4** | **2/4** | **2/4** | **2/4** |
| **Correctness score** | **0/4** | **0/4** | **0/4** | **0/4** |

## Key Takeaways

1. **No model can perform few-shot tasks correctly** — factual recall, classification, translation, and arithmetic all fail across all scales tested (50M–94M params).
2. **Format following improves with training time but plateaus** — the 5-min baseline only follows 1/4 formats, while all other models (iter_200, 10-min baseline, 10-min improvement) follow 2/4. The jump happens between 5-min baseline and the others, but scaling from 50M to 94M params doesn't unlock additional format following.
3. **Perplexity improves steadily across models** — PPL drops from 21.59 (5-min baseline) → 21.32 (iter_200) → 19.05 (10-min baseline) → 17.60 (10-min improvement), but this doesn't translate to better few-shot task performance.
4. **Perplexity ≠ few-shot ability at small scale** — the 10-min improvement model has 18% lower perplexity than the 5-min baseline, yet scores identically on few-shot tasks. This confirms that in-context learning is an emergent capability that requires a qualitative jump in model scale, not just incremental perplexity gains.
5. **All models handle arithmetic format** — they continue generating `X + Y = Z` pairs, but the actual answers are wrong. This is the easiest pattern to mimic (simple repeated structure).
6. **In-context learning requires much larger models** — reliable few-shot performance typically emerges at 1B+ parameters. At 50–94M, models can mimic surface-level patterns but lack the capacity for reasoning or factual knowledge.
