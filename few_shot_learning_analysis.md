# Few-Shot Learning Analysis

Evaluating in-context learning ability of our 50.3M param models on structured few-shot prompts.

## Models Tested

| Checkpoint | Training Budget | Model Config | Params | Wikitext-2 PPL | BPB |
|------------|----------------|-------------|--------|----------------|------|
| 5_mins_baseline | 5 min | 8L / 4H / 512D | 50.3M | 21.59 | 1.2803 |
| iter_200 (5 min) | 5 min | 8L / 4H / 512D | 50.3M | 21.32 | 1.2751 |
| 10_mins_baseline | 10 min | 8L / 4H / 512D | 50.3M | 19.05 | 1.2283 |

Both 5-min models share the same architecture; iter_200 adds sliding window attention and logit softcapping. The 10-min baseline uses the same architecture as the 5-min baseline but trained for twice as long.

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

## Summary

| Task | 5min Baseline Format | iter_200 Format | 10min Baseline Format | 5min Baseline Correct | iter_200 Correct | 10min Baseline Correct |
|------|---------------------|-----------------|----------------------|----------------------|------------------|----------------------|
| Q&A (capitals) | No | Yes | Yes | No | No | No |
| Sentiment | No | No | No | No | No | No |
| Translation | No | No | No | No | No | No |
| Arithmetic | Yes | Yes | Yes | No | No | No |

## Key Takeaways

1. **Neither model can perform few-shot tasks correctly** — factual recall, classification, translation, and arithmetic all fail at this scale (50M params).
2. **10-min baseline matches iter_200 on format following** — the 10-min baseline follows the Q&A format (short answers) just like iter_200, while the 5-min baseline reverted to long-form. This suggests that additional training time can achieve similar in-context pattern matching improvements as architectural changes (sliding window attention + logit softcapping).
3. **Perplexity improves with more training** — the 10-min baseline (PPL 19.05, BPB 1.2283) significantly outperforms both 5-min models (PPL ~21.3-21.6) on wikitext-2, confirming the model is undertrained at 5 minutes.
4. **All models handle arithmetic format** — they continue generating `X + Y = Z` pairs, but the actual answers are wrong.
5. **In-context learning requires larger models** — reliable few-shot performance typically emerges at 1B+ parameters. At 50M, the models can mimic surface-level patterns but lack the capacity for reasoning or factual knowledge.
6. **Training time vs architecture** — doubling training time (5 min → 10 min) yielded a larger perplexity improvement (~2.5 PPL points) than 200 iterations of architectural search within the same 5-min budget (~0.3 PPL points), underscoring that compute budget is the dominant factor at this scale.
