# WikiText-103 Perplexity Results

| Model | Params | Layers | Heads | Embed Dim | Vocab | Context | Perplexity | BPB |
|---|---|---|---|---|---|---|---|---|
| Custom (random weights) | 50.3M | 8 | 4 | 512 | 8192 | 2048 | 9919.16 | — |
| Custom (baseline) | 50.3M | 8 | 4 | 512 | 8192 | 2048 | 38.77 | 1.52 |
| Custom (iter 200) | 50.3M | 8 | 4 | 512 | 8192 | 2048 | 40.73 | 1.54 |
| GPT-2 (HuggingFace) | 124.4M | 12 | 12 | 768 | 50257 | 1024 | 24.98 | 1.02 |
