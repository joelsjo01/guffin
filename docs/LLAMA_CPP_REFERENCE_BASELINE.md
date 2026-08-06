> **Rerun notice:** Rerun llama.cpp only if the model, llama.cpp commit, macOS, hardware, benchmark prompts, sampling settings, or timing definitions change.

# Pinned llama.cpp Reference Baseline

This is the stored external reference for the standard local benchmark. It is
not run during ordinary local optimization work.

## Reference identity

- Date: 2026-08-06
- Hardware: MacBook Air, Apple M4
- macOS: 26.5.2 (25F84)
- Model: `models/gemma-2-9b-Q4.gguf`
- llama.cpp: `d48a56effbba99944a677938d4beb63a0065ecdf`
- llama-cli version: `9430 (d48a56eff)`
- Sampling: normal interactive defaults, seed 42
- Startup warm-up: llama-cli default enabled
- Stream mode: captured pipes
- Inter-turn pause: 5 seconds, excluded from timings
- Repetitions: one fresh process

Prompts:

1. `Hello, is AI getting smarter?`
2. `Okay, so will it always progress this fast?`

Both responses ran to their natural stop without a response-token cap.

## Stored result

| Turn | Evaluated | Cached | Sampled | TTFT | Prompt t/s | Generation t/s | Prompt duration | Generation duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| First | 16 | 0 | 239 | 0.4045 s | 39.6 | 14.1 | 0.4045 s | 16.9686 s |
| Follow-up | 263 | 11 | 257 | 1.9503 s | 134.8 | 13.9 | 1.9503 s | 18.5227 s |

The rates use llama.cpp's `result_timings`: prompt processing ends at the
first sampled token, and generation spans the first sampled token through the
final sampled token while counting every sampled token.

## Rerun command

```sh
make reference-benchmark
scripts/llama_reference_benchmark.py --model models/gemma-2-9b-Q4.gguf
```

This stored run is a stable planning reference. Re-run it alongside local when
making a claim smaller than ordinary system or thermal variation.
