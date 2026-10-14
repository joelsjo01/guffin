# Pinned llama.cpp reference baseline

This is the historical August planning reference for the two-turn development
benchmark. Its identity, protocol and recorded result are preserved below; old
local output bundles are no longer retained. The October
[final release campaigns](PERFORMANCE.md#final-release-results) use a different
llama.cpp executable and protocol.

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

The driver verifies the pinned commit and its Homebrew installation receipt;
it rejects the newer llama-cli used by the release campaigns. Use `--reference`
with the pinned installation if it is available.

The stored result is invalid as a direct comparator if the model, llama.cpp
commit, macOS, hardware, prompts, sampling settings or timing definition changes.
A replacement requires its own matched comparison and record. Small claimed
gains also require a contemporaneous comparison rather than this August result.
