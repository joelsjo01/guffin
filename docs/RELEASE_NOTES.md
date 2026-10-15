# Guffin v0.1.0

Guffin is a local C++20/Metal GGUF engine for Gemma chat. The tested setup is a
16 GiB MacBook Air M4, macOS 26.6.2 and Apple clang 21. The executable owns
inference and embeds its Metal source; it needs no separate inference runtime,
Python environment or network connection.

## Included

- Gemma 3 12B IT Q4_K_M, with native quantized weights and F16 KV.
- Optional Gemma 3 270M Q8_0 drafting, using H1, paired target verification and
  lossless sampling. Lossless preserves the configured target distribution;
  it does not require identical seeded answers.
- Gemma 2 9B IT Q4_K_M, with default suffix retrieval, fused prompt attention
  and explicit weight conversions.
- Installed `guffin` chat, streaming, conversation reuse, interruption, one-shot
  and piped output, timing metrics and GGUF inspection.

Install with `make -j4 install`. See the [quick start](../README.md#quick-start)
and [support contract](../README.md#everyday-use-and-support) for exact models,
commands and context limits. Model weights are downloaded separately.

## Evidence and limits

The three retained campaigns each cover 38 conversations and 80 turns per
engine. The [performance record](PERFORMANCE.md) links the complete results,
model and executable identities, timing definitions and comparison limits.
They compare each engine's chat defaults; changing speculative settings can
change the comparison. These are single-pass performance measurements, without
an answer-quality evaluation.

The [numerical boundary](PERFORMANCE.md#numerical-limits) remains unresolved:
Gemma 3 F16 KV exceeds the original independent full-model bounds on a
reproduced 128-token prompt; the F32 control passes. Lossless speculation
preserves the native target's law, including its precision. Wider-context
fidelity and other hardware remain unvalidated. Gemma 4 is not supported.

Project code is [MIT licensed](../LICENSE). MT-Bench fixtures retain their
[Apache-2.0 attribution](../benchmarks/chat/README.md#attribution); model weights
have their own terms.

The release tag is `v0.1.0`. `guffin --version` reports the compiled build
identity, normally the source commit with an optional `-dirty` suffix. When
building a source archive without Git metadata, use
`make -j4 install ENGINE_BUILD=v0.1.0` to retain the release identity.
