# Next Task

Use this as the implementation baton. Gates 1 through 5 of
`docs/CLI_MILESTONES.md` are complete; do not redo them. Keep this file short
and update it after the final gate.

## Objective

Implement Gate 6 from `docs/CLI_MILESTONES.md`: a matched, honest llama.cpp
Metal parity comparison for the supported Gemma chat application.

## Current State

- Persistent streaming chat, REPL commands, context trimming/rebuilds, greedy
  generation, and seeded sampled generation are complete.
- Greedy keeps Metal top-1 selection and one selected-token readback. Sampled
  mode reads back final vocabulary logits for engine-owned CPU filtering/RNG;
  model execution and KV remain Metal-resident.
- `/regen` restores pre-turn sampler state and exactly replays seeded sampled
  output.
- Gate 5 CI passes with 297 tests.

## Implement

- Compare against a current llama.cpp Metal interactive run using the same GGUF
  model, Gemma chat template, prompt turns, context length, stop conditions,
  sampler controls, seed, and relevant backend settings.
- Capture TTFT, prompt/prefill tokens/s, decode tokens/s, total tokens/s,
  memory notes, commands, transcript, and output differences.
- Classify differences as tokenizer/template, sampler, context policy, runtime
  performance, or known feature gaps.
- Keep benchmark artifacts under ignored `benchmarks/results`.

## Do Not

- Do not change runtime behavior merely to force matching visible text.
- Do not call the paths equivalent without matching tokenization and sampler
  semantics.
- Do not wrap llama.cpp as production inference.
- Do not commit model weights or benchmark result output.

## Verify

- Rebuild release binaries before measurement.
- Run both implementations from a normal Terminal with Metal available.
- Record exact commands and enough setup detail to reproduce the comparison.
- Run `bash scripts/ci.sh` if implementation or tracked behavior changes.

## Stop When

The comparison is reproducible, differences are honestly classified, and the
supported Gemma path can be assessed as a llama.cpp-style local interactive
application. Stop after reporting the final frontend gaps and results.
