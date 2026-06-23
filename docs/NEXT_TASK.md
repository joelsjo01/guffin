# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first scheduled Gemma attention probability setup primitive behind the
Metal runtime shell.

## Why

Gemma now reaches the real generate path far enough to load file-backed tensor
views, tokenize the prompt, upload the resident Metal model plan, build
separate prefill/decode schedules, turn prompt token IDs into backend-resident
hidden states with the Gemma embedding scale, normalize the first-layer
attention input with `blk.0.attn_norm.weight`, and produce explicit
backend-resident Q/K/V projection outputs for the first prefill layer. The
runtime shell also applies Gemma position/RoPE setup to the first-layer Q/K
projection outputs, keeps the positioned Q/K outputs explicit, and produces
explicit pre-softmax attention scores for the first layer with Gemma scaling,
softcap, GQA head mapping, and causal/window masking.

The next blocker is proving those scheduled score outputs can flow into
attention probabilities without breaking the resident model plan. Keep this
slice tiny and replaceable: enough to produce explicit softmax probability
outputs for the first prefill layer, but not enough to aggregate values or
pretend full attention or generation exists.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`, especially
  `## Current Inference Path And Bottlenecks` and
  `## Target Architecture Proposal`

## Touch

- Gemma engine/runtime planning code under `include/local_gguf_engine/engine`
  and `src/engine`
- Metal backend/tensor code only for narrow backend-buffer probability helpers
  and tests
- `tests/engine`
- `tests/tensor` only if adding or exposing a backend primitive
- `tests/frontends/cli/generate_cli_tests.cpp` only if the unsupported Gemma
  execution boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not add full Gemma generation or a full Gemma layer chain.
- Do not add KV-cache update, value aggregation, attention output projection,
  FFN work, logits, or sampling changes.
- Do not add full 9B CPU execution, broad graph abstractions, or benchmark
  work in this slice.
- Do not dequantize whole Gemma weight tensors into persistent F16/F32 buffers.
- Do not replace the resident plan with per-op host round trips.
- Do not make `local-gguf-generate` claim Gemma success yet.
- Do not read benchmark artifacts.

## Verify

- Add deterministic tests for primitive shape/resource ownership and CPU/Metal
  parity when a tensor primitive is introduced.
- Keep unavailable-Metal behavior explicit if the primitive creates Metal
  resources directly.
- Keep the real Gemma CLI failure as explicit unsupported Gemma execution until
  enough scheduled execution exists to return logits.
- Run focused engine/tensor/CLI tests touched by the primitive.
- Run local Gemma inspection/generation far enough to prove any remaining
  failure is still explicit unsupported Gemma execution.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

One small scheduled Gemma attention-probability setup primitive consumes the
runtime shell's first-layer score outputs and produces an explicit resident or
phase-local softmax probability output with deterministic tests, while full
Gemma generation remains explicitly unsupported.
