# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first Gemma generation runtime shell that builds the Metal-resident
plan plus prefill/decode schedules, then stops before executing Gemma layers.

## Why

Gemma now has file-backed loading, a Metal-resident model plan, first packed
Q4_K/Q6_K matmul kernels, and distinct resident-plan schedules for prompt
prefill and one-token decode. The CLI still rejects Gemma after metadata,
tokenizer, and weight binding. The next blocker is routing the real Gemma
generate path far enough to construct the resident Metal plan and phase
schedules for the encoded prompt, while keeping the actual layer execution
explicitly unsupported.

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
- Generation CLI only to route Gemma to the runtime shell and preserve the
  explicit unsupported execution error
- `tests/engine`
- `tests/frontends/cli/generate_cli_tests.cpp`
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not expand into unrelated cleanup.
- Do not optimize the old CPU/Llama path unless it directly proves the current
  slice.
- Do not read broad docs or benchmark artifacts unless named above.
- Do not add full Gemma generation, Gemma layer execution, full 9B CPU
  execution, sampling changes, broad graph abstractions, or benchmark work in
  this slice.
- Do not dequantize whole Gemma weight tensors into persistent F16/F32 buffers.
- Do not replace the resident plan with per-op host round trips.
- Do not make `local-gguf-generate` claim Gemma success yet.

## Verify

- Add deterministic tests proving the Gemma generate path creates the resident
  plan and prefill/decode schedules before the unsupported execution error.
- Preserve unsupported-template failures before runtime construction.
- Add unavailable-Metal behavior tests if the runtime shell now constructs
  Metal resources directly.
- Run focused engine/CLI tests touched by the runtime shell.
- Run local Gemma inspection/generation far enough to prove any remaining
  failure is still the explicit unsupported Gemma runtime error.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

The real Gemma generate path reaches a small runtime shell that owns or
constructs the resident Metal plan, creates prefill and one-token decode
schedules for the encoded prompt, and then fails with an explicit unsupported
Gemma execution error. Llama behavior remains unchanged.
