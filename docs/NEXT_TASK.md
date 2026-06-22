# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first scheduled Gemma execution primitive behind the Metal runtime
shell.

## Why

Gemma now reaches the real generate path far enough to load file-backed tensor
views, tokenize the prompt, upload the resident Metal model plan, and build
separate prefill/decode schedules. The next blocker is proving that a small
piece of actual Gemma execution can run through that shell while keeping
weights, KV, scratch, and layer state resident.

The first primitive should be tiny and replaceable: enough to move real token
IDs/control data into a resident execution step, but not enough to pretend
full generation exists.

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
- Metal backend/tensor code only for one narrow primitive and its tests
- `tests/engine`
- `tests/tensor` only if adding a backend primitive
- `tests/frontends/cli/generate_cli_tests.cpp` only if the unsupported Gemma
  execution boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not add full Gemma generation or a full Gemma layer chain.
- Do not add full 9B CPU execution, sampling changes, broad graph
  abstractions, or benchmark work in this slice.
- Do not dequantize whole Gemma weight tensors into persistent F16/F32 buffers.
- Do not replace the resident plan with per-op host round trips.
- Do not make `local-gguf-generate` claim Gemma success yet.
- Do not read benchmark artifacts.

## Verify

- Add deterministic tests for the primitive shape, resource ownership, and
  unavailable-Metal behavior if it creates Metal resources directly.
- Keep the real Gemma CLI failure as explicit unsupported Gemma execution until
  enough scheduled execution exists to return logits.
- Run focused engine/tensor/CLI tests touched by the primitive.
- Run local Gemma inspection/generation far enough to prove any remaining
  failure is still explicit unsupported Gemma execution.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

One small scheduled Gemma execution primitive is reachable from the runtime
shell with resident buffers and deterministic tests, while full Gemma generation
remains explicitly unsupported.
