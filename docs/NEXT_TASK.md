# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first Metal-resident Gemma model plan boundary.

## Why

Large Gemma tensor payloads now flow through validated file-backed views without
eager owned tensor copies on the Gemma load path. The next blocker is backend
residency: Gemma needs an explicit Metal-owned plan that can receive packed
weight views, own persistent buffers, and define KV/scratch/layer-state
ownership before real packed kernels and execution schedules land.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`, especially
  `## Target Architecture Proposal`

## Touch

- Gemma engine/runtime planning code under `include/local_gguf_engine/engine`
  and `src/engine`
- Metal buffer/context ownership code only where needed for persistent Gemma
  weight/KV/scratch handles
- `tests/engine`
- `tests/tensor` only for backend ownership/unavailable-device behavior
- CLI tests only if the unsupported Gemma runtime boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not expand into unrelated cleanup.
- Do not optimize the old CPU/Llama path unless it directly proves the current
  slice.
- Do not read broad docs or benchmark artifacts unless named above.
- Do not add full Gemma generation, full 9B CPU execution, packed matmul
  kernels, attention kernels, sampling changes, broad graph abstractions, or
  benchmark work in this slice.
- Do not copy all Gemma tensor payloads back into owned host `TensorStorage`
  before Metal upload.

## Verify

- Add deterministic tests for plan ownership, required weight coverage,
  unavailable Metal behavior, and no eager host tensor-copy regression.
- Run focused engine/tensor/CLI tests touched by the ownership change.
- Run local Gemma inspection/generation far enough to prove any remaining
  failure is still the explicit unsupported Gemma runtime error.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

Gemma has a small reusable Metal plan object that is constructed from borrowed
validated weight views, owns persistent backend resources for weights plus
declared KV/scratch/layer state, handles unavailable Metal deterministically,
and still stops before unsupported Gemma execution rather than drifting into
kernel or generation work.
