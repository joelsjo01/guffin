# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first packed Q4_K/Q6_K Metal matmul path for Gemma projection weights.

## Why

Gemma tensor payloads can now load through validated file-backed views and a
Metal-resident model plan can own packed weights, KV, scratch, and per-layer
state. The next blocker is execution of the dominant projection work: consume
packed Q4_K/Q6_K weight buffers directly on Metal for representative matmul
shapes, without whole-model dequantization or falling back to owned host copies.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`, especially
  `## Current Inference Path And Bottlenecks` and
  `## Target Architecture Proposal`

## Touch

- Metal tensor-op/kernel code under `src/tensor/backends/metal`
- Matmul op surface under `include/local_gguf_engine/tensor/ops` and
  `src/tensor/ops` only if the existing API cannot express packed inputs
- Gemma model plan code only if descriptors/weight handles need small metadata
  additions for packed matmul dispatch
- `tests/tensor`
- `tests/engine` only if plan integration changes
- CLI tests only if the unsupported Gemma runtime boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not expand into unrelated cleanup.
- Do not optimize the old CPU/Llama path unless it directly proves the current
  slice.
- Do not read broad docs or benchmark artifacts unless named above.
- Do not add full Gemma generation, full 9B CPU execution, attention kernels,
  prefill/decode scheduling, sampling changes, broad graph abstractions, or
  benchmark work in this slice.
- Do not dequantize whole Gemma weight tensors into persistent F16/F32 buffers
  just to make matmul easier.

## Verify

- Add deterministic parity tests for small packed Q4_K and Q6_K matmul cases
  against CPU/reference results.
- Add unavailable-Metal behavior tests for the new Metal matmul path.
- Run focused tensor/engine/CLI tests touched by the matmul change.
- Run local Gemma inspection/generation far enough to prove any remaining
  failure is still the explicit unsupported Gemma runtime error.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

Representative packed Q4_K and Q6_K matmul inputs can execute directly from
packed Metal buffers with deterministic CPU/reference parity, unavailable Metal
is handled explicitly, and the real Gemma CLI path still stops at unsupported
Gemma execution rather than drifting into full generation.
