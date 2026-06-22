# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first separate Gemma prefill/decode schedule boundary on the
Metal-resident plan.

## Why

Gemma now has file-backed loading, a Metal-resident model plan, and first
packed Q4_K/Q6_K matmul kernels that consume resident packed buffers directly.
The next blocker is execution structure: prompt prefill and one-token decode
need separate schedules over the resident plan so KV/scratch ownership,
command encoding, and phase-specific work are explicit before full generation
is wired.

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
- Metal tensor-op/kernel code only where a schedule needs a narrow backend
  primitive
- Gemma model plan code if schedule handles, phase scratch, or small control
  metadata need to be added
- `tests/engine`
- `tests/tensor` only for new backend primitives or parity helpers
- CLI tests only if the unsupported Gemma runtime boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not expand into unrelated cleanup.
- Do not optimize the old CPU/Llama path unless it directly proves the current
  slice.
- Do not read broad docs or benchmark artifacts unless named above.
- Do not add full Gemma generation, full 9B CPU execution, sampling changes,
  broad graph abstractions, or benchmark work in this slice.
- Do not dequantize whole Gemma weight tensors into persistent F16/F32 buffers.
- Do not replace the resident plan with per-op host round trips.

## Verify

- Add deterministic tests that prove prefill and decode are distinct schedule
  objects or paths, with explicit KV/scratch/layer-state ownership.
- Add unavailable-Metal behavior tests if the schedule creates new Metal
  resources directly.
- Run focused engine/tensor/CLI tests touched by the schedule change.
- Run local Gemma inspection/generation far enough to prove any remaining
  failure is still the explicit unsupported Gemma runtime error.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

Gemma has a small reusable schedule boundary that separates prompt prefill from
one-token decode on the resident Metal plan, keeps ownership explicit, has
deterministic tests for phase shape/resource behavior, and still stops before
claiming full Gemma generation.
