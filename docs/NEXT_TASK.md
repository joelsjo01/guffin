# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first reusable scheduled Gemma prefill block runner behind the Metal
runtime shell.

## Why

Gemma now reaches the real generate path far enough to load file-backed tensor
views, tokenize the prompt, upload the resident Metal model plan, build
separate prefill/decode schedules, run full scheduled blocks 0, 1, and 2
through FFN residual output, and hand layer 2's residual to block 3 as a
normalized attention input.

The next blocker is stopping bespoke layer-N shell expansion. Use the proven
block pattern behind one reusable scheduled runner before expanding through the
remaining Gemma layers.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` only if Gemma attention semantics
  are needed

## Touch

- Gemma engine/runtime planning code under `include/local_gguf_engine/engine`
  and `src/engine`
- Tensor/Metal backend code only if an existing primitive is not enough
- `tests/engine`
- `tests/tensor` only if adding or exposing a backend primitive
- `tests/frontends/cli/generate_cli_tests.cpp` only if the unsupported Gemma
  execution boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not add another bespoke layer-3 attention/FFN field chain.
- Do not add logits, sampling, KV updates, decode looping, or generation
  success.
- Do not add a broad graph abstraction or generic model-family operator stack.
- Do not dequantize whole Gemma weight tensors into persistent F16/F32 buffers.
- Do not replace the resident plan with per-op host round trips.
- Do not make `local-gguf-generate` claim Gemma success yet.
- Do not read benchmark artifacts.

## Verify

- Add deterministic tests for the runner's shape/resource ownership.
- Reuse CPU/Metal parity checks for the proven block operations.
- Keep unavailable-Metal behavior explicit if the runner creates Metal
  resources directly.
- Keep the real Gemma CLI failure as explicit unsupported Gemma execution until
  enough scheduled execution exists to return logits.
- Run focused engine/tensor/CLI tests touched by the runner.
- Run local Gemma inspection/generation far enough to prove any remaining
  failure is still explicit unsupported Gemma execution.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

One reusable scheduled Gemma prefill block runner can consume a normalized
layer input, run the proven attention/residual/FFN block pattern for a scheduled
layer without adding a new bespoke layer-N shell chain, and produce an explicit
FFN residual output with deterministic tests, while full Gemma generation
remains explicitly unsupported.
