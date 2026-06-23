# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first scheduled Gemma attention output projection setup primitive behind
the Metal runtime shell.

## Why

Gemma now reaches the real generate path far enough to load file-backed tensor
views, tokenize the prompt, upload the resident Metal model plan, build
separate prefill/decode schedules, turn prompt token IDs into backend-resident
hidden states with the Gemma embedding scale, normalize the first-layer
attention input, produce Q/K/V projection outputs, apply Q/K RoPE positioning,
produce pre-softmax attention scores, softmax those scores into explicit
phase-local attention probabilities, and aggregate those probabilities with the
first-layer V projection into an explicit phase-local attention value output.

The next blocker is proving that first-layer attention value output can project
through `blk.0.attn_output.weight` back to hidden width without breaking the
resident model plan. Keep this slice tiny and replaceable: enough to produce
the first projected attention output, but not enough to apply post-attention
norm/residual work or pretend full generation exists.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`, especially
  `## Current Inference Path And Bottlenecks`,
  `## Target Architecture Proposal`, and attention semantics if needed

## Touch

- Gemma engine/runtime planning code under `include/local_gguf_engine/engine`
  and `src/engine`
- Tensor/Metal backend code only for narrow backend-buffer projection helpers
  and tests if existing matmul helpers are not enough
- `tests/engine`
- `tests/tensor` only if adding or exposing a backend primitive
- `tests/frontends/cli/generate_cli_tests.cpp` only if the unsupported Gemma
  execution boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not add full Gemma generation or a full Gemma layer chain.
- Do not update KV cache.
- Do not implement post-attention norm, residual add, FFN work, logits, or
  sampling changes.
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

One small scheduled Gemma attention output projection setup primitive consumes
the runtime shell's first-layer attention value output and produces an explicit
resident or phase-local projected attention output with deterministic tests,
while full Gemma generation remains explicitly unsupported.
