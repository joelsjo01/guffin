# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first scheduled Gemma layer-2 attention score setup primitive behind
the Metal runtime shell.

## Why

Gemma now reaches the real generate path far enough to load file-backed tensor
views, tokenize the prompt, upload the resident Metal model plan, build
separate prefill/decode schedules, turn prompt token IDs into backend-resident
hidden states with the Gemma embedding scale, run the full first scheduled
layer through attention and FFN residual output, run layer-1 attention and FFN
through its residual output, hand that layer-1 FFN residual to block 2 by
normalizing it with `blk.2.attn_norm.weight`, project that layer-2 normalized
attention input through `blk.2.attn_q.weight`, `blk.2.attn_k.weight`, and
`blk.2.attn_v.weight`, and apply Gemma positions/RoPE to the layer-2 Q/K
outputs.

The next blocker is proving layer-2 pre-softmax attention scores without
taking the rest of layer-2 attention at once: consume
`prefill_third_layer_attention_position->query` and `key`, apply the scheduled
causal grouped attention score primitive, and produce an explicit resident or
phase-local layer-2 score workspace.

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
- Tensor/Metal backend code only for narrow backend-buffer attention score
  helpers and tests if existing helpers are not enough
- `tests/engine`
- `tests/tensor` only if adding or exposing a backend primitive
- `tests/frontends/cli/generate_cli_tests.cpp` only if the unsupported Gemma
  execution boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not add full Gemma generation or a full Gemma layer chain.
- Do not update KV cache.
- Do not implement layer-2 probabilities, value aggregation, output projection,
  residual work, FFN work, logits, or sampling changes.
- Do not run past layer-2 pre-softmax attention scores.
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

One small scheduled Gemma layer-2 attention score primitive consumes the runtime
shell's layer-2 positioned Q/K outputs and produces an explicit resident or
phase-local layer-2 pre-softmax score workspace with deterministic tests, while
full Gemma generation remains explicitly unsupported.
