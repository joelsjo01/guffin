# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first scheduled Gemma layer-1 FFN output setup primitive behind the
Metal runtime shell.

## Why

Gemma now reaches the real generate path far enough to load file-backed tensor
views, tokenize the prompt, upload the resident Metal model plan, build
separate prefill/decode schedules, turn prompt token IDs into backend-resident
hidden states with the Gemma embedding scale, run the full first scheduled
layer through attention and FFN residual output, normalize that first-layer FFN
residual into the layer-1 attention input, run layer-1 attention through Q/K/V,
RoPE, scores, probabilities, value aggregation, output projection, and then
apply `blk.1.post_attention_norm.weight` plus the layer-1 residual add. It now
normalizes that layer-1 post-attention residual with `blk.1.ffn_norm.weight`
and projects the normalized FFN input through `blk.1.ffn_gate.weight` and
`blk.1.ffn_up.weight`.

The next blocker is proving layer-1 FFN output setup without taking logits or
the full layer loop at once: consume
`prefill_second_layer_post_attention_residual->post_attention_residual` and
`prefill_second_layer_feed_forward_gate_up`, apply Gemma's gated FFN activation,
project through `blk.1.ffn_down.weight`, apply `blk.1.post_ffw_norm.weight`,
add the layer-1 post-attention residual, and produce an explicit resident or
phase-local layer-1 FFN residual. Keep this slice tiny and replaceable: enough
to prove layer-1 FFN output setup, but not enough to run logits, sampling, KV
cache updates, full layer chaining, or generation success.

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
- Tensor/Metal backend code only for narrow backend-buffer norm helpers and
  tests if existing helpers are not enough
- `tests/engine`
- `tests/tensor` only if adding or exposing a backend primitive
- `tests/frontends/cli/generate_cli_tests.cpp` only if the unsupported Gemma
  execution boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not add full Gemma generation or a full Gemma layer chain.
- Do not update KV cache.
- Do not implement logits or sampling changes.
- Do not run past layer-1 FFN residual output.
- Do not add layer-2 handoff in this slice.
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

One small scheduled Gemma layer-1 FFN output primitive consumes the runtime
shell's layer-1 post-attention residual plus layer-1 FFN gate/up outputs and
produces an explicit resident or phase-local layer-1 FFN residual with
deterministic tests, while full Gemma generation remains explicitly
unsupported.
