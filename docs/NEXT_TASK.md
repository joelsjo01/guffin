# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add the first scheduled Gemma layer-1 post-attention residual setup primitive
behind the Metal runtime shell.

## Why

Gemma now reaches the real generate path far enough to load file-backed tensor
views, tokenize the prompt, upload the resident Metal model plan, build
separate prefill/decode schedules, turn prompt token IDs into backend-resident
hidden states with the Gemma embedding scale, normalize the first-layer
attention input, produce Q/K/V projection outputs, apply Q/K RoPE positioning,
produce pre-softmax attention scores, softmax those scores into explicit
phase-local attention probabilities, and aggregate those probabilities with the
first-layer V projection into an explicit phase-local attention value output,
then project that attention output through `blk.0.attn_output.weight` back to
hidden width, apply Gemma post-attention norm plus the residual add, then
normalize that post-attention residual with `blk.0.ffn_norm.weight` and project
the normalized FFN input through `blk.0.ffn_gate.weight` and
`blk.0.ffn_up.weight`, apply Gemma's gated FFN activation, project through
`blk.0.ffn_down.weight`, then apply `blk.0.post_ffw_norm.weight` and the FFN
residual add, then normalize that first-layer FFN residual with
`blk.1.attn_norm.weight` when a second layer exists, and project that layer-1
normalized attention input through `blk.1.attn_q.weight`,
`blk.1.attn_k.weight`, and `blk.1.attn_v.weight`, then apply Gemma RoPE
positioning to the layer-1 Q/K projection outputs, then produce explicit
layer-1 pre-softmax attention scores with the Gemma scale, logit softcap, GQA
head mapping, and causal/window masking, then softmax those scores into
explicit layer-1 attention probabilities, then aggregate those probabilities
with the layer-1 V projection into an explicit layer-1 attention value output,
then project that attention value output through `blk.1.attn_output.weight`
back to hidden width.

The next blocker is proving layer-1 post-attention residual setup without
taking the full layer loop at once: consume
`prefill_second_layer_attention_output_projection->projected_attention_output`
and the layer-1 residual input from
`prefill_first_layer_feed_forward_output.feed_forward_residual`, normalize the
projected attention output with `blk.1.post_attention_norm.weight`, add the
residual input, and produce an explicit resident or phase-local layer-1
post-attention residual. Keep this slice tiny and replaceable: enough to prove
layer-1 post-attention residual setup, but not enough to run FFN, logits,
sampling, KV cache updates, or generation success.

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
- Do not implement layer-1 FFN, logits, or sampling changes.
- Do not run past layer-1 post-attention residual output.
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

One small scheduled Gemma layer-1 post-attention residual primitive consumes
the runtime shell's layer-1 projected attention output and layer-1 residual
input and produces an explicit resident or phase-local layer-1 post-attention
residual with deterministic tests, while full Gemma generation remains
explicitly unsupported.
