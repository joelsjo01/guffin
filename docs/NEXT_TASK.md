# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Move Gemma greedy next-token selection off full logits host readback.

## Why

Gemma now reaches a bounded real greedy generation path. Full scheduled prefill
runs, scheduled prefill K/V writes into the resident KV cache, the selected
prefill token feeds repeated one-token decode steps, each decode appends
resident K/V, and `local-gguf-generate ... --greedy` succeeds for the covered
path.

Latest reusable/runtime outputs:

- `prefill_reusable_blocks`
- `prefill_reusable_layer_inputs`
- `prefill_kv_cache_writes`
- `prefill_logits.logits`
- `prefill_next_token_selection.token_id`
- `first_decode_reusable_layer_inputs`
- `first_decode_reusable_blocks`
- `first_decode_terminal_feed_forward_output.feed_forward_residual`
- `first_decode_output_norm.normalized_hidden_states`
- `first_decode_logits.logits`
- `first_decode_next_token_selection.token_id`
- `generated_token_ids`
- `decode_steps`

The next performance blocker is that the current greedy selection boundary
copies the full logits tensor back to host and runs CPU top-1. Replace that with
a Gemma/Metal top-1 boundary that leaves logits resident and reads back only
the selected token/logit.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` only if Gemma attention semantics
  are needed

## Touch

- Gemma runtime shell code under `include/local_gguf_engine/engine` and
  `src/engine`
- Tensor/Metal top-1 selection code if a narrow op is needed
- `tests/engine`
- `tests/tensor` if adding a narrow top-1 op
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not change the generated-token loop semantics.
- Do not fold prefill layers 0-2 into the reusable prefill runner yet.
- Do not add non-greedy sampling, temperature, top-k, top-p, repetition penalty,
  or RNG support for Gemma yet.
- Do not add broad graph abstractions or a generic model-family operator stack.
- Do not dequantize whole Gemma weight tensors into persistent F16/F32 buffers.
- Do not replace the resident plan with per-op host round trips.
- Do not reintroduce full-logits host readback for Gemma greedy selection.
- Do not make unsupported Gemma sampling/debug paths claim success before
  coverage is explicit.
- Do not read benchmark artifacts.

## Verify

- Add deterministic CPU/reference parity tests for the Metal top-1 token/logit
  result on prefill and decode logits.
- Add shape/resource ownership tests proving selection consumes resident logits
  and exposes only the small selected-token boundary.
- Keep unavailable-Metal behavior explicit if new code creates Metal resources
  directly.
- Run focused tensor/engine/CLI tests touched by the selection boundary.
- Run release Gemma inspection.
- Run release Gemma greedy generation smoke; it should remain successful.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

Gemma greedy selection no longer reads back full logits for top-1; the bounded
generated-token loop and real `--greedy` generation still pass, while
non-greedy Gemma sampling remains explicitly unsupported.
