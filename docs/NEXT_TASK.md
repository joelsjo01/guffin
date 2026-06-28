# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Implement the first Gemma decode logits projection boundary.

## Why

Gemma now reaches the real generate path far enough to run full scheduled
prefill, write scheduled prefill K/V into the resident KV cache, select the
next token from prefill logits, embed that selected token for decode, normalize
the first decode layer input, and run one reusable scheduled decode loop across
scheduled decode layers for that selected token.

Latest reusable/runtime outputs:

- `prefill_reusable_blocks`
- `prefill_reusable_layer_inputs`
- `prefill_kv_cache_writes`
- `prefill_output_norm.normalized_hidden_states`
- `prefill_logits.logits`
- `prefill_next_token_selection.token_id`
- `first_decode_input.hidden_states`
- `first_decode_first_layer_input.normalized_attention_input`
- `first_decode_block.kv_cache_write`
- `first_decode_block.feed_forward_output.feed_forward_residual`
- `first_decode_reusable_layer_inputs`
- `first_decode_reusable_blocks`
- `first_decode_terminal_feed_forward_output.feed_forward_residual`

The next blocker is to normalize the terminal decode residual and project it
through the output weight into explicit decode logits for that one selected
token, without adding the generated-token loop yet.

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
- `tests/engine`
- Tensor/Metal backend code only if existing primitives are not enough
- `tests/tensor` only if adding or exposing a backend primitive
- `tests/frontends/cli/generate_cli_tests.cpp` only if the unsupported Gemma
  execution boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not add bespoke decode layer-N attention/FFN fields.
- Do not fold prefill layers 0-2 into the reusable prefill runner yet.
- Do not add a generated-token loop or generation success.
- Do not change sampling behavior.
- Do not add a broad graph abstraction or generic model-family operator stack.
- Do not dequantize whole Gemma weight tensors into persistent F16/F32 buffers.
- Do not replace the resident plan with per-op host round trips.
- Do not make `local-gguf-generate` claim Gemma success yet.
- Do not read benchmark artifacts.

## Verify

- Add deterministic tests for decode output normalization, raw logits,
  softcapped logits, shape/resource ownership, and terminal layer selection.
- Reuse CPU/Metal parity checks for the decode output norm/logits projection
  where possible.
- Keep unavailable-Metal behavior explicit if new code creates Metal resources
  directly.
- Keep the real Gemma CLI failure as explicit unsupported Gemma execution until
  the generation boundary is intentionally moved.
- Run focused engine/tensor/CLI tests touched by the decode-logits boundary.
- Run local Gemma inspection/generation far enough to prove any remaining
  failure is still explicit unsupported Gemma execution.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

The terminal first-decode FFN residual feeds Gemma output RMSNorm and logits
projection for one selected token, producing explicit raw and softcapped decode
logits without a generated-token loop or generation success.
