# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add end-to-end Gemma generation benchmark metrics.

## Why

Gemma now reaches a bounded real greedy generation path with resident top-1
selection. Full scheduled prefill runs, scheduled prefill K/V writes into the
resident KV cache, selected tokens feed repeated one-token decode steps, each
decode appends resident K/V, logits stay on Metal for greedy selection, and
only the selected token/logit is read back.

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
- `tensor::ops::top_logit_f32_from_backend`

The next blocker is evidence. The real Gemma path is still slow, but the major
GPU-residency boundary is now clean enough to measure. Add compact end-to-end
metrics to the same `local-gguf-generate` command path so the next optimization
slice is chosen from data rather than guesses.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` only if Gemma attention semantics
  are needed

## Touch

- `include/local_gguf_engine/frontends/cli/generate.hpp`
- `src/frontends/cli/generate.cpp`
- `benchmarks/frontends/cli` or benchmark report plumbing only if needed for
  shared formatting
- `tests/frontends/cli/generate_cli_tests.cpp`
- `tests/benchmarks` only if shared benchmark formatting changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not change Gemma generation semantics or token output.
- Do not add non-greedy Gemma sampling, temperature, top-k, top-p, repetition
  penalty, or RNG support.
- Do not add broad graph abstractions or a generic model-family operator stack.
- Do not refactor large Gemma runtime files in this slice.
- Do not dequantize whole Gemma weight tensors.
- Do not read benchmark result artifacts from prior runs; produce fresh metrics
  if benchmarking is run.
- Do not make unsupported Gemma sampling/debug paths claim success.

## Verify

- Add CLI tests for benchmark flag parsing/report shape and failure paths.
- Keep normal generation stdout stable when benchmarking is not requested.
- Report at least TTFT, prompt eval tokens/s, decode tokens/s, total tokens/s,
  prompt token count, generated token count, and total elapsed time.
- Run focused CLI/benchmark tests touched by metrics.
- Run release Gemma inspection.
- Run release Gemma greedy generation smoke; it should remain successful.
- Run a fresh release Gemma benchmark smoke if the command path is added.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

The real Gemma `--greedy` command can emit compact end-to-end performance
metrics for the same generation path, without changing generated text or
unsupported sampling behavior.
