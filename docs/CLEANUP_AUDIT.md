# Cleanup Audit

Living ledger for removing old code that distracts from the Gemma 2 Q4
Metal-first runtime. Keep this file compact.

## Cleanup Policy

- Product generation is Gemma 2, greedy, Metal-first, and model-resident.
- CPU code stays only when it is generic infrastructure or correctness
  scaffolding.
- The tokenizer may use GGUF/GGML names such as `tokenizer.ggml.model = llama`
  when Gemma files require them; that is compatibility metadata, not Llama
  execution support.
- Benchmarks should measure the real `local-gguf-generate` path. Standalone
  synthetic benchmark scaffolding should not be resurrected by default.

## Removed In Bold Pass

- Standalone benchmark binary and harness:
  - `local-gguf-bench` Makefile target and build inputs
  - `include/local_gguf_engine/benchmarks`
  - `benchmarks/benchmark_harness.cpp`
  - `benchmarks/engine`
  - `benchmarks/frontends/cli`
  - `benchmarks/model_format`
  - `benchmarks/support`
  - `benchmarks/tensor`
  - `tests/benchmarks`
- Llama execution runtime:
  - `include/local_gguf_engine/engine/llama_model.hpp`
  - `include/local_gguf_engine/engine/llama_layer.hpp`
  - `include/local_gguf_engine/engine/kv_cache.hpp`
  - `src/engine/llama_model.cpp`
  - `src/engine/llama_layer.cpp`
  - `src/engine/kv_cache.cpp`
  - Llama engine tests and Llama test support
- Old Llama Metal cached path:
  - `src/tensor/backends/metal/metal_llama_ops.*`
  - `src/tensor/backends/metal/metal_benchmark.*`
  - public `llama_cached_*` tensor op declarations
  - eager Llama cached pipeline creation in `metal_context.*`
  - embedded Llama cached kernels in `metal_backend.mm`
- Generate CLI behavior:
  - non-Gemma generation now fails through the Gemma-only model boundary
  - `--backend cpu` is rejected for Gemma generation
  - `--debug-first-token` was removed with the old Llama debug path

## Removed In Follow-Up Sweep

- Llama model metadata support:
  - `ModelArchitecture::llama`
  - `LlamaModelConfig`
  - `EngineModel::llama_config()`
  - `load_llama_config()`
  - the Llama branch in `load_engine_model_from_gguf`
  - eager owned tensor loading that existed only for Llama model compatibility
- Llama model tests:
  - Llama model-load coverage
  - missing/invalid Llama metadata coverage
  - artificial non-Gemma `EngineModel` construction in Gemma Metal plan tests
- Tokenizer legacy paths:
  - SmolLM2 chat-template detection and formatting
  - GPT-2/BPE tokenizer model support
  - GPT-2 byte alphabet and BPE merge helpers
  - SmolLM2/GPT-2 tokenizer tests and fixtures
- Generate CLI sampling surface:
  - `--temperature`
  - `--top-k`
  - `--top-p`
  - `--repeat-penalty`
  - `--seed`
  - sampled default generation settings
  - transition tests for rejected Llama generation and `--debug-first-token`
- Stochastic CPU sampling:
  - `SamplingConfig`
  - `SamplingState`
  - `sample_token_f32`
  - top-k, top-p, repetition penalty, and RNG sampling tests

## Kept Intentionally

- `TokenizerModel::llama`, because Gemma GGUF tokenizer metadata uses the GGML
  Llama tokenizer label.
- `top_logit_tokens_f32`, because Gemma parity tests use it for CPU/reference
  checks.
- CPU tensor/backend/reference primitives. These are correctness scaffolding
  for Gemma Metal parity.
- `benchmarks/results/.gitkeep`. Future real-path metrics can still write
  ignored artifacts there.

## Remaining Scan Candidates

1. Add real Gemma `local-gguf-generate --bench` metrics and keep all benchmark
   plumbing on that command path.
2. After `--bench` lands, scan README/docs again for stale benchmark wording.
3. Treat `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` as historical. Prefer
   current code search over its old Llama/SmolLM2 references.

## Next Cleanup Candidates

- Tests have more removable bulk than `src`. The clearest target is
  `tests/engine/gemma2_metal_plan_tests.cpp`: it is large, milestone-era, and
  over-asserts many staged Gemma runtime handoffs. Keep the high-signal coverage
  for plan shape, prefill, decode, generated-token loops, residency, and error
  boundaries; consolidate repeated layer-by-layer assertions.
- Tensor tests are large but mostly useful correctness scaffolding. Trim only
  obvious duplicates; keep Q4_K/Q6_K, backend residency, Metal parity,
  top-logit, and KV/cache coverage.
- GGUF parser fixtures still use some `llama.*` metadata as generic test data.
  This is not runtime Llama support, but the fixtures can be renamed or rebuilt
  around neutral/Gemma metadata to reduce misleading context.
- `src` has less obvious dead legacy left. `gemma2_runtime.cpp` is live but
  likely carries incremental milestone structure; prefer consolidation after
  benchmark evidence, not deletion. `metal_tensor_ops.mm` is also live backend
  code and is a later split-by-op-family candidate.
- Small low-value cleanup remains in unreachable non-Gemma defensive checks now
  that the model boundary is Gemma-only.

## Verification

- `bash scripts/ci.sh` passed after the bold deletion pass.
- `bash scripts/ci.sh` passed after the follow-up sweep with `286 test(s)
  passed`.
