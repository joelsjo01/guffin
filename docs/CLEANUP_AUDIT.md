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

## Removed In Prefill Loop Consolidation

- Gemma prefill milestone scaffolding:
  - named layer-2, layer-3, and layer-4 prefill activation fields on
    `Gemma2MetalGenerationRuntimeShell`
  - old optional second/third/fourth prefill helper functions
  - old terminal prefill selector that special-cased second and third layers
- Prefill now keeps the first layer explicit and runs later scheduled layers
  through the reusable block loop, matching the decode-side shape. The first
  layer stays separate because it starts from prompt embeddings instead of a
  previous block residual.
- `src/engine/gemma2_runtime.cpp` dropped from roughly 4.7k lines to roughly
  2.8k lines in this pass.

## Removed In Gemma Plan Test Consolidation

- `tests/engine/gemma2_metal_plan_tests.cpp` dropped from 6,656 lines to 3,474
  lines.
- Removed milestone-era staged tests that preserved old construction history:
  - first-layer projection-only and RoPE-only runtime shell tests
  - one-later-layer and two-later-layer prefill loop count tests
  - standalone reusable prefill "feed next scheduled layer" test
  - the very large first-layer feed-forward/output walkthrough
- Replaced the first-layer walkthrough with one compact boundary/parity test
  that proves the first prefill block starts from prompt embeddings, produces
  the first residual, and feeds the reusable later-layer input.
- Kept high-signal Gemma Metal coverage for plan shape, resident buffers,
  first prefill boundary, reusable prefill block runner parity, later prefill
  loop/output-token path, prefill KV writes, decode loop, generated-token loop,
  and invalid/unavailable Metal boundaries.

## Removed In Tensor Test Guard Consolidation

- `tests/tensor/tensor_tests.cpp` and
  `tests/tensor/tensor_backend_tests.cpp` dropped from 5,406 lines to 5,328
  lines.
- Merged milestone-style negative-path tests that repeated the same guard
  shape:
  - F32 add and multiply dtype/shape rejection tests now run through one
    elementwise invalid-input test.
  - F32 matmul dtype/rank/inner-dimension rejection tests now run through one
    matmul invalid-input test.
  - Metal-unavailable probes for GGUF matmul, K-quant matmul, backend copy,
    view copy, and allocation now run through one unavailable-backend test.
- Kept live tensor coverage for Q4_K/Q6_K layout, decode, dequant, row
  validation, F16/F32 accessors, CPU backend reference behavior, Metal parity,
  backend residency/copy, top-logit selection, prefill KV writes, and KV-cache
  attention.

## Removed In Shared Tensor Fixture Consolidation

- Moved duplicated packed Q4_K/Q6_K test fixture construction into
  `tests/support/tensor_test_support.*`.
- Removed local copies of the same K-quant byte-packing helpers from:
  - `tests/tensor/tensor_tests.cpp`
  - `tests/tensor/tensor_backend_tests.cpp`
  - `tests/engine/gemma2_metal_plan_tests.cpp`
- The tensor test files dropped from 5,328 lines to 5,014 lines, and
  `tests/engine/gemma2_metal_plan_tests.cpp` dropped from 3,474 lines to
  3,322 lines.
- Shared support now owns seeded Q4_K/Q6_K GGUF weight matrices, Q4_K/Q6_K
  reference tensors, projection-value fixtures, and the quantized reference
  matmul used by backend parity tests.
- Kept all packed-layout assertions, CPU/reference K-quant matmul checks,
  Metal Q4_K/Q6_K parity checks, and Gemma Metal plan/runtime coverage.

## Removed In GGUF Parser Fixture Cleanup

- Replaced generic parser-fixture Llama metadata with neutral model metadata:
  - `general.architecture = dummy`
  - `model.context_length`
  - `model.use_parallel_residual`
- Removed the remaining `llama.*` metadata from `tests/model_format` and
  `tests/fixtures`.
- Kept real `tokenizer.ggml.model = llama` metadata in tokenizer and CLI
  tests, because Gemma GGUF files require that GGML tokenizer label.

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

- `tests/engine/gemma2_metal_plan_tests.cpp` has had the main milestone-era
  bulk removed. Future trims should be opportunistic and keep the current
  high-signal boundaries intact.
- Tensor tests are still large but mostly useful correctness scaffolding. After
  guard and fixture consolidation, further trims should focus on repeated
  parity/invalid-input boilerplate and avoid weakening Q4_K/Q6_K, backend
  residency, Metal parity, top-logit, and KV/cache coverage.
- `src` has less obvious dead legacy left. `gemma2_runtime.cpp` is live but
  should now be treated as a live reusable prefill/decode runner, not
  layer-by-layer milestone scaffolding. `metal_tensor_ops.mm` is also live
  backend code and is a later split-by-op-family candidate.
- Small low-value cleanup remains in unreachable non-Gemma defensive checks now
  that the model boundary is Gemma-only.

## Verification

- `bash scripts/ci.sh` passed after the bold deletion pass.
- `bash scripts/ci.sh` passed after the follow-up sweep with `286 test(s)
  passed`.
- `make test MODE=debug` passed after Gemma plan test consolidation with
  `281 test(s) passed`.
- `bash scripts/ci.sh` passed after Gemma plan test consolidation with
  `281 test(s) passed`.
- `make test MODE=debug` passed after tensor test guard consolidation with
  `272 test(s) passed`.
- `bash scripts/ci.sh` passed after tensor test guard consolidation with
  `272 test(s) passed`.
- `make test MODE=debug` passed after shared tensor fixture consolidation with
  `272 test(s) passed`.
- `bash scripts/ci.sh` passed after shared tensor fixture consolidation with
  `272 test(s) passed`.
- `make test MODE=debug` passed after GGUF parser fixture cleanup with
  `272 test(s) passed`.
- `bash scripts/ci.sh` passed after GGUF parser fixture cleanup with
  `272 test(s) passed`.
