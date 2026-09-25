# Project Graph

Read when: locating code, tests, fixtures, or command surfaces for a task.
Do not read as history.

## Read Rule

1. Read `AGENTS.md`.
2. Read `docs/ENGINE_PRINCIPLES.md`.
3. Read `docs/NEXT_TASK.md`.
4. Read this file only when choosing files to inspect or edit.
5. Then read only the relevant code/tests below.

## Dependency Flow

```text
src/frontends/cli
  -> src/engine/gemma2_chat_session.*

src/benchmarks
  -> src/engine/gemma2_chat_session.*

src/engine/gemma2_chat_session.*
  -> src/engine/model.cpp
  -> src/model_format
  -> src/tokenizer
  -> src/engine/sampling.*
  -> src/engine/retrieval_proposer.*
  -> src/engine/gemma2_runtime.*
      -> src/engine/gemma2_metal_plan.*
      -> src/tensor
          -> src/tensor/ops
          -> src/tensor/backends/cpu_backend.*
          -> src/tensor/backends/metal_backend.*
          -> src/tensor/backends/metal/*

src/engine/speculative_controller.*
  -> src/engine/sampling.*

src/engine/target_verifier*.*
  -> src/engine/speculative_controller.*
  -> src/engine/gemma2_runtime.*
  -> src/tensor/backends/metal/*

src/engine/prepared_gemma2_plan.*
  -> src/engine/model.cpp
  -> src/engine/gemma2_metal_plan.*
  -> src/engine/standalone_chat.*

src/engine/standalone_chat.*
  -> src/engine/standalone_draft_artifact.*
  -> src/engine/gemma3_draft.*
  -> src/engine/chain_chat.*
  -> src/tokenizer

src/engine/chain_chat.*
  -> src/engine/speculative_controller.*
  -> src/engine/target_verifier.*

src/engine/feature_bank.*
  -> src/engine/target_verifier.*
  -> src/tensor/backends/metal/*

scripts/gemma3_reference_golden.py
  -> scripts/reference_gguf.py

tests
  -> module under test
  -> tests/support
  -> tests/fixtures
```

Benchmark output lives under ignored `benchmarks/results`.

## Build Entrypoints

| Path | Use |
| --- | --- |
| `Makefile` | build graph for engine sources, CLIs, and tests |
| `scripts/ci.sh` | repository check wrapper |
| `scripts/quick_benchmark.py` | canonical local real-chat benchmark driver |
| `scripts/memory_benchmark.py` | interleaved production-path startup and process-memory benchmark |
| `scripts/output_head_benchmark.py` | thermally spaced F16/Q8 output-head production-chat comparison |
| `scripts/storage_policy_benchmark.py` | paired immutable-weight Metal resource-policy screen |
| `scripts/capture_decode_gpu_trace.py` | one-shot production decode `.gputrace` driver |
| `scripts/llama_reference_benchmark.py` | infrequent pinned llama-cli reference driver |
| `scripts/gemma3_reference_golden.py` | independent full-checkpoint GGUF/Q8 Gemma 3 CPU logits and logical K/V goldens |
| `scripts/reference_gguf.py` | independent GGUF reader shared by offline checkpoint equations |
| `scripts/reference_requirements.txt` | pinned NumPy/PyTorch dependencies for offline CPU reference goldens |
| `docs/LLAMA_CPP_REFERENCE_BASELINE.md` | stored pinned reference identity, protocol, and result |
| `src/frontends/cli/generate_main.cpp` | `local-gguf-generate` binary entry |
| `src/frontends/cli/gguf_inspect_main.cpp` | `gguf-inspect` binary entry |
| `tests/test_main.cpp` | single test binary registry |

## Module Ownership

| Path | Owns | Does not own |
| --- | --- | --- |
| `include/local_gguf_engine` | public API declarations matching source modules | implementation |
| `src/model_format` | GGUF headers, metadata, tensor directories, payload views, payload-size validation | tokenizer logic, inference |
| `src/tokenizer` | tokenizer metadata, encode/decode, Gemma prompt/chat formatting | model execution |
| `src/runtime` | backend enum/error types shared across runtime layers | backend execution |
| `src/benchmarks` | secondary production-kernel probes and summaries | chat inference execution, reference execution, CLI formatting |
| `src/engine/model.cpp` | engine model loading from GGUF model views, architecture boundary | raw GGUF parsing internals |
| `include/local_gguf_engine/engine/model.hpp` | source-model and resident-target payload receipt types | training artifacts, file serialization |
| `src/engine/gemma2_model.*` | Gemma 2 config parsing and weight binding | backend upload/execution |
| `src/engine/gemma2_metal_plan.*` | Metal resident weight upload, KV cache buffers, prefill/decode/verifier schedules and transaction state | tokenization, CLI formatting |
| `src/engine/gemma2_runtime.*` | Gemma Metal prefill/decode/verifier execution, resident decode, metrics, argmax and selector-driven token loops | chat history, CLI parsing |
| `src/engine/gemma2_chat_session.*` | persistent chat state, prompt rendering, KV-prefix reuse, context trimming, stops, streaming, regeneration, sampler integration | CLI commands and formatting |
| `src/engine/gemma2_reference.*` | tiny CPU/reference Gemma numerics for parity tests | product CPU runtime |
| `src/engine/sampling.*` | greedy and seeded stochastic token selection, top-k/top-p, repetition penalty, token history, integer proposal laws, rejection/residual sampling, checkpoint-owned random streams | model execution |
| `src/engine/speculative_controller.*` | fixed-capacity proposal-graph validation, speculative token/callback/cache transitions and sampler tickets | model execution, learned drafting kernels, chat rendering |
| `src/engine/target_verifier*.*` | validated verifier requests, owning target tickets, target-only episodes, private speculative K/V, accepted-path publication, and fault seams | learned drafting, chat rendering |
| `src/engine/chain_chat.mm` | draft-independent chain graph, full/conditional target transactions, sampler forks and controller-driven callback prefixes | draft model execution, tokenizer conversion |
| `src/engine/standalone_draft_artifact.cpp` | pinned complete Gemma 3 artifact authentication and typed bounded target/draft tokenizer mapping | Metal model execution, target sampling |
| `src/engine/gemma3_draft.mm` | resident complete Q8 Gemma 3 transformer, hybrid attention caches, private draft transactions and full-vocabulary proposal laws | target verification, conversation tokenization |
| `include/local_gguf_engine/engine/standalone_draft_types.hpp` | native draft execution statistics | draft model execution, target verification |
| `src/engine/standalone_chat.cpp` | target/draft emission ledgers, bounded context conversion, explicit native-context restoration, draft-only scalar resolution and joint chain lifecycle | target sampler transforms, model weight conversion |
| `src/engine/prepared_gemma2_plan.*` | atomic target/standalone preparation, authenticated payload receipts and aggregate lifecycle | chat session composition |
| `src/engine/feature_bank.*` | typed feature capture/projection provenance, fixed-capacity Metal storage, draft cache substrate, and joint promotion lifecycle | drafter scoring, training |
| `src/engine/retrieval_proposer.*` | session-local bounded suffix retrieval, continuation evidence, and profitability gating | sampling, model execution, chat rendering |
| `src/tensor` | tensor descriptors/storage, dtype/layout, F16, quantization views, typed access, backend buffer wrapper | model-family policy |
| `src/tensor/ops` | CPU/reference tensor ops and backend-dispatched op front doors | device setup |
| `src/tensor/backends/cpu_backend.*` | CPU backend buffers and CPU backend execution helpers | product-speed Gemma path |
| `src/tensor/backends/metal_backend.*` | public Metal backend bridge, availability, buffer copy/allocation | CLI behavior |
| `src/tensor/backends/metal/metal_command_context.*` | resident prefill/decode command encoding, submission statistics, selected-token readback | model scheduling, chat state |
| `src/tensor/backends/metal/*` | Metal device context, buffers, kernels, command encoding, Metal op implementations | benchmark reporting |
| `src/frontends/cli` | command arguments, file IO, stdout/stderr formatting, error reporting | inference math |
| `scripts/gemma3_reference_golden.py`, `scripts/reference_gguf.py` | independent CPU checkpoint equations and authenticated GGUF reads | production inference |
| `tests` | deterministic coverage and fixtures | real model weights, generated benchmark artifacts |

## Task Routes

Gemma 2 model binding:

- `include/local_gguf_engine/engine/model.hpp`
- `include/local_gguf_engine/engine/gemma2_model.hpp`
- `src/engine/model.cpp`
- `src/engine/gemma2_model.cpp`
- `tests/engine/engine_model_tests.cpp`

Source and resident model identities:

- `include/local_gguf_engine/engine/model.hpp`
- `src/engine/model.cpp`
- `src/engine/gemma2_metal_plan.cpp`
- `include/local_gguf_engine/runtime/sha256.hpp`
- `src/runtime/sha256.cpp`
- `tests/engine/engine_model_tests.cpp`
- `tests/engine/standalone_draft_artifact_tests.cpp`

Gemma Metal plan and resident schedules:

- `include/local_gguf_engine/engine/gemma2_metal_plan.hpp`
- `src/engine/gemma2_metal_plan.cpp`
- `tests/engine/gemma2_metal_plan_tests.cpp`
- `tests/support/tensor_test_support.*`

Gemma Metal runtime and resident prefill/decode/verifier execution:

- `include/local_gguf_engine/engine/gemma2_runtime.hpp`
- `src/engine/gemma2_runtime.cpp`
- `src/tensor/backends/metal/metal_command_context.*`
- `include/local_gguf_engine/tensor/metal_trace_capture.hpp`
- `docs/METAL_GPU_TRACE.md`
- `tests/engine/gemma2_metal_plan_tests.cpp`

Gemma chat session and generation orchestration:

- `include/local_gguf_engine/engine/gemma2_chat_session.hpp`
- `src/engine/gemma2_chat_session.cpp`
- `include/local_gguf_engine/engine/sampling.hpp`
- `src/engine/sampling.cpp`
- `include/local_gguf_engine/engine/retrieval_proposer.hpp`
- `src/engine/retrieval_proposer.cpp`
- `tests/engine/retrieval_proposer_tests.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`

Gemma reference numerics:

- `include/local_gguf_engine/engine/gemma2_reference.hpp`
- `src/engine/gemma2_reference.cpp`
- `tests/engine/gemma2_reference_tests.cpp`
- `tests/support/gemma2_reference_test_support.*`

Gemma tokenizer/chat template:

- `include/local_gguf_engine/tokenizer/tokenizer.hpp`
- `src/tokenizer/tokenizer.cpp`
- `tests/tokenizer/tokenizer_tests.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`

GGUF parser and payload views:

- `include/local_gguf_engine/model_format/gguf_header.hpp`
- `include/local_gguf_engine/model_format/gguf_metadata.hpp`
- `include/local_gguf_engine/model_format/gguf_reader.hpp`
- `include/local_gguf_engine/model_format/gguf_tensor.hpp`
- `src/model_format/gguf_reader.cpp`
- `tests/model_format/gguf_reader_tests.cpp`
- `tests/fixtures/*.hex`

GGUF tensor bridge and Q4_K/Q6_K layout:

- `include/local_gguf_engine/model_format/gguf_tensor_bridge.hpp`
- `src/model_format/gguf_tensor_bridge.cpp`
- `include/local_gguf_engine/tensor/dtype.hpp`
- `include/local_gguf_engine/tensor/layout.hpp`
- `include/local_gguf_engine/tensor/quantization.hpp`
- `src/tensor/dtype.cpp`
- `src/tensor/layout.cpp`
- `src/tensor/quantization.cpp`
- `tests/model_format/gguf_tensor_bridge_tests.cpp`
- `tests/tensor/tensor_tests.cpp`

Tensor descriptors, storage, and typed access:

- `include/local_gguf_engine/tensor/tensor.hpp`
- `include/local_gguf_engine/tensor/shape.hpp`
- `include/local_gguf_engine/tensor/f16.hpp`
- `include/local_gguf_engine/tensor/typed_access.hpp`
- `include/local_gguf_engine/tensor/tensor_error.hpp`
- `src/tensor/tensor.cpp`
- `src/tensor/shape.cpp`
- `src/tensor/f16.cpp`
- `src/tensor/typed_access.cpp`
- `src/tensor/tensor_error.cpp`
- `tests/tensor/tensor_tests.cpp`

Tensor ops and CPU/reference behavior:

- `include/local_gguf_engine/tensor/ops/*.hpp`
- `src/tensor/ops/*.cpp`
- `src/tensor/ops/validation.hpp`
- `tests/tensor/tensor_tests.cpp`
- `tests/tensor/tensor_backend_tests.cpp`

Backend buffers and residency:

- `include/local_gguf_engine/tensor/backend.hpp`
- `include/local_gguf_engine/tensor/backend_buffer.hpp`
- `src/tensor/backend.cpp`
- `src/tensor/backend_buffer.mm`
- `src/tensor/backends/cpu_backend.*`
- `src/tensor/backends/metal_backend.*`
- `src/tensor/backends/metal/metal_buffer.*`
- `tests/tensor/tensor_backend_tests.cpp`

Metal kernels and backend-dispatched Metal ops:

- `src/tensor/backends/metal/metal_context.*`
- `src/tensor/backends/metal/metal_command_context.*`
- `src/tensor/backends/metal/metal_tensor_ops.*`
- `src/tensor/backends/metal_backend.*`
- `tests/tensor/tensor_backend_tests.cpp`
- `tests/engine/gemma2_metal_plan_tests.cpp`

Token sampling and top-logit selection:

- `include/local_gguf_engine/engine/sampling.hpp`
- `src/engine/sampling.cpp`
- `src/tensor/backends/metal/metal_tensor_ops.*`
- `tests/engine/sampling_tests.cpp`
- `tests/engine/lossless_sampling_tests.cpp`
- `tests/tensor/tensor_backend_tests.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`

Speculative sampler/control semantics:

- `include/local_gguf_engine/engine/speculative_controller.hpp`
- `src/engine/speculative_controller.cpp`
- `include/local_gguf_engine/engine/sampling.hpp`
- `src/engine/sampling.cpp`
- `tests/engine/speculative_controller_tests.cpp`
- `tests/engine/sampling_tests.cpp`

Target-verifier topology frontier:

- `include/local_gguf_engine/engine/target_verifier.hpp`
- `src/engine/target_verifier*.cpp`
- `src/engine/gemma2_verification_state_access.hpp`
- `src/tensor/backends/metal/metal_command_context.*`
- `src/tensor/backends/metal/metal_test_hooks.hpp`
- `tests/engine/gemma2_metal_plan_tests.cpp`
- `tests/engine/speculative_controller_tests.cpp`

Prepared target/draft plans, feature banks, and joint cache substrate:

- `include/local_gguf_engine/engine/prepared_gemma2_plan.hpp`
- `include/local_gguf_engine/engine/feature_bank.hpp`
- `src/engine/prepared_gemma2_plan.cpp`
- `src/engine/feature_bank.cpp`
- `src/engine/feature_bank_runtime.hpp`
- `src/engine/draft_cache_runtime.hpp`
- `src/engine/prepared_plan_test_hooks.*`
- `tests/engine/feature_bank_tests.cpp`
- `tests/engine/gemma2_metal_plan_tests.cpp`

Standalone Gemma 3 TLI drafting and shared chain verification:

- `include/local_gguf_engine/engine/standalone_draft_artifact.hpp`
- `include/local_gguf_engine/engine/standalone_draft_types.hpp`
- `include/local_gguf_engine/engine/gemma3_draft.hpp`
- `include/local_gguf_engine/engine/standalone_chat.hpp`
- `include/local_gguf_engine/engine/chain_chat.hpp`
- `src/engine/standalone_draft_artifact.cpp`
- `src/engine/gemma3_draft.mm`
- `src/engine/standalone_chat.cpp`
- `src/engine/chain_chat.mm`
- `scripts/gemma3_reference_golden.py`
- `tests/engine/standalone_draft_artifact_tests.cpp`
- `tests/engine/gemma3_draft_tests.cpp`
- `tests/engine/gemma3_golden_hashes.hpp`
- `tests/engine/standalone_chat_tests.cpp`
- `tests/scripts/gemma3_reference_golden_tests.py`

Offline Gemma 3 checkpoint reference equations:

- `scripts/gemma3_reference_golden.py`
- `scripts/reference_gguf.py`
- `scripts/reference_requirements.txt`
- `tests/scripts/gemma3_reference_golden_tests.py`
- `tests/scripts/reference_gguf_tests.py`
- `tests/engine/gemma3_golden_hashes.hpp`

Generation diagnostics and compatible `learned_*` metric fields:

- `include/local_gguf_engine/engine/gemma2_runtime.hpp`
- `src/engine/gemma2_runtime.cpp`
- `src/engine/gemma2_chat_session.cpp`
- `src/frontends/cli/generate.cpp`
- `scripts/quick_benchmark.py`
- `tests/scripts/quick_benchmark_script_tests.py`

Generation and interactive CLI:

- `include/local_gguf_engine/frontends/cli/generate.hpp`
- `src/frontends/cli/generate.cpp`
- `src/frontends/cli/generate_main.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`

Quick local/reference benchmark:

- `include/local_gguf_engine/benchmarks/feature_capture_benchmark.hpp`
- `include/local_gguf_engine/benchmarks/immutable_weight_benchmark.hpp`
- `include/local_gguf_engine/benchmarks/output_head_benchmark.hpp`
- `include/local_gguf_engine/benchmarks/quick_benchmark.hpp`
- `include/local_gguf_engine/benchmarks/target_verifier_benchmark.hpp`
- `include/local_gguf_engine/benchmarks/width_two_verifier_benchmark.hpp`
- `src/benchmarks/immutable_weight_benchmark.mm`
- `src/benchmarks/feature_capture_benchmark.cpp`
- `src/benchmarks/output_head_benchmark.cpp`
- `src/benchmarks/quick_benchmark.cpp`
- `src/benchmarks/target_verifier_benchmark.cpp`
- `src/benchmarks/width_two_verifier_benchmark.cpp`
- `src/frontends/cli/benchmark_main.cpp`
- `scripts/quick_benchmark.py`
- `scripts/chat_benchmark_common.py`
- `scripts/memory_benchmark.py`
- `scripts/output_head_benchmark.py`
- `scripts/storage_policy_benchmark.py`
- `scripts/capture_decode_gpu_trace.py`
- `tests/benchmarks/quick_benchmark_tests.cpp`
- `tests/scripts/quick_benchmark_script_tests.py`
- `tests/scripts/memory_benchmark_script_tests.py`
- `tests/scripts/output_head_benchmark_script_tests.py`
- `tests/scripts/storage_policy_benchmark_script_tests.py`

GGUF inspection CLI:

- `include/local_gguf_engine/frontends/cli/gguf_inspect.hpp`
- `src/frontends/cli/gguf_inspect.cpp`
- `src/frontends/cli/gguf_inspect_main.cpp`
- `tests/frontends/cli/gguf_inspect_tests.cpp`

Shared test fixtures:

- `tests/support/test_support.*`
- `tests/support/tensor_test_support.*`
- `tests/support/gemma2_reference_test_support.*`
- `tests/fixtures/*.hex`

Project docs:

- `docs/ENGINE_PRINCIPLES.md` for durable engineering judgment
- `docs/NEXT_TASK.md` for verified current state and the active implementation scope
- `docs/SPECULATIVE_DECODING_RETROSPECTIVE.md` for the speculative-decoding retrospective
- `docs/METAL_GPU_TRACE.md` for the production decode trace workflow
- `docs/LLAMA_CPP_REFERENCE_BASELINE.md` for the pinned external benchmark
- `docs/GRAPH.md` for this locator

## Test Map

| Change area | Tests |
| --- | --- |
| GGUF parser/payload views | `tests/model_format/gguf_reader_tests.cpp` |
| GGUF tensor bridge | `tests/model_format/gguf_tensor_bridge_tests.cpp` |
| Tensor layout/dtypes/quantization/accessors | `tests/tensor/tensor_tests.cpp` |
| Tensor ops/backends/Metal parity | `tests/tensor/tensor_backend_tests.cpp` |
| Engine model binding | `tests/engine/engine_model_tests.cpp` |
| Gemma Metal plan/runtime/resident command context | `tests/engine/gemma2_metal_plan_tests.cpp` |
| Gemma chat session, KV reuse, stops, streaming, context, regeneration | `tests/frontends/cli/generate_cli_tests.cpp` |
| Gemma reference numerics | `tests/engine/gemma2_reference_tests.cpp` |
| Token sampling and top-logit helpers | `tests/engine/sampling_tests.cpp`, `tests/tensor/tensor_backend_tests.cpp`, `tests/frontends/cli/generate_cli_tests.cpp` |
| Speculative proposal graphs, controller transitions and sampler tickets | `tests/engine/speculative_controller_tests.cpp`, `tests/engine/sampling_tests.cpp` |
| Target-verifier topology, transactions, K/V, Metal parity, and failure injection | `tests/engine/gemma2_metal_plan_tests.cpp`, `tests/engine/speculative_controller_tests.cpp` |
| Prepared plans, feature banks, draft caches, joint transactions, and capture failures | `tests/engine/feature_bank_tests.cpp`, `tests/engine/gemma2_metal_plan_tests.cpp`, `tests/engine/standalone_draft_artifact_tests.cpp` |
| Lossless sampling probability laws, proposal leases, and residual correction | `tests/engine/lossless_sampling_tests.cpp` |
| Offline GGUF directory parsing and open-mapping identity | `tests/scripts/reference_gguf_tests.py` |
| Complete Gemma 3 Q8 checkpoint, tokenizer bridge, hybrid private K/V, full-vocabulary sampling, native lifecycle, and independent CPU goldens | `tests/engine/gemma3_draft_tests.cpp`, `tests/engine/gemma3_golden_hashes.hpp`, `tests/engine/standalone_draft_artifact_tests.cpp`, `tests/engine/standalone_chat_tests.cpp`, `tests/frontends/cli/generate_cli_tests.cpp`, `tests/scripts/gemma3_reference_golden_tests.py` |
| Quick kernel summaries and real-chat timing parser validation | `tests/benchmarks/quick_benchmark_tests.cpp`, `tests/scripts/quick_benchmark_script_tests.py` |
| Production startup/process-memory benchmark parsing | `tests/scripts/memory_benchmark_script_tests.py` |
| Tokenizer/chat template | `tests/tokenizer/tokenizer_tests.cpp`, plus CLI tests when prompt rendering changes |
| Generate CLI | `tests/frontends/cli/generate_cli_tests.cpp` |
| Inspect CLI | `tests/frontends/cli/gguf_inspect_tests.cpp` |
| Shared fixtures | `tests/support`, `tests/fixtures` |
