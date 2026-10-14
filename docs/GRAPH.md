# Project graph

Locator for implementation, tests, fixtures, commands and project documents.

## Execution entrypoints

```text
frontends/cli: options -> interaction -> metrics
                         |
                         v
engine/gemma_chat_session
  -> tokenizer
  -> model + gemma_model + prepared_gemma_plan
  -> gemma2_runtime + gemma_target_runtime
       -> gemma2_metal_plan
       -> tensor/backends/metal
  -> standalone_chat
       -> gemma3_draft
       -> chain_chat
            -> speculative_controller + sampling
            -> target_verifier
```

## Build and frontend files

| Path | Locate |
| --- | --- |
| `Makefile` | Build, dependency, install and uninstall targets |
| `scripts/ci.sh` | Repository checks |
| `src/frontends/cli/main.cpp` | Installed `guffin` entrypoint |
| `src/frontends/cli/cli.cpp` | Interactive/one-shot IO, commands and streaming |
| `src/frontends/cli/options.*` | Arguments and defaults |
| `src/frontends/cli/metrics.cpp` | Timings and attribution formatting |
| `src/frontends/cli/gguf_inspect*.cpp` | Integrated and standalone GGUF inspection |
| `src/frontends/cli/benchmark_main.cpp` | `guffin-bench` entrypoint |
| `tests/test_main.cpp` | Native test registry |

## Engine and tensor files

| Path | Locate |
| --- | --- |
| `src/model_format` | GGUF parsing, metadata, payload views and file ownership |
| `src/tokenizer` | Encoding, decoding and Gemma chat formatting |
| `src/runtime` | Backend errors and SHA-256 |
| `include/local_gguf_engine/runtime/owned_read_only_bytes.hpp` | Shared read-only byte ownership and file advice |
| `src/engine/model.*`, `gemma_model.*` | Model capabilities, validation, identity and weight binding |
| `src/engine/gemma_chat_session.*` | Conversations, prefix reuse, context, stops and regeneration |
| `src/engine/prepared_gemma_plan.*` | Target/draft preparation |
| `src/engine/gemma2_metal_plan.*` | Target weights, scratch, KV, preparation and transactions |
| `src/engine/gemma2_runtime.*` | Target prefill/decode, selection, receipts and timings |
| `src/engine/gemma_target_runtime.*` | Shared target encoding and Gemma 3 execution |
| `src/engine/sampling.*` | Sampling transforms, history, RNG and lossless rejection/residual law |
| `src/engine/retrieval_proposer.*` | Suffix retrieval and profitability gate |
| `src/engine/speculative_controller.*` | Proposal graphs and token/cache transitions |
| `src/engine/target_verifier*` | Verifier arena, tickets, private rows and prefix publication |
| `src/engine/chain_chat.*` | Chain verification and sampler commit |
| `src/engine/standalone_draft_artifact.*` | Drafter authentication and tokenizer mapping |
| `src/engine/gemma3_draft.*` | Native Q8 drafter, proposal laws and draft KV |
| `src/engine/standalone_chat.*` | Target/draft ledgers and joint lifecycle |
| `src/tensor` | Descriptors, storage/views, precision and layouts |
| `src/tensor/quantization.*`, `kquant_packing.*`, `kquant_packing_internal.hpp`, `weight_preparation.hpp` | Conversion, packing and preparation workers |
| `src/tensor/ops` | Operation interfaces and CPU/reference equations |
| `src/tensor/backends/metal/metal_kernels.*` | Embedded Metal shaders |
| `src/tensor/backends/metal/metal_command_context.*` | Encoding, submission and readback |
| `src/tensor/backends/metal/metal_buffer.*` | Buffers, mapped views, initialization and residency |
| `src/tensor/backends/metal/metal_context.*` | Device, queue and compute pipelines |
| `src/benchmarks` | Developer probes and verifier diagnostics |
| `tests/reference`, `tests/support` | Independent Gemma 2 equations, fixtures and fault helpers |

## Test routes

| Behavior | Path |
| --- | --- |
| GGUF and model binding | `tests/model_format`, `tests/engine/engine_model_tests.cpp` |
| Tokenizers/templates | `tests/tokenizer/tokenizer_tests.cpp` |
| CLI/conversations/stopping | `tests/frontends/cli/cli_tests.cpp` |
| Sampling and lossless laws | `tests/engine/sampling_tests.cpp`, `lossless_sampling_tests.cpp` |
| Speculative controller | `tests/engine/speculative_controller_tests.cpp` |
| Target/verifier ownership, cache, rollback and fused attention | `tests/engine/gemma2_metal_plan_tests.cpp` |
| Gemma 2 numerical references | `tests/engine/gemma2_reference_tests.cpp`, `tests/reference` |
| Gemma 3 target/model contracts | `tests/engine/gemma3_model_plan_tests.cpp`, `gemma3_target_runtime_tests.cpp` |
| Drafter authentication/lifecycle | `tests/engine/gemma3_draft_tests.cpp`, `standalone_draft_artifact_tests.cpp`, `standalone_chat_tests.cpp` |
| Metal equations and storage | `tests/tensor/tensor_backend_tests.cpp`, `gemma3_metal_tests.cpp` |
| Descriptors, conversion/packing and tensor values | `tests/tensor/tensor_tests.cpp` |
| Developer probes and scripts | `tests/benchmarks`, `tests/scripts` |
| Release corpus, collector, metrics and reporter | `tests/scripts/chat_corpus_tests.py`, `release_chat_*_tests.py` |

## Commands, fixtures and documents

| Path | Locate |
| --- | --- |
| `scripts/gemma3_reference_golden.py`, `gemma3_target_reference.py` | Independent Gemma 3 checkpoint goldens |
| `scripts/reference_gguf.py`, `reference_requirements.txt` | Offline GGUF reader and optional reference dependencies |
| `scripts/quick_benchmark.py` | Two-turn development benchmark |
| `scripts/release_chat_benchmark.py`, `release_chat_metrics.py`, `release_chat_report.py` | Release collection, native adapters and offline reports |
| `benchmarks/chat` | Frozen workload and source snapshots |
| `benchmarks/results/gemma2-release-v2` | Final Gemma 2 campaign |
| `benchmarks/results/gemma3-normal-release-v2` | Final ordinary Gemma 3 campaign |
| `benchmarks/results/gemma3-speculative-release` | Final speculative Gemma 3 campaign |
| `scripts/memory_benchmark.py` | Startup and process-memory probe |
| `scripts/output_head_benchmark.py`, `storage_policy_benchmark.py` | Head/storage developer probes |
| `scripts/capture_decode_gpu_trace.py`, [trace guide](METAL_GPU_TRACE.md) | Decode Metal trace capture |
| `scripts/llama_reference_benchmark.py`, [reference record](LLAMA_CPP_REFERENCE_BASELINE.md) | Pinned historical reference |
| [AGENTS](../AGENTS.md), [principles](ENGINE_PRINCIPLES.md) | Operating contract and engineering principles |
| [Current baton](NEXT_TASK.md) | Current state and next task |
| [README](../README.md) | Public architecture and usage |
| [Benchmark guide](BENCHMARKING.md), [performance](PERFORMANCE.md) | Protocol and results |
| [Retrospective](SPECULATIVE_DECODING_RETROSPECTIVE.md), [development archive](archive/DEVELOPMENT_HISTORY.md) | Design narrative and historical record |
