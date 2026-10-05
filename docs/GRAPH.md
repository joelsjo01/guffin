# Project Graph

Use this map to locate current implementation and tests. Historical designs,
commands and measurements live in [development history](archive/DEVELOPMENT_HISTORY.md).
Read [engine principles](ENGINE_PRINCIPLES.md) and [current work](NEXT_TASK.md)
for policy and scope. The [README](../README.md) diagrams show the public
architecture and H1 cycle; the [retrospective](SPECULATIVE_DECODING_RETROSPECTIVE.md)
explains the design decisions.

## Execution flow

```text
frontends/cli: options -> interaction -> metrics
                         |
                         v
engine/gemma_chat_session
  -> tokenizer: rendering, encoding, incremental decoding
  -> engine/model + gemma_model: capabilities and weight binding
  -> engine/prepared_gemma_plan: atomic target/draft preparation
  -> engine/gemma2_runtime + gemma_target_runtime: native target execution
       -> gemma2_metal_plan: persistent weights, KV and scratch
       -> tensor/backends/metal: kernels, buffers, commands and residency
  -> engine/standalone_chat: target/draft token mapping and ledgers
       -> gemma3_draft: native drafter and its private caches
       -> chain_chat: lossless chain orchestration
            -> speculative_controller + sampling
            -> target_verifier: private target rows and cache publication
```

Shared session/configuration/weight/preparation APIs use `Gemma` names.
`gemma2_runtime` and `gemma2_metal_plan` retain their original execution names;
they contain the original Gemma 2 schedules and the shared target foundations
used by Gemma 3. Model-family differences remain explicit in capabilities,
validation and execution dispatch. Versioned authentication domains are stable
through source refactoring.

## Build and command entrypoints

| Path | Responsibility |
| --- | --- |
| `Makefile` | Object builds, compiler-generated dependencies, test isolation, install/uninstall |
| `scripts/ci.sh` | Python checks and explicit debug native tests |
| `src/frontends/cli/main.cpp` | Installed `guffin` entrypoint |
| `src/frontends/cli/cli.cpp` | Interactive and one-shot IO, commands, streaming and error presentation |
| `src/frontends/cli/options.*` | Argument parsing and CLI defaults; passes engine session options directly |
| `src/frontends/cli/metrics.cpp` | Compact timings and opt-in attribution formatting |
| `src/frontends/cli/gguf_inspect.cpp` | GGUF inspection without inference |
| `src/frontends/cli/gguf_inspect_main.cpp` | Minimal standalone developer inspector |
| `src/frontends/cli/benchmark_main.cpp` | Separate `guffin-bench` developer executable |
| `tests/test_main.cpp` | Native test registry |

Production and test objects are separate. Test objects enable fault injection;
references and diagnostic episode drivers are linked only where needed.
Standalone inspection links file-format and tensor value types without Metal.
Build output stays under `build/`; installation copies the executable.

## Module ownership

| Location | Owns |
| --- | --- |
| `src/model_format` | GGUF parsing, metadata, tensor directories and validated payload views |
| `src/tokenizer` | Tokenizer loading, encoding/decoding and supported Gemma chat formatting |
| `src/runtime` | Shared backend errors and SHA-256 |
| `src/engine/model.*` | Model configuration/capabilities, validation and source payload identity |
| `src/engine/gemma_model.*` | Shared weight binding with explicit family validation |
| `src/engine/gemma_chat_session.*` | Conversation state, prefix reuse, context handling, stops, streaming and regeneration |
| `src/engine/prepared_gemma_plan.*` | Atomic preparation and ownership of target/draft aggregates |
| `src/engine/gemma2_metal_plan.*` | Move-only target facade, stable weights/scratch/KV storage and transaction state |
| `src/engine/gemma2_runtime.*` | Target prefill/decode loops, selection, completion receipts and timings |
| `src/engine/gemma_target_runtime.*` | Shared target block encoding, Gemma 3 execution and retained width-two/fused-sums routing |
| `src/engine/sampling.*` | Sampling transforms, history, RNG forks and lossless rejection/residual sampling |
| `src/engine/retrieval_proposer.*` | Bounded suffix retrieval and its profitability gating |
| `src/engine/speculative_controller.*` | Validated proposal graphs and token/callback/cache transitions |
| `src/engine/target_verifier*` | Move-only verifier arena, owner-retaining tickets, private rows and accepted-prefix publication |
| `src/engine/chain_chat.*` | Draft-independent chain verification and authoritative sampler commit |
| `src/engine/standalone_draft_artifact.*` | Complete drafter authentication and bounded tokenizer mapping |
| `src/engine/gemma3_draft.*` | Native Gemma 3 Q8 drafter, proposal laws and private draft caches |
| `src/engine/standalone_chat.*` | Target/draft ledgers, native-context restoration and joint lifecycle |
| `src/tensor` | Descriptor admission, storage/views, precision and quantized layouts |
| `src/tensor/ops` | Tensor operation interfaces and CPU/reference implementations |
| `src/tensor/backends/metal/metal_kernels.*` | Embedded numerical Metal source |
| `src/tensor/backends/metal/metal_command_context.*` | Persistent command encoding, submission and compact readback |
| `src/tensor/backends/metal/metal_buffer.*` | Allocation/view ownership and residency lifetime |
| `src/tensor/backends/metal/metal_context.*` | Device and compute pipeline ownership |
| `src/benchmarks` | Developer probes and diagnostic verifier episodes using production mechanisms |
| `tests/reference` | Independent Gemma 2 numerical helpers |
| `tests/support` | Fixtures and test-only fault seams |

Gemma 3 drafting owns its private caches in `gemma3_draft`; `standalone_chat`
coordinates their durable frontier with the target execution receipts.

Target and verifier facades cannot copy mutable execution state. Tickets retain
both setup-owned storage objects, so plan moves/destruction preserve live
transactions and completion views. Their reusable runtime borrows stable
pointers during the lease and clears them when it ends. Caller access remains
serialized; arena reuse ends a borrowed completion view's validity.

Tensor construction uses one descriptor validator across host storage, external
views and Metal allocation/view boundaries. It accepts both canonical and
native GGUF quantized layouts, including output tails and padded K-quant tiles.
Owned host views check backing size without repeating full descriptor admission.

## Correctness routes

| Behavior | Tests |
| --- | --- |
| GGUF parsing and binding | `tests/model_format`, `tests/engine/engine_model_tests.cpp` |
| Tokenization and chat templates | `tests/tokenizer/tokenizer_tests.cpp` |
| CLI, streaming, conversation lifecycle and stopping | `tests/frontends/cli/cli_tests.cpp` |
| Sampling, lossless distributions and RNG/history | `tests/engine/sampling_tests.cpp`, `lossless_sampling_tests.cpp` |
| Proposal graph/controller transitions | `tests/engine/speculative_controller_tests.cpp` |
| Target/verifier moves and owner lifetimes, rollback and fault handling | `tests/engine/gemma2_metal_plan_tests.cpp` |
| Gemma 2 independent numerics | `tests/engine/gemma2_reference_tests.cpp`, `tests/reference` |
| Gemma 3 model contracts and native target equations | `tests/engine/gemma3_model_plan_tests.cpp`, `gemma3_target_runtime_tests.cpp` |
| Complete Gemma 3 drafter and target/draft lifecycle | `tests/engine/gemma3_draft_tests.cpp`, `standalone_draft_artifact_tests.cpp`, `standalone_chat_tests.cpp` |
| Metal numerics, packing and F16 behavior | `tests/tensor/tensor_backend_tests.cpp`, `gemma3_metal_tests.cpp` |
| Descriptor mutation, backing ownership, tensor values and CPU equations | `tests/tensor/tensor_tests.cpp` |
| Developer probes and scripts | `tests/benchmarks`, `tests/scripts` |

## Independent references and measurements

[Performance evidence](PERFORMANCE.md) records saved conversation results,
complete-generation clocks, configuration identities and comparison limits.

- `scripts/gemma3_reference_golden.py`: full-checkpoint Gemma 3 Q8 CPU goldens.
- `scripts/gemma3_target_reference.py`: bounded Q4_K/Q6_K target logits and KV goldens.
- `scripts/reference_gguf.py`: independent offline GGUF reader.
- `scripts/reference_requirements.txt`: optional pinned reference dependencies.
- `scripts/quick_benchmark.py`: canonical two-turn actual-chat driver.
- `scripts/memory_benchmark.py`: production-path startup and process memory.
- `scripts/output_head_benchmark.py` and `storage_policy_benchmark.py`: retained developer comparisons.
- `scripts/capture_decode_gpu_trace.py`: actual decode Metal trace capture.
- `scripts/llama_reference_benchmark.py` and [reference identity](LLAMA_CPP_REFERENCE_BASELINE.md): pinned external comparison.

Generated results are ignored under `benchmarks/results`. Model weights and
local training artifacts are not repository source and are left untouched.
