# Project Graph

Read when: locating code, tests, fixtures, or command surfaces for a task.
Do not read as history.

## Read Rule

1. Read `AGENTS.md`.
2. Read `docs/ENGINE_PRINCIPLES.md`.
3. Read `docs/NEXT_TASK.md`.
4. Read this file only when choosing files to inspect or edit.
5. Read `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` only for Gemma planning,
   architecture, performance, or upstream-comparison work.
6. Then read only the relevant code/tests below.

## Dependency Flow

```text
src/frontends/cli
  -> src/benchmarks
  -> src/engine/gemma2_chat_session.*

src/benchmarks
  -> src/engine/gemma2_chat_session.*

src/engine/gemma2_chat_session.*
  -> src/engine/model.cpp
  -> src/model_format
  -> src/tokenizer
  -> src/engine/sampling.*
  -> src/engine/gemma2_runtime.*
      -> src/engine/gemma2_metal_plan.*
      -> src/tensor
          -> src/tensor/ops
          -> src/tensor/backends/cpu_backend.*
          -> src/tensor/backends/metal_backend.*
          -> src/tensor/backends/metal/*

tests
  -> module under test
  -> tests/support
  -> tests/fixtures
```

Benchmark output lives under ignored `benchmarks/results`. The old standalone
benchmark harness is not a live source route.

## Build Entrypoints

| Path | Use |
| --- | --- |
| `Makefile` | build graph for engine sources, CLIs, and tests |
| `scripts/ci.sh` | repository check wrapper |
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
| `src/benchmarks` | generation benchmark cases, runner, records, derived metrics, and statistics | inference execution, CLI formatting |
| `src/engine/model.cpp` | engine model loading from GGUF model views, architecture boundary | raw GGUF parsing internals |
| `src/engine/gemma2_model.*` | Gemma 2 config parsing and weight binding | backend upload/execution |
| `src/engine/gemma2_metal_plan.*` | Metal resident weight upload, KV cache buffers, prefill/decode schedules | tokenization, CLI formatting |
| `src/engine/gemma2_runtime.*` | Gemma Metal prefill/decode execution, resident decode, metrics, argmax and selector-driven token loops | chat history, CLI parsing |
| `src/engine/gemma2_chat_session.*` | persistent chat state, prompt rendering, KV-prefix reuse, context trimming, stops, streaming, regeneration, sampler integration | CLI commands and formatting |
| `src/engine/gemma2_reference.*` | tiny CPU/reference Gemma numerics for parity tests | product CPU runtime |
| `src/engine/sampling.*` | greedy and seeded stochastic token selection, top-k/top-p, repetition penalty, token history, top-logit helpers | model execution |
| `src/tensor` | tensor descriptors/storage, dtype/layout, F16, quantization views, typed access, backend buffer wrapper | model-family policy |
| `src/tensor/ops` | CPU/reference tensor ops and backend-dispatched op front doors | device setup |
| `src/tensor/backends/cpu_backend.*` | CPU backend buffers and CPU backend execution helpers | product-speed Gemma path |
| `src/tensor/backends/metal_backend.*` | public Metal backend bridge, availability, buffer copy/allocation | CLI behavior |
| `src/tensor/backends/metal/metal_command_context.*` | resident prefill/decode command encoding, submission statistics, selected-token readback | model scheduling, chat state |
| `src/tensor/backends/metal/*` | Metal device context, buffers, kernels, command encoding, Metal op implementations | benchmark reporting |
| `src/frontends/cli` | command arguments, file IO, stdout/stderr formatting, error reporting | inference math |
| `tests` | deterministic coverage and fixtures | real model weights, generated benchmark artifacts |

## Task Routes

Gemma 2 model binding:

- `include/local_gguf_engine/engine/model.hpp`
- `include/local_gguf_engine/engine/gemma2_model.hpp`
- `src/engine/model.cpp`
- `src/engine/gemma2_model.cpp`
- `tests/engine/engine_model_tests.cpp`

Gemma Metal plan and resident schedules:

- `include/local_gguf_engine/engine/gemma2_metal_plan.hpp`
- `src/engine/gemma2_metal_plan.cpp`
- `tests/engine/gemma2_metal_plan_tests.cpp`
- `tests/support/tensor_test_support.*`

Gemma Metal runtime and resident prefill/decode execution:

- `include/local_gguf_engine/engine/gemma2_runtime.hpp`
- `src/engine/gemma2_runtime.cpp`
- `src/tensor/backends/metal/metal_command_context.*`
- `tests/engine/gemma2_metal_plan_tests.cpp`

Gemma chat session and generation orchestration:

- `include/local_gguf_engine/engine/gemma2_chat_session.hpp`
- `src/engine/gemma2_chat_session.cpp`
- `include/local_gguf_engine/engine/sampling.hpp`
- `src/engine/sampling.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`

Gemma reference numerics:

- `include/local_gguf_engine/engine/gemma2_reference.hpp`
- `src/engine/gemma2_reference.cpp`
- `tests/engine/gemma2_reference_tests.cpp`

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
- `tests/tensor/tensor_backend_tests.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`

Generation and interactive CLI:

- `include/local_gguf_engine/frontends/cli/generate.hpp`
- `src/frontends/cli/generate.cpp`
- `src/frontends/cli/generate_main.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`

Generation benchmark runner, records, and statistics:

- `include/local_gguf_engine/benchmarks/generation_benchmark.hpp`
- `src/benchmarks/generation_benchmark.cpp`
- `tests/benchmarks/generation_benchmark_tests.cpp`

GGUF inspection CLI:

- `include/local_gguf_engine/frontends/cli/gguf_inspect.hpp`
- `src/frontends/cli/gguf_inspect.cpp`
- `src/frontends/cli/gguf_inspect_main.cpp`
- `tests/frontends/cli/gguf_inspect_tests.cpp`

Shared test fixtures:

- `tests/support/test_support.*`
- `tests/support/tensor_test_support.*`
- `tests/fixtures/*.hex`

Project docs:

- `docs/ENGINE_PRINCIPLES.md` for durable engineering judgment
- `docs/NEXT_TASK.md` for the active baton
- `docs/LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md` for the current Gemma Metal structural comparison with `llama.cpp`
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
| Generation benchmark records, runner, and statistics | `tests/benchmarks/generation_benchmark_tests.cpp`, `tests/frontends/cli/generate_cli_tests.cpp` |
| Tokenizer/chat template | `tests/tokenizer/tokenizer_tests.cpp`, plus CLI tests when prompt rendering changes |
| Generate CLI | `tests/frontends/cli/generate_cli_tests.cpp` |
| Inspect CLI | `tests/frontends/cli/gguf_inspect_tests.cpp` |
| Shared fixtures | `tests/support`, `tests/fixtures` |
