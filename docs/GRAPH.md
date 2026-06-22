# Project Graph

Read when: locating code, tests, or benchmark surfaces for a task.
Do not read as history.

## Read Rule

1. Read `AGENTS.md`.
2. Read `docs/ENGINE_PRINCIPLES.md`.
3. Read `docs/NEXT_TASK.md`.
4. Read this file only when choosing files to inspect or edit.
5. Read `docs/GEMMA_Q4_MILESTONES.md` only when choosing or updating Gemma
   roadmap status.
6. Read `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` only for Gemma planning,
   architecture, or implementation work.
7. Then read only the relevant code/tests below.

## Dependency Flow

```text
src/frontends/cli
  -> src/engine
      -> src/model_format
      -> src/tokenizer
      -> src/tensor
          -> src/tensor/backends/cpu_backend.*
          -> src/tensor/backends/metal*

benchmarks
  -> src/engine / src/tensor / src/model_format

tests
  -> module under test
  -> tests/support when shared fixtures are needed
```

## Module Ownership

| Path | Owns | Does not own |
| --- | --- | --- |
| `src/model_format` | GGUF headers, metadata, tensor directories, payload views, payload-size validation | inference, tokenizer logic |
| `src/tokenizer` | tokenizer metadata, encode/decode, prompt/chat formatting | model execution |
| `src/engine` | model binding, model configs, generation orchestration, KV cache, sampling | CLI parsing, raw GGUF parsing internals, backend device setup |
| `src/tensor` | tensor descriptors/storage, dtype/layout, typed access, quantization views, numerical ops | model-family policy |
| `src/tensor/backends/cpu_backend.*` | CPU backend buffers and CPU backend fallback helpers | product-speed Gemma runtime |
| `src/tensor/backends/metal*` | Metal availability, buffers, persistent context, kernels, backend execution helpers | CLI behavior, benchmark reporting |
| `src/frontends/cli` | command arguments, file IO, stdout/stderr formatting, error reporting | inference math |
| `benchmarks` | local measurement harnesses and benchmark report formatting | production routing |
| `tests` | deterministic coverage and fixtures | real model weights, benchmark artifacts |

## Task Routes

Gemma 2 architecture/model binding:

- `include/local_gguf_engine/engine/model.hpp`
- `src/engine/model.cpp`
- `tests/engine/engine_model_tests.cpp`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`

Gemma Q4 roadmap/status:

- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`

Gemma tokenizer/chat template:

- `include/local_gguf_engine/tokenizer/tokenizer.hpp`
- `src/tokenizer/tokenizer.cpp`
- `tests/tokenizer/tokenizer_tests.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`

Q4_K/Q6_K GGUF and tensor layout:

- `include/local_gguf_engine/model_format/gguf_tensor_bridge.hpp`
- `src/model_format/gguf_reader.cpp`
- `src/model_format/gguf_tensor_bridge.cpp`
- `include/local_gguf_engine/tensor/dtype.hpp`
- `src/tensor/dtype.cpp`
- `src/tensor/layout.cpp`
- `src/tensor/quantization.cpp`
- `tests/model_format/gguf_reader_tests.cpp`
- `tests/model_format/gguf_tensor_bridge_tests.cpp`
- `tests/tensor/tensor_tests.cpp`

Packed quantized Metal matmul:

- `include/local_gguf_engine/tensor/ops/matmul.hpp`
- `src/tensor/ops/matmul.cpp`
- `src/tensor/backends/metal_backend.*`
- `src/tensor/backends/metal/metal_context.*`
- `src/tensor/backends/metal/metal_buffer.*`
- `src/tensor/backends/metal/metal_tensor_ops.*`
- `tests/tensor/tensor_backend_tests.cpp`
- `benchmarks/tensor`

Model-resident Gemma execution:

- `src/engine`
- `src/tensor/backend_buffer.mm`
- `src/tensor/backends/metal/metal_buffer.*`
- `src/tensor/backends/metal/metal_tensor_ops.*`
- future Gemma-specific engine/backend files if added
- `tests/engine`
- `tests/tensor`

Generation CLI and benchmark metrics:

- `include/local_gguf_engine/frontends/cli/generate.hpp`
- `src/frontends/cli/generate.cpp`
- `src/frontends/cli/generate_main.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp`
- `benchmarks/frontends/cli`
- `benchmarks/benchmark_harness.cpp` only if sharing benchmark report plumbing

Benchmark harness:

- `include/local_gguf_engine/benchmarks/benchmark_harness.hpp`
- `benchmarks/benchmark_harness.cpp`
- `benchmarks/frontends/cli`
- `benchmarks/support`
- the hot surface being measured
- `tests/benchmarks/benchmark_tests.cpp`

CLI inspection:

- `include/local_gguf_engine/frontends/cli/gguf_inspect.hpp`
- `src/frontends/cli/gguf_inspect.cpp`
- `src/frontends/cli/gguf_inspect_main.cpp`
- `tests/frontends/cli/gguf_inspect_tests.cpp`

## Test Map

| Change area | Tests |
| --- | --- |
| GGUF parser/layout | `tests/model_format` |
| Tensor ops/dtypes/backends | `tests/tensor` |
| Tokenizer/chat template | `tests/tokenizer`, plus CLI tests when prompt rendering changes |
| Engine/model binding/generation | `tests/engine` |
| CLI behavior | `tests/frontends/cli` |
| Benchmark parsing/report shape | `tests/benchmarks` |
| Shared fixtures | `tests/support` |
