# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add layout-only Q4_K/Q6_K GGUF/tensor support for the local Gemma 2 9B Q4 file.

## Why

This removes the first hard blocker: `models/gemma-2-9b-Q4.gguf` fails before
Gemma metadata, binding, tokenizer, or Metal planning can be exercised because
GGUF tensor types `12` and `14` are unsupported.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`, especially
  `## GGUF Quantization And Q4/K Format Findings`,
  `## Target Architecture Proposal`, and `## First Slice Recommendation`

## Touch

- `include/local_gguf_engine/tensor/dtype.hpp`
- `src/tensor/dtype.cpp`
- `src/tensor/layout.cpp`
- `src/tensor/quantization.cpp`
- `include/local_gguf_engine/model_format/gguf_tensor_bridge.hpp`
- `src/model_format/gguf_tensor_bridge.cpp`
- `src/model_format/gguf_reader.cpp`
- `src/frontends/cli/gguf_inspect.cpp` only if inspector support needs an
  explicit type-name/update
- `tests/model_format`
- `tests/tensor`

## Do Not

- Do not expand into unrelated cleanup.
- Do not optimize the old CPU/Llama path unless it directly proves the current
  slice.
- Do not read broad docs or benchmark artifacts unless named above.
- Do not add generation, full Gemma CPU execution, Metal kernels, or model
  architecture refactors in this slice.

## Verify

- Add deterministic tests for Q4_K/Q6_K row-size and payload-size validation,
  malformed row rejection, dtype conversion, and safe block-view access.
- Run the focused tests that cover `model_format` and `tensor`.
- Run `build/release/gguf-inspect models/gemma-2-9b-Q4.gguf` if a release
  inspector already exists; the expected improvement is that type IDs `12` and
  `14` no longer fail at payload conversion.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

The Gemma GGUF tensor directory can be inspected with Q4_K/Q6_K payload sizes
validated, tests cover the new layouts and rejection cases, and execution still
fails explicitly later for unsupported Gemma runtime work rather than at tensor
type conversion.
