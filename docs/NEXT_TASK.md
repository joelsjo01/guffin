# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add Gemma 2 config parsing and weight binding for the local Gemma 2 9B Q4 file.

## Why

Q4_K/Q6_K tensor layout support now lets `models/gemma-2-9b-Q4.gguf` inspect
through payload validation. The next blocker is architecture binding:
`general.architecture = "gemma2"` and Gemma tensor names/config keys need to
bind beside the current Llama path before tokenizer, reference fixtures, or
Metal planning can be exercised.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`, especially
  `## Gemma 2 Architecture And Tokenizer Findings` and
  `## Target Architecture Proposal`

## Touch

- `include/local_gguf_engine/engine/model.hpp`
- `src/engine/model.cpp`
- Gemma-specific engine config/weight files if a split keeps the slice small
- `tests/engine/engine_model_tests.cpp`
- `tests/model_format` only if additional GGUF metadata fixtures are needed
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not expand into unrelated cleanup.
- Do not optimize the old CPU/Llama path unless it directly proves the current
  slice.
- Do not read broad docs or benchmark artifacts unless named above.
- Do not add generation, tokenizer/chat-template support, full Gemma CPU
  execution, Metal kernels, or broad model architecture refactors in this
  slice.

## Verify

- Add deterministic tests for `gemma2` metadata parsing, required config keys,
  required tensor-name binding, mixed Q4_K/Q6_K projection binding, and missing
  or malformed metadata/tensor failures.
- Run the focused tests that cover engine model binding and any touched
  `model_format` fixtures.
- Run a local Gemma load/generation command far enough to prove any remaining
  failure is an explicit unsupported Gemma execution/runtime error, not
  architecture or tensor binding.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

The local Gemma GGUF metadata and required tensor names bind into a Gemma-owned
config/weights representation beside Llama, mixed Q4_K/Q6_K tensors remain
packed, and execution still fails explicitly later for unsupported Gemma
runtime work rather than architecture, metadata, tensor type, or binding.
