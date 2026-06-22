# Next Task

Use this as the implementation baton. Keep it short and update it after each
completed implementation slice.

## Objective

Add Gemma tokenizer and chat-template prompt formatting for the local Gemma 2
9B Q4 file.

## Why

Gemma 2 config parsing and weight binding now succeed far enough for
`models/gemma-2-9b-Q4.gguf` to fail explicitly at unsupported Gemma runtime
execution. The next blocker is prompt preparation: the file reports
`general.architecture = "gemma2"`, `tokenizer.ggml.model = "llama"`, and a
Gemma-specific `tokenizer.chat_template` that the current SmolLM2-only prompt
formatter does not support.

## Read

- `AGENTS.md`
- `docs/ENGINE_PRINCIPLES.md`
- `docs/GRAPH.md` only to locate files/tests
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`, especially
  `## Gemma 2 Architecture And Tokenizer Findings` and
  `## Target Architecture Proposal`

## Touch

- `include/local_gguf_engine/tokenizer/tokenizer.hpp`
- `src/tokenizer/tokenizer.cpp`
- `src/frontends/cli/generate.cpp` only to move the Gemma runtime stop after
  tokenizer/prompt validation
- `tests/tokenizer/tokenizer_tests.cpp`
- `tests/frontends/cli/generate_cli_tests.cpp` if the CLI boundary changes
- `docs/GEMMA_Q4_MILESTONES.md`
- `docs/NEXT_TASK.md`

## Do Not

- Do not expand into unrelated cleanup.
- Do not optimize the old CPU/Llama path unless it directly proves the current
  slice.
- Do not read broad docs or benchmark artifacts unless named above.
- Do not add Gemma generation execution, full Gemma CPU execution, Metal
  kernels, a broad Jinja runtime, or broad model architecture refactors in this
  slice.

## Verify

- Add deterministic tests for Gemma prompt formatting, role validation,
  BOS/EOS behavior, unsupported template rejection, and the generation prompt
  suffix.
- Run the focused tokenizer and generate CLI tests.
- Run a local Gemma load/generation command far enough to prove any remaining
  failure is an explicit unsupported Gemma runtime error after tokenizer and
  prompt preparation, not tokenizer/chat-template support.
- Run `bash scripts/ci.sh` after implementation changes.

## Stop When

The local Gemma GGUF tokenizer metadata can format and encode Gemma IT prompts
with deterministic tests, and generation still fails explicitly later for
unsupported Gemma runtime work rather than architecture, tensor binding,
tokenizer, or chat-template handling.
