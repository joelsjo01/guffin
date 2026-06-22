# Agent Instructions

Read this first. Keep future context small: load deeper docs only when the task
needs them.

## Aim

Build a from-scratch local GGUF inference engine optimized for this MacBook Air
M4. The production runtime owns GGUF loading, tensors, operators, tokenization,
KV cache, decoding, sampling, and backend selection.

Performance goal: beat `llama.cpp` on selected local GGUF models and
quantization formats on this machine. Broad portability and model coverage are
secondary.

Current main target: fast local Gemma 2 9B Q4 on Metal.

## Hard Rules

- Do not wrap `llama.cpp`, Ollama, MLX, PyTorch, TensorFlow, Core ML, or remote
  inference APIs for production inference.
- Reference libraries may be used only for benchmarks, compatibility checks,
  golden outputs, and debugging.
- For Gemma 2 9B Q4, the product path is Metal-first and model-resident. CPU
  work is correctness scaffolding only.
- Keep engine logic separate from CLI, server, UI, scripts, and benchmarks.
- Runtime inference must work locally without network access.
- Never commit model weights, generated binaries, local env files, or benchmark
  result output.
- Preserve user changes. Never revert unrelated work.

## Default Read Path

1. `AGENTS.md`
2. `docs/ENGINE_PRINCIPLES.md`
3. `docs/NEXT_TASK.md`
4. `docs/GRAPH.md` only when locating modules, tests, or task-specific docs
5. `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` only for Gemma planning/work
6. Relevant code/tests only

Do not bulk-read docs.

## Workflow

1. Inspect relevant code and tests.
2. Make the smallest coherent change that advances the Gemma Q4 Metal-first
   direction.
3. Add or update tests for behavior changes.
4. Update docs only when public behavior, architecture, backend policy, model
   support, toolchain direction, or the next task changes.
5. Run `bash scripts/ci.sh` after implementation changes. Doc-only edits do
   not need CI.
6. Report changed files, tests run, and known gaps.

## Testing Expectations

- Add deterministic tests for behavior changes.
- Use CPU/reference fixtures to prove correctness when needed.
- Metal work needs parity or golden checks plus unavailable-device behavior.
- CLI/API changes need success and failure-path tests.
- Performance work needs a benchmark or a clear reason one was not run.

Manual testing is useful, but it does not replace automated tests.

## Dependencies

Allowed: build tooling, test frameworks, small parsers, CLI parsing,
logging/tracing, and dev-only comparison tools.

Restricted: production inference runtimes, tensor frameworks used as production
executors, model-serving frameworks, and remote inference APIs.

## Comments

Explain non-obvious numerics, cache behavior, memory
layout, backend constraints, hardware assumptions, or performance choices.
Do not have comments that merely restate the code.

## Doc Map

- Engine principles: `docs/ENGINE_PRINCIPLES.md`
- Current implementation baton: `docs/NEXT_TASK.md`
- Where to look: `docs/GRAPH.md`
- Gemma milestone map: `docs/GEMMA_Q4_MILESTONES.md`
- Gemma planning and findings: `docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md`
