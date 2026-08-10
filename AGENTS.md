# Agent Instructions

Read this first. This file is the repository operating contract. Current
implementation state belongs in `docs/NEXT_TASK.md`.

## Mission

Build a from-scratch local GGUF inference engine for this MacBook Air M4. The
production runtime owns the full inference stack from GGUF loading through
token selection.

Performance goal: beat `llama.cpp` on selected local GGUF models and
quantization formats on this machine. Broad portability and model coverage are
secondary.

Current main target: fast local Gemma 2 9B Q4 on Metal.

## Non-Negotiable Boundaries

- Production inference must remain engine-owned. Do not wrap `llama.cpp`,
  Ollama, MLX, PyTorch, TensorFlow, Core ML, model servers, or remote inference
  APIs.
- Reference libraries may support benchmarks, compatibility checks, golden
  outputs, and debugging, but must not execute the production path.
- The Gemma 2 9B Q4 product path is Metal-first and model-resident. CPU work is
  correctness scaffolding, not a parallel product runtime.
- Keep engine logic separate from CLI, server, UI, scripts, diagnostics, and benchmarks.
- Runtime inference must work locally without network access.
- Production dependencies must not provide inference execution or tensor
  framework semantics. Development and infrastructure tooling is acceptable.
- Never commit model weights, generated binaries, local environment files, or
  benchmark result output.
- Preserve user changes. Never revert or rewrite unrelated work.

## Read Path

1. `AGENTS.md`
2. `docs/ENGINE_PRINCIPLES.md`
3. `docs/NEXT_TASK.md`
4. `docs/GRAPH.md` only when locating modules, tests, fixtures, or task docs
5. Relevant code and tests only

Do not bulk-read docs or source trees when the task has a narrower route.

## Work Loop

1. Follow the objective, constraints, verification, and stop condition in
   `docs/NEXT_TASK.md`; do not silently broaden the task.
2. Inspect the relevant implementation and tests, then make the smallest
   coherent change that proves the intended behavior or hypothesis end to end.
3. Add deterministic automated coverage for behavior changes. Metal work needs
   parity or golden evidence and unavailable-device behavior; CLI/API work needs
   success and failure paths.
4. For performance work, run a comparable real-path benchmark or explain why
   the current slice cannot produce one.
5. Update durable docs only when their owned information changes.
6. Run `bash scripts/ci.sh` after implementation changes. Doc-only edits do not
   need CI.
7. Stop at the baton boundary and report changed files, verification performed,
   benchmark evidence when applicable, and known gaps.

## Document Ownership

- `AGENTS.md`: mission, non-negotiable boundaries, read path, and workflow.
- `docs/ENGINE_PRINCIPLES.md`: immutable engineering and coding principles.
  Do not modify this document unless the user explicitly requests a change to
  the principles themselves.
- `docs/NEXT_TASK.md`: verified current state, one active baton, constraints,
  verification, and stop condition.
- `docs/GRAPH.md`: mechanically maintained locator for code, tests, fixtures,
  commands, and task docs; no policy, reasoning, measurements, or project
  history.
- `docs/LLAMA_CPP_REFERENCE_BASELINE.md`: pinned external reference identity,
  protocol, result, invalidation rule, and rerun command.

Do not put milestones, benchmark results, completed-work history, or the active
bottleneck in this file.

## Benchmarking

Build and run the canonical local benchmark with:

```sh
make MODE=release benchmark
scripts/quick_benchmark.py --model models/gemma-2-9b-Q4.gguf
```

It drives the actual local chat executable through the fixed two-turn sampled
conversation, lets both responses stop naturally, and reports TTFT, prompt
tokens/s, and generation tokens/s with raw counts and durations. One fresh
process with enforced cooldowns is the quick thermal-safe screen; the
separately launched local kernel probes are diagnostics only. The pinned
llama.cpp reference is a separate, infrequent benchmark documented in
`docs/LLAMA_CPP_REFERENCE_BASELINE.md`. On this fanless machine, repeated pairs
systematically heat-soak and throttle later runs, so a retention decision uses
one candidate and one baseline in fresh processes with at least four minutes of
idle cooldown before each. Require at least 2% aggregate generation improvement,
wins on both turns, exact workload/token-hash correctness equivalence, and no
greater-than-2% regression in another headline metric. Do not substitute a
back-to-back repetition burst for thermal isolation. Never put measured results
here.
