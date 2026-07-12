# Engine Principles

Read this after `AGENTS.md`. Keep it short; this file is the current project
compass, not a history log.

## Goal

Build a from-scratch, engine-owned local GGUF inference engine optimized for
this MacBook Air M4. The main target is `models/gemma-2-9b-Q4.gguf`.

The product path is Metal-first and model-resident: packed quantized weights,
GPU-resident weights/KV/scratch, explicit prefill/decode paths, and real
end-to-end generation throughput.

The current Gemma path already has the important Metal residency boundary:
packed weights, KV cache, prefill/decode schedules, logits, and greedy top-1
selection stay on Metal, with only the selected token/logit read back. The next
project focus is measuring that real path and using the numbers to choose
optimization work.

CPU work is correctness scaffolding only: metadata checks, tokenizer checks,
tiny reference kernels, fixtures, and golden comparisons. Do not build a full
Gemma 2 9B Q4 CPU runtime as the main track.

## Hard Rules

- Production inference must stay engine-owned.
- Do not wrap `llama.cpp`, Ollama, MLX, PyTorch, TensorFlow, Core ML, or remote
  inference APIs.
- Reference runtimes are allowed only for benchmarks, compatibility checks,
  golden outputs, and debugging.
- Runtime inference must work locally without network access.
- Do not commit model weights, generated binaries, local env files, or benchmark
  result output.
- Preserve user changes.

## Architecture Bias

- Prefer focused Gemma 2 support over preserving old Llama execution paths.
- Make reusable blocks at ownership boundaries: GGUF metadata, tokenizer,
  tensor layout, weight binding, execution planning, backend buffers, Metal
  kernels, KV cache, greedy token selection, and benchmark metrics.
- Keep the Gemma Q4 hot path explicit and specialized when that is faster or
  clearer.
- Do not add a broad model graph, virtual operator stack, or generic
  architecture layer before at least two real model families prove the shape.

## Comment Style

Good comments explain:

- why a path exists
- numerical assumptions and precision choices
- quantized block layout and packing details
- tensor shape/order invariants
- Metal buffer ownership, residency, and synchronization
- backend constraints and fallback behavior
- performance tradeoffs that are not obvious from the code

Avoid comments that restate names or describe obvious control flow. Prefer a
short comment before a tricky block over many line-by-line comments.

## Current Gemma State

See `docs/CLI_MILESTONES.md` for the CLI application route map and status. The
core Metal path is no longer a checklist of missing pieces; it is a working
greedy path that now needs a persistent llama.cpp-style chat application on top
of the engine-owned session/runtime state.

- Gemma 2 metadata, tokenizer/chat template, Q4_K/Q6_K layout, and weight
  binding are in place.
- Packed quantized weights, KV cache, scratch, prefill/decode schedules, logits,
  and greedy token selection are on the Metal path.
- The active application blocker is Gate 1 from `docs/CLI_MILESTONES.md`: add
  `local-gguf-generate --chat` with a persistent Gemma session and nonzero
  cache-position append.

## Benchmark Rule

Generation performance must be measured on the real generation path.

- Normal `local-gguf-generate` output should stay stable unless the CLI contract
  intentionally changes.
- `--bench` should report metrics for the same command path.
- Report prompt eval tokens/s, decode tokens/s, total tokens/s, and TTFT.
- Keep benchmark reports under ignored `benchmarks/results`.
- Do not resurrect the deleted standalone `local-gguf-bench` scaffold unless a
  specific kernel question cannot be answered on the real generate path.

## Implementation Loop

Before implementing, ask:

1. Does this move us closer to fast local Gemma 2 9B Q4 on MacBook Air M4?
2. Does it advance the Metal/model-resident path rather than polish the old CPU
   path?
3. Is any CPU work only correctness scaffolding?
4. Does it remove a real blocker or produce measurable proof?
5. What is a good coherent slice, and where should I stop?

After implementation:

- Run the relevant tests; normally run `bash scripts/ci.sh`.
- For performance work, include a real benchmark or explain why it was not run.
- Stop after the verified slice instead of drifting into speculative cleanup.
