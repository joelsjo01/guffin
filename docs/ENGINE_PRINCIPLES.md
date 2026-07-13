# Engine Principles

Read this after `AGENTS.md`. Keep it short; this file is the current project
compass, not a history log.

## Goal

Build a from-scratch, engine-owned local GGUF inference engine optimized for
this MacBook Air M4. The main target is `models/gemma-2-9b-Q4.gguf`.

The product path is Metal-first and model-resident: packed quantized weights,
GPU-resident weights/KV/scratch, explicit prefill/decode paths, and real
end-to-end generation throughput.

The current Gemma path has the important Metal residency boundary: packed
weights, KV cache, prefill/decode schedules, logits, and greedy top-1 selection
stay on Metal, with only the selected token/logit read back. Production decode
now encodes one complete token in one command buffer with one wait and no
transient Metal buffers. Its one-row Q4_K and Q6_K projections use measured,
format-specific SIMD-cooperative kernels; the next focus is the tied F16
vocabulary projection.

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

The core Metal path, persistent chat frontend, and integrated generation
benchmark are working. The next focus is measured optimization on the real
engine-owned session/runtime path.

- Gemma 2 metadata, tokenizer/chat template, Q4_K/Q6_K layout, and weight
  binding are in place.
- Packed quantized weights, KV cache, scratch, prefill/decode schedules, logits,
  and greedy token selection are on the Metal path.
- Persistent chat, streaming, commands, context management, greedy generation,
  and seeded sampled generation are complete.
- The six-gate benchmark foundation and real release Metal baseline are
  complete.
- Resident one-submit decode is complete and improved warm greedy generation
  from about `0.287 t/s` to `0.930 t/s` mean in a five-repetition release run.
- The Q4_K SIMD decode matvec then improved the five-repetition warm greedy
  result to `2.404 t/s` mean (`2.366 t/s` median) while preserving the
  one-submit resident path.
- The Q6_K SIMD decode matvec improved the result again to `3.130 t/s` mean
  (`3.000 t/s` median); its real `[14336, 3584]` kernel is about `4.16x` faster
  than the scalar Q6_K path.
- The active task is the tied F16 vocabulary-projection decode kernel.

## Benchmark Rule

Generation performance must be measured on the real generation path.

- Normal chat will permanently print compact prompt and generation throughput
  after Gate 2 of the benchmark roadmap.
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
