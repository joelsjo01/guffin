# Engine Principles

Read this after `AGENTS.md`. Keep it short; this file is the current project
compass, not a history log.

## Goal

Build a from-scratch, engine-owned local GGUF inference engine optimized for
this MacBook Air M4. The main target is `models/gemma-2-9b-Q4.gguf`.

The product path is Metal-first and model-resident: packed quantized weights,
GPU-resident weights/KV/scratch, explicit prefill/decode paths, and real
end-to-end generation throughput.

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

- Prefer focused Gemma 2 support beside the current Llama path.
- Make reusable blocks at ownership boundaries: GGUF metadata, tokenizer,
  tensor layout, weight binding, execution planning, backend buffers, Metal
  kernels, KV cache, sampling, and benchmarks.
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

## Current Gemma Blockers

See `docs/GEMMA_Q4_MILESTONES.md` for the compact route map and status.

- `gemma2` metadata and model binding.
- Gemma 2 tokenizer/chat template support.
- Q4_K/Q6_K tensor layout and validation.
- Packed quantized Metal matmul.
- Model-resident Metal weights, KV cache, and scratch.
- Separate prefill and decode execution plans.
- Real generation benchmarking with tokens/s and time-to-first-token.

## Benchmark Rule

Generation performance must be measured on the real generation path.

- `local-gguf-generate` should always report compact throughput.
- `--bench` should produce detailed metrics for the same command path.
- Report prompt eval tokens/s, decode tokens/s, total tokens/s, and TTFT.
- Keep benchmark reports under ignored `benchmarks/results`.

## Implementation Loop

Before implementing, ask:

1. Does this move us closer to fast local Gemma 2 9B Q4 on MacBook Air M4?
2. Does it advance the Metal/model-resident path rather than polish the old CPU
   path?
3. Is any CPU work only correctness scaffolding?
4. Does it remove a real blocker or produce measurable proof?
5. What is the smallest coherent slice, and where should I stop?

After implementation:

- Run the relevant tests; normally run `bash scripts/ci.sh`.
- For performance work, include a real benchmark or explain why it was not run.
- Stop after the verified slice instead of drifting into speculative cleanup.
