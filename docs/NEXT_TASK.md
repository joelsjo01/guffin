# Next Task: Cooperative Decode RMSNorm and Measured Fusion

## Verified Current State

Production single-token decode now uses engine-owned fused Metal attention.
One workgroup owns each KV head and its paired query heads, streams 32-token
K/V blocks, applies Gemma logit softcapping, and accumulates stable online
softmax without materializing scores or probabilities.

The path has:

- one direct-output dispatch through 256 attended tokens;
- 2-32 partial workgroups plus one reduction dispatch above that boundary;
- persistent 516 KiB partial scratch and no dispatch-time allocation;
- unchanged sliding/global attention windows and F32 K/V;
- no decode call to the generic score, softmax, or value encoders;
- one compute encoder, zero blit encoders, one commit, and one wait per token.

Exact 16-Q/8-KV/256-dimensional parity passes at 1, 31, 32, 33, 256,
257, 4096, and 8192 tokens. The resident two-layer decode test independently
checks logits, top-logit selection, and appended K/V. All 316 tests and the
repository CI pass.

The canonical short greedy benchmark is neutral within noise: 4.94 tok/s mean
versus 4.92 tok/s before (+0.5%). Resident scratch is about 508 KiB smaller.
Medium/long real-path attention gains remain unmeasured because the current
2 tok/s prefill makes repeated 512/2048-token benchmark setup both slow and
thermally confounded. Do not claim a long-context speedup until it is measured.

Prefill intentionally retains the generic three-pass attention path. It is
live code and must not be deleted before a separately proven batched fused
implementation.

## Objective

Take ranked finding 3 from `LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md`: replace the
serial decode RMSNorm kernel with a cooperative SIMD/threadgroup reduction,
then determine which adjacent norm/residual operation can be fused for a real
whole-token gain without changing Gemma arithmetic.

Before implementation, produce a focused design in this file using the pinned
llama.cpp Metal implementation and measurements from this engine. The design
must distinguish decode from prefill and delete the superseded decode-specific
pipeline/tests rather than retain a fallback.

## Required Investigation

1. Attribute decode GPU time to RMSNorm, residual add, projection, and barrier
   boundaries with one benchmark process at a time.
2. Inspect llama.cpp's cooperative Metal row reductions and relevant graph
   fusion rules at pinned commit
   `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`.
3. Design for Gemma 2's actual decode widths: 3584 hidden, 4096 attention
   output, and 14336 feed-forward, with one F32 row.
4. Preserve multiplicative Gemma RMSNorm semantics, F32 accumulation, in-place
   safety, and the current concurrent-encoder dependency model.
5. Compare standalone cooperative RMSNorm with concrete fused candidates,
   especially post-projection norm plus residual add. Reject fusion that
   duplicates reads or forces an extra temporary.
6. Specify independent parity tests, exact structural count changes, deletion
   targets, and thermally controlled real-model acceptance measurements.

## Constraints

- Do not change attention, projection math, KV dtype/layout, prefill scheduling,
  sampling, tokenizer, or chat behavior.
- Do not add a generic graph optimizer.
- Keep production execution engine-owned and Metal-first.
- Keep generic tensor RMSNorm live while prefill or independent backend tests
  still use it.
- Never overlap real-model benchmark processes. Cool the machine between
  sustained runs.

## Stop Condition

Stop the investigation when this file contains one implementation-ready
cooperative RMSNorm/fusion design grounded in this engine's exact layouts,
upstream evidence, deletion map, correctness gates, and measurable performance
gate. Do not implement that next finding in the same investigation turn unless
the user explicitly asks for implementation.
