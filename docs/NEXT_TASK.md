# Next Task: Finding 5 — Fused Gate/Up Projection Experiment

## Verified Current State

The serial multi-row Q4_K/Q6_K Metal projection path is deleted. Both the
resident command context and the public backend-buffer API now use one shared
Metal selector owned by `metal_context.*`:

- M=1 retains the previous 64-thread cooperative decode matvec unchanged;
- M=2 selects a compile-time two-row kernel;
- M=3-4 selects a compile-time four-row kernel;
- M=5-8 selects a compile-time eight-row kernel;
- M>=9 selects a compile-time sixteen-row kernel and tiles additional rows in
  the dispatch grid.

Q4_K and Q6_K have separate precompiled pipelines because their native GGUF
block layouts differ. Every multi-row kernel uses four 32-lane SIMD groups per
128-thread workgroup. One SIMD group owns one output column, each lane decodes
eight values from a native 256-value block once, and those decoded F32 values
update all live row accumulators before the next packed block is read. This
changes a 16-row projection from sixteen logical weight decodes to one while
preserving native 144-byte Q4_K and 210-byte Q6_K storage.

The retained sixteen-row tier uses SIMD-group reduction and register reuse,
not half-operand SIMD-group matrix multiply. It was retained because the
simpler F32 path cleared the whole-product performance gate by a large margin,
preserved deterministic model behavior, and avoided the numerical and
threadgroup-memory costs of an F16 matrix candidate. Do not add a matrix path
without new profiling evidence that projection remains the limiting family.

The deleted legacy surface includes `q4_k_value`, `q6_k_value`, both serial
`matmul_f32_q*_k_gguf_weight_kernel` functions, their two pipeline objects and
creation/error paths, and both duplicated 16x16 host dispatch branches.

## Correctness Evidence

The device-backed suite has 315 passing tests. The old three row-two tests are
replaced by one independent table-driven suite for both Q4_K and Q6_K and both
TensorStorage and backend-buffer entry points. It covers:

- M = 2, 3, 4, 7, 8, 9, 15, and 16;
- K block counts = 1, 14, 16, and 56, including all exact Gemma K widths;
- N = 1, 3, 4, and 5, covering tails around the four-column workgroup tile;
- nonzero Q4_K scale/min terms, negative Q6_K scales, and mixed large/small
  activation values.

The oracle is the independent test-support decoder. The stress sweep passes at
`0.05 + 1e-5 * abs(expected)` for Q4_K and
`0.1 + 1e-5 * abs(expected)` for Q6_K. Existing one-row packed-block and output
tail tests remain unchanged.

Resident prefill coverage now uses an exact 16-row chunk followed by a
seven-row remainder and compares the combined execution against incremental
16+7 execution. Logits, selected token/logit, and every appended K/V cache byte
match. Existing decode/reference coverage proves the retained M=1 tier. The
resident assertions still prove one command buffer per chunk, one final
synchronous wait, no transient allocation, and the expected commit/KV-write
counts.

`bash scripts/ci.sh` passes, and a separate device-backed run of
`./build/debug/local_gguf_engine_tests` passes all 315 tests on this M4.

## Real-Path Performance Evidence

Canonical release benchmark: `models/gemma-2-9b-Q4.gguf`, prompt
`Explain why the sky is blue.`, 16 prompt tokens, eight generated tokens,
greedy Metal, one warmup, three measured repetitions.

| Metric | Serial baseline | Row-reuse path | Change |
| --- | ---: | ---: | ---: |
| warm prompt throughput mean | 2.176 tok/s | 8.545 tok/s | +292.7% |
| warm prompt throughput median | 2.176 tok/s | 8.567 tok/s | +293.8% |
| runtime TTFT mean | 7.354 s | 1.873 s | -74.5% |
| runtime TTFT standard deviation | 9.9 ms | 10.2 ms | comparable |
| warm decode throughput mean | 8.282 tok/s | 8.581 tok/s | no regression |

An immediately interleaved run of the preserved pre-change binary measured
2.171 prompt tok/s and 7.371 s runtime TTFT, ruling out thermal drift as the
source of the gain. Its 7.945 decode tok/s was below the new run. Seeded decode
was effectively unchanged (7.626 versus 7.610 tok/s), which is expected because
the M=1 kernels did not change.

A one-token prompt benchmark produces ten formatted prompt tokens and exercises
the M=9-15 tier at 7.710 warm prompt tok/s. The canonical benchmark's cached
second turn evaluates 17 tokens through a 16+1 multi-chunk path at 8.533 tok/s
mean. Small-batch M=2-8 is covered by the device-backed exact-K sweep and the
resident 16+7 execution; the current formatted CLI template cannot produce a
first-turn prompt below nine tokens.

Benchmark output remains uncommitted. The preserved comparison binary is in
`/tmp`, not the repository.

## Active Bottleneck

Projection is no longer known to dominate prefill. The next work must profile
the new real path before choosing another kernel family. The previously
identified reversible opportunity is gate/up fusion: both projections consume
the same normalized 16x3,584 activation and produce matching 16x14,336 shapes,
then a standalone GELU-tanh multiply consumes both outputs.

## Objective

Measure the post-projection-optimization prefill command buffer by projection,
attention, normalization, elementwise, and synchronization families. Only if
gate/up plus GELU remains a material whole-path share, implement one fused
Q4_K/Q6_K gate/up kernel that reuses the activation tile and writes the gated
result directly.

## Constraints

- Keep production inference engine-owned, Metal-first, and model-resident.
- Keep native GGUF Q4_K/Q6_K storage; do not pre-dequantize, reorder, or retain
  a second weight copy.
- Preserve the shared projection selector and all M=1 behavior.
- Gate and up dtypes vary by tensor. Fusion must select each input by its actual
  dtype and support mixed Q4_K/Q6_K pairs or reject the candidate.
- Do not fuse Q/K/V, change attention/KV layout, alter the 16-token chunk,
  change sampling/tokenization, or add a tensor framework.
- Do not delete existing scratch buffers or elementwise kernels until a fused
  replacement passes parity and wins end to end.

## Verification

1. Attribute GPU duration on exact M=16 and remainder chunks before coding.
2. Add independent parity for Q4/Q4, Q4/Q6, Q6/Q4, and Q6/Q6 gate/up pairs at
   M=2, 7, 9, 15, and 16 and exact Gemma K/N dimensions.
3. Compare the final gated output, downstream down-projection output, logits,
   selected token, and all appended K/V rows against the unfused schedule.
4. Prove dispatch/barrier and resident scratch reductions without adding
   per-projection allocations or global dequantized weights.
5. Run `bash scripts/ci.sh` and the device-backed suite.
6. Run interleaved release real-model benchmarks with the same canonical
   command and report prompt tok/s, TTFT, mean, median, variance, decode tok/s,
   and deterministic token behavior.

## Stop Condition

Stop after one reversible gate/up plus GELU candidate either improves prompt
throughput and TTFT beyond run variance with parity intact, or is removed and
its measured rejection is recorded here. Do not proceed to Q/K/V fusion in the
same slice.
