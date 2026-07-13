# Next Task: M4 Tied-F16 Decode Projection Bake-Off

## Verified Starting Point

One-row Q4_K and Q6_K decode projections now use format-specific
SIMD-cooperative Metal kernels.

- Both produce two output columns per 32-lane SIMD group and use two SIMD
  groups per 64-thread threadgroup.
- Q4_K divides four packed blocks across four 8-lane subgroups.
- Q6_K pairs lanes so sixteen lanes cover each of two interleaved packed
  blocks, matching its separate low/high bit planes and signed scales.
- Both reuse activations across two output columns, accumulate in F32, finish
  with `simd_sum`, and use no threadgroup memory.
- Specialized selection requires one row, a 256-aligned inner width, a
  32-thread execution width, and capacity for a 64-thread threadgroup.
- Multi-row kernels remain scalar fallbacks.
- Production decode remains one command buffer, one commit, one wait, zero
  transient buffers, and no intermediate CPU readback.

Host Metal CI passes with `313` tests. Q6 parity covers packed-block counts
`1`, `2`, `14`, `16`, and `56`, an odd output tail, both low nibbles, all four
high-bit planes, signed scales, full six-bit values, and mixed-sign activations
against the CPU dequantized F32 oracle.

## Q6_K Result

All `42` target-model Q6_K tensors are feed-forward down projections with
shape `[14336, 3584]`.

| Resident Q6_K kernel | GPU duration | Relative |
| --- | ---: | ---: |
| Scalar | `9.006 ms` | `1.00x` |
| SIMD | `2.166 ms` | `4.16x` |

The exact eight-token release benchmark used one warmup and five measured
repetitions. Warm-first-turn results were:

| Repetition | Generation tokens/s |
| ---: | ---: |
| `0` | `2.996` |
| `1` | `3.000` |
| `2` | `2.765` |
| `3` | `3.054` |
| `4` | `3.835` |
| Mean | `3.130` |
| Median | `3.000` |

This is a further `30.2%` mean improvement over the Q4-only `2.404 t/s`
result and about `3.37x` the original `0.930 t/s` scalar baseline.

## Remaining Major Matvec

The tied token-embedding/output tensor is F16 with shape `[3584, 256000]` and
streams about `1.84 GB` once per generated token. Its current decode kernel
assigns one output column to one thread and serially walks all `3584` input
values.

This is now the obvious next matvec to measure and optimize. Do not assume the
Q4/Q6 two-output mapping wins unchanged: the unquantized F16 layout has
different arithmetic intensity, a very large output width, and no unpacking
cost.

## Implementation Gates

### Gate 1: Lock The F16 Baseline

- Add deterministic one-row F16 GGUF-weight matvec parity fixtures with odd
  output tails and mixed-sign values.
- Preserve existing multi-row F16 behavior.
- Measure the allocation-free resident `[3584, 256000]` projection over enough
  repetitions to reduce timer noise.
- Record pipeline execution width, maximum threads per threadgroup, and static
  threadgroup memory.

### Gate 2: Compile-Time Decode GEMV Bake-Off

Start with a one-dimensional 32-lane SIMD group cooperatively reducing the
`3584`-value dot product. Compare small compile-time layouts:

1. one output column per SIMD group;
2. two output columns per SIMD group with activation reuse;
3. four output columns only if pipeline capacity and measured performance
   justify the additional registers.

Start with two SIMD groups per threadgroup. Test other group counts only if the
isolated duration remains meaningfully different. Use `simd_sum`, F32
accumulation, no threadgroup staging, and no intermediate weight conversion.
Remove losing candidates; do not add runtime autotuning.

### Gate 3: Integrate Narrowly

- Select the winner only for one-row F16 GGUF-weight matvec with a 32-thread
  execution width and sufficient threadgroup capacity.
- Encode through `MetalCommandContext` into the existing resident logits
  buffer.
- Keep token embedding and multi-row F16 paths unchanged.
- Preserve the one-submit/no-transient-buffer production invariants.

### Gate 4: Prove End-To-End Value

- Run CPU-oracle parity, resident diagnostic parity, full host Metal CI, and
  `git diff --check`.
- Repeat the exact eight-token, one-warmup, five-repetition release benchmark.
- Compare against `3.130 t/s` mean and `3.000 t/s` median.
- Reject a candidate whose improvement is within noise or whose numerical
  tolerance, residency, or submission behavior regresses.

## Do Not

- Do not alter the completed Q4_K or Q6_K kernels.
- Do not change prefill, token embedding, attention, KV, sampling, context
  handling, model loading, or CLI behavior.
- Do not add permanent kernel-level CLI output, a benchmark executable,
  dashboards, databases, or runtime autotuning.
- Do not begin the llama.cpp throughput comparison.

## Stop Condition

Stop when one measured one-row F16 vocabulary-projection kernel has replaced
the scalar kernel only on the resident decode path, all correctness and
structural tests pass, and the exact five-repetition real-generation benchmark
shows an improvement beyond noise.
