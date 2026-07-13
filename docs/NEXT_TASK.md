# Next Task: Measure The Remaining Decode Matvec Split

## Verified Starting Point

The one-row Q4_K decode kernel is complete.

- One 32-lane SIMD group produces two output columns.
- Two SIMD groups form a 64-thread threadgroup and produce four columns.
- Four 8-lane subgroups divide packed blocks, reuse activations for both
  columns, accumulate in F32, and finish with `simd_sum`.
- The specialized path is selected only for one-row Q4_K inputs with a
  256-aligned inner width on a pipeline whose execution width is 32.
- Multi-row Q4_K, Q6_K, and F16 paths retain their previous kernels.
- Production decode remains one command buffer, one commit, one wait, zero
  transient Metal buffers, and no intermediate CPU readback.

Deterministic Metal parity covers packed-block counts `1`, `4`, `14`, `16`,
and `56`, an odd five-column output tail, nonzero minimum terms, all scale/min
packing branches, both nibbles, and mixed-sign activations against the CPU
dequantized F32 oracle.

## Measured Result

Allocation-free resident measurements used five encoded matvecs per command
buffer and divided the command-buffer GPU duration by five:

| Q4_K shape `[K, N]` | Scalar | SIMD | Speedup |
| --- | ---: | ---: | ---: |
| `3584 x 2048` | `3.765 ms` | `0.435 ms` | `8.65x` |
| `4096 x 3584` | `4.348 ms` | `0.711 ms` | `6.11x` |
| `14336 x 3584` | `10.395 ms` | `2.701 ms` | `3.85x` |

The exact release benchmark used eight generated tokens, one warmup, and five
measured repetitions. Warm-first-turn generation results were:

| Repetition | Generation tokens/s |
| ---: | ---: |
| `0` | `2.686` |
| `1` | `2.366` |
| `2` | `2.180` |
| `3` | `2.016` |
| `4` | `2.770` |
| Mean | `2.404` |
| Median | `2.366` |

The recorded scalar baseline was `0.930 t/s` mean, so the retained Q4_K
kernel improved real warm greedy throughput by about `2.58x`. Host Metal CI
passes with `312` tests.

## Why Measurement Comes Next

One generated token still encodes:

- `42` scalar Q6_K matvecs streaming about `1.01 GB` of packed weights;
- one scalar tied F16 vocabulary projection streaming about `1.84 GB`.

Weight bytes alone no longer identify the dominant kernel. The Q6_K calls
have dispatch and repeated-input costs, while the single vocabulary projection
has a very large output width. Measure both on the resident production layout
before choosing the next optimization.

## Implementation Gates

### Gate 1: Enumerate The Real Workload

- Enumerate the actual target model's Q6_K decode shapes and call counts from
  the resident plan.
- Confirm the tied F16 output projection shape and call count.
- Keep this model-specific and read-only; do not introduce a generic graph or
  profiling framework.

### Gate 2: Measure Resident Kernel Cost

- Use temporary internal instrumentation or a temporary allocation-free probe,
  not permanent CLI timing output.
- Reuse already uploaded buffers and caller-owned outputs.
- Exclude model load, plan creation, buffer allocation, terminal output, and
  CPU token selection.
- Measure representative individual Q6_K shapes and the tied F16 projection
  with enough repetitions to reduce timer noise.
- Record pipeline execution width, maximum threads per threadgroup, and static
  threadgroup memory for both pipelines.
- Estimate per-token aggregate Q6_K and F16 projection time using the real call
  counts, then validate the ranking on a complete resident decode token where
  possible.

### Gate 3: Choose One Next Kernel

- Select Q6_K only if its measured aggregate cost is larger.
- Select the tied F16 projection only if its measured cost is larger.
- If results overlap materially, repeat under the same thermal conditions and
  use the complete-token result as the tiebreaker.
- Put the chosen kernel, exact shapes, baseline timings, parity requirements,
  candidate mapping, integration boundary, and rejection criteria into this
  file before implementation.

## Do Not

- Do not optimize Q6_K or F16 in this measurement slice.
- Do not alter the completed Q4_K kernel.
- Do not change prefill, attention, KV, sampling, context handling, model
  loading, or CLI behavior.
- Do not add permanent kernel-level CLI output, a benchmark executable,
  dashboards, databases, or runtime autotuning.
- Do not begin the llama.cpp throughput comparison.

## Stop Condition

Stop when the real resident Q6_K and tied-F16 costs have been measured with a
repeatable method, one is identified as the next bottleneck beyond noise, and
this baton contains a narrow implementation plan for that one kernel. Wait for
approval before implementing it.
