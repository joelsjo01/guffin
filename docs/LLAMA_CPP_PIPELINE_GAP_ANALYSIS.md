# Gemma 2 Metal Performance Gap Versus llama.cpp

## Purpose

This document explains the current performance difference between this engine
and `llama.cpp` for the same Gemma 2 9B Q4 model on the base Apple M4. It
replaces the earlier milestone-by-milestone pipeline analysis. Completed work
and old implementation history do not belong here.

The conclusion is based on controlled real-path measurements, inspection of
the current engine, and source-level comparison with the Metal backend in
`llama.cpp`. It is a diagnosis and optimization direction, not an active
implementation baton. The active task remains owned by `docs/NEXT_TASK.md`.

## Executive Conclusion

The performance gap is real. It is not principally caused by chat formatting,
sampling, Flash Attention, KV placement, or CPU work. The dominant remaining
problem is the quality of the Q4_K and Q6_K Metal projection kernels.

The engine's decode kernels now resemble `llama.cpp` at the outer dispatch
level, but they do not yet resemble it internally. `llama.cpp` does much more
packed, vectorized, compile-time-specialized work per activation load and per
quantized block. The local kernels perform more scalar unpacking, indexing,
scale/min extraction, and loop control. Consequently, the local engine obtains
far less useful work from the M4's available memory bandwidth.

Prompt processing has an additional, larger problem: the local 16-row kernels
reuse weights across rows but process the rows serially inside each lane. They
are not true tiled matrix-matrix kernels. `llama.cpp` switches to direct-
dequantizing SIMD-group matrix multiplication for this shape. That algorithmic
difference explains why the prompt gap is much larger than the decode gap.

## Controlled Measurements

### Test environment

- Hardware: base Apple M4 with Apple-family SIMD-group reduction and matrix
  multiplication support.
- Model: `models/gemma-2-9b-Q4.gguf`.
- Reference executable: Homebrew `llama.cpp` build 9430, commit `d48a56eff`.
- Local path: release `local-gguf-generate` resident Metal path.
- Reference command family: `llama-bench`, five repetitions, full Metal
  offload, with Flash Attention and microbatch size varied independently.

### Decode

| Engine | Controlled generation rate |
|---|---:|
| This engine | approximately 9.0 tokens/s |
| `llama.cpp` | approximately 14.2 tokens/s |

`llama.cpp` is approximately 58% faster. The local result is stable across
greedy, sampled, and longer 32-token generation runs, so it is not a short-run
sampling artifact.

### Prefill

| Prompt tokens | `llama.cpp`, microbatch 16 | `llama.cpp`, microbatch 512 |
|---:|---:|---:|
| 16 | 72.4 tokens/s | 70.1 tokens/s |
| 32 | 74.9 tokens/s | 136.9 tokens/s |
| 128 | 77.5 tokens/s | 153.3 tokens/s |

The local canonical 16-token prompt runs at approximately 8.8 tokens/s. At the
same 16-row scale, `llama.cpp` is therefore about 8.3 times faster. Prompt
content is not byte-for-byte identical between the two benchmark drivers, but
the projection shapes and amount of model work are comparable and the gap is
far beyond measurement noise.

Large microbatches make `llama.cpp` substantially faster at 32 and 128 tokens,
but do not explain its 16-token advantage. Merely increasing the local chunk
size cannot close the initial gap; the projection algorithm must change first.

### Flash Attention control

With a 512-token microbatch, `llama.cpp` measured approximately:

| Workload | Flash Attention off | Flash Attention on |
|---|---:|---:|
| 16-token prefill | 70.1 tokens/s | 68.2 tokens/s |
| 128-token prefill | 153.3 tokens/s | 151.3 tokens/s |
| 32-token generation | 14.0 tokens/s | 14.2 tokens/s |

Flash Attention improved short decode by only about 1.5% and slightly reduced
short-prompt throughput. It is not the cause of the observed gap.

## Bandwidth Diagnosis

The resident model payload is approximately 6.837 GB. A useful first-order
decode estimate is therefore:

- local engine: `6.837 GB * 9.0 tokens/s = 61.5 GB/s`;
- `llama.cpp`: `6.837 GB * 14.2 tokens/s = 97.1 GB/s`.

Apple specifies 120 GB/s unified-memory bandwidth for the base M4. Under this
simplified weight-streaming measure, `llama.cpp` extracts about 81% of that
limit while the local engine extracts about 51%.

These numbers are effective estimates, not literal DRAM counters. Cache hits,
metadata, activation traffic, and the 1.835-GB tied F16 output matrix make the
physical traffic more complicated. The comparison is nevertheless strong
evidence: both engines execute the same model on the same memory system, while
`llama.cpp` completes substantially more weight traversal per second. The
remaining loss is primarily kernel instruction efficiency, occupancy, and
scheduling around memory traffic.

## Finding 1: Decode Projection Geometry Is Not Kernel Equivalence

The local Q4_K and Q6_K decode paths use two SIMD groups and compute two output
rows cooperatively. This matches important outer parameters used by
`llama.cpp`, but matching those numbers does not make the kernels equivalent.

The `llama.cpp` Q4_K and Q6_K kernels:

- partition packed block decoding carefully across SIMD lanes;
- load activation values into registers and reuse them for adjacent output
  rows;
- use packed integer loads and masks instead of repeated scalar extraction;
- accumulate several independent products in vector registers;
- use compile-time-unrolled block loops;
- walk adjacent weight rows with precomputed pointer strides;
- extract packed quantization metadata with a lane-aware schedule;
- specialize pipeline behavior through Metal function constants; and
- reduce results cooperatively across the SIMD group.

The local kernels still perform relatively scalar byte handling. Q4_K scale
and minimum extraction is repeated inside the hot path, and each lane performs
small scalar accumulation loops for both output columns. Q6_K has the same
general issue around its packed bitplanes and scalar arithmetic schedule.

This difference repeats across the model's quantized projections in all 42
layers. Even modest wasted work inside one block becomes a large whole-token
cost.

### Required direction

Replace the current decode Q4_K and Q6_K inner loops with shape-specialized,
packed cooperative kernels. The objective is not to copy a thread count. It is
to minimize instructions, metadata decoding, and activation loads per useful
multiply-accumulate while preserving direct operation on GGUF blocks.

The first isolated benchmarks must cover the exact Gemma projection shapes and
separate Q4_K from Q6_K. Whole-token retention remains mandatory because an
isolated kernel gain can be lost through occupancy or scheduling interactions.

## Finding 2: Prefill Uses Serial Row Reuse, Not Tiled GEMM

The production prefill chunk size is 16. The current multi-row Q4_K and Q6_K
kernels assign a SIMD group to an output column, decode one weight per lane,
and then loop over every activation row while retaining one scalar accumulator
per row. For a 16-row tile, each lane owns an array of 16 sums and updates those
sums serially for every decoded weight.

This avoids decoding the same weight 16 separate times, but it has three major
costs:

1. the token dimension is serialized inside each lane;
2. the accumulator array creates substantial register pressure; and
3. it does not use the M4's SIMD-group matrix-multiply instructions.

`llama.cpp` selects algorithms by row count. Its current Metal dispatcher uses
specialized matrix-vector variants for small row counts and switches larger
inputs to SIMD-group matrix-matrix kernels when supported. The Q4_K and Q6_K
matrix-matrix paths dequantize blocks directly into tiled computation and use
`simdgroup_half8x8` operations rather than materializing a complete temporary
F16/F32 weight matrix.

At 16 prompt tokens, `llama.cpp` is already on a genuine matrix-matrix path,
while the local engine is executing serial row loops inside a matvec-derived
kernel. This is the best-supported explanation for the approximately 8.3-times
prefill difference.

### Required direction

Use a shape-dispatched projection family:

- one row: packed cooperative Q4_K/Q6_K matrix-vector kernel;
- approximately two to eight rows: a small-batch cooperative kernel that
  amortizes block decoding without excessive row accumulators;
- larger batches: direct-dequantizing tiled SIMD-group matrix multiplication;
- exact Gemma shapes: function-constant or separately compiled specialization
  for widths, strides, row tiles, and quantization format.

After the tiled path is competitive, benchmark larger resident prefill chunks.
A larger chunk should be retained only when its scratch requirements and real
prompt latency are favorable.

## Finding 3: Fixed Dispatch and Barrier Cost Is Material but Secondary

The current 42-layer decode top-logit path performs 635 compute dispatches and
466 explicit buffer barriers per token in one compute encoder and one command
buffer. Keeping one encoder and command buffer removed a major earlier cost,
but the remaining schedule is still conservative.

`llama.cpp` tracks buffer ranges and inserts a barrier when a later operation
actually conflicts with preceding writes. This permits independent projections
such as Q/K/V and gate/up to be encoded without unnecessary synchronization.

Hundreds of dispatches and barriers are material at a 70-111 millisecond token
time. They are unlikely to explain the full gap while projection kernels stream
several gigabytes per token, but their relative importance will grow after the
quantized kernels improve.

### Required direction

First obtain per-operation GPU timings or exact-shape isolated projection
timings. After projection optimization, replace fixed barriers with verified
resource dependency tracking and fuse cheap neighboring operations only where
whole-token measurements show a benefit.

## Finding 4: The F16 Output Head Is No Longer the Main Explanation

The tied F16 output matrix is approximately 1.835 GB and is read once per
generated token. Its old kernel severely underused the dispatched threads. The
new cooperative implementation uses multiple SIMD groups, multiple output rows
per group, vector loads, and SIMD reductions.

That rewrite improved real decode by approximately 3.2-3.6%. The result proves
that the head mattered, but also proves that it was not the majority of the
remaining 58% gap. The quantized layer projections now have the stronger
evidence and higher priority.

## Finding 5: Factors Ruled Out as Primary Causes

### Flash Attention

Controlled on/off measurements change short decode by approximately 1.5% and
do not improve the tested short prefill shapes. It remains valuable at longer
contexts but cannot explain this gap.

### Sampling and CPU logits work

The local greedy, sampled, and 32-token decode measurements are all near 9.0
tokens/s. Sampling work is therefore not responsible for the controlled decode
difference. Chat sampling improvements can still matter independently.

### KV cache location and format

The production path keeps its model and F16 KV cache resident in Metal-visible
unified memory. At short context, KV traffic is small compared with streaming
the projection weights. KV optimization matters increasingly at long context,
but is not the present short-decode cause.

### More threads by themselves

Using every possible thread does not guarantee more throughput. Extra threads
can duplicate weight traffic, increase reductions, consume more registers, and
reduce occupancy. The correct target is maximum useful SIMD work per loaded
block, not the largest legal threadgroup.

### Metal repacking

The `llama.cpp` command-line repack option is enabled by default, but its Metal
backend does not appear to use the CPU-style repacking path as the source of
this advantage. The relevant Metal kernels operate directly on supported
quantized block layouts.

### Chat token-rate presentation

Rates printed for tiny interactive prompts are noisy. Tokenization, chat
templates, cached-prefix length, synchronization boundaries, and fixed timing
overhead can make later prompt rates appear dramatically higher. These factors
explain why the displayed 20-90 tokens/s figures vary, but they do not invalidate
the controlled benchmark gap.

## Recommended Investigation and Implementation Order

1. Add trustworthy GPU timing around the exact Q4_K and Q6_K projection shapes,
   or build isolated commands that reproduce the production layouts and
   dispatches exactly.
2. Replace the Q4_K decode inner kernel with packed, vectorized, activation-
   reusing cooperative SIMD work.
3. Apply the proven structure to Q6_K, measuring it independently because its
   bitplane unpacking and arithmetic balance differ.
4. Implement small-batch kernels for roughly two to eight rows.
5. Implement direct-dequantizing SIMD-group tiled matrix multiplication for
   16-row and larger prefill shapes.
6. Tune the resident prefill chunk size only after the tiled kernel exists.
7. Re-measure whole-token dispatch and barrier cost, then introduce dependency-
   driven barriers and targeted fusion.

## Acceptance Criteria

Every replacement must satisfy all of the following:

- independent CPU or golden parity for exact Q4_K and Q6_K block decoding;
- deterministic correctness at the exact Gemma projection shapes;
- correct unavailable-Metal behavior;
- real greedy and seeded sampled generation parity;
- no fallback to a legacy production kernel for the target Gemma path;
- an isolated improvement outside measurement noise;
- a comparable end-to-end `local-gguf-generate --bench` improvement; and
- no meaningful prefill regression while improving decode, or vice versa.

Stop and reject a proposed kernel when it improves an artificial microbenchmark
but does not improve the model's real production path.

## Primary References

- Apple M4 memory bandwidth:
  <https://www.apple.com/newsroom/2024/10/apple-introduces-m4-pro-and-m4-max/>
- `llama.cpp` Metal dispatcher and dependency handling:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp>
- `llama.cpp` Metal quantized matrix-vector and matrix-matrix kernels:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal.metal>
- `llama.cpp` Metal quantization/kernel parameters:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-impl.h>
- GGUF quantized block layouts:
  <https://github.com/ggml-org/llama.cpp/wiki/Tensor-Encoding-Schemes>

Local implementation anchors:

- `include/local_gguf_engine/engine/gemma2_metal_plan.hpp`
- `src/engine/gemma2_runtime.cpp`
- `src/tensor/backends/metal_backend.mm`
- `src/tensor/backends/metal/metal_command_context.mm`
- `src/tensor/backends/metal/metal_context.mm`
- `tests/engine/gemma2_metal_plan_tests.cpp`
