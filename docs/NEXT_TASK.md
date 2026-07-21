# Next Task: Replace the Decode Q4_K/Q6_K Inner Kernels

## Objective

Replace the one-row Metal Q4_K and Q6_K projection kernels with packed,
compile-time-specialized cooperative kernels for the exact Gemma 2 9B shapes.
Prove Q4_K and Q6_K independently, delete the superseded decode kernels and
their pipeline identities, and retain the work only if the real resident decode
path improves beyond noise without changing generated tokens.

This is Finding 1 from `docs/LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md`. It is a decode
kernel task, not a prefill, fusion, scheduling, repacking, or quantization-format
task.

## Why This Is the Active Bottleneck

The controlled local result is approximately 9.0 tokens/s for Gemma 2 9B Q4
decode; `llama.cpp` reaches approximately 14.2 tokens/s on the same base M4 and
model. The resident model payload gives a useful first-order effective bandwidth
comparison of approximately 61.5 GB/s locally versus 97.1 GB/s in `llama.cpp`.
The local runtime already has one compute encoder, one command buffer, one
commit, one wait, resident weights, an F16 ring KV cache, and compact token
selection. The tied F16 head rewrite improved decode by only about 3.2-3.6%.
The repeated quantized projections therefore have the strongest remaining
decode evidence.

The current source contains a misleading near-match. Both local decode kernels
dispatch 64 threads as two 32-lane SIMD groups, and each SIMD group produces two
adjacent output columns. Those are also the current `llama.cpp` outer constants.
The inner work is not equivalent:

- local Q4_K loops over eight quantization groups and then four scalar items,
  repeatedly calls scalar scale/min extraction, and indexes bytes separately;
- local Q6_K has the right lane/block partition in outline but still expresses
  the hot arithmetic as scalar loads, scalar unpacking, and scalar accumulation;
- the upstream kernels cache activation fragments in registers, use packed
  integer loads and masks, keep several independent `float4` accumulators, fully
  unroll the fixed block work, and advance precomputed row strides for the
  adjacent output row;
- those extra instructions occur in seven projections in every one of 42
  layers: five Q4_K projections and two Q6_K projections per decoded token.

The old `NEXT_TASK.md` conclusion that the decode kernels already embodied the
upstream winner was based on dispatch geometry and is superseded by the source-
level inner-loop comparison below.

## Exact Local Production Workload

The resident decode call graph is:

`run_gemma2_metal_resident_decode_step`
-> `MetalCommandContext::encode_matmul`
-> `select_quantized_matmul_dispatch`
-> the dtype-specific one-row pipeline.

For the target model, all production decode inputs have one F32 activation row,
all quantized inner dimensions are native 256-value block multiples, and all
output counts are multiples of four. The important projection families are:

| Projection | Dtype | K | N | Blocks per output |
| --- | --- | ---: | ---: | ---: |
| query | Q4_K | 3,584 | 4,096 | 14 |
| key | Q4_K | 3,584 | 2,048 | 14 |
| value | Q6_K | 3,584 | 2,048 | 14 |
| attention output | Q4_K | 4,096 | 3,584 | 16 |
| FFN gate/up | Q4_K | 3,584 | 14,336 | 14 |
| FFN down | Q6_K | 14,336 | 3,584 | 56 |

Q4_K is the first implementation and measurement target because it owns five
of the seven quantized projections per layer and most of the quantized weight
traffic. Q6_K follows only after the Q4_K result is understood.

## Upstream Research

Research was refreshed on 2026-07-21 against current source, not inferred from
thread counts or documentation summaries.

### `llama.cpp`: the primary same-format oracle

Current `llama.cpp` commit `846e991ec3c7ccec49112ff2c5b00b710e5f551d`
defines `N_R0_Q4_K = 2`, `N_SG_Q4_K = 2`, `N_R0_Q6_K = 2`, and
`N_SG_Q6_K = 2`. Its dispatcher compiles shape-specialized pipeline variants
with Metal function constants and records the output-row and SIMD-group tile in
the selected pipeline.

Its Q4_K decode kernel maps a lane as four 8-lane block workers. Each lane:

- caches 32 activation values covering four 64-value quantization regions;
- computes four activation sums once for the Q4_K minimum terms;
- reads packed quant values as `uint16_t`, not four independent bytes;
- reconstructs the eight 6-bit scale/min values with four packed masks;
- accumulates quant products in two `float4` values;
- reuses the activation cache for two adjacent output rows by advancing weight,
  scale, and half-scale pointers by a precomputed row stride; and
- performs one SIMD reduction per output row.

Its Q6_K kernel assigns lane pairs to two blocks, caches 16 activation values,
uses one `float4` for the four low/high-bitplane products, advances directly to
the second output row, and performs a SIMD reduction per row. Both kernels
operate directly on native GGUF blocks; neither materializes dequantized weights
nor requantizes the activation row.

Primary source:

- [Q4_K/Q6_K Metal kernels](https://github.com/ggml-org/llama.cpp/blob/846e991ec3c7ccec49112ff2c5b00b710e5f551d/ggml/src/ggml-metal/ggml-metal.metal#L8300-L8656)
- [decode tile constants](https://github.com/ggml-org/llama.cpp/blob/846e991ec3c7ccec49112ff2c5b00b710e5f551d/ggml/src/ggml-metal/ggml-metal-impl.h#L54-L61)
- [pipeline specialization and dispatch selection](https://github.com/ggml-org/llama.cpp/blob/846e991ec3c7ccec49112ff2c5b00b710e5f551d/ggml/src/ggml-metal/ggml-metal-device.cpp#L820-L950)

### MLX: independent Apple-GPU scheduling evidence

Current MLX commit `30a19f7234abfb4d136af58623eaac289e63f5d2` uses a
different affine/FP quantization representation, so its arithmetic cannot be
copied into a GGUF K-quant kernel. Its scheduling decisions are still useful:

- it selects distinct vector, small-batch-wide, matrix, and split-K families by
  M, N, K, quantization mode, and Apple GPU generation;
- its ordinary one-row quantized vector path also uses two SIMD groups;
- its small-batch path reuses weights across a bounded number of input vectors;
  it does not grow an unbounded row-accumulator array; and
- split-K is used when the orientation/shape lacks enough parallelism, while
  tiled quantized matrix kernels are used once M crosses a shape-dependent
  threshold.

This supports compile-time shape specialization and separate decode/prefill
families. It does not support adding split-K to the present Gemma decode: these
projections already expose thousands of output columns, and split-K would add
an intermediate buffer and reduction.

Primary source:

- [MLX quantized Metal dispatch and vector kernels](https://github.com/ml-explore/mlx/blob/30a19f7234abfb4d136af58623eaac289e63f5d2/mlx/backend/metal/quantized.cpp#L183-L552)
- [MLX shape-family selection](https://github.com/ml-explore/mlx/blob/30a19f7234abfb4d136af58623eaac289e63f5d2/mlx/backend/metal/quantized.cpp#L1463-L1552)

### Candle: a useful older GGUF comparison, not the target

Current Candle commit `31f35b147389700ed2a178ee66a91c3cc25cc80d`
contains GGUF Q4_K/Q6_K Metal kernels derived from an older ggml design. It
confirms the value of packed masks, vector accumulators, and direct GGUF-block
operation. Its host geometry differs: Q4_K uses four outputs in one SIMD group,
while Q6_K uses one output per SIMD group. This is a plausible tuning comparison
but not stronger evidence than the newer same-model `llama.cpp` result. Do not
make it the initial design or retain a second production implementation merely
for compatibility.

Primary source:

- [Candle GGUF Metal host dispatch](https://github.com/huggingface/candle/blob/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-metal-kernels/src/kernels/quantized.rs#L28-L160)
- [Candle packed Q4_K/Q6_K Metal kernels](https://github.com/huggingface/candle/blob/31f35b147389700ed2a178ee66a91c3cc25cc80d/candle-metal-kernels/src/metal_src/quantized.metal#L4887-L5293)

### Metal facilities

Use Metal function constants, or equivalently separately instantiated kernel
templates, so the compiler sees the production block count and tail contract.
Apple documents function constants as compile-time specialization inputs.
For temporary profiling, first check whether this M4 supports timestamp counter
sampling at dispatch boundaries; Apple documents counter sample buffers and
explicit compute-encoder sampling for this purpose.

- [Metal function constants](https://developer.apple.com/documentation/metal/mtlfunctionconstantvalues)
- [Metal GPU counters and counter sample buffers](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)
- [Sampling counters at command boundaries](https://developer.apple.com/documentation/metal/sampling-gpu-data-into-counter-sample-buffers)

## Selected Design

Implement a new packed one-row kernel family in
`src/tensor/backends/metal_backend.mm` with one shared ownership contract:

- 64 threads per threadgroup;
- two 32-lane SIMD groups;
- two adjacent output columns per SIMD group;
- four output columns per threadgroup;
- direct reads from native 144-byte Q4_K and 210-byte Q6_K blocks;
- one activation load schedule reused for both output columns;
- F32 accumulators and `simd_sum` reduction;
- no threadgroup-memory staging; and
- exact output-tail handling for the public backend contract.

Create specialized pipeline variants for the production block counts:

- Q4_K: 14 and 16 blocks per output column;
- Q6_K: 14 and 56 blocks per output column.

The compiler must see the block count as a constant so it can unroll the block
loop and fold address arithmetic. Keep one generic packed variant for valid
public/test shapes not in that set; it must use the same new inner algorithm,
not the legacy kernel. The production selector chooses only by dtype and shared
width. There is no runtime autotuner.

The Q4_K inner loop should follow the proven packed schedule, expressed in this
engine's row-major GGUF weight convention:

1. derive the two output-column base pointers once;
2. map lanes to `block_mod_4`, quantization half, and packed-pair index;
3. cache the fixed activation fragment in thread-local arrays/registers;
4. compute the four activation sums once;
5. load Q4 nibbles and scale/min metadata through aligned packed integer views;
6. accumulate four independent regions in `float4` values with explicitly
   unrolled fixed loops;
7. advance each weight pointer by one full column stride for the second output;
8. apply `d` and `dmin` after the packed dot products; and
9. reduce and write only valid columns.

The Q6_K loop should similarly preserve the lane-pair/two-block partition while
replacing scalar item arithmetic with a register-cached, unrolled `float4`
schedule and precomputed adjacent-column strides.

Do not blindly transliterate upstream tensor indexing or argument structures.
The local weight descriptor is logically `[K, N]` but stored as consecutive
native blocks per output column. Preserve that local invariant and use the
independent local CPU decoder as the oracle.

## Alternatives Rejected for This Baton

- **Activation Q8 requantization:** current Metal `llama.cpp` K-quant matvec
  consumes F32 activations directly. Adding a quantization pass changes the
  numerical contract and adds traffic before evidence says it is needed.
- **Weight repacking:** the same-format reference reaches 14.2 tokens/s directly
  from supported K-quant blocks. A second resident layout increases startup,
  memory, and ownership complexity and is not the first missing mechanism.
- **Split-K decode:** output parallelism is already abundant; partial outputs
  and a reduction add work. MLX's split-K decision applies to different tensor
  orientation, quantization, and shape conditions.
- **Four outputs per SIMD group as the initial kernel:** Candle shows this older
  alternative, but it increases simultaneous weight-row and accumulator state.
  Screen it only after the packed two-output kernel works, and only in temporary
  code, if GPU timing shows the retained kernel is occupancy-limited.
- **Prefill row kernels or tiled GEMM:** those are Finding 2 and remain untouched
  so decode attribution stays clean.
- **Barrier or fusion changes:** dispatch dependency work is Finding 3 and would
  confound the kernel A/B.

## Implementation Sequence

1. Preserve the current release binary as the end-to-end baseline. Record the
   current commit and dirty-tree state without altering unrelated changes.
2. Build a temporary, non-production timing probe around model-resident exact
   projection shapes. Prefer Metal timestamp counter samples when dispatch-
   boundary sampling is supported. Otherwise measure repeated format-specific
   command buffers with GPU start/end timestamps. Keep Q4_K and Q6_K totals
   separate, warm pipelines first, repeat enough times to report median and
   spread, and remove the probe before completion.
3. Implement and validate Q4_K specialized and generic packed kernels. Compare
   isolated Q4_K totals and the full resident decode before touching Q6_K.
4. If Q4_K improves the real path beyond noise, implement Q6_K with the same
   ownership but its own packed bitplane schedule. Measure it independently.
5. Only if the packed two-output kernel remains demonstrably occupancy-limited,
   temporarily screen the one-SIMD/four-output Q4_K geometry or one-output Q6_K
   geometry. Retain exactly one winner per dtype and delete every rejected
   variant.
6. Delete legacy production code and implementation-coupled tests as specified
   below, run the complete correctness suite, then run interleaved baseline and
   candidate real-model benchmarks.

## Legacy Deletion Boundary

After a replacement is retained, delete rather than alias:

- `matvec_f32_q4_k_simd_2x2_kernel` and
  `matvec_f32_q6_k_simd_2x2_kernel`;
- `matvec_q4_k_simd_2x2_pipeline` and
  `matvec_q6_k_simd_2x2_pipeline`;
- their old function lookup, pipeline construction, comments, and dispatch
  identities;
- any temporary tuning pipelines, counter instrumentation, diagnostic entry
  points, benchmark-only engine APIs, and generated benchmark output; and
- any test that exists only to assert an old kernel name, pipeline identity,
  thread geometry, implementation reuse, or compatibility alias.

Do not delete behavioral evidence. Keep and adapt the independent CPU parity
coverage, the combined packed-block/output-tail decode test, full resident
decode parity, unavailable-Metal behavior, and real generation checks. The
current multi-row Q4_K/Q6_K kernels and their tests are not legacy in this baton;
they are the live prefill path and must remain unchanged.

## Correctness Verification

Required deterministic coverage:

- Q4_K and Q6_K one-row parity against
  `reference_f32_quantized_gguf_weight_matmul`;
- block counts `1, 2, 14, 16, 56` and output tails `1, 2, 3, 4, 5, 7`, ensuring
  generic and every specialized block-count path execute;
- mixed-sign, nonuniform activations and nonzero scale/min metadata;
- exact production shapes for all six projection families in the table above,
  using compact deterministic fixtures where full real weights are unsuitable
  for a unit test;
- unchanged multi-row Q4_K/Q6_K parity, proving prefill dispatch did not move;
- resident Gemma decode parity against the independent CPU reference, including
  logits within the justified reduction-order tolerance, selected token, and KV
  state;
- unavailable-Metal error behavior; and
- exact greedy and seed-42 sampled token IDs against the preserved baseline.

Choose tolerances from F32 accumulation-order error. Do not loosen the existing
Q4_K `0.1` or Q6_K `0.25` absolute tolerance merely to accept a failure; first
compare error distributions and locate the arithmetic discrepancy.

Run `bash scripts/ci.sh` after implementation changes.

## Performance Verification

Use `models/gemma-2-9b-Q4.gguf`, release builds, prompt
`Explain why the sky is blue.`, one warmup, and at least five measured
repetitions. Preserve the established greedy and sampled configurations. Use a
32-token run to confirm decode rather than prompt setup owns the change.

Report:

- isolated aggregate Q4_K GPU time before/after and its spread;
- isolated aggregate Q6_K GPU time before/after and its spread;
- warm greedy generation tokens/s and decode milliseconds for 8 and 32 tokens;
- seeded sampled generation tokens/s and decode milliseconds;
- exact token IDs for every compared run;
- resident weight, KV, and scratch byte counts; and
- dispatch, barrier, encoder, command-buffer, commit, wait, allocation, and
  compact-readback counts, which must remain structurally unchanged.

Interleave baseline and candidate runs when possible to reduce thermal drift.
An isolated kernel gain is necessary but not sufficient. Retain a dtype rewrite
only when the full production decode improves beyond run-to-run noise and no
meaningful prefill regression appears.

## Constraints

- Keep inference engine-owned and offline; upstream code is a research oracle,
  not a runtime dependency.
- Preserve direct GGUF block operation and model residency.
- Preserve the current F16 output-head winner, F16 ring KV cache, compact token
  selection, one-submit resident decode, and zero full-vocabulary CPU readback.
- Preserve the separate multi-row prefill pipelines exactly.
- Add no runtime autotuning, benchmark switches, fallback to deleted kernels,
  compatibility aliases, or benchmark-only public APIs.
- Do not mix in prefill tiled GEMM, attention, fusion, barriers, sampling, or
  unrelated cleanup.
- Do not commit models, binaries, temporary probes, Metal captures, or benchmark
  result files.
- Preserve unrelated user changes in the current dirty worktree.

## Stop Condition

Stop after both dtype decisions are evidence-backed and the retained code has
exactly one generic/new packed implementation family plus the required exact-
shape pipeline variants per dtype. Update this document with measured
correctness and performance evidence, deleted legacy surfaces, changed files,
and any rejected candidate result.

If Q4_K does not improve whole-token decode beyond noise, delete the candidate
and stop before Q6_K. Record the negative result and select a new bottleneck;
do not keep speculative kernel complexity. If Q4_K wins but Q6_K does not,
retain Q4_K, restore no old aliases, leave the proven existing Q6_K kernel as
the only Q6_K implementation, and record that bounded result explicitly.
