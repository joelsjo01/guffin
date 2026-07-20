# Next Task: Competitive Multi-Token Q4_K/Q6_K Projection

## Verified Current State

The serial Metal RMSNorm gap is closed. Every RMSNorm row is now owned by one
256-thread workgroup. Threads read strided `float4` values when the hidden width
is divisible by four, accumulate squares in F32, reduce first with `simd_sum`
inside each SIMD group and then through eight F32 threadgroup partials, and
cooperatively write the normalized row. The scalar tail path remains for generic
tensor shapes. Input and output may alias because both threadgroup barriers
complete the reduction reads before any output write.

Gemma 2 decode also has a dedicated fused `RMSNorm * weight + residual` kernel.
It replaces the standalone post-attention norm/add and post-FFN norm/add pairs;
the superseded decode calls and their barriers are deleted rather than retained
as a fallback. The projection output remains the norm input, while the residual
is read directly and the final residual is written directly, so the fusion adds
no temporary and does not duplicate a full-row read.

The actual row-width analysis matters:

- attention produces 4,096 values before the output projection;
- the output projection produces the 3,584-value row that post-attention
  RMSNorm consumes;
- gate/up produce 14,336 values, but the down projection again produces the
  3,584-value row that post-FFN RMSNorm consumes;
- attention-input, FFN-input, post-projection, and final norms therefore all
  reduce exactly one 3,584-value F32 row during production decode.

The fixed 256-thread choice is an M4/Gemma specialization: 3,584 floats become
896 `float4`s, or three to four vector loads per lane. It avoids runtime pipeline
selection in the hot path while providing eight resident 32-lane SIMD groups.
The generic kernel still supports prefill and small/tail tensor tests; only the
decode-specific serial norm/add sequence was removed.

## Upstream and Hardware Findings

Pinned llama.cpp commit `b15ca938ad00aa6b3ee6c2edda7363fd02826b18`
uses the same essential hierarchy in
[`ggml-metal.metal`](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-metal/ggml-metal.metal#L3108-L3181):
vectorize by four when aligned, reduce lane partials with `simd_sum`, exchange
one value per SIMD group through threadgroup memory, then use a second
`simd_sum`. Its host encoder grows a power-of-two workgroup from 32 up to the
vectorized row width or pipeline maximum and can select fused RMSNorm/MUL/ADD
variants in
[`ggml-metal-ops.cpp`](https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/ggml/src/ggml-metal/ggml-metal-ops.cpp#L3405-L3548).

Apple documents 32-lane Apple-GPU SIMD groups and the two-level reduction
pattern in [Discover Metal enhancements for A14
Bionic](https://developer.apple.com/videos/play/tech-talks/10858/?time=747).
Apple also recommends threadgroup sizes that are multiples of a pipeline's
`threadExecutionWidth` in [Metal thread execution width
documentation](https://developer.apple.com/documentation/metal/mtlcomputepipelinestate/threadexecutionwidth),
and requires explicit synchronization even on Apple silicon in [Porting Metal
code to Apple silicon](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon).

The engine-specific optimization beyond copying the generic upstream choice is
to specialize the production fused kernel for the known one-row, 3,584-wide,
F32, four-aligned Gemma layout and to fuse the residual in the same write. A
generic graph optimizer would add complexity without improving this fixed
schedule.

## Correctness and Structural Evidence

- All 317 repository tests pass on Metal.
- Generic cooperative RMSNorm passes scalar-tail and batched CPU parity.
- A dedicated 3,584-wide test independently computes CPU RMSNorm plus add and
  checks the fused Metal result within `2e-5`, one dispatch, zero internal
  barriers, and zero transient allocations.
- The resident two-layer golden checks logits, selected token, appended K/V,
  one encoder, one commit, one wait, and no blit or transient allocation.
- Parallel F32 reduction changes association. The largest observed synthetic
  KV difference was 0.00318 at magnitude 1,554.3 (about 2 ppm); its explicit
  cache gate is 0.005. Logits retain the existing 0.003 gate.
- A real 42-layer selected-token decode now encodes 635 compute dispatches and
  466 explicit barriers, down from 719 and 550. The two fusions remove exactly
  84 dispatches and 84 barriers per token.

## Real-Path Performance Evidence

Canonical Gemma 2 9B Q4 release benchmark, prompt `Explain why the sky is
blue.`, eight generated tokens, greedy Metal, one warmup and five measured
repetitions:

- new warm decode: 8.16 tok/s mean, 8.05 median, 0.22 standard deviation;
- new cached decode: 8.65 tok/s mean, 8.88 median;
- preserved pre-change binary, run immediately afterward with the same command
  and one measured repetition: 5.21 tok/s warm and 5.19 tok/s cached;
- previously verified five-run baseline: 4.94 tok/s mean.

The same-session preserved-binary comparison is +56.6% for warm decode. Prefill
was 2.17-2.20 tok/s in both binaries, which supports attribution to the
decode-only norm/fusion change rather than a machine-wide speed shift. Benchmark
output remains in the terminal only and is not committed.

## Active Bottleneck

Prefill is now the clearest product-path gap. Q4_K and Q6_K projections with
multiple activation rows still assign one output element to one thread and
serially unpack the full inner dimension. They do not tile activations, reuse
dequantized weights across rows, or select a small-batch versus matrix strategy.
At roughly 2.2 prompt tok/s, this dominates time-to-first-token and makes long
context attention benchmarking unnecessarily expensive.

## Objective

Implement a competitive engine-owned multi-token Q4_K/Q6_K Metal projection
path for Gemma prefill. Choose the first slice from measured prompt batch sizes,
not from speculative generality, and preserve the existing cooperative
single-row matvec unchanged.

## Roadmap

1. Attribute prefill GPU duration by Q/K/V/O, gate/up/down, attention, and
   elementwise families for chunk sizes 1, 2, 4, 8, 16, 32, and 64. Record the
   row-count distribution used by the resident 64-token chunk schedule.
2. Inspect pinned llama.cpp Q4_K/Q6_K `mul_mm`, small-batch, and `mul_mv`
   dispatch thresholds and Metal layouts. Derive bytes read, dequantization
   work, activation reuse, occupancy, and threadgroup memory for Gemma's exact
   3,584, 4,096, and 14,336 dimensions.
3. Prototype only the highest-value shape family. Prefer a tiled kernel with a
   workgroup cooperatively loading an activation tile and reusing each decoded
   weight block across several rows. Keep K-quant decoding engine-owned and do
   not introduce MPS or another tensor runtime.
4. Compare at least two row-tile sizes and threadgroup shapes in a release
   microbenchmark, then retain only the winner. Handle output-column tails and
   Q4_K/Q6_K block boundaries explicitly.
5. Add independent CPU parity for both formats at row counts around the chosen
   dispatch threshold, inner/output tails, nonzero quant scales/minima, and the
   unavailable-Metal path. Add resident prefill parity and structural dispatch
   assertions.
6. Delete the superseded multi-row serial Metal branch and tests that assert its
   structure. Keep the single-row cooperative matvec and CPU reference paths.
7. Run `bash scripts/ci.sh`, then the canonical five-repetition real-model
   benchmark in one process. Accept only a prompt-throughput and TTFT improvement
   beyond variance with unchanged decode throughput and deterministic tokens.

## Constraints

- Do not change quantized weight layout, model residency, attention math, KV
  dtype/layout, tokenizer, sampling, or chat behavior.
- Do not regress the Q4_K/Q6_K one-row decode matvec.
- Do not add a generic graph optimizer or production inference dependency.
- Do not retain rejected kernels, dispatch branches, diagnostic allocations, or
  benchmark artifacts.
- Never overlap real-model benchmark processes; compare release builds under a
  controlled thermal state.

## Stop Condition

Stop at the first end-to-end multi-row projection slice that passes independent
Q4_K/Q6_K parity and repository CI, deletes the superseded serial multi-row
path, and produces a repeatable real-model prefill/TTFT gain without decode
regression. If no candidate clears that gate, remove it and leave the measured
rejection evidence here.
