# Gemma 2 Metal Pipeline Gaps Versus llama.cpp

## Scope

This is a source-level comparison of the current Gemma 2 9B Q4 production path
against `ggml-org/llama.cpp` commit
`b15ca938ad00aa6b3ee6c2edda7363fd02826b18` (2026-07-16).

It ranks structural differences that are likely to explain large inference
gaps. It is not a measured profile, so the order below is a recommendation for
what to measure first, not proof of exact milliseconds saved.

## Executive Summary

The main problem is not one slow kernel. The engine now has resident model
state, resident prefill scratch, one command buffer per 16-token prefill chunk,
and a one-command-buffer/one-compute-encoder decode path. The largest remaining
gaps are:

1. attention is three materialized F32 passes rather than fused attention;
2. RMSNorm and softmax reductions are serial within each row;
3. multi-token quantized projection is a scalar inner loop, while only
   single-token decode has a cooperative SIMD kernel;
4. KV is a full-context F32 allocation for every layer;
5. sampled generation copies and fully sorts all `256000` logits on the CPU.

`llama.cpp` instead treats inference as a reserved, reusable compute graph. Its
Metal backend keeps one compute encoder open across many graph nodes, inserts
memory barriers only for real dependencies, fuses supported node sequences,
uses fused attention, selects different projection kernels by batch shape, and
uses cooperative reductions.

The operation-at-a-time prefill gap and decode encoder fragmentation described
in the original analysis have been removed. The next structural target is the
largest measured fused kernel family, beginning with decode attention.

## Ranked Findings

### Completed: prefill used to synchronize hundreds of times

Prefill now executes through `run_gemma2_metal_resident_prefill` with plan-owned
scratch, a fixed 16-token chunk, one command buffer per chunk, no transient
activation-buffer allocation, queue-ordered chunks, and one final synchronous
wait. Greedy, sampled, chat append, context rebuild, and benchmark paths share
this executor. The staged all-intermediate prefill shell and its coupled tests
were deleted.

The remaining prefill gap is kernel quality: the resident executor still uses
separate encoders and scalar multi-row Q4_K/Q6_K projection kernels.

### 1. Completed: decode had one submission but extreme encoder fragmentation

`run_gemma2_metal_resident_decode_step` now keeps one concurrent compute encoder
open for the complete token, uses explicit fixed-schedule memory barriers, and
writes K/V directly into their cache rows. The 42-layer top-logit path performs
`719` compute dispatches through `1` compute encoder, `0` blit encoders, one
command buffer, one commit, and one wait.

`llama.cpp` creates one Metal encoder for a command-buffer graph slice, encodes
many nodes through it, and uses memory barriers plus tracked buffer ranges for
dependencies. It can also encode independent graph regions concurrently and
split a large graph across a small number of command buffers.

The staged decode executor, its activation hierarchy and tests, schedule-owned
resident buffer copies, and decode K/V scratch buffers were removed.

**Measured impact:** Gemma 2 9B Q4 greedy decode improved by `11.4%` on warm
first turns and `13.3%` on cached second turns in five-repetition release
benchmarks. Seeded sampled decode improved by `4.4%`; warm prefill was unchanged
within noise.

**Status:** complete and retained after independent nonzero CPU-reference tests,
all `315` repository tests, structural count checks, and real-path benchmarks.

### 2. Decode completed: fused online-softmax attention; prefill remains

Prefill still performs:

1. an F32 QK score kernel;
2. a separate F32 softmax kernel;
3. a separate F32 probability-times-V kernel.

Its scores and probabilities are written to full intermediate buffers. The value
kernel then rereads probabilities and walks the KV sequence again. At longer
contexts this creates substantial global-memory traffic.

`llama.cpp` enables flash attention automatically when supported. Its Metal
path has dedicated single-query and multi-query variants that tile Q/K/V in
threadgroup memory, compute softmax online, include logit softcapping, and emit
the attention result without materializing the full score and probability
matrices.

Single-token decode no longer uses that sequence. One paired-query Metal
workgroup owns each KV head and its two Gemma query heads, streams K/V in
32-token blocks, and maintains two stable online-softmax states without writing
scores or probabilities. Contexts through 256 tokens write the normalized
result directly. Longer contexts use 2-32 persistent partials and one
cooperative reduction dispatch. Sliding and global layers pass their existing
contiguous attention windows unchanged.

Decode scratch now owns 516 KiB of reusable partial state instead of 1 MiB of
score/probability buffers. The all-short 42-layer top-logit path is 719
dispatches and 550 barriers; a layer using partitioned attention adds one
dispatch and one barrier. It remains one compute encoder, one command buffer,
one commit, and one wait.

Exact 16-Q/8-KV/256-dimensional CPU parity covers token counts 1, 31, 32, 33,
256, 257, 4096, and 8192. The production decode integration matches independent
CPU logits, selected token, selected logit, and appended K/V state. All 316
repository tests pass.

**Measured impact:** the canonical short greedy benchmark is neutral within
noise: 4.94 versus 4.92 tok/s mean (+0.5%). That prompt never reaches the
partitioned path, so medium/long-context speedup still needs a thermally
controlled real-path measurement. Resident scratch fell by about 508 KiB.

**Status:** decode structural work is complete. Batched fused prefill attention
remains a separate task after competitive multi-token projection kernels; its
current three-pass operators remain live and tested.

**Remaining structural change:** add batched fused prefill attention after the
prefill projection strategy is fixed. Preserve alternating global/sliding
attention semantics.

### 3. Reduction kernels are serial and small ops are not fused

The local RMSNorm kernel assigns one GPU thread to a row. That thread loops over
the entire hidden width to compute the sum, then loops over it again to write
the result. Decode has only one row, so almost the entire dispatched threadgroup
is idle. The local softmax similarly assigns one thread per attention row and
serially makes three passes over the context.

`llama.cpp` uses all threads in a threadgroup for row reductions, combines
SIMD-group partial sums through threadgroup memory, vectorizes aligned data, and
can fuse RMSNorm with following multiply/add nodes. Its graph optimizer also
fuses supported elementwise chains.

This repo separately dispatches norm, residual add, GELU multiply, and final
softcap operations. Even when each kernel has little arithmetic, each adds an
encoder, launch, and full-buffer pass.

**Likely impact:** high in aggregate because these patterns repeat across all
42 layers.

**Required structural change:** first make RMSNorm and softmax cooperative;
then fuse measured chains such as post-projection norm plus residual, and
gate/up activation work. Judge fusion by aggregate token time, not one launch.

### 4. Quantized projection has no competitive multi-token execution strategy

The local Q4_K/Q6_K single-row decode path uses cooperative SIMD matvec kernels.
That is directionally correct. For more than one row, however, the fallback
assigns one thread to one output and serially loops over the entire input width,
repeatedly unpacking quantized blocks. It has no tiled activation reuse and does
not use SIMD-group matrix operations.

`llama.cpp` selects among:

- specialized matrix-vector kernels for one token;
- small-batch kernels for roughly 2-8 rows;
- SIMD-group matrix-matrix kernels for larger batches;
- format- and shape-specialized pipelines using function constants.

Thus the largest projection difference is currently prefill, not proof that the
decode matvec alone explains the decode gap.

**Likely impact:** very high for prompt processing; uncertain but important for
decode until aggregate Q4_K/Q6_K time is measured.

**Required structural change:** add shape-specific small-batch and matrix-matrix
Q4_K/Q6_K kernels after prefill becomes a single resident execution. Keep a
separate decode path, but select its next rewrite only from whole-token
projection totals.

### 5. KV layout doubles bandwidth and over-allocates sliding layers

The local plan allocates F32 K and V for all 8192 positions in all 42 layers:
about `5.64 GB`. Sliding-window layers still reserve the full context even
though only 4096 tokens are attended.

`llama.cpp` defaults K/V to F16 and gives Gemma 2 separate global and
sliding-window cache handling. Its sliding cache is sized near the window
rather than the full context. F16 also halves KV read bandwidth for attention.

This is a startup, memory-pressure, and long-context bandwidth problem. It may
not dominate very short decode, but it reduces the memory headroom available to
the rest of the pipeline.

**Likely impact:** medium at short context, high at long context and startup.

**Required structural change:** move to F16 KV with parity tests, then use a
ring-buffered 4096-token allocation for sliding layers while retaining full
history for global layers.

### 6. Sampled generation performs a full-vocabulary CPU round trip

For non-greedy chat, the engine copies the entire F32 logits buffer from Metal
to a new CPU tensor each token. `TokenSampler` builds 256000 candidates, applies
repetition checks with linear history searches, and fully sorts the vocabulary
before truncating to top-k.

`llama.cpp` has partial top-k selection, reusable sampler state, asynchronous
output extraction, and an experimental backend-sampling graph. Its greedy path
can execute argmax on the backend.

This does not explain a greedy Metal benchmark, but it can materially slow the
actual sampled chat product and hide gains made in decode.

**Likely impact:** high for sampled chat; negligible for the resident greedy
top-logit path.

**Required structural change:** at minimum, perform GPU top-k and read back only
the retained candidates, or use an in-place CPU partial selection without
copying/sorting the full vocabulary.

## Recommended Order

1. Consolidate decode encoders and add dependency barriers.
2. Implement fused single-query attention and cooperative RMSNorm/softmax.
3. Add small-batch and matrix-matrix Q4_K/Q6_K projection kernels.
4. Convert KV to F16 and compact sliding-window storage.
5. Remove the sampled full-logit CPU round trip.

The stop rule for every item should remain end-to-end: retain a change only if
the real `local-gguf-generate --bench` path improves beyond run-to-run noise.

## Primary Upstream References

- `src/models/gemma2.cpp`
- `src/llama-context.cpp`
- `src/llama-graph.cpp`
- `src/llama-kv-cache-iswa.cpp`
- `src/llama-sampler.cpp`
- `ggml/src/ggml-metal/ggml-metal-context.m`
- `ggml/src/ggml-metal/ggml-metal-ops.cpp`
- `ggml/src/ggml-metal/ggml-metal.metal`

Upstream repository: <https://github.com/ggml-org/llama.cpp/tree/b15ca938ad00aa6b3ee6c2edda7363fd02826b18>
