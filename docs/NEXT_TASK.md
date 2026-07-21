# Next Task: Finding 5 — Native F16 KV Cache and Exact Sliding Rings

## Implementation Result (2026-07-21)

Finding 5 is implemented and accepted on the Gemma 2 9B Q4 Metal product
path. There is one production cache representation: F16 K/V, with 8192 rows
for global layers and an exact 4096-row power-of-two ring for sliding layers.
The resident readers convert half vectors to float and retain F32 dot products,
softmax state, and outputs.

The implementation also landed the intended producer and scheduling changes:

- decode projects V into an 8 KiB F32 `raw_value` scratch and uses one fused
  K-RoPE/F16-K/F16-V store dispatch;
- prefill uses the same fused store, has no separate K scratch or KV blit, and
  splits chunks at a sliding-ring boundary;
- decode and prefill cache readers map logical positions with the layer mask;
- the old full-context F32 allocation/offset path, KV blit encoder, public
  test-only F32 cache APIs, their helpers, and their representation-only tests
  are deleted rather than retained as fallbacks.

Device-backed verification passes 312 tests. It covers exact F16 V conversion,
K RoPE within the host/device transcendental error bound, physical rows 7 and
0 across a wrap, F16-cache decode attention at 1/31/32/33/256/257/4096/8192
tokens, sliding/global resident decode, boundary-split prefill, one final wait,
zero transient allocations, zero KV blits, incremental parity, deterministic
selection, generic F32 attention isolation, and unavailable-Metal behavior.

The preserved pre-change binary and the accepted release candidate used the
same local model, prompt, generation settings, one warmup, and five measured
repetitions. The generated token IDs were unchanged in every corresponding
case. Key results:

| Metric | Full F32 cache | F16/exact rings | Change |
|---|---:|---:|---:|
| Resident KV bytes | 5,637,144,576 | 2,113,929,216 | -62.5% |
| Plan creation | 14.185 s | 11.968 s | -15.6% |
| Warm prompt throughput | 2.147 tok/s | 8.327 tok/s | 3.88x |
| Warm greedy decode throughput | 4.804 tok/s | 8.352 tok/s | 1.74x |
| Warm total generation time | 9.119 s | 2.880 s | -68.4% |

These short-context gains exceed the cache-read-only ceiling because the old
5.25 GiB allocation caused severe unified-memory pressure for the resident
6.37 GiB model. The result therefore accepts the F16 path without a fallback.

The remaining measurement gap is real end-to-end profiling immediately around
4096 and near 8192 tokens with per-layer GPU counters. Exact-shape 4096/8192
correctness is covered, but producing and interleaving those long resident
sessions would take tens of minutes with the rejected baseline. Do that as a
measurement-only follow-up; it is not a reason to retain old code or reopen the
accepted representation.

## Scope Correction

Finding 5 in `docs/LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md` is the KV-cache
layout problem, not gate/up fusion. The active baton is therefore to replace
the full-context F32 KV cache with the fastest low-risk layout for the Gemma 2
9B Metal product path. Gate/up fusion is not part of this slice.

This is a replacement, not a second cache mode. Once parity and performance
gates pass, no production F32/full-context KV implementation or tests that
only preserve that implementation should remain.

## Verified Current State

The resident plan allocates separate K and V buffers for every layer with
shape `[context_length, 8, 256]` and dtype F32, including the 21 sliding layers.
For the target's 42 layers and 8192-token context this is:

`42 * 8192 * 8 * 256 * 2 * 4 = 5,637,144,576 bytes = 5.25 GiB`.

The current execution paths are coupled to that representation:

- decode projects V directly into an absolute F32 cache byte offset and RoPE
  writes K directly into a second absolute F32 cache byte offset;
- prefill produces F32 K and V scratch, then uses a blit encoder to copy both
  into full-context cache rows;
- decode fused attention accepts only rank-3 F32 K/V caches and indexes an
  unwrapped absolute `[first_position, cache_position]` range;
- prefill score and value kernels also read absolute F32 cache rows;
- plan accounting, exact-shape attention tests, resident decode/reference
  tests, and prefill byte-equality tests encode the F32/full-context contract;
- the public `write_*kv_cache_f32*` and
  `causal_grouped_attention_*_f32_with_kv_cache*` backend-buffer APIs are not
  used by the resident production executor. Their only repository callers are
  tests, so they are legacy surface rather than a required fallback.

The schedule already computes the correct logical attention length:
sliding layers use `min(total_tokens, sliding_window)` and global layers use
all populated tokens. What is missing is a physical-row mapping independent of
the logical token position.

## Research Verdict

Implement F16 K and V caches, keep all dot products, online-softmax state, and
attention outputs in F32, and allocate each sliding layer as an exact
power-of-two ring. For the target:

- 21 global layers: `[8192, 8, 256]` F16 K plus V;
- 21 sliding layers: `[4096, 8, 256]` F16 K plus V;
- physical sliding row: `absolute_position & (4096 - 1)`;
- physical global row: the absolute position (or the same masked mapping with
  an 8191 mask while the context invariant is enforced).

The resulting target allocation is:

`(21 * 8192 + 21 * 4096) * 8 * 256 * 2 * 2`
`= 2,113,929,216 bytes = 1.96875 GiB`.

This removes `3,523,215,360` bytes (`3.28125 GiB`, `62.5%`) from the current
cache. F16 alone halves every attention K/V read. Compact sliding storage does
not reduce bytes read by a correctly windowed attention kernel, but it removes
allocation and memory pressure and prevents cold, unreachable sliding rows
from consuming unified memory.

This is the best first production format for the M4 because Apple explicitly
states that 16-bit types reduce register and memory footprint and that
half/float conversions are free on the relevant GPU family. Native half loads
therefore cut cache traffic without adding scale metadata, integer unpacking,
or a lossy calibration contract. Accumulation must remain F32.

## What llama.cpp Actually Establishes

The pinned comparison commit
`b15ca938ad00aa6b3ee6c2edda7363fd02826b18` establishes four useful facts:

1. The context defaults K and V cache types to F16.
2. Gemma 2 is declared as standard sliding attention, defaults to a 4096-token
   window, and alternates sliding/global layers. Google's reference 9B config
   independently specifies 42 layers, 16 Q heads, 8 KV heads, 256 head width,
   and `[LOCAL_SLIDING, GLOBAL] * 21`.
3. `llama_kv_cache_iswa` splits global and sliding layers into separate caches.
   In compact mode its sliding capacity is
   `pad256(min(context, window * sequences + microbatch))`, not simply the
   window. The extra microbatch cells prevent a whole prefill batch from
   overwriting rows still needed by early queries in that batch.
4. llama.cpp's library default sets `swa_full=true`, while its common CLI
   default is false and explicitly overrides that library default. The pipeline
   document's statement that llama.cpp compacts sliding storage describes the
   normal CLI configuration, not every API initialization.

Do not copy llama.cpp's general cell allocator. This engine owns one sequence,
one fixed target model, and a fixed maximum prefill chunk. It can use a simpler
and smaller exact ring while preserving the same overwrite invariant.

## Engine-Specific Fast Path

### 1. Split prefill at a ring boundary

An exact 4096-row ring is unsafe if a multi-token chunk writes across the wrap
before its early queries attend: those writes can overwrite still-live old
rows. Prevent that in the chunk scheduler rather than over-allocating every
sliding layer.

Choose each resident prefill chunk as:

`min(16, remaining_tokens, 4096 - (absolute_position & 4095))`.

This preserves the current 16-token tier everywhere except the single chunk
that reaches each 4096-token boundary. It guarantees that all K/V rows needed
by every query in a submitted chunk survive until that chunk's attention has
consumed them. It also keeps writes contiguous within the physical ring.

The target context and window are powers of two, so attention can map a logical
row with one integer AND instead of division/modulo. Validate those target
invariants when constructing the plan; do not add a generic slow modulo path.

### 2. Fuse RoPE, conversion, and cache store

Do not add an F32-to-F16 conversion dispatch after the existing producers.
Replace the current K-RoPE destination plus V cache write with one
Gemma-specific Metal kernel:

`raw K (F32) + raw/projected V (F32) -> RoPE(K) F16 cache + V F16 cache`.

One thread owns one head element for one live row. It computes the existing
NeoX RoPE expression in F32, converts the final K and V values to half, and
writes both to the selected physical row. K and V have the same 8x256 target
shape, so the two stores share a grid and row mapping.

- Prefill: use existing `raw_key` and `value` F32 scratch; remove the separate
  F32 `key` scratch, the key-RoPE dispatch, and the KV blit encoder.
- Decode: redirect the V projection to a new 1x2048 F32 `raw_value` scratch;
  replace the current K-RoPE-to-cache dispatch with the fused K/V store. This
  changes no per-layer dispatch count and adds only 8 KiB of decode scratch.
- Q RoPE remains unchanged and F32.

This producer-store fusion is preferable to duplicating every Q4_K/Q6_K
projection pipeline with half-output variants. It keeps the shared projection
selector intact, avoids pipeline proliferation, and removes the prefill blit.

### 3. Read half, accumulate float

Replace, rather than duplicate, all resident cache readers:

- decode online-softmax attention loads aligned `half4` K/V vectors, converts
  them to `float4`, and retains its F32 scores, maxima, denominators, and value
  accumulators;
- prefill score and probability-times-V kernels load F16 cache operands and
  retain F32 query/probability/output arithmetic;
- each kernel receives the logical first position and a cache-row mask, and
  maps each logical token to a physical row before its vector loads;
- the row is 4096 bytes for one F16 K or V token (`8 * 256 * 2`), so every row
  and `half4` access remains naturally aligned.

Do not transpose V or introduce a second cache layout in this slice. The fused
decode kernel coalesces lanes across a token row, and the planned batched fused
prefill-attention work also wants token-major tiled K/V. A dimension-major V
copy would trade one present kernel's stride for extra storage and conversion.

## More Aggressive Research Outside llama.cpp

Low-bit cache work is real but is not the active implementation:

- KIVI shows that K and V have different error structure: K benefits from
  per-channel quantization, V from per-token quantization, and a recent
  high-precision residual window is important for difficult generation.
- TurboQuant and related rotation-based schemes reduce cache traffic further,
  but add transforms, metadata, packing, and a new model-quality contract.
- A May 2026 preprint reports a fused INT4 Metal cache faster than F16 by
  roughly 3-8% on Gemma 3 1B over 256-4096-token prefixes, with 3x persistent
  compression. That is promising Apple-specific evidence, but it is a v1
  preprint on different, much smaller models; its reported Gemma quality delta
  is not a sufficient correctness basis for Gemma 2 9B.

The original follow-up worth preserving is a fused rotated INT4 cache reader
inside this engine's online-softmax kernel, not a standalone quantize/dequantize
pipeline. Consider it only after the F16 implementation proves from GPU
counters that long-context attention is bandwidth-bound. It must be a separate
reversible experiment with Gemma 2 9B perplexity/KL, retrieval, deterministic
prompt, and real token-time evidence. Do not carry INT4 scaffolding into the
F16 baton.

## Required Code Changes

1. Extend `Gemma2MetalKVCacheLayer` with the physical row capacity/mask (or a
   single explicit layout object) so logical position, capacity, dtype, and
   sliding/global policy are not inferred independently at call sites.
2. Allocate F16 cache descriptors per layer: context rows for global layers,
   sliding-window rows for sliding layers. Reject non-power-of-two target
   context/window configurations in the Metal product plan.
3. Make the resident prefill chunker stop at the next sliding-ring boundary.
4. Add the fused F32 raw-K/raw-V to RoPE-F16-K/F16-V cache-store kernel and one
   command-context entry point shared by decode and prefill.
5. Add decode `raw_value` scratch and delete prefill `key` scratch.
6. Replace decode fused attention and both prefill cache readers with F16,
   ring-aware kernels using vector half loads and F32 accumulation.
7. Update byte accounting and externally reported resident-KV metrics to the
   actual heterogeneous allocation.
8. Keep the pure CPU F32 attention and model reference code as independent
   oracles, but explicitly round each reference K/V store through F16 and map
   logical sliding positions through the ring when comparing production state.

## Legacy Removal Boundary

Delete these only in the same change that lands their proven replacement:

- full-context F32 cache allocation and `sizeof(float)` cache-offset arithmetic
  in `gemma2_metal_plan.cpp` and `gemma2_runtime.cpp`;
- `MetalCommandContext::encode_kv_cache_write` and its blit implementation;
- the F32-only resident decode-attention kernel/pipeline/validation path;
- the F32-cache prefill score/value kernel variants; retain no dtype selector
  or dormant F32 pipeline;
- the test-only public backend-buffer functions
  `write_kv_cache_f32_to_backend`,
  `write_prefill_kv_cache_f32_to_backend`,
  `causal_grouped_attention_scores_f32_with_kv_cache_to_backend`, and
  `causal_grouped_attention_values_f32_with_kv_cache_to_backend`, plus their
  Metal helpers and validation;
- the tests named `writes prefill KV cache to CPU backend buffer`,
  `matches prefill KV cache write to Metal backend buffer when available`,
  `applies causal grouped attention to CPU backend KV cache buffers`, and
  `matches causal grouped attention KV cache to Metal backend buffers when
  available`;
- F32/full-capacity descriptor expectations, F32 cache-byte formulas, and raw
  full-buffer byte-equality assertions in the Gemma plan tests.

Replace those tests with tests of the production F16/ring contract. Do not
delete independent CPU math or end-to-end coverage merely because it exposed
the old representation.

## Verification

### Deterministic correctness

1. Plan test: assert F16 dtype, 4096 sliding capacity, 8192 global capacity,
   power-of-two masks, and exactly `2,113,929,216` target bytes.
2. Store-kernel test: compare K against CPU RoPE followed by the repository's
   exact F32-to-F16 conversion and V against exact F16 conversion. Cover row 0,
   the last physical row, the first wrapped row, mixed signs, large/small
   values, and F16 rounding boundaries.
3. Ring-liveness test: prefill across `window - 1`, `window`, and
   `window + 1`; seed overwritten slots with sentinels and prove every logical
   live row and only the intended physical rows are read/written. Use the tiny
   model's 8-token window for fast integration coverage and exact 4096 for the
   device kernel.
4. Decode-attention parity: retain token counts 1, 31, 32, 33, 256, 257, 4096,
   and 8192 for global layers; add sliding logical end positions 4095, 4096,
   4097, 8191, and 8192. The CPU oracle must consume values after F16 rounding.
5. Resident decode parity: compare logits, selected token/logit, and the
   logical live cache contents for both a sliding and global layer. Include a
   decode immediately after a ring wrap.
6. Resident prefill parity: compare one-call and incremental execution across
   a ring boundary, including logits, selection, logical K/V state, chunk
   count, one final wait, zero transient allocations, and zero KV blit encoders.
7. Preserve unavailable-Metal failure behavior and all context-overflow checks.
8. Run `bash scripts/ci.sh` and the complete device-backed test binary.

Choose tolerances from measured F16 cache error. Do not compare against the old
F32 cache with an arbitrarily widened tolerance, and do not require byte-equal
logits after intentionally changing cache precision. Require exact F16 cache
bits, bounded F32 outputs, and unchanged selected tokens for the canonical
deterministic runs.

### Performance and memory

Measure release builds and interleave the preserved pre-change binary:

1. Plan creation: wall time, reported KV bytes, process resident memory, and
   allocation success under normal memory pressure.
2. Canonical short prompt: prompt tok/s, TTFT, decode tok/s, mean/median/stddev,
   and deterministic tokens. This is a regression gate, not evidence for the
   long-context bandwidth claim.
3. Real Gemma 2 9B contexts immediately before and after the 4096 wrap and near
   8192: measure per-token GPU duration and decode tok/s over repeated tokens
   from one warmed session. Attribute sliding and global attention GPU time
   separately if counters permit.
4. Record cache bytes read, achieved bandwidth, occupancy, and time for the
   exact-shape decode-attention kernel at 256, 4096, and 8192 tokens. The
   expected traffic ceiling from F32 to F16 is 2x; do not claim a 2x end-to-end
   speedup.
5. Confirm that boundary-aware chunk splitting affects only wrap-crossing
   chunks and does not add a synchronous wait.

## Constraints

- Keep inference engine-owned, Metal-first, model-resident, and offline.
- Optimize only the Gemma 2 9B target shapes; no generic paged-cache allocator,
  multi-sequence slot manager, dtype configuration surface, or CPU production
  cache is needed.
- Preserve 8192 logical context semantics, alternating global/sliding layers,
  RoPE positions, attention softcap, GQA mapping, and the current online
  softmax algorithm.
- Never keep both F32 and F16 production cache paths after acceptance.
- Do not combine this work with batched fused prefill attention, gate/up
  fusion, sampling, or lower-bit cache quantization.

## Stop Condition

Stop when one coherent F16/exact-ring production path has replaced the F32
cache end to end, all replacement and integration tests pass, the old code and
old representation-only tests are deleted, target KV allocation is 1.96875 GiB,
the canonical short path does not regress beyond variance, and real
long-context measurements show whether the change improves token time.

If F16 cache reads regress long-context token time beyond variance, remove the
candidate completely and record the measured rejection here. Do not retain a
fallback or proceed to INT4 in the same slice.

## Primary Sources

- llama.cpp pinned Gemma 2 model:
  <https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/src/models/gemma2.cpp>
- llama.cpp pinned interleaved sliding/global cache:
  <https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/src/llama-kv-cache-iswa.cpp>
- llama.cpp pinned context defaults:
  <https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/src/llama-context.cpp>
- llama.cpp pinned common CLI defaults:
  <https://github.com/ggml-org/llama.cpp/blob/b15ca938ad00aa6b3ee6c2edda7363fd02826b18/common/common.h>
- Google Gemma reference configuration:
  <https://github.com/google/gemma_pytorch/blob/main/gemma/config.py>
- Apple Metal shader performance guidance:
  <https://developer.apple.com/videos/play/tech-talks/111373/>
- KIVI:
  <https://arxiv.org/abs/2402.02750>
- TurboQuant:
  <https://openreview.net/forum?id=tO3ASKZlok>
- Apple-Silicon INT4 KV-cache preprint:
  <https://arxiv.org/abs/2605.05699>
