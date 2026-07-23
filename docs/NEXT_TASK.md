# Next Task: Attribute and Remove Redundant Metal Synchronization

## Verified Current State (2026-07-23)

The direct-dequantizing prefill rewrite is complete. Q4_K and Q6_K now each
have one packed small-batch family for rows 2-8 and one direct-dequantizing
SIMD-group tiled family for rows 9 and above. The tiled family has a generic
64x32x32/four-SIMD schedule and an exact 64x16x32/two-SIMD specialization of
the same algorithm. All legacy 2/4/8/16 row-serial kernels, selectors, pipeline
identities, helpers, and implementation-coupled tests are deleted. One-row
packed decode is unchanged.

The resident prefill chunk is now 128. Measured resident allocation is:

| Resource | Bytes |
| --- | ---: |
| weights | 6,837,368,832 |
| F16 KV cache | 2,113,929,216 |
| scratch | 181,146,892 |

Warm release measurements on the base M4, with five repetitions where noted:

| Prompt tokens | Median prefill | Prompt rate | Spread / note |
| ---: | ---: | ---: | --- |
| 16 | 174.834 ms | 91.515 tok/s | 174.619-175.275 ms, 5 reps |
| 32 | 245.811 ms | 130.182 tok/s | 245.440-246.209 ms, 5 reps |
| 121 | 851.431 ms | 142.114 tok/s | 848.944-854.214 ms, 3 reps |

Decode remains approximately 14.2 tok/s. Plan creation measured 11.3-11.8
seconds with the retained pipelines. The standard greedy eight-token output
matches the preserved row-serial binary exactly: `The sky appears blue due to
a phenomenon`.

The numerical stress distribution that selected the permanent tolerances was:

| Dtype | Full-F32 max / RMS | F16-staged max / RMS |
| --- | ---: | ---: |
| Q4_K | 2.383 / 0.399 | 3.256 / 0.744 |
| Q6_K | 0.000977 / 0.000047 | 0.001465 / 0.000140 |

The Q4_K distribution reflects large stress magnitudes plus F16 operand
rounding and SIMD-group accumulation order. Small-batch kernels retain the
tighter full-F32 contract. Permanent tests use dtype- and family-specific
combined absolute/relative tolerances rather than the old shared thresholds.

## Active Finding

Finding 3 in `docs/LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md` is now the first
unfinished finding. One decoded token currently uses 635 compute dispatches
and 466 explicit buffer barriers in a single compute encoder and command
buffer. The resident schedule inserts barriers conservatively around operation
boundaries even when adjacent projections only read the same activation and
write disjoint buffers.

Projection kernels are no longer the dominant unexplained difference at the
controlled shapes. The next task is to measure how much GPU and CPU time is
spent in dispatch/barrier structure, then remove only barriers proven
unnecessary by exact resource-range dependencies.

## Objective

Implement engine-owned Metal resource dependency tracking for the resident
Gemma schedule, replacing fixed barriers with barriers derived from read/write
hazards. Preserve one command buffer and one compute encoder per chunk/token.
Retain the change only if whole decode or prefill improves beyond noise without
changing logits, selected tokens, KV contents, cache position, or unavailable-
device behavior.

Do not fuse operations in the first slice. Dependency tracking and fusion must
remain separately attributable.

## Required Investigation

1. Enumerate every `MetalCommandContext` encode operation's buffer ranges and
   access mode, including aliased scratch views and nonzero output offsets.
2. Map every explicit barrier in resident prefill and decode to the actual
   RAW, WAR, or WAW dependency it protects.
3. Add temporary GPU timing around representative layer regions if Metal
   counter sampling is supported; otherwise use command-buffer GPU timestamps
   with controlled schedule variants.
4. Establish the maximum possible gain by timing a temporary barrier-minimized
   schedule only after proving its hazards manually. Do not retain an unsafe
   no-barrier mode.
5. Choose the smallest dependency representation that handles exact buffer
   byte ranges. Whole-buffer identity is insufficient where scratch views or
   output offsets alias.

## Selected Design Boundary

Dependency state belongs to `MetalCommandContext`, which owns command encoding
and barriers. Engine scheduling may describe buffer access but must not own
Metal hazard mechanics.

The first retained design should:

- record reads and writes as backend-buffer identity plus byte range;
- insert a barrier before an operation only when its accesses overlap an
  unresolved prior write or when a write overlaps an unresolved prior read;
- clear or advance dependency state after an inserted barrier;
- treat unknown or overflowed ranges conservatively;
- preserve explicit lifecycle barriers that protect command-buffer/CPU
  visibility rather than intra-encoder tensor hazards; and
- expose no benchmark switch or public dependency API.

Do not add a general graph compiler, runtime autotuner, external dependency,
or second execution path.

## Correctness Evidence

Add deterministic tests for:

- disjoint read/read, read/write, and write/write ranges;
- overlapping RAW, WAR, and WAW ranges;
- exact-touching non-overlapping boundaries;
- subviews and nonzero output offsets;
- overflow/unknown ranges taking the conservative path;
- unchanged resident prefill logits, selected token, F16 KV bytes, cache
  position, chunk/ring boundaries, and submission statistics;
- unchanged decode logits/tokens and one-row projection parity; and
- graceful unavailable-Metal behavior.

Run `bash scripts/ci.sh` after implementation changes.

## Performance Evidence

Use release builds, warm pipelines, controlled thermals, and at least five
measured repetitions. Report before/after:

- decode latency and tok/s for the standard prompt and a longer context;
- 16-, 32-, and approximately 128-token prefill latency and tok/s;
- dispatch, barrier, encoder, command-buffer, commit, wait, allocation, and
  readback counts;
- GPU duration when available, runtime TTFT, and end-to-end TTFT;
- plan creation time and resident bytes; and
- exact greedy and seed-42 sampled outputs.

Interleave baseline and candidate when thermal drift could bias the result.
The candidate must reduce actual barriers and improve a whole production path;
a counter-only change with no measurable latency benefit is not enough.

## Constraints

- Preserve engine-owned, offline, Metal-first, model-resident inference.
- Preserve native GGUF weights, the new projection families, the 128-row
  chunk, F32 outputs, F16 ring KV, compact selection, and decode behavior.
- Do not mix fusion, Flash Attention, quantization, kernel geometry, sampling,
  tokenizer, CLI, server, or unrelated cleanup into this task.
- Add no compatibility alias, silent backend fallback, benchmark-only public
  API, or generated benchmark artifact.
- Preserve unrelated user changes.

## Stop Condition

Stop when dependency tracking is correct for every resident operation, all
superseded fixed intra-encoder barriers are deleted, deterministic parity and
CI pass, and real decode or prefill improves beyond noise. Delete all timing
probes and rejected schedule variants.

If a safe barrier-minimized candidate does not improve a whole production
path, delete it, retain the current explicit schedule, record the negative
result in this file, and choose a fresh measured bottleneck. Do not retain
dependency machinery solely because it is architecturally attractive.
