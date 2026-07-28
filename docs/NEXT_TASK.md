# Next Task: Prove and Reduce Decode Barrier Cost

## Objective

Replace the decode path's whole-buffer-scope Metal barriers with the smallest
verified dependency-aware synchronization candidate, then retain it only if the
fixed product conversation improves beyond its measured noise without changing
any output, cache, memory, or lifecycle fact.

This is Finding 3 from `docs/LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md`. Do not combine
it with kernel fusion, projection math changes, prefill scheduling, generation
ordering, or a new command-buffer lifecycle.

## Verified Starting State (2026-07-25)

The product-level benchmark task is complete:

- schema 3 scenario 1 runs the four-turn conversation through the production
  `Gemma2ChatSession`, resident plan, runtime, Metal backend, sampler, stop
  handling, synchronous stream callback, and session history;
- one warmup and every measured repetition are retained as raw JSONL records;
- exact rendered prompt IDs and selected IDs must remain stable or validation
  fails and affected summaries are suppressed;
- primary timing excludes per-turn memory sampling, with one separate labelled
  memory-audit conversation;
- a PTY calibration showed callback-complete TTFT differs from externally read
  first assistant bytes by only 28-68 microseconds in three probes;
- the baseline/candidate gate checks exact workload and state, requires at least
  five paired conversations and four wins, rejects material per-turn or memory
  regressions, and requires improvement beyond
  `max(2%, 2 * baseline_MAD / baseline_median)`; and
- the old independent four-case generation benchmark is removed.

The prior long-generation nondeterminism was a real engine bug, not benchmark
noise. `MetalCommandContext::encode_embedding` dispatched only one 16-row
threadgroup in the token dimension, so prefills longer than 16 tokens left
later rows of the resident scratch buffer stale. The dispatch now covers
`ceil(tokens / 16)` threadgroups. A 17-token Metal embedding parity test guards
the boundary. With that fix, all five measured conversations have identical
rendered prompts and selected lengths of 32, 23, 97, and 128 tokens.

### Local release baseline

Model identity:
`models/gemma-2-9b-Q4.gguf`, 6,843,425,696 bytes, mtime
`1780675613669034390`, on Apple M4 / macOS Darwin 25.5.0. Build identity was
`bf6a82d-dirty`; generated result files remain ignored local artifacts.

One warmup plus five measured conversations produced:

| Turn | Median frontend-complete TTFT | MAD | Median complete turn |
| --- | ---: | ---: | ---: |
| 1 | 331.988 ms | 19.218 ms | 2.524 s |
| 2 | 261.612 ms | 9.764 ms | 1.831 s |
| 3 | 344.751 ms | 19.877 ms | 7.193 s |
| 4 | 372.853 ms | 31.954 ms | 9.546 s |

The median complete conversation was 21.094 seconds with MAD 0.346 seconds.
The primary aggregate (sum of the four frontend-complete TTFTs per
conversation) had median 1.300 seconds and MAD 68.955 ms, so the current
retention threshold is 10.605%. Late samples slowed materially from thermal
drift; do not weaken this threshold or discard the tail.

Plan creation was 12.599 seconds. Logical resident bytes were 6,837,368,832
weights, 2,113,929,216 KV, and 181,146,892 scratch. Metal current allocation
was 9,133,539,328 bytes. The separate memory audit remained stable across all
turns.

### Pinned reference

The verified reference is Homebrew `llama.cpp` build 9430 at commit
`d48a56effbba99944a677938d4beb63a0065ecdf`. Both required views ran one warmup
plus five measured conversations with the same model and passed reference
validation.

Reference median external-observed TTFT was 431.429, 642.707, 895.380, and
2,025.191 ms for turns 1-4. Median complete conversation was 24.193 seconds.
Exact selected IDs matched the local engine on all four turns in all five
measured repetitions when the exact paired prompt IDs were submitted.

The paired-workload view produced the reference medians above and matched exact
selected IDs on every turn in every repetition. The independent product view
matched local prompt and selected IDs through turns 1-3. Before turn 4,
`llama.cpp`'s template trims trailing whitespace from the prior assistant text,
producing 227 prompt IDs instead of the local token-native session's 229; it
then selects 122 tokens instead of the local limit of 128. The product-view run
occurred immediately after sustained benchmark load and was severely thermally
throttled, so its latency is retained in raw output but is not a controlled
speed baseline.

The paired comparator records the template mismatch while submitting the 229
exact local IDs for the fair compute-work view. The pinned server also reports
only 10 cached tokens on follow-ups versus the local engine's 47, 86, and 204;
this is a measured reference product-path difference, not a validation failure.
Slot erase is a hard prerequisite between repetitions and the launched server
uses an isolated temporary `--slot-save-path` because this pinned build
otherwise disables the erase endpoint.

## Finding 3 Hypothesis

A one-token 42-layer decode currently encodes approximately 635 compute
dispatches and 466 explicit buffer barriers in one compute encoder and one
command buffer. The benchmark observes 465 barriers per resident decode call
in the current selected-token path (the historical 466 count included a
slightly different endpoint). Every barrier currently uses
`MTLBarrierScopeBuffers`, even though each dependency involves a known, small
set of producer and consumer buffers.

The first candidate is resource-scoped synchronization for the existing decode
schedule. It may reduce GPU synchronization cost even when the number of true
dependency boundaries is unchanged. If it does not improve the real path
beyond noise, remove it rather than retaining a generalized dependency layer.
Only after that measurement may a follow-up task consider eliminating a proven
redundant boundary or fusing neighboring cheap operations.

## Implementation Boundary

1. Add a narrow Metal command-context operation that encodes a memory barrier
   for explicit backend buffers using `memoryBarrierWithResources:count:`.
   Validate non-null Metal-owned resources and preserve unavailable-device
   behavior.
2. In resident one-token decode only, replace each whole-buffer-scope barrier
   with the exact written resources required by the next dependent operation.
   Keep independent Q/K/V projections and gate/up projections together as they
   are now; do not insert synchronization between independent writes.
3. Keep the existing global-scope operation for call sites outside this slice.
   Do not introduce a graph scheduler, general hazard tracker, or public tensor
   API unless repeated concrete cases prove that abstraction necessary.
4. Propagate barrier kind/count through existing execution statistics so raw
   benchmark records can distinguish global-scope from resource-scoped
   barriers. Measurement must add no command buffer, encoder, wait, allocation,
   or readback.
5. If resource-scoped barriers are unsupported on the target API/device, fail
   the candidate clearly and leave the production path unchanged; do not
   silently omit synchronization.

Likely anchors:

- `src/tensor/backends/metal/metal_command_context.hpp`
- `src/tensor/backends/metal/metal_command_context.mm`
- `src/engine/gemma2_runtime.cpp`
- `include/local_gguf_engine/engine/gemma2_runtime.hpp`
- benchmark JSON serialization and focused Metal/runtime tests

## Correctness Proof

Add deterministic coverage that:

- validates the exact resource list and rejects invalid/non-Metal buffers;
- records resource-scoped and global barrier counts separately;
- preserves one command buffer, one compute encoder, one commit, one wait, zero
  transient allocations, and the existing selector/readback behavior per
  decode;
- covers unavailable Metal behavior;
- passes existing tensor, runtime, chat-session, CLI, and benchmark tests; and
- preserves exact selected IDs, visible IDs/text, stop reason, rendered prompts,
  suffixes, cache lengths, plan allocation identities, and memory facts in a
  real generation run.

Run `bash scripts/ci.sh` after implementation changes.

## Performance Verification

Build a release baseline binary before changing synchronization and preserve it
outside the build tree. Run baseline and candidate with the same model, backend,
scenario, warmup, repetitions, sampler, machine power state, and synchronous
frontend. Interleave binaries when practical to reduce thermal-order bias.

Use `scripts/compare_local_conversation.py` for the verdict. A candidate passes
only when:

- both JSONL validation records are true;
- every exact workload, output, stop, cache, memory, and plan fact matches;
- at least five measured conversations are paired and at least four improve;
- primary aggregate improvement exceeds the baseline-derived threshold; and
- no individual turn's TTFT or complete latency, nor memory audit fact, has a
  material regression.

Also report per-decode global/resource barrier counts, GPU duration, command
buffers, encoders, dispatches, waits, compact readback, and decode throughput to
test the claimed mechanism. These diagnostics explain the result; they do not
override the product gate.

## Constraints

- Preserve engine-owned, offline, Metal-first, model-resident inference.
- Do not call or link `llama.cpp` from production code.
- Do not change kernel math, tensor layout, cache semantics, prompt formatting,
  sampler/stop semantics, generation order, or callback order.
- Do not remove a barrier without an explicit producer/consumer dependency
  proof and parity evidence.
- Do not add per-kernel waits, timers, allocations, or polling to primary
  timing.
- Preserve unrelated user changes and do not commit generated benchmark output.

## Stop Condition

Stop after one coherent resource-scoped decode-barrier candidate is fully
tested and measured. Retain it only if the product gate passes and diagnostics
support the synchronization-cost hypothesis. Otherwise remove the candidate,
record the negative result here, and choose the next measured bottleneck rather
than broadening this task into fusion or scheduler work.
