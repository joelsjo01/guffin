# Next Task: Fuse Decode Waves Without Sacrificing Concurrency

## Objective

Reduce Finding 3's fixed Metal decode cost without serializing work that can
run concurrently. Keep one concurrent compute encoder and replace the current
standalone Q/K/V, Q/K/V postprocess, and gate/up/GELU dispatch groups with
three fused, target-shaped kernel families. Synchronize only the exact produced
resources required by each dependent wave.

Retain the implementation only if the fixed Gemma 2 9B Q4 product conversation
improves beyond measured noise with exact output, cache, memory, dispatch,
command-lifetime, and frontend parity.

This is Finding 3 from `docs/LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md`. Do not combine
it with serial dispatch, attention math changes, prefill scheduling, output-head
changes, generation reordering, command-buffer reuse, indirect command buffers,
or Metal 4 migration.

## Decision

Do not trade concurrency for fewer barriers. Preserve
`MTLDispatchTypeConcurrent` and remove fixed cost by doing less work:

1. encode Q, K, and V projections as one dispatch whose threadgroups cover all
   three output grids;
2. encode Q RoPE plus K-RoPE/V cache storage as one dispatch whose independent
   threadgroups cover both postprocess regions;
3. compute each matching gate/up row pair and its GELU-tanh product in one
   cooperative dispatch, writing only the gated result; and
4. replace whole-buffer-scope barriers with resource-scoped barriers naming
   the exact outputs crossing each dependency boundary.

This retains parallel GPU scheduling inside each independent wave, reduces
dispatch count, removes two FFN activation round trips, removes one true
per-layer dependency boundary, and narrows every remaining barrier. It is the
best no-compromise direction for the current fixed Gemma target.

Do not keep the existing standalone production sequence as a fallback for the
target model. If the fused candidate fails correctness or the product gate,
remove it completely and restore the pre-task path.

## Verified Starting State (2026-07-28)

### Live production counts

`run_gemma2_metal_resident_decode_step` currently creates one command buffer,
one concurrent compute encoder, one commit, one synchronous wait, zero blit
encoders, and zero transient backend-buffer allocations per decoded token.

The current selected top-1 path is:

```text
dispatches = 1 embedding + 42 * 15 layer dispatches
           + 1 final RMSNorm + 1 output projection + 1 selector
           = 634

barriers = 1 after embedding + 42 * 11 layer barriers
         + 1 after final RMSNorm + 1 after output projection
         = 465
```

The existing `635`/`466` formulas are stale. A former separate final
softcap/endpoint dispatch disappeared when softcap moved into compact token
selection. The live two-layer test already observes 34 dispatches with
selection, 33 without selection, and 25 barriers.

For contexts through 256 tokens, fused decode attention uses one dispatch. At
longer contexts its partition reduction adds one internal dispatch and one
dependency boundary; that dynamic count remains separate from the fixed
short-context formulas.

### Exact target weight facts

Inspection of `models/gemma-2-9b-Q4.gguf` and the plan bindings confirms:

- all 42 query, key, value, attention-output, gate, and up matrices are Q4_K;
- all 42 down matrices are Q6_K;
- Q/K/V share the same 3,584-element input;
- gate/up share the same `[3584, 14336]` layout; and
- Q/K/V and gate/up already use the same packed cooperative Q4_K decode
  primitive family.

The GGUF contains 252 Q4_K tensors, exactly six Q4_K matrices per layer, and 42
Q6_K tensors, exactly one down matrix per layer. The proposed fused families
therefore specialize a repeated production fact rather than inventing a
general multi-matrix abstraction.

### Existing benchmark

Product conversation scenario 1 already exercises the production chat
session, resident plan/runtime, Metal backend, selection, stops, synchronous
streaming, cache reuse, and four-turn history. It preserves raw records and
validates exact prompt and selected IDs.

The last verified local run had median complete-conversation latency of 21.094
seconds with MAD 0.346 seconds. Its current comparison script uses summed TTFT
as the primary metric. TTFT is dominated by prefill plus first-token selection
and cannot prove a subsequent-decode optimization. The performance section
below adds a decode-specific mode while preserving TTFT mode for prefill and
frontend work.

## Research Findings

### Metal concurrency and barriers

Apple documents that a concurrent compute encoder performs no automatic
synchronization between commands that access common resources; the application
must encode the required barriers. Apple also recommends encoding independent
work early and using GPU traces to prove overlap rather than assuming it.

Primary references:

- Apple, Resource synchronization:
  <https://developer.apple.com/documentation/metal/resource-synchronization>
- Apple, `MTLDispatchType.concurrent`:
  <https://developer.apple.com/documentation/metal/mtldispatchtype/concurrent>
- Apple, Porting Metal code to Apple silicon:
  <https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon>
- Apple, Scale compute workloads across Apple GPUs:
  <https://developer.apple.com/videos/play/wwdc2022/10159/>
- Apple, Optimizing GPU performance:
  <https://developer.apple.com/documentation/xcode/optimizing-gpu-performance/>

The current encoder choice is therefore sound when independent work exists.
The problem is that 634 separately encoded dispatches and 465 broad barriers
express the schedule at too fine and too conservative a granularity.

### What llama.cpp contributes—and what it does not

Both pinned llama.cpp commit
`d48a56effbba99944a677938d4beb63a0065ecdf` and current upstream keep a
concurrent encoder and use source/destination byte-range tracking to decide
when a node conflicts with the active wave. On a conflict they still call
`memoryBarrierWithScope:MTLBarrierScopeBuffers`; they do not issue a
resource-scoped barrier.

Primary source:

- pinned operation scheduler:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal-ops.cpp>
- pinned encoder/barrier wrapper:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal-device.m>
- current operation scheduler:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp>
- current encoder/barrier wrapper:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m>

The relevant lesson is to expose independent work in one concurrent wave. A
generic range tracker is not appropriate here: the Gemma decode schedule and
buffer ownership are fixed and already explicit. It would rediscover almost
all true dependencies while adding alias metadata and a second scheduling
model.

### Why fusion is the higher-ceiling answer

LLM inference research identifies fragmented operator execution, off-chip
intermediate traffic, and launch/scheduling overhead as recurring decode
costs. Fusion is strongest when it both removes a dispatch and avoids an
intermediate memory round trip.

References:

- ClusterFusion, expanded operator fusion for LLM inference:
  <https://arxiv.org/abs/2508.18850>
- TaxBreak, inference overhead decomposition:
  <https://doi.org/10.1109/ISPASS69572.2026.00014>

The three chosen fusions have different benefits:

- fused Q/K/V projection removes two host dispatch commands while preserving
  the same weight reads and exposing all projection threadgroups together;
- fused Q/K/V postprocess removes one dispatch while preserving independent
  query and cache threadgroups; and
- fused gate/up/GELU removes two dispatches, one dependency boundary, the F32
  gate and up writes, their later reads, and two resident scratch buffers.

The gate/up fusion is especially well matched to the retained packed Q4_K
kernel. Its cooperative inner loop already accumulates two values from a
shared activation traversal. The new specialization pairs the same output row
from the gate and up matrices, applies stable Gemma GELU-tanh once both F32
accumulators are reduced, and emits their product.

### Why not use serial dispatch

Serial dispatch would delete explicit barriers but would also prevent Q/K/V
and gate/up from being scheduled concurrently. These groups stream large
matrices and may contend for bandwidth, but no local trace proves their overlap
worthless. Removing concurrency first would exchange one unmeasured cost for
another.

The fused design avoids that bet. Q/K/V threadgroups remain available to the
scheduler in one grid, and gate/up work is coupled at the row level so the pair
shares activation loads and completes as one useful result.

### Why resource-scoped barriers belong after fusion

`memoryBarrierWithResources:count:` orders only named Metal resources. It does
not eliminate a dependency boundary, but after fusion it prevents unrelated
buffers and independent work from being included in that boundary. This is a
narrow fixed-schedule primitive, not a generalized hazard tracker.

Reference:

- Apple, `memoryBarrier(resources:)`:
  <https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/memorybarrier%28resources%3A%29>

The implementation must still be retained only on real-path evidence. Apple
does not promise a specific latency reduction over `MTLBarrierScopeBuffers`.

### Alternatives

| Candidate | Decision | Reason |
| --- | --- | --- |
| Concurrent fused waves plus resource barriers | Implement | Preserves parallel scheduling, removes dispatches and activation traffic, and narrows synchronization. |
| Serial encoder with no barriers | Reject | Removes synchronization commands by giving up unmeasured independent overlap. |
| Resource barriers without fusion | Reject | Retains 465 commands and all intermediate traffic; too small a ceiling. |
| Generic range/dependency tracker | Reject | Duplicates a fixed schedule with generalized alias machinery. |
| More command buffers or encoders | Reject | Reintroduces transitions and submission complexity removed by `39dce7e`. |
| Indirect command buffers | Reject | Dynamic bindings and dependent waves still require synchronization. |
| Metal 4 migration | Reject | Broad API/lifetime change without isolated evidence for this bottleneck. |
| Persistent whole-layer kernel | Reject | Metal has no grid-wide synchronization within an ordinary dispatch; dependent layer stages cannot safely become one monolithic kernel. |

## Target Decode Schedule

For short-context attention, each layer becomes ten dispatches:

1. attention RMSNorm;
2. fused Q/K/V projection;
3. fused Q-RoPE/K-RoPE/V-cache postprocess;
4. fused decode attention;
5. attention output projection;
6. fused post-attention RMSNorm-add;
7. feed-forward RMSNorm;
8. fused gate/up/GELU-tanh projection;
9. down projection; and
10. fused post-FFN RMSNorm-add.

There is one resource-scoped dependency barrier after each layer dispatch
because every next wave consumes its predecessor. The last barrier also makes
the new residual visible to the next layer. Embedding, final RMSNorm, output
projection, and token selection retain their existing order.

Expected selected top-1 short-context structure:

```text
dispatches = 1 embedding + 42 * 10 fused-layer dispatches
           + 1 final RMSNorm + 1 output projection + 1 selector
           = 424

barriers = 1 after embedding + 42 * 10 resource barriers
         + 1 after final RMSNorm + 1 after output projection
         = 423
```

This removes 210 dispatches and 42 barriers per selected token. More
importantly, it removes gate/up intermediate traffic while keeping the GPU's
ability to schedule independent Q/K/V work concurrently.

At attention token counts above 256, add the existing partition-reduction
dispatch and one resource barrier over the three partial buffers. Keep that
dynamic dependency inside the attention encoder operation and account for it
separately.

## Kernel Design

### 1. Fused Q/K/V packed projection

Add one Gemma decode Q4_K kernel that accepts:

- one normalized F32 activation vector;
- Q, K, and V Q4_K weight buffers;
- raw Q, K, and V F32 output buffers; and
- the shared input block count plus three output-row counts.

Use a single one-dimensional threadgroup grid partitioned into Q, K, and V
regions. A threadgroup belongs to exactly one region, so its branch is uniform.
Within each region, reuse the retained packed cooperative Q4_K inner loop and
two-output-row geometry unchanged. Do not concatenate or repack weights and do
not allocate an argument/descriptor buffer per token.

The grid must cover each region's odd-row tail independently. Bounds must be
based on that region, not the combined row count. Bind three weight and three
output buffers explicitly so ownership and validation remain visible.

### 2. Fused Q/K/V postprocess

Add one dispatch with independent uniform threadgroup regions for:

- applying NeoX RoPE to raw Q and writing `scratch.query`; and
- applying NeoX RoPE to raw K while converting/storing K and raw V into the
  current F16 ring-cache rows.

Reuse the current numerical formulas, paired-lane behavior, position, head
counts, dimensions, row mask, and checked cache addressing exactly. Do not
materialize an extra packed QKV buffer and do not make raw K in-place; the
existing paired-lane RoPE dependency still requires a distinct raw input.

### 3. Fused gate/up/GELU-tanh packed projection

Add one Q4_K decode kernel that accepts one normalized F32 activation, gate and
up Q4_K weights, and the gated F32 output.

For output row `r`, cooperatively accumulate gate row `r` and up row `r` during
one activation traversal. Preserve F32 accumulation and the current stable
Gemma GELU-tanh expression. After SIMD reduction, write exactly:

```text
gated[r] = gelu_tanh(gate_dot[r]) * up_dot[r]
```

Do not write standalone gate or up outputs. Preserve packed block decoding,
activation reuse, compile-time block-count specialization, lane ownership, and
tail handling from the retained Q4_K decode kernel where applicable.

Specialize the production pipeline for Gemma's 14 Q4_K input blocks. A generic
block-count function-constant variant may exist only if deterministic focused
tests need it; the target path must resolve to the exact specialization.

## Resource-Barrier Design

In `MetalCommandContext`, replace the zero-argument global barrier with a
narrow operation taking `std::span<const BackendBuffer>`.

It must:

- reject an empty list;
- reject CPU/non-Metal storage;
- reject nil resources;
- deduplicate identical `id<MTLResource>` values without allocation on the hot
  path, or reject duplicates with a clear contract;
- use a bounded stack array sized for the largest fixed dependency set;
- call `memoryBarrierWithResources:count:`; and
- separately count resource-scoped and whole-buffer-scope barriers during the
  candidate measurement.

The final target path must use resource barriers for every fixed decode
boundary. Keep a whole-buffer-scope operation only if a real non-target call
site still requires it. Do not preserve a dead global operation for tests or
future speculation.

Exact boundary resources include:

- embedded/new layer residual;
- normalized attention input;
- raw Q, K, and V together;
- query plus the current layer's key and value cache resources;
- attention output, or its partial buffers before reduction;
- projected attention output;
- post-attention residual;
- normalized FFN input;
- fused gated activation;
- projected FFN output;
- final layer residual;
- final normalized hidden; and
- logits before selection.

Weights are read-only and must never appear in a barrier list.

## Code Placement

### Metal shader and pipelines

- Add the three kernels beside the retained packed Q4_K decode and Gemma cache
  kernels in `src/tensor/backends/metal_backend.mm`.
- Put shared packed Q4_K block decoding/accumulation in the smallest private MSL
  helper that is genuinely shared. Do not create a generic fusion framework.
- Add exactly three target pipeline states and their construction in
  `src/tensor/backends/metal/metal_context.hpp` and `.mm`.
- Keep pipeline creation eager with the other resident plan pipelines; never
  compile a pipeline in the token hot path.

### Command encoding

In `src/tensor/backends/metal/metal_command_context.hpp` and `.mm`:

- add `encode_gemma2_qkv_projection`;
- add `encode_gemma2_qkv_postprocess`;
- add `encode_gemma2_gate_up_gelu_projection`;
- replace the global barrier API with explicit-resource synchronization;
- validate exact dtype, rank, dimensions, capacities, cache range, and Metal
  ownership at the command-context boundary; and
- add fused-family and barrier-kind counts to `MetalCommandStats`.

Encoding must add no command buffer, encoder, wait, allocation, readback, or
CPU-built per-token metadata buffer.

### Runtime integration

In `src/engine/gemma2_runtime.cpp`:

- replace the three Q/K/V `encode_matmul` calls with the fused Q/K/V encoder;
- replace Q RoPE plus KV store with the fused postprocess encoder;
- replace gate projection, up projection, and GELU multiply with the fused FFN
  encoder;
- pass exact produced resources at every barrier boundary;
- preserve operation order outside those three groups; and
- preserve selection, cache, metrics, callbacks, and command submission.

Do not route prefill through these one-token kernels.

### Resident plan cleanup

After parity passes, remove `feed_forward_gate` and `feed_forward_up` from
`Gemma2MetalDecodeScratch` in
`include/local_gguf_engine/engine/gemma2_metal_plan.hpp` and
`src/engine/gemma2_metal_plan.cpp`.

Delete their allocations, byte accounting, allocation-identity exposure,
validation, and tests. Keep `feed_forward_gated`; it remains the down
projection input. Do not retain the removed buffers behind diagnostics, a
fallback path, or a compile flag.

The generic Q4_K matvec kernel/pipeline remains because attention-output and
other single-matrix projections still use it. It is shared production code,
not legacy. Remove only target-path standalone calls and state made unreachable
by the fused schedule.

### Execution statistics and benchmark schema

Add counters for:

- fused Q/K/V projection dispatches;
- fused Q/K/V postprocess dispatches;
- fused gate/up/GELU dispatches;
- resource-scoped barriers; and
- remaining whole-buffer-scope barriers.

Propagate them through decode execution statistics and JSONL. Remove the old
single ambiguous `barriers` field. This changes the versioned record shape, so
increment conversation benchmark schema 3 to 4 while keeping scenario version
1.

To make the performance A/B schema-comparable, first create a reporting-only
schema-4 baseline that records current global barriers and zero fused/resource
counts without changing GPU execution. Preserve that release binary. Then
implement and build the fused candidate with the same schema.

Do not change `scripts/calibrate_local_chat_ttft.py`; its schema number belongs
to a separate calibration record.

Likely files:

- `src/tensor/backends/metal_backend.mm`
- `src/tensor/backends/metal/metal_context.hpp`
- `src/tensor/backends/metal/metal_context.mm`
- `src/tensor/backends/metal/metal_command_context.hpp`
- `src/tensor/backends/metal/metal_command_context.mm`
- `include/local_gguf_engine/engine/gemma2_metal_plan.hpp`
- `src/engine/gemma2_metal_plan.cpp`
- `include/local_gguf_engine/engine/gemma2_runtime.hpp`
- `src/engine/gemma2_runtime.cpp`
- `include/local_gguf_engine/benchmarks/conversation_benchmark.hpp`
- `src/frontends/cli/generate.cpp`
- `scripts/compare_local_conversation.py`
- `scripts/compare_llama_conversation.py`
- focused Metal, plan/runtime, benchmark, CLI, and script tests

## Correctness Proof

### Independent fused-kernel tests

Add isolated deterministic tests for each fused family using independent CPU
or composed retained-operation oracles. The fused implementation must never be
its own oracle.

Q/K/V projection coverage:

- nonzero Q4_K blocks and nonzero activation;
- different Q, K, and V row counts;
- one and multiple packed blocks;
- each region's two-row boundary and odd tail;
- exact Gemma input block count and production output shapes; and
- full output comparison, not selected-token comparison.

Q/K/V postprocess coverage:

- nonzero position;
- paired NeoX RoPE values;
- query, key, and value heads;
- F32 query and F16 cache parity;
- ring-mask wrap and first/last cache rows; and
- invalid position, mask, dimension, and capacity errors.

Gate/up/GELU coverage:

- independent nonzero Q4_K gate and up weights;
- positive, negative, small, and saturation-region accumulators;
- exact stable GELU-tanh parity at the existing tolerance;
- two-row group boundary and odd output tail;
- generic test block count and exact 14-block specialization; and
- proof that no standalone gate/up destination is allocated or written.

Every fused kernel test must cover unavailable Metal behavior.

### Runtime parity

Retain and update the existing nonzero two-layer CPU-reference resident decode
test. It already covers F16/Q4_K/Q6_K weights, sliding and global attention,
nonzero cache position, full logits, selected token/logit, and every written
K/V cache row.

Require for selected and no-selection paths:

- full-logit parity against the independent CPU reference;
- exact selected candidates where requested;
- exact F16 K/V ring contents;
- one command buffer, one concurrent compute encoder, one commit, one wait,
  zero blit encoders, and zero transient allocations;
- expected fused-family, legacy projection-family, dispatch, and resource
  barrier counts; and
- no whole-buffer-scope barrier in target decode.

Retain multi-partition attention tests and compact-selector merge/finalize
tests. Convert their internal barriers to exact resources and keep numerical
parity. Do not delete a useful numerical test merely because its old barrier
count changes.

### Test cleanup

Delete or replace:

- the stale `635`/`466` formula assertions;
- the old 25-barrier tiny-decode assertions;
- assertions that treat all barriers as one undifferentiated count;
- decode scratch descriptor/byte/identity assertions for standalone gate and
  up buffers;
- target-path tests that require three Q/K/V calls or separate gate/up/GELU
  calls; and
- schema-3 JSON fixtures for the conversation benchmark.

Do not delete generic Q4_K matvec parity, prefill tests, Q6_K down-projection
coverage, attention parity, selector parity, chat/session behavior, or CLI
success/failure coverage.

Add schema-4 serialization tests requiring separate resource/global barrier
counts and the three fused-family counts.

## Performance Verification

### Decode-aware product gate

Extend `scripts/compare_local_conversation.py` with an explicit primary metric
mode. Preserve its current TTFT mode and add decode mode. Schema-4 turn records
must serialize raw integer `decode_duration_ns`; do not reconstruct it from the
rounded `decode_tps` value.

For each paired measured conversation, decode mode compares:

1. the sum of all four turns' `decode_duration_ns` as the mechanism metric; and
2. complete conversation `total_ns` as the product metric.

The candidate passes only when:

- both validation records are true;
- schema, scenario, model, sampler, frontend, exact prompt/suffix/selected/
  visible IDs, final text, stops, cache facts, plan identities, and resident
  memory facts match;
- at least five measured conversations are paired;
- at least four of five improve for aggregate decode duration and complete
  conversation duration;
- both median improvements exceed their own baseline-derived threshold
  `max(2%, 2 * baseline_MAD / baseline_median)`;
- no turn's frontend-complete TTFT, total latency, or memory audit fact has a
  material regression;
- command buffers, encoders, commits, waits, compact readback, and unaffected
  projection-family counts match; and
- the candidate records 424 short-context selected dispatches, 423 resource
  barriers, zero global barriers, and one count for each fused family per layer.

The exact count comparison must account for long-context attention partition
reduction and the final no-selection decode step. Derive it from recorded
attention partitions and selector dispatches rather than weakening the check.

### Run protocol

1. Make the schema/statistics-only changes and build a release concurrent
   baseline outside the build tree.
2. Implement the complete fused/resource-barrier candidate and build a
   separate release binary.
3. Run `models/gemma-2-9b-Q4.gguf`, product scenario 1, greedy selection, the
   same synchronous frontend, one warmup, and at least five measured
   conversations per binary.
4. Interleave baseline and candidate conversations when practical. Retain late
   thermal samples and do not weaken thresholds.
5. Run the decode-mode comparator for the verdict.
6. Report per-turn and aggregate decode duration, decode throughput, GPU
   duration, complete turn/conversation duration, TTFT, paired wins, MADs,
   thresholds, commands, dispatches, fused-family counts, both barrier kinds,
   readback, and memory audit.
7. Take one Metal System Trace of each binary if available. Verify that the
   fused Q/K/V grid exposes all three regions without new GPU gaps and that the
   gate/up fusion improves rather than reduces occupancy. Trace evidence
   explains the mechanism but cannot override the product gate.

Run `bash scripts/ci.sh` and `git diff --check` after the final implementation.
Generated JSONL, traces, binaries, and model data remain ignored local
artifacts.

## Cleanup Audit

If the candidate is retained, use `rg` to prove the target decode path contains
none of:

- three standalone Q/K/V matmul calls;
- separate Q RoPE and KV-store calls;
- standalone gate/up projection calls;
- standalone GELU-tanh multiply after gate/up;
- `feed_forward_gate` or `feed_forward_up` decode scratch;
- zero-argument barriers in target decode;
- a target global-scope barrier;
- ambiguous serialized `"barriers"` fields; or
- stale 635/466 formulas.

Do not remove:

- shared generic Q4_K/Q6_K/F16 projection kernels still used by real paths;
- prefill scratch or prefill operations;
- Metal shader `threadgroup_barrier` or `simdgroup_barrier` calls; or
- independent reference and parity tests.

Remove temporary pipeline variants, debug counters, benchmark switches,
intermediate binaries inside the repository, and rejected kernel candidates.

## Constraints

- Preserve engine-owned, offline, Metal-first, model-resident inference.
- Do not call or link llama.cpp from production code.
- Preserve one concurrent compute encoder, one command buffer, one commit, one
  wait, zero blit encoders, zero transient allocations, and compact readback.
- Do not serialize Q/K/V or gate/up as the method of removing barriers.
- Do not change mathematical precision, tensor layouts, cache semantics,
  attention math, output-head math, prompt formatting, sampling/stops,
  generation order, or callback order.
- Do not repack or duplicate resident weights.
- Do not add a generic fusion graph, scheduler, hazard tracker, per-token
  allocation, per-kernel wait, timer, polling loop, or readback.
- Preserve unrelated user changes and never commit generated benchmark output.

## Stop Condition

Stop after the three fused decode families and exact resource barriers are
integrated, independently tested, and measured together as one coherent
no-compromise candidate. Retain them only if the decode mechanism gate and
complete-conversation product gate both pass and trace/count evidence supports
the claimed cause.

If the candidate fails, remove all three fused pipelines, runtime calls,
resource-barrier additions that have no remaining owner, scratch cleanup,
statistics/schema changes, and tests that exist only for the rejected path.
Restore the complete pre-task production path, record the negative result here,
and choose a new baton rather than keeping half of the fusion or broadening the
task.
