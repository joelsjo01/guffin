# Next Task: Profile the Rejected Seven-Wave Decode Before Redesigning Finding 3

## Implementation Outcome (2026-07-29)

The seven-wave candidate described below was implemented end to end, verified
on the real model, measured, and reverted because it did not pass the mandatory
performance gate. Do not reimplement it unchanged.

The candidate achieved its structural targets:

- 297 dispatches for a short-context 42-layer greedy decode token;
- zero explicit command-encoder barriers and one serial encoder;
- one fused entry, one direct QKV/cache operation, two dual bridges, and one
  fused gate/up/GELU operation per layer;
- exact real-model text parity and independent tiny-model full-logit, selected
  token, F16 cache, fused-entry, and dual-bridge parity;
- 147,460 fewer scratch bytes: the specified 144 KiB of raw Q/K/V and gate/up
  intermediates plus the now-unneeded four-byte decode token staging buffer; and
- product first-turn dispatches reduced from 21,089 to 10,305.

It did not produce a material end-to-end win. The schema-4 baseline's five
complete-conversation samples had a 21,094,457,876 ns median and 32,823,876 ns
MAD. The first complete candidate had a 23,169,434,249 ns median and severe
thermal drift because its initial generic Q6_K V subregion was compute-heavy.
Replacing that subregion with the retained packed Q6_K arithmetic removed the
regression, but a cooled screening conversation was 21,038,642,333 ns: only
about 0.27% faster than the baseline median, far below the required 2% gate and
not enough evidence to justify five retained paired runs. A single-traversal
four-accumulator gate/up variant was slower at 21,321,882,707 ns, consistent
with register-pressure/occupancy loss, and was removed before the full revert.

The exact local GGUF also corrected a research premise: `attn_v.weight` is
Q6_K (`DType` 14), not Q4_K. The direct candidate handled it natively without
repacking, but any future design must begin from this verified dtype.

The production execution and decode scratch are therefore back at the verified
starting state below. Schema 4, raw `decode_duration_ns`, omission of serialized
barrier state, and the four zero-valued reserved fused-family counters remain so
future candidates can use a compatible baseline. Before selecting another
Finding 3 implementation, collect a Metal trace or isolated family timings that
identify a design with a plausible greater-than-2% whole-conversation ceiling.

## Rejected Objective

Fix Finding 3 from `docs/LLAMA_CPP_PIPELINE_GAP_ANALYSIS.md` by replacing the
current fine-grained concurrent decode schedule with a Gemma-2-specific fused
serial schedule. The target is seven dependent GPU dispatches per transformer
layer, zero explicit command-encoder memory barriers, no target-path legacy
Q/K/V, RoPE/cache-store, gate/up/GELU, or standalone bridge-normalization
dispatches, and no dead decode scratch retained for those operations.

This baton is implementation work, but the design below is the result of the
research-only task completed on 2026-07-29. Do not code a different design
without first updating this document with measured evidence that invalidates a
stated premise.

Retain the replacement only if it preserves full logits, selected tokens,
F16 ring-cache contents, model residency, command lifetime, and frontend
behavior and improves both decode time and complete product-conversation time
beyond baseline noise.

## Rejected Candidate Design

Use one ordinary serial `MTLComputeCommandEncoder` and specialize the fixed
Gemma 2 one-token graph into these operations:

1. fused embedding lookup, embedding scaling, residual write, and first
   attention RMSNorm;
2. fused Q4_K Q/K/V projection that writes rotated F32 Q and rotated/narrowed
   F16 K plus narrowed F16 V directly to their final destinations;
3. the retained fused decode-attention operation;
4. the retained attention-output projection;
5. a dual-normalization residual bridge that performs post-attention RMSNorm,
   residual add, and feed-forward RMSNorm in one threadgroup;
6. fused Q4_K gate/up projection and GELU-tanh multiplication that writes only
   the gated activation;
7. the retained Q6_K down projection;
8. the same dual-normalization residual bridge, now performing post-FFN
   RMSNorm, residual add, and either the next layer's attention RMSNorm or the
   final output RMSNorm.

Items 2-8 are the seven per-layer waves. Item 1 runs once before layer zero.
The output projection and token selector remain after the layers.

This is the highest-ceiling legal fusion for the current Metal execution
model. It removes dispatches and intermediate traffic where a single
threadgroup or independent output rows make fusion safe. It stops at the two
large projection boundaries whose consumers require results produced by many
threadgroups; Metal provides no ordinary-dispatch grid-wide barrier that would
make fusion across those boundaries safe.

Do not implement the earlier three-family design consisting of fused Q/K/V,
then a separate Q/K/V postprocess, plus resource-scoped barriers. Direct final
Q/K/V writes remove one more dispatch and dependency boundary, while the
serial schedule makes all explicit resource barriers unnecessary.

## Verified Starting State

### Live short-context schedule

`run_gemma2_metal_resident_decode_step` currently creates one command buffer,
one `MTLDispatchTypeConcurrent` compute encoder, one commit, one synchronous
wait, zero blit encoders, and zero transient backend-buffer allocations.

Each of 42 layers encodes 15 dispatches:

```text
attention RMSNorm
Q projection | K projection | V projection
Q RoPE | K RoPE + K/V cache store
decode attention
attention output projection
post-attention RMSNorm + residual add
feed-forward RMSNorm
gate projection | up projection
GELU-tanh(gate) * up
down projection
post-FFN RMSNorm + residual add
```

Vertical bars mark independent commands that the concurrent encoder may run
together. Every other boundary is a true producer/consumer dependency.

For greedy top-1 selection at an attention length through 256:

```text
dispatches = 1 embedding + 42 * 15 layer dispatches
           + 1 final RMSNorm + 1 output projection + 1 top-1 selector
           = 634

explicit barriers = 1 after embedding + 42 * 11 layer barriers
                  + 1 after final RMSNorm + 1 after output projection
                  = 465
```

The `635`/`466` assertions in the current two-layer test and the corresponding
headline in the gap analysis are stale by one. Final softcap is now performed
inside compact selection rather than by a standalone dispatch.

Above 256 attention tokens, each affected layer's retained partitioned
attention adds one reduction dispatch and currently one internal explicit
barrier. Non-greedy compact selection adds a tile dispatch, binary merge
dispatches, and a finalize dispatch; its current internal barriers are also
unnecessary in a serial encoder.

### Exact target facts

The selected `models/gemma-2-9b-Q4.gguf` has 42 layers and repeats the same
decode geometry:

- hidden width 3,584, exactly 14 Q4_K blocks;
- 16 query heads and 8 K/V heads of dimension 256;
- Q width 4,096 and K/V widths 2,048;
- feed-forward width 14,336;
- Q, K, attention-output, gate, and up weights are Q4_K;
- V and down weights are Q6_K; and
- Q/K/V share one normalized activation, as do gate/up.

The current packed Q4_K decode kernel maps one SIMD group to two adjacent
output rows and shares activation registers across those two rows. The direct
Q/K path can instead map those two accumulators to one NeoX RoPE pair within a
head. It preserves two-row activation reuse while making both values available
before the SIMD group writes either final rotated result. V retains adjacent
row pairing.

The current decode scratch unnecessarily keeps F32 `raw_query`, `raw_key`,
`raw_value`, `feed_forward_gate`, and `feed_forward_up`. The target direct/fused
families make all five unreachable. At the exact model shape this removes
32 KiB of raw Q/K/V scratch and 112 KiB of gate/up scratch, in addition to
their repeated global-memory writes and reads.

### Existing correctness evidence

`tests/engine/gemma2_metal_plan_tests.cpp` already contains an independent
nonzero two-layer CPU-reference decode. It checks full logits, selected output,
F16 cache contents, sliding/global cache behavior, command counts, and both
selection/no-selection paths. It is the integration oracle to update, not
delete.

The focused suites also already cover generic packed Q4_K/Q6_K projection,
NeoX RoPE, F16 KV storage, fused attention at partition boundaries, RMSNorm-add,
compact selection, unavailable Metal, plan byte accounting, and real generation
append/replay. Preserve the numerical coverage that remains semantically live.

## Research Findings

### Decode is bandwidth-bound, but fixed work is still removable

Autoregressive decode streams the model's large projection weights once per
token. That makes model traffic the first-order lower bound and explains why
fusion cannot produce an arbitrary speedup for a 9B model. It does not make
634 dispatches and 465 synchronization commands free. Cheap normalization,
RoPE, cache stores, activations, and selector stages are short enough that
dispatch/scheduling cost and avoidable activation traffic are a meaningful
secondary term.

BaseRT is the closest published Apple-Silicon evidence. Its native Metal
runtime reports that decode gains are largest where fixed per-token cost is a
larger fraction of latency, and describes direct dequantization, zero-allocation
decode, chip-specific fusion, and custom dispatch as its core techniques. Its
public conversion format also canonicalizes Q/K/V and gate/up as fused
topologies. The runtime shader source is not public, so use this as directional
and measured system evidence, not as an implementation oracle.

References:

- BaseRT paper: <https://arxiv.org/html/2607.00501>
- BaseRT format topology:
  <https://github.com/basecompute/baseRT/blob/main/base-convert/FORMAT.md>

### Serial Metal execution is the correct final schedule

Apple documents that an ordinary `computeCommandEncoder` dispatches commands
serially. A concurrent compute pass removes that ordering and makes the
application responsible for synchronizing conflicting accesses. Serial here
means dispatch-to-dispatch ordering; it does not serialize the threadgroups or
SIMD groups inside a dispatch.

After direct Q/K/V fusion and gate/up/GELU fusion, every remaining dispatch
consumes the preceding dispatch's output. There is no dispatch-level overlap
left to preserve. Keeping a concurrent encoder would therefore retain an
explicit barrier at every boundary without exposing useful independent work.
Resource-scoped barriers would narrow a wait that has no unrelated work to
allow through.

Use the default serial encoder, keep one encoder and one command buffer, and
delete the explicit barrier abstraction. This is simpler and has a stronger
correctness contract than rebuilding serial order manually.

Primary references:

- Apple resource synchronization:
  <https://developer.apple.com/documentation/metal/resource-synchronization>
- Apple default serial compute encoder:
  <https://developer.apple.com/documentation/metal/mtlcommandbuffer/makecomputecommandencoder%28%29>
- Apple dispatch types:
  <https://developer.apple.com/documentation/metal/mtldispatchtype>
- Apple compute-dispatch profiling:
  <https://developer.apple.com/documentation/xcode/analyzing-draw-command-and-compute-dispatch-performance-with-pipeline-statistics>

### llama.cpp solves a more general problem

The pinned comparison commit `d48a56eff` and current upstream llama.cpp both
keep a concurrent Metal encoder, track source/destination byte ranges for graph
nodes, and issue `memoryBarrierWithScope:MTLBarrierScopeBuffers` when a new node
conflicts with the active wave. They do not use resource-scoped barriers as a
special performance mechanism.

That design is appropriate for a general graph executor whose independent
nodes and aliases vary. This engine owns a fixed Gemma decode schedule and
already knows every dependency. Adding a generic range tracker would duplicate
the runtime schedule, allocate or maintain hot-path metadata, and rediscover
that the post-fusion graph is a chain. Do not copy that abstraction.

Primary sources:

- pinned scheduler:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal-ops.cpp>
- pinned Metal encoder wrapper:
  <https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal-device.m>
- current scheduler:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-ops.cpp>
- current encoder wrapper:
  <https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-metal/ggml-metal-device.m>

### Fusion should follow communication boundaries

Inference-engine and research systems consistently fuse elementwise work,
normalization, residuals, activation functions, and compatible projection
epilogues to remove launches and off-chip intermediates. DeepSpeed's stated
fusion rule is particularly relevant: preserve access patterns and stop where
different thread blocks would need to communicate through main memory.

The target schedule follows that rule:

- a Q or K SIMD group owns both values of one RoPE pair, so projection and
  rotation can be fused without cross-threadgroup communication;
- a V output row can narrow directly into its final cache row;
- a gate/up SIMD worker can accumulate matching rows from both matrices during
  one activation traversal and apply GELU after local reduction;
- hidden-width normalization/residual bridges use one 256-thread threadgroup,
  so two reductions can communicate through threadgroup memory; and
- attention-output and down projections use many threadgroups, so fusion past
  either projection would require a grid-wide synchronization Metal does not
  provide inside an ordinary dispatch.

ClusterFusion demonstrates deeper QKV-attention-output fusion on hardware with
cluster-level collective primitives. That is evidence for the benefit of the
idea and simultaneously evidence that a communication primitive is required.
Do not emulate one with global atomics or a spin barrier: occupancy is not
guaranteed, so waiting threadgroups can prevent the producers they await from
ever becoming resident.

References:

- DeepSpeed Inference paper: <https://arxiv.org/abs/2207.00032>
- DeepSpeed kernel-fusion design:
  <https://www.deepspeed.ai/2021/03/15/inference-kernel-optimization.html>
- ClusterFusion: <https://arxiv.org/html/2508.18850>
- BaseRT kernel-fusion discussion:
  <https://arxiv.org/html/2607.00501#S3.SS3>

### Candidate decisions

| Candidate | Decision | Reason |
| --- | --- | --- |
| Seven fused waves in a serial encoder | Implement | Removes all independent branches first, then uses Metal's native ordering for the remaining chain. |
| Separate fused Q/K/V and Q/K/V postprocess | Reject | Materializes raw Q/K/V and preserves one dispatch and one dependency boundary that row ownership can remove. |
| Resource-scoped barriers in a concurrent encoder | Reject for final path | After branch fusion there is no unrelated dispatch to overlap across a narrow barrier. |
| Generic range/dependency tracker | Reject | Solves dynamic graph aliasing that the fixed schedule does not have. |
| Serialize the current 15-dispatch layer unchanged | Diagnostic only | It removes explicit barriers but also serializes currently independent Q/K/V and gate/up work. |
| Argument buffers | Reject for this slice | They may reduce CPU binding cost but do not remove GPU dispatches, dependencies, or intermediate traffic. |
| Indirect command buffers | Reject | The graph is CPU-known and fixed; ICB execution does not remove producer/consumer ordering or kernel work. |
| Persistent whole-layer kernel/global atomic barrier | Reject | Ordinary Metal dispatches lack a safe grid-wide barrier and cannot guarantee all waiting groups are resident. |
| Fuse normalization independently into every projection group | Reject | It would redundantly reduce the same 3,584-element vector thousands of times. |
| Fuse across attention-output or down projection | Reject | The next normalization needs results from many projection threadgroups. |
| Change quantization, precision, attention, or sampling | Out of scope | Those are separate numerical and performance decisions, not Finding 3. |

## Target Schedule and Counts

### One-time entry

`encode_gemma2_embedding_first_norm` reads the selected F16 embedding row,
applies the existing embedding scale, writes the F32 residual, reduces its
squared norm, applies layer zero's attention norm weight, and writes
`normalized_hidden` in one 256-thread threadgroup.

### Seven waves per layer

```text
1  direct Q/K/V projection + Q/K RoPE + F16 K/V cache store
2  fused decode attention (plus its reduction dispatch only when partitioned)
3  attention output projection
4  post-attention RMSNorm + residual add + FFN RMSNorm
5  gate projection + up projection + GELU-tanh multiply
6  down projection
7  post-FFN RMSNorm + residual add + next-attention/final-output RMSNorm
```

For greedy top-1 and attention lengths through 256:

```text
dispatches = 1 fused embedding/first norm + 42 * 7 layer dispatches
           + 1 output projection + 1 top-1 selector
           = 297

explicit command-encoder barriers = 0
command buffers = encoders = commits = waits = 1
```

This removes 337 dispatches (53.2%) and all 465 explicit barriers from the
short-context greedy path. Required producer/consumer ordering still exists;
it is expressed by the serial compute pass rather than 465 application-encoded
barrier commands.

For general selection, derive the exact count as:

```text
297 - 1 top1 + selector_dispatch_count
    + sum(max(attention_partition_count(layer) - 1, 0) != 0 ? 1 : 0)
```

The retained attention implementation currently adds exactly one reduction
dispatch whenever partition count exceeds one, not one dispatch per partition.
Tests and metrics must encode that fact directly rather than hard-code only a
short-context total.

## Kernel Design

### Fused embedding and first normalization

Add one Gemma-decode-only kernel with one threadgroup. It accepts token ID,
F16 embedding weights, layer-zero attention norm weight, F32 residual and
normalized destinations, embedding scale, hidden width, vocabulary size, and
epsilon.

Preserve the exact current order at each element:

```text
residual[i] = float(embedding[token, i]) * embedding_scale
normalized[i] = residual[i] * rsqrt(mean(residual^2) + epsilon) * norm_weight[i]
```

Use the same SIMD and threadgroup reduction order as `rms_norm_f32_kernel`
where possible. Validate token and buffer bounds on the host before encoding.
Do not retain a standalone embedding dispatch or first RMSNorm call in decode.

### Direct Q/K/V projection, rotation, and cache store

Add one native K-quant production kernel specialized for 14 blocks. Bind the normalized
activation, three independent native GGUF weight buffers, Q destination, K/V
cache buffers, cache position/mask, three output widths, head geometry, RoPE
base, and block strides. Do not concatenate, repack, or duplicate weights.

Partition one grid into Q, K, and V regions with uniform threadgroup branches:

- Q and K: each SIMD group accumulates the two rows forming one NeoX pair in
  one head, reduces both F32 dots, computes sine/cosine once, and writes both
  rotated values; Q writes F32 scratch and K writes F16 cache;
- V: retain adjacent two-row packed Q6_K accumulation and write F16 directly to the
  current value-cache row; and
- each region computes its own bounds and tails; no region may borrow another
  region's output count.

The local target's `attn_v.weight` was verified during implementation as Q6_K
(`DType` 14), correcting the research-only draft's Q4_K claim. The exact target
has even 256-wide heads, but focused tests need an explicit
validated contract for even RoPE width and independent region tails. Use the
retained packed Q4_K decode arithmetic as a private helper only where doing so
does not pessimize the production specialization. Avoid a generic fusion
framework.

There must be no raw Q/K/V global output, no postprocess dispatch, and no
inter-dispatch synchronization between projection and rotation.

### Dual-normalization residual bridge

Add one one-token, one-threadgroup kernel used twice per layer. It accepts:

- a projected branch input and its post-branch norm weight;
- the incoming residual;
- the residual output;
- the following pre-operation norm weight and normalized output; and
- hidden width plus both epsilons.

It computes:

```text
branch_norm = rms_norm(projected, post_weight, epsilon)
new_residual = incoming_residual + branch_norm
next_normalized = rms_norm(new_residual, next_weight, epsilon)
```

Preserve F32 accumulation and vectorized float4 access for hidden width 3,584.
The first reduction finds the projected input RMS. After forming each residual
element, perform a second SIMD/threadgroup reduction for its RMS and write both
the residual and normalized output. A threadgroup barrier is legal because the
entire hidden vector belongs to one threadgroup; this is not a command-encoder
barrier and must not be counted as one.

For the attention bridge, `next_weight` is the same layer's FFN norm weight.
For the FFN bridge, it is the next layer's attention norm weight, or the model's
output norm weight for layer 41. The runtime must select the already-resident
weight; no per-token metadata buffer or allocation is allowed.

Screen two implementations if register pressure requires it:

1. retain each thread's residual float4 values through the second reduction;
2. write residual, use a device-memory threadgroup barrier, then reread it for
   the final normalized write.

Choose by exact-shape isolated timing and full parity. Do not keep both
production variants after the decision.

### Fused gate/up/GELU-tanh projection

Add one Q4_K kernel specialized for 14 blocks. Matching gate and up output rows
must share one activation traversal. Each SIMD group accumulates two adjacent
rows from both matrices, reduces four F32 sums, then writes:

```text
gated[r] = gelu_tanh(gate_dot[r]) * up_dot[r]
```

Preserve the current precise Gemma GELU-tanh expression. Start with the same
four final rows per two-SIMD-group threadgroup as the retained Q4_K kernel; this
keeps threadgroup count at 3,584 rather than trading dispatch reduction for a
larger grid. Measure register occupancy. If four accumulators materially reduce
throughput, screen a two-final-row geometry, but remove the losing pipeline.

Never write standalone gate or up outputs.

## Code Placement

### Metal shader and pipelines

In `src/tensor/backends/metal_backend.mm`, place the four new shader families
beside their retained numerical primitives:

- embedding + first norm;
- direct Q/K/V projection/postprocess;
- dual RMSNorm-add-next-norm; and
- gate/up/GELU projection.

The dual bridge is one pipeline used twice, so the final target adds four
pipeline states, not five. Construct them eagerly in
`src/tensor/backends/metal/metal_context.hpp` and `.mm`. Compile function
constants at model-load/process initialization, never in the token loop.

### Command context

In `src/tensor/backends/metal/metal_command_context.hpp` and `.mm`:

- remove the `concurrent_compute` constructor argument and state;
- always create the ordinary serial compute encoder;
- remove `encode_memory_barrier` and `memory_barrier_count`;
- remove explicit barriers inside partitioned attention and compact selection;
- add narrowly typed encoders for the four new pipeline families;
- validate dtype, rank, exact dimensions, buffer capacity, cache position,
  power-of-two row mask, Metal ownership, and nonaliasing destinations; and
- add family counters for fused embedding/norm, direct QKV/cache, dual norm
  bridges, and fused gate/up/GELU.

Encoding must retain one command buffer/encoder/commit/wait and add no blit,
allocation, CPU-built argument buffer, polling, or readback.

### Runtime

In `src/engine/gemma2_runtime.cpp`:

- replace embedding plus the layer-zero attention RMSNorm with the fused entry;
- replace Q/K/V matmuls, Q RoPE, and KV store with direct QKV/cache encoding;
- replace post-attention RMSNorm-add plus FFN RMSNorm with one dual bridge;
- replace gate/up matmuls plus GELU multiply with fused gated projection;
- replace post-FFN RMSNorm-add plus the next attention RMSNorm with the second
  dual bridge;
- pass output norm as the second bridge's next norm on the final layer;
- remove the standalone final RMSNorm; and
- delete every explicit barrier call.

Do not route prefill through these one-token kernels. Prefill already uses a
serial command context and its row-batched kernels have different geometry.

### Decode scratch and dead state

After isolated parity passes, remove these fields from
`Gemma2MetalDecodeScratch` and its allocation/byte accounting:

- `raw_query`;
- `raw_key`;
- `raw_value`;
- `feed_forward_gate`; and
- `feed_forward_up`.

Keep `query`, `feed_forward_gated`, both residual buffers, normalized buffers,
projection outputs, attention partials, logits, and selection workspace.

Do not remove similarly named prefill scratch. Prefill still owns raw batched
Q/K and standalone gate/up intermediates. Do not remove generic Q4_K/Q6_K/F16,
RoPE, KV-store, RMSNorm, RMSNorm-add, or GELU pipelines while a real prefill,
tensor-op, or test path still uses them. “Remove legacy” means delete the old
decode execution and state completely, not break shared live functionality.

### Statistics and benchmark schema

Remove `memory_barrier_count` through Metal stats, runtime execution stats,
human output, and JSONL. Add the four fused-family counters and propagate them
through prefill/decode aggregation. Bump conversation benchmark schema 3 to 4;
keep scenario version 1.

For the preserved schema-4 baseline binary, omit the legacy barrier field from
serialized output and report zero for the new fused-family counters while the
old execution still keeps its private barrier counter. Remove that private
counter when the old execution is deleted. Baseline and candidate JSON then
have the same schema without forcing dead barrier state into the final engine.

Schema 4 must retain raw `gpu_duration_ns`, compute dispatches, command buffers,
encoders, commits, waits, allocation counts, projection-family counts,
attention dispatch/partition counts, selector dispatches, compact readback, and
memory facts. It must add raw per-turn `decode_duration_ns` if that value is not
already serialized, so a decode optimization is not judged from rounded
tokens/second or TTFT.

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
- focused Metal, plan/runtime, benchmark, CLI, and script tests

## Replacement Test Plan

### Independent kernel tests

Use composed CPU/reference operations as oracles. Never use the new fused
kernel to generate its own expected output.

Fused embedding/first norm:

- nonzero and signed F16 embedding values;
- embedding scale other than one;
- zero and last valid token IDs;
- exact hidden width 3,584 and a small deterministic width;
- full residual and normalized output parity; and
- invalid token, dtype, rank, hidden width, and capacity failures.

Direct Q/K/V/cache:

- independently nonzero Q4_K Q, K, and V blocks;
- different Q/K/V output counts and region boundaries;
- NeoX first-half/second-half pairs in multiple heads;
- nonzero position and frequency base;
- exact F32 query and exact F16 key/value cache parity;
- cache row zero, last row, and ring-mask wrap;
- generic focused block count if needed plus the exact 14-block specialization;
- output tails around one SIMD group and one threadgroup; and
- invalid weight dtype/layout, odd RoPE width, head geometry, mask, position,
  capacity, aliasing, and non-Metal buffers.

Dual residual bridge:

- independently nonzero projected input, residual, and two norm weights;
- signed and large/small finite values;
- both full intermediate residual and next-normalized output parity;
- exact 3,584 width plus vector boundary widths;
- tolerance derived from two F32 reductions, not loosened after failure; and
- invalid width, epsilon, dtype, capacity, alias, and backend errors.

Gate/up/GELU projection:

- independent nonzero Q4_K gate/up weights;
- positive, negative, near-zero, and GELU saturation-region dots;
- exact stable GELU-tanh expression;
- row boundaries and tails;
- exact 14-block specialization; and
- proof that only the gated destination is required and written.

Every public new Metal operation needs unavailable-device behavior. Where the
operation is command-context-internal, retain the existing plan-creation
unavailable-Metal test as the public boundary and do not invent an unreachable
mock device path.

### Runtime integration

Update the existing two-layer CPU-reference resident decode rather than
replacing it with a weaker token-only test. It must require:

- full-logit parity for selected and no-selection execution;
- exact selected candidate IDs and expected logit tolerance;
- exact F16 K/V ring contents for every live row;
- sliding and global attention layers;
- partitioned attention behavior;
- one command buffer, one serial compute encoder, one commit, one wait, zero
  blit encoders, and zero transient allocations;
- one fused entry, one direct QKV/cache dispatch per layer, two dual bridges per
  layer, and one fused gated projection per layer;
- no target-path standalone Q4_K count for Q/K/V/gate/up, while retaining the
  expected Q4_K attention-output and Q6_K down counts; and
- dispatch totals derived from layer count, attention reductions, and selector
  dispatches.

Add an exact target-shape test when the local model is available only as an
explicit integration/benchmark command, not as a CI dependency on model
weights.

### Tests to delete or replace

Delete or rewrite all tests whose owned behavior disappears:

- stale `635`/`466` formulas;
- fixed 25-barrier tiny-decode assertions;
- any expectation for concurrent decode encoding;
- standalone barrier-count tests and the `memory_barrier_count` field;
- tests that require separate decode Q/K/V projections;
- tests that require decode Q RoPE plus separate KV store;
- tests that require standalone decode gate/up/GELU;
- decode scratch descriptor, byte-count, or identity checks for the five
  removed buffers; and
- schema-3 fixtures and serialized `"barriers"` fields.

Replace them with the fused numerical, ownership, serial-order, family-count,
schema-4, and exact-dispatch tests above. A count assertion alone is not a
replacement for a numerical test.

Preserve:

- generic Q4_K/Q6_K/F16 projection parity used by live paths;
- prefill Q/K/V, RoPE/cache, gate/up/GELU, and scratch tests;
- standalone tensor-op RMSNorm, RMSNorm-add, RoPE, KV-store, and GELU tests;
- decode attention numerical and partition-boundary tests, changing only their
  obsolete explicit-barrier expectations;
- compact selector numerical tests, changing only serial ordering/counts;
- CPU reference, generation append/replay, stale-cache, sampling, chat/session,
  and CLI success/failure tests; and
- unavailable Metal coverage.

## Implementation and Measurement Order

Keep an external release baseline binary before changing execution. Temporary
candidate binaries and results stay outside the repository.

1. Correct the reporting schema and preserve a schema-4 baseline binary whose
   GPU execution is still the old schedule.
2. Implement and isolate-test direct QKV/cache and fused gate/up/GELU while the
   existing schedule remains available only in the baseline binary.
3. Integrate those two families, then compare final serial encoding against a
   temporary concurrent/resource-barrier candidate. The shipped path is serial
   unless the concurrent candidate wins beyond noise; if it does, update this
   design with trace evidence before retaining it.
4. Implement the fused entry and dual bridge, test each independently, and
   integrate the seven-wave schedule.
5. Delete old decode calls, buffers, explicit barrier API/state, losing pipeline
   variants, and obsolete tests. Do not retain a runtime flag or fallback.
6. Run focused tests, `bash scripts/ci.sh`, and `git diff --check`.
7. Run the real product benchmark and retain or revert the complete candidate
   according to the gates below.

This order permits attribution during development without leaving two
production schedules in the final tree.

## Performance Verification

### Required measurements

Use release binaries and `models/gemma-2-9b-Q4.gguf`. Preserve model, prompt,
sampler, seed, stop conditions, cache history, frontend, output mode, warmup,
and repetition count between baseline and candidate.

Measure both:

1. controlled short-context greedy generation, which must show 297 dispatches
   per selected token and zero explicit barriers; and
2. product conversation scenario 1 with its real seeded sampler and four-turn
   cache reuse.

Use at least one warmup and five measured runs per binary. Interleave baseline
and candidate runs when practical to reduce thermal drift. Record median, MAD,
paired wins, CPU wall decode duration, command-buffer GPU duration, tokens/s,
complete turn/conversation duration, TTFT, dispatch/family counts, attention
partitions, selector dispatches, readback, and memory audit.

If Xcode Metal capture is available, capture one baseline and candidate token.
Confirm:

- no GPU gaps were introduced;
- direct QKV regions occupy the GPU as expected;
- gate/up fusion did not lose occupancy through register pressure;
- dual bridges are one dispatch each; and
- the command stream contains no explicit buffer barrier commands.

Trace evidence explains the mechanism but cannot override end-to-end results.

### Retention gates

All must pass:

- `bash scripts/ci.sh` and all new deterministic parity tests;
- exact prompt/suffix/selected/visible token IDs, final text, stops, cache
  facts, plan identity, and resident-memory facts in paired product records;
- one command buffer, encoder, commit, and wait; zero blits and transient
  allocations;
- the expected 297 short greedy dispatches and fused-family counts;
- at least four of five paired runs improve aggregate decode duration;
- median aggregate decode improvement exceeds
  `max(2%, 2 * baseline_MAD / baseline_median)`;
- median complete-conversation improvement exceeds its separately computed
  threshold using the same formula; and
- no material prefill, TTFT, memory, or frontend regression.

The 53.2% command-count reduction is not a 53.2% latency promise. Large weight
streams remain. Accept only measured whole-path improvement outside noise.

If the full candidate fails, use the staged measurements to identify the
losing family, remove it completely, and update this document before choosing
a smaller retained design. Never keep dead scratch, a disabled pipeline, a
legacy fallback, or tests that exist only for a rejected candidate.

## Cleanup Audit

Before stopping, use `rg` and code inspection to prove target decode contains
none of:

- `MTLDispatchTypeConcurrent` or a concurrent constructor switch;
- `encode_memory_barrier` or `memory_barrier_count`;
- standalone decode embedding or first/final RMSNorm calls;
- three standalone Q/K/V matmul calls;
- standalone decode Q RoPE or KV-cache-store calls;
- standalone decode gate/up matmul or GELU-multiply calls;
- `raw_query`, `raw_key`, `raw_value`, `feed_forward_gate`, or
  `feed_forward_up` in decode scratch;
- schema-3 benchmark fixtures or a serialized `"barriers"` field;
- stale 635/466 formulas; or
- rejected pipeline states, counters, flags, and temporary diagnostics.

Also prove prefill still contains and tests every generic operation and scratch
buffer it genuinely owns. Cleanup must follow ownership, not name matching.

## Constraints

- Preserve engine-owned, offline, Metal-first, model-resident inference.
- Do not call or link llama.cpp, BaseRT, MLX, MPSGraph, or another inference
  runtime from production.
- Preserve direct native GGUF Q4_K/Q6_K operation; do not repack, concatenate,
  duplicate, or dequantize resident weights.
- Preserve F32 accumulation, exact Gemma RMSNorm/GELU/RoPE/softcap semantics,
  F16 KV storage, ring-cache addressing, attention math, and output-head math.
- Preserve generation order, sampling, stops, callbacks, chat templates, and
  compact readback.
- Keep one command buffer, one serial encoder, one commit, one wait, zero blits,
  zero token-hot-path allocations, and no CPU/GPU mid-token synchronization.
- Do not add a generic graph/fusion scheduler, hazard tracker, argument buffer,
  indirect command buffer, global atomic grid barrier, per-token descriptor
  buffer, or fallback decode path.
- Do not change prefill execution in this task except shared command-context
  cleanup required by removing redundant serial barriers.
- Preserve unrelated user changes and keep generated models, binaries, traces,
  and benchmark records out of version control.

## Stop Condition

Stop after the seven-wave serial decode is independently tested, fully
integrated, stripped of the old decode schedule/state/tests, and measured on
the real Gemma 2 9B Q4 product path.

Report changed files, exact tests and CI, old/new dispatch and scratch counts,
greedy and sampled benchmark distributions, Metal trace evidence when
available, and known gaps. Do not proceed to output-head fusion, multi-token
GPU loops, attention changes, quantization changes, Metal 4 migration, or a
new performance finding in the same implementation task.
