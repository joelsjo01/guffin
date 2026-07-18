# Next Task: Consolidate Resident Decode Encoders

## Verified Current State

Gemma 2 prefill now uses one production Metal executor:

- `Gemma2MetalModelPlan` owns fixed 16-token prefill scratch.
- Prompts are split into consecutive 16-token chunks.
- Each chunk uses one command buffer and resident activation buffers.
- Chunks rely on command-queue ordering and the complete prefill waits once.
- Greedy, sampled, chat append, context rebuild, and benchmarks share the path.
- Structural tests cover exact-size and cross-boundary prompts, nonzero cache
  positions, alternating sliding/global layers, logits, selected token, KV
  parity, invalid inputs, and zero transient Metal-buffer allocations.
- The staged prefill shell, intermediate wrappers, helper chain, validation
  code, and implementation-coupled tests were deleted.

For a two-layer 17-token fixture, the verified counts are two command buffers,
two commits, one synchronous wait, four layer executions, four KV writes, and
zero transient buffer allocations.

The first comparable release sample on the 16-token real Gemma prompt moved
warm greedy prefill from `1.90` to `1.95` prompt tokens/s. Two repetitions are
not enough to distinguish that result from thermal variance. The benchmark
runner executes four cases and repeats nested cached turns, so its requested
10-warmup/50-repetition setting would take tens of minutes on the current
runtime. Treat the structural win as proven and the throughput magnitude as
unresolved.

## Objective

Reduce resident decode from roughly 803 compute encoders and 42 blit encoders
per 42-layer token to a small intentional number of encoder phases without
adding command buffers, waits, allocations, or numerical changes.

## Required Design

1. Keep one command buffer and one final synchronous wait per decode token.
2. Keep a compute encoder open across compatible consecutive operations.
3. Insert explicit Metal memory barriers for scratch dependencies.
4. Replace per-layer blit encoder KV writes with a compute copy kernel, or batch
   them into a small number of deliberate blit phases.
5. Preserve resident weights, KV, decode scratch, top-logit selection, and all
   current Gemma semantics.
6. Expose structural statistics so tests assert the new compute/blit encoder
   counts without making per-kernel timing a production API.

Do not change prefill, attention math, Q4_K/Q6_K arithmetic, KV dtype/layout,
sampling, tokenizer behavior, or chat orchestration in this task.

## Correctness Gates

- Resident decode logits and selected-token parity against the diagnostic
  decode path.
- KV append parity for every layer.
- Sliding-window and global-attention layer coverage.
- One command buffer, one commit, one wait, and zero transient allocations.
- Unavailable Metal and invalid decode input behavior remain intact.
- All chat, context rebuild, sampling, tokenizer, and CLI tests remain intact.

## Performance Gate

Measure the exact release `local-gguf-generate --bench` path before and after.
Record decode encoder counts, GPU duration, runtime decode duration, and
generation tokens/s. Keep the change only if the real path improves beyond
run-to-run noise without prefill regression.

## Verification

1. Run `bash scripts/ci.sh`.
2. Run `git diff --check`.
3. Build the release generator.
4. Run comparable release benchmarks.
5. Confirm no alternate decode executor or rejected instrumentation remains.

## Stop Condition

Stop when decode uses one command buffer, one final wait, zero transient
allocations, and a small documented encoder count; parity and CI pass; and the
release benchmark supports the end-to-end result. Do not begin fused attention
without approval.
