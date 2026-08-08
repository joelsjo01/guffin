# Next Task

## Verified current state

Stages 2A through 2C remain retained for the exact Gemma 2 9B Q4 target. The
production model uses copied shared/default/tracked Metal weights, a separate
Q8_0 output head while preserving the tied F16 embedding, and load-time
Q6_K-to-Q4_K conversion for the 21 Q6_K FFN-down matrices. Scheduled decode
weight traffic is 5,699,040,256 bytes/token and logical resident weights are
7,534,041,088 bytes. The last cooled retained chats produced 15.96 and 15.82
tok/s; startup was about 22 seconds with 10.88 GB peak RSS and no swaps.

Stage 2D is complete as a rejected candidate. The attempted kernel preserved
the current GQA/ring/partition ABI, computed 32 exact softcapped scores into a
1 KiB threadgroup tile, reduced the block maximum once, rescaled the prior F32
value state once per block, and made one V pass. It was selected only above 256
tokens. CPU-reference parity passed across block/partition boundaries through
8,192 tokens and a wrapped full-cache window. No approximate math, new device
buffer, hot-path allocation, prefill change, or Stage 2A-C accounting change was
introduced.

The identical 42-layer release-Metal harness gave these two-run GPU intervals:

| Context | Retained kernel | Block-online, original partitions | Result |
| --- | ---: | ---: | ---: |
| 512 | 3.279 / 2.940 ms | 2.246 / 2.367 ms | clear mechanism win |
| 1K | 3.751 / 3.638 ms | 3.549 / 3.585 ms | about 3.5% median gain |
| 2K | 9.470 / 9.010 ms | 10.064 / 9.186 ms | about 4.2% median regression |
| 4K | 16.144 / 16.201 ms | 16.208 / 16.415 ms | about 0.9% median regression |
| 21 local 4K + 21 global 8K | 22.897 / 22.943 ms | 22.797 / 24.490 ms | about 3.2% median regression |

One bounded partition rescue kept the winning 512/1K geometry and halved
partitions at 2K/4K/8K. It improved 2K to 7.316 / 7.289 ms, but worsened 4K to
17.889 / 18.542 ms and the real 4K/8K mixture to 24.505 / 25.291 ms. The long
regime is dominated by compulsory KV traffic; extra score-tile synchronization
cost more than reduced rescaling saved. The candidate failed the declared
whole-token long-context gate and was deleted completely, including its
pipeline, selector, partition changes, temporary probe, and candidate-only test
expansion. There is no old/new CLI, environment switch, dead kernel, or
alternate production path.

The restored retained source passes `bash scripts/ci.sh`: 21 Python tests and
319 C++/Metal tests. A full chat benchmark was intentionally not run after the
focused production-kernel gate failed; it could not rescue a regression in the
only context regime Stage 2D was intended to improve. Stage 2C therefore remains
the production performance baseline.

The short-context conventional plateau is reached. Remaining one-row cleanup
ceilings are individually below the 2% gate: all norm/residual work plus GELU is
about 0.53 ms, RoPE plus KV stores about 0.15 ms, token selection about 0.285 ms,
and the measured host envelope about 0.55 ms. These operations may be fused or
rearranged later when they otherwise multiply inside a multirow verifier, but
Stage 2E does not justify another standalone one-token optimization pass.

## Next baton: research the Stage 3 width-two causal verifier

Do deep implementation research and write one focused width-two verifier plan
into this file. Do not implement it until the user approves that plan.

The research must trace the exact production path and answer one decision:
whether the existing multirow projection kernels and a new causal two-row
attention/KV transaction can execute a target verification pass below 1.35x the
retained one-token latency. Cover:

- sustained one-row versus two-row cost for every target projection class,
  including converted Q4_K FFN down, remaining Q4_K/Q6_K transformer weights,
  F16 projections, and the Q8_0 output head;
- which current small-batch or tiled kernels actually serve two rows, their
  reduction order and numerical relationship to the one-token kernels, and any
  shapes that still fall back to unsuitable generic work;
- the exact two-row transformer schedule: normalization/residual ordering,
  position-dependent RoPE, causal attention between the two rows, alternating
  4K sliding/global windows, and production logits for both verified positions;
- a transactional KV design that stages both proposed rows, exposes the first
  row to the second during verification, commits only the accepted prefix, and
  restores or ignores rejected speculative rows without corrupting the resident
  ring;
- independent row-by-row logits/KV parity against two canonical one-token
  steps, including full/partial acceptance, rejection at the first proposal,
  ring wrap, softcap, GQA, cache-full behavior, and Metal-unavailable behavior;
- exact command, barrier, scratch, output, and resident-memory changes, with no
  production CLI mode or second inference runtime; and
- one diagnostic that establishes the width-two cost mechanism, one real-path
  verification benchmark, and a direct proceed/stop gate before any retrieval
  or learned proposer is added.

The plan should be one coherent implementation pass, not a sequence of tiny
milestones. Prefer extending the existing model plan, schedule, resident runtime,
and Metal kernels over building a parallel verifier abstraction. A diagnostic
verifier is allowed to return both logits and acceptance-independent KV state;
it must not claim a generation speedup before proposal and exact speculative
sampling exist.

## Constraints and stop condition

- Production inference remains engine-owned, local, Metal-first,
  model-resident, and allocation-free in the steady hot path.
- Preserve the retained one-token path, Stage 2A-C byte/residency behavior,
  prefill, sampling distribution, and current chat semantics.
- Do not add a drafter, retrieval proposals, speculative acceptance/sampling,
  width four, tree attention, KV quantization, or Stage 2E cleanup in this task.
- The research must derive an end-to-end verifier-cycle bound, not extrapolate
  from one isolated fast matrix.
- Stop after replacing this baton with the researched implementation plan. Do
  not begin width-two implementation until the user approves it.
