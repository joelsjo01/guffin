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
320 C++/Metal tests. A full chat benchmark was intentionally not run after the
focused production-kernel gate failed; it could not rescue a regression in the
only context regime Stage 2D was intended to improve. Stage 2C therefore remains
the production performance baseline.

The short-context conventional plateau is reached. Remaining one-row cleanup
ceilings are individually below the 2% gate: all norm/residual work plus GELU is
about 0.53 ms, RoPE plus KV stores about 0.15 ms, token selection about 0.285 ms,
and the measured host envelope about 0.55 ms. These operations may be fused or
rearranged later when they otherwise multiply inside a multirow verifier, but
Stage 2E does not justify another standalone one-token optimization pass.

## Stage 3 retained state

Stage 3 is complete and retained. Its exact width-two verifier extends the
existing plan, schedule, runtime, and Metal command context rather than creating
a second inference path. It has two-row activation/logit scratch, per-layer
tentative F16 K/V, an explicit pending transaction, and zero/one/two
accepted-prefix resolution. Verification backs up the two destination ring rows,
stores both proposed rows in one dispatch, executes row-zero then causal row-one
attention, stages the resulting K/V, and restores the resident rows before
leaving each layer. Rejection is therefore a logical discard and cannot destroy
a live row at cache-full pressure.

The projection path includes dedicated packed width-two Q4_K, Q6_K, and Q8_0
kernels. The Q4_K/Q6_K forms retain the canonical one-row worker assignment and
accumulation order while immediately rescanning the same column for row two; the
second scan benefits from cache residency and proved faster than the generic
small-batch kernel on the complete verifier. The Q8_0 head loads each packed
weight once and updates both row accumulators. Fused RMSNorm-add now accepts and
correctly indexes an explicit row count, and decode attention accepts explicit
query/output rows.

The retained verifier adds 3,928,072 descriptor-accounted bytes on the exact target.
Steady verification and commit allocate no transient buffers. Width-two Q4_K,
Q6_K, and Q8_0 dispatch families have explicit counters. The focused suite covers
two-row quantized projection and RMSNorm-add parity, two-row logits against
sequential decode, resident-cache preservation before resolution, two-row
commit, one-row commit across a sliding-cache wrap, zero-row discard,
pending-transaction exclusion, and unavailable-Metal behavior. The current
focused suite reports 320 passing C++/Metal tests.

The initial development measurements established the mechanism: generic
small-batch projections took 97.441 ms versus 60.466 / 60.139 ms one-row
commands, while packed width-two projections reduced that to 81.704 ms versus
62.960 / 61.748 ms. The latter was 1.310x mean one-row latency and 0.655x two
sequential commands, with a 0.061 ms two-row commit. Those runs used an invalid
raw-logit/byte-equality correctness verdict and are performance evidence only.

Corrected thermally separated adjudication first found the fused RMSNorm-add
row-indexing defect. Once repaired, it produced exact printed parity: maximum
raw and post-softcap logit error 0.000, mean KL -0.000, top-40 overlap 1.000,
top-1 agreement, maximum committed-KV error 0.000, exact rollback, and an exact
sequential cache after commit. It took 81.455 ms versus 60.091 / 59.817 ms
one-row commands, or 1.359x.

The final bounded optimization combines the two per-layer KV stores while a
ring still has unused capacity, removing 42 dispatches and 42 barriers in the
measured short-context target without changing results. The cooled measurement
improved the verifier to 80.715 ms versus 60.178 / 59.339 ms, with a 0.062 ms
commit and exact quality/cache parity. This is 1.351x mean one-row latency and
0.675x two sequential rows. The literal 1.35x boundary was missed by about
0.041 ms, or 0.05%, which is below observed thermal and run variation. By the
user's explicit completion decision, this is accepted as practically meeting
the gate; further verifier micro-optimization is not a Stage 3 requirement.

A subsequent nonzero-weight cache-wrap audit proved that the second proposed
row aliases the oldest row still needed by row-zero attention once a sliding
window is full. Full-window layers therefore retain the causal two-store order;
global layers and sliding layers with unused capacity keep the combined fast
path. The deterministic integration test now exercises this nonzero wrapped
one-row commit case.

## Next baton: select the Stage 4 proposer and acceptance design

Ordinary generation and chat do not invoke the retained verifier yet; their
production loop still executes one resident decode per selected token. Stage 3
owns the verified engine primitive, resident scratch, and transactional KV
resolution. Stage 4 must supply the proposer and exact acceptance controller,
extract the required verifier-row distributions/candidates without a full-logit
host readback, aggregate speculative metrics, warm the width-two route, and wire
that controller into the existing generation/chat orchestration.

Research and choose the smallest engine-owned proposer that can supply two-token
drafts cheaply enough for the retained verifier to produce an end-to-end decode
gain. Specify exact speculative acceptance for greedy and sampled generation,
including rejection recovery and zero/one/two KV commit behavior. Measure draft
acceptance on the canonical local conversation before implementing a production
proposer. Preserve the existing sampler distribution and deterministic seeded
behavior; do not weaken acceptance correctness to inflate the rate.

Do not broaden immediately to width four or tree attention. First connect one
bounded width-two proposer through the retained transaction and compare actual
generation tokens/s, output equivalence, acceptance distribution, and proposer
overhead against ordinary decode under the repository's thermal-safe retention
protocol.
