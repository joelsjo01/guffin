# Next Task

## Current state

Stage 4 is complete and retained. The production Gemma 2 9B Q4 chat path now
uses exact, session-local retrieval speculation with a width-two Metal verifier.
The final thermally isolated candidate/baseline comparison improved aggregate
generation time by 2.77%, won both turns, matched every selected-token hash and
cache position, and did not regress TTFT or prompt throughput.

The next baton is **Stage 5 proposer feasibility and data design**. Freeze the
retained verifier/controller protocol. Before adding a learned or self-draft
runtime, define the smallest engine-owned proposer that fits the recovered
memory, latency, and thermal budget, and define how target-aligned proposal data
will be collected without adding work to ordinary production generation.

## Retained Stage 4 implementation

### Session-local proposer

`engine/retrieval_proposer.*` owns a bounded retrieval proposer over the exact
live prompt, conversation, and selected output token IDs:

- suffix orders 2 through 16 are searched longest first;
- rolling prefix hashes reject almost every non-match cheaply, while exact token
  comparison makes collisions correctness-neutral;
- competing continuations use support count, total observations, empirical
  confidence, and stable lower-token-ID ties;
- one-off long matches fall through to shorter supported matches;
- history, hashes, powers, and continuation scratch are reserved once, so
  propose/observe does not allocate;
- state is rebuilt after regeneration or context trimming and never learns a
  rejected proposal or cross-session data.

The final gate requires support of at least two and exact integer confidence of
at least 2/3. Cooled telemetry showed 0 matches in 6 cycles below 2/3, 3 in 9
cycles from 2/3 to below 1, and 14 in 17 confidence-1 cycles. The middle bucket
is retained because it was 3/4 in the first turn and remains near the optimized
verifier's break-even point; only the demonstrated losing region was removed.

### Exact sampled controller

The engine samples before decoding. For selected target token `x0` and retrieved
continuation `d1`, it verifies `[x0, d1]`, samples target token `x1` from verifier
row zero, and accepts `d1` only when `x1 == d1`:

```text
match:    retain x0,x1; use verifier row-one candidates next
mismatch: retain x0; carry already sampled x1 into the next cycle
```

This is exact target-token-match acceptance for the engine's seeded sampling
sequence. A mismatch performs no recovery decode. Preview RNG/history is adopted
only for the retained target prefix. Stop tokens, a one-token remaining budget,
insufficient context, and an unprofitable/no proposal use ordinary one-row
decode. `--no-speculative` disables only this controller.

Sampler preview uses one persistent fixed-capacity copy. The hypothetical
row-one repetition request is derived with a stack-only
`selection_request_after_observing` operation. Speculative cycles therefore do
not allocate sampler-history vectors.

### Width-two Metal execution

The verifier uses the retained exact width-two causal schedule for embedding,
Q/K/V/O projections, RoPE positions, per-row attention bounds, FFN, final norm,
and both output rows. K-quant and Q8 output projections use width-two families.

Compact selection for both rows is batched in one tile/merge/finalize command
sequence with independent histories and result lanes. No full vocabulary logit
row crosses to the host. Greedy uses one batched top-1 dispatch. The selection
workspace is persistent and shared with one-row execution.

### Zero-copy K/V transaction

The optimized verifier backs up the two original resident cache rows per layer,
writes verified K/V directly into the resident ring, and leaves those rows
provisional after completion:

- accepting both rows is a host-side state transition with zero GPU command,
  zero K/V copy, and zero wait;
- accepting one row restores only the rejected second row;
- discarding restores both rows;
- a callback stop after delivery of the first matched token restores row two;
- wrapped full-window caches preserve the original rows required by row-zero
  attention and restore the exact rejected suffix.

This replaced the earlier tentative-copy/restore/commit sequence. It removed two
per-layer K/V copy passes from every verifier and removed the accepted-pair
commit command. The retained candidate used 590 generation command buffers for
598 selected tokens, exactly `baseline + misses - matches` (`598 + 9 - 17`).
Scratch is 107,667,996 bytes, 688,128 bytes below the first Stage 4 candidate and
2,546,952 bytes above the Stage 3 plan.

### Warm-up and observability

Startup warms 16-token prefill, one-row decode/selection, and a state-neutral
width-two verifier/batched selector. Steady generation compiles no pipeline and
allocates no Metal buffer.

Decode diagnostic schema 3 reports a deterministic 64-bit selected-token hash;
fixed proposal buckets over match length, support, confidence, cycles, and
matches; proposer/selection/GPU/wall time; speculative outcomes; command and
projection attribution; compact readback; residency; and cache positions.

The benchmark driver validates bucket totals and hash format. Its default
cooldown is four minutes. On this fanless M4, do not replace isolated runs with
five interleaved pairs: later repetitions measure heat soak and throttling rather
than the candidate.

## Final correctness evidence

`bash scripts/ci.sh` passes with 21 Python tests and 327 C++/Metal tests.
Deterministic coverage includes:

- proposer longest-match, support, stable-tie, rebuild/incremental, bounds, and
  2/3 gate behavior;
- stack-only hypothetical selection-request equality with an independently
  copied/observed sampler;
- independent two-row Metal candidate parity with distinct histories;
- exact verifier logits, softcapped candidates, and K/V against sequential
  execution;
- zero-dispatch two-row acceptance and exact one/zero-row suffix restoration at
  cache wrap;
- seeded speculative/baseline selected-token and hash equality;
- forced match, mismatch/carry, stop, exception, and callback rollback paths;
- final K/V byte equality after callback rollback;
- warm-up state neutrality, unavailable-Metal behavior, and CLI schema paths.

Run `bash scripts/ci.sh` after any implementation change.

## Final isolated retention comparison

The final release candidate and `--no-speculative` baseline ran in separate
fresh processes with at least four minutes of idle cooldown before each. Both
used the canonical two-turn seed-42 sampled conversation, natural stops, the
same model/binary, and the five-second inter-turn pause. No kernel probe or
benchmark repetition burst ran between them.

| Turn | Retained candidate | Baseline | Result |
| --- | ---: | ---: | ---: |
| First, 276 tokens | 16.3 tok/s, 16.965 s | 15.7 tok/s, 17.524 s | 3.19% less time |
| Follow-up, 322 tokens | 15.9 tok/s, 20.268 s | 15.5 tok/s, 20.769 s | 2.41% less time |
| Aggregate, 598 tokens | 37.233 s | 38.293 s | **2.77% less time** |

Correctness identity was direct:

| Turn | Candidate hash | Baseline hash | Final cache |
| --- | --- | --- | ---: |
| First | `49080569719b1cea` | `49080569719b1cea` | 292 |
| Follow-up | `369b09d9b01ff1e1` | `369b09d9b01ff1e1` | 633 |

Workload counts also matched: 16/19 newly evaluated prompt tokens, 276/322
selected tokens, and initial cache positions 0/292. Candidate TTFT was 164.626
ms and 535.191 ms versus 168.350 ms and 571.295 ms baseline. Prompt throughput
was 97.2/35.5 tok/s versus 95.0/33.3 tok/s. Thus neither secondary headline
metric regressed in this isolated pair.

Final proposal behavior was 26 cycles, 17 matches, 9 misses, and 553 gated/no-
proposal fallbacks. Proposer time was 3.392 ms total, about 0.0057 ms per selected
token. The candidate used 590 command buffers versus 598 baseline. Aggregate
decode GPU time was 36.744 s versus 37.411 s baseline.

## Research decision

[`REST`](https://arxiv.org/abs/2311.08252) supplies the applicable exact
target-token-match rule for a token-only retriever.
[`SuffixDecoding`](https://arxiv.org/abs/2411.04975) and
[`SAM Decoding`](https://aclanthology.org/2025.acl-long.595/) support current-
context retrieval, longest suffix evidence, and later O(1)-average suffix-
automaton lookup. SAM is not justified here because the bounded rolling-hash
proposer is only 0.0057 ms/token and is not the bottleneck.

Recent [`LogitSpec`](https://aclanthology.org/2026.findings-acl.1655/) and
[`RACER`](https://aclanthology.org/2026.findings-acl.998/) work combine
retrieval with next-next-logit cues;
[`DReSD`](https://aclanthology.org/2025.findings-acl.1017/) uses contextual dense
retrieval. These approaches may raise acceptance,
but they require additional proposal state, semantic computation, or wider
verification. They belong in Stage 5 comparisons, not as unmeasured complexity
inside the retained Stage 4 path. Approximate mismatch acceptance is excluded
because exact seeded target output is an invariant.

## Next baton: Stage 5 feasibility and target-aligned data

Do not modify the retained Stage 4 verifier/controller while comparing Stage 5
proposers. The next slice is analytical and diagnostic-first:

1. derive hard resident-memory and cycle-latency budgets from the current
   7,534,041,088-byte weights, 2,113,929,216-byte K/V cache, 107,667,996-byte
   scratch, peak RSS, and swap behavior;
2. define comparable engine-owned candidates: a tiny recurrent/feature drafter,
   a parallel masked-position drafter, and a lower-bit or heavily skipped
   self-draft;
3. define a bounded opt-in trace of target-aligned prefixes, proposals, target
   tokens, acceptance, and workload class; ordinary production generation must
   pay no trace overhead;
4. score every candidate by committed tokens divided by complete cycle latency,
   including proposal execution, verification, selection, rollback, memory,
   startup, and thermals;
5. include current retrieval alone and retrieval-plus-learned routing as
   baselines; add width four only if measured acceptance clears its full-cycle
   cost threshold;
6. stop before training or adding a production drafter until one candidate has
   a credible local memory/latency budget and a deterministic correctness plan.

The Stage 5 feasibility slice is complete when it leaves one recommended
candidate (or a documented decision to keep retrieval alone), a concrete data
schema, a memory/cycle budget, deterministic verification requirements, and a
clean implementation baton.

## Invariants

- Production inference remains engine-owned, offline, Metal-first, and
  model-resident.
- Sampling transforms and their order do not change.
- Exact seeded target-selected tokens remain the correctness oracle.
- Rejected drafts never alter retained K/V, sampler/proposer history, output, or
  subsequent target distributions.
- No full vocabulary logit row crosses to the host in production.
- Cache wrap, transaction lifetime, row histories, and logical token counts are
  explicit and tested.
- Ordinary one-row decode remains the only fallback and is directly measurable
  with `--no-speculative`.
- Stage 5 must not silently introduce an external inference runtime, model
  server, tensor framework, or remote dependency.
