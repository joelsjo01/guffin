# Decode Acceleration Master Plan

## Purpose

This document turns the completed three-phase research in
`docs/DECODE_RESEARCH_PROGRAM.md` into the durable implementation strategy for
faster Gemma 2 9B Q4 generation on the base M4. It is not an active task list;
`docs/NEXT_TASK.md` owns the complete currently approved implementation scope
and its evidence boundary.

The strategy is to extract the worthwhile gains from the current single-token
engine first, while building only the foundations that directly reduce risk for
the later step-change. Once the unchanged decoder reaches a measured plateau,
the program moves to multi-token verification and speculative decoding, then to
the higher-risk Phase 1 hardware and algorithm experiments.

## Ground Truth

Planning starts from these local measurements, not from isolated paper results:

- Current generation is approximately 13.9 tok/s, or about 72 ms per token.
- A token schedules 6,837,376,000 logical weight bytes: the previously recorded
  6,837,368,832 projection/norm bytes plus the 7,168-byte selected F16 embedding
  row actually read by decode. Sustained Q4_K, Q6_K, generic Q4_K, and F16
  projection probes converge near 98-103 GB/s.
- The 98 GB/s weight-only floor is approximately 69.8 ms. Even the M4's
  documented 120 GB/s peak gives an idealized 57.0 ms floor, so an unchanged
  one-token pass has at most about 1.22x theoretical headroom before attention
  and other work.
- FFN projections own 57.3% of weight bytes, the F16 output head owns 26.8%,
  and attention projections own 15.8%. At short context, attention itself is
  about 2.55 ms; at the real maximum-context mixture it rises to about 23.2 ms.
- CPU sampling is about 3.4 microseconds per token. The total wall-minus-GPU
  envelope is about 0.55 ms per token. Neither is a headline bottleneck.
- Mode-major sustained measurement corrected the earlier width estimates.
  Width-two Q4_K and Q6_K cost approximately 1.7x their one-row paths; width
  four costs approximately 3.2x and 3.5x. F16 output is effectively flat at
  width two and approximately 1.07x at width four. These projection results do
  not establish full causal-verifier economics.
- Warm-up currently materializes a 2.114 GB KV allocation by clearing it twice.
  This is avoidable memory pressure, not a demonstrated decode-throughput win.

The governing optimization metric is:

```text
committed-token latency = complete cycle latency / expected committed tokens
```

For ordinary decode, expected committed tokens is one. For speculation, the
complete cycle includes proposal generation, target verification, sampling,
KV commit or rollback, synchronization, and recovery after rejection. Kernel
bandwidth, acceptance rate, and verifier latency are explanatory metrics; none
is independently the product result.

## Non-Negotiable Success Contract

Every retained change must preserve the engine-owned, offline Metal production
path and comply with the repository benchmark contract. In particular:

1. Correctness is established with an independent reference or golden oracle,
   including shape tails, cache boundaries, reset/regeneration, context shift,
   unavailable-Metal behavior, and relevant sampling modes.
2. A performance candidate graduates only after the canonical real chat path
   shows at least 2% median improvement, four of five paired interleaved wins,
   workload and correctness equivalence, and no greater-than-2% regression in
   another headline metric.
3. Diagnostics must observe the production implementation or the same retained
   kernel. A fast diagnostic-only execution path is not a product result.
4. Each approved `NEXT_TASK.md` scope proves its hypotheses end to end. Rejected
   code and temporary instrumentation are removed before moving on.
5. Report negative results. They narrow the search space and prevent attractive
   but locally invalid ideas from being repeated.

### Numerical policy gate

The primary lane preserves the current checkpoint's numerical and sampling
semantics within explicitly justified tolerances. Methods such as exact target
verification and a certified output shortlist belong in this lane.

Quantizing the F16 output matrix, compressing other weights, approximating
softmax functions, pruning, and lossy KV formats create a different numerical
model. They may be tested only as a clearly labeled opt-in lane with explicit
approval, separate quality evidence, and no silent replacement of the canonical
path. A speed result in that lane is never described as exact checkpoint parity.

## Program Order

The order below is dependency-driven. Work within a stage is ranked, not a
requirement to retain every item. A failed gate moves the program forward rather
than triggering an open-ended rewrite.

### Stage 0: Make Performance Evidence Trustworthy

**Goal:** establish a stable measurement and correctness foundation before
changing the hot path.

1. Add a sustained diagnostic mode for representative Q4_K, Q6_K, F16, and
   multirow shapes. It must prevent one-dispatch GPU ramp and underfill results
   from being mistaken for steady-state bandwidth.
2. Add production-path attribution for token GPU time, wall time, command and
   dispatch counts, context length, and bytes by projection category without
   introducing a second scheduler.
3. Capture Metal counter samples or a repeatable Xcode GPU trace for the key
   pipelines: bandwidth, occupancy, register pressure, ALU utilization,
   control-flow stalls, and cache behavior. Counter tooling is diagnostic and
   must remain optional at runtime.
4. Freeze correctness oracles for one-row logits, per-row multirow logits,
   sampled candidate transforms, and KV state transitions. Record the canonical
   benchmark environment and refresh the llama.cpp comparison only under its
   existing invalidation rule.

**Exit gate:** repeated diagnostics reproduce the production bandwidth regime,
the baseline passes the canonical benchmark protocol, and later candidates can
be attributed without temporary changes to inference semantics.

### Stage 1: Remove Avoidable Work and Memory Pressure

**Goal:** correct high-confidence whole-system inefficiencies before adding
more resident state.

1. Remove full KV-cache warm-up clears and rely on enforced populated-row
   bounds. Prove fresh session, warm-up restoration, reset, regeneration,
   prefix reuse, context shift, ring wraparound, and mixed 4K/8K window safety.
   Measure startup, TTFT, resident/compressed memory, and decode; retain it for
   lifecycle and memory improvements even if decode is neutral, provided no
   headline metric regresses.
2. Release or advise away GGUF tensor-payload pages after Metal upload while
   retaining valid metadata ownership. Measure load time, prompt processing,
   steady decode, page faults, and compressor behavior.
3. Skip final norm/output work when a turn is already known to terminate and no
   next-token distribution is needed. Treat this as tail-latency and energy
   work, not a generation-throughput claim.
4. Recover safe memory headroom from prefill scratch aliasing or liveness only
   when ownership is clear and measurements justify the complexity.
5. Replace quadratic whole-prefix streaming decode with incremental text
   decoding before speculative bursts or long outputs make callback work
   visible. Preserve UTF-8 and tokenizer boundary behavior.

**Exit gate:** avoidable KV materialization is gone, enough memory headroom
exists for verifier state and a small proposer, and no remaining housekeeping
item has a plausible 2% current-workload throughput ceiling.

### Stage 2: Push the Current Single-Token Decoder to Its Useful Limit

**Goal:** pursue Phase 2/3 changes that remove meaningful bytes or attack a
measured context-dependent cost; do not polish already bandwidth-saturated
kernels without counter evidence.

#### 2A. Weight residency and access

- Probe immutable shared, write-combined, private, and hazard-tracking choices
  on representative sustained matrices before changing production ownership.
- Test an offset-aware contiguous weight buffer or safe no-copy mapping only if
  it reduces physical duplication or counters show better access behavior.
- Retain a storage change only with real decode improvement or a material,
  independently measured memory reduction that enables a later stage without
  regressing load, prompt, or decode performance.

#### 2B. Output head

- In the canonical lane, keep the existing F16 head until the certified Phase 1
  shortlist is ready; projection/top-k fusion has a sub-millisecond ceiling and
  is cleanup unless it is needed by the verifier.
- In the explicitly approved altered-numerics lane, prototype output-only Q8
  first. Its first-order bound is approximately 15.8 tok/s from a 13.9 tok/s
  baseline. Advance to Q6 or Q4 only if Q8 quality and measured bandwidth justify
  the larger divergence.
- Measure logit error, top-40 overlap, post-transform KL divergence, perplexity,
  deterministic chat cases, representative quality tasks, memory, TTFT, and
  end-to-end throughput. Account for tied embedding storage explicitly.

#### 2C. FFN bytes and quantized projections

- Use counters to decide whether any hot shape is limited by memory traffic,
  unpack arithmetic, occupancy, or register pressure. The generic 56-block Q4
  path already sustains about 99 GB/s, so specialization by name alone is not a
  hypothesis.
- Test repacking, interleaving, or an alternate low-bit representation only when
  it physically removes fetched bytes or resolves a measured hardware limit.
  Include conversion/load cost, alignment, tails, and all Q4/Q6 layer mixtures.
- Do not retain gate/up/GELU fusion solely for dispatch reduction; the weights
  dominate and all norm/residual/GELU work is only about 0.53 ms per token.

#### 2D. Attention by context regime

- Preserve the present short-context attention unless a candidate has a
  measured whole-token bound above noise.
- Build and tune block-online softmax for 256, 512, 1K, 2K, 4K, and 8K contexts,
  reducing per-element rescaling while preserving Gemma attention softcap,
  GQA, causal masks, ring behavior, and alternating local/global windows.
- Consider partition autotuning next. Consider KV quantization or paging only
  for the long-context lane and only when conversion/dequantization is included.

#### 2E. Bounded cleanup

RoPE lookup, projection/top-k fusion, schedule reuse, argument buffers, indirect
commands, and GPU-resident sampling are admitted only when they support a later
multi-token cycle or their combined production-path bound exceeds the retention
threshold. Serial encoding, broad barrier removal, CPU sampler tuning, a Q6-only
rewrite, and wholesale CUDA/llama.cpp kernel transplants are rejected directions
unless new measurements invalidate the research evidence.

**Conventional plateau gate:** move on when either (a) two consecutive highest-
value conventional candidates fail their retain gates, or (b) every remaining
candidate for the headline short-context workload has a credible whole-path
ceiling below 2%. Record the new retained baseline and bottleneck decomposition.
Do not wait for the long-context track to finish before beginning Stage 3.

### Stage 3: Build the Multi-Token Verification Foundation

**Goal:** prove that one target-weight cycle can correctly process enough
positions to reduce committed-token latency. This is the bridge from conventional
optimization to Phase 1 and should begin as soon as the conventional plateau is
clear.

1. Define a multirow target API owned by the runtime, not by diagnostics or the
   CLI. Keep proposal policy separate from verification mechanics.
2. Implement a width-two causal chain verifier that emits both rows' logits,
   uses the exact per-position masks and RoPE positions, and writes tentative KV
   without corrupting committed state.
3. Implement explicit KV transaction semantics: begin, tentative append,
   commit accepted prefix, discard rejected suffix, and restore after stop,
   context shift, regeneration, or error. Avoid copying the full cache.
4. Prove every row against independent one-token execution before measuring it.
   Validate greedy and sampled transforms, local/global layer boundaries,
   cache-window wraparound, rejection at every position, EOS, and partial
   acceptance.
5. Measure a sustained full verifier, including all output rows and KV work.
   Derive its gate from complete verifier latency, proposal overhead, measured
   acceptance/commit distribution, and baseline token latency. Do not inherit a
   fixed width-two target from isolated or projection-only kernels.
6. Measure width four only if width two passes. Do not pursue width 16 merely
   because an isolated exact-16 kernel is fast; full causal attention, logits,
   acceptance, and KV economics decide width.

For a verifier cycle with latency `V`, proposal overhead `D`, baseline token
latency `B`, and expected committed tokens `E`, speculation can beat ordinary
decode only when:

```text
E > (V + D) / B
```

The retain gate remains stricter because it also requires the repository's 2%
and four-of-five result. If width two misses its latency target, attribute the
gap to quantized projections, F16 output rows, causal attention, KV handling,
or synchronization and attempt one evidence-backed repair. If it still cannot
beat the economic inequality with realistic acceptance, stop before building a
drafter and reconsider byte reduction or the Phase 1 hardware experiments.

### Stage 4: Obtain the First Multi-Token Speedup Without Training

**Goal:** validate the entire speculative protocol and acceptance economics
with a cheap, removable proposer.

1. Build a suffix-automaton or equivalent retrieval proposer over the prompt,
   conversation, and prior output. It must not own target semantics or mutate
   committed state.
2. Implement exact speculative sampling for the engine's temperature, top-k,
   top-p, minimum-p, repetition penalty, token history, and seeded PRNG rules.
   Greedy equivalence alone is insufficient for the product workload.
3. Report proposal length, acceptance by position, expected committed tokens,
   proposal time, verification time, rollback time, total cycle time, and
   committed tok/s by workload class. Include repetitive text, chat prose, code,
   and adversarial low-reuse prompts.
4. Start with a width-two chain. Add width four only when measured acceptance
   crosses its much higher cost threshold. Do not add a tree in this stage.

**Exit gate:** retain retrieval speculation only if the canonical chat path
passes correctness and performance retention. Even if it does not, keep the
verifier only when its API and measurements justify the learned-proposer stage;
remove proposal-specific complexity that failed.

### Stage 5: Add a Learned or Self-Speculative Proposer

**Goal:** raise accepted tokens per cycle enough to create the main decode
step-change without overwhelming 16 GB unified memory.

1. Freeze the target-verifier protocol and collect target-aligned proposal and
   acceptance data. Select training data by the intended chat, code, and
   structured-output workload, not by generic perplexity alone.
2. Compare a tiny recurrent/feature drafter, a parallel masked-position drafter,
   and a heavily skipped or lower-bit self-draft using the same local score:
   committed tokens divided by full cycle latency. Include model load, resident
   memory, and thermal behavior.
3. Train or adapt the smallest candidate only after an analytical memory and
   latency budget fits the recovered headroom. Production inference must remain
   engine-owned; training may use development tooling outside the runtime.
4. Start with a chain and dynamic proposal length. Combine retrieval and learned
   candidates only when their errors are complementary in local acceptance data.
5. Add hardware-aware tree verification only after a chain shows that branching
   should improve expected committed tokens more than it increases output-head,
   attention, KV, and sampling cost. Optimize tree shape for measured M4 cycle
   time, not for acceptance alone.

**Exit gate:** the selected proposer must pass distribution/correctness tests,
fit the memory budget without compressor-induced regressions, and satisfy the
canonical paired retention protocol. Otherwise retain the simpler successful
stage and move to the independent Phase 1 byte-reduction tracks.

### Stage 6: Phase 1 High-Risk, High-Upside Tracks

These are controlled experiments, not mandatory production features. Each
starts as an isolated proof with a predeclared success threshold and a clean
removal path.

#### 6A. Certified exact top-40 output search

Build a low-rank or clustered index with rigorous residual upper bounds, then
evaluate full F16 rows only where the bound cannot exclude a token from the
exact top 40. The result must match exhaustive selection after Gemma softcap,
temperature, penalties, and all configured sampling transforms. Measure index
memory, candidate count distribution, worst-case fallback, verifier-width
scaling, and full-path latency. Abandon it if bounds are usually loose enough
to fetch most of the 1.835 GB head.

#### 6B. ASTC HDR fixed-function weight decoding

First prove Metal ASTC HDR texture fetch, layout, precision, and sustained
effective bandwidth on one representative matrix. Then test the F16 output
head and Q6_K tensors before considering Q4_K, because they offer the clearest
byte reduction. Quantify encoding error, required minima/adapters, texture-size
constraints, coordinate overhead, cache behavior, load conversion, and quality.
Treat it as an altered-representation lane unless exact equivalence is proved.

#### 6C. More aggressive representations and sparsity

Consider structured contiguous sparsity, hardware-shaped sub-4-bit weights, or
activation skipping only if skipped data is physically not fetched and the
Metal kernel demonstrates a full-cycle benefit. Irregular sparse formats,
paper speedups from unrelated NVIDIA serving workloads, and format conversion
without a local bandwidth proof are insufficient reasons to enter production.

#### 6D. Long-context KV methods

After block-online attention is established, test KV quantization, paging, or
eviction against the 4K/8K context lane. Preserve mixed window semantics and
report quality separately. These methods do not compete for priority with the
short-context headline until attention becomes its measured bottleneck.

**Exit gate:** promote an unconventional method only if its production-path
benefit survives correctness or explicitly approved quality policy, memory and
load costs, thermal testing, and the standard paired retention protocol.

### Stage 7: Integrate the Best Compatible Methods

**Goal:** compose retained wins without assuming their speedups multiply.

1. Rebaseline after every major stage and repeat the bottleneck decomposition.
   A successful output or FFN compression changes verifier economics; successful
   speculation changes the importance of host, callback, attention, and
   selection overhead.
2. Combine only two already-retained changes at a time. Re-run correctness and
   paired performance because bandwidth, residency, cache, and thermal effects
   interact on unified memory.
3. Choose verifier width and proposal budget dynamically from measured context,
   workload-class, acceptance, and M4 cost curves. Keep a simple one-token path
   as an explicit fallback, not an accidental second runtime.
4. Compare the final engine with the pinned llama.cpp reference using its stored
   protocol only after the local baseline/candidate retain gate passes.

The target is not a predetermined tok/s number. A conventional result near the
17.5 tok/s idealized peak-bandwidth roofline would already imply that unchanged
single-token work is exhausted. Sustainable results beyond that must come from
fewer bytes per target cycle, multiple committed tokens per cycle, or both.

## Prioritization Rules for Every New Baton

Before writing a task into `NEXT_TASK.md`, answer all of the following:

1. Which measured whole-path cost does it attack?
2. What is its optimistic end-to-end bound, including all new work?
3. Does it reduce bytes, increase committed tokens per pass, solve a measured
   long-context cost, or merely rearrange sub-millisecond work?
4. Which numerical, tensor-layout, residency, KV-lifetime, synchronization, and
   sampling invariants change?
5. What independent oracle proves correctness or quantifies approved quality
   loss?
6. What diagnostic explains the result, and what canonical benchmark decides
   retention?
7. What exact threshold advances the idea, and what result stops it?
8. Can all candidate complexity be removed cleanly if it fails?

An enabling task whose immediate speed ceiling is below 2% is allowed only when
it is necessary for a named later gate, has its own measurable deliverable, and
does not claim a throughput win.

## Recommended `NEXT_TASK.md` Baton Sequence

The active document should select one item at a time from this sequence and be
rewritten only after the prior stop condition is met:

1. Sustained diagnostics and correctness oracles.
2. Remove full KV warm-up clears and prove cache lifecycle safety.
3. Release payload-page pressure after upload.
4. Run the highest-value approved single-token byte-removal experiment,
   selected by counters and the numerical policy gate.
5. Declare and document the short-context conventional plateau using the gate
   above. Block-online attention may continue as a separate long-context lane,
   but it must not delay the verifier when it cannot improve the headline path.
6. Implement the diagnostic-correct width-two causal verifier and KV transaction.
7. Add retrieval proposals and exact speculative sampling.
8. Select and test the smallest learned or self-speculative proposer justified
   by measured acceptance economics.
9. Explore certified output search, ASTC HDR, wider trees, and other Phase 1
    methods in the order dictated by the new bottleneck decomposition.
10. Integrate retained methods and run the final pinned reference comparison.

This sequence may change when a retained result moves the bottleneck, but a
change in order must cite new local evidence. It must not be changed merely
because an isolated kernel number or external headline is attractive.

## Deferred and Rejected Directions

Unless new production measurements overturn the research, do not prioritize:

- CPU sampling or generic host micro-optimization;
- broad barrier removal or serial command encoding;
- short-context KV quantization;
- standalone norm, GELU, RoPE, or selector rewrites;
- a Q6-only rewrite without hardware-counter evidence;
- width four or sixteen verification before width two proves its economics;
- tree speculation before chain acceptance is measured;
- a trained drafter before the verifier passes its latency gate;
- architecture replacement when accelerating the current Gemma checkpoint;
- direct transplantation of llama.cpp, MLX, CUDA, or serving-framework runtime
  code into the production path.

## Completion Condition

The program is complete when the engine has retained all compatible candidates
that still clear their evidence gates, the remaining ideas have ceilings below
the retention threshold or have failed their declared stop conditions, and the
final production path has current correctness, quality, thermal, memory,
canonical chat, and pinned-reference evidence. Until then, `NEXT_TASK.md`
records the complete currently approved implementation scope and its verified
outcome rather than treating this master plan as an unchecked task list.
