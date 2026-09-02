# Next Task

## Active baton

Build and evaluate a **full-fidelity, engine-owned learned speculative decoder**
for Gemma 2 9B Q4 on this base M4 MacBook Air. The registered external-drafter
proposal horizons are H=3, 4, 7, 8, 15, 16, and 23 future edges, corresponding
to total post-full verifier blocks of 4, 5, 8, 9, 16, 17, and 24 rows. H4 is
required by ReDrafter's L5 source control and H8 by EAGLE-3. H=16 is not a
padded H=15 run: it is required by PCTree and by separately labeled JetSpec-
latent and TreeFlash H16 transfers while JetSpec's contradictory H16 paper
description remains unresolved. The pinned TreeFlash and JetSpec executable
source fixtures are H15 and are tested independently.
The target-integrated arm registers the exact Speculative Streaming gamma4/k3
tree plus the finite, explicitly non-Cartesian transfer controls declared
below.

The candidate is a target-conditioned, bounded-horizon drafter plus a measured
target-verifier topology frontier. The H=3 conditional cascade of two retained
width-two calls is the bring-up point and currently strongest measured
topology; new tiled width-4/8/16 calls, conditional tile cascades, and
prefix-closed forests remain coequal until faithful measurements or
shape-specific valid bounds eliminate them. The winner must run through the
real Metal chat path, preserve the existing seeded target-token sequence and
cache semantics, and be judged by direct end-to-end execution.

This document deliberately replaces the former “shortlist feasibility, then
monolithic verifier, then one small pilot” baton. That sequence could reject
the idea for reasons that are not properties of the intended system:

- a selector fed target hidden states does not test a selector fed learned
  drafter states;
- fixed 2,048/4,096 vocabularies do not cover the useful hardware frontier for
  Gemma's 256K vocabulary;
- an arbitrary 2.0 ms draft allowance can force an under-capacity model before
  its acceptance is known;
- a fixed three-edge horizon cannot reject block-parallel models trained for
  8/16/24-row blocks or recurrent/tree methods whose benefit appears later;
- a monolithic width-four target kernel is not the conditional target schedule
  selected by measurements on this machine;
- a rank-256 factor used as the entire semantic head is not a faithful test of
  a semantic drafter with a rank-256 local correction;
- the previously opened 40-conversation training set cannot establish
  generalization;
- proxy recall, teacher argmax, total variation, and memorization are useful
  diagnostics, but none is the product objective.

The work is complete only when the actual trained architecture is executed in
the actual production topology and reaches one of the predeclared statistical
terminal states below, including the explicitly labeled fixed-N inconclusive
state. An imprecise result is reported as imprecise; it is not converted into a
success or a family-level rejection for convenience.

### The no-proxy rule

A reduced experiment may debug a mechanism or establish a valid optimistic
bound for the exact component it instantiates. It may not admit the product,
and its failure may not close a broader architecture.

In particular:

- a synthetic kernel may reject that kernel shape, not learned drafting;
- a full-vocabulary draft oracle may quantify shortlist loss, not serve as an
  inference input;
- perfect recorded proposals may reject a target schedule whose perfect upper
  bound is insufficient, but may not establish learned quality;
- a CPU implementation may prove algebra, packing, and decisions, but not Metal
  cost;
- replay may guide architecture selection, but only live execution may admit a
  candidate;
- a single small model, three-edge horizon, shortlist, training seed, or corpus
  size may not close its family.

There is no pass based on an easier approximation of the proposed system.

## Verified starting point

### Retained production baseline

Stage 4 remains the production baseline and must stay available unchanged by
feature flag while this work is developed.

It consists of:

- session-local exact suffix retrieval over the live token history;
- one-token proposals gated at support at least two and confidence at least
  2/3;
- the sampled target-token-match controller;
- the specialized width-two Metal verifier and batched full-vocabulary target
  selector;
- direct-to-resident K/V writes with transactional suffix restoration;
- exact stop, callback, regeneration, context-shift, and sampler-history
  handling.

Its thermally isolated canonical comparison selected the same 598 tokens as
ordinary decoding and reduced aggregate generation time from 38.293 s to
37.233 s, or 2.77%. It won both turns, matched both selected-token hashes and
final cache positions, and did not regress TTFT or prompt throughput.

The retained candidate used 26 speculative cycles: 17 matches, 9 misses, and
553 fallbacks. It issued 590 generation command buffers for 598 selected tokens
versus 598 command buffers for ordinary decoding. Retrieval itself cost about
0.0057 ms per selected token and is not the next bottleneck.

The current correctness suite gives useful one-edge coverage: width-two
candidate/K/V parity on the retained paths, sliding-cache wrap transactions, a
broad accepted-route callback rollback case, verifier/readback failures,
warm-up neutrality, and unavailable Metal. It is not an exhaustive stop,
budget, callback, or mismatch matrix. In particular, it does not prove the new
x1/x2/x3 carry and terminal-endpoint cases. Any implementation change still
requires the complete repository CI.

### What the rejected experiments actually established

The archived learned-speculation work is evidence, not production code.

- The old fixed 32K/rank-256 head generalized poorly after its training bug was
  corrected. That rejects that representation and training run; it does not
  reject a semantic backbone with a local rank-256 correction.
- The old 40-training/20-diagnostic-conversation corpus contained highly
  correlated windows, had already influenced design, and capped 46.4% of turns
  at 384 tokens. It is development data only.
- The monolithic width-four target projection measured 205.333 ms wall,
  204.708 ms GPU, and 5.697 GB of logical scheduled weight bytes. Its
  free/perfect ceiling was 1.178x. That rejects the measured monolithic
  width-four implementation only.
- Two retained width-two chunks measured 167.284 ms wall and 164.325 ms GPU
  for four sequential rows, giving a 1.488x free/perfect target-only ceiling.
  The diagnostic ran the second chunk unconditionally and did not implement the
  required transaction boundary, so it is a cost control rather than a product
  result.
- The later feature-tiled broad-head probe showed that the old serial
  projection was unnecessarily slow, but it omitted the live feature binding,
  full selector, real learned weights, three-edge recurrence, routing, and
  production readback. Its roughly 0.5 ms warm kernel timing is not a drafter
  timing.

No trained three-edge semantic drafter—let alone a registered 7/15-edge
member—complete device-resident rollout, conditional second verifier call,
learned production controller, or cooled learned end-to-end benchmark has yet
existed.

### Current code boundaries that must change

The existing structure is a sound Stage 4 baseline but is not an adequate
container for the new controller:

- include/local_gguf_engine/engine/gemma2_runtime.hpp exposes only a
  RetrievalProposer pointer to sampled generation.
- src/engine/gemma2_runtime.cpp owns one pending token, one preview sampler, one
  proposal edge, and retain-one/retain-two outcomes.
- include/local_gguf_engine/engine/sampling.hpp can build a hypothetical
  selection request after only one token.
- include/local_gguf_engine/engine/gemma2_metal_plan.hpp owns exactly two
  verifier rows, one transaction backup arena, two target-selection lanes, and
  no learned artifact, feature state, or draft cache.
- the target encode contract is also structurally width-two: fixed result
  arrays/workspaces and the per-layer `second_attention_token_count` cannot
  express a 24-row chain or the 1,025-row rooted verifier input required by a
  published 1,024-draft-node DDTree. Repeating the width-two API cannot
  establish one-invocation or tree geometry.
- src/engine/gemma2_runtime.cpp constructs, submits, waits for, and reads a
  width-two verifier inside one function. A fused draft-plus-first-verifier
  topology requires separate encode, submission, completion, and transaction
  boundaries without duplicating target execution.
- src/tensor/backends/metal/metal_command_context.* stages token IDs and
  repetition histories from the host. It cannot yet consume a device-produced
  proposal token as the second verifier input or as a hypothetical history
  suffix.
- src/engine/gemma2_chat_session.cpp owns only the retained retrieval proposer.
  Reset, regeneration, sampler-option changes, and context rebuilding have no
  learned-cache or artifact lifecycle.
- the archived Stage 5R feature scaffolding has four raw F32 intermediate
  workspaces plus the final-normalized hidden state. Decode/verifier rows can
  redirect those buffers, but prefill captures only the last row of the final
  chunk. It is not a five-tap, all-prompt-row, full-context feature substrate.
- session construction currently creates and uploads the Metal plan before a
  learned artifact could be authenticated, and EngineModel has no
  cryptographic tensor-payload receipt. The load path must be split into a CPU
  prepared plan and an atomic upload step.
- target upload also transforms weights: the current plan repacks the retained
  Q6_K FFN down projections to Q4_K in addition to generating the output-head
  Q8 payload. Source-GGUF identity alone therefore does not identify the target
  arithmetic actually resident on GPU.
- callback exceptions can escape `Gemma2ChatSession::generate_user_turn` after device K/V
  and local generation state advanced but before `cached_token_ids_` and the
  session sampler are copied back. The current outer session then appears
  reusable with mutually inconsistent host/device state.
- the current public generation-duration field ends at the last token
  selection. Final K/V-only work, restoration, finalization, and tail callback
  work can fall outside it, so it cannot be the admission clock unchanged.
- only sliding-attention layers wrap. Global-attention storage is bounded by
  the context length and the current runtime rejects work beyond that boundary.
- the target output tensor is semantically [3,584, 256,000], and a token is a
  physical packed **column**, not a C++ row. New indexed-head code must retain
  the existing column-stride convention.
- the current metrics describe a one-edge route and derive scheduled bytes from
  a one-row formula. That formula is invalid for learned work plus conditional
  width-two calls.

These boundaries are to be redesigned. Do not preserve them by weakening the
experiment.

## Exact product hypothesis

Let:

- x0 be the target token already selected at the current origin but not yet
  written to K/V;
- H be a registered external proposal horizon in {3,4,7,8,15,16,23};
- d1 through dH be the learned draft chain;
- x1 through xH be authoritative target samples from verifier rows/nodes;
- M_i be the event x_i = d_i;
- G be the frozen causal decision to launch call two, evaluated only after M1
  and M2;
- P1 be Pr[M1] and P12 be Pr[M1 and M2];
- P123 be Pr[M1 and M2 and M3] in the explicit always-launch control;
- P12G be Pr[M1 and M2 and G];
- \(P_{12\bar G}\) be Pr[M1 and M2 and not G];
- P123G be Pr[M1 and M2 and G and M3].

The equations and event table immediately below instantiate the mandatory H=3
bring-up controller; they do not cap the learned architecture. At an eligible
origin, use the causal target feature state that produced x0,
the registered root-only or full projected target-feature context, x0, the
exact sampler history, and proposal-only previews of the next target RNG draws
to produce d1 through dH. For H=3, execute:

~~~text
draft on GPU: d1, d2, d3

target call 1: [x0, d1]
  row 0 -> sample x1; validate d1
  row 1 -> sample x2 only if x1 == d1; validate d2

target call 2: [d2, d3]
  issue only if x1 == d1, x2 == d2, no endpoint intervenes,
  and the causal second-call benefit gate selects it
  row 0 -> sample x3; validate d3
  row 1 -> next target candidates after a full match
~~~

The boundary-free outcomes under the H=3 conditional policy are:

| Accepted draft edges | Target calls | Canonical K/V retained | Carry |
| ---: | ---: | --- | --- |
| 0 | 1 | x0 | x1 |
| 1 | 1 | x0, d1 | x2 |
| 2 | 2 | x0, d1, d2 | x3 |
| 3 | 2 | x0, d1, d2, d3 | none; use call-two row-one candidates |

If both early edges match but the second-call gate declines, finalize call one,
adopt the already sampled d2/S2 as a pending token, and process d2 through an
ordinary logits-producing decode before delivery. Use K/V-only mode only when
d2 itself is a stop token or exhausts the selected-token budget. Insufficient
room for width two chooses the ordinary one-row/context-policy path; it is not
silently treated as an endpoint. The gate is causal: it may use d1/d2,
precomputed c3, route/context/budget state, sampler options, retrieval state,
and frozen costs, but not x3, a future target result, or an evaluation-workload
label.

With the second verifier always selected after the first two matches, expected
**canonical target-K/V-frontier progress** per completed cascade episode is:

\[
E[L] = 1 + P_1 + P_{12} + P_{123G},
\]

where always-launch makes P123G equal P123. A selected mismatch/correction carry
is intentionally excluded until the later target call that caches it; that
later call and its progress are each charged exactly once. The gated episode has the same base
P12 term because its ordinary fallback still processes x2; only a launched and
matched third edge adds the fourth retained token. This is genuine speculative
decoding. The target still computes every authoritative distribution, and the
draft affects only which already-computed target rows can be retained. No
rejected proposal can select an output token.

This displayed progress formula is a boundary-free scale diagnostic: it assumes
budget, stop, callback, and context survival through the episode. The exact
controller multiplies each deeper contribution by its causal survival event and
uses complete trajectory replay/live measurement. For H=4/7/8/15/16/23 it repeats the
same rule over registered tile boundaries: sample authoritative target rows in
order, stop at the first nonmatching edge, expose callbacks one selected token
at a time, commit only that prefix, and carry/cache the first target mismatch.
No closed-form iid extrapolation substitutes for those longer live paths.

### Why conditional two-plus-two is the measured reference

The conditional cascade is the first topology to build for semantic and
measured hardware reasons, not because it is easier to implement and not as an
architectural lock.

- Call two is useful only after the first two proposal edges match. Waiting for
  those target samples avoids an entire 5.697 GB target sweep on every earlier
  miss.
- Even after two matches, call two competes with ordinary processing of d2.
  Its calibrated c3 and measured incremental value must justify the wider call;
  low-benefit suffixes return to ordinary decode without changing output.
- The retained width-two family is already specialized and measured at
  83.642 ms. Two such calls measured 167.284 ms, roughly 38 ms below the failed
  monolithic width-four implementation even before conditional skipping.
- Finalizing call one before call two makes the current single backup arena
  sufficient for this topology. A wider or forest verifier must receive its own
  speculative K/V arena; the current arena is not a reason to exclude it.
- Sampling, stop handling, token budgets, and callbacks occur at the same
  per-token boundaries as ordinary generation. Work computed beyond a later
  callback stop is legal only while isolated and fully discardable, but its
  cost remains real.

Build genuinely M4-tiled single-invocation width
3/4/5/7/8/9/15/16/17/23/24 chain controls
with isolated speculative K/V and compare them against every registered
conditional tile partition on the same draft outcomes. Build or validly exclude
each predeclared k/N forest before selection. Any topology may replace the
cascade if its complete live cost wins. The failed Stage 6 kernel is not
evidence against a different production-correct implementation.

## Correctness contract

### Target authority

The target verifier always uses the complete Gemma vocabulary and the retained
production-Q8 output head. A dynamic vocabulary is allowed only inside the
drafter. This follows the separation used by
[SpecVocab](https://arxiv.org/abs/2602.13836) and
[DynaSpec](https://arxiv.org/abs/2510.13847): shortlisting reduces draft-head
work; target verification remains exhaustive.

The controller samples x1 through xH from the full target compact-candidate
path and accepts each edge only on identity with d1 through dH. It does not use
the standard min(1, p/q) rejection rule or a residual distribution. Standard speculative
sampling is distribution-preserving but generally consumes randomness
differently, so equal seeds need not produce equal sequences
([Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html),
[Chen et al.](https://arxiv.org/abs/2302.01318)).

For a proposal or child set fixed independently of the authoritative target
draw:

- a deterministic single proposal d matches with target probability p(d);
- an independently sampled draft D from q matches with probability
  sum_v p(v)q(v);
- a proposed child set S matches with target mass sum_{v in S} p(v).

None of these is generally 1 minus total variation. TV remains a distribution
diagnostic, not an acceptance label or admission gate.

When a proposal is generated from a common random ticket, it is not independent
of the target draw. Its match probability is the joint collision probability
under the frozen coupling and must be measured/derived from that joint law; do
not substitute p(d) or the independent sum_v p(v)q(v).

### Seeded sampler identity

The authoritative sampler retains its current contract and transform order:

1. Gemma final logit softcap;
2. repetition penalty over the exact bounded history;
3. stable logit-descending, token-ID-ascending ordering;
4. top-k;
5. the current pre-temperature top-p computation;
6. min-p;
7. temperature;
8. the current libc++ uniform_real_distribution<double> driven by
   mt19937_64.

Add fixed-capacity authoritative sampler checkpoints S0 through SH
(`ExternalHmax+1=24` checkpoints) and a span-based hypothetical-history
request. The controller
adopts only the checkpoint for tokens authoritatively selected or delivered
after the preceding callback
boundary, including an uncached carry. A preview behind a callback that stops
or throws is neither selected nor observed and is discarded. The controller
never consumes the unused row-one sample after a full match.

The sampled drafter arm may use common random numbers, but must not migrate
target sampling to Gumbel-max or otherwise change the existing seed-to-token
mapping. External arms preview up to H opaque draw tickets from a copy of the
sampler state. Because Arm S emits next trees for every possible accepted depth
before that depth is known, it previews the exact offset range needed by every
anchor, up to `2*gamma+1` draws and a fixed `StreamHmax=23` capacity of 47;
each anchor/tree
receives only its registered future slice. Leave the authoritative state
untouched and pass those tickets to the GPU drafter. A lower-capacity Arm S
must use deterministic proposal selection or defer proposal sampling until the
host has selected the anchor; it may not reuse the wrong offset. The common API
is preview_ticket/sample_with_ticket; a ticket is
used only by proposal sampling, not fed into the semantic recurrent state,
unless a separately registered draw-conditioned classifier is explicitly
defined. Before refactoring, pin golden sampled streams. Prove equality of raw
engine outputs, post-engine state, normalized variate mapping, boundary
behavior, and selected token under the pinned libc++ implementation. Equal
engine-call count alone is insufficient because the current implementation
draws uniform_real_distribution<double> over the distribution's total mass,
not necessarily a reusable unit interval by an implementation-independent
mapping. Greedy paths issue no stochastic ticket and consume no RNG; each
stochastic ticket contains enough raw evidence to reproduce exactly one future
target draw under an arbitrary prepared total.

Tree proposal randomness is explicit rather than hidden in a top-k operation.
The child on the coupled primary path uses the target ticket for that causal
depth. Additional siblings are either deterministic stable top-k alternatives
or use artifact-bound, path- and lane-domain-separated **proposal-only**
subdraws from a private proposal RNG. They never consume, alias, or reorder
authoritative target tickets. The registry fixes this choice per tree
constructor, and CPU/Metal goldens cover sibling order, duplicate merging, and
every root-to-leaf ticket mapping.

This is an unproven scalar inverse-CDF coupling inspired by, but **not an
implementation of**, [DISD](https://arxiv.org/abs/2408.07978). DISD's guarantee
comes from token-ID-indexed shared exponential/Gumbel randomness; independently
sorted scalar CDFs inherit no such guarantee and may couple poorly. A true DISD
arm would deliberately change the target randomness mapping and is out of this
same-seed contract. A scalar-ticket failure disables only that proposal arm; it
never changes target output correctness. A deterministic draft arm remains a
required control.

### Numeric target identity

The current width-two parity tolerance is useful evidence but is not, by
itself, a universal same-seed proof. Before admission, committed target K/V
must be bit-identical to ordinary one-row execution for the same causal
path. If a multi-row kernel cannot provide that, recompute or copy canonical
one-row K/V before continuing; a current-row logit margin cannot certify
arbitrary future state.

Compact current-row target selection must also be bit-identical, or a
conservative error bound must detect every candidate-order/CDF ambiguity and
recompute that row through the ordinary selector before the authoritative
sample. The ambiguity escape hatch applies only to current selection, never to
non-identical retained K/V.

Real-model randomized parity must cover all supported quantized projection
families, close candidate ties, adversarial draw boundaries, sliding wrap,
global-context tail/boundary, and long contexts. Run full K/V hashing/readback
in a separate untimed correctness pass so it cannot contaminate headline
timing. Zero mismatches on the sealed trajectories remains a
mandatory observable check, not a substitute for the numeric contract.

Persist raw selected token IDs and SHA-256 in admission output in addition to
the existing diagnostic FNV hash. Compare visible bytes, stop reason and token,
callback sequence, sampler checkpoint, cache position, retrieval/learned
history, and final logical K/V.

## Learned architecture

The experiment tests recurrent, block-parallel, latent-mixture, source-exact
conditional/corrected tree, target-integrated multi-stream, and heterogeneous routed
architectures that share target authority, vocabulary machinery, artifacts,
and verification. A single under-capacity or short-horizon pilot may not close
any family.

### Causal target feature state

Define F_t as the target feature state whose output head produced the
distribution from which x_t was selected. The learned input at an origin is
only:

- F_t and its causal projected context;
- x_t;
- the sampler request and proposal-only draw tickets;
- immutable model and artifact state.

F_{t+1} and later target states are labels. They may never enter the drafter at
that origin.

Capture multiple target residual levels during target work already being
performed. Register three conditioning families:

1. **DFlash shared fusion:** one fused context representation from five
   uniformly spaced low-to-high taps, with a three-tap low/middle/high control;
2. **DFlare layer-wise fusion:** the published-fidelity control has seven draft
   layers, and each learns its own softmax mixture of nine target layers
   selected by the frozen proportional zero-based Gemma map
   `{1,6,11,15,20,25,30,34,39}`. A
   separate Gemma-specific extension exposes all 42 block outputs plus the
   registered embedding/final-normalized endpoints; a result from that wider
   bank is not reported as the published DFlare control;
3. **DFly hybrid conditioning:** global transformed target context is combined
   with layer-specific target views and a predecessor-conditioned head.

The broad bank must be accumulated/projected while target layers are live; do
not retain 42 raw 3,584-wide histories at runtime. Before any trace is
collected, freeze every source tensor/block index, whether final-normalized
hidden is included, the shared three/five subsets, layer-wise mixing rule,
resident/trace precision and rounding, row source, stride/layout, and model
identity. This is new infrastructure: the archived four raw F32 workspaces plus
final-normalized output are not silently relabeled as any of these controls.

The feature substrate must support both current/root-only conditioning and full
projected-context conditioning. Root-only recurrence is a coequal Arm R design,
not merely an ablation: it preserves ReDrafter's defining low-cost path. Full
context K/V is primary for Arm P and an optional Arm R member whose additional
acceptance must repay its maintenance and attention traffic.

For the full-context production members:

- accumulate the shared or per-draft-layer learned mixtures/projections without
  a host copy;
- retain only the fused feature or per-draft-layer K/V projections, not raw
  target-wide histories;
- project every accepted target row into the K/V cache of every draft layer;
- let each draft layer attend to the entire accepted projected target context;
- keep speculative draft K/V writes in the same retain/restore transaction as
  target K/V;
- identify every live feature with a typed TargetStateView containing its row
  source, plan epoch, tap contract, artifact identity, and logical position.

The H=3 conditional-bring-up row mapping is explicit:

| Route outcome | Feature state for the next selected anchor |
| --- | --- |
| ordinary decode | ordinary row zero |
| first-call edge-one miss | call-one row zero |
| first-call edge-two miss | call-one row one |
| second-call edge-three miss | call-two row zero |
| all three edges match | call-two row one |
| second-call decline/context fallback | ordinary-x2 row zero after fallback |
| prefill origin | final evaluated prefill row |

Target and draft caches use related but different logical coordinates. If x0
is written to target position p, target rollback begins at p, while the feature
emitted by processing x0 predicts the anchor at p+1 and its draft-cache
transaction begins at p+1. Conditional call one emits projected feature rows
for p+1 and p+2; call two emits p+3 and p+4. Target and draft caches therefore
need independent logical offsets, masks, ring victims, and backup ranges even
though one transaction ticket commits or restores them atomically. Never expose
independent target-only and draft-only commit calls.

Prefill must fuse/project every prompt row needed to bootstrap the full-context
draft K/V, not only copy the final row of the final chunk. Cached-turn
continuation, context rebuild, ordinary decode, verifier rows, and terminal
K/V-only rows must extend the same feature sequence. Sliding draft state wraps
under its declared mask; global target context stops/rebuilds at the existing
global-context boundary rather than pretending that global layers wrap.

A raw buffer pointer plus row index is not an adequate lifetime contract.
Stale epochs, overwritten verifier scratch, wrong tap identities, and future
rows must fail closed. `TargetStateView` is borrow-only while its owning target
ticket is live. Before finalize/reuse, the selected ordinary/linear/forest row
must either be transactionally promoted/copied into persistent feature storage
and returned as a `TargetStateLease`, or the lease must keep the owning ticket
and scratch alive until consumption. Promotion participates in the same commit/
rollback and covers root-only fused state as well as every full-context child;
a view into finalized verifier scratch is invalid by construction.

For a heterogeneous bundle, one `FeatureSinkSet` owns a child-indexed
`CacheFrontier {artifact_digest, epoch, logical_position}` and one joint cache
journal. Each full-context child declares exactly one policy:

- **eager:** every accepted target row is projected into that child at the time
  it is produced, whether or not the child drafted that origin; all child
  maintenance traffic and writes are charged and rolled back atomically; or
- **lazy:** the child is marked stale immediately. It cannot be routed until a
  charged replay/bootstrap reconstructs its exact K/V through the required
  frontier from a persistent artifact-neutral feature basis proved sufficient,
  or by re-executing the target context. If the retained basis cannot reproduce
  all of that child's DFlash/DFlare/DFly inputs exactly, target rebuild is the
  only legal path.

Target-layer capture may be shared, but child-specific fusion/projection is not
free. Root-only children likewise need their selected-row fused state promoted
while its source is live or must pay re-execution. Cross-child switching checks
the frontier before proposal work; stale K/V is never treated as a route
decline or silent Stage 4 fallback.

This design takes the important lesson from
[EAGLE-3](https://arxiv.org/abs/2503.01840)—multi-level target features, direct
token prediction, and training-time draft rollout—and the full-context lesson
from [DFlash](https://arxiv.org/abs/2602.06036): target features are injected
as reusable K/V into the draft layers rather than fused only once at the first
position. [DFlare](https://arxiv.org/abs/2606.02091) prevents a negative shared-
fusion result from closing layer-specific feature use, while
[AngelSpec/DFly](https://arxiv.org/abs/2607.25852) supplies the registered hybrid
global-plus-layer-specific control.

### Arm R: recurrent direct-token MTP

Arm R is the high-entropy conversational candidate:

- either the current fused F_t alone or one full-context attention/fusion step
  initializes a learned semantic state;
- a shared gated recurrent, GRU, or compact SwiGLU/attention cell rolls that
  state through d1 to dH;
- each edge consumes the preceding proposed token embedding; its draw ticket
  is applied only when sampling that edge's proposal distribution;
- each edge predicts the next token distribution directly;
- the recurrent state is self-generated after the first edge; future teacher
  features never replace it.

The capacity frontier includes widths 512, 768, and 1,024 plus separate Gemma
target-width 3,584 architecture transfers. The compact frontier additionally
includes a gated-residual cell, standard GRU, and two-layer semantic cell, but
none may stand in for the source controls below.

The [ReDrafter](https://arxiv.org/abs/2403.09919) control uses
`s_t=SiLU(W_no_bias e_t+U_with_bias s_{t-1})`, with initial-state semantics
bound by the upstream golden, concatenates `[s_t,h_origin]` to width 2h, applies
two residual blocks `x+silu(Linear_bias(x))`, and maps through the learned
bias-free 2h-to-vocabulary head while sharing target token embeddings. Its MLX
**executable-source** control has recurrent state width h, exit width 2h, root-
inclusive code parameter L=4, and fixed beam-width cells W=1/2/3/4. All beams
begin with the same target-sampled init token, which is not yet in canonical
K/V, so L4 contains that root plus only three recurrent draft edges. The pinned
MLX verifier explicitly performs **no prefix compression**: it sends W*L
block-diagonal physical rows, duplicating the identical root W times, for exact
row counts 4/8/12/16. Preserve that repeated-root topology as a distinct
validated `RepeatedRootBeamBatch`; a unique-root tree is not byte- or cost-
faithful MLX-executable evidence.

This conflicts with the article, which defines beam length as draft/RNN-call
count and describes prefix removal. Register a separate article-described
four-draft-step, prefix-deduplicated control (root plus four edges; equivalent
to code-L5 geometry) at W1/2/3/4, with row bounds 5/9/13/17. Withhold an
unqualified source-reproduction label until a pinned author golden resolves
the length/compression conflict.

The pinned PyTorch packer is a separate executable source-method control. It prefix-dedups
the common-root beams into one dynamic tree, with checked root-inclusive bound
`1+W*(L-1)`: L4/W1..4 yields 4/7/10/13 rows. The README/PyTorch L5/W45 point
has at most 181 tree rows (180 non-root), while its raw rectangular beam and
post-verification expanded-K/V storage are 225 entries. A tree-native path may
avoid that expansion only after exact cache-equivalence proof. A deduplicated
L4 Apple-Silicon geometry is an explicit transfer, not the MLX-executable
runtime. Freeze the official top-50 draft-logit warp, score arithmetic, exact
pinned backend/version equal-score membership/order, uncompressed MLX packing,
compressed PyTorch dedup/order/packing, and exact output bytes from pinned upstream commit
`bd8586bb9bbfa761644c36f932a0cd1eb3f1cdf9`. Its training fixture uses two epochs, learning rate
`1e-3`, zero weight decay, 0.1 warmup then cosine decay, BF16, maximum sequence
2,048, target horizon 5, micro-batch 4, accumulation 4, eight workers/global
batch 128. Gemma training may scale capacity, but first has to reproduce the
upstream fixture's exact beam/tree bytes.

The [EAGLE-3](https://arxiv.org/abs/2503.01840) control captures the hidden
state **before** executing target-block loop indices
`{2,floor(N/2),N-3}`, concatenates those three features, and applies the
official bias-free `Linear(3*target_hidden,draft_hidden)` fusion. For Gemma's
42 blocks the frozen pre-block tap indices are `{2,21,39}`, equivalently the
outputs after zero-based blocks `{1,20,38}`. This convention is part of the
feature receipt; calling them outputs "after layers `{2,21,39}`" is incorrect.
The draft decoder does not receive an ordinary additive embedding. Its exact
input contract separately RMS-normalizes the selected-token embedding and the
fused current target-or-draft feature, concatenates them to width `2h`, and
uses the official bias-free `2h` Q/K/V projections in its one decoder layer
before the shared target LM head. After the first edge, unavailable future
target feature g is replaced by the drafter's preceding self-produced feature
a. Training-time-test self-rollout and its loss mask are mandatory. Bind the
upstream fixture's constructor `depth=7`, which emits an initial depth-one
proposal and then performs seven expansion iterations: maximum non-root path
depth 8, 60 root-inclusive verifier rows (`total_token=60` becomes 59 non-root
candidates plus the root), and expansion top-k 10. Before global pruning the
builder materializes `10+7*10^2=710` non-root candidate entries across one
10-wide first level and seven 100-entry expansion sets; its root-inclusive
pre-tree bound is 711 and its compressed expansion-parent-row bound is
`1+7*10=71` for score/token metadata. The iterative decoder forwards ten
active nodes in each of seven iterations (70 forward rows total), and its last
expansion mask has ten queries by 80 keys. Candidate entries may share token
IDs and are not mislabeled as unique nodes. Freeze confidence threshold,
expansion score, exact pinned PyTorch/device `topk` equal-score membership/
order, and byte-exact packing behavior from pinned official commit
`cb7e0841fe0c206c6ed74a197ad5e2a1f13f5a2b`. A stable deterministic tie policy
is a separately labeled production transfer, not attributed to upstream
`topk`. A generic recurrent direct-token head is not EAGLE-3 evidence.

Do not label a ReDrafter/EAGLE hybrid as evidence for either.
Train every registered capacity with exactly three predeclared initialization
seeds unless the finite registry gives a valid optimistic-bound exclusion. A
production-shaped Metal lower bound may eliminate the target-width control only
if favorable uncertainty still leaves perfect proposals below the product
threshold; a smaller arm's failure does not eliminate the larger arm.

ReDrafter is the closest published Apple-Silicon precedent for recurrent
token-conditioned drafting. Its reported M1 Max results also warn that wider
beams can lose despite accepting more tokens; geometry is hardware-dependent
and must be measured here.

### Arm D: one-pass block-diffusion drafting

Arm D is a concrete family, not shorthand for a generic parallel MTP head. Its
source-mechanism DFlash/DFlare transfers use an input block of `B=H+1`: position
zero is the clean target-selected anchor and positions one through H are mask
tokens. One draft Transformer invocation predicts all H masked positions. The
masked positions attend bidirectionally within their own block and to the
causal injected target context, but never to a future target token/feature or a
different training block. The inference denoising-step count is exactly one;
adding iterative denoising is a separately registered architecture and cost.

Training first runs the frozen target over complete target-generated sequences,
then randomly samples response anchors, constructs masked blocks at those
anchors, injects the corresponding target feature sequence as K/V at every
draft layer, and minimizes weighted token CE over all H future tokens. Multiple
blocks may share a sparse training batch only if the attention mask proves
block isolation. The source DFlash mechanism shares and freezes the target
embedding and target-width LM head, uses one shared fusion of uniformly spaced
target taps, and registers one-, three-, and five-layer drafters. Its mandatory
Qwen3-4B B16/H15 five-layer source cell uses target taps `{1,9,17,25,33}`,
concatenates the five views through shared Wc and then RMSNorm, and uses the
**same** layer-specific K/V projection matrices for injected target features
and draft anchor/mask states. Its corpus is approximately 800K Nemotron Post-
Training Dataset V2 plus CodeAlpaca prompts with target-regenerated responses.
Train six epochs with AdamW `6e-4`, gradient clip 1.0, 0.04 warmup then cosine
decay, maximum sequence length 3,072, up to 512 random anchors, and fixed
position-loss gamma=7. Freeze source revisions, corpus proportions, global
batch, regeneration template/sampler/seeds, optimizer details, and every other
unpublished field in the Phase-0 resolution ledger; do not inherit them from
DFlare. Reproduce upstream mask/fusion/KV/logit bytes before the Gemma transfer.

The pinned executable Qwen3-4B DFlare control binds source target taps
`{1,5,9,13,17,21,25,29,33}`, draft width 2,560, FFN width 9,728, seven draft
layers, 32 query heads, 8 KV heads, head dimension 128, and B16/H15. The Gemma
DFlare architecture transfer is separately identifiable: authenticated target
width 3,584, seven draft layers, B16/H15, the frozen proportional target-tap
map `{1,6,11,15,20,25,30,34,39}`, a learned
softmax fusion vector per draft layer, an RMSNorm **after** that layer's
weighted target-state sum, and **different K/V projection matrices** for
injected target context versus the clean decode anchor/masked draft states.
For draft layer i its source-faithful fusion is

\[
\alpha^{(i)}=\operatorname{softmax}(W^{\mathrm{fuse}}_{i,:}),\qquad
f_t^{(i)}=\operatorname{RMSNorm}\!\left(\sum_{j=1}^{9}
\alpha_j^{(i)}h_t^{(j)}\right).
\]

The source-mechanism training cell is six epochs of AdamW at learning rate `6e-4`,
gradient clipping 1.0, cosine decay with warmup ratio 0.04, maximum sequence
length 3,072, 512 random anchors per sequence, and global batch 64 on the full
2.4M target-regenerated mixture of NVIDIA Nemotron Post-Training Dataset V2,
CodeAlpaca, and Step-3.5-Flash-SFT (target sampling temperature 0.6), with
response-only loss and removal of every training sample having at least a
32-gram overlap with the test set. Freeze source revisions, exact subset
proportions, filters/dedup, templates, and regeneration seeds in Phase 0; any
field not published remains blocked rather than inferred from another DFlash
recipe. Its
progressive position loss uses `exp(-(k-1)/gamma)`. The pinned official run
script actually fixes gamma=7 throughout all six epochs; register that exact
fixed7 **executable-code control**. Appendix A.1 instead gives six active epoch
values `gamma={4.5,5.5,6.5,7.5,8.5,9.5}` from gamma0=4.5 and +1 per completed
epoch; register it separately as the paper-appendix control. Equation 5 gives a
third, continuous per-step interpolation to a gamma_max whose numeric value is
not published. Never invent gamma_max=10.5. Withhold an unqualified
“source-exact reproduction” label for the paper schedule until a pinned author
config/checkpoint resolves the appendix/equation/code conflict. The all-42-tap
bank, shared K/V, alternative fixed/continuous gamma, and smaller-data variants
outside these three named controls are labeled extensions/ablations, not
DFlare mechanism-transfer evidence. DFlash/DFlare published checkpoints and
target layer maps are Qwen/GPT-OSS rather than Gemma 2. Before interpreting the
Gemma result, an upstream model/checkpoint fixture must reproduce the source
mask, tap/fusion, heterogeneous-K/V, loss, and draft-logit behavior. The Gemma
cell then tests a target-matched architecture transfer; it cannot by itself
confirm or falsify the published target-specific result.

Register native blocks B4/B8/B16 and trained B24. B24 is not obtained by
running a B16 checkpoint with a longer mask: that known distribution shift is
only a negative control. The DBloom controls are (A) a frozen B16 checkpoint
post-trained directly at B24 and (B) the same B16 checkpoint first continued at
B16 and then expanded to B24. The shared expansion step uses exactly 20K
OpenMathInstruct-2 prompts plus 10K long-competitive rStar-Coder prompts from
high-ceiling domains, target-self-distilled one-response English examples,
32-gram evaluation decontamination, B24 CE with weight 1 at offsets 1..15 and
weight 3 at offsets 16..23 with no decay, learning rate `1e-5`, four epochs,
and global batch 64. Generate this B24 expansion data greedily and train CE on
the target-generated/argmax token IDs. Arm B's preceding B16 continuation uses 250K
OpenMathInstruct-2, 300K code prompts from opc-sft-stage2/rStar-Coder/
OpenCodeInstruct, 60K Nemotron-v2 STEM replay, and 60K Nemotron-v2 chat replay
(approximately 670K total), target response sampling temperature 0.6, learning
rate `5e-5`, and two epochs; its B24 step
then changes only the initialization and reuses Arm A's expansion corpus and
knobs. Both stages use maximum sequence length 3,072, target-response cap 2,048,
512 strided/random block-start anchors per sequence, AdamW, gradient clip 1.0,
and cosine decay. Freeze anchor-order details, target regeneration sampler/
template/seeds and every corpus filter
in the manifest rather than treating “approximately 30K” as a recipe.
Preserve the full per-cycle accepted-length histogram and ceiling-bin mass. The
development-only ceiling indicator can prioritize training order, but B24
remains in the finite registry and can be omitted only by a valid family-level
cost bound, never because a mean acceptance statistic looked flat.

DDTree is Arm D's faithful tree constructor: one Arm-D block pass produces the
per-depth marginal distributions, then a GPU-resident best-first builder ranks
prefixes by cumulative marginal log probability under the registered node
budget. The marginals remain branch-independent; do not silently add a
parent-conditioned correction and continue calling it DDTree. Its tree is
verified by the real prefix-closed target verifier and all builder, head,
selection, and node costs are charged.

[CaDDTree](https://arxiv.org/abs/2606.01813) remains the same branch-independent
DFlash marginal model and DDTree heap; it is **not** parent-conditioned. It
adaptively stops after n non-root nodes to maximize the source surrogate

\[
\widehat\theta^*(n;\ell)=
\frac{1+\Phi^*(n)}{C_d+C_v(n;\ell)},\qquad n\in\{0,\ldots,1024\}.
\]

Start at the legal empty draft `n=0`; after each next heap pop, compute the new
surrogate and stop at the first non-improvement. This first-stop rule is valid
only under the paper's diminishing benefit/cost-ratio assumption; a monotone
convex C_v in n is its sufficient condition. The local cost artifact profiles
every context grid point `ell={0,1K,...,8K}` and n=1..1024 with 5 warmups plus
20 timed target trials and uses their median, measures C_d as the mean of 500
draft rounds after 5 warmups, separately records/validates C_v(0;ell), fits a
monotone convex piecewise-linear curve, linearly interpolates between context
grid points, and authenticates raw samples, interpolation, fit code, and fit
digest. Behavior at exactly 8K and beyond the profiled range is frozen: it
must conservatively clamp/fallback or add preregistered profile knots, never
silently extrapolate through a context-shift boundary.
The fixed DDTree B=16/32/64/128/256/512/1024 grid is the oracle/baseline; it is
not CaDDTree's policy. Production capacity must accept every adaptive physical
row count 1..1,025, including root-only n=0.

### Arm P: semi-autoregressive backbone plus sequential correction

Arm P is the bandwidth-efficient challenger:

- a block-parallel masked-token backbone emits H base future states from
  the same full target context and learned mask/position inputs;
- target-context visibility is causal, while masked draft slots may attend
  bidirectionally inside the block only when the frozen mask exposes no future
  target token or feature;
- a lightweight sequential correction conditions proposal i on d1 through
  d(i-1);
- rank 128, 256, and 512 Markov factors are compared as local corrections;
- pure Markov, gated, and GRU corrections are registered controls rather than
  conditionally added after looking at held-out outcomes;
- the zero-correction member is evaluated beside, but is not classified as,
  the actual Arm-D DFlash baseline; it is not itself enough to claim faithful
  diffusion training;
- the semantic backbone, not the low-rank term, carries language meaning.

This is the relevant idea in
[DSpark](https://arxiv.org/abs/2607.05147): use a parallel semantic backbone,
restore intra-block dependence with a cheap sequential module, and schedule
verification from actual prefix-survival confidence. The rank-256 factor is not
used as a complete replacement for Gemma's semantic output space.

The named DSpark source control has exact mechanics. With source gamma=7, its
one backbone input is the selected anchor plus gamma-1 mask rows; the anchor
row itself emits the base logits for proposal one. This differs from DFlash's
anchor-plus-gamma-mask layout and must have its own row/position golden. The
default Markov correction for depth k is the rank-256 token-conditioned bias
`W1[token_previous] * W2` added to shared base logits U_k. Its RNN ablation is
the source cell—not a GRU shorthand. With `s_0=0` and token vector
`e_{k-1}=W1[x_{k-1}]`, form the ordered concatenation
`z_k=[s_{k-1};e_{k-1};h_k]`. One source projection is split into
`W_g/W_c/W_o`; compute `g_k=sigmoid(W_g z_k)`,
`shat_k=tanh(W_c z_k)`,
`s_k=g_k*s_{k-1}+(1-g_k)*shat_k`, and correction
`b_k=W2^T tanh(W_o z_k)`. Bind every matrix orientation, bias, activation, and
precision from a pinned source implementation before packing.

Its online confidence scheduler is equally normative. For depth k compute
`c_k=sigmoid(w^T[h_k;e_{k-1}])` and train toward the soft label
`c*_k=1-0.5*||p_draft,k-p_target,k||_1`. Fit one temperature scalar per depth
sequentially from shallow to deep because scheduling uses cumulative survival
`a_{r,j}=product_{i<=j} c_{r,i}`. At runtime globally sort candidate tile
extensions by descending a with a frozen total tie key; initialize verifier
batch B and launch count tau to the source's initial R, score `tau*SPS(B)`,
admit candidates one by one, and stop at the first non-improvement. Freeze the
exact SPS receipt and source initialization in Phase 0. The paper's offline
source-control experiment disables this scheduler; keep that row distinct from
the online source-faithful schedule rather than averaging them.

The faithful upstream fixture is a five-layer gamma7 DFlash/DSpark model,
default Markov head, trained for ten epochs on approximately 1.3M
target-regenerated non-thinking Open-PerfectBlend examples with the source
CE/total-variation/confidence objective weights `0.1/0.9/1.0`. Reproduce its
chain logits and confidence schedule first. PCTree then changes only inference
construction over this same checkpoint. Rank128/512, GRU, other H, and Gemma
widths are declared transfers/ablations, not DSpark reproduction.
[DBlast](https://arxiv.org/abs/2608.05448) is an additional warning that
independent parallel positions are especially fragile under stochastic,
high-entropy output; its dependent proposal ideas belong in Arm P, while its
standard rejection-verification math does not replace this controller.

Test H=3/4/7/8/15/16/23, widths 512/768/1,024, and depths 1/2/3/5/7/8. Add a target-width
3,584, five-layer, H=15 DFlash control sharing the frozen embedding and full
target head, plus its eight-layer acceptance control. Add the published-shape
Domino control: target-width five-layer H=15 backbone, GRU width 1,024, and
rank-256 correction. A cell may be eliminated before full training only by the
registered optimistic-bound rule. Arm P can batch the semantic selector scan
for all H positions. Arm R has a new hidden state after each proposed token and
normally pays a selector pass per edge. No cost report may describe the latter
as one batched scan.

### Arm L: dependent latent-mixture block drafting

[DBlast](https://arxiv.org/abs/2608.05448) is not reduced to a generic CP loss.
Its one-pass proposal is

\[
q_{\mathrm{soft}}(d_{1:H}\mid F)=
\sum_{z=1}^{K}q(z\mid F)\prod_{i=1}^{H}q_i(d_i\mid F,z).
\]

The DFlash trunk emits width-d state h_i at each block position. Its category
expander is exactly a biased linear map with weight `[K*d,d]` and bias `[K*d]`;
reshape its category-major output to `[position,K,d]` and broadcast-add h_i to
obtain `h_{i,z}=h_i+g_z(h_i)`. The frozen target LM head is reused on every
`h_{i,z}`. A separate biased prior with weight `[K,d]` and bias `[K]` consumes
the declared clean-anchor row and emits q(z|F). This layout reproduces the
source parameter count `K*d^2+2*K*d+K`; a bias-free or differently packed
head is not equivalent. Bind orientation, anchor row, and category packing in
checkpoint/logit goldens. All K branches are evaluated in one packed pass. Register
K=1/2/4/8, with K=4 and H=15 as the primary source transfer, and preserve the
four mechanism controls `K1+NLL`, `K1+AL`, `K4+NLL`, and `K4+AL`.

The literature-fidelity controller is a separate, non-product control. For
every category z it computes the deterministic greedy branch
`hat_d_{i,z}=argmax q_i(.|F,z)`, samples one z from the temperature-calibrated
prior (or takes its argmax at category temperature zero), and verifies that
branch with the **same finite-mixture conditional that generated it**. After an
accepted prefix d_<i, let C_i contain categories whose greedy branches share
that prefix. Renormalize their prior mass and form the sparse proposal

\[
q_{\mathrm{greedy}}(v\mid F,d_{<i})=
\sum_{z\in C_i}\widetilde\pi_{i,z}
\mathbf 1\{v=\hat d_{i,z}\},
\]

whose support is at most K. Accept with `min(1,p(v)/q_greedy(v))`; on the first
rejection sample from normalized `[p-q_greedy]_+`; after a full match sample
the target bonus. This is distribution-lossless standard speculative sampling,
but it consumes proposal/acceptance randomness and deliberately does not
preserve this engine's existing seed-to-token mapping. It can reproduce the
published method, but cannot enter exact-seed product admission.

The exact-seed product transfer uses the same trained DBlast head without
pretending to reproduce that controller. Enumerate all K deterministic greedy
branches, stable-deduplicate them into a prefix trie, attach to each node the
sum of prior mass from categories still consistent with that prefix, and let
the ordinary authoritative target sampler walk the trie by exact token match.
No proposal or acceptance draw changes target authority. A separately labeled
single-category hard-match chain measures the cheaper alternative. The
all-category trie, single-category chain, mixture prior/expander/full-head work,
deduplication, target rows, and off-path discard are all charged; success of
either product transfer is evidence for that transfer, not for DBlast's
published rejection controller.

The source-exact training control starts from the official five-full-attention-
layer DFlash checkpoint, uses Tulu3 prompts with target-regenerated responses at
temperature 0.7/top-p 0.8/top-k 20, sequence length 2,048, H=15, one epoch,
Adam at `1e-4`, warmup ratio 0.04 then cosine decay, gradient max-norm 1.0, and
micro/global batch 4. The AL cell uses the paper's threshold-truncated,
all-prefix log surrogate with `tau=0.1` and stop-gradient prefix membership;
it is neither an unbiased expected-acceptance estimator nor a lower bound. For
a target-sampled continuation x define, for every prefix l,

\[
q_{1:l}=\sum_z\pi_z\prod_{i=1}^{l}q_i(x_i\mid z),\qquad
p_{1:l}=\prod_{i=1}^{l}p_i(x_i),\qquad r_l=q_{1:l}/p_{1:l},
\]

derive `q(x_i|x_<i)=q_{1:i}/q_{1:i-1}` from that same soft joint—not from a
greedy category—and set

\[
A_i=\min(1,p_i(x_i)/q(x_i\mid x_{<i})),\qquad
S_l=\sum_{i=1}^{l}\prod_{j=1}^{i}A_j.
\]

With `l_tau=min({l:r_l<tau} union {H})` and
`T_tau={1,...,l_tau}`, minimize the signed negative of
`sum_{l in T_tau}(log r_l+log S_l)`; only membership in T_tau is stop-gradient.
Target p_i comes from the previous-position target hidden/head and uses exactly
the same temperature/top-p/top-k filter that generated x. Implement mixture
products/conditionals in log space, bind extended-zero-support behavior and
any finite numerical floor to the pinned upstream fixture, and never insert an
unstated clamp. A per-position, greedy-mixture, or full-block-only AL term is a
different ablation. A
Gemma-trained instance is an architecture/algorithm transfer. Upstream Qwen
fixture reproduction must first match the published equations/checkpoint
behavior before the Gemma result can be attributed to the method.

### Arm T: distinct causal and corrected tree mechanisms

Arm T does not merge unlike mechanisms under “branch-conditioned.” It registers
PCTree, TreeFlash, JetSpec, DominoTree, DARTree, and a separately named full-
path extension with source-specific topology, score, budget convention, and
serialization. Every product member uses exact authoritative target-token
matching; a paper's rejection controller remains a labeled reproduction
control where applicable.

**PCTree.** One DSpark backbone pass emits B shared depth logits. At each of the
B sequential Markov stages, batch the current concrete parent frontier
(size one initially and at most k thereafter), add each parent's Markov bias to
the shared depth logits, take each parent's local top-k children, append them to
the all-node pool with cumulative conditional log scores, and retain the
current layer's global top-k as the next frontier. After all stages, globally
sort the pool **including the root** by score descending, depth ascending, then
stable node ID, and retain top N. Non-positive log increments put every ancestor
before its descendants, so the result is prefix-closed. Register source block/
maximum depths B=7 and B=16, main `k=4,N=32`, k=1/2/4/8 controls at N32, and
the complete source budget sweep `N={8,16,24,32,48,64,96,128}` at k4 for both
B7 and B16. In particular B16/k4/N64 is mandatory because it is the reported
source-sweep speed optimum; N32 remains the pre-specified main-table point, not
a substitute for the sweep. Also register the
parent-independent shared-Markov tree plus k=1 DSpark-chain mechanism controls.
Confidence scheduling is disabled in the source control. B is a draft depth;
N is a root-inclusive verifier-node budget. Neither may be substituted for the
other, and PCTree H=16 is a real registered proposer/verifier shape.

**TreeFlash.** This is immediate-predecessor conditioning, not arbitrary path-
state conditioning. Bind the exact approximator rather than implementing a
generic SwiGLU sketch. Let r_i be the unnormalized draft hidden,
`n_i=DraftFinalRMSNorm(r_i)`, and
`p_i=RMSNorm_eps1e-6(embedding(predecessor_i))`. The paper feeds
`c_i=[n_i;p_i]`, while pinned code feeds `c_i=[p_i;n_i]`; each named control
binds its own order and trained weight layout. Compute

\[
 (u_i,g_i)=\operatorname{split}(\operatorname{Linear}_{bias}
 (c_i;2f)),\quad
 \delta_i=\operatorname{Linear}_{no\ bias}
 (\operatorname{SiLU}(g_i)\odot u_i;h),\quad r'_i=r_i+\delta_i.
\]

The down projection is zero-initialized. Apply the drafter final RMSNorm again
to r'_i, then the frozen target LM head and full-vocabulary log-softmax at the
draft temperature. Freeze f, every bias, norm epsilon, matrix orientation, and
checkpoint bytes in an upstream-logit fixture. Depth one uses the root/target-
token embedding. Construction is exactly two-stage: build the
initial top-M-per-depth DFlash marginal tree, evaluate one root-conditioned
corrected distribution at depth one and M predecessor-conditioned
distributions at each depth 2 through gamma, and run OPT-Tree using those
corrected distributions. Here too paper and code have an off-by-one conflict:
the paper calls gamma/block size 16 and iterates q'1 through q'16, whereas the
pinned executable uses DFlash `block_size=16`, drops the anchor hidden row, and
emits H=15 corrected distributions. Semantically there are
`1+(15-1)*M`: 225 unique states at released M16 and 897 at M64. Register a separate H16/961
paper-description mechanism control and Gemma transfer. H16 may not substitute
for the executable H15 result, and neither receives an unqualified source-
reproduction label until a pinned checkpoint plus upstream tree/logit golden
resolves the horizon conflict; the concat-order conflict is retained in the
same resolution ledger.

Pinned code does **not** exploit that semantic deduplication. It constructs an
H-by-M tensor and broadcasts the root predecessor over all M depth-one lanes,
then executes normalization and the full target head for every physical row.
The released H15/M16 control therefore executes 240 rows, and an H15/M64 code
transfer executes 960; an analogous H16/M64 code-shaped transfer executes
1,024. The duplicate depth-one distributions are semantically identical but
their work/storage remain charged. A 225/897/961-row root-deduplicated kernel is
a valid optimized transfer only after exact logits/tree bytes are proven; it
cannot supply released-code timing. A popped prefix may
be expanded only if its terminal predecessor belongs to the initial depth's
top-M set; consequently every non-leaf final node is in the initial M-tree,
while final leaves may be outside it. The paper's B excludes the root, so
B=16/32/64/128/256/512 means rows 17/33/65/129/257/513. Pinned code instead
treats `tree_size` as root-inclusive, so the same numeric grid means
16/32/64/128/256/512 physical rows and 15/31/63/127/255/511 non-root nodes.
Register both typed conventions, with paper main B16/B64 and executable
`tree_size=64` benchmark reproduction, M=16/32/64, default M=16, and draft
temperature 1. Passing B+1 to code is a paper-algorithm transfer, not released-
benchmark reproduction. The paper's OPT-Tree keeps each corrected q' over full
V and may request sibling ranks through B. Pinned code instead truncates every
corrected distribution to `topk_candidates`, default 16, before building, and
its released benchmark exposes no override. Bind
`TreeFlashCorrectedCandidateDomain={PaperFullVLazy,ExecutableTopK}` in the
artifact. The exact released control is H15/M16/topK16/root-included
tree_size64; M32/M64, full-V code paths, changed tree_size, and crossed
paper/code fields are separately labeled mechanism transfers. Training teacher-forces the
ground-truth predecessor, uses forward KL and DFlash depth scaling factor 7,
starts from a pretrained DFlash trunk with a zero-initialized approximator,
jointly fine-tunes trunk and head for one epoch on 100K Nemotron-v2/
CodeAlpaca target-generated examples, effective batch 128, 128 anchors per
sample, sequence length 3,072, AdamW peak `1e-4`, 128 warmup steps, then cosine
decay. Frozen-trunk, linear, two-predecessor, CE, and no-scaling rows remain
explicit source ablations.

**JetSpec.** The paper and pinned executable disagree in two material ways, so
the ambiguous paper combination is not silently made executable. The paper
describes distributions r1 through r16, maximum draft depth 16, and later
marginals anchored to earlier argmax extensions, but does not specify a
discrete-feedback implementation that preserves its one-forward claim. At pinned
commit `2c7b3fa`, the primary one-pass input is instead the anchor followed by
`block_size-1` mask-token embeddings; `block_size=16` yields exactly 15 draft
logit rows and the engine enforces `tree_depth=15`. Under its causal head mask,
each later position attends to earlier **latent mask-position states**; no
earlier argmax token is embedded or fed back. The heap reuses each resulting
full-vocabulary depth distribution r_d for every node at depth d, so an
off-path branch receives a shared latent-depth distribution not conditioned on
its concrete parent token. Register three separately labeled runnable members:
(1) the pinned executable H15 latent-mask control; (2) a one-pass H16
latent-mask **extension** using anchor plus 16 masks; and (3) a sequential H16
discrete-argmax-feedback **diagnostic** that actually embeds each selected
argmax and charges all dependent steps. The latter two are not paper-faithful,
and the H16 member may not pad or duplicate the fifteenth executable logit. A
pinned checkpoint/config plus upstream golden trace must resolve the paper's
conditioning/depth contradiction before the paper combination becomes a
runnable source control or receives a source-reproduction label. None may be
silently upgraded to per-branch causal attention. Build each root-inclusive
tree with W=7 and best-first accumulated untempered log-probability (`T=1`),
using insertion-order heap ties exactly. Register root-inclusive
B=16/32/64/128/256 plus the distinct B=255 ablation wherever the member's
declared maximum depth permits the topology.

The Qwen3-8B source head concatenates target taps `{1,9,17,25,33}` of 36,
projects 5d to d=4,096 with a bias-free matrix plus RMSNorm, then uses five
Qwen-style decoder layers with 32 query heads, 8 KV heads, head dimension 128,
FFN 12,288, and injected target-context K/V at every layer. The executable
control binds `block_size=16/H=15`; the H16 latent extension binds
`block_size=17`. Use up to 512 random anchors per sequence, isolated
block-causal masks, micro-batch 2 on each of eight H100s, and target-regenerated
780K Nemotron-v2 examples (all code/math plus randomly sampled STEM/chat) plus
20K CodeAlpaca, with type-specific non-thinking chat templates. Training uses
forward KL multiplied by `T_KD^2`, no DFlash depth weighting in the main causal
control, and default learning rate `3e-4`. The source does not publish numeric
T_KD, global batch/accumulation, epoch count, optimizer/betas/weight decay, or
target-regeneration sampling knobs. Every one remains a blocked manifest field;
no value is inherited from DFlash or invented to obtain a “source-exact” label.

Gemma's JetSpec port is explicitly a target-matched transfer, not a Qwen shape
reproduction. It binds Gemma width/head/FFN geometry from the authenticated
model config and freezes the proportional zero-based tap map
`{1,11,20,29,39}` from the Qwen source taps into Gemma's 42 blocks before data
are opened. The upstream Qwen fixture remains the method-reproduction oracle;
the Gemma experiment tests whether the same mechanism transfers.

**DominoTree.** Reuse the frozen Domino B16 block backbone and rank-256/GRU
correction. Its recurrent cell is a single-layer **bias-free GRU** over frozen
target token embeddings. Before depth one, advance zero state with the carried
root token. For each concrete parent, compute
`delta_L=W2*SiLU(W1*[H_i;S_{i-1}])` at rank 256, take that depth's marginal
top-M token IDs once, gather both their base logits and corresponding W2 rows,
add the correction on exactly that M-slice, and log-softmax **over that slice**.
Select k children from those normalized scores; each selected child's frozen
token embedding advances its own GRU state. A full-V member streams the exact
full domain and normalizes over full V. Full-V normalization, zero root state,
a biased GRU, or DSpark's Markov/RNN equation is not DominoTree. Freeze matrix
orientations, ties, and checkpoint-byte/upstream-logit/tree fixtures.

The normative semantic oracle is sequential best-first: pop exactly one
highest-scoring prefix, apply the concrete-parent correction above, push its k
children, and repeat until B non-root nodes are selected. The production Metal
constructor must also implement the paper's host-synchronization-free depth-
synchronous form. At each of D depths, each of at most W active parents scores
its complete top-M slice (or streamed full V), selects k children, records the
resulting at most `W*k` generated children in a `[D,W*k]` ledger, then retains
the best W active prefixes. Use W=B to recover the sequential heap tree,
including exact ties via an insertion-order-compatible total key, and prove
exact node/parent/score/state bytes against the sequential CPU oracle. For
registered maximum B64/H15/k8/M256, bound W=64 active rows, 16,384 explicit
corrected slice scores per depth, 512 generated children per depth, 7,680
generated-child ledger entries, and 960 correction/GRU active-lane steps;
charge `sum_depth active_depth*M` or every full-V streamed score, never merely
W*k. The source point is B=16, M=64, k=8; register B=16/32/64 and
M=16/32/64/128/256/full-V. A per-pop host-synchronized Metal
implementation is diagnostic only and cannot close the architecture. Any
depth batching that changes the oracle tree is a new method, not an
optimization.

**DARTree.** Run one block-parallel backbone, take the base top-K candidate
menu at each of 16 depths, batch every active prefix and its path-specific
correction state depth-wise, and initialize depth-one recurrent state by
advancing zero state with the carried root token. For every parent, evaluate
the correction only on that depth's K base-logit candidates and log-softmax
over those K corrected values; full-V normalization is not equivalent. Freeze
matrix orientation, candidate order, root state, and upstream score/tree bytes.
The pruned variant retains the depth's global top-W corrected children,
accumulates the W-wide supertree for all 16 depths, and then selects the global
top-B non-root nodes by

\[
s_\beta(u_{1:d})=\sum_{i=1}^{d}\log\widetilde q_i(u_i\mid u_{<i})+\beta d.
\]

Require `beta<=0` and ancestor-first stable ties; then child score never exceeds
parent score and top-B is prefix-closed. The source pruned point is
K64/W12/B64/beta=-0.2. The fixed variant has beta=0, assigns the next depth
`ceil(remaining_nodes/remaining_depths)` nodes, and directly verifies that
balanced tree without global pruning; at B64/H16 this is four nodes per depth.
Register one-factor legal sweeps, not an accidental Cartesian product: vary
W=4/8/12/16/32 at B64, and vary B=32/64/128/192 at W12, with
`B<=16*W` for every pruned cell. Register the corresponding fixed balanced
B controls separately; W and beta=-0.2 do not leak into them. The primary
backbone is the pinned Domino checkpoint/bias-free GRU with K=64. DSpark-Markov
is a transfer control, not the primary backbone.

**FullPath-T extension.** Preserve the stronger hypothesis without leaving its
choices to implementation time. Cross four frozen first-stage scaffolds:
(A-code) the pinned JetSpec executable's shared latent-mask-depth W7 heap at
maximum depth 15, scaled only in node budget; (A-latent16) the explicit
one-pass H16 latent-mask extension; (A-argmax16) the dependent sequential
argmax-feedback diagnostic with all 16 calls charged; and (B) the
registered DFlash/DDTree independent-marginal top-M64 heap at maximum depth 16.
Each final-B member builds exactly one root-inclusive, prefix-closed scaffold
of 1,025 nodes with stable member-specific score/tie rules. The JetSpec-based scaffolds' B=1,025 are FullPath-only
extensions and are not attributed to JetSpec's published budget registry; an
unresolved paper mechanism is never inferred from either H16 diagnostic. Run a
second five-layer Gemma-shaped
tree-causal Transformer over **every** scaffold node. It uses target taps
`{1,11,20,29,39}`, bias-free 5d-to-d projection plus RMSNorm, Gemma Q/KV/head/
FFN geometry, injected target-context K/V at every layer, and an ancestor-only
mask so node u sees the actual root-to-u scaffold path and no sibling.

The parent output assigns the indexed full-head log probability to each
existing child token; accumulate real path-conditional log scores and select
the global top B non-root scaffold nodes with ancestor-first/stable-ID ties for
B=16/64/256. Since probabilities are at most one, this ordering is prefix-
closed. Candidate support is intentionally the first-stage scaffold; report
coverage separately for each scaffold so refinement quality cannot hide a
support miss. Train on the same target-regenerated 780K Nemotron-v2/20K
CodeAlpaca corpus, all materialized scaffold paths rather than only the greedy
path, three seeds, forward KL at frozen `T_KD=1`, with a hard-CE ablation. Bind
optimizer and remaining schedule in Phase 0 before traces are opened.

Use independent `FullPathSupertreeMax=1,025`,
`FullPathScaffoldEdgeMax=1,024`, and checked per-node state/mask/indexed-score
scratch. This member pays both proposer passes, every scaffold node, mask, K/V
row, indexed head score, and selection; a reduced scaffold or sampled-path
training cannot stand in for it. It is neither JetSpec nor TreeFlash
reproduction.

Published checkpoints and exact layer maps for these methods are Qwen-family.
For every method, distinguish (1) an upstream model/checkpoint fixture that
validates source equations and tree bytes from (2) the Gemma 2 architecture
transfer used for product selection. Failure of an unvalidated Gemma port may
not be reported as falsifying the published method; failure of the complete,
predeclared transfer still closes that exact registry member.

### Arm S: lossless target-integrated multi-stream control

The target-integrated family is a coequal candidate, not a fallback permitted
only after an external drafter loses. Implement the complete paper-described
lossless pipeline as a separately labeled Gemma paper transfer of
[Speculative Streaming](https://aclanthology.org/2025.emnlp-main.986/):

- the published primary cell is `Ns=4`, `gamma=4`, and `k=3`; the exact raw
  prior tree contains `1+sum_{j=1}^gamma k^j=121` main nodes including its
  carried root. It has four streams, four MSA layers, parallel stream-adapter
  rank 8, early-exit projection rank 8, the identity stream initializer used
  in all reported Section-3 experiments, and lossless weights
  `alpha0=0,alpha1=1`;
- run **every raw prior-tree node** through ordinary lower target layers
  `[0,N-Ns)` with tree-causal main-row attention before any pruning;
- at layer `N-Ns`, score each raw edge from its parent's lower-layer hidden
  state, prune failed edges and their entire descendant subtrees, and only then
  create streams for surviving main nodes;
- initialize stream j at surviving anchor u exactly at that split as
  `S^(N-Ns)_{u,j}=f_eta(M^(N-Ns)_u)+P_j`. The paper permits a rank-eta linear
  map in the equation but Appendix K.1 reports `f_eta=Identity` for every
  Section-3 experiment; the paper-primary transfer therefore uses identity
  plus the learned stream-ID/relative-position embedding. Stream rows never
  execute the lower `N-Ns` layers;
- execute the top `Ns` layers with lossless multi-stream attention (MSA), while
  preserving the main-stream target function. Main rows use the frozen base
  MLP; stream rows use the paper's parallel stream adapter; and
- in that same target invocation, verify the incoming tree and produce the
  per-surviving-anchor stream distributions from which the next raw tree can be
  issued after the authoritative accepted anchor is known.

The pruner order is a correctness requirement, not a scheduling preference.
For raw edge parent p -> child token c at the split layer, apply the trained
rank-8 early-exit projection `o_theta` to the **parent main row**, reuse the
frozen target LM head, normalize those early-exit logits, and use the probability
assigned to c as the transition score. A failed edge removes all descendants.
Define explicit typed states

`RawPriorTreeLease -> LowerLayerTreeState -> SurvivingPriorTree -> UpperMSAResult`.

Appendix K.1 further says that the **application-specific experiments** pass 32
tree nodes after pruning; Table 3 and its surrounding text identify those
experiments as **Shared** Speculative Streaming. This is not a published
lossless configuration and cannot be crossed into the exact-output product arm
under a source label. Preserve the literal source field as
`ReportedSharedTable3PostPruneDraftNodeCount=32` in a non-product Shared-SS
fixture.
The paper does not say whether 32 is a maximum or exact retained count, whether
the carried root is included, whether thresholding alone produced it, whether
a cap followed thresholding, or how an over-budget tree was scored, made
prefix-closed, and tie-broken. Run each still-plausible complete interpretation
as a separately tagged **Shared source-resolution fixture**. A lossless
32-node run is permitted only as a local cross-mode cost/accuracy ablation and
is never evidence for source lossless fidelity.

For lossless mode, the paper also does not publish the numeric threshold, a
more precise normalization recipe than normalized early-exit logits, or the
exact parallel-adapter algebra/sharing. The released code omits the early-exit
pruner and cannot disambiguate these fields. Therefore a source-exact
reproduction label remains blocked until a pinned checkpoint/config or author
golden supplies them.

The Phase-0 registry consequently separates: (1) the lossless paper-primary
geometry transfer with `f_eta=Identity` and `EarlyExitRank=8`; (2) its
threshold-only lossless Gemma transfers with every numeric threshold and
survivor behavior frozen on development data but no invented source cap; (3)
explicitly labeled learned-initializer and lossless-32 cross-mode ablations;
and (4) the non-product Shared-SS 32-passed-node source-resolution fixtures.
Freeze adapter algebra/sharing, normalization, threshold sweep, any
over-budget prefix-closed selection/score/tie/NaN behavior, and the selected
threshold. Never report those local transfer choices as published values, and
never use an easier 32-node truncation as a stand-in for the unresolved
lossless pruning rule.

Register a separate released-code architecture control pinned at official
commit `f5ad87edde0d52cb43d9c148a352297c04b4dc66`: its bias-free SwiGLU
stream adapter projects hidden-to-rank with gate/up branches then rank-to-hidden
with a down projection, installs adapters at all decoder layers, and creates
streams from special token embeddings at input rather than at an N-Ns split.
Bind one explicit r from the conflicting code/config defaults and the README's
recommended r=8 in the source receipt. Its checked-in training command runs
three epochs at per-device batch two with stream-adapter rank 8, while the
trainer actually constructs PyTorch AdamW and defaults to BF16, learning rate
`2e-5`, cosine scheduling, warmup ratio 0.1, weight decay 0.01, and three
streams unless overridden, with maximum gradient norm 1.0. This conflicts with Appendix K.1's AdamQ/`5e-4`/
linear description and is a separate receipt, never a disambiguation. The
pinned Ns4 path is itself not a
byte-executable oracle: its position-ID helper defaults to three streams while
the mask builder uses configured Ns=4. Register either the literal internally
inconsistent failure fixture plus a separately labeled Ns3 code control, or an
**intended-code repair** that passes Ns4 to both helpers and is validated by an
independent mathematical mask/RoPE golden. Never call the repair byte-exact
released behavior. This control is executable evidence but is not the paper
architecture. It has no split-layer `f_eta`, early-exit pruner, or Shared
Table-3 32-node implementation; its checked-in rank-8 training flag is the
stream-adapter rank and cannot resolve any of those paper fields.

The lossless 2-D attention mask is exact. Let Anc(u) include main-tree node u
and its root-to-u main ancestors. In every upper MSA layer:

- main query u sees canonical cached main K/V plus Anc(u), and no stream K/V;
- stream query (u,j) sees the same canonical/main path plus streams
  `(u,1..j)` at **that identical anchor only**;
- neither query sees a sibling main subtree, another anchor's streams, nor a
  future same-anchor stream.

The paper specifies the visibility relation but not RoPE IDs. Type
`StreamPositionContract={ReleasedCodeShiftedRoPE,PaperTransferFrozen}`.
Released code—and the independently corrected intended-code fixture—uses main
`past_length+depth(u)` and stream `(u,j)` at
`past_length+depth(u)+j`; siblings at equal depth share a position. The Gemma
paper-transfer position rule is frozen before data and remains a transfer field,
not a published fact, until an author checkpoint/golden resolves it. Main K/V
is persistent. Same-anchor earlier-stream K/V required by
the equation exists transiently inside each upper-layer invocation, but no
stream K/V survives the invocation. Poison tests must mutate siblings, other
anchors, and future streams independently and prove no forbidden dependency;
mutating a legal same-anchor earlier stream must affect its allowed successors.

All raw lower-layer main K/V and surviving upper-layer main K/V for off-path
nodes live in the node-indexed speculative target arena, never the canonical
ring. Only the walked root and accepted path are scattered/retained. Lazy
backtracking of lower-layer raw-tree storage is allowed only behind the same
transaction: it may remain until next-tree generation completes, but it cannot
become visible canonical state.

The overlap path applies Gemma's final normalization, logit softcap, and real
full 256K target head to gamma terminal stream states for **every survivor**
before the accepted anchor is known, performs stable top-k on device, and keeps
only compact per-stream menus. It tiles the survivor-by-gamma head rows, reuses
head-weight tiles, and never materializes or reads back the dense logit matrix.
For the gamma4/k3 cell, lower-layer work is exactly 121 raw main rows. A
lossless threshold-only transfer can execute as many as 121 upper main plus
`121*4=484` stream rows; pruning may reduce only upper work, never the 121 lower
rows. The non-product Shared-SS source-resolution fixture separately records
32 passed nodes without silently treating that statement as a lossless cap or
assuming its root convention. Report all work counts separately. A post-walk
variant may score only the chosen anchor's gamma streams in a second submission,
but it pays the wait/submission and is not evidence for verify-and-speculate
overlap.

Do not cross independent maxima and call the result published fidelity. The
finite Arm-S registry contains:

- the primary exact raw geometry `(Ns,gamma,k)=(4,4,3)`;
- threshold-only lossless controls and separately named lossless-32 cross-mode
  ablations;
- non-product Shared-SS 32-passed-node source-resolution fixtures, with every
  unresolved complete threshold/cap/root interpretation separately tagged;
- `Ns=2/4/8` transfer ablations at the same gamma4/k3 raw tree;
- exact smaller branching controls gamma3/k3 and gamma4/k1/k2; and
- exact deep-stream chain transfers gamma7/15/23 with k1.

For every entry, `RawPriorNodeCount=1+sum_{j=1}^gamma k^j` is computed with
checked integer arithmetic and all raw nodes really traverse the lower layers.
Any high-gamma/k>1 capped, beam-pruned, or adaptive pre-tree is a new explicitly
named constructor with its exact pre-cap semantics; it cannot close Speculative
Streaming. Impossible exponential crosses such as gamma23/k8 are not silently
represented by a 256-row buffer: they require a valid implementation-independent
cost certificate or remain untested, and are not part of the claimed source
registry.

Keep independent safety ceilings `StreamRawPriorNodeMax=256`,
`StreamSurvivorMax=256`, `StreamHmax=23`, and
`StreamTransitionBranchMax=8`. Derived checked ceilings are
`StreamUpperRowMax=StreamSurvivorMax*(StreamHmax+1)=6,144`,
`SNextNodeMax=65,536`, and `SRawChildMax=47,104`; actual allocations use the
artifact-declared legal pair, not the Cartesian maximum. The non-product Shared
source fixture binds the literal
`ReportedSharedTable3PostPruneDraftNodeCount=32`, not a claimed survivor cap;
its task-specific selected gamma/k and exact cost remain unresolved source
fields. In the separately labeled lossless gamma4/k3-plus-32 cross-mode
diagnostic, the two root conventions have 32 or 33 anchors, at most
`32*121=3,872` or `33*121=3,993` compact next-tree nodes, and
`32*4*3=384` or `33*4*3=396` raw top-k entries. The lossless threshold-only
transfer retains independent safety bounds of
`121*121=14,641` compact nodes and `121*4*3=1,452` raw top-k entries. These
arrays are separate from the external forest's 1,025 verifier rows and output
frontier.

Arm S is a pipeline, so it has an explicit cross-invocation protocol. The final
prefill target row cannot yet publish a rooted prior tree: x0, its adopted
sampler checkpoint, and its post-x0 history do not exist until authoritative
selection and callback delivery. Prefill and an ordinary MSA rebootstrap
therefore return a `PendingStreamBootstrap` bound to the pre-x0 state, target-
plan epoch, main-stream row, and candidate future-tree arrays. Event one
samples/adopts x0; only after its callback continues and eligibility is known
does the controller bind x0/S0/post-history into a `PriorTreeLease`, otherwise
it discards the pending bootstrap.

The resulting `RawPriorTreeLease` is tied to the target-plan epoch,
root sampler/history fingerprint, main-stream row, and future-tree node arrays.
Each steady invocation consumes that lease, executes/prunes/verifies its raw
tree, while MSA produces bounded candidate menus for every surviving node that could emit
the authoritative correction/next-anchor token. After target sampling walks
one path and all preceding callbacks continue, the controller couples that
row's sampled correction carry and sampler checkpoint to **that same row's**
next tree; it does not require the correction to have been an input token or
insert an extra target pass. The next steady invocation inputs/caches the carry
while verifying its attached tree. This is the published verify-and-speculate
overlap.

For a full-match row whose main correction has not yet been sampled, retain one
`PendingTargetCandidateWithTree` until the normal next-origin sample, then bind
the selected correction and checkpoint to that row's next tree. Every off-path
set is discarded. A missing/incompatible lease or an intentional route switch
may require an ordinary MSA bootstrap/drain; an endpoint caches the correction
without publishing unused future work. Stop, callback stop/throw, rollback,
reset, regeneration, context rebuild, and poison discard every unpublished
lease.

The target-integrated ticket therefore owns the prior lease, verified target
rows, transient speculative streams, candidate `NextTreeSet`, target-selection
workspace, and target K/V transaction as one unit. Its completion result
contains bounded prior-tree verification outputs and bounded next-tree handles;
no speculative tree becomes session state merely because a GPU command
completed. Prefill bootstrap, steady verify-plus-emit, ordinary drain/rebase,
and shutdown are separate controller states with exhaustive transition tests.

The paper-described literature controller is `Ns=4`, `gamma=4`, transition
branching `k=3`, with a virtual-draft-distribution rejection/residual sampler.
Appendix E does not fully specify multi-path top-k acceptance order, RNG mapping,
or residual/path tie semantics, and released code instead uses target argmax
plus equality tree walking. Until a pinned author golden resolves this, freeze
each complete interpretation as a labeled transfer rather than claiming a
source-exact controller. A correctly specified rejection interpretation is
distribution-lossless but does not preserve this engine's
existing seed-to-token mapping, so it is a research/reproduction control and
cannot enter same-seed product admission. Product members use exact target-
token match under the current authoritative sampler. Report the distinction,
and scope any Arm-S family conclusion to this stricter exact-seed product
contract; never call the hard-match port a reproduction of the paper's
acceptance controller.

A separate top-layer suffix drafter run before target verification is not this
architecture: it destroys the defining overlap between verification and next-
draft generation. The shared mode is excluded from the exact-output product
claim because it lets speculative streams modify the main stream; it may be a
quality experiment, but cannot satisfy the retained seeded-target-token oracle.
Published speedups on other models are motivation only. The source model is
Qwen-family, so first validate mask/position/cache bytes against the pinned
official implementation, then treat Gemma as an architecture transfer. The
official release is useful evidence for the 2-D mask, positions, and accepted-
main-cache selection, but is not a full oracle: it omits the paper's pruner and
does not implement the paper's lower-MHA/split-birth pipeline exactly. The
engine must implement the paper pipeline rather than wrap or imitate that
incomplete runtime. Arm S must pass the same bit-exact target, cache, callback,
resource, and live timing gates here.

### Finite candidate registry and elimination

Before architecture-selection validation data are opened, publish a
content-addressed finite registry. Each entry fixes family,
root-only/full-context feature mode,
tap set, width, depth/cell/correction, curriculum, training seed, precision,
selector type and d-prime, shortlist policy, proposal sampler, eager/conditional
draft schedule, route, and verifier topology. Every declared Arm R/D/P/L/T/S
capacity, horizon, feature, branch, loss, seed, and vocabulary/precision
control—and every predeclared heterogeneous combination—must appear either as
a direct entry or with an entry-specific mathematical exclusion certificate.
The registry explicitly marks which axes are crossed and which published-
fidelity controls are standalone; an omitted factorial interaction is an
unclosed member, not an implicit negative result.

A timing interval from an implementation can eliminate only that exact code,
artifact, topology, and shape digest. It cannot prove that a faster
implementation or larger member of the architecture family is impossible.
Family/capacity exclusion requires one of two stronger forms:

1. an implementation-independent lower bound on unavoidable bytes, MAC/FLOPs,
   serial dependency depth, submissions, and synchronization for the claimed
   family member, divided by rigorously measured **upper** ceilings on
   achievable M4 throughput/bandwidth and combined with a simultaneous upper
   bound on useful accepted progress; or
2. a complete, production-correct implementation frontier covering every
   registered realization of that member, with simultaneous live intervals.

The favorable bound must include maintenance, fallback, endpoint, callback,
tree construction, selection, and thermal cost, and must still fall below the
decision threshold. A slow implementation, smallest/largest scaling trend,
teacher loss, shortlist recall, parameter count, isolated kernel time, the old
width-four kernel, or a 512-window memorization check is not such a certificate.
An untested interaction is not inferred away from marginal trends.

The 512-window balanced set remains an algebra/overfit diagnostic. Registered
members train to a frozen convergence rule: maximum steps, evaluation cadence,
early-stop patience and minimum delta, optimizer schedule, and restart policy
are fixed before training, and all three seed outcomes are retained. Quantized
members include both post-training quantization and packed-aware/QAT fine-tuning
from the same generalized F16 checkpoint. A PTQ loss cannot reject a precision
or family that QAT recovers. A larger or slower member survives whenever its
actual joint prefix distribution can repay its complete live cost.

### Direct distribution training

Train token distributions, not hidden-state regression and not only the one
sampled label.

For each reached edge, retain the exact target prepared distribution under the
bound sampler contract and optimize a predeclared combination of:

- forward target-to-draft KL or cross-entropy on the exact prepared target
  support;
- direct token classification;
- a common-draw/CDF-alignment objective for the sampled proposal arm;
- calibrated conditional prefix-survival loss;
- an optional weak future-state auxiliary whose zero-weight arm is mandatory.

At every H, cross uniform position weights and DFlash's frozen exponential
early-position weighting with two deliberately distinct D-PACE-derived
controls. The faithful
[D-PACE](https://arxiv.org/abs/2605.18810) control uses the target-policy token
confidence `q_i`, smooths only inside the detached weight with
`q_tilde_i=(1-alpha)q_i+alpha`, computes
`w_j=stop_gradient(sum_(m=j)^B product_(i=1)^m q_tilde_i)`, and multiplies that
weight by the **unsmoothed** `-log q_j`. The generalized product control derives
weights from this engine's expected useful prefix after stop/budget/callback
survival and is labeled `controller-PACE`; it is not reported as reproduction
evidence for D-PACE.

Likewise, an entropy-order-only curriculum is retained only as an ablation.
The faithful [EDSD](https://aclanthology.org/2026.acl-long.2145/) control bundles
all four published mechanisms: Mask-p_t with target-output-entropy-ranked
easy-to-hard token exposure, Step-n_t with the training-time rollout horizon
increasing from one to the registered maximum, EDSelect, and EDFuse. EDSelect
computes matrix-entropy-derived Curvature and LiDAR for every Gemma layer on a
development-only calibration subset, chooses the best two model-specific
feature sources, and freezes them before architecture selection. EDFuse uses
the paper's token-dominant two-stream path: gated-SiLU target features plus
normalized/up-projected token features, aggregation, and down-projection, with
both hidden-width and intermediate-width controls. Report full EDSD/EDArch,
EDTrain-only, EDSelect/EDFuse ablations, and the entropy-only curriculum
separately. None changes target verification at inference.

Use full-H backpropagation and compare two faithful token-prefix curricula:

1. target-token prefixes with self-rolled latent states, plus a
   [Domino](https://arxiv.org/abs/2605.29707)-style base-backbone anchor whose
   weight decays from one to zero so the sequential correction cannot bypass
   the semantic model;
2. deployment-style own-token rollout with losses and confidence labels masked
   after the first mismatch, because deeper work is not reached by the
   controller.

Teacher-path loss may remain as an explicitly weighted auxiliary in the second
curriculum. Report teacher-prefix and deployment-prefix results separately.
EAGLE-style training-time testing justifies self-rolled latent states; it does
not prove that every token-prefix curriculum must use the drafter's own token.
Do not close Arm P without comparing both.

For Arm S, retain a paper-described Section-3 lossless Gemma-transfer control:
freeze every base Gemma parameter, set the main-stream next-token loss weight
to zero, set `f_eta=Identity`, train only stream identifiers/adapters on all
`gamma` future-token losses, and train the rank-8 early-exit pruning projection
on its declared next-token objective. It has no invented 32-node cap.
Learned-initializer, lossless-32 cross-mode, threshold-only/uncapped, D-PACE,
and entropy-weighted Arm-S variants are separately labeled members. The Shared
Table-3 32-node fixture is non-product and does not inherit Section-3 identity
or optimizer fields unless another pinned source proves them. Segmented
attention may reduce training memory only if it preserves the paper's stream
visibility exactly; do not label any transfer choice as the published
objective or runtime rule.

Appendix K.1 records BF16 training, “AdamQ,” learning rate `5e-4`, and a linear
scheduler specifically for its Section-3 experiments. Preserve those literal
fields in that source receipt. If neither the paper nor pinned code
defines AdamQ's update, betas, epsilon, weight decay, warmup, and step-count
semantics, source-training reproduction remains blocked rather than silently
substituting AdamW. Register the fully specified Gemma optimizer recipe as a
transfer and compare it with any later source-resolved recipe.

For all explicitly non-source optimizer controls, use the registry's frozen
AdamW, global-norm clipping, warmup plus decay, deterministic data order, three
model seeds, and convergence/overfit diagnostics. A target feature or token
from the future must be poisonable without changing the pre-loss draft rollout.
Changing d_i must change the registered causal state or correction used for
d(i+1).

### Dynamic draft vocabulary

The drafter head is:

~~~text
draft semantic state
  -> learned down-projection
  -> approximate full-vocabulary selector
  -> stable top-k candidate IDs
  -> indexed dot against the artifact-bound selected-token draft table
  -> selected-token sequential correction
  -> exact frozen proposal transforms
  -> deterministic or common-draw proposal
~~~

If the semantic state is narrower than 3,584, a learned up-projection produces
the selected-table input space before indexed scoring. The primary Q8 member
aliases the existing resident target-Q8 columns and their quantization
semantics. F16/Q6/Q4 controls own deterministic dequantized/repacked copies
derived from the same bound target output/embedding payload, with separate
tensor digests, kernels, resident-byte charges, and startup traffic. They never
replace the production-Q8 table used by authoritative target selection.
“Exact selected column” means exact for that registered draft-table format; it
does not claim that the learned state is an exact target hidden state.

Register both reuse of the frozen target F16 token-embedding vector followed by
a learned narrow projection and an independently learned full-vocabulary
semantic-embedding control. The latter may be excluded only by the registered
resident-traffic/lower-cost optimistic bound, not by implementation
convenience.

The selector is trained jointly from the actual deployed drafter states. A
target-hidden selector is only an oracle diagnostic. Compare at least:

- static target-generated vocabulary;
- learned SpecVocab-style projection;
- clustered or routed DynaSpec-style vocabulary;
- selector bottleneck d-prime in 112, 224, and 448 (1/32, 1/16, and 1/8 of
  Gemma's hidden width), subject only to the registered exclusion rule;
- k in 1K, 2K, 4K, 8K, 16K, 32K, 64K, and 128K;
- an adaptive k selected from measured benefit;
- full-vocabulary draft scoring as the quality ceiling and cost control.

Cross selector bottleneck and shortlist explicitly in the finite registry. A
cell can be skipped only with its own optimistic dominance certificate; do not
silently fix one axis while claiming to test the other. Include selector,
target-space up-projection, and selected-head precision in the F16/Q8/Q6/Q4
registry, including each non-Q8 table's actual memory and traffic. No format is
represented by quantizing only a tiny correction while leaving its claimed
selected head in another precision.

The candidate union must include repetition-history IDs, stop/control IDs,
retrieval-forced IDs, and any predeclared transition candidates. Deduplication
and ties are stable by token ID. For Arm P, explicitly account for tokens that a
Markov correction can promote outside the parallel semantic shortlist:

- either add a measured transition candidate index and train/evaluate the union;
- or score the corrected full vocabulary for the control;
- never report the uncorrected shortlist's recall as the corrected drafter's
  recall.

Proposal sampling is non-authoritative but still reproducible. Define and bind
in the artifact one GPU-compatible arithmetic contract for proposal logits,
exp/normalization, stable order, ticket mapping, and boundary fallback; the CPU
golden executor must emulate that contract exactly. Do not claim that Metal F32
is bit-identical to the authoritative CPU-double TokenSampler. Any conservative
near-boundary fallback affects only which proposal is attempted. The target
compact sampler remains on the existing CPU-double path and remains the sole
authority for selected tokens.

A shortlist miss reduces performance but cannot change target correctness.
Failure of 2K and 4K advances to wider/adaptive/full controls; it does not close
the semantic drafter. SpecVocab's published k is a result on different models
and accelerators, not a constant for Gemma 2 on M4.

Define the training loss on a hard shortlist without mathematical sleight of
hand. KL(p||q) is infinite when p assigns mass to a token omitted by q.
Therefore:

- train the approximate full-vocabulary selector with an explicit
  full-vocabulary, sampled-softmax, or formally normalized contrastive
  objective and a separately reported miss penalty;
- apply selected-head distribution distillation only conditional on target
  support coverage, while counting every omitted target-support token in the
  all-opportunity metrics;
- never inject teacher-positive IDs into a training candidate set unless the
  same causal mechanism can supply them at inference.

### Retained retrieval and routing

Retrieval remains because it is nearly free and already profitable. Freeze and
compare two real policies on development data:

1. retrieval-first: use the retained one-edge Stage 4 route when profitable and
   run the learned bounded-H route only at residual eligible origins;
2. retrieval-anchored: use a confident retrieved d1 and let the learned model
   extend d2 through dH through the same target-conditioned state.

Do not pretend the existing one-token RetrievalProposal already supplies a
multi-token block. Both policies need explicit controller states and metrics.

Also register co-resident heterogeneous routers among specialized Arm R/D/P/L/T
artifacts and, where its fused target ownership is compatible, Arm S. A
`HeterogeneousSpeculationBundle` binds the ordered child-artifact digests,
shared target/model/sampler/feature-bank compatibility, router tensors and
configuration, route precedence, the resolved horizon/topology per child, and
the sum of all resident allocations. `PreparedGemma2Plan` validates the whole
bundle and session construction uploads/publishes it atomically; a missing,
malformed, incompatible, or
partially-uploaded requested child is setup failure, never a silent route
drop. Metrics identify the bundle, chosen child, and full residency as well as
executed-arm, eager inactive-child maintenance, lazy replay/bootstrap, scout,
and Arm-S pipeline traffic.

At runtime, only a validated causal **pre-submit** no-route/decline may fall
back to Stage 4. Once any target or draft work has been submitted, the joint
ticket rules below apply; uncertain completion poisons the entire bundle, and
fallback is legal only after verified atomic rollback. This tests AngelSpec's
workload heterogeneity without leaking dataset labels: choosing one universal
drafter cannot reject the routed architecture.

Confidence predicts actual conditional target-token matches and expected saved
wall time at reached origins. The deployed router may consume only causally
available production state: proposal scores/features, route, logical
position/context, remaining budget, sampler options, and retrieval state.
Dataset, source, split, benchmark, and oracle workload labels are evaluation
metadata and never enter the model or router. Workload-stratified calibration
is diagnostic only. Add a poison test proving arbitrary evaluation-metadata
changes cannot alter proposals or routing. A decline remains in the
all-opportunity denominator. The route threshold is chosen from measured
target/draft costs and joint outcomes, not from a fixed confidence percentage.

Pre-routing and scouting are distinct registered policies. A zero-scout router
may choose an arm only from shared already-resident causal target/session fields;
it cannot inspect an arm-specific proposal score that has not been computed.
A scout router may execute one or more bounded arm-specific prefixes and use
their scores, but its complete scout kernels, reads, barriers, and abandoned
work are charged on every routed opportunity. Arm S can be immediately
selectable only when the preceding target invocation paid its MSA/tree-emission
maintenance; otherwise routing to it first pays the ordinary MSA bootstrap and
cannot claim that origin's speculative saving. Full bundle residency is always
charged as a resource constraint, even when only one arm executes deeply.

The artifact has an exhaustive `CalibratablePolicyParameters` schema. Its
fixed-dimensional, bounded fields are only the preregistered confidence-map
parameters and numeric cutoffs for: arm/no-route choice, adaptive **draft
shortlist** k among already implemented sizes, dynamic-horizon stopping among
already implemented edges, conditional tile launch, fixed-shape tree pruning,
and Arm-S early-exit pruning. The fitting loss, constraints, causal inputs,
tie/priority rule, default on empty cells, and whether parameters are shared or
context-bucketed are fixed in Phase 0. Evaluation workload/source labels are
never legal buckets.

Family, weights, features/taps, Ns/gamma, H set, selector algorithm/d-prime,
available shortlist sizes, tree branch factor/node cap/constructor, pruning
algorithm/insertion layer, recurrence/correction, precision/packing, cache
policy, target topology, command schedule, and router feature set/order are
architecture fields and cannot be refit in Phase 9. Any numeric or categorical
choice absent from the calibratable schema is architecture-frozen. Architecture
and policy have distinct digests.

There are two explicitly separated fits. Before architecture selection, each
registry entry receives a **provisional development policy** by applying its
preregistered fitting rule only to training/architecture-development live
outcomes; freeze that policy before the untouched selection partition is
opened. Phase 8 measures those frozen policies and never tunes a threshold.
After one architecture is selected, Phase 9 refits only that architecture's
final confidence/tile/router thresholds on the calibration partition. The
provisional and final parameters have distinct digests and labels.

## Production controller and transaction design

Implement the learned controller as a pure state machine used by production,
replay, and exhaustive tests. Do not extend the current one-edge conditional
tree of local variables.

Its result type must identify:

- eligibility and proposal source;
- a bounded tagged `ProposalGraph` plus the linear d1-through-dH view when its
  tag is chain;
- verifier topology, draft schedule, and bounded per-tile/branch decisions;
- accepted edge count 0 through H;
- rows/nodes retained in each target call;
- carry token and its sampler checkpoint;
- next candidate and owning `TargetStateLease` source (with a borrow-only
  `TargetStateView` during completion);
- optional consumed `PriorTreeLease`, selected-path node sequence, and
  publish/discard/rebase action for an Arm S `NextTreeSet`, including the
  full-match `PendingTargetCandidateWithTree` case;
- endpoint/fallback action and every skipped/declined reason, including the
  H=3 `second_call_confidence_decline` instance;
- every incurred command, wait, selection, restoration, and route cost.

`ProposalGraph` is the topology-neutral external-arm contract. It fixes
`kind={chain,forest,repeated_root_beam_batch}`, horizon, materialized node
count, proposal-edge count, root count/indices, and fixed-capacity structure-of-arrays for token, parent, optional child-
node index, depth, edge/conditional-prefix confidence, path-history descriptor,
proposal-ticket offset, feature provenance, and device-arena epoch. For chain/
forest, node zero is the one current authoritative root. The ReDrafter MLX tag
instead carries 1..4 byte-identical root occurrences with isolated beam masks
and one shared authoritative root identity. An edge may lead to another
materialized node or be an output-only frontier proposal. The completion result returns the
selected node/edge path, never only an accepted integer. Arm-S tree leases use
their separate pipelined envelope but embed the same validated graph
description for each prior/next tree.

### Exact event order

The numbered trace below is the H=3 external-chain bring-up for Arms R/D/P/L/T.
Arm S uses the same authoritative per-token/callback rules through the forest
walker plus its prior-tree/next-tree states; it is not forced through a
pre-target drafter call.

1. Sample and observe x0 through the ordinary authoritative sampler; x0 is
   selected and S0 is the resulting checkpoint.
2. Before drafting, apply the ordinary endpoint and eligibility rules. If x0
   is a stop token or exhausts the selected-token budget, run ordinary K/V-only
   x0, adopt S0, then stop without a callback for a stop token or invoke the
   normal callback for a non-stop budget endpoint. If call one cannot fit,
   follow the ordinary logits-producing decode/context policy. Do not draft or
   execute call one in any of these cases.
3. Obtain the registered eager or conditional subset of d1 through d3 without
   mutating target sampler/history, then execute call one on [x0,d1].
4. Sample x1 from row zero on a fork of S0, producing the provisional S1.
5. If x1 differs from d1, retain x0 and resolve x0's callback. A stop or throw
   discards x1 and S1: x1 was only a preview and receives no K/V write. Only if
   x0 continues does x1 become authoritative and S1 become adopted. Queue it
   as a carry when nonterminal with budget remaining; otherwise process x1
   through ordinary K/V-only, omit its callback when it is a stop token, and
   invoke its callback before a non-stop budget termination.
6. If x1 equals d1, provisionally retain [x0,d1] and resolve x0's callback. A
   stop or throw there restores d1, retains only x0, discards S1, and leaves S0
   authoritative.
7. After x0 continues, select x1 and adopt S1. If x1 is a stop, finalize two
   rows without invoking its normal callback. Otherwise resolve x1's callback;
   a stop or throw finalizes two rows with S1. If the budget ends at x1,
   finalize two rows after the callback and do not sample x2.
8. Otherwise sample x2 from row one on a fork of S1, producing S2. All earlier
   callbacks have continued, so x2 immediately becomes authoritative: append it
   to selected/history state, adopt S2, and update retrieval/learned history
   exactly once before the mismatch, endpoint, gate, or deferred-d3 branch.
9. If x2 differs from d2, finalize both call-one rows. Because all prior
   callbacks continued, x2 is already authoritative under S2. Queue x2 as a
   carry when nonterminal with budget remaining; otherwise run ordinary
   K/V-only x2 and apply the same stop/no-callback or budget/callback rule.
10. If x2 equals d2, x2 is authoritative but not cached. If it is a stop or
    exhausts the budget, finalize call one, write x2 through ordinary K/V-only,
    retain the already adopted S2, omit a stop-token callback, and invoke the
    normal callback for a non-stop budget endpoint.
11. If continuation remains possible, evaluate the frozen causal second-call
    gate. On confidence decline or width-two context failure, finalize call
    one only after transactionally promoting call-one row one; queue x2 once
    with its already adopted S2 and that TargetStateLease. Do not resample or
    re-observe it. Process x2 through the
    ordinary logits-producing decode/context-policy path before its callback;
    use that ordinary row-zero output as the candidate/feature anchor for x3.
12. On gate acceptance, finalize call one completely, generate d3 now if the
    registered draft schedule deferred it, and execute call two at the next
    target position on [x2,d3]. There is never more than one live verifier
    backup transaction in the conditional topology.
13. Sample x3 from call-two row zero on a fork of S2, producing provisional S3.
14. If x3 differs from d3, retain x2, restore d3, and resolve x2's callback. A
    stop or throw discards x3/S3 without a K/V write. Only after x2 continues
    does x3 become authoritative and S3 become adopted. Queue it as a carry
    when nonterminal with budget remaining; otherwise ordinary-K/V-only cache
    x3 and apply the stop/no-callback or budget/callback rule.
15. If x3 equals d3, provisionally retain [x2,d3] and resolve x2's callback
    first. A stop or throw restores d3 and keeps S2. After x2 continues, select
    x3 and adopt S3. A stop token finalizes both rows without a callback;
    otherwise invoke x3's callback and finalize both rows on stop/throw or
    budget exhaustion. When callback and budget both continue, finalize call
    two only after copying compact row-one candidates and promoting its feature
    state to the next origin's owning lease; no live transaction ticket crosses
    the cycle boundary.
16. Do not sample call-two row one during this cycle. Every authoritative
    selected x0/x1/x2/x3 is eventually target-K/V cached; every preview blocked
    by an earlier callback is discarded.

Selected-token vectors, generated/visible vectors, sampler state, retrieval
history, learned route state, target K/V, draft feature K/V, and callbacks must
advance as one logical transaction.

Callback events expose the logical committed count at that exact token
boundary, never the physically provisioned speculative-row count. This retains
the current Stage 4 convention for the first callback of a width-two result.

The split encoder API uses one owning multi-command transaction ticket. Encode
failure before submission creates no pending state. The **first successful
mutating submission**—shared, target-only, external-draft-only, router scout,
feature maintenance, Arm S, or restoration—activates the ticket and marks every
referenced subplan pending; later submissions join that ownership. A known-
complete draft/scout followed by route decline must verify discard/restoration
before ordinary fallback. Any wait failure with unknown completion poisons all
referenced target/draft/bundle children because commands may still own or have
mutated their buffers. After known-complete readback/callback failure, the
ticket attempts restoration to the last authoritative prefix; if commit,
finalize, discard, or restoration submission/wait itself fails, all referenced
plans remain poisoned and require destruction/rebuild. The ticket owns every
command/readback/lease until completion and is the only object allowed to commit
or restore. Ordinary fallback is legal only before the first mutating
submission or after verified successful joint rollback; a draft-only, scout,
or fused-command failure never silently falls back.

The joint ticket is not the whole session transaction. The **chat-session
layer** owns a bounded `SessionMutationJournal` for selected/generated/visible tokens,
sampler checkpoint, retrieval history, learned/router state, published Arm-S
lease, target/draft logical positions, incremental UTF-8 decoder, reverse-stop
automaton/search offset, and partial emitted/final text. At every callback
boundary the pure controller emits a low-level `SessionCommitReceipt`
describing the last authoritative token/cache/sampler prefix, including the
token whose callback was invoked even when that callback throws. It does not
own tokenizer, UTF-8, reverse-stop, messages, or visible text; the chat-session
journal applies the receipt to those concerns.

Callback invocation is the externally irreversible commit boundary. Before
invocation, make that token's authoritative target/draft/sampler/cache prefix
recoverable under its receipt while every later preview remains isolated. If
the callback throws, retain that token and all earlier committed state,
discard only later previews/off-path rows/unpublished trees, persist a
recoverable partial assistant turn plus decoder/reverse-stop state through the
receipt, then rethrow. Reuse begins after the delivered token; the engine never
rolls back and redelivers it. The callback's own partial external side effects
are unknowable, but engine delivery is at most once.

The outermost public `generate_user_turn` takes a `TurnStartCheckpoint` before
**any** prompt append, context-shift erase, cache trim, prefill, or target/
proposer work. `regenerate_last_turn` takes its checkpoint before moving or
erasing the previous assistant/user turn. The checkpoint covers `messages_`,
`cached_token_ids_`, `turn_cache_positions_`, `turn_sampler_states_`, session
sampler, proposer/router/cache epochs, visible/decoder/reverse-stop state,
authenticated target/draft/Arm-S frontiers, and device cache position.
Implement this with staged ownership, vector-size/mutation journals, and
moved-turn handles rather than a deep copy of the full conversation each turn.

One catch-all boundary surrounds the entire operation through durable message
and cache commit—not merely the user callback. It handles proposer/router,
tokenizer/UTF-8/final-text, callback, allocation/encode/submit/wait/readback,
commit/finalize/restore, context-shift prefill, and regeneration failures. A
pre-submit failure before any callback restores staged host state. After a
callback boundary, the last durable `SessionCommitReceipt`, not the pre-turn
checkpoint, is the rollback floor. A callback throw follows the committed-
partial-turn rule above. Any other known-complete device failure must jointly
rollback or deterministically rebuild to the latest receipt (or the pre-turn
checkpoint if none); an unknown completion or failed restore/rebuild poisons
the public session. Only after equality of all host/device state is verified does
`generate_user_turn` or `regenerate_last_turn` rethrow and permit reuse. The
poison flag gates generation, regeneration, sampler-option changes, and normal
clear; only explicit destroy/reopen or the authenticated reset-and-rebuild path
may clear poison. External callback side effects are not claimed reversible.

### General horizon and topology semantics

The H=3 event order above is the executable reference instance. Generate the
H=4/7/8/15/16/23 state tables mechanically from the same transition model and review the
golden tables into the artifact/tests; do not hand-copy another nested branch
tree. At every tile boundary, a frozen causal gate chooses the next tile versus
the ordinary one-row fallback using prefix survival and complete conditional
cost. Every already sampled x_i becomes authoritative only after all earlier
callbacks continue, then its sampler checkpoint and retrieval/learned history
are adopted exactly once. Every authoritative carry is cached before delivery;
every preview behind a stopped/thrown callback is discarded.
Before any ticket releases or reuses scratch, compact candidates are copied to
their owning result and the selected row/node feature is promoted to a
TargetStateLease (or the ticket remains pinned); this applies to mismatches,
tile carries, full-match rows, and forest leaves.

Every linear descriptor states `input_row_count`, `validated_edge_count`, and
one role per output row; input width alone does not imply how many proposals
are validated. In a terminal-carry call with m input rows
`[carry,d_a,...,d_(a+m-2)]`, all m outputs are verification rows and validate
`d_a` through `d_(a+m-1)`; the last matched proposal is not an input row and is
therefore an uncached carry. In a post-full call, the last proposal is included
as an input, only the preceding H proposal edges are validation outputs, and
the final row is explicitly `post_full_candidate`—it does not validate an
imaginary `d_(H+1)`. A subsequent logits-producing target call—ordinary or the
next eligible verifier—must cache a terminal carry before its callback, and
K/V-only is used only at its stop/budget endpoint. A width-four H=3 call
includes d3, caches it after full acceptance, and exposes row-three candidates
only after all callbacks continue and its transaction is finalized.
Width-three is terminal-carry. Dynamic horizons one and two obey this same rule
rather than borrowing the H=3 d2/d3 branches.

For a forest, target rows/nodes are isolated until the host walks one causal
path. At each depth it samples once from the parent's full-vocabulary target
distribution, follows the matching child if present, or ends with that target
token as the carry. It resolves the preceding selected token's callback before
making the preview authoritative, commits only the walked prefix, discards all
off-path nodes, and never samples unused leaves. Parent-specific sampler
history, absolute position, sliding/global mask, feature provenance, and
logical callback cache count follow the selected path only.

For Arm S, that forest is the prior invocation's immutable proposal lease and
the same target work also emits candidate next leases. Target completion alone
does not choose one: the pure controller walks the old tree under authoritative
sampling and callbacks, then returns exactly one publish/discard/rebase action.
The correction carry adopts only the next tree emitted by the main row whose
distribution produced that correction—even though the correction itself was
not an input row—and never a sibling's tree. Route switches must either pay an
MSA maintenance/bootstrap policy that preserves a correctly based lease or mark
Arm S unavailable until an ordinary MSA bootstrap; stale leases never survive
an epoch, history, or root mismatch.

### First-call command topology

Implement and directly compare these complete production topologies:

- **fused eager:** encode the drafter and call one in a single command buffer,
  with d1 and the hypothetical row-one history remaining device-resident;
- **host-gated:** submit and wait for the complete drafter, use its calibrated
  benefit decision on the host, then issue either call one or ordinary decode;
- **GPU-gated control:** first record whether this Metal command topology can
  actually skip target work from a device decision without host proposal
  readback; if supported, implement and measure it unless its registered
  favorable bound fails.

Fused eager requires the existing verifier to be refactored into encode and
completion operations. The compatibility wrapper must retain current Stage 4
behavior. At every fused tile boundary, embedding and target selection must
accept device-produced proposal IDs and device history suffixes—not only d1—
with explicit Metal barriers.

That suffix is a dynamic selection request, not just a token buffer. Add a
bounded chronological `SamplerHistorySnapshot` containing the raw ordered
repetition-window ring—including duplicates, head, length, and window limit—
plus a `DeviceSelectionRequest` containing that snapshot, a device suffix of
length zero through H, and suffix count. Do not derive it from the current
deduplicated `TokenSelectionRequest`, because that representation cannot model
which occurrence is evicted as a hypothetical suffix grows. For each row/path,
append its causal suffix to the raw history, truncate to the configured recent
window, apply the CPU sampler's exact repetition logic—including its duplicate
semantics—to that ordered window, and only then deduplicate the candidate-ID
union. Derive
unique history/base-candidate counts and preserve stable token-ID ties. Cover
duplicate/nonduplicate d_i at
repetition-window positions 0, 63, and 64; the current fixed host
base_candidate_count contract cannot be reused unchanged.

No topology wins from an isolated wait count. Run the actual learned route and
choose by whole-path development time. In the H=3 bring-up, call two remains a
separate conditional submission because x1/x2 and user callbacks are
authoritative host decisions. At larger H, each later conditional tile obeys
the same boundary unless a capability-tested GPU topology can preserve every
intervening authoritative sampling/callback decision without speculative host
effects.

Also compare three complete draft schedules at every registered external H.
Eager-H produces the
whole proposal block before target work. Conditional tiles produce only the
next registered tile after the prior accepted prefix/gate; H=3 includes the
explicit d1/d2-then-d3 schedule, optionally encoding d3 in the second command
buffer. Dynamic horizon stops at any registered edge using only causal
production confidence. This comparison is mandatory for recurrent arms, where
deep eager selector/context passes are wasted on every early miss or tile-gate
decline. A negative result from eager-H cannot close that family.

A deferred d_i resumes the origin-frozen drafter state after d(i-1) and clamps
full-context attention to the origin cache frontier. It may not read feature or
draft-K/V rows emitted by earlier verification: those contain later target
information unavailable to eager drafting. A verifier-reanchored continuation
is a distinct registered/trained/costed architecture. Poisoning verifier
features must leave an ordinary deferred proposal suffix unchanged.

### Target-verifier topology frontier

The conditional cascade is the measured reference, not an architectural lock:

Never reuse a paper's bare `B` or `N` as an array extent. Every registered
fixed-budget constructor carries a typed
`SourceTreeBudget { value, root_convention }`, where
`root_convention` is `Excluded` or `Included`, and resolves it before allocation
to `draft_node_cap` (non-root speculative nodes) and `input_row_cap` (the carried
root plus those nodes). The source value and convention remain in the artifact,
trace, topology digest, and benchmark output. Fixed DDTree, DominoTree,
DARTree, and paper-described TreeFlash budgets exclude the root and therefore
use `input_row_cap=B+1`; executable TreeFlash records root-inclusive
`tree_size` and uses `draft_node_cap=tree_size-1,input_row_cap=tree_size`.
JetSpec initializes the root inside its B-node tree and
uses `draft_node_cap=B-1,input_row_cap=B`. CaDDTree instead records an adaptive
non-root n in 0..1,024 plus its maximum, and PCTree records distinct
`draft_depth B` and root-inclusive `verifier_node_budget N`; neither is passed
through a misleading generic B field. Other constructors must record their
source convention from the pinned paper/code rather than inherit either choice
by name. EAGLE-3 uses 60 root-inclusive final rows; DBlast's all-category trie
uses actual root-inclusive rows through 121; ReDrafter's compressed trees are
root-inclusive, while its exact MLX topology is a separately tagged repeated-
root block-diagonal batch rather than a falsely deduplicated tree.

The ABI supports the largest faithful registered case, rather than clipping it
to the old 256-row sketch:

- `ExternalTreeDraftNodeMax=1,024` non-root speculative nodes;
- `VerifierRowMax=1,025` physical target input rows. Ordinary forest/tree forms
  include exactly one root; only the tagged ReDrafter MLX repeated-root batch
  may contain its declared 1..4 byte-identical root rows;
- `MaterializedTreeEdgeMax=1,024`, because a rooted 1,025-row tree has exactly
  1,024 incoming parent edges;
- `MaterializedDegreeMax=1,024`, needed because a legal DDTree/CaDDTree root can
  own all `B` selected depth-one nodes;
- `OutputFrontierBranchMax=32` and
  `OutputFrontierEdgeMax=VerifierRowMax*32=32,800` for separately stored,
  output-only proposals that are not materialized verifier rows;
- `BestFirstCandidateRankMax=1,024` and constructor-specific heap/frontier
  ledgers: DDTree/CaDDTree use `K=min(B,|V|)`, at most B pops and 2B pushes;
  TreeFlash `PaperFullVLazy` permits lazy sibling ranks through B=512 even
  though only M predecessor tokens per depth are expandable. M is not a child-
  rank cap there and leaves may lie outside top-M. ExecutableTopK instead has
  candidate-rank cap `topk_candidates` (16 in the released control);
- `DDTreeHeapLiveMax=1,025` and `CaDDTreeHeapLiveMax=1,025`: their root-seeded
  heaps can grow by one live entry per pop when the literal algorithm expands
  even the final selected node. Initialization, live extent, and total push
  events are separate metrics. `TreeFlashCursorHeapLiveMax=513` applies the
  analogous literal-final-expansion rule at B512. A smaller B/B+0 allocation is
  legal only with a byte-exact golden proving a no-final-expand optimization;
- `JetSpecHeapLiveMax=255` bounds the published root-inclusive B256 builder;
  `FullPathJetSpecHeapLiveMax=1,024` bounds the 1,025-node JetSpec-derived
  extension scaffold, while `FullPathDDTreeHeapLiveMax=1,025` preserves the
  DDTree-derived scaffold's literal final expansion. These are indexed heap
  capacities, not aliases for selected verifier nodes;
- `ReDrafterRawBeamEntryMax=225`, `ReDrafterRectangularKVRowMax=225`,
  `ReDrafterMLXPhysicalRowMax=16`,
  `ReDrafterMLXRepeatedVerifierRootMax=4`,
  `ReDrafterPyRawRootOccurrenceMax=45`,
  `ReDrafterPyUnpackedCacheRowMax=225`, and `ReDrafterPyTreeRowMax=181`; the L4
  compressed subbounds are 4/7/10/13 and the MLX uncompressed subbounds are
  4/8/12/16;
- `EAGLE3RawCandidateNodeMax=710`, `EAGLE3RawSupertreeRowMax=711`,
  `EAGLE3ScoreParentRowMax=71`, `EAGLE3ExpansionActiveMax=10`,
  `EAGLE3ExpansionForwardRowMax=70`, `EAGLE3ExpansionMaskKeyMax=80`, and
  `EAGLE3RetrievePathRowMax=9`, and `EAGLE3VerifierRowMax=60`;
  `DBlastCategoryTrieRowMax=121` covers K8/H15;
- `ConditionalExplicitMenuMax=256` for materialized DominoTree candidate-menu
  ablations, plus a separately tiled `ConditionalCandidateDomain=FullV` path
  that scans the artifact-bound vocabulary without allocating V children;
- independent constructor fields `DominoTreeBranchMax=8`,
  `DARTreeCandidateWidth=64`, `DARTreeLayerWidthMax=32`, and
  `JetSpecBranchMax=7`;
- pre-prune builder bounds `PCTreePoolMax=969` for B16/k8,
  `TreeFlashReleasedCorrectionStateMax=225` for H15/M16,
  `TreeFlashCodeTransferCorrectionStateMax=897` for H15/M64, and
  `TreeFlashSemanticCorrectionStateMax=961` for the H16/M64 transfer;
  independently, `TreeFlashReleasedPhysicalCorrectionRowMax=240`,
  `TreeFlashCodePhysicalCorrectionRowMax=960`, and
  `TreeFlashH16PhysicalCorrectionRowMax=1,024` bind actual H*M execution,
  `DARTreeSupertreeMax=512` for H16/W32,
  `DARTreeScoredPerDepthMax=2,048`, and
  `DARTreeScoreEventMax=30,784` (`64+15*32*64`) for a bounded per-depth
  streamed score buffer, and
  `DominoGeneratedChildMax=512`/`DominoHeapLiveMax=449` for the sequential
  B64/k8 oracle;
- `DominoFrontierActiveMax=64`,
  `DominoConditionalScoreTileMax=16,384` for M256 plus a separately tiled
  exact full-V domain, `DominoFrontierGeneratedPerDepthMax=512`,
  `DominoFrontierGeneratedLedgerMax=7,680`, and
  `DominoCorrectionLaneStepMax=960` for the exact depth-synchronous W=B Metal
  construction;
- TreeFlash checked rank scratch distinguishes
  `TreeFlashReleasedRankEntryMax=240*16=3,840` for released H15/M16/topK16,
  `TreeFlashCodeTop16RankEntryMax=960*16=15,360` for H15/M64/topK16, and
  H15/M64 with topK64
  `TreeFlashCodeTop64RankEntryMax=960*64=61,440`, optional H15 code/full-V-lazy
  `960*512=491,520`, H16 paper/full-V-lazy `961*512=492,032`, and H16
  code-shaped/full-V `1,024*512=524,288` when physically cached. A root-dedup
  optimization records its smaller semantic scratch separately; an exact
  streamed implementation instead charges every rescan; and
  `FullPathSupertreeMax=1,025`/`FullPathScaffoldEdgeMax=1,024`.

Materialized parent edges and output-only frontier edges are different arrays;
there is no million-edge `VerifierRowMax*MaterializedDegreeMax` allocation.
Each constructor validates its own realized per-parent degree, candidate rank,
active-layer width, depth, and budget against the typed descriptor. In
particular, DARTree's K=64 candidate menu and W active layer are not 64 or W
materialized children by definition, while DDTree's rank K can genuinely reach
1,024. A bridging `k=1`, H=23 chain with an oversized draft-node cap still
materializes at most 23 non-root nodes; a cap never fabricates unreachable
depth.

Before encode, validate exactly one root with a parent sentinel for every
unique-root topology; the ReDrafter `RepeatedRootBeamBatch` instead validates
1..4 identical root token/position/history rows, block-diagonal isolation, and
its declared W/L. Then validate connected prefix closure per tree/beam; acyclic topological parent-before-child order; parent/depth
consistency and exactly one matching incoming edge for every nonroot node;
legal token/absolute-position/mask/history/ticket ranges; stable
duplicate-free sibling token IDs (or the one preregistered deterministic merge
rule); node, edge, branch, and H bounds; and a declared row-output role. Each
materialized node is one target input row whose output distribution either
verifies declared outgoing proposal edges or is the post-full correction/
candidate row. A sampled materialized child can commit its row K/V; a sampled
matching output-only frontier child is an uncached carry; a materialized leaf's
row supplies the correction/next candidate after the full accepted path.

1. **conditional tile cascades:** register ordered **edge-count** partitions
   `(e1,...,et)` with positive parts summing to H and map them explicitly to
   input-row widths. A nonfinal tile begins with the uncached authoritative
   carry, has `e_j` input rows, and validates `e_j` proposal edges; its last
   matching proposal becomes the next carry. The final terminal-carry variant
   also has `e_t` rows, so total scheduled rows sum to H and the last matched
   proposal needs a later target call. The final post-full-candidate variant
   includes that last proposal, has `e_t+1` rows, and total scheduled rows sum
   to H+1. Thus the named H=3 `2+2` row cascade is the edge partition `(2,1)`
   with a post-full final row, not an ambiguous partition of three edges into
   four. For H=4/7/8/15/16/23 enumerate the registered partitions whose mapped row widths
   use production tiles 2/3/4/5/8/9/16/17/24, and gate each later tile after authoritative
   prefix survival; compare only after all ordinary fallbacks are included;
2. **single-invocation chains:** for H=3/4/7/8/15/16/23 build both the terminal-carry
   widths 3/4/7/8/15/16/23 and post-full-candidate widths
   4/5/8/9/16/17/24 with isolated speculative
   K/V. The former validates dH but carries/caches it separately on a full
   match; the latter includes dH, can retain every row, and emits post-full
   candidates. Commit only the authoritative prefix after host sampling;
3. **prefix-closed forests:** register bridging non-root draft-node budgets
   4/8/16/32/64/128/192/256; fixed DDTree non-root budgets
   B=16/32/64/128/256/512/1,024; CaDDTree adaptive non-root n=0..1,024;
   TreeFlash paper non-root B=16/32/64/128/256/512 and executable
   root-inclusive tree_size=16/32/64/128/256/512; DominoTree B=16 plus B=32/64;
   DARTree B=32/64/128/192; PCTree root-inclusive
   N=8/16/24/32/48/64/96/128 at k4 with independent draft depths 7/16, plus
   N32 k1/2/4/8 controls; JetSpec's root-inclusive main-table B=16/32/64/128/256 plus
   the distinct B=255 ablation; and FullPath-T non-root B=16/64/256.
   Register the source's branch/width controls rather than forcing them through
   one `k`: general bridging k=1/2/3/4/7/8/12, DominoTree k=8 and
   M=16/32/64/128/256/full-V, DARTree K=64 and W=4/8/12/16/32 plus
   fixed/deferred-pruned variants under the declared one-factor B/W pairings
   and `B<=16W` legality rule, TreeFlash M=16/32/64, JetSpec W=7, and
   PCTree k=1/2/4/8. Include
   independent-marginal, recurrent dynamic-tree,
   PCTree parent-conditioned Markov, DominoTree sequential-oracle/exact
   depth-synchronous best-first, DARTree fixed/deferred-pruned, JetSpec shared
   causal latent-depth,
   TreeFlash immediate-predecessor two-stage, and FullPath-T concrete-path
   constructors. Keep source budget/root convention, resolved draft
   nodes/input rows, raw candidate width M/K, active layer width W, realized
   published child degree, and target verification budget as different typed
   fields.

Every single-invocation implementation must be designed around M4 tiling; the
archived generic/width-four kernel is only a negative control. Compare complete
target selection, transactions, waits, callbacks, all accepted prefixes, and
minimum-prefix/global-context-tail/sliding-wrap outcomes. A fixed one-command,
fixed two-command, fixed horizon, or fixed node-budget preference is forbidden.
A topology may be omitted only when a favorable uncertainty-aware upper-speed
bound using its actual row/node shape is below the registered threshold.

## Bandwidth and memory analysis

The new path does not “go over” a 120 GB/s memory-bandwidth limit. That value is
a hardware cap on delivered traffic, not a budget that makes a program invalid.
GPU commands, CPU work, and unified-memory activity can overlap and contend;
excess demand appears as latency, queuing, and thermal throttling. The purpose
of verification is to reduce large target-weight traversals per selected token.

Apple specifies 120 GB/s unified-memory bandwidth for this
[M4 MacBook Air](https://support.apple.com/en-us/122209). The measured target
traversal schedules about 5.697 GB of weights, whose unattainable roofline floor
is about 47.5 ms at 120 GB/s. The measured ordinary row is 62.216 ms and a
width-two verifier is 83.642 ms, which is consistent with a bandwidth-dominated
target plus additional multi-row arithmetic. This machine has 16 GB of unified
memory, so resident capacity, OS headroom, compression, and swap are independent
constraints even when scheduled bytes per cycle look favorable.

At full three-edge acceptance:

- two target sweeps advance four selected tokens;
- target logical weight traffic is about 2.85 GB per selected token before
  draft work, versus about 5.70 GB for ordinary decode;
- the bandwidth ceiling is approached with more useful arithmetic per loaded
  target byte, not exceeded.

For the boundary-free H=3 gated cascade diagnostic, let V1 and V2 be the measured call-one/call-two costs,
B2 the ordinary logits-producing x2 fallback cost at the same reached states,
D the expected cost of the registered draft schedule, `T_ord` the measured
ordinary per-token completion time on the identical trajectory and thermal
regime, and C, R, E the measured
control/sampling, restoration, and endpoint costs. A complete decision episode
includes the ordinary x2 fallback after a gate decline:

\[
E[L_{episode}] =
1 + P_1 + P_{12G} + P_{12\bar G} + P_{123G}
= 1 + P_1 + P_{12} + P_{123G},
\]

\[
S_{cascade} =
\frac{T_{ord}\,E[L_{episode}]}
     {D + V_1 + P_{12G}V_2 + P_{12\bar G}B_2 + C + R + E}.
\]

If accounting stops at an uncached pending x2 instead, both its unit progress
and its later ordinary B2 cost are excluded; do not mix that convention with
the complete-episode equation. Live trajectory replay is authoritative because
costs, gates, matches, fallbacks, and context are correlated.

At the second-call decision, the target-only local scale check is
V2 < B2(1+p3), where p3 is calibrated Pr[M3 | M1,M2,G] at the reached state.
With the old 83.642 ms and 62.216 ms values, p3 must exceed roughly 34.4%
before draft-three, control, restoration, or thermal costs. The actual gate
uses measured conditional distributions and costs, not that scalar threshold.

The old always-launch scale check is the special case G=true, V1=V2=V,
B2 absent, and C=R=E=0:

\[
S =
\frac{T_{ord}(1 + P_1 + P_{12} + P_{123G})}
     {D + V(1 + P_{12})}.
\]

Under the deliberately simplified iid match rate 77.4% and D = 2 ms:

- P12 is about 0.599;
- expected canonical-frontier progress is about 2.837 tokens;
- target calls per selected token are about 0.564;
- target wall time per selected token is about 47.2 ms, roughly 24% below the
  62.2 ms ordinary row before draft, restoration, and control work.

The 77.4% and 2 ms numbers are explanatory only. Actual joint match/gate
events and complete costs determine whether any trained arm wins.

For each topology, also report expected logical scheduled bytes per selected
token:

\[
W_{spec} =
\frac{W_D + W_{V1} + P_{12G}W_{V2} + P_{12\bar G}W_{B2}
      + W_{restore} + W_{endpoint} + W_{feature}}
     {1 + P_1 + P_{12} + P_{123G}}.
\]

Every width 3/4/5/7/8/9/15/16/17/23/24 chain, conditional tile partition, and k/N forest
shape receives a separate ledger; none inherits another topology's sweep
count. H=4/7/8/15/16/23 admission economics come only from complete correlated trajectory
replay/live execution, not exponentiating the H=3 match rate.

### Illustrative draft traffic

For a two-layer, width-768, three-position Arm P with five target taps,
d-prime 224, k = 2,048, and rowwise Q4/Q8 packing, an initial logical-byte
ledger is approximately:

| Component | Approximate scheduled bytes per draft cycle |
| --- | ---: |
| five 3,584-to-768 tap projections, Q4 including block scales | 7.7 MB |
| two width-768 attention/SwiGLU layers, Q4 | 8.0 MB |
| one batched 256K by 224 selector table, Q8_0 | 60.9 MB |
| three sets of 2,048 selected 3,584-wide target Q8_0 columns | 23.4 MB |
| full two-layer F16 draft K/V read at context 8,192 | at most 48 MiB if shared across the three queries |
| target-space up projection and selected Markov token vectors | low single-digit MB |

This is roughly 0.15 GB per cycle before scratch and index overhead, versus
5.697 GB for one target sweep. A recurrent arm may reread its selector and
context per edge and plausibly schedule 0.25-0.45 GB instead. Those are design
estimates, not performance claims.

The Q8_0 selector alone is 60.9 MB because each 224-element logical token vector
occupies seven 34-byte blocks. F16 would be 114.7 MB. A selected target Q8_0
packed column is 3,808 bytes; 2,048 columns across three edges are 23.4 MB. A
full target-head traversal per draft edge would be nearly 1 GB and is a required
cost control, not the assumed product.

Longer blocks do not multiply target **weight** bytes inside a correctly tiled
single invocation: post-full width 4/5/8/9/16/17/24 is intended to reuse each loaded weight tile
across more rows. They do multiply activation arithmetic, scratch, attention,
selection lanes, and speculative K/V, so they can move the GPU from bandwidth-
bound toward compute-bound. Conditional cascades make the opposite trade: each
launched tile can incur another target sweep, but later sweeps are skipped after
early rejection. Arm S also does not get a free draft: its MSA top layers reuse
the target pass's resident weights but add up to `gamma` streams per surviving
node, adapter/attention FLOPs, transient activations, pruning, and next-tree
fan-out. Measure those additions; never count them as zero because no separate
model was launched.

The largest **incremental speculative-stream-head** shape in the legal Arm-S
registry is the gamma23/k1 chain: 24 raw/surviving anchors times 23 streams, or
552 full-vocabulary stream rows.
At vocabulary 256,000 and hidden width 3,584 that is 506,462,208,000
dot-product multiply-accumulates (about 1.013 TFLOP when multiply and add count
separately) before normalization/top-k. A naive stream-only F32 logit slab
would be 565,248,000 bytes (about 0.526 GiB). Main verification distributions
are a separate required head ledger. The reported Shared Table-3 32-node fact
does not publish one selected per-task gamma/k tuple, so no full cost is
attributed to it. In the separately labeled **lossless gamma4/k3 plus
cross-mode-32** diagnostic, a root-inclusive interpretation executes 32 upper
main plus 128 stream distributions: 160 rows,
146,800,640,000 MACs (about 0.294 TFLOP), and a hypothetical
163,840,000-byte/0.153-GiB F32 slab. A convention with 32 non-root nodes plus a
separate carried root has 165 rows, 151,388,160,000 MACs (about 0.303 TFLOP),
and a 168,960,000-byte/0.157-GiB slab. These are local cross-mode ledgers, not
lossless source or product claims. Both still execute the 121-row raw lower
tree before pruning. The lossless threshold-only gamma4/k3 transfer is the largest
total-head case: as many as 121 upper main plus 484 stream distributions, or
605 rows, 555,089,920,000 MACs (about 1.110 TFLOP), and a hypothetical
619,520,000-byte/0.577-GiB F32 slab. Its actual survivor count is measured; it
is not described as an unpruned published configuration.
The required tiled top-k removes the dense slab and can reuse head-weight
tiles, but it cannot remove the arithmetic. The Cartesian safety
envelope `256*23=5,888` is not a legal registry cell, is never allocated as a
candidate, and supplies no performance conclusion. These real legal shapes are
already likely compute-heavy; only a production-shaped measurement or valid
implementation-independent unavoidable-work certificate may exclude one. This
is why a 120 GB/s-only model is insufficient for Arm S.

For each live topology report the empirical roofline **diagnostic**

\[
T_{roof,predict}=
\max(W_{estimated}/BW_{empirical},F_{counted}/P_{empirical}),
\]

using development-measured sustained bandwidth/compute envelopes and clearly
labeling any logical-byte substitute. It predicts where a measured
implementation may be bottlenecked; it is not a lower bound and can never close
a family. An achieved microbenchmark rate is not an upper bound on what a
better kernel could achieve.

An implementation-independent exclusion certificate instead needs
conservatively proven unavoidable work and genuine hardware upper ceilings.
For serial phases q that cannot overlap, use

\[
T_{min}\;\ge\;\sum_q
\max\!\left(
  W^{min}_q/BW^{max},
  F^{min}_q/P^{max},
  T^{min}_{serial,q}
\right),
\]

and use the corresponding critical-path construction for phases proven able to
overlap. `W_min`/`F_min` must count work every faithful implementation performs;
`BW_max`/`P_max` must be conservative physical or analytically justified device
maxima, not observed sustained rates. The 120 GB/s M4 nameplate may serve as a
bandwidth upper ceiling; if a defensible compute ceiling or unavoidable-work
minimum is unavailable, this certificate is unavailable and the member needs a
complete live implementation. Logical scheduled bytes and empirical rooflines
remain diagnostics only. Report achieved bytes/s, arithmetic intensity, target
sweeps per canonical-frontier token, canonical-frontier tokens per target
sweep, selected-but-uncached carries, and the residual gap to measured wall
time.

### Mandatory traffic evidence

Measure and report:

- logical bytes by target projection, target head, feature fusion, external
  draft backbone, context K/V, selector, selected-token scorer, correction,
  Arm S stream initializer/adapters/attention/full-vocabulary head/top-k/pruner,
  router scouts, copies, and rollback;
- prompt-row tap capture, projection, and draft-K/V writes; cached-turn
  continuation; context rebuild/reset; terminal K/V-only projection; artifact
  upload; and startup traffic separately from steady generation;
- wall time around encode, commit, wait, readback, CPU sampling, callbacks, and
  transaction work;
- production command-buffer GPU start/end time. Per-component GPU times are
  reported only when supported counters exist or from separately launched
  diagnostic commands, and are labeled non-additive rather than fabricated
  inside a fused command;
- commands, encoders, dispatches, barriers, commits, waits, allocations, and
  compact readback bytes;
- available Metal counter-set and Xcode GPU-capture DRAM/utilization evidence
  in diagnostic runs;
- resident target, target K/V, scratch, artifact, selector, correction, and
  draft K/V bytes;
- Arm S transient/persistent stream arenas and every heterogeneous child/router,
  with peak live overlap rather than a sum of individually measured minima;
- RSS, memory pressure, compression, and swap before and after cold start and
  after heat soak.

Logical scheduled bytes are not physical DRAM bytes and must be labeled as
such. Profiler runs are diagnostic because capture changes timing.

The initial width-768 design is expected to add roughly tens to a few hundred
megabytes of resident state, including a roughly 48 MiB maximum-context draft
K/V cache. There is no arbitrary byte pass gate. The hard requirements are:

- all persistent state fits without swap or memory-pressure compression;
- there is no warm steady-state allocation or artifact growth;
- the generation loop incurs zero major faults and zero candidate-attributable
  swap-in/swap-out or compressed-byte growth under symmetric predeclared
  environment invalidation rules;
- both cold isolated and thermally saturated end-to-end performance pass.

## Data and trace protocol

### Immutable split boundary

Anything previously opened or used to choose an architecture is development
data. Create new immutable source-grouped partitions:

1. **training/architecture development:** model fitting, debugging, topology
   construction, optimistic-bound work, family-completion decisions, and
   unrestricted iteration; legacy traces may be used here;
2. **architecture-selection validation:** one-time capacity, feature,
   vocabulary, routing, precision, and complete live runtime-topology selection
   after every registered family is implemented or validly bounded;
3. **calibration:** thresholds and confidence calibration only after model
   architecture is frozen;
4. **sealed admission:** opened once after code, weights, quantization,
   selector, router, statistics program, and gates are content-addressed.

All decisions to add, remove, resize, retrain, or reimplement a chain, forest,
heterogeneous bundle, or target-integrated member use partition 1 only. The
architecture-selection partition is not opened to decide whether another
family is needed and then reused to select the resulting candidate. If an
unforeseen family is added after partition 2 is exposed, the old partition
becomes development evidence and a newly source-disjoint partition 2 must be
generated before selection resumes.

Before assigning any partition, build connected components over root source
identity, prompt family/template, topic, exact token/text duplicates, and
MinHash-style near duplicates. Assign each complete connected component once,
then keep its entire conversations, turns, seeds, continuations, and derived
windows in that partition. Split before target response generation/windowing.
An automated sealed-set collision check may return only pass/fail and a digest;
it must not reveal sealed contents or influence architecture choices.

The independent population unit is the highest shared source cluster, not an
overlapping token window. If multiple conversations share a source document,
repository, prompt generator, or near-duplicate component, that common parent
is the cluster. Preserve the crossed calendar-day factor with the frozen
two-way resampling and power simulation; never collapse it into source-level
replication.

### Workload coverage

Freeze mutually exclusive primary workload labels and product weights before
collection. Context positions are possibly overlapping secondary labels, not
additional primary strata. Include at least:

- multi-turn instruction chat;
- factual and grounded QA/RAG;
- code generation and code explanation;
- mathematics and structured reasoning;
- summarization and translation;
- structured output and tool-like formats;
- rare names, numbers, uncommon tokens, and non-English inputs when within
  product scope;
- natural stops, short answers, long answers, and explicit token caps;
- early context, middle context, sliding-window wrap, global-context tail and
  boundary, near-8K context, and context-shift rebuild;
- repeated-text cases where retained retrieval is active.

Do not use random-token or templated expansions as generalization evidence.
[SPEED-Bench](https://arxiv.org/abs/2604.09557) specifically motivates
workload-stratified speculative-decoding evaluation because acceptance and
speed are data-dependent.

### Scale

The old 40 conversations are a debugging cohort, not a fair training attempt.
Collect nested target-generated scales of approximately 100K, 250K, 500K, 1M,
and 2.4M eligible origins with five frozen sampling seeds per conversation.
The DFlash fidelity cell includes its approximately 800K-example scale; the
DFlare fidelity cell reaches 2.4M; DBloom expansion and continuation corpora are
separately counted rather than relabeled as base examples. The 500K scale is a
floor, not an admission claim or a hidden stopping cap. Register a maximum
trained scale per member before opening architecture-selection validation. Use
ordinary learning-curve extrapolation only to schedule development resources,
never as a family-closing bound. If held-out joint-prefix or predicted-live-cost
performance is still improving at the registered maximum, add the next
geometric scale under a versioned registry before architecture-selection
validation or state that the conclusion applies only to the tested scale.
Resource exhaustion
does not become evidence that more data cannot help.

Report unique source clusters, source groups, and conversations beside origins.
The five seeds remain nested inside their source cluster and do not inflate the
independent sample count.
For context, SpecVocab reports roughly 464K target-generated examples and
DFlash reports a much larger target-generated training regime on different
models and hardware; those papers do not validate a 40-conversation shortcut.

### Trace vNext

Build a bounded, checksummed, streamable trace whose records include:

- conversation/source/workload/split identity and logical position;
- exact prompt/turn/seed derivation and sampler scalar bit patterns;
- one atomic causal tuple {F_t, p_t, u_t, x_t, S_pre_t, S_post_t}: the feature
  state and exact prepared target distribution that produced x_t, raw draw
  evidence, selected token, and sampler states before/after that selection;
- compact candidate logits, totals, and prepared probabilities serialized as
  binary doubles with bit patterns—not rounded F32 text;
- all frozen target taps or their losslessly specified projection inputs;
- TargetStateView provenance and cache generation;
- stop, remaining budget, retrieval eligibility/proposal, context bucket, and
  wrap state;
- target source and resolved-resident-transformation receipts, resolved output
  policy, production Q8 head, tokenizer, chat template, code, build, Metal
  library, and trace-schema digests.

The current sampler's exact top-k support is sufficient to store its complete
prepared sampling distribution; do not mistake that for a full target-vocab
logit row. Store additional teacher top logits, hard negatives, and full
log-sum-exp only where the frozen training objective requires them. Full logits
may be used offline, but no full row crosses to the host in production.

Construct an H-edge label only by joining H+1 consecutive atomic records
inside the same source, conversation, turn, split, workload, context generation,
and logical/cache sequence. Require each record's selected token to be the next
record's anchor and prove the F_t/p_t/x_t indexing with a scalar reader. Never
cross a stop. Store one feature sequence plus indices rather than materializing
overlapping windows.

An independent reader must recompute split membership, causal windows, sampler
distributions, denominators, and SHA-256 digests. Malformed length, shape,
nonfinite, overflow, checksum, or identity fields fail closed.

## Artifact and production ownership

Create a new artifact format. Reuse the archived bounded-reader, digest,
atomic-publication, and CPU-golden patterns only after audit; do not reuse its
fixed vocabulary, OTHER bucket, independently free output table, one-edge
controller, or weak fingerprint.

Refactor construction around a CPU PreparedGemma2Plan. It binds GGUF tensors,
computes a verified model-identity receipt, resolves requested versus actual
execution/output-head policy, generates and hashes the production Q8 bytes,
performs and hashes every other upload-time target transformation—including
the retained Q6_K-down-to-Q4_K repack and its algorithm/version/layout—and
optionally loads/validates the learned artifact or heterogeneous bundle,
feature-bank contract, sampler contract, and cache maxima before
allocation/upload. Target-only and
Stage 4 construction remain valid with no learned artifact. When learned mode
is requested, only a fully valid prepared target+artifact/bundle graph may
atomically create every required GPU plan; a missing/mismatched child fails
rather than silently uploading half the system. This reverses the current
session/plan construction order and is not implemented by merely adding a
check to Gemma2ChatSessionState.

The identity receipt includes the SHA-256 of the complete GGUF plus a canonical
typed encoding of every consumed metadata value and the resolved
`Gemma2ModelConfig`: architecture, dimensions, layer/head/KV-head counts,
context and sliding-window limits, RoPE values, RMS epsilon, attention/logit
softcaps, tokenizer, special-token IDs, and chat template. Tensor bytes alone
cannot identify execution semantics. Make `Gemma2ChatSessionState` opaque or
constructible only from one unforgeable prepared-plan receipt; callers may not
independently combine a file model, borrowed EngineModel, Tokenizer, target
Metal plan, and learned artifact from different preparations.

After successful upload, release all bulk CPU staging—including the roughly
gigabyte-scale generated Q8 payload and every Q6_K-to-Q4_K transformed target
buffer—while retaining compact identity receipts and configuration. Clear/reset
zero or rebuild caches in place and do not keep a duplicate target solely for
reset. A poisoned-plan destruction/rebuild may deterministically recreate all
transformed staging from the original GGUF; that exceptional cost is not
resident steady state.

Retain the learned side as an authenticated `FileBackedSpeculationArtifact` or
bundle source: an already-open immutable descriptor/mapping plus payload
receipt, not merely a pathname. Poison rebuild reuses that authenticated byte
source, or reopens and rehashes the entire file before upload. Deletion,
replacement, truncation, or metadata change between construction and rebuild
must either leave the original immutable mapping valid or fail closed; it may
never rebuild router/drafter/Arm-S weights from unauthenticated new bytes.

The artifact binds cryptographically:

- the complete GGUF/tensor payload identity for every consumed target tensor;
- the whole-GGUF digest, canonical consumed-metadata receipt, and resolved
  `Gemma2ModelConfig` values/types;
- the complete **resolved resident target** identity after every upload-time
  conversion, with requested/resolved policy, source ranges, conversion
  algorithm/version, dtype/block layout, output ranges, and SHA-256—including
  every retained FFN down-projection Q6_K-to-Q4_K result;
- tokenizer and chat-template metadata;
- requested and resolved output-head policy separately;
- generated production-Q8 output-head bytes, quantizer algorithm, block layout,
  and version;
- sampler transforms, scalar bit patterns, supported option ranges, and draw
  ticket semantics;
- exact feature bank/tap layers, residual/norm semantics, shared/layer-specific
  fusion layout, resident/trace precision, and rounding;
- family, width, depth, FFN, recurrence/correction, horizon, context,
  ProposalGraph node/edge/branch capacities, Arm-S initializer/early-exit-rank/
  post-prune policy and fan-out capacities, row roles, and mask contract;
- selector algorithm, d-prime, k policy, candidate unions, and stable ties;
- artifact role, immutable architecture digest, calibratable-policy schema,
  fitting-code receipt, and unset/provisional/final policy digest;
- every tensor's name, dtype, dimensions, stride, packing, byte range, and
  SHA-256;
- corpus/split/trace/training-code/config/environment/checkpoint receipts;
- golden H=3/4/7/8/15/16/23 CPU rollouts, topology paths, and expected route results.

A standalone artifact has one family payload. A heterogeneous bundle has its
own schema and digest and contains an ordered manifest of child artifact
digests, a router payload/configuration, legal child routes, route precedence,
shared feature-bank and cache-compatibility receipts, per-child resolved
horizon/topology, eager/lazy cache-maintenance policy and replay basis, and the
exact total persistent-memory envelope. Child ranges remain independently
bounded and authenticated. Preparation validates all
children before allocating any of them; upload success publishes one bundle
epoch, and failure tears down the unpublished graph. Reset, poison, rebuild,
and allocation-identity checks operate on the complete graph rather than only
the arm selected for the last token.

Artifact role is explicit and monotonic. Phase 6 emits a nonselectable
`WeightsArtifact` (or weights bundle) with no fitted policy; only an explicit
development harness may run it under a registry-fixed always-propose/scripted
policy. Phase 7 applies the preregistered fitting code to partition 1 and emits
a `SelectionArtifact` that references the immutable weights digest and binds
the provisional policy digest. Phase 9 emits an `AdmissionArtifact` that
references the selected weights/architecture and binds the calibration-only
policy. A Phase 8 loader rejects weights-only, unset-policy, wrong-role, or
stale-policy artifacts; no weights are repacked merely because policy scalars
changed.

Hash target bytes while they are already traversed for load/upload where
possible; do not add an unmeasured second multi-gigabyte startup sweep. Validate
the artifact fully before Metal upload and expose no partially initialized
plan. Malformed, oversized, mismatched, nonfinite, unsupported, or unavailable-
Metal cases fail before generation.

Development training may use external tensor frameworks and accelerators.
The packed artifact loader, all production draft execution, routing, target
verification, sampling, and token selection remain engine-owned and work
offline.

Add every generated artifact extension and training-output directory to
.gitignore and to scripts/ci.sh's forbidden-weight/output scan before producing
weights. No learned artifact, trace, model weight, benchmark result, or packed
binary is committed.

## Implementation plan

Work on an experiment branch. Keep retained Stage 4 selectable and behaviorally
frozen until the new path passes.

### Phase 0 — preregister identities, registry, and experiment

1. Record the Stage 4 commit, release binary, compiler/SDK/OS identity, GGUF
   SHA-256, canonical consumed-metadata/`Gemma2ModelConfig` receipt,
   tokenizer/template digest, requested and resolved Q8 policy, sampler golden
   streams, every upload-time target-transformation receipt including FFN
   Q6_K-to-Q4_K, and canonical benchmark output.
   For every named literature control, also freeze a source-method receipt:
   exact paper version/PDF hash, upstream repository commit, checkpoint/config
   digest, license, and an explicit resolution ledger for paper-versus-code
   discrepancies. Any still-unresolved field—currently JetSpec's numeric T_KD
   plus paper-versus-code H16/H15 and argmax-versus-latent-mask conditioning;
   TreeFlash's paper-gamma16 versus executable-H15 row convention,
   `[n;p]` versus `[p;n]` concat order, and root-excluded B versus
   root-inclusive executable `tree_size`, plus full-V lazy paper ranks versus
   executable topK16 truncation;
   ReDrafter's article four-step/deduplicated beam versus executable L4-as-
   root-plus-three/no-compression semantics and backend-dependent tie order;
   Arm-S lossless pruning normalization/threshold, the separately scoped Shared
   Table-3 32-node root/count/threshold/cap/selection/tie interaction and
   per-task gamma/k, parallel-adapter algebra/sharing, undefined Section-3 AdamQ
   optimizer details, paper-transfer positional encoding,
   Ns4 mask-versus-three-stream position-ID bug, and multi-path rejection/RNG
   semantics—remains blocked rather than
   drifting when upstream changes; the DFlare paper/code gamma discrepancy is
   preserved in that ledger with both registered interpretations.
2. Freeze the complete target feature bank, shared three/five subsets,
   layer-specific/hybrid mixing contracts, pre/post-norm meanings, resident
   precision/layout, proposal arithmetic, target/draft logical positions, and
   the exhaustive calibratable-policy schema plus development-only provisional
   gate/router fitting algorithms.
3. Publish the finite candidate registry, exact three training seeds,
   convergence/restart rules, factorial cells, direct-test obligations,
   optimistic-bound calculation, learning-curve extension rule,
   H=3/4/7/8/15/16/23 external target topologies; the exact Arm-S
   Section-3 lossless gamma4/k3 cell with identity initializer and rank-8
   early-exit projection but no invented 32-node cap; its Ns2/4/8 transfers,
   gamma3/k3 and gamma4/k1/k2 controls, learned-initializer,
   threshold-only/uncapped, and explicitly cross-mode lossless-32 ablations,
   and gamma7/15/23-k1 transfers; separately, the non-product Shared Table-3
   `ReportedSharedTable3PostPruneDraftNodeCount=32` resolution fixtures; the typed source-budget/root-convention tree
   registry through fixed DDTree B=1,024, adaptive CaDDTree n=0..1,024,
   TreeFlash, PCTree, JetSpec, DominoTree, DARTree, and FullPath-T; ReDrafter's
   MLX repeated-root, article-described, and PyTorch compressed beam forms;
   EAGLE-3's 711-row pre-tree/60-row final tree; DBlast's K-category tries; and explicit
   resolved-row, materialized-edge,
   output-frontier, candidate-rank, branch, and builder-width rules for every
   registered tree/heterogeneous construction, and
   target-integrated raw/survivor/builder rules before architecture-selection
   data exist.
4. Freeze mutually exclusive workload weights, source clustering, split
   generator, invalid-pair rules, weighted estimator, bootstrap interval and
   multiplicity method, fixed sample size, performance gates, and cold/sustained
   thermal protocol before collecting admission data. Freeze the separate
   llama.cpp blocked-comparison size and interval rule before its final rerun.
5. Add an experiment manifest that records a dirty-diff digest rather than
   silently describing a dirty build as its parent commit. Re-run the retained
   correctness and cooled baseline when binary or model identity changes.

Exit: an independent reader can reconstruct a content-addressed baseline,
candidate registry, feature contract, and statistical protocol without an
unrecorded choice.

### Phase 1 — exact sampler and pure controller foundation

1. Add an ordered raw `SamplerHistorySnapshot` ring and bounded span/
   `DeviceSelectionRequest` forms for zero through `ExternalHmax=23`
   hypothetical tokens.
   Reproduce append-then-window-truncate repetition semantics before candidate
   deduplication, with no hot allocation.
2. Add sampler checkpoints/forks and preview_ticket/sample_with_ticket while
   pinning the existing target stream bit-for-bit. Provide 23 external-arm
   tickets and the Arm-S `2*StreamHmax+1=47` offset pool without advancing authority;
   verify the anchor-to-ticket mapping at every accepted depth.
3. Implement a standalone BoundedSpeculativeController with fixed safety
   capacities `ExternalHmax=23`, `StreamHmax=23`,
   `StreamRawPriorNodeMax=256`, `StreamSurvivorMax=256`,
   `StreamUpperRowMax=6,144`, `StreamTransitionBranchMax=8`,
   `SNextNodeMax=65,536`, `SRawChildMax=47,104`,
   `ExternalTreeDraftNodeMax=1,024`, `VerifierRowMax=1,025`,
   `MaterializedTreeEdgeMax=1,024`, `MaterializedDegreeMax=1,024`,
   `OutputFrontierBranchMax=32`, `OutputFrontierEdgeMax=32,800`,
   `BestFirstCandidateRankMax=1,024`, `DDTreeHeapLiveMax=1,025`,
   `CaDDTreeHeapLiveMax=1,025`, `TreeFlashCursorHeapLiveMax=513`,
   `JetSpecHeapLiveMax=255`, `FullPathJetSpecHeapLiveMax=1,024`,
   `FullPathDDTreeHeapLiveMax=1,025`,
   `ReDrafterRawBeamEntryMax=225`, `ReDrafterRectangularKVRowMax=225`,
   `ReDrafterMLXPhysicalRowMax=16`,
   `ReDrafterMLXRepeatedVerifierRootMax=4`,
   `ReDrafterPyRawRootOccurrenceMax=45`,
   `ReDrafterPyUnpackedCacheRowMax=225`, `ReDrafterPyTreeRowMax=181`,
   `EAGLE3RawCandidateNodeMax=710`, `EAGLE3RawSupertreeRowMax=711`,
   `EAGLE3ScoreParentRowMax=71`, `EAGLE3ExpansionActiveMax=10`,
   `EAGLE3ExpansionForwardRowMax=70`, `EAGLE3ExpansionMaskKeyMax=80`,
   `EAGLE3RetrievePathRowMax=9`, `EAGLE3VerifierRowMax=60`,
   `DBlastCategoryTrieRowMax=121`,
   `PCTreePoolMax=969`,
   `TreeFlashReleasedCorrectionStateMax=225`,
   `TreeFlashCodeTransferCorrectionStateMax=897`,
   `TreeFlashSemanticCorrectionStateMax=961`,
   `TreeFlashReleasedPhysicalCorrectionRowMax=240`,
   `TreeFlashCodePhysicalCorrectionRowMax=960`,
   `TreeFlashH16PhysicalCorrectionRowMax=1,024`, `DARTreeSupertreeMax=512`,
   `DARTreeScoredPerDepthMax=2,048`, `DARTreeScoreEventMax=30,784`,
   `TreeFlashReleasedRankEntryMax=3,840`,
   `TreeFlashCodeTop16RankEntryMax=15,360`,
   `TreeFlashCodeTop64RankEntryMax=61,440`,
   `TreeFlashH15FullVRankEntryMax=491,520`,
   `TreeFlashH16FullVRankEntryMax=492,032`,
   `TreeFlashH16CodeFullVRankEntryMax=524,288`,
   `DominoGeneratedChildMax=512`, `DominoHeapLiveMax=449`,
   `DominoFrontierActiveMax=64`, `DominoConditionalScoreTileMax=16,384`,
   `DominoFrontierGeneratedPerDepthMax=512`,
   `DominoFrontierGeneratedLedgerMax=7,680`,
   `DominoCorrectionLaneStepMax=960`,
   `FullPathSupertreeMax=1,025`, `FullPathScaffoldEdgeMax=1,024`, and
   `ConditionalExplicitMenuMax=256`, a full-vocabulary streamed conditional
   candidate domain, `DominoTreeBranchMax=8`,
   `DARTreeCandidateWidth=64`, `DARTreeLayerWidthMax=32`, and
   `JetSpecBranchMax=7`, plus the validated tagged
   `ProposalGraph` and exact source-budget/topology/path/outcome/cost records.
   Bind `ReportedSharedTable3PostPruneDraftNodeCount=32` as source-fixture
   metadata, not as a lossless runtime safety capacity or pruning rule.
   Generate and inspect
   H=3/4/7/8/15/16/23 transition tables and every legal Arm-S raw/pruned/stream
   transition table from one state machine. Explicitly exhaust H16 accepted
   depths, tickets, histories, callbacks, endpoints, and overflow paths rather
   than treating H15 plus a width-16 kernel as equivalent; likewise exhaust
   ReDrafter L5/H4 and EAGLE-3 depth8 rather than routing either through a
   neighboring horizon.
4. Exhaust accepted lengths zero through H, remaining budgets one through H+1,
   stops/callbacks at every depth, carries, every uncached terminal endpoint,
   context exhaustion, every tile-gate outcome, retrieval priority, dynamic
   horizons one through H, forests, Arm-S pending-bootstrap/bind/steady/
   pending-correction/drain/rebase lease transitions, sibling ticket maps, and
   exceptions.
5. Implement corrected event/trajectory replay using measured conditional
   outcome costs; retain iid equations only as diagnostics.

Primary files:

- include/local_gguf_engine/engine/sampling.hpp
- src/engine/sampling.cpp
- new speculative_controller.hpp/.cpp
- tests/engine/sampling_tests.cpp
- new tests/engine/speculative_controller_tests.cpp

Exit: the pure controller and baseline loop produce identical selected tokens,
sampler states, callbacks, and logical cache decisions for every scripted
outcome. This proves control semantics, not speed.

### Phase 2 — real target-verifier topology frontier

1. Split width-two verification into validated encode, owning **target-only**
   ticket, submit/wait, target-candidate completion, provisional commit,
   finalize, discard, and poisoned-plan states while preserving Stage 4.
2. Introduce bounded `VerifierTopologyDescriptor`, `VerifierEncodeRequest`,
   `VerifierAttentionSchedule`, `TargetSelectionWorkspace`, owning ticket, and
   completion-result types. They carry actual input-row/node count, parents,
   depths, proposal-edge count, per-row verify/post-full role, absolute
   positions, per-node causal-history span, sliding/global masks, per-layer
   attention counts, selection lanes, topology kind (unique-root tree versus
   the exact ReDrafter repeated-root beam batch), source budget/root convention, resolved
   draft/input-row counts, and tile schedule within
   ExternalHmax/VerifierRowMax/materialized-edge/output-frontier capacities.
   Replace fixed two-lane arrays and
   `second_attention_token_count`; invalid topology fails before encode.
3. Add scripted/perfect H=3/4/7/8/15/16/23 proposals through the exact controller,
   including causal tile gates and ordinary logits fallbacks.
4. Add device proposal/history inputs through ExternalHmax for chains and
   VerifierRowMax with separate materialized and output-frontier edge arrays for
   forests, dynamic per-path candidate-union counts, and explicit Metal
   dependencies needed at every fused tile/node batch.
5. Implement the generic target-only ordinary K/V endpoint for every terminal
   authoritative uncached token. Phase 3 attaches feature/draft-cache
   maintenance to the same hook after that substrate exists.
6. Implement new M4-tiled linear verifiers at widths
   3/4/5/7/8/9/15/16/17/23/24 and the registered conditional tile partitions, each
   with private speculative K/V. In this phase—not deferred to drafter work—
   implement Gemma's production-Q8 full-vocabulary target head, exact sampler
   transforms/stable selection, and compact per-parent candidate output for
   every required row. Do not port the rejected generic-N kernel or time a
   row-by-row proxy.
7. Implement target-side forest arenas, ancestor masks, production-Q8 full-V
   target scoring/selection, and accepted-path scatter for every registered
   source-budget/root-convention/k/M/K/W shape not
   excluded by a valid shape-specific perfect-proposal/lower-cost bound. Cover
   fixed DDTree rows 17/33/65/129/257/513/1,025; TreeFlash paper rows
   17/33/65/129/257/513 and executable-code rows 16/32/64/128/256/512;
   CaDDTree every adaptive physical row count 1..1,025; PCTree
   8/16/24/32/48/64/96/128; DARTree
   33/65/129/193; DominoTree 17/33/65; JetSpec 16/32/64/128/255/256; and the
   FullPath-T shapes. Also cover DBlast's K1/2/4/8 actual-count category tries
   through 121 root-inclusive rows; EAGLE-3's 60-row final tree after its real
   711-row pre-builder; ReDrafter MLX repeated-root batches 4/8/12/16,
   code-L4 PyTorch compressed trees 4/7/10/13, article-described four-draft-
   edge trees 5/9/13/17, and L5/W45 PyTorch row count 181 with raw rectangular
   and unpacked-cache storage through 225. Every row/node uses its own raw chronological repetition
   history and exact target sampler transform, with stable full-vocabulary
   top-k/sampling over 1 through 1,025 simultaneous lanes; a host-selection,
   readback, or row-by-row proxy does not satisfy this item. A linear buffer
   masquerading as a tree does not satisfy it either.
8. Put Metal allocation, encode, commit, wait, readback, restore, and finalize
   behind a deterministic failure-injection seam. Test every accepted
   prefix/path, callback/stop/budget/tile boundary, minimum
   valid prefix, early/middle/sliding-wrap/global-context-tail position,
   transaction failure, and numeric/K/V identity for each surviving topology.
   Include 513- and 1,025-row root-heavy DDTree/CaDDTree fixtures and a
   TreeFlash root whose selected leaves require sibling ranks above 256, proving
   that degree/rank limits are not silently clipped. Exercise exact maximum
   builder fixtures for PCTree pool 969; TreeFlash released/code-transfer/
   paper-transfer semantic states 225/897/961, physical rows 240/960/1,024,
   and rank-storage cases 3,840/15,360/61,440/491,520/492,032/524,288; DARTree
   supertree/scored-row bounds 512/2,048 and
   30,784 total score events, Domino generated/live/frontier/explicit-score/
   generated-ledger bounds 512/449/64/16,384/7,680 plus its full-V streamed
   domain, FullPath scaffold 1,025, ReDrafter raw/packed/KV bounds 225/181/225,
   EAGLE-3 raw/final bounds 711/60 and 10-by-80 expansion mask, and DBlast trie
   121.
9. Measure live perfect/scripted outcomes with all selection, restoration,
   callback, endpoint, fallback, and command costs on identical trajectories.

Exit: every topology is either production-correct and directly measured or has
an uncertainty-aware favorable upper-speed bound below the registered
threshold. At least one must retain a perfect-proposal ceiling capable of the
product objective for an external or target-integrated path. The archived
width-four result is not an exclusion certificate.

### Phase 3 — prepared plan, causal features, and joint caches

1. Define and implement the bounded artifact envelope/schema, authenticated
   loader, and deterministic fixture-artifact writer before PreparedGemma2Plan
   depends on them; final trained packing remains Phase 6.
2. Introduce PreparedGemma2Plan and the verified EngineModel payload receipt.
   Hash the whole GGUF, canonical consumed metadata, source tensors, and every
   resolved/transcoded resident target byte. Target-only mode needs no
   artifact; learned mode validates the complete artifact/bundle descriptors
   before allocation. Phase 3 atomically uploads the target plus only the
   fixture feature/cache substrate defined here; the concrete external, router,
   and Arm-S plan types are built in Phase 6 and atomically composed at session
   setup in Phase 7. Validation order is fixed without pretending those later
   plans already exist.
3. Add typed FeatureBankSpec/TargetStateView provenance. Pass an optional
   `FeatureSinkSet`/binding into prefill, ordinary, K/V-only, linear-verifier,
   and forest encode loops so target execution can stream shared capture plus
   every eager child projection into state owned outside the target plan without
   cyclic ownership. Track each lazy child's stale frontier and charged rebuild.
4. Capture/fuse/project every required prompt and generated row for shared,
   layer-wise, and hybrid feature controls without changing target output.
5. Build a parameterized draft-cache substrate sized to ExternalHmax=23,
   VerifierRowMax=1,025, and
   the finite registry's maximum depth/width for fixtures and bring-up. In
   Phase 6, every member that reaches live selection receives its own
   artifact-resolved final-shaped packed allocation, memory ledger, and warm
   identity; Phase 8 never times a generic oversized layout and then assumes a
   selected specialization will behave the same.
6. Extend the Phase 2 target ticket into the sole atomic target/external-draft/
   Arm-S ticket. Implement target-p versus draft-(p+1) coordinates, independent
   masks/victim ranges, origin-clamped deferred proposal suffixes, and bounded
   pending-stream-bootstrap/prior-tree/next-tree lease ownership. Every chunk
   of multi-command prefill belongs to this ticket; partial first, middle, or
   final chunk completion can never leave canonical target/feature/draft K/V
   outside the rollback/rebuild/poison decision. Inactive components remain
   empty typed subtransactions rather than alternate commit APIs.
7. Extend commit/finalize/discard, cached-turn continuation, regeneration,
   clear/reset, sampler reconfiguration, context rebuild, sliding wrap, global
   boundary, cross-child switch/replay, and every terminal K/V hook. Promote
   selected scratch rows into owning TargetStateLease storage before finalize.
   Exercise allocation/encode/commit/wait/readback/restore/finalize failures at
   every prefill chunk and every bundle child through the deterministic backend
   seam. Add bounded trace-only readback.
8. Implement `speculation_trace.*`, its bounded writer/reader/auditor, and a
   trace-producing developer CLI before any training collection. Wire their
   sources and deterministic malformed/round-trip/causality tests into the
   Makefile and CI; Phase 5 consumes this implementation rather than inventing
   trace infrastructure after collection.
9. Release bulk CPU staging after upload and measure startup, TTFT, prompt
   throughput, feature-bank/cache traffic, and every lifecycle path with
   capture disabled/enabled.

Exit: capture off is behaviorally identical to Stage 4. Capture on has scalar
and Metal parity, all-row prefill bootstrap, no future leakage, correct
provenance/offsets, independent rebuild equality for draft K/V, atomic failure
restoration for known-complete recoverable failures, mandatory poison/
destruction/rebuild for unknown completion or failed restore/finalize, and
measured prompt/startup cost. Deterministically injected failure after any
multi-command prefill chunk or child-plan operation leaves either exact pre-call
state or a publicly visible poisoned session—never a reusable partial cache.

### Phase 4 — production-shaped vocabulary and scoring kernels

1. Implement the **drafter** selector's full-V projected scan for one through
   ExternalHmax states, with specialized 1/3/4/7/8/15/16/23-state forms in F16 and
   every registered packed format. H16 is a real state batch for TreeFlash's paper-description transfer,
   PCTree, and JetSpec's one-pass latent-mask extension, not padded H15; JetSpec's pinned
   executable-source control is independently H15. The target verifier's rooted-row
   full-V kernels already belong to Phase 2 so its topology timings are real;
   do not move them back here or let a replay stand in for them.
2. Implement stable top-k, dynamic history/candidate unions, device
   deduplication, transition expansion, and 1K through 128K/full-V controls.
3. Implement indexed F16/Q8/Q6/Q4 artifact-bound packed-column dot products
   without materializing a k-by-3,584 matrix. Q8 may alias the resident target
   head; all other formats own authenticated candidate bytes and are charged
   in residency/startup/traffic.
4. Add target-space up projection, final normalization/softcap,
   selected-token correction scores, the artifact-bound GPU proposal transform,
   and proposal selection.
5. Implement full-vocabulary corrected scoring and CPU goldens that emulate
   proposal arithmetic exactly. TreeFlash released/code-shaped controls execute
   the real target head on all H*M physical rows: 240 for released H15/M16,
   960 for H15/M64, and 1,024 for H16/M64. The paper-description H16 path may
   use its 961 semantic rows. None may use d-prime instead of evaluating its
   full target head; a 225/897-row code optimization is separately timed only
   after exact duplicate-root elimination proof. After that scan, reproduce
   executable topK16 truncation exactly for the released control. Keep H15
   topK64 and FullVLazy as separately labeled transfers. Only PaperFullVLazy
   requires lazy stable ranks through B=512. Tile full 256K scoring/top-rank
   without a dense matrix and prove identical constructor pop/tie order to the
   matching CPU golden. The H16 paper-semantic path is 881,721,344,000 MACs
   (about 1.763 TFLOP); the H16 code-shaped physical path is
   939,524,096,000 MACs (about 1.879 TFLOP). Their forbidden dense-F32 slabs are
   about 0.916 and 0.977 GiB. Bind physical cached-rank cases
   `240*16=3,840`, `960*16=15,360`, `960*64=61,440`, optional H15 full-V
   `960*512=491,520`, H16 paper full-V `961*512=492,032`, and H16 code-shaped
   full-V `1,024*512=524,288`; a streamed alternative records every rescan cost
   and remains allocation-free.
6. Implement Arm S's real target-Q8 full-vocabulary stream-head path for each
   legal per-entry survivor-by-gamma shape: the local lossless gamma4/k3-plus-32
   cross-mode diagnostics have 160 or 165 total upper-head rows under the two
   root conventions; the deep gamma23/k1 transfer has up to 552 incremental
   stream states; and the lossless threshold-only gamma4/k3 transfer has up to
   605 total upper main-plus-stream distributions. The non-product Shared
   Table-3 source fixture retains its unresolved per-task gamma/k and is not
   assigned these lossless costs. Use tiled final norm/softcap, exact
   per-stream stable top-k, rank-8 transition-pruner inputs, compact tree output,
   and a post-walk selected-anchor control. Separately charge all 121 raw lower-
   layer main rows and the realized surviving upper main rows. The unreachable
   5,888 Cartesian safety envelope is not a benchmark cell. Measure FLOPs,
   scratch, head reuse/traffic, and all-anchor latency with production weights
   before training can exclude a shape.
7. Measure fixture-weight recurrent-per-edge and parallel-batched kernel shapes
   with live target features and candidate distributions. Label these as
   component timings/lower bounds, not trained end-to-end results.

Exit: CPU and Metal agree exactly or invoke the declared conservative proposal
fallback for candidate IDs, ties, transforms, and complete fixture rollouts.
No family is stopped here unless its registered optimistic live-system bound
fails.

### Phase 5 — immutable traces and faithful training

1. Use the already tested Phase-3 trace vNext writer/reader/auditor to publish
   immutable source-cluster split manifests, then collect natural-stop,
   full-context data on the geometric scale rule.
2. Train every unexcluded Arm R/D/P/L/T/S registry member using exactly the
   frozen seeds and convergence/restart rules, including the target-integrated
   stream identifiers/adapters and rank-8 pruner, each separately labeled
   learned-initializer ablation, and every bundle router declared in the finite
   registry. The paper-primary Arm-S member keeps `f_eta=Identity`.
3. For Arm D, instantiate the actual random-anchor masked-block/sparse-mask
   objective and one-pass inference distribution; train faithful DFlash,
   DFlare, and B16-continuation/B24-DBloom cells through their declared data
   scales, progressive/fixed/D-PACE losses, tap fusion, and heterogeneous-K/V
   controls. Train no generic parallel head and relabel it block diffusion.
4. Train ReDrafter's source control with teacher-forced multi-horizon
   recurrence over shifted ground-truth/target-generated token embeddings and
   its exact depth-shifted CE; keep an own-token rollout as a labeled ablation.
   Train EAGLE-3's beam/dynamic-tree member with its mandatory training-time
   self-rollout and per-node hidden state. Train JetSpec's H15/H16 single block-
   causal latent rollouts over anchor-plus-mask positions and separately cost
   the H16 dependent argmax-feedback diagnostic; TreeFlash's teacher-forced immediate-predecessor
   approximator; FullPath-T's concrete-path refiner; and DBlast's exact NLL/AL
   control matrix. PCTree, DominoTree, and DARTree reuse their declared frozen
   corrected backbone/head and alter construction only; DDTree and CaDDTree
   reuse branch-independent marginals. A constructor is called training-free
   only when its source method is.
5. Train selectors jointly on actual deployed drafter states and compare both
   registered token-prefix curricula.
6. Evaluate raw, uncalibrated conditional prefix outcomes for deterministic and
   common-draw policies, every registered d-prime/k/precision cell, transition
   union, and full-V control.
7. Produce source-cluster learning curves and favorable uncertainty bounds;
   preserve every seed and failed member rather than selecting a convenient
   run.

Exit: every registered family proceeds to at least one complete packed live
implementation unless its upper joint-progress/lower complete-live-cost bound
is already below the product threshold. Teacher loss, replay, shortlist recall,
or isolated kernel time cannot block Phase 6.

### Phase 6 — candidate artifacts and complete Metal drafters

1. Use the Phase 3 authenticated envelope/loader; implement the training
   checkpoint converter, deterministic packer, independent packed-artifact
   inspector, and exact CPU golden executor, then package every still-eligible
   standalone candidate and heterogeneous bundle as nonselectable
   `WeightsArtifact`s. Provisional and final calibration parameters remain
   unset. Retain an authenticated file-backed source mapping for every packed
   weights/router/Arm-S payload needed by a later rebuild.
2. Allocate external-arm weights, semantic/recurrent state, selector state,
   scratch, compact output, and parameterized K/V persistently in
   `Gemma2SpeculativePlan`. Own Arm S adapters, transient streams,
   raw-prior/surviving/next-tree leases, lower-tree/pruning/upper-MSA workspace,
   and MSA pipelines in a dedicated
   `Gemma2MultiStreamPlan` composed into the target plan and its target ticket;
   allocate its artifact-resolved RawPrior/Survivor/UpperRow/SNextNode/
   SRawChild maxima and do not
   pretend it is a pre-target external drafter.
3. Encode every registered eager/conditional/dynamic d1-through-dH schedule:
   feature/context
   use, backbone, recurrence/correction, selector, indexed score, proposal
   transform, token choice, and causal confidence.
4. Keep intermediate proposal IDs on device with explicit barriers. Drafter
   readback is the bounded tagged `ProposalGraph`: chain members use
   ExternalHmax; forest members use `VerifierRowMax` row metadata,
   `MaterializedTreeEdgeMax` parent edges, and `OutputFrontierEdgeMax`
   output-only proposals with actual counts; repeated-root ReDrafter members
   use their explicit beam/root/rectangular-cache extents. DDTree/CaDDTree/TreeFlash best-
   first ranks and independently capped B+1 live heaps/cursor queue, PCTree's
   pre-prune pool, TreeFlash's corrected-state/rank scratch, DominoTree's
   sequential generated/live heap plus depth-synchronous active/conditional-
   score/generated-ledger/lane-step storage, and DARTree's supertree/per-depth score scratch are
   independently typed and bounded; none aliases verifier or
   output-frontier storage. It also
   contains route/status and bounded draft diagnostics.
   Target candidates live in distinct ordinary or verifier result types and
   never in the drafter result.
5. Implement the registered Arm-D DFlash/DFlare/DBloom one-pass masked-block
   kernels; DDTree marginal best-first and CaDDTree's same marginal heap plus
   frozen C_d/C_v adaptive stopping; DBlast latent mixtures and both controllers;
   PCTree's parent-conditioned DSpark Markov constructor;
   ReDrafter/EAGLE3 recurrent beam/dynamic trees with per-node hidden/cache/
   selector work; DominoTree's sequential CPU oracle and exact
   depth-synchronous W=B Metal best-first correction;
   DARTree depth-batched correction plus fixed and deferred-pruned supertrees;
   TreeFlash immediate-predecessor two-stage construction; JetSpec's H15/H16
   shared causal latent/mask-state depth distributions and separately labeled
   sequential H16 argmax diagnostic; FullPath-T's concrete-path second
   pass; and Speculative Streaming's lower-main/prune/upper-MSA paths as actual packed
   proposal/tree constructors—not replay adapters around an R/D/P chain. Arm S
   includes the non-product Shared Table-3 32-passed-node source-resolution
   fixtures, separately labeled lossless-32 cross-mode diagnostics, every legal
   threshold-only survivor-by-gamma full-head/top-k path, raw lower-tree work,
   Section-3 identity-initializer/rank-8-pruner fields,
   the exact-seed product
   controller, the separately labeled virtual-distribution literature control,
   and the post-walk cost control.
6. Build bundle construction, atomic upload, route dispatch, CPU/Metal parity,
   malformed/incompatible-child failure, poison/rebuild, and full-residency
   accounting tests for every registered heterogeneous candidate.
7. Warm every learned pipeline/topology state-neutrally and prove zero hot
   allocation, compilation, upload, or unbounded artifact growth.

New ownership boundaries:

- learned_speculation_artifact.* — bounded format and identity;
- heterogeneous_speculation_bundle.* — ordered child graph, router, and atomic
  residency contract;
- gemma2_speculative_plan.* — resident learned weights/cache/scratch;
- gemma2_speculative_runtime.* — Metal encode/completion and feature-cache work;
- gemma2_multi_stream_plan/runtime.* — Arm S target-integrated MSA,
  prior-tree/next-tree pipeline, and pruning ownership;
- speculative_controller.* — backend-independent state machine;
- speculation_trace.* plus the Phase-3 offline writer/reader/auditor — immutable
  causal records, without framework code in the runtime;
- PreparedGemma2Plan/model identity — CPU validation before atomic upload.

Do not put learned math into the CLI or grow gemma2_runtime.cpp into the
artifact, trainer, controller, and cache owner.

Exit: every unexcluded registry family has a real packed CPU/Metal-parity
proposal chain/tree/stream pipeline and production-shaped full-path timing.

### Phase 7 — session integration, failure safety, and observability

1. Add explicit standalone-artifact or bundle path/policy and independently
   controllable retrieval, learned-family, target-integrated, and
   all-speculation options to session and CLI APIs.
2. Construct the target, every bundle child, router, external speculative plan,
   and optional Arm-S multi-stream plan atomically from PreparedGemma2Plan
   during setup; never compile/upload on the first generated token. Remove or
   close the public component-wise `Gemma2ChatSessionState` construction escape
   hatch and test file/model, tokenizer/template, target-plan/output-policy,
   learned-child, and artifact-epoch mismatches.
3. Integrate retrieval-first/anchored routes, heterogeneous causal routing,
   all R/D/P/L/T/S paths, all target topologies, and all draft schedules behind
   the pure controller.
4. Revalidate or explicitly disable learned routing on sampler changes.
5. Extend warmup, clear/reset, regeneration, cached-turn continuation, context
   rebuild, reverse stop, interruption, and the entire failure taxonomy.
6. Add the chat-session-owned mutation journal/commit receipt and outer
   pre-mutation TurnStartCheckpoint. Wrap all of `generate_user_turn` and
   `regenerate_last_turn`; restore/rebuild full messages/turn-checkpoint/
   sampler/history/cache/lease/decoder/reverse-stop state after every host or
   known-complete device exception, and poison after unknown completion or
   failed reconciliation. Test failure before/during context-shift prefill,
   every prefill chunk, every callback depth, final decode/text assembly, and
   every bundle child without deep-copying full session state on the hot path.
7. Add separate, non-overloaded clocks. `generation_completion_duration`, the
   primary speed clock, starts at the synchronized post-prefill-ready boundary
   immediately before authoritative x0 selection/callback and any generation-
   phase route work; it ends only after terminal K/V-only work, the final
   callback, UTF-8/reverse-stop decoding/finalization, and backend
   synchronization. `full_turn_duration` starts immediately before the user-
   turn mutation and includes prompt/context processing through that same end.
   TTFT retains its own start/end receipt. Retain the legacy first/last-
   selection metric separately for canonical history. Candidate work moved
   into prompt preparation remains visible in full-turn, TTFT, and prompt
   gates; it is never silently omitted from all headline clocks.
8. Emit schema-versioned topology/route/resource metrics and update the CLI,
   benchmark header, JSONL writer/parser, validation tests, and quick benchmark.
9. On training/architecture-development data only, run complete live paths to
   fit and freeze each entry's preregistered provisional route/tile policy;
   emit a content-addressed `SelectionArtifact`/bundle referencing the unchanged
   weights before Phase 8.

Exit: one release binary runs ordinary, Stage 4, capture-only, learned-only,
hybrid, and every registered verifier control with fail-closed lifecycle tests.

### Phase 8 — live architecture-selection validation and attribution

Run complete live paths for every unexcluded registry entry on the untouched
architecture-selection validation partition: Stage 4; maintenance-only;
perfect proposals; every registered conditional tile partition; linear widths
3/4/5/7/8/9/15/16/17/23/24; every unexcluded typed-budget forest; every Arm
R/D/P/L/T/S F16 and packed member; every heterogeneous bundle;
deterministic/common-draw; full-V and every shortlist; retrieval policy; draft
schedule; and fused/host/GPU-gated command topology. GPU gating may be excluded
only by a recorded Metal capability failure or its registered bound. Every live
route/tile decision uses the development-fitted provisional policy frozen at
Phase 7.

A provisional-policy result may select which architecture advances, but it may
not close an architecture whose Phase-9 calibratable policy could improve it.
Any Phase-8 rejection certificate must upper-bound performance optimistically
over the **entire preregistered `CalibratablePolicyParameters` space**, including
route, tile, horizon, shortlist, and pruning thresholds, with uncertainty and
all runtime costs. A bound at only the fitted provisional point is invalid. If
that policy-uniform certificate cannot be proved, at least one candidate
advances; a complete-family rejection instead requires preregistered per-entry
policy fits on an independent calibration partition followed by simultaneous
bounds on a further untouched closure partition. That closure partition is
neither Phase 8 nor the winner's sealed admission data.

Use a separately frozen crossed source-by-calendar-day design, atomic product
unit, thermal regimes, arm-blind invalidation rules, and joint two-way-cluster
max-t procedure with simultaneous candidate-by-regime bounds—the same
inferential structure specified below for admission, but on disjoint selection
sources and dates. The manifest must declare the operating-day population and
date-selection rule; source-only uncertainty cannot select an architecture from
day/session noise. Replay may explain a live result but cannot select an
unexecuted winner. Fit no final route threshold on this split. Freeze exactly
one complete architecture/checkpoint, precision, selector, draft schedule,
target topology, and routing feature set.

Exit: exactly one candidate advances, or simultaneous upper bounds plus valid
policy-uniform certificates reject every registry member against 1.30. A
backup is development evidence only and requires its own untouched sealed set.

No family is introduced or retrained from Phase 8 evidence. R/D/P/L/T/S,
registered tree constructors, and heterogeneous bundles must already have
completed Phases 2–7 or carry valid preregistered certificates. An unforeseen
architecture discovered here invalidates this partition for selection and
requires the source-disjoint replacement rule above.

### Phase 9 — calibration-only fit and final freeze

1. With architecture fixed, refit only the declared
   `CalibratablePolicyParameters` on the calibration split using the frozen
   algorithm, complete live costs, and causal production fields.
2. Validate probability calibration, every tile/branch/arm decision, and
   declines by workload diagnostically while poisoning all evaluation metadata
   inputs.
3. Emit/content-address the `AdmissionArtifact` over the unchanged selected
   weights plus final policy, and freeze binary, router, thresholds, topology,
   model, statistics program, environment rules, fixed N, and sealed manifest.
   No architecture or weight changes follow.

Exit: one immutable production candidate is registered for one sealed study.

### Phase 10 — sealed admission and retention

1. Run the fixed-N paired Stage 4/candidate study with balanced randomized
   AB/BA order, fresh processes, and both predeclared thermal regimes.
2. Evaluate exact identity, inclusive end-to-end time, weighted workload and
   context claims, full resource counters, and every **sealed internal Stage-4
   gate** once.
3. If and only if the sealed internal gates pass, run the canonical cooled
   two-turn retention gate with the same synchronized inclusive-completion
   clock. If and only if that gate passes, run the separately pinned llama.cpp
   comparison.
4. Merge only a candidate that passes all three ordered decisions and its
   tests/metrics/artifact contract/docs. Remove
   rejected temporary instrumentation while retaining authenticated evidence.

Exit: the admission and stop conditions below decide the baton.

### Repository touch points

The implementation is expected to touch these ownership seams:

- `engine/model.hpp`, `engine/model.cpp`, `engine/gemma2_model.*`, and
  `model_format/gguf_reader.*`, `model_format/gguf_metadata.*`, and tensor views
  for a complete file/tensor/canonical-metadata receipt, lifetime-safe mapped
  bytes, requested/resolved output policy, and staged-Q8 release/recreation;
- `tokenizer/tokenizer.*` and chat-template/session construction so tokenizer,
  special-token, and exact template bytes participate in artifact/trace
  identity rather than only a coarse architecture name;
- `engine/gemma2_metal_plan.*`, `engine/gemma2_runtime.*`,
  `engine/gemma2_chat_session.*`, and `engine/sampling.*`, plus the new
  controller/artifact/bundle/speculative-plan/speculative-runtime/trace units;
- `tensor/backends/metal_backend.mm`, backend buffer allocation seams,
  `metal_context.*`, Metal buffer/command-context headers and implementations,
  deterministic failure injection, and new Metal kernels for feature streaming,
  indexed scoring, multi-row/tree attention, MSA, packing, and transactions;
- benchmark headers/emitters, CLI option/schema code, SHA-256 utilities,
  `scripts/quick_benchmark.py`, `scripts/chat_benchmark_common.py`, and
  `scripts/memory_benchmark.py` for inclusive timing and route/resource output;
- new offline trace generation/audit, training, checkpoint conversion,
  artifact packing/inspection, calibration, power-simulation, bootstrap, and
  experiment-manifest scripts, each with deterministic fixture tests;
- unit/integration/golden/CLI/script tests and `tests/test_main.cpp`
  registration, Makefile source/test lists, `scripts/ci.sh`, `.gitignore`, and
  `docs/GRAPH.md`.

Replace the current fixed `std::array<...,4> allocation_identities()` plan and
session APIs with a typed `PlanAllocationIdentity` tree that separately hashes
target resident buffers, speculative caches/scratch, each bundle child, router,
and transaction arenas. Warm-state tests compare the complete tree. This list
prevents a hidden feature from being wedged only into `gemma2_runtime.cpp`; it
is not permission to alter an unrelated layer.

### Dependency graph

~~~text
identities + exact tap/sampler/registry/statistics contracts
  -> pure controller + sampler/device-history semantics
  -> measured target topology frontier + target-only ticket
  -> CPU prepared plan + all-row feature capture + parameterized draft cache
  -> joint target/external-draft/Arm-S/bundle ticket
  -> trace vNext ------------------------------+
  -> selector/scorer production kernels -------|
                                                v
                           registered R/D/P/L/T/S + router training
                              -> weights artifacts + Metal drafters/MSA
                              -> session/controller/live topology integration
                              -> development policy fit + SelectionArtifacts
                              -> architecture-selection validation
                              -> calibration-only route fit + final artifact
                              -> fixed-N sealed direct admission
                              -> canonical + llama.cpp retention comparison
~~~

Trace collection and production-kernel work may proceed in parallel only after
their shared taps, offsets, precision, sampler, artifact, and registry contracts
are frozen.

## Required correctness matrix

### Pure controller

Cover the cross-product, reducing only provably equivalent cases:

- accepted edges 0 through each registered H=3/4/7/8/15/16/23, including every tile and
  tree-depth boundary;
- remaining budget 1 through H+1 and greater than H+1;
- stop at x0 through xH;
- callback continue/stop/throw after every deliverable token;
- route decline, retrieval-first, retrieval-anchored, and learned route;
- linear/forest/repeated-root-beam ProposalGraph paths, output-only frontier carry, materialized
  leaf correction, and every invalid graph rejection;
- Arm-S pending prefill bootstrap, post-x0 lease binding, steady mismatch/full
  match, pending full-match
  correction binding, attached-tree publish/discard, missing/incompatible
  lease, intentional switch, endpoint, callback stop/throw, poison/reset,
  drain/rebase, and shutdown;
- every tile gate accept/decline, every draft schedule, and the ordinary carry
  fallback at each boundary;
- context capacity at each call and every x0-through-xH endpoint;
- deterministic, common-draw, and greedy proposal policies.

Assert exact selected/generated/visible vectors, callback sequence, sampler
checkpoint, retrieval/draft history, retained rows, carry, next candidates,
feature source, cache position, and stop reason.

### Sampler

Cover:

- span histories of length 0 through ExternalHmax at repetition-window positions 0,
  1, 63, and 64, including suffixes that cross the 64-token window boundary;
- repeated IDs, history deduplication, top-k/history candidate union, and stable
  ties;
- greedy and stochastic paths;
- draw tickets against the unchanged current sampler for many seeds,
  distributions, totals, and draw boundaries;
- raw engine outputs, copied/post engine state, total-mass mapping, exact binary
  doubles, and GPU-proposal arithmetic/fallback goldens;
- DeviceSelectionRequest duplicate/nonduplicate suffixes at history positions
  0, 63, and 64 with raw chronological eviction, multiplicity, and exact
  dynamic base-candidate counts;
- checkpoint adoption for every exit;
- Arm S ticket slices for every possible accepted anchor through the
  `2*gamma+1` frontier, proving equality with draws previewed directly from that
  anchor's eventual checkpoint;
- no extra **authoritative** RNG draw after a full match or callback stop;
- no heap allocation in the steady proposal loop.

### Target verifier and caches

Cover every registered conditional tile partition, linear width
3/4/5/7/8/9/15/16/17/23/24, and unexcluded source-budget/root-convention/k/M/K/W forest against independent sequential
ordinary execution at:

- minimum valid real prefix, early, middle, sliding-wrap, global-context-tail,
  and exact context-boundary positions;
- every accepted prefix/node path and every terminal uncached-token endpoint;
- distinct row histories and proposal tokens;
- zero/multiple roots in a unique-root topology; malformed/nonidentical/
  cross-visible roots in `RepeatedRootBeamBatch`; cycles, disconnected nodes, child-before-parent,
  parent/depth mismatch, duplicate siblings, over-H depth, node/edge/branch
  overflow, invalid row role, token/position/mask/history/ticket bounds, and
  legal deterministic sibling merge;
- poisoned output, scratch, independent target/draft backups, and ring victims;
- failure at allocation/encode/commit, before submit, during wait, after
  completed readback, restoration, and finalization—including every first,
  middle, and last multi-command prefill chunk and every bundle child;
- callback rollback after every provisional row;
- callback-throw commit receipts at every row/node, followed by session reuse
  when reconciliation succeeds and mandatory poison/reset when it cannot;
- repeated cycles proving no stale backup or feature epoch;
- bit-identical committed target K/V or its canonical recomputation, byte-exact
  target restoration, and draft K/V equality against independent rebuild and
  CPU/Metal goldens.

For H=3 call two, and for **every** later H=4/7/8/15/16/23 tile, the command must be
observably absent after an earlier miss, stop, budget endpoint, callback stop/
throw, gate decline, or insufficient context. On every gate/context decline,
the authoritative carry must take the ordinary logits-producing path whenever
generation continues; no deeper proposal/tile state may advance.

### Features and learned execution

Cover:

- every TargetStateView row mapping, owning TargetStateLease promotion/pinning,
  finalize/reuse lifetime, and stale/overwritten epoch rejection for ordinary,
  linear, and forest rows;
- all 42 block outputs plus registered embedding/final-normalized endpoints,
  exact three/five subsets, the faithful nine-tap DFlare maps, every wider
  DFlare per-draft-layer mixture map, heterogeneous source-specific K/V, and every
  DFly global/layer-specific/predecessor-conditioned mapping across prompt,
  ordinary, verifier, forest, terminal, and rebuild rows; resident-versus-trace
  precision/rounding parity and capture-disabled target identity;
- poison-future causality and previous-proposal sensitivity;
- origin-clamped deferred-d_i identity at every tile boundary when intervening
  verifier features are poisoned;
- target-p/draft-(p+1) offsets, separate wrap victims, terminal projection,
  cached-turn continuation, and context rebuild;
- heterogeneous eager-child maintenance and lazy stale/replay behavior across
  every child switch, accepted/mismatch row, rollback, wrap, reset,
  regeneration, and rebuild, with per-child frontier equality;
- full-context attention bounds and no speculative/future cache visibility;
- Arm S prefill bootstrap, steady prior-tree verify-plus-next-tree emit,
  full-match pending-correction publication, and the normal **no-extra-target-
  pass** mismatch/terminal-carry continuation with the emitting row's attached
  next tree; test drain/rebase only for a missing/incompatible lease,
  intentional route switch, or endpoint, plus callback/stop discard,
  reset/rebuild, stale-lease rejection, and poison recovery for every
  Ns/gamma/tree shape;
- Arm S Section-3 identity-initializer and rank-8 early-exit equation goldens;
  the non-product Shared Table-3 literal 32-passed-node fixture under every
  predeclared root/count/threshold/cap/score/tie interpretation; separately
  labeled lossless-32 cross-mode cases; and a 121-survivor lossless
  threshold-only boundary distinct from all of them;
- bit-identical Arm S main-stream logits and target K/V with MSA disabled versus
  active lossless streams, plus proof that speculative streams allocate no
  persistent K/V and cannot influence main-stream attention;
- CPU/Metal feature fusion, one-pass masked-block diffusion, recurrent,
  semi-parallel, selector, every indexed
  selected-table format, correction, latent/tree construction, MSA/pruning,
  transform, and route parity;
- F16 and packed close-score/tie cases, vocabulary tails, odd dimensions,
  Q4 negative nibbles/scales, and nonwinner rows;
- complete d1-through-dH rollout and every registered tree constructor rather
  than independent one-step outputs;
- malformed/nonfinite/overflow artifacts and unavailable Metal;
- warm-up state neutrality and persistent allocation identities.

### Public path

Cover CLI/session success and failure for:

- valid/missing/mismatched standalone artifact and heterogeneous bundle,
  including incompatible or absent child/router payloads;
- each learned family on/off, target-integrated on/off, retrieval on/off, and
  all-speculation off;
- sampler reconfiguration;
- clear, reset, regenerate, context shift, reverse stop, interruption, and
  stream callback exception;
- post-callback-exception reuse/reset for ordinary, Stage 4, every linear/tile/
  tree route, and Arm S, with messages, turn checkpoints, visible decoder/
  reverse-stop state, host/device history, and sampler equality;
- pre-context-shift and pre-regeneration exception restoration, every
  non-callback host/device failure class, public poison gating, and authenticated
  rebuild after learned-artifact deletion/replacement/truncation;
- prepared-plan construction mismatch for GGUF metadata/config, file/model,
  tokenizer/template, target plan/output policy, artifact child, and epoch;
- JSON schema totals and raw-token/SHA-256 fields;
- inclusive completion/full-turn timing versus the legacy canonical timing;
- exact baseline equivalence when learned routing is disabled or declines.

## Observability contract

Replace the fixed one-edge metrics with additive route/cascade accounting:

- total opportunities and ineligibility reasons;
- retrieval, R/D/P/L/T/S, heterogeneous-bundle child, anchored, scouted, and
  declined routes, including all abandoned scout costs;
- draft length and accepted-prefix histogram 0 through actual H, including
  native-block ceiling-bin mass and DBloom expansion deltas;
- per-edge conditional match probability and cumulative prefix survival for
  edges 1 through H, with uncertainty and reached-origin denominators;
- verifier topology, node/row count, draft schedule, per-tile target-call
  counts, ordinary carry fallbacks, and every skipped-call reason including
  tile-specific confidence decline;
- carried tokens, endpoints, stops, callback exits, and rollback rows;
- Arm S prior-tree consumes, next-tree candidates/publications/discards,
  bootstrap/drain/rebase calls, pruned nodes, and maintenance-only work;
- feature, draft backbone, selector, selected scorer, correction, target,
  selection, readback, and restoration wall time; command-level production GPU
  time plus separately labeled non-additive component diagnostics;
- commands, commits, waits, encoders, dispatches, barriers, allocations, and
  bytes by category;
- resident memory per target/child/router plus bundle total, RSS, memory
  pressure, compression, and swap;
- cache/context bucket, thermal window, and route confidence/calibration;
- raw selected tokens, SHA-256, existing FNV hash, final cache, and sampler
  state fingerprint;
- primary `generation_completion_duration`, separate `full_turn_duration` and
  TTFT clocks with start/end receipts, and the separately named legacy
  first/last-selection duration.

Emit per-invocation scheduled-byte totals. Do not infer learned or verifier
traffic by multiplying ordinary per-token bytes.

## Statistical and thermal protocol

### Primary statistic

Let j identify a frozen candidate; admission has exactly one j. Let
`h in {cold,sustained}` identify the two co-primary thermal regimes. Let s index
mutually exclusive predeclared workload strata with product weights w_s that
sum to one, g the highest independently sampled source cluster in stratum s,
and d a calendar-day block. Every source is run on every selected day and in
both regimes, so source and day are crossed and regime is explicit rather than
silently pooled.

The manifest defines the atomic product unit before data are opened. One unit a
is one exact natural-stop conversation-and-sampler-seed record; a single-turn
prompt is a one-turn conversation, and all turns of a multi-turn conversation
stay in that unit. It fixes prompt/template bytes, turn order, seed, sampler,
maximum-token rule, whether a selected EOS is counted as a completion token,
and the engine-independent terminal-work boundary. Prompt variants,
continuations, derived windows, and repeated timings are nested measurements,
not independent units. Freeze cluster weights `v_sg` with
`sum_g v_sg=1` within every stratum; if cluster inclusion probabilities differ,
these are the normalized inverse-probability/product-mixture weights rather
than observed sample fractions. Fixed within-cluster unit weights `pi_sga` sum
to one and encode the product mixture rather than allowing clusters with more
variants, turns, or tokens to receive extra weight.

For arm A in `{B=retained Stage 4,C=candidate j}`, let
`t^A_sgdhar` be unit a's **inclusive generation-completion** time on repetition
r. Its start is the synchronized post-prefill-ready event immediately before
authoritative x0 selection/callback and all generation-phase route work. Its
end follows terminal K/V, the final callback, decoder/reverse-stop finalization,
and backend synchronization. The distinct full-turn and TTFT clocks enforce
the prompt-work gates but do not enter S_jh. The fixed technical-repetition reduction is the arithmetic
mean over all r complete repetitions—no trimming or best-of selection—and the
source-cell functional is

\[
A_{sgdh}=\sum_a \pi_{sga}\,R^{-1}\sum_{r=1}^{R}t^A_{sgdhar}.
\]

If a candidate-attributable incomplete run leaves any required t undefined,
the cell and S are not estimated from the remaining runs. Only a censor rule
frozen before data may yield a conservative closure bound; otherwise use the
ordered non-estimable label below.

The primary regime-specific speedup for candidate j is

\[
\widehat S_{jh}=
\frac{\sum_s w_s\sum_g v_{sg}D^{-1}\sum_d B_{sgdh}}
     {\sum_s w_s\sum_g v_{sg}D^{-1}\sum_d C_{jsgdh}},
\qquad
\widehat Q_j=\min_h\widehat S_{jh}.
\]

This is weighted completion-time speedup over identical Stage-4/candidate
trajectories, not a mean of per-run token/s ratios. Raw unweighted sums are
equivalent only if allocation exactly matches all product and inclusion
weights. Context buckets are secondary labels and do not duplicate records in
the primary estimator.

The same conversation, seed, sampler, target-selected token trajectory, model,
release binary, and source-by-day environment are paired. Before either
selection or admission, its manifest defines an eligible operating-day
population and uses a seeded uniform draw without replacement of D distinct
dates from a fixed future test window. Eligibility and any replacement-date
rule depend only on arm-blind environmental measurements fixed before results;
candidate behavior cannot select a date. This makes the day clusters draws from
the stated operating-day population. If that sampling contract cannot be met,
the study must instead preregister a conditional-on-these-days within-day
randomization/CR2 analysis before opening data; it may not resample days or
switch interpretations afterward.

A one-way source bootstrap is invalid. Validate in development and freeze a
studentized max-t two-way-cluster procedure that resamples source clusters
within strata and sampled calendar days independently, takes their Cartesian
cells, and keeps the paired arms and every nested record together. Compute all
candidate, regime, workload, context, TTFT, and prompt contrasts jointly with at
least 100,000 replicates. Let `[L_jh,U_jh]` be the resulting simultaneous
one-sided 95% bounds. Under their joint coverage,
`L_Qj=min_h L_jh` and `U_Qj=min_h U_jh` bound the worst-regime product quantity.
Simulate source effects, day effects, interaction, thermal autocorrelation,
heteroskedasticity, arm-blind invalidity, and the actual small-D regime. If
coverage misses the frozen target, choose and validate a two-way
wild-cluster/randomization or crossed mixed-effects/CR2 alternative before the
partition opens—never after seeing which is favorable.

Report every S_jh, Q_j, simultaneous lower and upper bound, workload/context
bounds by regime, source distribution, day/thermal variance components, and raw
paired rows. Windows, tokens, turns, seeds, and repetitions are never promoted
to independent observations.

Determine fixed n_s, D, and R by crossed-design power simulation of this exact
functional, date sampling, both co-primary regimes, multiplicity, and frozen
failure rules. Require at least 90% simulated power at the 1.35x design
alternative to have every `S_jh>=1.30` and every `L_jh>1.25` while satisfying
secondary gates. At a truth exactly on a point boundary, no finite design can
guarantee a 90% pass probability. There is no interim look or appended tranche;
an extension is a new independently sealed study with a preregistered
combination rule.

### Thermal execution

Headline runs use:

- release build and exact artifact/model hashes;
- fixed AC/power mode, OS build, ambient band, display policy, and background
  workload policy;
- separate fresh processes;
- balanced randomized AB/BA ordering within day/session blocks;
- no profiler, Metal capture, or debug logging;
- all environmental invalidations retained with their declared reason.

Each source-by-day cell receives its own AB/BA thermal pair block; no other
source shares that pair block. Multiple source blocks may occur on a day in a
frozen randomized order, and every source appears on every selected day, so
calendar day is genuinely crossed rather than falsely nested inside source.
Only arm-blind external conditions measured independently of candidate behavior
may invalidate a pair under the frozen symmetric rule. A completed slow run,
including one with a page fault or throttling, retains its measured time and may
also fail a resource gate. A candidate-attributable timeout, crash, OOM,
artifact failure, poison, or missing output has no invented completion time: it
automatically fails the nonperformance gate, is retained in the censor/failure
ledger, and yields no admissible speed claim. Such a member may contribute to
family closure only if the manifest already specified a conservative censored-
time construction that gives it a valid speed upper bound; otherwise it remains
unclosed. Successful route declines and completed ordinary fallbacks are normal
candidate execution and stay in `C_jsgdh`.

The manifest fixes external-invalid replacement before data open. Each planned
r has at most two same-date reserve pair slots. An arm-blind external
invalidity discards the entire Stage-4/candidate pair—not one arm—and uses the
next reserve slot; R counts externally valid scheduled pair units, not attempts.
A candidate-attributable incomplete consumes one such unit but leaves its t
undefined. Only if arm-blind external invalidations exhaust the reserve slots
before a cell supplies R externally valid scheduled units, discard that
**entire crossed calendar-day block** for all s/g/h and rerun it on the next of
two uniformly pre-sampled reserve dates.
Exhausting either cap terminates the study as a design failure with no speed
claim; it never triggers an appended data-dependent tranche. Candidate-
attributable timeout/crash/OOM/poison is retained, consumes its slot, and is
never converted into an external invalidity or replacement run.

Run two separate regimes:

1. cold, isolated paired latency: after fresh-process launch, idle for a
   minimum 240 seconds; poll the arm-blind readiness vector every 5 seconds and
   require 12 consecutive in-band samples. Freeze exact sensors, bands, slopes,
   counter tolerances, and missing-sensor behavior numerically on development
   data before Phase 8. Start within 5 seconds of the twelfth sample. If
   readiness is not achieved within 900 seconds total, externally invalidate
   the whole pair under the reserve rule above—never wait selectively;
2. sustained latency: satisfy the identical cold-start predicate, then run a
   frozen symmetric heat-in workload in that same process and arm mode. Freeze
   its exact prompt/token bytes, sampler, repetition count, work order, and
   terminal backend barrier from development equilibrium experiments. Sealed
   heat-in stops after that fixed work count, never on candidate throughput,
   observed temperature, or a convenient equilibrium crossing. Begin the timed
   workload within 1 second of its terminal barrier with **no** intervening
   idle. Freeze rolling-window width, temperature/power proxies, throughput-
   slope/CV diagnostics, and development equilibrium evidence numerically
   before Phase 8; diagnostics cannot extend sealed heat-in.

Do not let sealed baseline behavior choose or extend heat-in or measurement.
The completion-time ratio therefore remains the defined estimator; a
fixed-duration unequal-token throughput count is not substituted silently.
Both regimes are separately powered and jointly covered; each must clear the
same 1.30 point/1.25 lower-bound performance gates. A cold-only win is not
admissible.

Capture process RSS/compressed bytes, system pressure state, swap totals,
major-fault counts, and thermal/power proxies immediately before/after every
arm. Pre-existing system swap is recorded, not misattributed. Symmetric
environment invalidation limits and counter-resolution tolerances are frozen on
development data; invalid pairs remain visible with their declared reason.

## Ordered admission and retention decisions

### A. Sealed internal Stage-4 admission gates

The direct live engine study is internally validated against Stage 4 only when
all of the following hold:

- `S_jh>=1.30` and simultaneous `L_jh>1.25` separately for cold and sustained;
- no workload or predeclared context bucket has point speedup below 1.00x, and
  simultaneous workload/context lower bounds exceed 0.98 in both regimes;
- define `R_TTFT=B_TTFT/C_TTFT` and
  `R_prompt=C_prompt_throughput/B_prompt_throughput` using the same fixed
  product aggregation; both point ratios and their simultaneous one-sided
  lower bounds are at least 0.98;
- every selected token, visible byte, stop/callback event, sampler checkpoint,
  cache position, target/draft state, and final logical K/V comparison passes;
- no warm allocation, generation-loop major fault, candidate-attributable swap
  or compression growth beyond the frozen counter tolerance, pressure failure,
  or unbounded state growth occurs;
- there is no candidate-attributable incomplete run; every successful route
  decline and completed fallback remains in candidate time;
- the result is produced by the frozen packed artifact and production Metal
  topology, not CPU inference, replay, or an oracle.

Assign one mutually exclusive, speed-only label in this order:

1. **speed not estimable—candidate-attributable incomplete:** at least one
   required candidate time is missing and no preregistered admissible censor
   construction identifies S; this is also an automatic nonperformance-gate
   failure. A conservative censor bound may support family closure only, never
   candidate admission;
2. **strong below-useful-floor:** `U_Qj<1.25`;
3. **misses product objective:** `1.25<=U_Qj<1.30`;
4. **speed-qualified:** every `S_jh>=1.30` and every `L_jh>1.25`;
5. **fixed-N inconclusive:** none of the preceding conditions holds.

Then report gate status orthogonally. **Internally validated against Stage 4**
requires `speed-qualified` plus every correctness/resource/secondary gate.
**Not internally validated—nonperformance gate failed** applies regardless of
an attractive speed interval and rejects only this frozen implementation/
artifact absent complete-family evidence. An inconclusive or candidate-level
speed miss does not reject its family.

### B. Canonical repository retention gate

Only an internally validated candidate reaches the fixed cooled two-turn
retention run. Run Stage 4 and the candidate from the same release binary in
fresh processes under the repository cooldown. The headline generation field
is the Phase-7 `generation_completion_duration`, starting at its synchronized
post-prefill-ready/pre-x0 boundary and ending after terminal K/V, callbacks,
decoder/finalization work, and backend synchronization; legacy
first-to-last-selection time is continuity-only. The candidate must win both
turns, improve aggregate inclusive generation throughput by at least 2%,
preserve the exact workload/token hashes, and regress no other headline metric
by more than 2%. Rerun both arms under this clock; do not compare a new
candidate clock to a historical Stage-4 clock. This pair enforces repository
retention but does not own the 25%/30% product claim.

If it fails, report **internally validated against Stage 4; canonical retention
failed**. Do not merge or tune from the exposed pair; a retry needs a newly
preregistered candidate and untouched sealed data.

Exactly one candidate uses this sealed set. Before any result is exposed, an
alternative design may instead register multiple candidates and execute every
one on every sealed unit with a frozen max-t/closed-testing procedure, but no
backup may be added after seeing a primary result. There is no optional
stopping, post-hoc outlier removal, workload reweighting, threshold retuning,
or sealed-set reuse.

A family-level closure needs simultaneous bounds over candidate and regime:
for every registry member j form `U_Qj=min_h U_jh`, and close the family only if
`max_j U_Qj<1.30`. The observed best member's UCB is selection-biased; every
unexecuted or incomplete member requires the registered valid optimistic
upper-progress/lower-cost or conservative censoring certificate.

### C. Separately pinned llama.cpp claim

Only a canonical-retention winner reaches this external, output-unpaired
comparison. Before running it, preregister fixed process-block count N, turn
weights, calendar-day/process clustering, AB/BA randomization, EOS/completion-
token convention, timing boundaries, and arm-blind invalid-pair rule. For block
b and turn t define

\[
Y_{bt}=\log\frac{N^{candidate}_{bt}/T^{candidate}_{bt}}
                    {N^{llama}_{bt}/T^{llama}_{bt}},
\]

where N counts every generated target token under the frozen engine-independent
rule (including a selected terminal EOS if that rule says so, never prompt
tokens) and T starts after synchronized prompt/prefill readiness and ends only
after the last token's K/V, stop/callback, decoder/finalization work and backend
synchronization. The aggregate ratio is the exponential of the frozen weighted
mean Y; use a day/process-clustered simultaneous one-sided 95% interval over the
aggregate and turns. Historical first-to-last-selection time is secondary only.
The new clock invalidates the old pinned measurement, so update
`docs/LLAMA_CPP_REFERENCE_BASELINE.md` and rerun both the pinned commit and
candidate symmetrically.

Assign exactly one result in order: (1) **beats llama.cpp claim established** if
aggregate point throughput is at least 1.02x, its lower bound is above 1.00x,
and each turn point is above 1.00x; (2) **claim failed at fixed N—turn-point
gate failed** if any turn point is at most 1.00x; (3) **claim failed at fixed
N—aggregate demonstrated below target** if the aggregate upper bound is below
1.02x; otherwise (4) **beats llama.cpp not established—fixed-N inconclusive**.
If conditions 2 and 3 both hold, report condition 2 as the primary label and
also report the aggregate bound. Report raw counts, times, output differences,
and interval. Neither failure nor inconclusion is trajectory-identity evidence
or rejection of the speculative architecture, but both are terminal for this
baton.

## Failure attribution and complete family closure

Before rejecting an arm, localize the loss with paired, faithful controls:

| Question | Required comparison |
| --- | --- |
| Is semantic quality insufficient? | converged F16 full-vocabulary drafter versus target distribution and exact target-match rollout |
| Is the shortlist responsible? | same frozen drafter with full-V, 128K, 64K, 32K, 16K, 8K, 4K, 2K, and 1K scoring across registered d-prime values |
| Is quantization responsible? | same checkpoint and route in F16 versus each packed precision |
| Is context missing or too costly? | root/current-feature Arm R versus full-context Arm R/D/P/L/T, including prompt and terminal maintenance |
| Is one-pass block diffusion responsible? | faithful DFlash versus DFlare, B16 versus trained/naive B24, and linear versus DDTree on the same frozen backbone |
| Is sequential dependence missing? | Arm P without/with Markov/GRU correction versus Arm R and parent-conditioned Arm T |
| Is one universal family hiding workload structure? | each standalone R/D/P/L/T/S member versus the frozen co-resident heterogeneous bundles, charging total residency |
| Is common randomness useful? | deterministic versus proposal-only common-draw using the same target sampler |
| Is routing responsible? | always-propose, calibrated decline, retrieval-first, and anchored policies |
| Is wasted deep drafting responsible? | eager-H versus every registered conditional tile schedule versus causal dynamic horizon at H=3/4/7/8/15/16/23 |
| Is control overhead responsible? | fused eager, host-gated, and capability-supported GPU-gated live topologies |
| Is target geometry responsible? | identical perfect/scripted proposals through every registered tile partition, width 3/4/5/7/8/9/15/16/17/23/24 chain, and source-budget/root-convention/k/M/K/W forest through 1,025 verifier rows |
| Is Metal implementation responsible? | independent CPU reference, trained Metal path, and production-shaped traffic/timing |

A single failed row closes only that row.

### Prefix-closed tree family

A chain is the primary M4 geometry because retained width two is fast and Apple
Silicon evidence warns that beam width can cost more than it saves. It is not a
universal rejection of trees. ReDrafter's exact MLX repeated-root beam batch is
the one declared non-tree topology in this closure and is never mislabeled
prefix-closed compression.

On training/architecture-development data, compute exact target mass, joint
path coverage, and live cost ceilings for every registered typed source
budget/root convention: bridging non-root 4/8/16/32/64/128/192/256,
fixed DDTree non-root B=16/32/64/128/256/512/1,024, adaptive CaDDTree
n=0..1,024, TreeFlash paper root-excluded B=16/32/64/128/256/512 and code
root-inclusive tree_size=16/32/64/128/256/512, PCTree B7/B16 with
root-inclusive N=8/16/24/32/48/64/96/128, DominoTree non-root B=16/32/64,
legal one-factor DARTree B/W cells, JetSpec root-inclusive
B=16/32/64/128/255/256, FullPath-T B=16/64/256 after its 1,025-row scaffold,
DBlast tries through 121 rows, EAGLE-3's 60-row tree/711-row builder, and every
ReDrafter MLX repeated-root/article/PyTorch shape through 225 raw rows. Include
every declared legal k/M/K/W and candidate-domain shape.
Before architecture-selection validation is opened, each
shape is either implemented as a real prefix-closed verifier/proposer pair or
has its own implementation-independent unavoidable-work certificate below
1.30x under the registry rule. An empirical interval from a particular slow
tree implementation closes only that digest. Implemented trees require:

- explicit parent/depth/token arrays;
- committed K/V plus speculative node storage;
- path-causal sliding/global attention and absolute RoPE;
- path-specific repetition histories;
- full target-vocabulary selection for every required node;
- accepted-path scatter and rejected-node isolation;
- per-token callback/stop rollback.

Do not claim a tree result from chain replay or independent row kernels.
[Sequoia](https://arxiv.org/abs/2402.12374) supplies the hardware-aware tree
baseline. Cross independent-marginal construction with
[PCTree](https://arxiv.org/abs/2608.02123)'s parent-conditioned conversion,
[DDTree](https://arxiv.org/abs/2604.12989)'s block-diffusion tree, the
cost-aware [CaDDTree](https://arxiv.org/abs/2606.01813) policy,
[DominoTree](https://arxiv.org/abs/2607.08642)'s per-path corrected best-first
tree, [DARTree](https://arxiv.org/abs/2608.13524)'s depth-batched corrected
supertree, TreeFlash's immediate-predecessor correction, JetSpec's executable
shared causal latent-depth construction and separately blocked paper
description, plus the FullPath-T concrete-path extension. A published tree policy transfers
only after its proposal scores, parent states, pruning decisions, exact target
nodes, and complete live costs are instantiated here.

### Target-integrated family closure

Arm S is registered before selection, because an external draft model's loss
cannot reject speculation generated inside the target pass. Evaluate the
complete predeclared lossless Gemma-transfer members only in the finite legal
pairings: `(Ns=2/4/8,gamma=4,k=3)`,
`(Ns=4,gamma=3,k=3)`, `(Ns=4,gamma=4,k=1/2)`, and
`(Ns=4,gamma=7/15/23,k=1)`. The `(4,4,3)` paper-primary geometry transfer
binds the Section-3 `f_eta=Identity` and early-exit rank 8 but no 32-node cap.
Learned-initializer, threshold-only, and lossless-32 cross-mode cases stay
explicitly labeled Gemma ablations. The reported Shared Table-3 32-passed-node
fixture is evaluated separately as a non-product source-resolution control and
cannot admit or reject the lossless product family. Use Gemma's real top layers,
stream adapters, stream dependencies, prior-tree verification, next-tree
generation, and early-exit pruning in one target invocation. Use the same exact
controller and live admission protocol. A suffix drafter, chain replay, or
Qwen/PyTorch reproduction is not evidence for this engine. Shared mode cannot
enter the exact-output product comparison because it modifies the target main
stream. Run the Ns4/gamma4/k3 paper-described virtual-distribution
interpretations as separately labeled controls, not as unqualified literature
fidelity, and do not use their different seed trajectory to admit or
reject the exact-target-token-match product family.

## Stop condition

Stop this baton only when one of these is true:

1. a complete frozen learned candidate passes correctness, sealed live
   internal Stage-4 admission, the canonical cooled retention gate, and the
   separately pinned llama.cpp claim protocol; or
2. a candidate is internally validated against Stage 4 and passes canonical
   retention, but the separately pinned llama.cpp result is either a fixed-N
   claim failure or fixed-N inconclusive. Record its exact external label and
   that the beats-llama.cpp claim is not established; or
3. a frozen candidate reaches its sealed decision and a correctness, resource,
   thermal, workload/context, TTFT/prompt, or other nonperformance admission
   gate fails. Record the exact failed gate and that only this implementation/
   artifact is rejected absent complete-family evidence; or
4. internal Stage-4 admission passes but the canonical cooled retention pair
   fails. Record **internally validated against Stage 4; canonical retention
   failed**; do not merge or tune against that exposed pair; or
5. the complete finite R/D/P/L/T/S and heterogeneous-bundle registry, including
   every registered chain, tree, and target-integrated member, has direct
   simultaneous candidate-by-regime bounds after independent per-entry policy
   fitting and an untouched closure set, or valid optimistic certificates
   uniform over each entry's full preregistered calibratable-policy space,
   for which `max_j min_h U_jh<1.30`. If that maximum worst-regime upper bound
   is also below 1.25, record the stronger below-useful-floor conclusion; or
6. the one frozen candidate reaches the preregistered fixed final N without the
   `speed-qualified` label. Record its exact ordered speed label and orthogonal
   gate status, that it is not internally admitted, and that neither its broader
   family nor learned speculation is rejected without item 5 evidence.

Items 2–4 and 6 consume that candidate's sealed set. Any continuation needs a
new preregistered candidate/study and untouched sealed data; the old admission
or canonical observations cannot be recycled for tuning.

Do not stop because:

- a target-hidden calibration oracle fails;
- k = 2,048 or 4,096 fails;
- a model exceeds 2 ms in isolation;
- a 512-window overfit threshold misses;
- one rank, depth, width, seed, quantization, or feature set fails;
- a replay, microbenchmark, synthetic kernel, or teacher-forced metric misses;
- the old monolithic width-four kernel remains slow;
- implementation requires refactoring the current controller, sampler, plan,
  Metal command boundaries, or cache ownership.

The retained Stage 4 path remains the production default until a candidate
passes. If the full experiment fails, remove its production complexity cleanly
and retain the authenticated evidence needed to avoid repeating the same
architecture.

## Invariants

- Production inference remains engine-owned, offline, Metal-first, and
  model-resident.
- The target verifier and target token selection remain exhaustive over the
  full vocabulary.
- Target sampling transforms, order, RNG, and authoritative draw count do not
  change.
- Exact seeded target-selected tokens are the correctness oracle.
- Rejected or undelivered drafts never alter retained target K/V, draft K/V,
  sampler, retrieval history, learned state, output, or future distributions.
- Every authoritative selected token is eventually represented in canonical
  target K/V; a preview blocked by a callback boundary is never selected.
- Target feature provenance, cache generation, logical position, and lifetime
  are explicit and validated.
- No full target or draft vocabulary logit row crosses to the host in
  production.
- All learned weights, cache, scratch, and compact output are persistent after
  warm-up.
- Ordinary one-row decode and retained Stage 4 remain directly measurable from
  the same binary.
- Development frameworks may train artifacts but never execute the production
  inference path.
- Every performance conclusion is scoped to the exact model, artifact,
  topology, workload, machine, thermal regime, and code identity measured.

## Research basis

The architecture and experiment above synthesize:

- [Gemma 2](https://arxiv.org/abs/2408.00118) for the target's alternating
  local/global attention, 9B geometry, and 256K vocabulary;
- [Leviathan et al.](https://proceedings.mlr.press/v202/leviathan23a.html) and
  [Chen et al.](https://arxiv.org/abs/2302.01318) for exact speculative
  decoding and target-verification economics;
- [DISD](https://arxiv.org/abs/2408.07978) for communication-free shared-random
  coupling. This plan does not implement its token-indexed construction or
  inherit its collision guarantee; it tests a legacy-sampler-preserving scalar
  common-random-number control;
- [EAGLE-3](https://arxiv.org/abs/2503.01840) for multi-level feature fusion,
  direct token prediction, and training on draft-generated states;
- [DFlash](https://arxiv.org/abs/2602.06036) for full target-context feature
  K/V injection and efficient parallel block drafting;
- [DFlare](https://arxiv.org/abs/2606.02091) for its seven-layer/B16 control,
  nine-tap draft-layer-specific fusion, heterogeneous target-versus-draft K/V,
  progressive position weighting, and 2.4M-example scaling; and
  [DBloom](https://arxiv.org/abs/2608.30427) for ceiling-bin diagnostics and
  continuation/expansion training from B16 to B24;
- [AngelSpec/DFly](https://arxiv.org/abs/2607.25852) for hybrid global/layer-
  specific conditioning and heterogeneous speculative architectures;
- [DSpark](https://arxiv.org/abs/2607.05147) for semi-autoregressive semantic
  drafting, lightweight sequential correction, and prefix-survival scheduling;
- [Domino](https://arxiv.org/abs/2605.29707) for target-token-prefix training,
  base-backbone anchoring, and the five-layer/GRU/rank-256 controls;
- [DBlast](https://arxiv.org/abs/2608.05448) for the stochastic weakness of
  independent parallel positions and dependent latent proposal modes;
- [D-PACE](https://arxiv.org/abs/2605.18810) for detached,
  asymmetrically-smoothed expected-prefix-aware training weights and
  [EDSD](https://aclanthology.org/2026.acl-long.2145/) for Mask-p/Step-n
  entropy scheduling plus EDSelect/EDFuse architectural controls;
- [ReDrafter](https://arxiv.org/abs/2403.09919) for recurrent
  previous-token-conditioned drafting and measured Apple-Silicon beam-width
  tradeoffs;
- [SpecVocab](https://arxiv.org/abs/2602.13836) and
  [DynaSpec](https://arxiv.org/abs/2510.13847) for adaptive draft-only
  vocabulary reduction;
- [Sequoia](https://arxiv.org/abs/2402.12374) for hardware-aware speculative
  tree geometry; [PCTree](https://arxiv.org/abs/2608.02123) for
  parent-conditioned conversion of parallel/Markov proposals;
  [DDTree](https://arxiv.org/abs/2604.12989) and
  [CaDDTree](https://arxiv.org/abs/2606.01813) for block-diffusion and
  cost-aware tree construction;
  [DominoTree](https://arxiv.org/abs/2607.08642) for per-path top-M corrected
  best-first construction; [DARTree](https://arxiv.org/abs/2608.13524) for
  depth-wise batched correction and deferred pruning; and
  [TreeFlash](https://arxiv.org/abs/2606.03819) for immediate-predecessor
  correction and [JetSpec](https://arxiv.org/abs/2606.18394) for executable
  one-pass shared causal latent-depth construction, with their paper/code
  discrepancies kept explicit; FullPath-T is the separately named concrete-
  path extension;
- [Apple Speculative Streaming](https://aclanthology.org/2025.emnlp-main.986/)
  and its [official implementation](https://github.com/apple/ml-speculative-streaming)
  for the coequal target-integrated Arm S;
- [SPEED-Bench](https://arxiv.org/abs/2604.09557) for workload-sensitive
  speculative-decoding evaluation;
- [REST](https://aclanthology.org/2024.naacl-long.88/) and
  [SAM Decoding](https://aclanthology.org/2025.acl-long.595/) for retaining the
  nearly free session-local retrieval route;
- [Apple Metal synchronization guidance](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)
  for explicit barriers when dependent GPU work uses concurrent dispatch;
- [Kalibera and Jones](https://kar.kent.ac.uk/33611/) and
  [NIST experimental-design guidance](https://www.itl.nist.gov/div898/handbook/pri/section3/pri332.htm)
  for blocked, randomized, variance-aware systems measurement.

Published throughput values are motivation only. Different target models,
samplers, CUDA/server workloads, and accelerators do not establish performance
on this M4 Air; only the protocol above can do that.
