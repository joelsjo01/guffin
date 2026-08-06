# Decode Acceleration Research Program

## Objective and Process

Find a path to substantially faster end-to-end token generation for the
Metal-first Gemma 2 9B Q4 runtime on this MacBook Air M4, while preserving the
engine-owned production stack and proving quality, correctness, and real-path
performance. Each phase is conducted only after explicit approval; its findings
are written into this document, then work stops until the next phase is approved.

## Phase 1: Scientific Frontier and Unconventional Methods

Conduct a deep search of academic papers, technical reports, and other credible
scientific sources for the fastest known or plausible inference methods,
including unconventional, recent, and cross-disciplinary ideas that could
produce step-change decode gains. Evaluate both established and original
combinations against Gemma 2 9B Q4 and Apple M4 constraints, clearly separating
measured evidence from hypotheses and accounting for quality, memory, and
implementation tradeoffs.

### Phase 1 findings (completed 2026-08-06)

#### Executive conclusion

The present decoder is already close enough to the base M4's single-token
weight-streaming limit that ordinary kernel improvement cannot deliver a
step-change by itself. The best research direction is a compound design that
(1) makes one target-model weight pass validate several tokens, (2) reduces the
bytes in that pass, and (3) attacks Gemma 2's unusually expensive 256K-vocabulary
output projection without changing the configured sampling distribution.

The strongest evidence-backed candidate is learned speculative decoding with a
small recurrent or parallel drafter and a tree/block target verifier. The most
promising unconventional hardware experiment is to store selected or all
matrix weights as ASTC HDR textures and use the M4 GPU's fixed-function texture
decoder as a neural-weight decompressor; the most promising original algorithmic
experiment is a certified top-k output search that avoids scanning most of the
1.835 GB tied output matrix while returning exactly the same top 40 logits.

#### Target-specific constraint and opportunity

- The model file is 6.843 GB and the measured generation rate is 13.9 tok/s.
  Treating file size as a first-order proxy for bytes streamed per token gives
  about 95.1 GB/s, or 79% of the MacBook Air M4's documented 120 GB/s memory
  bandwidth. The corresponding idealized one-full-file-read roofline is only
  17.5 tok/s, so even reaching peak bandwidth would improve generation by only
  about 26%; this is a planning bound, not a claim that every file byte is read
  on every step. [Apple documents 120 GB/s for this M4 MacBook Air](https://support.apple.com/en-us/122209),
  while prior work independently characterizes batch-one generation as
  memory-bandwidth-bound ([SqueezeLLM](https://arxiv.org/abs/2306.07629)).
- Gemma 2 9B has 42 layers, width 3584, interleaved local/global attention,
  grouped-query attention, and a 256,128-token tied vocabulary
  ([Gemma 2 technical report](https://arxiv.org/abs/2408.00118)). The local
  diagnostic shows that the 3584 x 256000 F16 output projection alone reads
  about 1.835 GB and takes 23.3 ms out of an approximately 71.9 ms/token decode;
  eliminating its cost entirely would cap the gain near 1.48x, or 20.6 tok/s.
- These bounds divide methods into two classes. A faster implementation of the
  same single-token pass has limited remaining headroom; a method that reads
  fewer weight bytes or obtains more than one accepted token per pass can cross
  the roofline and is capable of a large gain.

#### Evidence map

| Direction | Best relevant evidence | Fit for this target | Phase 1 verdict |
| --- | --- | --- | --- |
| Learned speculative decoding | Classical speculative sampling preserves the target distribution and reports 2-3x speedups ([Leviathan et al.](https://arxiv.org/abs/2211.17192), [Chen et al.](https://arxiv.org/abs/2302.01318)). Feature/recurrent drafters report 2.7-3.5x for EAGLE, 3.05-4.26x for EAGLE-2, up to 6.5x for EAGLE-3, and up to 2.3x specifically on Apple Silicon for ReDrafter ([EAGLE](https://arxiv.org/abs/2401.15077), [EAGLE-2](https://arxiv.org/abs/2406.16858), [EAGLE-3](https://arxiv.org/abs/2503.01840), [ReDrafter](https://arxiv.org/abs/2403.09919)). | Very high: verification can reuse each streamed target weight across several proposed positions, which directly attacks this machine's bottleneck. It requires a Gemma-specific drafter and an efficient causal multi-row/tree verifier; headline results on larger NVIDIA models are not transferable, but ReDrafter's Metal result is direct evidence. | **Highest-priority route.** Train a tiny target-conditioned drafter only if a target verification pass is first shown to be nearly flat for useful proposal widths. |
| Parallel, tree, and hardware-aware verification | Dynamic trees improve accepted length; Sequoia and OPT-Tree optimize tree shape/budget, with reported speedups up to 4.04x and 3.2x respectively ([Sequoia](https://arxiv.org/abs/2402.12374), [OPT-Tree](https://arxiv.org/abs/2406.17276)). DART replaces autoregressive draft rollout with parallel masked-position prediction and reports 2.03-3.44x, 30% above EAGLE-3 on average ([DART](https://arxiv.org/abs/2601.19278)). | High: the local Q4 exact-16 diagnostic being faster than one-row decode is unusually favorable evidence that verification width can amortize weight traffic. Tree masks, KV commit/rollback, and the large output head remain major costs. | **Pair with the first route.** Benchmark chain verification before tree complexity; adopt a tree only if branching raises accepted tokens more than its output/KV overhead. |
| Model-free retrieval proposals | REST retrieves drafts without another neural model and reports 1.62-2.36x on 7B/13B models ([REST](https://arxiv.org/abs/2311.08252)). Suffix automata update and retrieve in average O(1), and add 3.28-11.13% even on top of EAGLE-2 ([SAM Decoding](https://arxiv.org/abs/2411.10666)); suffix-tree methods also work from prior outputs ([SuffixDecoding](https://arxiv.org/abs/2411.04975)). | High as a cheap first proposer: CPU-side lookup can exploit repeated chat phrases, formatting, code, and the existing conversation without another weight set. Open-ended prose acceptance may be modest, so it should be a source in a hybrid proposal tree rather than the only strategy. | **Low-cost, high-information experiment.** Preserve exactness through normal target verification and measure accepted tokens by workload class. |
| Self-speculation from the target | Training-free layer skipping reports up to 1.99x and CLaSp reports 1.3-1.7x while preserving target outputs/distribution ([Draft & Verify](https://arxiv.org/abs/2309.08168), [CLaSp](https://arxiv.org/abs/2505.24196)). QuantSpec combines shared architecture with quantized weights/KV and reports up to about 2.5x for long context ([QuantSpec](https://arxiv.org/abs/2502.10424)); QSpec reports up to 1.64x using complementary precision ([QSpec](https://arxiv.org/abs/2410.11305)). | Medium: it avoids a second model, but every sequential draft token still streams a substantial fraction of Gemma's layers. It becomes compelling only if skipped/low-bit passes are much cheaper and acceptance stays high on Gemma 2. | **Fallback or hybrid drafter**, behind retrieval and a tiny recurrent/parallel drafter. |
| Training-free parallel decoding | Lookahead Decoding uses Jacobi trajectories, needs no draft model, preserves greedy output, and reports up to 1.8x on one GPU ([Lookahead Decoding](https://arxiv.org/abs/2402.02057)). | Medium-low for the sampled chat target: it trades extra parallel FLOPs for fewer serial steps and its exactness claim is naturally aligned with greedy decoding, not automatically with the configured sampled distribution. | **Interesting diagnostic, not the lead product route.** |
| More aggressive weight compression | AQLM and QuIP# make 2-bit-ish post-training compression substantially more accurate than scalar methods; AWQ, GPTQ, and SpQR establish strong 3-4 bit baselines ([AQLM](https://arxiv.org/abs/2401.06118), [QuIP#](https://arxiv.org/abs/2402.04396), [AWQ](https://arxiv.org/abs/2306.00978), [GPTQ](https://arxiv.org/abs/2210.17323), [SpQR](https://arxiv.org/abs/2306.03078)). | High theoretical fit because bytes/token set the roofline, but codebook lookup, transforms, outliers, and unpacking must be cheaper than the bandwidth saved on this GPU. Quality risk is material for a 9B model below 3 bits and published CUDA kernels do not establish Metal speed. | **Pursue hardware-shaped compression, not a format transplant.** ASTC is the standout M4-specific candidate below. |
| Static or contextual sparsity | SparseGPT reports 50% one-shot pruning with little perplexity loss, DejaVu predicts contextual sparsity and reports over 2x latency reduction, and TEAL reports up to 1.53x/1.8x at 40%/50% activation sparsity ([SparseGPT](https://arxiv.org/abs/2301.00774), [DejaVu](https://arxiv.org/abs/2310.17157), [TEAL](https://arxiv.org/abs/2408.14690)). | Medium-low: irregular sparse accesses can lose to dense streaming on Apple GPU, and approximate activation skipping changes numerics/quality. Structured blocks or large contiguous neuron groups are the only promising forms. | **Research reserve.** Consider only after measuring a representation whose skipped bytes are physically not fetched. |
| KV-cache compression/eviction | KIVI provides tuning-free 2-bit KV quantization; SnapKV reports 3.6x generation gain at 16K inputs; H2O and StreamingLLM reduce retained context ([KIVI](https://arxiv.org/abs/2402.02750), [SnapKV](https://arxiv.org/abs/2404.14469), [H2O](https://arxiv.org/abs/2306.14048), [StreamingLLM](https://arxiv.org/abs/2309.17453)). | Low for the current few-hundred-token benchmark: Gemma already uses GQA and alternating sliding-window attention, while model weights dominate traffic. Relevance rises sharply near the 8K context limit. | **Do not expect a headline short-context gain.** Keep as a long-context track. |
| Replace or retrain the architecture | Mamba reports 5x Transformer inference throughput, RecurrentGemma offers a fixed-size recurrent state at 9B, Mixture-of-Depths reports up to 50% faster sampling, and multi-token-prediction training reports up to 3x inference speed ([Mamba](https://arxiv.org/abs/2312.00752), [RecurrentGemma](https://arxiv.org/abs/2404.07839), [Mixture-of-Depths](https://arxiv.org/abs/2404.02258), [multi-token prediction](https://arxiv.org/abs/2404.19737)). | Low compatibility: these are different trained models or require substantial continued training, so they do not accelerate the current Gemma 2 checkpoint while preserving its behavior. They show that co-designing training and inference has the highest long-run ceiling. | **Out of scope for the current weight-compatible mission**, but important if model identity is later relaxed. |

#### Unconventional hypothesis 1: fixed-function ASTC weight decoding

Apple reports using 6 x 6 ASTC HDR blocks as a 3.56-bit-per-weight
post-training representation for its server model. Its GPU's fixed-function
ASTC unit decompresses weights on demand without consuming general-purpose
shader compute, the per-block minimum restoration is fused into matrix
operations, and Apple explicitly identifies autoregressive weight bandwidth as
the benefit ([Apple Intelligence Foundation Models 2025, sections 6.2-6.3](https://arxiv.org/html/2507.13575v3#S6.SS2)).
This is not merely analogous hardware: Metal exposes `astc_6x6_hdr`, M4 belongs
to Apple GPU family 9, and Apple's feature table marks HDR ASTC filtering as
supported from family 6 onward
([Metal pixel format](https://developer.apple.com/documentation/metal/mtlpixelformat/astc_6x6_hdr),
[Metal feature tables](https://developer.apple.com/metal/capabilities/)).

The original experiment is to reinterpret each large matrix as one or more
read-only 2D ASTC textures, fetch decompressed FP16 values through texture reads
inside a matvec/matmul kernel, add a per-block minimum, and accumulate normally.
The format's raw 128/36 ratio is 3.56 bits/weight; a separate FP16 minimum per
36 weights makes a conservative effective estimate about 4.0 bits/weight before
alignment or adapters. Applying 3.56-4.0 bits to roughly 9.24B parameters gives
about 4.1-4.6 GB, suggesting approximately 20.6-23.1 tok/s at today's inferred
bandwidth efficiency, before quality-recovery or texture-access overhead.

This is a hypothesis, not a forecast. Apple's reported quality includes its own
models and low-rank recovery adapters; Gemma's post-training error, ASTC encoder
availability/quality, texture-coordinate overhead, filtered-read semantics,
cache behavior, maximum texture dimensions, and whether arbitrary matrix access
sustains the advertised bandwidth all require isolated proof. A selective
variant is safer: start with the 1.835 GB F16 output/embedding matrix and Q6_K
matrices, where the byte reduction is largest, while leaving Q4_K tensors in
their proven format.

#### Unconventional hypothesis 2: certified top-40 output search

The configured sampler applies top-k 40 before top-p, so it needs the exact 40
largest adjusted logits, not necessarily a dense scan of all 256K vocabulary
rows. Build a compact low-rank or clustered index of the tied output embeddings;
use it to produce an upper bound for every token or cluster, evaluate full FP16
dot products only for candidates whose bounds can still beat the current 40th
score, and stop when every unevaluated bound is lower. This remains exactly
equivalent because the bound certifies the top 40; Gemma's final tanh soft-cap
and positive-temperature scaling are monotone, while repeated-token penalties
can be handled by exactly evaluating the finite set of affected token IDs.

One concrete certificate decomposes each embedding row into a rank-r projection
and residual. For hidden state `h` and token row `w_i`, compute the cheap
projected score and bound the omitted term by
`norm(h_residual) * norm(w_i_residual)` using Cauchy-Schwarz; only rows whose
upper bound exceeds the current threshold need the 3584-dimensional read. The
method fails safely—poor bounds fall back to all rows and preserve output—but a
successful index can attack up to the observed 23.3 ms/token output cost; prior
vocabulary-selection work supports the existence of the bottleneck but generally
trades exactness for speed ([LSH vocabulary selection](https://aclanthology.org/2022.wmt-1.79/),
[VQ-Logits](https://arxiv.org/abs/2505.10202)), making the certificate the novel
and essential part here.

#### Unconventional hypothesis 3: a multi-source speculative forest

Use a CPU suffix automaton over the prompt, conversation, and prior outputs to
offer virtually free candidate continuations; merge those paths with candidates
from a tiny target-conditioned recurrent or DART-style parallel drafter; then
select a hardware-budgeted verification tree using expected accepted tokens per
measured target-pass millisecond. This combines complementary error modes rather
than making one drafter larger: retrieval should dominate copied phrases and
structured text, while the learned drafter covers novel prose.

The verifier applies the engine's actual temperature, top-k, top-p, minimum-p,
repeat-penalty, and PRNG semantics and uses rejection/correction sampling, so
candidate generation is approximate but committed output preserves the target
distribution. The controller should maximize `expected committed tokens /
cycle latency`, not acceptance rate alone, and must learn separate budgets for
chat, code, and repetitive structured output; Sequoia's hardware-aware tree
optimization and SAM Decoding's gains when combined with EAGLE-2 support this
composition, but no source establishes it on Gemma 2 or base M4.

#### Quantitative performance envelope

These are bounds and test targets, not promised outcomes:

1. A conventional single-token, unchanged-format decoder is bounded near 17.5
   tok/s by the coarse full-file/peak-bandwidth model. The realistic ceiling is
   lower because attention, synchronization, arithmetic, and non-weight traffic
   also consume time.
2. Removing the measured output projection cost entirely gives about 20.6
   tok/s; this is the maximum isolated benefit for the certified top-40 idea
   before accounting for its index work.
3. ASTC at an effective 3.56-4.0 bits/weight gives a coarse 20-23 tok/s range at
   current bandwidth utilization. Selective ASTC may deliver less but has lower
   quality and implementation risk.
4. If one target verification pass costs roughly one current pass and commits
   two tokens on average, the gross ceiling is near 28 tok/s before draft and
   tree overhead. Three committed tokens per pass raises the gross ceiling near
   42 tok/s; combining lower-byte ASTC verification with two to three committed
   tokens makes roughly 30-50 tok/s a defensible research envelope, not a
   benchmark prediction.

The decisive quantity is therefore not a paper's reported speedup but the local
ratio `accepted tokens / (draft time + verification time + rollback/sampling
time)`, with identical token counts, output, sampling rules, and thermal protocol.

#### Ranked conclusions for later planning

1. **Prove multi-position target verification economics.** Everything above 20
   tok/s depends on amortizing target weights, and the current exact-16 Q4 result
   is the strongest local clue that this may work.
2. **Prototype ASTC HDR as a standalone matrix experiment.** It is the rare
   unconventional idea both published by Apple for neural weights and directly
   supported by M4 fixed-function hardware.
3. **Test model-free suffix proposals before training anything.** They cheaply
   reveal the verifier, rollback, and acceptance economics and remain useful in
   a later hybrid.
4. **If verification is favorable, train a very small recurrent or parallel
   Gemma-specific drafter.** Optimize end-to-end latency and acceptance jointly;
   do not select it by perplexity alone, consistent with the finding that draft
   language-model capability poorly predicts speculative speed
   ([Decoding Speculative Decoding](https://arxiv.org/abs/2402.01528)).
5. **Investigate certified output shortlisting independently and inside the
   verifier.** It has a hard 1.48x isolated ceiling today but can prevent the
   giant vocabulary head from erasing speculative gains at wider verification.
6. **Defer KV eviction, irregular sparsity, and architecture replacement.** They
   either target a different bottleneck, carry high quality/hardware risk, or
   change the model being accelerated.

#### Phase boundary

Phase 1 is complete. No local pipeline audit, implementation, or benchmark was
performed beyond read-only constraint calculations; Phase 2 requires explicit
approval.

## Phase 2: Local Whole-System Opportunity Audit

Investigate the local engine end to end for large gains in non-obvious places,
including tensor representation and lifetime, memory residency and traffic,
KV-cache organization, synchronization, command submission, allocation,
scheduling, CPU/GPU overlap, and avoidable work. Use measurement and source-level
inspection to expose hidden bottlenecks, quantify each opportunity's maximum
whole-path benefit, and record experiments that could validate the strongest
hypotheses.

### Phase 2 findings (completed 2026-08-06)

#### Executive conclusion

The local audit changes the recommended order of attack. The best immediate
enabler for a step-change is a native two- or four-row causal verifier: on this
M4, the expensive output matrix is almost flat from one to two rows and the
quantized projections are actually faster at two rows than at one. The existing
prefill route cannot serve as that verifier because it only produces logits for
its final row, while the decode route explicitly rejects more than one row.

The largest surprising whole-system defect is not a matmul at all. Startup
warm-up synchronously zeroes the full 2.114 GB KV allocation twice, materializing
pages that short chats never use and pushing a 16 GB unified-memory machine into
needless pressure. Removing those full clears is a high-confidence memory and
startup improvement, although this phase did not establish a retained decode
throughput win. The current per-token CPU sampler, compact readback, allocation,
and command submission together are too small to explain the missing speed.

#### Audit method and evidentiary limits

- Traced the production route from the persistent chat session through schedule
  creation, the resident Metal command context, all plan-owned buffers, compact
  GPU candidate selection, CPU sampling, callback streaming, and cache reuse.
- Added temporary diagnostic-only counters to the existing production path,
  built release binaries, ran a fixed-seed eight-token chat, and then removed all
  instrumentation. No candidate implementation remains in the source tree.
- Temporarily extended the existing kernel diagnostic to measure identical F16,
  Q4_K, and Q6_K matrices at 1, 2, 4, 8, and 16 rows, then removed the extension.
  These are explanatory microbenchmarks, not retained end-to-end results.
- Tested two isolated hypotheses: suppressing warm-up's full KV clears, and using
  a serial compute encoder with no explicit buffer barriers. Runs were intentionally
  diagnostic rather than five-run retention comparisons; thermal and VM state
  make absolute cross-run latency non-comparable unless explicitly paired below.

#### Measured production-path anatomy

The target file is 6,843,425,696 bytes, of which 6,837,368,832 bytes are supported
tensor payloads uploaded as resident Metal weights. The plan also reports
2,113,929,216 bytes of KV and 181,146,892 bytes of scratch, for 9,132,444,940
bytes (8.50 GiB) of logical resident plan storage before the retained file mapping,
tokenizer, process, driver, and operating system. The machine has 16 GB unified
memory, so residency policy is part of decode performance rather than merely a
capacity concern.

One sampled token has this static command shape:

| Item | Per token without next-token selection | Additional sampled top-40 selection |
| --- | ---: | ---: |
| Command buffers / synchronous waits | 1 / 1 | 0 |
| GPU dispatches | 633 | 11 |
| Explicit buffer barriers | 465 | 10 |
| Q4_K / Q6_K / F16 projections | 252 / 42 / 1 | 0 |
| Transient engine buffer allocations | 0 | 0 |

The 633 dispatches are exactly one embedding, 42 layers times 15 dispatches, and
final norm plus output projection. The 465 barriers are one after embedding, 42
layers times 11 dependency boundaries, and two final boundaries. Candidate
selection divides 256K logits into 500 tiles and performs nine merge levels plus
finalization, explaining its 11 dispatches and 10 barriers.

In the paired baseline half of the synchronization diagnostic, eight decodes
took 572.27 ms of reported GPU time and 576.70 ms wall time: 71.53 and 72.09
ms/token respectively. CPU sampling itself took only 26.8 microseconds total,
or 3.4 microseconds/token; the residual decode-call wall-minus-GPU envelope was
about 0.55 ms/token. This bounds schedule-vector construction, command-buffer
creation, Objective-C encoding, dynamic buffer validation, waiting, and compact
readback near 0.8% for this short run. Sampling is separately negligible;
streaming callbacks lie outside this interval. Removing a CPU wait without also
batching target work therefore cannot create a large speedup.

#### Decisive finding: two- and four-row target verification are unusually cheap

Within one release diagnostic process, the measured medians were:

| Matrix | 1 row | 2 rows | 4 rows | 8 rows | 16 rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| F16 3584 x 256000 output | 18.4 ms | 18.5 ms | 19.9 ms | 32.0 ms | 60.0 ms |
| Q4_K 3584 x 14336 | 0.8 ms | 0.5 ms | 0.9 ms | 2.1 ms | 0.8 ms |
| Q6_K 14336 x 3584 | 1.4 ms | 0.7 ms | 1.4 ms | 2.9 ms | 1.1 ms |

The exact values are thermal-state diagnostics, but their within-run ratios are
unambiguous. Two output rows cost only 0.5% more than one, four cost only 8%, and
the two-row Q4_K/Q6_K paths cost roughly 63%/50% of their one-row paths. Width
eight is a bad local dispatch point for both K-quant formats, while the special
exact-16 path recovers high reuse; verifier width must therefore be selected by
the measured hardware cost curve, not by accepted-length theory alone.

This is stronger evidence than Phase 1's isolated exact-16 result. A chain
verifier with two or four proposed positions can amortize the full target weight
stream, including the vocabulary head, before any tree or trained drafter is
added. A crude decomposition using the current roughly 72 ms pass and 18.4 ms
output cost suggests a two-row target pass could plausibly land around 50-55 ms
if the rest of the graph follows the observed K-quant ratios; that is a hypothesis
to measure, not a projected result. The required experiment must produce and
validate every row's logits, apply causal attention and tentative KV writes, and
commit or roll back exactly. Current prefill computes only its last row's final
norm/logits, so timing it would falsely omit the core verifier cost.

#### Highest-confidence strange-place win: stop materializing unused KV

Gemma 2's alternating cache layout is logically sensible: 21 layers reserve
4,096 sliding rows and 21 reserve 8,192 global rows. At eight KV heads, width
256, F16 K and V, this exactly explains the 2,113,929,216-byte allocation. At
the user's roughly 238-token follow-up context, however, attention reads only
about 81.9 MB of KV per generated token across all 42 layers, roughly 1.2% of
resident model-weight bytes. KV arithmetic or quantization therefore cannot be
a large short-context throughput win by itself.

The memory lifecycle is different. Warm-up calls a host `memset` over every K/V
buffer before its 16-token prefill and repeats it afterward. The buffer layer's
own comment notes that this operation synchronously prefaults the complete
allocation. It writes 4.228 GB in total and leaves 2.114 GB of zero cache pages
physically materialized even though attention bounds guarantee that unwritten
rows are never read.

A temporary no-clear diagnostic preserved the same fixed-seed generated text.
At the prompt boundary, the system-wide anonymous-page count was about 7.74 GB
instead of 10.67 GB in the preceding baseline observation, a noisy 2.93 GB
difference whose direction and scale match the removed KV materialization. The
candidate's decode was slower because the machine was already in a different
thermal/compressor state, so no throughput claim is attached. The production
experiment should replace global clears with one of these, in order:

1. Do not clear at all and rely on the already enforced populated-row bounds;
   prove fresh-session, reset, regeneration, context shift, and wraparound parity.
2. If diagnostics require restoration, clear only the 17 warm-up rows per layer.
3. Separately consider capacity-on-demand or paged growth for the global cache;
   this is more complex and unnecessary if untouched shared-buffer pages remain
   physically lazy after the first fix.

The plausible direct decode benefit is unknown, but removing two gigabytes of
avoidable pressure protects hot weight pages from compression and eviction and
is a prerequisite for adding a drafter or alternate weight representation on a
16 GB machine.

#### Weight residency and file-mapping opportunity

Every one of the 464 tensors is copied from the private read-only GGUF `mmap`
into a separate `MTLStorageModeShared` buffer using `newBufferWithBytes`. The
file-backed mapping remains alive for the session because `EngineModel` retains
borrowed tensor views and chat formatting still reads its owned metadata. This
is not automatically 6.84 GB of permanent physical duplication—clean file pages
are reclaimable—but upload transiently reads the full mapping and creates 6.84
GB of anonymous Metal storage while the machine has only 16 GB.

Three bounded experiments are warranted:

- After upload, issue page-aligned `madvise` for tensor-payload pages while
  retaining only the small metadata region; measure startup, prompt latency,
  compressor activity, and steady decode. This is the smallest intervention.
- Test immutable weight buffers with shared/write-combined, private, and explicit
  hazard-tracking choices in an isolated large-matrix probe. Do not assume that
  a GPU-only storage mode is faster on unified memory.
- Prototype one offset-aware Metal buffer that wraps or stages the contiguous
  GGUF payload, rather than 464 zero-offset buffers. A file-backed no-copy buffer
  could eliminate upload duplication, but page alignment, buffer-length limits,
  file-backed-memory support, cache behavior, and resource lifetime must all be
  proved locally before changing ownership.

The last option is architecturally attractive because a buffer-plus-byte-offset
view also enables heaps, argument tables, and faster binding, but it is not yet
a decode-speed claim. Its primary demonstrated value is memory headroom.

#### Synchronization, dispatch, and static-graph replay

The current implementation already avoids the common catastrophic mistakes:
weights, KV, scratch, and candidate buffers persist; a whole token uses one
command buffer and one compute encoder; there are no engine-level transient
buffers; Q/K/V and gate/up are placed between the same dependency barriers so
the concurrent encoder can overlap them; and only compact candidates cross to
the CPU.

To test whether explicit barriers themselves were the hidden cost, a temporary
candidate used a serial encoder and made all explicit barriers no-ops. It
preserved the exact fixed-seed text but took 582.10 ms GPU / 586.16 ms wall for
eight decodes, versus the immediately following restored baseline's 572.27 /
576.70 ms. The serial candidate lost about 1.6-1.7%, so this direction is
rejected; the concurrent groups are worth more than the barriers cost.

There is still a longer-term replay opportunity. Tensor bindings and 42-layer
topology are static, while token ID, cache position, attention length, sampling
history, and RNG state are small dynamic inputs. An argument-buffer or indirect
command design could remove hundreds of host API calls, or a GPU-resident token
loop could remove the per-token wait. The measured 0.55 ms/token host envelope
caps the isolated benefit today, so neither deserves priority unless it is the
mechanism that enables multi-token verification or coarse-grained streaming.

#### Other non-obvious findings and bounds

- **Final-token work:** even when a sampled stop or token limit means no next
  candidate is required, the runtime still executes final RMSNorm and the full
  output projection after updating KV. Skipping the head can save the observed
  18-23 ms once per completed turn. It does not materially change long-response
  tokens/s, and the displayed generation interval already ends at final
  selection, but it improves tail latency and avoids needless energy.
- **Scratch liveness:** 181.1 MB is persistent, dominated by 128-row prefill
  scores/probabilities. Decode activations are tiny beside weights. In-place
  prefill softmax, cross-phase aliasing, and decode liveness reuse can recover
  tens of megabytes, not a step-change in generation speed; pursue this as
  memory headroom for a verifier/drafter.
- **Per-token host allocation:** schedule creation allocates and fills a 42-entry
  vector every token, and command encoding repeatedly performs descriptor checks,
  dynamic casts, labels, and `setBytes`. These are avoidable but live inside the
  measured sub-millisecond host envelope.
- **Streaming text is quadratic:** the chat callback decodes the entire visible
  token prefix after every token, making display work O(output_tokens squared).
  It was not isolated by the decode-call timing and is insignificant at eight
  tokens, but an incremental detokenizer is appropriate before very long
  generation or a much faster speculative target exposes it.
- **KV layout is long-context work:** the row-major cache makes each head walk
  with an 8 KB row stride while all eight KV-head workgroups collectively consume
  the row. At 238 tokens the hard traffic ceiling is too small for a headline
  gain; at full context, alternating global/sliding K/V traffic reaches the full
  2.114 GB per token and layout, paging, or quantization becomes important.
- **CPU sampling is solved for the present workload:** 3.4 microseconds/token in
  the diagnostic makes changes to `std::sort`, RNG, or top-p arithmetic irrelevant
  until target inference is tens of times faster.

#### Ranked conclusions for later planning

1. **Build a diagnostic-only width-2 causal verifier first.** It must emit both
   rows' exact logits and expose target-pass time, tentative KV cost, and rollback;
   then test width 4 and hardware-favored width 16. This is the gate for every
   speculative method and currently has the highest plausible throughput return.
2. **Stop full-cache warm-up clearing.** Prove untouched-row safety and measure
   cold startup, VM pressure, TTFT, and thermally controlled decode. This is the
   clearest small implementation with a large resource win.
3. **Use the recovered memory headroom for proposals, not a larger cache.** Start
   with suffix/retrieval proposals; train a drafter only after verifier economics
   are measured.
4. **Prototype payload-page release and offset-aware/no-copy weight storage.** Keep
   this behind isolated correctness and bandwidth probes because a memory win
   does not guarantee faster GPU reads.
5. **Skip the output head on terminal decode and alias prefill scratch.** These are
   bounded latency/memory cleanups, useful but not the central throughput route.
6. **Do not pursue serial dispatch, CPU sampler tuning, or short-context KV
   quantization as headline work.** Local measurements put their ceilings below
   the mission's required step-change.

#### Phase boundary

Phase 2 is complete. All temporary source instrumentation and experimental
behavior were removed; only this research document changed. Phase 3 requires
explicit approval and should use these results to narrow its kernel/pipeline
deep dive around a width-2/4 verifier, output projection, and the remaining
roughly 54 ms of non-output target work.

## Phase 3: Decode Pipeline and Kernel Deep Dive

Dissect the conventional token-by-token decode path operation by operation,
covering graph execution, quantized matvec/matmul kernels, attention and KV-cache
access, normalization, activations, logits, sampling, fusion, dispatch, and
hardware utilization. Compare the implementation with current primary sources
and strong reference techniques, then rank concrete optimizations by expected
end-to-end gain, correctness risk, engineering cost, and validation method.

### Phase 3 findings (completed 2026-08-06)

#### Executive conclusion

The conventional one-token decoder is substantially closer to its hardware
limit than the isolated kernel table suggests. When the existing Q4_K and Q6_K
decode kernels are repeated in a sustained, dependency-separated command stream,
both deliver about 98 GB/s; the generic 56-block Q4_K path also delivers about
99 GB/s. The F16 output projection reaches roughly 97-103 GB/s in favorable
runs. These independent paths converge on the same number, strongly indicating
that the long production command is a unified-memory bandwidth stream rather
than a collection of individually slow arithmetic kernels.

The model contains 6.837 GB of weights used by a token pass. At 98 GB/s the
weight-only floor is 69.8 ms, strikingly close to the measured 71.5 ms GPU and
72.1 ms wall time per token. Even a perfect 120 GB/s pass has a 57.0 ms weight
floor, so ordinary tuning of the unchanged single-token graph has an absolute
ceiling near 1.22x before attention and other work are included. The practical
conclusion is not that kernel work is finished, but that a large gain must remove
weight bytes, reuse one weight pass across tokens, or change the output-head
algorithm; launch, normalization, RoPE, and selection cleanup can only supply
small secondary gains.

This phase also corrects one Phase 2 interpretation. The apparent advantage of
two-row Q4/Q6 over one row came from comparing isolated commands that were too
short to reach the sustained GPU state. In repeated serial streams, width two
costs about 1.23-1.24x for Q4_K and 1.32-1.33x for Q6_K; width four costs about
2.32-2.44x and 2.59-2.60x respectively. A width-two verifier is still highly
promising because two target positions for roughly 1.3 passes is good economics,
and the F16 head was nearly flat from one to two rows, but the earlier speculative
50-55 ms full-pass estimate should not be used. Only a complete sustained causal
verifier can establish its real cost.

#### Method and limits

- Traced every production decode operation and its buffer dependencies, then
  reconciled the GGUF tensor directory with the plan's 294 projections and the
  633-dispatch token schedule.
- Temporarily extended the existing release diagnostic with sustained serial
  matrix streams, row-width reversals, aggregate elementwise/attention tests,
  pipeline-resource queries, and full tensor-category inspection. All temporary
  source changes were removed after measurement.
- Repeated the one/two/four-row quantized comparison in both `1,2,4` and `2,1,4`
  order. Absolute frequency changed with thermal state, but the row-cost ratios
  reproduced, which is the evidence needed to reject the isolated-command
  interpretation.
- Aggregate operation diagnostics use the real kernels, resident buffers, and
  barriers but not meaningful model values. They attribute cost and scaling;
  they are not additive end-to-end benchmarks and do not satisfy the project's
  retention rule.
- Compared the local implementation with Apple's current Metal guidance, the
  FlashAttention/Flash-Decoding family of algorithms, quantized-GEMM research,
  and the exact pinned llama.cpp Metal source used for the stored reference.

#### Exact weight-traffic anatomy

The 464 tensors are not uniformly quantized. Query, key, attention output,
gate, and up matrices are Q4_K in every layer. Value and down matrices are each
Q4_K in 21 layers and Q6_K in 21 layers, producing the observed totals of 252
Q4_K and 42 Q6_K tensors. The tied embedding/output matrix is the sole F16 tensor.

| Weight work performed every token | Bytes | Share of all resident weights | Approximate stream time at 98 GB/s |
| --- | ---: | ---: | ---: |
| FFN gate + up projections | 2,427,715,584 | 35.51% | 24.77 ms |
| FFN down projections | 1,492,033,536 | 21.82% | 15.22 ms |
| F16 output projection | 1,835,008,000 | 26.84% | 18.72 ms |
| Q/K/V/O attention projections | 1,080,188,928 | 15.80% | 11.02 ms |
| All F32 norm weights | 2,422,784 | 0.04% | 0.02 ms |
| **Total** | **6,837,368,832** | **100%** | **69.77 ms** |

This table is a much stronger guide than counting dispatches. FFN projections
own 57.3% of weight traffic, the vocabulary head owns 26.8%, and all attention
projections together own only 15.8%. Saving half of a tiny operation cannot
compete with removing 500 MB from gate/up/down or the output matrix.

It also explains the headline result without an unexplained residual. At the
locally observed sustained rate, weights alone account for nearly the complete
GPU interval; the remaining few milliseconds contain attention, activation
traffic, barriers, and imperfect overlap, while caching and concurrent Q/K/V or
gate/up dispatches can move the simple sum in the other direction. The earlier
coarse file-size roofline is therefore validated by direct sustained kernels.

#### Quantized projection kernels: already strong, with bounded tuning space

The decode kernels consume native GGUF blocks directly and fuse dequantization
with F32 dot accumulation; they never materialize an intermediate dequantized
matrix. Q4_K assigns four eight-lane block workers per SIMD group and reuses an
activation fragment for two adjacent output columns. Q6_K walks two blocks per
lane pair and likewise produces two adjacent columns. All high-frequency Gemma
inner widths are recognized: 3,584 maps to 14 quant blocks, 4,096 to 16, and
14,336 to 56. Even the unspecialized Q4_K 56-block down path reached 99.3 GB/s
in a sustained stream, so adding a function constant for that shape is unlikely
to matter by itself.

The isolated diagnostic's roughly 28-36 GB/s is therefore misleading for a
decode decision. Sixteen to 42 serial dispatches keep the GPU active and raise
Q4_K/Q6_K to about 72-99 GB/s depending on thermal and ramp state; with 42
repetitions both settled near 98 GB/s. Future matrix experiments must include a
sustained mode or they will optimize GPU power-state latency instead of the
production kernel.

Research kernels such as Marlin retain low-bit speed at larger batches through
striped partitioning and latency-hiding, while FLUTE restructures weights offline
to reduce unpacking and lookup overhead
([Marlin](https://arxiv.org/abs/2408.11743),
[FLUTE](https://arxiv.org/abs/2407.10960)). Those CUDA results do not transfer
directly to M4, but they suggest two valid local probes: an offline Metal-native
column/interleave layout, and a small search over columns per SIMD group,
threadgroup size, vector load width, and accumulator count. Neither should be
retained unless Apple GPU counters show an ALU/control-flow limiter and the
sustained production pass improves; 98/120 GB/s bounds the best conceivable
unchanged-byte benefit near 22%.

Apple specifically advises profiling occupancy, register pressure, limiter
counters, and bandwidth rather than treating occupancy alone as success
([occupancy](https://developer.apple.com/documentation/xcode/finding-your-metal-apps-gpu-occupancy),
[bandwidth counters](https://developer.apple.com/documentation/xcode/measuring-the-gpus-use-of-memory-bandwidth)).
For Apple-family-9 GPUs it also warns that copying device data into threadgroup
memory may lose to direct cached access, and recommends 16-bit values where
their precision is acceptable
([Metal shader practices](https://developer.apple.com/videos/play/tech-talks/111373/)).
That guidance applies mainly to the multirow tiled kernel, which stages an 8 KiB
tile; the one-row Q4/Q6 paths already read device memory directly and use no
threadgroup storage. Metal 4 tensor primitives are worth an isolated multirow
comparison, but Apple's neural-accelerator benefit starts with Apple-family-10
hardware, not this M4
([inline Metal ML operations](https://developer.apple.com/documentation/metal/running-inline-ml-operations-in-a-shader-with-metal-4)).

The pinned llama.cpp source independently supports the chosen design family. It
contains fused native Q4_K/Q6_K Metal matvecs, multirow `r1_2` through `r1_5`
variants, simdgroup-matrix matmul, and fused RMSNorm/multiply/add kernels
([pinned Metal kernels](https://github.com/ggml-org/llama.cpp/blob/d48a56effbba99944a677938d4beb63a0065ecdf/ggml/src/ggml-metal/ggml-metal.metal)).
The local runtime is not missing a fundamentally different conventional kernel
class; its sustained rate and slightly faster stored decode result show that
copying a reference kernel wholesale would be a weak bet. The useful comparison
is individual work partitioning and row-width ideas under the local retention
protocol.

#### Width-two and width-four verification economics, revised

The sustained row experiment measured these ratios relative to the same
one-row kernel in the same release process:

| Kernel | Two-row cost / one-row cost | Four-row cost / one-row cost | Interpretation |
| --- | ---: | ---: | --- |
| Q4_K 3584 x 14336 | 1.23-1.24x | 2.32-2.44x | Two rows reuse weights well; four rows still beat four independent passes |
| Q6_K 14336 x 3584 | 1.32-1.33x | 2.59-2.60x | More unpack/accumulator work, but useful two-row amortization remains |
| F16 3584 x 256000 | not yet measured sustained | not yet measured sustained | Isolated cost was 1.005x at two rows and 1.08x at four, but needs a multirow output buffer and sustained confirmation |

For the quantized 5.000 GB portion, a byte-weighted two-row ratio is roughly
1.25x. A complete verifier will also scale attention, elementwise work, KV
writes, output projection, and selection. A reasonable measurement target is
therefore a width-two pass below about 1.35x current latency, not the earlier
50-55 ms absolute guess. If it commits both proposed tokens, 1.35x verification
cost alone corresponds to a gross 1.48x gain before drafting and rejection;
1.25x corresponds to 1.60x. Those are useful ceilings, not forecasts.

Correctness is as important as cost. The width-two/four small-batch K-quant
kernel dequantizes and accumulates in F32, but it does not necessarily use the
same reduction order as the one-row packed kernel. The exact-16 name describes
an exact tile shape, not numerical identity: that path stages weights and
activations as F16 before F32 matrix accumulation. A target verifier must define
and test its target distribution explicitly, compare every row's logits against
the canonical one-row path, and must not assume that an `exact-16` dispatch is a
bit-exact oracle.

#### Attention and KV: solved for short context, second bottleneck at long context

The decode attention shader already implements the core FlashAttention ideas:
it streams K/V once, maintains online maximum/denominator/value state in
registers, shares each KV head between its two query heads, and avoids a score
or probability tensor. Beyond 256 tokens it splits the sequence into 2, 4, 8,
16, or 32 partitions and performs a second exact rescaling reduction, which is
the sequence-parallel strategy described by Flash-Decoding
([online softmax](https://arxiv.org/abs/1805.02867),
[FlashAttention](https://arxiv.org/abs/2205.14135),
[Flash-Decoding explanation](https://princeton-nlp.github.io/flash-decoding/)).

Aggregate local medians for all 42 layers were about 2.55 ms at 256 tokens,
18.7 ms when every layer was capped at 4,096, and 23.2 ms for the model's real
full-context mixture of 21 sliding 4,096-token layers and 21 global 8,192-token
layers. At the current short chat context, attention is only around 3.5% of a
72 ms token and cannot produce a headline win. At full context it becomes tens
of milliseconds and deserves its own optimization track.

The best conventional attention experiment is a block-online kernel: calculate
a 16- or 32-token score block, reduce its maximum once, exponentiate the block,
and rescale the accumulated value state once per block rather than once per
token. This preserves the online-softmax algebra while reducing non-matmul
operations, the same broad objective as FlashAttention-2's improved work
partitioning ([FlashAttention-2](https://arxiv.org/abs/2307.08691)). Benchmark
partition thresholds and block sizes at 256, 512, 1K, 2K, 4K, and 8K; compare
`precise::tanh`/`exp` only against an explicitly bounded approximation because
Gemma's attention softcap is model semantics, not cosmetic postprocessing.

KV quantization should remain a long-context option. QServe demonstrates that
weight, activation, and KV precision must be co-designed with kernels rather
than treated as storage formats ([QServe](https://arxiv.org/abs/2405.04532)),
but its NVIDIA serving results do not establish a Metal win. Locally, any F16 to
Q8/Q4 cache conversion must beat dequantization while preserving Gemma's GQA,
ring masks, mixed windows, and output quality. The measured 2.55 ms short-context
attention ceiling makes this inappropriate for the headline benchmark today.

#### The output head is the largest conventional byte-removal opportunity

The tied 3584 x 256000 F16 matrix is 1.835 GB, 26.8% of every token's weight
stream. Its kernel already reaches about 100 GB/s, so changing threadgroup
geometry has little room. Fusing top-k into the projection would remove a
roughly 1 MB logits write/read and the measured 0.285 ms selector, but it would
still fetch every F16 weight and therefore cannot be a large gain.

If exact checkpoint behavior is not mandatory, quantizing only this matrix at
load time is the clearest conventional way to remove bytes while leaving the
already-quantized transformer untouched. Assuming the observed 98 GB/s can be
maintained per compressed byte, these are first-order bounds:

| Output representation | Approximate bytes | Head time | Whole-token bound from a 72 ms baseline | Main risk |
| --- | ---: | ---: | ---: | --- |
| Existing F16 | 1.835 GB | 18.72 ms | 13.9 tok/s | None; canonical behavior |
| Q8_0-style | 0.975 GB | 9.95 ms | 15.8 tok/s | Small but nonzero logit/distribution change |
| Q6_K-style | 0.753 GB | 7.68 ms | 16.4 tok/s | Larger ranking and probability error |
| Q4_K-style | 0.516 GB | 5.27 ms | 17.1 tok/s | Material quality and sampling divergence risk |

These are bandwidth bounds, not measured candidates. Because the matrix is tied,
the clean design either replaces it and dequantizes the single embedding row on
lookup, or keeps F16 embedding storage plus a separate quantized output copy at
the cost of memory. Evaluation must include per-token max/mean logit error,
top-40 overlap, KL divergence after softcap and configured sampling transforms,
perplexity, deterministic chat regressions, and representative quality suites.
It is a different numerical model even if generated prose often looks similar.

If exact behavior is required, Phase 1's certified output shortlist remains the
only proposal with comparable byte-removal potential. Approximate vector-index
output work confirms that maximum-inner-product search can attack large
vocabularies, but it changes results without a certificate
([vector-index output embeddings](https://openreview.net/forum?id=nYQVHA4prY)).
The local selector itself does not justify a replacement: although radix and
row-wise GPU top-k algorithms are much more sophisticated than the present
bitonic tiles ([RadiK](https://arxiv.org/abs/2501.14336),
[RTop-K](https://arxiv.org/abs/2409.00822)), 0.285 ms caps even a perfect
selector at about 0.4% of current token latency.

#### Fusion, RoPE, normalization, and dispatch ceilings

An aggregate diagnostic comprising all four norm/residual operations plus GELU
for all 42 layers took about 0.53 ms. All query RoPE and fused K-RoPE/F16 KV
stores took about 0.15 ms. These values show that several attractive code-level
ideas have very small whole-token ceilings:

- Interleave gate/up weights and fuse GELU to avoid their intermediate writes;
  the weights still dominate and all norms plus GELU together cost only 0.53 ms.
- Fold normalization into Q/K/V, FFN, or output projections; this either repeats
  a global reduction or merely saves tens of kilobytes of activation traffic.
- Precompute one RoPE sin/cos table per position or fuse query rotation into
  attention; this may save energy and dispatches but is bounded by the measured
  0.15 ms aggregate.
- Fuse output projection and top-k; it is a sensible cleanup after larger work,
  with a sub-millisecond ceiling.
- Replace barriers, command encoding, or sampling; Phase 2 already bounded the
  host envelope near 0.55 ms and rejected serial/no-barrier execution.

Fusion can still matter inside a multirow verifier because it prevents small
costs from multiplying by row count, but it should be evaluated as part of that
verifier rather than as a standalone decode mission. Apple's guidance also
requires barriers for conflicting resources in concurrent dispatches, so broad
barrier removal without a resource-level proof is invalid
([Apple silicon Metal synchronization](https://developer.apple.com/documentation/apple-silicon/porting-your-metal-code-to-apple-silicon)).

#### Ranked Phase 3 conclusions for final planning

1. **Keep the native width-two causal verifier as the main step-change gate, but
   budget it from sustained ratios.** Target less than 1.35x one-token latency,
   emit both logits, enforce causal attention, maintain tentative KV, and prove
   target-distribution parity. Measure width four only after width two; it needs
   substantially more than two accepted tokens to justify its 2.3-2.6x
   quantized-projection cost.
2. **Add sustained and hardware-counter diagnostics before tuning matvecs.** A
   one-dispatch result measures ramp/underfill. Record per-pipeline bandwidth,
   ALU, memory, control-flow, occupancy, and register-limit evidence with Metal
   counter samples or an Xcode GPU trace
   ([Metal counter sample buffers](https://developer.apple.com/documentation/metal/gpu-counters-and-counter-sample-buffers)).
3. **Choose an output-head policy explicitly.** If numerical model changes are
   acceptable, prototype Q8 output-only repacking first; it has the safest
   quality/speed trade and an approximately 14% whole-token ceiling. If exact
   behavior is required, pursue the certified shortlist instead. Do not spend
   the same effort merely replacing the 0.285 ms selector.
4. **Treat FFN weight bytes as the largest conventional target.** Gate/up/down
   own 57.3% of the stream. Test Metal-native repacking or alternate low-bit
   representations only when they physically reduce bytes or counter evidence
   shows unpack ALU limiting; the current native-GGUF kernels already sustain
   about 98 GB/s.
5. **Optimize attention for long context, not the current short benchmark.** A
   block-online softmax, context-specific partition autotuning, and eventually
   KV quantization have a measured 20-plus-ms full-context opportunity but only
   about 2.55 ms at 256 tokens.
6. **Use fusion as verifier support and cleanup.** Gate/up/GELU, RoPE lookup,
   final norm/output, and projection/top-k fusions are individually bounded
   below roughly one millisecond in today's one-row path.
7. **Do not expect a large result from generic launch reduction, CPU sampling,
   a Q6-only rewrite, or blindly transplanting llama.cpp/CUDA kernels.** Direct
   local measurements put the decoder at the memory roofline and identify bytes
   per committed token—not missing arithmetic—as the governing metric.

#### Phase boundary

Phase 3 is complete. All temporary diagnostics were removed and no production
behavior was retained. All three research phases are now complete; the next
step is the separate implementation-plan phase and requires explicit approval.

## Final Planning Gate

Planning was explicitly approved after all three research phases completed.
The combined implementation strategy now lives in
[`DECODE_ACCELERATION_MASTER_PLAN.md`](DECODE_ACCELERATION_MASTER_PLAN.md);
individual implementation slices remain owned by `docs/NEXT_TASK.md`.
