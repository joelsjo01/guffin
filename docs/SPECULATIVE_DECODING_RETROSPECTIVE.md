# Trying to make speculative decoding worthwhile on a MacBook Air

*A project retrospective, September 2026.*

The goal was simple: make a local language model respond faster on my MacBook
Air M4. The engine already ran Gemma 2 9B Q4 at roughly 16 tokens per second in
our short chat benchmark. Speculative decoding looked like a way to go further.

The result was less exciting than the idea. A small retrieval-based version
worked and stayed. The learned drafters did not produce a convincing speedup.
Along the way we trained models, integrated downloaded ones, found real bugs,
built too much infrastructure, and eventually removed experiments that were
not earning their place.

This is the account I wish I had before starting: what actually ran, what its
numbers meant, and which conclusions we can reasonably draw.

## The idea, and the first small win

Ordinary generation asks the large model for one token at a time. Speculation
lets something cheaper suggest several future tokens, then asks the large
model to check them together. If enough suggestions survive, fewer expensive
target passes are needed.

The important word is *cheaper*. A draft that predicts well can still lose if
drafting and checking cost more than the work they save. Checking a batch is
also not automatically as cheap as checking one token.

The roughly 17.5-token/s "limit" we discussed was a weight-streaming estimate
for ordinary one-token generation, not a hard limit on what the computer can
ever produce. Reusing a weight pass across several useful tokens can beat that
estimate. The difficult part is making those tokens useful cheaply enough.

Our first useful version, called Stage 4, needed no model training. It looked
for repeated token sequences in the current conversation and proposed one
continuation when the evidence was strong enough. A specialized two-row Metal
verifier checked the proposal while preserving the ordinary sampler's exact
choices.

It worked: the same 598 generated tokens took 37.233 seconds instead of
38.293, a 2.77% reduction in generation time. Both turns improved, with identical
output hashes and cache positions. Retrieval itself was extremely cheap.

It was not a dramatic acceleration. There were only 26 speculative cycles,
including 17 matches, against hundreds of ordinary fallbacks. But it was a real
end-to-end gain, and it became the baseline the later experiments had to beat.

## Some early failures were our mistakes

Earlier branches tried compact learned output heads: a low-rank representation
and a fixed 32,000-token vocabulary instead of Gemma's full vocabulary. One
training run initially achieved only 6.19% validation proposal acceptance.

An audit found a genuine training problem. An inverse-square-root scaling had
been applied twice, making the feature-path gradients 16 times smaller relative
to the output bias. Correcting that and replacing the short SGD procedure with
AdamW recovered trainability. Validation acceptance rose to 22.20%.

That was still far from useful, but the explanation changed. The original
number had not established the model's potential. It had measured a collapsed
training run. The repaired head learned the training data much better than it
generalized to other conversations. Float and quantized validation results were
almost identical, so blaming four-bit weights would have been wrong too.

Later compact-head experiments managed to fit a small training cohort very
well and still failed on held-out conversations. Those were representation
experiments, not complete recurrent drafters producing faster chat.

This became a recurring lesson: reject the thing actually tested. A broken
trainer does not reject an architecture; a slow kernel does not reject every
possible implementation of speculative decoding.

## We built a research programme around a hobby project

I wanted a correct implementation, not a deliberately weakened approximation
that failed for avoidable reasons. That was a reasonable goal. The scope that
grew around it was not always reasonable for the immediate question.

The plan expanded into many model families, chain lengths, trees, beam layouts,
source-identity records, training controls and admission protocols. Phase 0
included machinery for attestations, simulations and evidence receipts. The
statistical design covered multiple workloads and days, cold and sustained
runs, and large bootstrap simulations.

There were valid ideas underneath this. Repeated windows from one conversation
are not independent examples. Paper settings and released code sometimes
differ. A fanless laptop can throttle. But building a comprehensive framework
for all of those concerns became work of its own, before we had a useful learned
decoder. Some of that administrative machinery was removed.

We also built real target-verification infrastructure for long chains, forests
and repeated-root beam batches. Some shapes ran against the actual Gemma model
with perfect or scripted proposals. That established correctness and optimistic
costs for those target shapes. It did not mean we had trained all the proposed
architectures, or that a tree decoder had demonstrated a chat speedup.

The better distinction is between implementing a chosen experiment faithfully
and implementing an entire field before choosing the next experiment.

## The three models we really trained

The main learned control was Arm R: a recurrent drafter conditioned on target
features and previous draft tokens. H3 meant three proposed future tokens.

We trained three random seeds of that same configuration, not three different
architectures. The preparation involved real target-generated traces. An initial
large corpus was valid but not balanced enough for the frozen workload splits.
A replacement collected only the final-normalized features this model needed,
making each captured feature bundle 44 times smaller without changing its
inputs. Total corpus sizes also depended on how many examples were collected.

I completed the collection and training: 2,415 traces, 100,000 balanced
training origins, and 83,340 validation origins. Each seed ran six epochs and
37,500 optimizer updates.

The best weighted validation accepted-prefix scores were approximately 0.802,
0.802 and 0.791 out of three. That did not mean 80% acceptance. It meant about
0.8 consecutive draft tokens accepted per evaluated origin under that offline
metric. Accepting the first suggestion but rejecting the second counts as one;
a plausible third suggestion cannot rescue a broken prefix.

The live chat result was worse: about 0.37 accepted draft tokens per learned
cycle, and generation dropped from 16.11 to 8.28 tokens per second. The offline
validation score and this short, routed chat measurement were different
evaluations, not interchangeable numbers.

The implementation had learned something. It simply was not saving enough
target work to pay for its own drafting and verification.

## “Same quality” was not “the same seeded sentence”

An important clarification came late. We had originally enforced identical
token choices for the same seed. What I actually needed was no degradation of
the target model's output distribution, not identical wording on every run.

Standard lossless speculative sampling supports that distinction. It samples
from the draft distribution, accepts with a target-to-draft probability ratio,
and uses a corrected distribution after rejection. The correction is what
preserves the target's probability law; merely accepting plausible-looking
tokens would not. This is the mechanism described by
[Leviathan and colleagues](https://proceedings.mlr.press/v202/leviathan23a.html).

We implemented it with explicit proposal probabilities, separate checkpointed
randomness, and rollback at callback boundaries. The actual finite-grid sampler
law mattered, not an ideal softmax substituted for what the device really did.

We reused the existing R weights. They had been trained for the earlier
common-draw objective, not retrained for the new sampler. Acceptance improved
to roughly 0.42 draft tokens per cycle, but the run still achieved only
8.50 tokens per second against a 15.83 baseline.

Changing the correctness contract enabled the standard algorithm. It did not,
by itself, solve the cost problem. Lossless also means preserving the chosen
target distribution, not guaranteeing that every individual answer is good.

## A bigger design that never became a trained result

Arm P explored a more substantial parallel drafting backbone with sequential
correction. Work reached source-based offline equations, gradients and a
five-tap target-feature boundary. The declared full-width Gemma transfer had
about 1.19 billion learned parameters.

The memory ledger already exceeded 17 GiB for float weights, gradients and Adam
state alone, before embeddings and activations. That was a resource warning,
not a measured speed verdict. No full-size training run happened, and there was
no native P backbone or P chat benchmark. It belongs in the story as unfinished
work, not another trained drafter that failed.

## Downloading a drafter got us closer

Next we tried existing pretrained models instead of training another small
predictor ourselves.

Vicuna-68M was genuinely small: two layers and a 32,000-token vocabulary.
We implemented the complete checkpoint in our own Metal runtime. Different
tokenizers required an explicit bridge, independent draft positions and careful
handling of target corrections. The vocabulary-intersection approach came from
[research on lossless speculation with heterogeneous vocabularies](https://arxiv.org/abs/2502.05202v3).
Our quantized Gemma/Mac execution was a transfer, not a reproduction of the
paper's hardware benchmark.

Vicuna drafted cheaply, but averaged only 0.54 accepted draft tokens per cycle.
Generation was 7.26% slower than Stage 4. The target verifier cost about 98 ms
per cycle, against roughly 4.55 ms of draft GPU work. A tiny draft model was
not enough when too little of its work survived.

Gemma 3 270M did better. This was still a Gemma 2 target, now paired with a
complete 18-layer Gemma 3 drafter and its full 262,144-token vocabulary. It
averaged 1.42 accepted draft tokens per cycle. Yet it still failed retention.

Here are representative real-chat screens. Each row has its own baseline;
these are not one controlled tournament across identical machine states.

| Experiment | Stage 4 tokens/s | Candidate tokens/s | Accepted draft tokens/cycle |
| --- | ---: | ---: | ---: |
| Trained R H3, exact-match | 16.11 | 8.28 | 0.37 |
| Same R weights, lossless | 15.83 | 8.50 | 0.42 |
| Vicuna-68M H3, conditional | 15.98 | 14.82 | 0.54 |
| Gemma 3 270M H3, initial screen | 15.13 | 15.41 | 1.42 |
| Gemma 3 270M H3, later sweep | 15.91–16.03 | 15.37 | 1.42 |

The apparent Gemma 3 win in the first screen was not sufficient: it missed the
aggregate gate and lost on the follow-up. Lossless runs also produced different
text, stop lengths and subsequent contexts. Unlike the exact-match R row, they
were not identical-work comparisons.

## More tokens, different scheduling, small kernel wins

We tested draft lengths two, three, four and six, with full-batch and conditional
verification. Conditional means later target tiles run only after the earlier
prefix survives. It keeps the requested horizon; it does not silently shorten
the model's proposals.

H3 conditional remained best. H6 raised average acceptance to 1.64 tokens but
fell to 13.28 tokens per second. The sixth proposal reached an actual decision
in only 26 of 220 cycles. Generating a longer chain was mostly buying work we
never used.

Scheduling mattered substantially. H3 full verification achieved 11.94 tokens
per second versus 15.37 conditional, with identical outputs and acceptance.
Changing draft-only temperature from 0.8 to either 0.6 or 1.0 made things worse.
That tested two settings, not every possible calibration or distillation method.

Kernel experiments were mixed too. A four-row weight-sharing implementation
passed numerical checks but slowed an R run to about 5.2 tokens per second.
A repeat confirmed the regression, and it was removed.

The final two-row scheduling change was much less dramatic: 15.25 to 15.56
tokens per second, about 2.03%, against the previous speculative implementation.
It preserved the outputs exactly. But whole-process time was 5% worse, paging
differed, and this was neither an 18-token/s result nor a win against Stage 4.
It was removed at my request as well.

## What was worth keeping from all this?

Correctness work did catch real defects. Draft-cache half-precision rounding
boundaries prompted a move to float draft K/V rather than weaker tolerances.
Regeneration could not safely reconstruct old decoded tokens as one new prefill
when that changed rounded cache values. Rejection and callback interruption
needed actual state restoration, not just a smaller token counter.

Those checks were useful. They are different from building elaborate machinery
for hypothetical experiments. The balance should have been a complete chosen
path, focused independent tests, and early measurement of the real executable.

Measurement also needed perspective. Later screens used fresh processes and
four-minute cooldowns, but compression and system swapping still occurred.
No thermal warning is not proof of unchanged GPU clocks. These results were
useful development evidence, not a precise universal ranking.

The central lesson is simple: acceptance, draft cost and verification cost have
to work together. A better score alone is not a faster engine. Neither is a
faster isolated kernel. And a negative result should stay attached to its
actual model, sampler, workload and implementation.

Gemma 3 as both target and drafter remains a possible next investigation, not
something these runs tested. Gemma 4 is interesting for a stronger reason than
sharing a family name: it has dedicated assistants using target embeddings and
activations, with specialized output selection in its smaller variants.
[Google's MTP description](https://ai.google.dev/gemma/docs/mtp/overview)
explains that design. Neither family has yet demonstrated speculative speedups
in this engine. They would require new integration and measurements.

I still think speculation can be worthwhile. The next attempt should begin
with evidence for a promising pair, then build the actual mechanism needed for
that pair. The goal is faster chat, not a larger collection of experiments.

## Where the detailed history lives

The full pre-cleanup research state and detailed `docs/NEXT_TASK.md` are
preserved on `archive/speculative-research-2026-09-25` at
`4579c7f5b903006e7db4f7c82e6e2dce9b6533da` (`4579c7f`). This article summarizes
that history; it does not replace the original measurements or imply every
planned architecture was implemented. Ignored weights and raw benchmark files
are local evidence, not files stored by the Git archive.

Earlier branch records remain separately identifiable:

- `archive/stage5r-b-rejected-2026-08-20` at
  `063e6f617c1e6182ea334795f3d94a0d1db1975e`: broad-head cost experiments.
- `stage6/verifier-feasibility` at
  `14c8a2fe89e497d0b7aaff85809089ff58abeece`: width-four feasibility work.
- `experiment/target-head-recurrent-r256` at
  `e525bcadc0ba1d83594c50c2a9274d3f06c2a5a0`: compact-head training repairs
  and generalization experiments.

For example, the complete archived handoff can be read without changing the
working tree:

```sh
git show 4579c7f:docs/NEXT_TASK.md
```
