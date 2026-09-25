# Next Task

## Active baton: archive research and prepare a clean Gemma foundation

The user requested a plain-language speculative-decoding retrospective and a
clean main branch for the next Gemma 3 / Gemma 4 investigation. This supersedes
the previous research and verifier-optimization batons. Do not start training,
download models, implement a new target family, or promote speculative defaults
during this cleanup.

Status: cleanup, source review, full CI and retained-path checks are complete.
Stop for user review. Do not begin the next target integration automatically.

## The detailed plan is preserved, not discarded

The entire pre-cleanup source tree, previously uncommitted work, and detailed
5,599-line experimental contract are preserved on the local branch:

`archive/speculative-research-2026-09-25`
at `4579c7f5b903006e7db4f7c82e6e2dce9b6533da`.

The earlier `DECODE_RESEARCH_PROGRAM.md` and
`DECODE_ACCELERATION_MASTER_PLAN.md` also live in this archive, not as competing
implementation plans on main. The history includes source identities,
candidate definitions, training commands, numeric decisions, full experimental
protocols, results, and
unresolved source fields. The old architecture-family contract is historical,
not an instruction to automatically resume every arm on main.

Read it without changing branches:

```sh
git show archive/speculative-research-2026-09-25:docs/NEXT_TASK.md
```

See [the retrospective](SPECULATIVE_DECODING_RETROSPECTIVE.md) for the readable
account and earlier branch identities. Ignored weights, training outputs,
traces, environments and benchmark evidence remain local; Git does not archive
those files. Nothing is pushed remotely by this task.

Stale debug/release trace-collector and recurrent-inspector executables were
moved out of the active build directories to the ignored local directory
`build/archived-research-tools-WMH9US/`; they were not deleted.

## What remains implemented

- Native GGUF loading, tokenization, Q4_K/Q6_K/Q8_0/F16 operations, model-resident
  Metal execution, and the working Gemma 2 9B chat engine.
- Stage 4 retrieval speculation as the unchanged default/control, with exact
  sampled-token matching. No learned candidate earned a default promotion.
- The complete pinned Gemma 3 270M-IT Q8 drafter, native full-vocabulary
  selection, control-preserving tokenizer bridge, separate logical positions,
  F32 local/global K/V, private transactions and pending final-token state.
- Explicit lossless speculative sampling: the target probability law is
  authoritative. Seeded output need not equal Stage 4. The finite-grid proposal
  law, residual correction, sampler checkpoints, callbacks, stops and rollback
  remain part of correctness.
- Shared chain verification, full/conditional execution, validated graphs,
  forests/repeated-root topology coverage, private target K/V and accepted-path
  publication. Existing capacities are not reduced to the current H3 screen.
- Authenticated source and resident-target identities, fixed-capacity feature
  capture and draft-cache ownership. These are foundations, not a finished
  Gemma 4 MTP assistant.
- Independent Gemma 3 CPU equations and pinned real-checkpoint logits/K/V
  goldens, along with native CPU/Metal and lifecycle coverage.

The supported chat target is still **Gemma 2**. Having a Gemma 3 drafter does
not implement Gemma 3 target inference. Gemma 4 target or assistant inference
is not implemented. The current tokenizer bridge deliberately rejects other
target profiles; no compatibility guard is relaxed as part of cleanup.

## What belongs in the archive

The trained Arm-R runtime, compact/corrected vocabulary experiments, Vicuna
model executor/conversion, learned-artifact envelope, trace collector, R/P
training and export tools, associated CLI controls, and their dedicated tests.
The unused 1,246-entry research-shape registry and standalone simulation
helpers also belong there; they are not the live graph validator or the target
and draft transaction owners.
Reusable ownership, numerical tests and tokenizer functionality are retained
or separated before deleting their old owning modules.

The R implementation, its three-seed training configuration, and unfinished P
equations remain recoverable from the archive. Trained checkpoints stay in the
ignored local outputs, not in Git. Those tests are not evidence that every
planned architecture trained or ran in production. Archive removal is a scope
decision, not a new scientific rejection of those families.

## Boundaries for the next investigation

After this cleanup, choose one actual target/checkpoint and its intended
drafter before authorizing integration. Gemma 3 12B with the existing 270M
drafter is a candidate, not an established speedup; memory residency and real
agreement must be measured. Gemma 4's dedicated MTP assistants are a different
architecture with target-feature/K/V coupling, not drop-in standalone drafts.

For either choice:

1. Authenticate the precise model, quantization and tokenizer semantics.
2. Implement that target's actual attention, positions, normalization, head,
   cache and chat-template rules; do not rename the Gemma 2 plan.
3. Implement the chosen drafter's actual source-defined mechanism. Preserve
   distinctions between independent token histories and target-integrated
   feature/K/V ownership. Do not guess unpublished source fields.
4. Keep production inference engine-owned, local and Metal-first. External
   libraries may train models or produce independent goldens, not execute chat.
5. Preserve target probabilities and numerical fidelity, transactional K/V,
   stop/callback behavior, regeneration and context shifting. Fail closed on
   malformed/incompatible artifacts, overflow or unavailable Metal.
6. Test the changed mechanism independently, then measure the real executable.
   Acceptance and a weight-bandwidth estimate alone are not speed evidence.

Use the existing short real-chat screen and the thermal isolation required by
AGENTS.md. Different lossless outputs imply different natural-stop workloads;
retain counts, times and hashes and do not call those exact-work comparisons.
A candidate only replaces Stage 4 after its applicable retention gates pass.

## Cleanup verification and handoff

Full release `bash scripts/ci.sh` passed **49 Python and 423 C++ tests** with
real Metal access and no compiler warnings. Log:
`build/gemma-foundation-cleanup-final-ci.log`. Independent Gemma 3 equations,
pinned full-checkpoint logits/all-layer K/V, full-context overflow, unavailable
device, callback, rollback, regeneration and context-shift tests all remain.
All nine shared standalone lifecycle tests were migrated to full-geometry
Gemma 3 fixtures. The sampler, chain verifier and core target verifier were
not rewritten. Archived-feature tests were removed with their implementations,
not relaxed to make retained behavior pass.

Four fresh production-chat processes used the canonical two-turn, seed-42,
natural-stop workload and a separate 240-second idle cooldown before each.
Order: archived Stage 4, cleaned Stage 4, archived Gemma 3 H3/conditional/lossless,
cleaned Gemma 3 with identical settings. No concurrent build/test workload or
kernel proxy was used. Streams were captured, not rendered in a terminal.

| Mode / turn | Tokens, both builds | Before generation s | After generation s | Before / after TTFT ms |
| --- | ---: | ---: | ---: | ---: |
| Stage 4 first | 276 | 17.307748 | 16.855057 | 167.213 / 166.665 |
| Stage 4 follow-up | 322 | 20.546937 | 20.298186 | 1547.331 / 641.388 |
| Gemma 3 first | 337 | 21.100417 | 21.088412 | 191.506 / 193.029 |
| Gemma 3 follow-up | 346 | 23.688253 | 23.733111 | 420.493 / 429.312 |

Stage 4 aggregate generation was **15.7973 → 16.0955 tokens/s**; Gemma 3 was
**15.2494 → 15.2382 tokens/s** (-0.073%). Within each mode, both output hashes,
token/cache positions and checked execution counters matched exactly. Gemma 3
also preserved all acceptance/rejection histograms, 273 cycles and 388 accepted
draft tokens, plus draft transfer/dispatch counts. This is an exact-work
cleanup comparison within each mode, not an exact-work Stage-4-versus-Gemma-3
comparison or a new speculative-default admission.

Performance caveats remain: Gemma 3 follow-up TTFT rose **2.10%**; rounded prompt
throughput was 83.5→82.9 and 45.2→44.3 tokens/s. Its complete child-process wall
time (including the fixed inter-turn pause) was 124.954→123.855 seconds, while
peak child RSS was 9,868,525,568→10,716,856,320 bytes. Reported target resident
buffer sizes, zero transient allocations and execution work were unchanged;
draft-owned storage decreased by 16 bytes after removing the old wrapper.
OS page residency and swap differed.
The later runs had roughly 2.7–3.5 GB of system swap in use. `pmset` reported no
recorded thermal/performance warning; no GPU clock or temperature trace proves
equal thermal state. The first Stage 4 run did not record process wall/RSS.
These checks establish preserved observed behavior and essentially unchanged
Gemma 3 generation, **not a performance win or satisfaction of promotion
gates**. Do not turn the startup/TTFT/RSS variation into a causal claim.

Raw timing, counter, hash and available system records are under ignored
`benchmarks/results/gemma-foundation-cleanup-3iDPvN/`. Archived executable SHA:
`4b027cbab15ef5b1b3eab6d619c26b83e6b310a7ec28a7fcdd1d57743bfebb92`;
cleaned executable SHA:
`184085ea12c9bb8c418a290df2fe4c2fc7b57051000e9d8daa59894519edf580`.

The reviewed cleanup is handed off through a local fast-forward of main.
The full research archive remains separate and no remote push is performed.

No training process is started by this task. The stop condition is the
verified cleanup and documentation handoff, not a new speculative-decoding
implementation.
