# Current state and next work

*Updated October 14, 2026. This file owns the current handoff; completed
experiments belong in the [retrospective](SPECULATIVE_DECODING_RETROSPECTIVE.md)
and [performance record](PERFORMANCE.md).*

## Verified production state

Guffin is an engine-owned, local C++20/Metal GGUF engine for a 16 GiB MacBook
Air M4. Gemma 3 12B is the primary target; Gemma 2 9B remains supported.
Installed chat starts with `guffin -m MODEL`. Gemma 4 remains future work.

- Gemma 3 uses native Q4/Q6 weights, F16 target KV with F32 attention and a
  16K default context allocation. Bulk prompt execution shares quantized
  weights across rows while preserving F32 projection arithmetic.
- `--draft-model FILE` enables the authenticated native Gemma 3 270M Q8_0
  drafter, H1, full verification and lossless sampling. Its F32 KV caps paired
  context at 32K. Lossless preserves the configured target law; it does not
  require identical seeded text. Larger horizons have no retained chat gain.
- Gemma 2 uses F16 KV, bounded suffix retrieval by default and fused bulk
  attention. Its Q6-down → Q4 and F16-head → Q8 conversions remain explicit
  numerical policy. The transferred target improvements stayed; learned
  recurrent drafting did not.
- Shared session, model and preparation APIs retain explicit family behavior.
  Move-only target/verifier facades and storage-retaining tickets protect
  transactions; tensor admission checks descriptor and backing contracts.
  Numerical references, fault helpers and benchmark drivers stay outside
  shipping objects.
- `make install` builds an optimized, self-contained executable, with
  `PREFIX`/`DESTDIR` support. Chat shows prompt/generation rates and internal
  TTFT; one-shot timings use `--stats`, attribution uses `--decode-diagnostics`.

## Release evidence and review

The three retained release campaigns are complete: **38 conversations / 80
turns per engine**, with no failed, interrupted or unattempted cases. They use
the same frozen user messages, fresh processes, 300-second launch cooldowns,
120-second inter-turn pauses and each CLI's inference defaults.

| Campaign | Core generation, Guffin / llama.cpp |
| --- | ---: |
| Gemma 2 default | 16.76 / 13.63 t/s |
| Gemma 3 ordinary | 12.57 / 12.40 t/s |
| Gemma 3 speculative | 18.43 / 8.43 t/s |

[Performance evidence](PERFORMANCE.md) links the committed manifests, receipts,
logs and reports, and records TTFT, memory and interpretation limits. These are
single-pass performance observations, not answer-quality or numerical
correctness evaluations. Gemma 2 retains its distinct executed-weight policy;
the speculative comparison retains each engine's own algorithm and defaults.

The October 14 Metal/runtime review found no blocking defect. The resulting
source fixes preserve shader arithmetic, schedules and precision while aligning
unknown-token decoding and hardening benchmark resource/receipt handling.
Offline validation checked all **76 receipts / 160 turns per campaign**, answer
hashes and log references; regenerated JSON/CSV/Markdown/SVG reports were
byte-identical. Final CI passes **537 registered native cases and 132 Python
cases**, with 15 existing Python skips and no compiler warnings. Artifact-dependent native
reference cases retain their availability gates.
No performance campaign was rerun. The README is unchanged in this
pass; the next baton below owns its deferred refresh.

## Remaining numerical boundary

The retained Gemma 3 F16 target exceeds the original independent full-model
bounds on a reproduced natural 128-token prompt. Its F32 control passes. The
investigation found amplified rounding feedback without establishing an
incorrect model equation, cache index or half conversion. Later exact prompt
optimizations preserve the original logits and caches; neither the release
campaigns nor those parity checks resolve this independent boundary.

Half-operand prompt candidates also failed their unchanged matching-policy
logit/cache gates and were removed. No tolerance was relaxed. Wider-context
and answer-quality fidelity remain unestablished; lossless H1 preserves the
configured native target law. See [numerical limits](PERFORMANCE.md#numerical-limits).

## Next baton: refresh the README

**Objective:** at the next user prompt, refresh the README using the three
final campaigns and the current architecture. The user explicitly deferred
that edit during the present pass; this review and cleanup are complete.

**Constraints:** preserve its explanatory structure; attach every headline to
its model, mode, suite and measurement boundary. Keep ordinary and speculative
comparisons distinct, disclose Gemma 2's numerical policy, slower large-prompt
TTFT and higher reported physical footprint with mapped-weight accounting
limits, and retain the F16 fidelity limit.
Do not infer quality correctness from successful conversations. Use the
retrospective for the longer development story and GRAPH only to locate code.

**Verification:** check figures and links against the retained manifests,
receipts and reports without rerunning inference. Check Markdown/diagram
rendering and ensure no headline depends on removed development artifacts.

**Stop condition:** the README accurately presents the released engine,
measured benefits and limitations. New inference optimization, precision
qualification or model coverage is a separate task.
