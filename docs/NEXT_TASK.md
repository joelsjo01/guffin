# Current state and next work

*Updated October 16, 2026. This file owns the current handoff; completed
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

The four retained campaigns are complete: **38 conversations / 80 turns per
engine**, with no failed, interrupted or unattempted cases. They use the same
frozen user messages, fresh processes, 300-second launch cooldowns and
120-second inter-turn pauses. The original three use each CLI's inference
defaults; the H1 follow-up overrides only llama.cpp's maximum draft length to
one with `LLAMA_ARG_SPEC_DRAFT_N_MAX=1`. Both engines' logs confirm H1.

| Campaign | Core generation, Guffin / llama.cpp |
| --- | ---: |
| Gemma 2 default | 16.76 / 13.63 t/s |
| Gemma 3 ordinary | 12.57 / 12.40 t/s |
| Gemma 3 speculative defaults (H1 / H3) | 18.43 / 8.43 t/s |
| Gemma 3 speculative, both H1 | 18.53 / 13.26 t/s |

[Performance evidence](PERFORMANCE.md) links the committed manifests, receipts,
logs and reports, and records TTFT, memory and interpretation limits. These are
single-pass performance observations, not answer-quality or numerical
correctness evaluations. Gemma 2 retains its distinct executed-weight policy;
speculative algorithms remain engine-specific. The H1 follow-up records
39.75% higher core throughput and 31.24% higher supplementary throughput in
Guffin. Its binaries, models and corpus match the earlier Gemma 3 campaigns;
an October 16 offline audit checked all 76 receipts / 160 turns and raw metrics.

The source review preserves shader arithmetic, schedules and precision.
Offline validation of the original three campaigns checked all **76 receipts /
160 turns per campaign**, answer hashes and log references; regenerated
JSON/CSV/Markdown/SVG reports were byte-identical. October 15 local CI passes **537 registered native cases and
132 Python cases**, with 15 existing Python skips and no compiler warnings.
Artifact-dependent native reference cases retain their availability gates.
Metal final-buffer initialization now releases failed allocations and drains
its autorelease pool before propagating callback errors; regression coverage
observes the actual buffer lifetime rather than device-wide memory totals.

A fresh export of the intended working tree, without cached build products or
model payloads, builds and installs successfully. The staged optimized binary
links only system libraries and works outside the checkout. Installed Gemma 2,
Gemma 3 and H1 two-turn chat, cache reuse, H1 interruption and continuation,
piped output, one-shot interruption, model errors and uninstall all pass.
A prefix containing spaces was exercised; the existing global installation was
preserved. No performance campaign was rerun for these correctness checks.
The README presents the final results with generation and TTFT charts,
separate first/follow-up medians and large-prompt weaknesses, architecture and
H1 diagrams, supported usage and concise comparison limits. The H1-versus-H1
results and their raw evidence are included alongside the original default
comparison. Figures and claims were checked against raw campaign counts and
current execution code. No performance campaign was rerun for the
documentation refresh.

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

## Release handoff: v0.1.0

**Identity:** the initial release is `v0.1.0` on `main`, with
[release notes](RELEASE_NOTES.md), an MIT license, the final README and the
three original benchmark campaigns. The later H1 comparison is additional
published evidence on `main`; the release tag is unchanged. The repository is
`joelsjo01/guffin`; its SSH remote and account authentication are verified.

**Scope:** keep the release limited to the tested M4 Gemma paths and retain
the documented numerical boundary. Local correctness and installation checks
are complete. No further optimization, model support or benchmark rerun is
requested for this release.

**Verification:** the October 15 CI and installation checks cover the current
runtime source. The maintainer authorized publication without waiting for
GitHub CI; its result must remain unconfirmed until observed. Verify the
pushed source commit, annotated tag and published GitHub Release separately.

**Stop condition:** the release source is on `main`, `v0.1.0` identifies that
commit, and the GitHub Release is published with the prepared notes.
