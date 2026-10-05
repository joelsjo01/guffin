# Current state and next work

*Updated October 5, 2026. This is the current handoff; completed research lives
in the linked records.*

## Production state

Guffin is an engine-owned, local C++20/Metal GGUF engine validated on a 16 GiB
MacBook Air M4. Gemma 3 12B is the primary target; Gemma 2 9B remains supported.
Gemma 4 is future work. Installed chat starts with `guffin -m MODEL`.

- Gemma 3: native Q4/Q6 target execution, F16 target KV with F32 attention,
  16K default capacity, and retained M32 prompt tiles for at least 32 useful rows.
  Larger capacities must satisfy model/device limits.
- `--draft-model FILE` enables the authenticated native Gemma 3 270M Q8_0
  drafter, H1, full verification and lossless sampling. The drafter uses F32 KV
  and caps paired context at 32K. Lossless preserves the configured target law;
  identical seeded text is unnecessary. Larger horizons remain diagnostics
  without an established retained chat advantage.
- Gemma 2: F16 KV and bounded suffix retrieval enabled by default. Its existing
  Q6-down → Q4 and F16-head → Q8 conversions remain explicit numerical policy.
  The target kernel port stayed; the October Arm R integration was removed.
- Chat metrics show prompt tokens/s, generation tokens/s and internal TTFT.
  `--stats` requests one-shot/piped timings; `--decode-diagnostics` opts into raw
  attribution. `make install` installs an optimized, self-contained executable;
  `PREFIX`/`DESTDIR` support custom/staged installation.

## Cleanup and verification completed

Shared session, model and preparation APIs use Gemma names, while original
family runtime schedules and authentication domains remain explicit. CLI
parsing, interaction and metrics are separate. Numerical references, fault
helpers and diagnostic verifier drivers are excluded from shipping objects.
Abandoned feature banks, hidden-state taps, training capture and auxiliary
research caches are removed.

Target/verifier facades are move-only, with stable setup-owned storage and
owner-retaining tickets. Tensor admission validates descriptor and backing
contracts before allocation. Drafter/chain resource names and control flow are
clearer without another runtime layer. Retained shader bytes, numerical
encoding, submission order, sampler operations and cache algorithms are
unchanged; no new token-loop allocation was added.

The last source-validation pass completed 504 registered native cases without
failure and 64 Python cases with 15 skips. Some native checkpoint/reference
cases also return when artifacts are absent. Debug/release builds had no
compiler warnings. Installed Gemma 3 ordinary/H1 two-turn chats and Gemma 2
ordinary generation passed; saved text and non-timing summaries matched.
H1 diagnostics matched 83/84 non-timing fields on both turns; the remaining
field records a 16-byte verifier host facade. Clean source-export, dependency,
install/uninstall and shipping-symbol checks passed. These were correctness
and lifecycle checks, not new performance benchmarks.

## Documentation completed

The [README](../README.md) now leads with installation/chat and includes
architecture and H1 diagrams, retained mechanisms and an original performance
figure. The [retrospective](SPECULATIVE_DECODING_RETROSPECTIVE.md) explains the
successful and rejected designs. [Performance evidence](PERFORMANCE.md)
recomputes saved rates using complete-generation timing and states comparison
limits. The pinned llama.cpp reference and historical archive are unchanged.
No engine source, models or benchmarks changed during this documentation pass.

## Remaining boundaries and release decisions

An independent 128-token natural-prompt CPU comparison exceeds existing
numerical bounds for the baseline and a removed attention candidate. The cause
is unresolved; top-1 agreement does not waive it. Retained M32 projections are
bit-identical and did not introduce the discrepancy. See the
[numerical boundary](SPECULATIVE_DECODING_RETROSPECTIVE.md#numerical-validation-has-a-boundary).

The older llama.cpp baseline differs in OS, timing, response history and Gemma 2
executed-weight policy. Manual Gemma 3 comparisons are not frozen controlled
results. Do not present a current percentage competitor lead from those data.
A fresh matched comparison would be separate, explicitly requested work.

Release identity and licensing remain maintainer decisions; no license, tag or
release was created. Confirm hosted CI for the eventual release revision.
Use [the code map](GRAPH.md) and [principles](ENGINE_PRINCIPLES.md) for future
implementation, and [development history](archive/DEVELOPMENT_HISTORY.md) for
historical experiments and commands.
