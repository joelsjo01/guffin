# Next Task

## Status

Stage 1 Baton 2 is complete as of 2026-08-07. Stage 1 is complete and no
implementation task is active. Stop here until the user explicitly authorizes
Stage 2; do not begin storage-mode, kernel, matmul, quantization, or speculative
work implicitly.

## Retained Baton 2 implementation

### Terminal body-only decode

Resident decode now has an explicit `logits` versus `kv_only` output mode. When
the generation loop knows before encoding that the selected token is a stop
token or the final token allowed by the limit, `kv_only` still executes the
embedding, every transformer layer, and every K/V store, but omits final
RMSNorm, the F16 output projection, selection, and their two barriers. A null
selection request still produces full logits by default, preserving diagnostic
and API behavior.

The schema-2 production attribution reports the number of terminal output-head
elisions and the exact skipped weight bytes. On the target, one terminal step
skips the 1,835,008,000-byte output head plus the 14,336-byte output norm, for
1,835,022,336 bytes total. Scheduled bytes for that step fall from
6,837,376,000 to 5,002,353,664, while its cache position and decoded-token count
remain unchanged.

Callback, reverse-stop, and cancellation termination deliberately retain full
output execution. Their decisions occur only after the current decode succeeds
and its KV is resident. Moving delivery before GPU success would change error
and cache ordering; splitting every continued token into a second output
command would damage the hot path.

### Physical prefill-scratch liveness

Three exact, equal-descriptor aliases are retained in the serial prefill path:

- attention probabilities replace dead attention scores in place;
- one hidden-width workspace serves normalized hidden, projected attention,
  normalized FFN input, and projected FFN output at disjoint live ranges;
- GELU output replaces its dead gate input in place.

This removes 79,953,920 persistent bytes. Prefill scratch falls from
176,707,584 to 96,753,664 bytes and total plan scratch falls from 181,146,892
to exactly 101,192,972 bytes, a 44.14% reduction. The plan counts shared storage
once, and tests assert both the intended storage identities and numerical
prefill parity. No arena, offset view, cross-prefill/decode alias, or hazard-mode
change was added.

### Linear incremental streaming decode

`Tokenizer::append_decoded_token()` is now the single implementation of token
decoding, and batch `decode()` delegates to it. Chat streaming appends only the
new token instead of decoding the complete growing prefix, changing token-piece
visits from quadratic to linear. Reverse-stop cleanup uses the same append
operation, while matching rescans only the previous possible boundary overlap
plus new bytes.

SentencePiece space markers, skipped special/control tokens, byte fallback,
partial multibyte UTF-8 bytes, callback chunking, and invalid-ID errors remain
byte-for-byte compatible. Reverse-stop tests include a marker spanning two
generated tokens.

## Verified target result

Five paired, interleaved one-token terminal runs with ten-second cooldowns
produced five of five wall/GPU wins:

| Metric | Baton 1 median | Baton 2 median | Change |
| --- | ---: | ---: | ---: |
| Terminal decode wall | 73.638 ms | 53.748 ms | -27.01% |
| Terminal decode GPU | 71.629 ms | 53.036 ms | -25.96% |
| Compute dispatches | 633 | 631 | -2 |
| Memory barriers | 465 | 463 | -2 |
| F16 projections | 1 | 0 | -1 |
| Scheduled weight bytes | 6,837,376,000 | 5,002,353,664 | -26.84% |
| Persistent scratch bytes | 181,146,892 | 101,192,972 | -44.14% |

Startup-to-ready was neutral at -0.42%. Process RSS varied widely in both
directions because the lazy shared allocations and file-backed pages were not
consistently resident at the prompt; it is not used as evidence for the exact
allocation reduction. Peak RSS changed by -0.49%. The allocation byte count,
storage identities, and virtual-size reduction are deterministic.

The canonical two-turn sampled chat passed with unchanged prompts, seed,
natural stops, and timing definitions:

| Turn | TTFT | Prompt t/s | Generation t/s | GPU/decode token | Wall/decode token |
| --- | ---: | ---: | ---: | ---: | ---: |
| First | 174.6 ms | 91.6 | 13.9 | 71.08 ms | 71.92 ms |
| Follow-up | 460.8 ms | 41.2 | 13.7 | 71.97 ms | 73.19 ms |

Only one terminal token per turn is elided, so sustained generation throughput
is expected to remain neutral. The improvement is turn-tail latency and future
memory headroom, not a claim that ordinary full-output decode is 27% faster.

## Correctness and verification

- Body-only and full-output decode produce byte-identical live K/V at a
  nonzero cache position against the independent CPU reference.
- Body-only mode rejects a selection request and removes exactly final norm,
  output projection, and two barriers; full-logits-without-selection remains.
- Stop/max paths elide exactly one output head. Callback cancellation and
  reverse-stop paths assert zero elisions.
- Existing prefill chunk/ring parity passes with the liveness aliases.
- Incremental and batch tokenizer decoding match across SentencePiece,
  special/control, and multibyte byte-token boundaries.
- `bash scripts/ci.sh`: passed, 17 Python tests and 308 C++ tests.
- `make MODE=release benchmark`: passed after the final accounting change.
- Five-pair production memory/terminal-decode benchmark: passed.
- Canonical `scripts/quick_benchmark.py` run: passed with schema-2 attribution
  and sustained kernel diagnostics. The final accounting-only follow-up was
  smoke-tested on the target and reported exactly 1,835,022,336 skipped bytes.
- `git diff --check`: rerun at final handoff.

No model, generated binary, benchmark output, production inference dependency,
alternate scheduler, or framework runtime is tracked.

## Next authorized boundary

Stage 2 begins only on explicit user approval. Its first decision should use
the Stage 0 attribution and Stage 1 headroom to compare weight residency/access
changes against conventional single-token decode fusion and byte reduction.
Callback-tail elision remains deferred unless a design preserves
successful-execution-before-delivery ordering without adding a command boundary
to every nonterminal token.
