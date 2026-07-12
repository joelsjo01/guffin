# CLI Milestone Map

Purpose: define the path from the current one-shot generator to a usable local
CLI application that behaves like llama.cpp interactive chat for the supported
Gemma Metal path.

Use this as the durable roadmap. Use `docs/NEXT_TASK.md` for the active
implementation baton and `docs/INTERACTIVE_CLI_RESEARCH.md` for source notes.

Status legend: `Done`, `Current`, `Next`, `Planned`, `Blocked`.

## Non-Negotiable Target

The CLI is not complete when it merely loops over one-shot generation calls.
The target is a persistent local application:

- Model is loaded once per process.
- The engine owns session state: GGUF lifetime, `EngineModel`, tokenizer,
  `Gemma2MetalModelPlan`, KV cache, cached token cursor, chat messages, stop
  tokens, sampler state, and generation settings.
- The CLI owns terminal behavior only: argument parsing, stdin, prompts,
  commands, streaming output, interrupts, and errors.
- New user/template tokens append into resident KV at the current cache
  position. Full replay may exist only as a correctness oracle or debug
  fallback, not as the production milestone exit.
- Gemma chat always uses the model chat template. Do not fake chat with raw
  `User:` / `Assistant:` text.
- One-shot `local-gguf-generate` behavior stays backward compatible.
- No production path wraps `llama.cpp`, Ollama, MLX, PyTorch, TensorFlow, Core
  ML, model servers, or remote inference APIs.

Final application shape:

```text
local-gguf-generate models/gemma-2-9b-Q4.gguf --chat --max-new-tokens 256 --backend metal

local-gguf-engine chat
model: gemma2
commands: /exit /quit /clear /reset /regen /tokens /read

> Hello
assistant text streams here

> Make that shorter
assistant text streams here

> /exit
```

## Milestones

| Status | Milestone | Exit Proof |
| --- | --- | --- |
| Done | One-shot Gemma generate foundation | Current CLI can load a Gemma GGUF, apply the Gemma chat template for one user prompt, run greedy Metal generation, decode output, and exit. |
| Done | CLI application roadmap | This document replaces the old Gemma implementation checklist and defines the end-to-end interactive application path. |
| Done | Persistent chat session | `--chat` starts one process, loads the model once, accepts at least two user turns, appends new turn tokens into resident KV, and keeps one-shot behavior unchanged. |
| Done | Streaming turn output | Assistant text prints as tokens are generated, and EOS / Gemma `<end_of_turn>` / reverse stop strings end only the assistant turn. |
| Done | REPL command surface | `/exit`, `/quit`, `/clear`, `/reset`, `/regen`, `/tokens`, multiline input, and `/read` work with deterministic session/KV behavior. |
| Done | Context-window management | Long conversations have explicit budget checks, shift/trim behavior, cache-position correctness, and useful user-facing recovery messages. |
| Done | Sampler and regeneration parity | Greedy remains deterministic, sampled chat gains seeded persistent sampler state, and `/regen` follows the documented seed/sampler policy. |
| Current | llama.cpp parity comparison | The supported chat path is compared against current llama.cpp Metal with matched model, template, context, stop, sampler, prompt turns, and hardware conditions. |

## Gate 1: Persistent Chat Session

Status: Done on 2026-07-10.

Implemented:

- `Gemma2ChatSession` owns the file-backed GGUF lifetime, `EngineModel`,
  tokenizer, resident `Gemma2MetalModelPlan`, chat messages, cached token ids,
  stop token ids, and generation settings.
- `local-gguf-generate` supports `--chat`, `--conversation`, `-cnv`, and
  `--max-new-tokens N` while preserving the one-shot positional command.
- Runtime generation can run against an existing plan and append a rendered
  chat suffix at a nonzero cache position.
- Prefill attention now writes the current layer K/V into resident KV before
  reading cache-backed attention, so RoPE positions, masks, sliding-window
  counts, KV writes, and decode cache positions honor nonzero cache positions.
- Tests prove two-turn session reuse, CLI `/exit`, append-vs-full-replay
  parity, stale-row isolation, old one-shot behavior, and CI success.

Original scope:

- Add `Gemma2ChatSession` or an equivalent engine-owned session layer.
- Session state includes file-backed GGUF lifetime, `EngineModel`, tokenizer,
  `Gemma2MetalModelPlan`, cached token ids, cached token count, chat messages,
  stop token ids, and generation settings.
- Add CLI flags without breaking old positional one-shot usage:
  `--chat`, `--conversation`, `-cnv`, and `--max-new-tokens N`.
- Add a runtime path that generates against an existing
  `Gemma2MetalModelPlan&` instead of building a fresh plan every turn.
- Render the new chat state through the Gemma chat-template formatter.
- Determine the newly appended token suffix from the previous rendered/tokenized
  prompt and the new rendered/tokenized prompt.
- Prefill appended tokens at `cache_position = cached_token_count` and
  `total_kv_token_count = cached_token_count + appended_token_count`.
- Make RoPE positions, causal masks, Gemma sliding-window attention, KV writes,
  and decode cache positions correct for nonzero cache positions.
- Preserve current Metal-only Gemma backend rejection behavior.

Exit proof:

- An injected chat test proves one session accepts two user turns and exits on
  `/exit`.
- The second turn reuses the same plan/session and appends at a nonzero cache
  position.
- Incremental append matches full replay on a deterministic tiny Gemma fixture
  for generated token ids or a stricter available oracle.
- Tests prove stale KV rows outside the active context cannot affect the next
  token.
- Existing one-shot CLI tests pass unchanged.
- `bash scripts/ci.sh` passes.

## Gate 2: Streaming Output And Turn Stops

Status: Done on 2026-07-10.

Implemented:

- Add a token callback or stream interface from the generate/decode loop.
- Decode and print assistant text incrementally while preserving correct
  SentencePiece leading-space rendering.
- Stop assistant generation on EOS and Gemma `<end_of_turn>` when present.
- Add reverse-stop-string CLI support and apply it during streaming.
- Keep stop/control tokens out of assistant history.
- Handle interrupts so one Ctrl+C stops generation and returns to input; a
  documented second Ctrl+C may exit.

Exit proof:

- Chat tests prove output is emitted through the streaming callback before
  `generate_user_turn` returns.
- Tests cover EOS, `<end_of_turn>`, reverse stop strings, stored-history
  cleanup, and the session cancellation hook used by Ctrl+C.
- CLI tests prove streamed visible tokens and reverse-stop trimming.
- One-shot generation still returns the same final text shape and existing
  error behavior.
- `bash scripts/ci.sh` passes.

## Gate 3: REPL Commands And Input UX

Status: Done on 2026-07-12.

Implemented:

- `/clear` resets history and the logical cached-token cursor while retaining
  the resident Metal model plan; stale KV rows are outside the active prefix.
- `/reset` rebuilds the Metal session plan from the retained model/settings and
  starts with empty history.
- `/regen` rolls back to the recorded cache position before the last user turn,
  then deterministically prefills that suffix again without full replay.
- `/tokens` reports real cached, remaining-context, and configured response
  token counts.
- A trailing `\` continues terminal or piped input on the next line, and
  `/read <path>` submits file contents as one prompt.
- Invalid commands, empty regeneration, and unreadable paths report errors
  without ending the session.
- Deterministic session/frontend tests cover every command and input convention;
  `bash scripts/ci.sh` passes with 288 tests.

Implement:

- `/exit` and `/quit` terminate cleanly.
- `/clear` drops message history and resets cached tokens/KV to an empty chat.
- `/reset` rebuilds the current session state from retained settings and empty
  history without reusing stale KV.
- `/regen` removes the last assistant turn and regenerates from the previous
  user turn with correct sampler/session state.
- `/tokens` reports current context tokens, remaining context budget, and
  configured max-new-tokens.
- Multiline input has an explicit terminal convention.
- `/read <path>` or equivalent transcript input feeds prompt text from disk
  without changing model/session ownership.
- Invalid commands fail politely without ending the session.

End this gate only when:

- Every command has deterministic CLI or frontend tests.
- `/clear`, `/reset`, and `/regen` prove cached token count and KV state are
  consistent after the command.
- Piped input and interactive stdin both have defined behavior.
- `bash scripts/ci.sh` passes.

## Gate 4: Context-Window Management

Status: Done on 2026-07-12.

Implemented:

- Every turn reserves the fully rendered prompt plus configured response budget
  against the model context length before generation.
- Normal overflow removes whole oldest user/assistant pairs until the retained
  structured history and new turn fit; no unsupported pinned prefix is kept.
- Overflow rebuilds only the retained live context from cache position zero.
  The existing Metal plan and uploaded weights stay resident, and the rebuilt
  prefill overwrites the active KV prefix.
- A turn that cannot fit even with empty history is rejected without mutating
  session state, with guidance to shorten the prompt or lower
  `--max-new-tokens`; the REPL remains alive.
- The CLI reports how many oldest turns were dropped and that retained KV was
  rebuilt.
- Deterministic session/CLI tests cover successful overflow, cache position,
  output stability, no-room state preservation, recovery messaging, and REPL
  continuation. `bash scripts/ci.sh` passes with 291 tests.

Implement:

- Track context length, cached token count, incoming user-turn tokens, and
  requested response budget before generation.
- Reject impossible turns with a clear error when no valid trim/shift can fit.
- Add llama.cpp-style context shift or history trimming for normal overflow.
- Preserve a pinned prefix only after the repo has an explicit supported Gemma
  policy for that prefix; do not invent unsupported `system` behavior.
- Prefer true KV shift when correct. If the correct path must rebuild the
  trimmed live context, keep weights resident and make rebuild behavior explicit.

End this gate only when:

- A long injected chat crosses the context threshold and continues according to
  the documented policy.
- Tests cover no-room failure, successful trim/shift, pinned-prefix behavior if
  supported, and cache-position correctness after trim/shift.
- The user-facing message explains what happened and how to recover.
- `bash scripts/ci.sh` passes.

## Gate 5: Sampler State And Regeneration

Status: Done on 2026-07-12.

Implemented:

- Greedy chat remains the default and retains the existing Metal top-1 path
  with one selected-token readback.
- The engine owns persistent seeded sampler state, token history, temperature,
  top-k, top-p, and repetition penalty.
- Sampled mode reads back only final vocabulary logits for engine-owned CPU
  filtering/RNG; model execution, weights, KV, and decode state remain on Metal.
- Chat flags are `--temperature`, `--top-k`, `--top-p`,
  `--repeat-penalty`, and `--seed`. Sampling modifiers require positive
  temperature, and incompatible/invalid combinations fail clearly.
- `/regen` restores the sampler snapshot from before the previous user turn,
  so seeded sampled regeneration exactly reproduces that response. Clear,
  reset, and overflow rebuilds keep sampler history aligned with live context.
- Deterministic sampler, session, and CLI tests cover seeded reproducibility,
  top-k filtering, RNG/history snapshots, sampled regeneration, all exposed
  flags, and failure paths. `bash scripts/ci.sh` passes with 297 tests.

Implement:

- Keep `--greedy` deterministic and backward compatible.
- Add persistent sampler state before exposing sampled chat flags.
- Add seeded RNG support for reproducible sampled chat.
- Add the sampled controls selected for llama.cpp comparison, such as
  temperature, top-k, top-p or min-p, repetition penalty, and seed.
- Make `/regen` interact correctly with sampler state and seed policy.
- Reject unsupported sampler combinations clearly.

End this gate only when:

- Greedy chat is token-id deterministic for the same prompt and settings.
- Seeded sampled chat is reproducible in tests.
- Sampler flags are documented in CLI usage and tested for success/failure
  paths.
- `bash scripts/ci.sh` passes.

## Gate 6: llama.cpp Parity Comparison

Status: Current.

Implement:

- Compare against a current llama.cpp Metal interactive run.
- Match GGUF model, chat template, prompt turns, context length, stop
  conditions, sampler settings, seed, thread/backend settings where applicable,
  and hardware state.
- Capture TTFT, prompt/prefill tokens per second, decode tokens per second,
  total tokens per second, memory notes, command transcript, and output
  differences.
- Keep benchmark artifacts under ignored paths.

End this gate only when:

- The comparison commands are recorded in docs or ignored artifacts.
- Differences are classified as tokenizer/template, sampler, context-policy,
  runtime-performance, or known feature gaps.
- The supported Gemma path can honestly be called a llama.cpp-style interactive
  local CLI application.

## Baton Rule

When implementation starts, copy the first incomplete gate into
`docs/NEXT_TASK.md` as the active slice. Keep this file as the roadmap, not the
working log.
