# Interactive CLI Research

Date: 2026-07-10

Scope: research and implementation direction only. This does not change the
active `docs/NEXT_TASK.md` baton.

## Goal

`local-gguf-generate` is currently a one-shot command:

```text
local-gguf-generate <model.gguf> <prompt> <max-new-tokens> [--greedy] [--backend cpu|metal]
```

The requested behavior is a persistent local conversation mode: load/start the
engine once, keep the model/session alive, then accept multiple user messages
from stdin and generate assistant responses until the user exits.

## Short Answer

Do not make the terminal loop own model state. Add an engine-owned chat/session
layer first, then put a small CLI REPL on top of it.

The first useful slice should keep the GGUF file lifetime, `EngineModel`,
tokenizer, and `Gemma2MetalModelPlan` alive for the whole process. For each
turn, render the full chat history with the existing Gemma chat-template helper,
run greedy generation with the resident plan, append the assistant response to
history, and wait for the next line.

That first slice gives the user the persistent llama.cpp-like experience and
keeps weights resident, but it can still replay the full rendered history each
turn. The faster second slice should add incremental prefill at a nonzero cache
position so new user-turn tokens append to the existing KV cache instead of
replaying the whole conversation.

## Current Repo State

`src/frontends/cli/generate.cpp` is one-shot by shape:

- `run_generate(...)` requires model path, prompt, and max-new-tokens positionals.
- It reads a file-backed GGUF model inside the command invocation.
- `generate_text_from_gguf_model_view(...)` loads `EngineModel`, loads the
  tokenizer, formats one user prompt, calls Gemma greedy generation, decodes the
  newly generated token span, prints one line, and exits.
- `GenerateOptions` only has `backend`; `--greedy` is accepted as a no-op because
  Gemma generation is currently greedy-only.

The engine already has most of the state an interactive mode needs, but it is not
packaged as a reusable session:

- `Gemma2MetalModelPlan` owns uploaded weights, resident KV cache, decode
  scratch, layer state, and byte counts.
- `create_gemma2_metal_generated_token_loop(...)` currently creates a new plan
  through `create_gemma2_metal_generation_runtime_shell(...)` for each call.
- Prefill schedules are fixed to `cache_position = 0` and
  `total_kv_token_count = prompt_token_count`.
- Decode schedules already use `cache_position = cached_token_count` and
  `total_kv_token_count = cached_token_count + 1`.

The tokenizer side is ahead of the CLI:

- `format_chat_prompt_from_gguf_metadata(...)` already supports multi-turn Gemma
  chat prompts.
- It validates Gemma role alternation: user, assistant/model, user, assistant/model.
- It rejects `system` today.
- It renders `<bos><start_of_turn>user...<end_of_turn>` and appends
  `<start_of_turn>model` when generation should begin.

## llama.cpp Findings

llama.cpp has two relevant user-facing patterns.

`llama-completion` is the closest low-level reference. Its README shows
one-shot generation with `-no-cnv`, conversation mode with a chat template, and
single-turn mode for one query. Its source loads model/context once, initializes
chat templates, then keeps `chat_msgs`, token history, sampler state,
`n_past`, and llama memory alive while looping between generation and user
input.

Important details from `llama-completion`:

- Conversation mode is auto-enabled when a supported chat template is available.
- Conversation mode also enables interactive input unless `--single-turn` with a
  predefined prompt disables it.
- `common_chat_format_single(...)` formats only the newly added message by
  comparing the full rendered prompt before and after the new message.
- The main loop preserves token/KV memory with `n_past` and only evaluates new
  tokens when possible.
- End-of-generation or reverse prompts return control to the user in interactive
  mode.
- Context overflow is explicit: it can stop, or shift context by keeping a fixed
  prefix and discarding part of the older middle.
- Prompt caching exists, but upstream documents that restoring a prompt cache is
  not the same as exact deterministic session restore.

Current `llama-cli` is a higher-level UX reference. It is now a client around a
local or remote `llama-server` rather than only a direct token loop. It keeps a
message array client-side, streams `/v1/chat/completions`, and exposes commands
such as `/exit`, `/clear`, `/regen`, `/read`, plus optional media commands.
That command UX is worth borrowing later, but this engine should not copy the
server subprocess architecture for the first local CLI mode.

## Gemma 2 Conversation Requirements

Gemma instruction-tuned models require the chat template for conversational use.
The Gemma model card shows the expected turn structure:

```text
<bos><start_of_turn>user
...<end_of_turn>
<start_of_turn>model
```

For this repo, that means interactive mode must not append raw text like:

```text
User: ...
Assistant: ...
```

It should keep structured messages and call
`format_chat_prompt_from_gguf_metadata(...)` so tokenizer behavior, BOS handling,
special-token encoding, role names, and generation prompts stay consistent with
the existing tested Gemma path.

Stop tokens also need attention. Current CLI passes only
`tokenizer.special_tokens().eos_token_id` to the greedy loop. A chat session
should stop response generation on EOS and on Gemma's turn terminator
`<end_of_turn>` when that token exists in the vocabulary. The response appended
to history should exclude EOS/end-of-turn control tokens.

## Recommended Architecture

Add an engine/session layer between `src/frontends/cli` and the current Gemma
runtime.

Suggested shape:

```text
Gemma2ChatSession
  owns/borrows:
    - file-backed GGUF owner or bytes owner needed by EngineModel tensor views
    - EngineModel
    - Tokenizer
    - Gemma2MetalModelPlan
    - chat history as vector<ChatMessage-owned-strings>
    - rendered/tokenized context ids for the current turn
    - stop token ids: EOS plus end_of_turn if available

  methods:
    - add_user_message(string)
    - generate_response(max_new_tokens) -> string
    - clear()
    - token_count() / context_remaining()
```

The CLI should only own terminal behavior:

- argument parsing
- stdin line reading
- command handling such as `/exit` and `/clear`
- printing prompts/responses/errors
- process-level interrupt handling

Keep `local-gguf-generate` one-shot behavior stable. Add a separate mode rather
than changing positional parsing for existing calls. A clean CLI shape would be:

```text
local-gguf-generate <model.gguf> --chat --max-new-tokens 128 [--backend metal]
```

Aliases can mirror llama.cpp:

```text
--chat
--conversation
-cnv
```

Avoid auto-enabling chat mode at first. llama.cpp can do that because it has a
broad mature CLI; this repo should preserve the existing command contract until
interactive mode is well tested.

## Implementation Phases

### Phase 1: Persistent Chat Process, Resident Plan, Replay History

Goal: user can start the binary once, type multiple messages, get multiple
responses, and exit.

Work:

- Add chat-mode option parsing without changing the current one-shot path.
- Extract a reusable model/session setup path that keeps the file-backed GGUF
  owner, `EngineModel`, tokenizer, and `Gemma2MetalModelPlan` alive.
- Add a runtime overload that can run the existing greedy generation loop against
  an existing `Gemma2MetalModelPlan&`.
- For each turn, render full chat history with `format_chat_prompt_from_gguf_metadata(...)`.
- Run generation from position 0 into the existing plan. This overwrites the
  active prefix of the resident KV cache; older rows beyond
  `total_kv_token_count` are ignored by schedules.
- Stop on EOS or `<end_of_turn>`.
- Append the decoded assistant response to history.
- Support `/exit` and `/clear`.

Tradeoff: this keeps weights and buffers resident, but recomputes the full chat
history each turn. That is acceptable as the first correctness and UX slice.

### Phase 2: Incremental Prefill Append

Goal: avoid full-history replay after each user message.

Work:

- Add an append-prefill schedule that accepts `token_count`, `cache_position`,
  and `total_kv_token_count`.
- Ensure RoPE position start, causal attention window, sliding-window layer
  attention counts, and KV writes all use the nonzero `cache_position`.
- Format only the new chat suffix, like llama.cpp's `common_chat_format_single`
  approach, and verify that the previously rendered prefix still matches.
- Prefill only new user-turn/template tokens, then continue one-token decode
  from the current cached token count.

Tradeoff: this is the real performance path, but it touches schedule semantics
and KV correctness. It should be a separate slice.

### Phase 3: Context Policy

Goal: make long conversations predictable.

Start simple:

- If rendered tokens plus requested generation exceed context length, return a
  clear error telling the user to `/clear` or use fewer max-new-tokens.

Later:

- Add a context trimming strategy.
- Keep the initial instruction only if this repo adds a supported system prompt
  policy for Gemma.
- Consider llama.cpp-style `--keep` only after incremental prefill is stable.

### Phase 4: Streaming And Interrupts

Goal: more llama.cpp-like responsiveness.

Current generation returns the final text after the greedy loop. A later slice
can expose a token callback from the decode loop so the CLI prints text as tokens
arrive. Ctrl+C should interrupt generation and return to input; a second Ctrl+C
can exit.

Do this after Phase 1 works, because streaming changes the runtime/API contract
and complicates tests.

## Minimal CLI UX

First version:

```text
$ local-gguf-generate models/gemma-2-9b-Q4.gguf --chat --max-new-tokens 128
local-gguf-engine chat
model: gemma2
commands: /exit, /clear

> Hello
...
> Can you explain that shorter?
...
> /exit
```

Do not add many commands initially. Good later commands:

- `/regen`: remove the last assistant turn and generate again.
- `/tokens`: show current context token count and remaining budget.
- `/save` or transcript output: plain text transcript, not model state.

Avoid for now:

- broad sampling controls
- server mode
- prompt-cache persistence
- system prompt support unless Gemma formatting policy is deliberately extended

## Tests To Add

Phase 1 deterministic tests:

- Existing one-shot CLI tests remain unchanged.
- Chat-mode argument parsing accepts `--chat --max-new-tokens N`.
- Chat mode rejects missing/invalid max-new-tokens.
- Chat mode rejects `--backend cpu` with the existing Metal-only Gemma error.
- Tiny Gemma fixture can process two user inputs from an injected input stream
  and exits on `/exit`.
- `/clear` removes prior message history and returns to an empty conversation.
- Unsupported chat template still fails before runtime.
- Context-full path reports a clear error.
- EOS and `<end_of_turn>` both stop assistant response generation when present.
- Metal-unavailable environment reports the existing backend error prefix.

Runtime tests:

- Existing generated-token-loop behavior remains identical when using the old
  one-shot helper.
- New plan-reuse overload produces the same token ids as the old helper for a
  fixed prompt.
- Reusing a plan for a second full-history replay does not depend on stale KV
  rows beyond the active prompt length.

Phase 2 tests:

- Append-prefill at nonzero cache position matches full replay for a small
  two-turn prompt.
- Sliding-window layers use the same attention token count as full replay.
- RoPE position start matches the appended token position.
- Context boundary rejects when there is no room for the next user turn or
  requested response.

## Open Decisions Before Coding

- Flag shape: prefer `--chat --max-new-tokens N` while keeping
  `<prompt> <max-new-tokens>` unchanged for one-shot.
- System prompt: keep unsupported for first Gemma chat mode, matching the current
  tokenizer formatter.
- Streaming: defer until after the session object works.
- Incremental prefill: defer until after persistent plan plus full replay works.
- Docs baton: update `docs/NEXT_TASK.md` only when this becomes the approved
  implementation slice.

## Sources

- llama.cpp `tools/completion` README:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/README.md
- llama.cpp `tools/completion` source:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/completion/completion.cpp
- llama.cpp `tools/cli` README:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/README.md
- llama.cpp `tools/cli` source:
  https://github.com/ggml-org/llama.cpp/blob/master/tools/cli/cli-context.cpp
- llama.cpp chat formatting helper:
  https://github.com/ggml-org/llama.cpp/blob/master/common/chat.cpp
- Gemma 2 9B IT model card and chat template guidance:
  https://huggingface.co/google/gemma-2-9b-it

