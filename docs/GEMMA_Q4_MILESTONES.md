# Gemma Q4 Milestone Map

Purpose: keep the big route to fast local Gemma 2 9B Q4 on Metal visible
without turning every handoff into a full architecture reread.

Use this as the route map. Use `docs/NEXT_TASK.md` for the active slice and
`docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` for details and evidence.

Status legend: `Done`, `Current`, `Next`, `Planned`, `Blocked`.

| Status | Milestone | Exit Proof |
| --- | --- | --- |
| Done | Q4_K/Q6_K layout support | Gemma GGUF tensor directory gets past type `12`/`14` validation with deterministic layout tests. |
| Current | Gemma 2 config and weight binding | `gemma2` metadata and required tensor names bind beside Llama, with unsupported execution errors where runtime work is unfinished. |
| Planned | Gemma tokenizer and chat template | Local Gemma IT prompt formatting, role validation, BOS/EOS behavior, and generation prompt suffix have tests. |
| Planned | Tiny reference correctness fixtures | Small Q4_K/Q6_K and Gemma semantic fixtures exist for parity, without becoming a full 9B CPU runtime. |
| Planned | File-backed loading | Large tensor payloads can flow from validated file-backed views to backend upload without eager duplicate host copies. |
| Planned | Metal resident model plan | Packed weights, KV cache, scratch, and per-layer state are owned by a reusable Gemma Metal plan. |
| Planned | Packed Q4_K/Q6_K Metal matmul | Representative Gemma projection shapes consume packed weights directly and pass CPU/reference parity. |
| Planned | Separate prefill and decode | Prompt eval and one-token decode use distinct schedules with resident KV and clear phase metrics. |
| Planned | End-to-end Gemma benchmark metrics | `local-gguf-generate --bench` reports TTFT, prompt eval tokens/s, decode tokens/s, and total tokens/s on the real Gemma path. |
| Planned | llama.cpp comparison loop | Same model, prompt, context, sampling mode, and hardware settings are compared against a current llama.cpp Metal baseline. |

Update rule: change milestone status only when a verified slice lands. Keep
implementation details in `docs/NEXT_TASK.md` or the diagnosis doc, not here.

Cleanup rule: isolate or remove old Llama/CPU-first paths only when a Gemma
replacement or correctness fixture makes them unnecessary. Do not turn cleanup
into a broad refactor, but do not let obsolete paths shape the Gemma Metal-first
architecture.
