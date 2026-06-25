# Gemma Q4 Milestone Map

Purpose: keep the big route to fast local Gemma 2 9B Q4 on Metal visible
without turning every handoff into a full architecture reread.

Use this as the route map. Use `docs/NEXT_TASK.md` for the active slice and
`docs/GEMMA_Q4_PERFORMANCE_DIAGNOSIS.md` for details and evidence.

Status legend: `Done`, `Current`, `Next`, `Planned`, `Blocked`.

| Status | Milestone | Exit Proof |
| --- | --- | --- |
| Done | Q4_K/Q6_K layout support | Gemma GGUF tensor directory gets past type `12`/`14` validation with deterministic layout tests. |
| Done | Gemma 2 config and weight binding | `gemma2` metadata and required tensor names bind beside Llama, with unsupported execution errors where runtime work is unfinished. |
| Done | Gemma tokenizer and chat template | Local Gemma IT prompt formatting, role validation, BOS/EOS behavior, and generation prompt suffix have tests. |
| Done | Tiny reference correctness fixtures | Small Q4_K/Q6_K and Gemma semantic fixtures exist for parity, without becoming a full 9B CPU runtime. |
| Done | File-backed loading | Large tensor payloads can flow from validated file-backed views to backend upload without eager duplicate host copies. |
| Done | Metal resident model plan | Packed weights, KV cache, scratch, and per-layer state are owned by a reusable Gemma Metal plan. |
| Done | Packed Q4_K/Q6_K Metal matmul | Representative Gemma projection shapes consume packed weights directly and pass CPU/reference parity. |
| Done | Separate prefill and decode | Prompt prefill and one-token decode have distinct resident-plan schedules with explicit KV, scratch, and layer phase shapes. |
| Done | Gemma generation runtime shell | The real Gemma generate path builds the resident Metal plan and phase schedules, then stops at an explicit unsupported execution boundary. |
| Done | First scheduled Gemma execution primitive | Prompt token IDs become backend-resident Gemma hidden states through the runtime shell, with CPU/Metal parity and validation tests. |
| Done | First scheduled Gemma layer input primitive | First-layer attention RMSNorm consumes resident prefill hidden states and produces explicit scheduled layer input, without claiming full generation. |
| Done | First scheduled Gemma attention projection setup | First-layer normalized attention input produces explicit Q/K/V projection outputs, without attention softmax, KV update, output projection, or logits. |
| Done | First scheduled Gemma attention position setup | First-layer Q/K projection outputs are prepared for Gemma positions/RoPE as explicit scheduled outputs, without KV update, attention softmax, output projection, or logits. |
| Done | First scheduled Gemma attention score setup | First-layer positioned Q/K outputs produce an explicit pre-softmax score workspace, without KV update, value aggregation, output projection, or logits. |
| Done | First scheduled Gemma attention probability setup | First-layer score outputs produce explicit softmax probabilities, without KV update, value aggregation, output projection, or logits. |
| Done | First scheduled Gemma attention value aggregation setup | First-layer softmax probabilities and V projection outputs produce an explicit attention value output, without KV update, output projection, residual work, FFN, or logits. |
| Done | First scheduled Gemma attention output projection setup | First-layer attention value output projects back to hidden width, without residual work, FFN, logits, or generation success. |
| Done | First scheduled Gemma post-attention residual setup | First-layer projected attention output applies Gemma post-attention norm and residual add, without FFN, logits, or generation success. |
| Done | First scheduled Gemma FFN gate/up setup | First-layer post-attention residual is normalized and projected into FFN gate/up buffers, without activation, down projection, logits, or generation success. |
| Done | First scheduled Gemma FFN output setup | First-layer FFN gate/up outputs apply Gemma activation, down projection, post-FFN norm, and residual add, without logits or generation success. |
| Done | First scheduled Gemma layer handoff setup | First-layer FFN residual becomes the next layer's normalized attention input, without running layer 1 attention, logits, or generation success. |
| Done | First scheduled Gemma layer-1 attention projection setup | Layer-1 normalized attention input produces explicit Q/K/V projection outputs, without position setup, attention, logits, or generation success. |
| Done | First scheduled Gemma layer-1 attention position setup | Layer-1 Q/K projection outputs are prepared for Gemma positions/RoPE, without attention scores, logits, or generation success. |
| Done | First scheduled Gemma layer-1 attention score setup | Layer-1 positioned Q/K outputs produce an explicit pre-softmax score workspace, without probabilities, value aggregation, logits, or generation success. |
| Done | First scheduled Gemma layer-1 attention probability setup | Layer-1 score outputs produce explicit softmax probabilities, without value aggregation, logits, or generation success. |
| Done | First scheduled Gemma layer-1 attention value aggregation setup | Layer-1 softmax probabilities and V projection outputs produce an explicit attention value output, without output projection, logits, or generation success. |
| Done | First scheduled Gemma layer-1 attention output projection setup | Layer-1 attention value output projects back to hidden width, without residual work, logits, or generation success. |
| Done | First scheduled Gemma layer-1 post-attention residual setup | Layer-1 projected attention output applies Gemma post-attention norm and residual add, without FFN, logits, or generation success. |
| Done | First scheduled Gemma layer-1 FFN gate/up setup | Layer-1 post-attention residual is normalized and projected into FFN gate/up buffers, without activation, down projection, logits, or generation success. |
| Done | First scheduled Gemma layer-1 FFN output setup | Layer-1 FFN gate/up outputs apply Gemma activation, down projection, post-FFN norm, and residual add, without logits or generation success. |
| Done | First scheduled Gemma layer-2 handoff setup | Layer-1 FFN residual becomes the next layer's normalized attention input, without running layer 2 attention, logits, or generation success. |
| Current | First scheduled Gemma layer-2 attention projection setup | Layer-2 normalized attention input produces explicit Q/K/V projection outputs, without position setup, attention scores, logits, or generation success. |
| Planned | End-to-end Gemma benchmark metrics | `local-gguf-generate --bench` reports TTFT, prompt eval tokens/s, decode tokens/s, and total tokens/s on the real Gemma path. |
| Planned | llama.cpp comparison loop | Same model, prompt, context, sampling mode, and hardware settings are compared against a current llama.cpp Metal baseline. |

Update rule: change milestone status only when a verified slice lands. Keep
implementation details in `docs/NEXT_TASK.md` or the diagnosis doc, not here.

Cleanup rule: isolate or remove old Llama/CPU-first paths only when a Gemma
replacement or correctness fixture makes them unnecessary. Do not turn cleanup
into a broad refactor, but do not let obsolete paths shape the Gemma Metal-first
architecture.
