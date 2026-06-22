# Gemma Q4 Performance Diagnosis

## Scope And Current Goal

This document diagnoses what must change for this repo to become a fast,
engine-owned local GGUF runtime with the concrete first major target:

- Model: `models/gemma-2-9b-Q4.gguf`
- Machine: MacBook Air M4
- Runtime: local, offline, from-scratch C++/Metal inference
- Correctness reference: CPU/reference fixtures, not the product runtime
- Performance target: eventually beat `llama.cpp` on selected local workloads,
  not broad compatibility

This is not a proposal to wrap `llama.cpp`, Ollama, MLX, PyTorch, Core ML, or
another production inference runtime. Those projects are useful references and
benchmark baselines only.

The fastest maintainable path is to add focused Gemma 2 and Q4/K support beside
the current Llama/SmolLM2 path, use small CPU/reference fixtures to prove
behavior, and move the hot Gemma Q4 execution path into model-resident Metal
kernels. A large generic architecture should wait until the common pieces have
been proven by at least two real model families and multiple quantization
formats.

## Current Repo Architecture

The current repo is cleanly layered:

- `src/model_format`: parses GGUF headers, metadata, tensor directories, aligned
  payload views, and tensor payload size checks.
- `src/tensor`: owns tensor descriptors, byte storage, dtype layouts,
  quantization block views, CPU ops, backend-buffer wrappers, and private CPU
  and Metal backends.
- `src/tokenizer`: loads tokenizer metadata from GGUF, builds a narrow
  vocabulary, encodes/decodes tokens, and handles the current SmolLM2 prompt
  template.
- `src/engine`: binds tensors into model weights, runs Llama blocks, owns KV
  cache objects, sampling, and generation loops.
- `src/frontends/cli`: reads files, parses CLI options, calls engine APIs, and
  formats output. It does not own inference math.
- `benchmarks`: measures synthetic tensor, engine, resident-KV, and model-load
  rows. Result files live under ignored `benchmarks/results`.
- `tests`: deterministic unit and integration tests grouped by domain.

The architectural separation is worth keeping. The main issue is not that the
repo is "spaghetti"; it is that the supported execution model is still the
small SmolLM2 F16 proof path:

- `ModelArchitecture` currently has only `llama`.
- `EngineModel` always carries a `LlamaModelConfig`.
- `load_engine_model_from_gguf` parses only `general.architecture = "llama"`
  and then unconditionally reads `llama.*` metadata.
- `bind_llama_model_weights` expects Llama tensor names such as
  `token_embd.weight`, `blk.N.attn_q.weight`, `blk.N.attn_k.weight`,
  `blk.N.ffn_gate.weight`, and `output_norm.weight`.
- Tensor Core supports only GGUF type IDs `0` F32, `1` F16, and `8` Q8_0.
- Q8_0 has block views and dequantization, but no quantized matmul execution
  path is claimed.
- All current Llama projection execution assumes F32 activations and F16 GGUF
  weight matrices.

The backend split has moved in the right direction. Metal code is no longer one
huge file: context, buffers, generic tensor ops, Llama ops, benchmark helpers,
and the kernel source string are separated. The public backend buffer API is
small and opaque, which is good for ownership. However, it is currently an
operation-level buffer API rather than a model execution planner.

## Current Inference Path And Bottlenecks

Current CPU generation is:

1. CLI reads the whole GGUF file into a `std::vector<uint8_t>`.
2. GGUF parser builds metadata and tensor payload views over those bytes.
3. `load_engine_model_from_gguf` copies each supported tensor payload into
   owned `TensorStorage`.
4. Tokenizer metadata is loaded from GGUF.
5. Prompt formatting is SmolLM2/Llama-specific.
6. Llama weights are bound as borrowed references into `EngineModel` tensors.
7. Prompt tokens are embedded on CPU into F32 hidden states.
8. Cached generation uses an append-only F32 KV cache.
9. Sampling and text decoding remain CPU-owned.

Current Metal generation is explicit opt-in through `--backend metal`:

1. Llama weights are copied into backend buffers.
2. A backend KV cache is allocated.
3. The model-block path can keep block outputs backend-resident across layers.
4. Each fused cached block still runs through Llama/F16-specific kernels.
5. Final logits and sampling still return through CPU-owned code.

The biggest current bottlenecks for Gemma 2 9B Q4 are structural:

- Full-file eager reads and eager owned tensor copies are expensive for a 6.4G
  GGUF. A competitive runtime should avoid unnecessary duplicate host copies and
  should prepare model-resident backend buffers directly from validated tensor
  payload views where possible.
- The dtype bridge rejects Q4/K tensor types before execution can begin.
- Projection kernels only consume F16 weights. Gemma Q4 needs direct packed
  quantized matmul kernels. Dequantizing the whole model to F16/F32 would lose
  the memory-bandwidth advantage and likely exceed the performance target.
- The CPU matmul and normalization code is scalar reference code. It is useful
  for correctness but not a competitive 9B CPU path.
- The Metal matmul kernel maps one output element per thread and loops over the
  entire inner dimension. It is simple and parity-friendly, but it does not use
  tiled/shared-memory style data reuse, SIMD-group reductions, packed quant
  unpacking, or matrix multiply hardware patterns.
- The backend API exposes buffers and individual ops, not a compiled model plan
  with owned scratch, persistent weights, per-layer schedules, and separate
  prefill/decode kernels.
- Existing Metal code frequently creates shared buffers and synchronously waits
  at operation boundaries. That is acceptable for parity tests and small
  synthetic rows, but the target path needs fewer command-buffer boundaries,
  reusable scratch, and model-resident execution.
- The current KV cache is F32. That is useful as a correctness reference, but a
  fast 9B path probably needs a GPU-resident cache layout and may later
  need lower-precision KV variants after F32 parity is proven.
- Existing benchmarks are mostly synthetic Llama/F16 and resident-KV rows. They
  do not yet measure Gemma 2 Q4 load, quantized matmul, real prefill/decode
  split, tokenizer behavior, or end-to-end throughput against `llama.cpp`.

The code worth preserving is the narrow layering, explicit unsupported errors,
deterministic CPU tests, backend-unavailable behavior, benchmark harness
selection, and CPU-vs-Metal parity habit. The code that should not be stretched
too far is the Llama-specific config, tensor binding, tokenizer prompt-template
handling, and F16-only projection path.

## Gemma 2 9B Q4 Support Gap

Local model inventory:

- `models/gemma-2-9b-Q4.gguf`: 6.4G
- Other local models include SmolLM2 F16/Q4_K_M/Q8_0 and Qwen2.5
  FP16/Q8_0, but Gemma 2 Q4 is the target for this diagnosis.

Existing repo tools fail on the Gemma file:

```text
$ build/release/gguf-inspect models/gemma-2-9b-Q4.gguf
error: unsupported tensor type during payload conversion: GGUF tensor type 14 is not supported by Tensor Core

$ build/release/local-gguf-generate models/gemma-2-9b-Q4.gguf "Hello" 1 --greedy
error: invalid GGUF: unsupported GGUF tensor type
```

That first failure occurs in GGUF tensor payload sizing/type conversion, before
engine model binding. A read-only metadata/directory probe shows the file is
otherwise structurally readable:

- GGUF version: 3
- Tensor count: 464
- Metadata entries: 33
- `general.architecture`: `gemma2`
- `general.name`: `gemma-2-9b-it`
- `general.file_type`: `15`
- `general.quantization_version`: `2`
- `gemma2.block_count`: 42
- `gemma2.context_length`: 8192
- `gemma2.embedding_length`: 3584
- `gemma2.feed_forward_length`: 14336
- `gemma2.attention.head_count`: 16
- `gemma2.attention.head_count_kv`: 8
- `gemma2.attention.key_length`: 256
- `gemma2.attention.value_length`: 256
- `gemma2.attention.sliding_window`: 4096
- `gemma2.attention.layer_norm_rms_epsilon`: about `1e-6`
- `gemma2.attn_logit_softcapping`: 50
- `gemma2.final_logit_softcapping`: 30
- `tokenizer.ggml.model`: `llama`
- `tokenizer.ggml.tokens`: 256000 strings, first tokens `<pad>`, `<eos>`,
  `<bos>`
- `tokenizer.ggml.bos_token_id`: 2
- `tokenizer.ggml.eos_token_id`: 1
- `tokenizer.ggml.unknown_token_id`: 3
- `tokenizer.chat_template`: present, Gemma-specific, and not supported by the
  current SmolLM2-only prompt formatter
- `quantize.imatrix.*`: present, indicating importance-matrix quantization data
  was used by the converter

Tensor dtype mix in this file:

| GGUF type ID | Count | Observed role |
| --- | ---: | --- |
| 0 | 169 | F32 norm weights |
| 1 | 1 | F16 `token_embd.weight` |
| 12 | 252 | K-quant matrix weights |
| 14 | 42 | K-quant matrix weights |

Per-suffix tensor type pattern:

| Tensor suffix | Type IDs |
| --- | --- |
| `token_embd.weight` | type 1, one tensor, shape `[3584, 256000]` |
| `output_norm.weight` | type 0, one tensor |
| `blk.N.attn_norm.weight` | type 0, 42 tensors |
| `blk.N.post_attention_norm.weight` | type 0, 42 tensors |
| `blk.N.post_ffw_norm.weight` | type 0, 42 tensors |
| `blk.N.ffn_norm.weight` | type 0, 42 tensors |
| `blk.N.attn_q.weight` | type 12, 42 tensors, shape `[3584, 4096]` |
| `blk.N.attn_k.weight` | type 12, 42 tensors, shape `[3584, 2048]` |
| `blk.N.attn_output.weight` | type 12, 42 tensors, shape `[4096, 3584]` |
| `blk.N.ffn_gate.weight` | type 12, 42 tensors, shape `[3584, 14336]` |
| `blk.N.ffn_up.weight` | type 12, 42 tensors, shape `[3584, 14336]` |
| `blk.N.attn_v.weight` | 21 type 12, 21 type 14, shape `[3584, 2048]` |
| `blk.N.ffn_down.weight` | 21 type 12, 21 type 14, shape `[14336, 3584]` |

The first exact blocker is Tensor Core dtype support:

- `model_format::tensor_type_layout` rejects type IDs other than 0, 1, and 8.
- `dtype_from_gguf_tensor_type` rejects type IDs other than 0, 1, and 8.
- `DType` has only `f32`, `f16`, and `q8_0`.
- The current inspector also treats only 0, 1, and 8 as supported.

The second blocker is architecture support:

- `parse_architecture` only accepts `general.architecture = "llama"`.
- `EngineModel` has only `LlamaModelConfig`, and `load_engine_model_from_gguf`
  unconditionally loads `llama.*` metadata after parsing architecture.
- Tokenizer validation also rejects non-`llama` architecture metadata.

The third blocker is execution semantics:

- Gemma 2 has four norm tensors per block in this file:
  `attn_norm`, `post_attention_norm`, `ffn_norm`, and `post_ffw_norm`. The
  current Llama block uses only attention norm and FFN norm.
- Gemma 2 metadata includes sliding-window attention and attention/final logit
  softcapping. Current Llama execution has neither.
- The current projection path requires F16 matrices, so even after type layout
  support, there is no Q4/K or Q6/K matmul path.
- The output weight appears tied to `token_embd.weight` because there is no
  separate `output.weight` tensor, which the current Llama binder can already
  handle for SmolLM2-style tied output.

Bottom line: Gemma 2 9B Q4 is not close to runnable. The parser can read the
directory and metadata, but production loading fails at the first unsupported
K-quant tensor. After that, the engine still needs Gemma 2 metadata parsing,
Gemma 2 block execution semantics, tokenizer/template support, quantized
reference kernels or fixtures for correctness checks, and a model-resident Metal
Q4/K execution path.

## llama.cpp / ggml Metal Architecture Findings

Upstream snapshot used for this phase:
`ggml-org/llama.cpp@75f460ac289e61eb3c2bb63c9487794a1ed514d1`
from 2026-06-20.

Current `llama.cpp`/`ggml` Metal is not a single "Gemma Metal" path. It is a
generic ggml backend that executes a graph containing architecture-specific
ops. The source is split into a small Metal backend directory:

- `ggml-metal.cpp`: backend interface wiring, including
  `graph_compute`.
- `ggml-metal-context.m`: Metal context, command-buffer submission,
  synchronization, graph encoding, capture support, and async tensor copies.
- `ggml-metal-device.*`: device, command queue, capability flags, memory
  properties, buffer allocation, and residency/shared-buffer policy.
- `ggml-metal-ops.cpp`: per-ggml-op dispatch and pipeline selection.
- `ggml-metal.metal`: kernel implementations and template instantiations.
- `ggml-metal-common.*`, `ggml-metal-impl.h`: common wrappers and internal
  types.

Graph scheduling is the first major difference from this repo's current
operation-level Metal calls. `ggml_backend_metal_graph_compute` forwards a
whole ggml graph to `ggml_metal_graph_compute`. That function creates command
buffers for the graph, encodes the first slice on the caller thread, and can
encode the rest through extra threads. Upstream caps this at eight extra
command buffers plus the main one and comments that 1 or 2 extra command
buffers were empirically best on M1 Pro and M2 Ultra Llama workloads. The graph
compute path generally leaves work asynchronous; explicit synchronization waits
on the last command buffer and checks command-buffer status.

For this repo, the useful lesson is not "copy ggml's scheduler." It is that
competitive decode/prefill cannot be a public API that waits after every small
operation. Gemma Q4 needs a model execution plan that owns persistent buffers,
scratch, command encoding, and per-token/per-prefill schedules behind the
backend boundary. Parity tests can still exercise individual ops, but the fast
path should batch a block or phase worth of work into fewer command-buffer
boundaries.

Quantized matmul is direct, not "dequantize the whole model first." In
`ggml_metal_op_mul_mat`, upstream selects between:

- small-batch matrix-vector extension kernels for F32 activations with batches
  around 2 to 8, including K-quant weights for batches 4 to 8;
- matrix-matrix kernels when inputs are untransposed, the device has
  simdgroup matrix support, the inner dimension is at least 64, and the batch
  exceeds the small-batch threshold;
- regular matrix-vector kernels otherwise.

The Metal kernel file registers direct Q4_K and Q6_K matrix-matrix,
matrix-vector, indirect matrix-matrix, and `get_rows` kernels. The templates
take packed quantized block types and dequantization functions, load/dequantize
sub-blocks inside the kernel, use threadgroup memory, and then use either
simdgroup matrix operations or simdgroup multiply-accumulate loops. The
important performance pattern is: packed weights stay packed in model memory;
unpacking is fused into the dot product.

KV cache handling is graph-level rather than a special Metal-only cache API.
The model graph creates/updates K and V tensors through memory/cache context
objects, and Metal sees the resulting tensors as backend buffers. Flash
attention is represented as `ggml_flash_attn_ext(q, k, v, mask, scale, bias,
logit_softcap)`. The Metal op handles masks, optional sinks, KV padding, temp
buffers, softcap flags, and reductions. For sliding-window attention, the graph
chooses SWA-aware masks and K/V contexts before the Metal backend sees the
attention op.

Gemma 2 support in upstream is model-specific above the backend:

- `LLM_ARCH_GEMMA2` maps to `llama_model_gemma2`.
- Gemma 2 hparams set standard SWA by default, default `n_swa = 4096`, read
  `attention.sliding_window`, read attention/final logit softcapping, and set
  SWA layer patterns.
- Gemma 2 tensor loading includes `attn_norm`, `post_attention_norm`,
  `ffn_norm`, and `post_ffw_norm` per block, plus tied output through
  duplicated token embedding.
- The Gemma 2 graph uses SWA-aware KV attention input, applies attention
  post-norm before the residual add, applies FFN post-norm before the second
  residual add, and applies final logit softcapping after the output
  projection.

The backend lesson for this repo is clear: Gemma-specific semantics belong in
an engine-owned Gemma 2 model/layer path. Metal should accelerate generic
tensor/backend operations and specialized fused hot paths, but it should not
become the owner of Gemma architecture semantics.

## GGUF Quantization And Q4/K Format Findings

The observed local Gemma file tensor IDs map to current ggml tensor types as:

| GGUF tensor type ID | ggml type | Meaning for this file |
| --- | --- | --- |
| 0 | `GGML_TYPE_F32` | norm weights |
| 1 | `GGML_TYPE_F16` | token embedding |
| 12 | `GGML_TYPE_Q4_K` | most projection/FFN matrices |
| 14 | `GGML_TYPE_Q6_K` | higher-precision K-quant matrices |

`general.file_type = 15` corresponds to the llama.cpp policy label
`LLAMA_FTYPE_MOSTLY_Q4_K_M`. Treat that as a model-file quantization policy,
not as the dtype of every tensor. The tensor directory is authoritative. The
local Gemma file proves this: most matrices are Q4_K, but some `attn_v` and
`ffn_down` tensors are Q6_K. A correct loader must preserve per-tensor dtype.

Relevant block layouts from current ggml:

| Type | Block size | Packed layout summary | Effective bits/weight |
| --- | ---: | --- | ---: |
| `Q4_0` | 32 values | one F16 scale plus 4-bit nibbles | 4.5 |
| `Q4_1` | 32 values | F16 scale and min plus 4-bit nibbles | 5.0 |
| `Q4_K` | 256 values | 8 groups of 32; F16 super-block scale/min-scale, 12 bytes of quantized scales/mins, and 4-bit quants | 4.5 |
| `Q6_K` | 256 values | 16 groups of 16; lower 4 bits, upper 2 bits, int8 scales, and F16 super-block scale | 6.5625 |

The K-quant formats are not just smaller scalar dtypes. Q4_K and Q6_K are
structured super-block formats with sub-block scales, bit planes, and layout
constraints. Supporting them correctly means adding:

- exact payload sizing and row-size validation for 256-value K blocks;
- byte-level block views for Q4_K and Q6_K;
- CPU reference dequantization and row matmul tests;
- direct quantized matmul kernels for the fast path;
- per-tensor dtype dispatch in model binding and execution.

Do not design Gemma Q4 around whole-model dequantization to F16/F32. That
would discard the memory-bandwidth advantage, inflate the resident model, and
avoid the actual work needed to beat a direct packed-weight runtime. A CPU
reference can dequantize rows or blocks for correctness, but production Metal
should consume packed Q4_K/Q6_K weights directly.

The first parser-side implementation should be deliberately boring: teach
Tensor Core the layouts for type IDs 12 and 14, reject malformed row sizes, and
expose safe block views. The first execution-side implementation should be a
CPU reference Q4_K/Q6_K matrix-vector path that is obviously correct, even if
slow. Only after that should the Metal path specialize Gemma's hot projections.

## Apple Metal Performance Findings

Apple's current Metal guidance lines up with the bottlenecks already visible in
this repo.

Threadgroup sizing should be derived from the compute pipeline, not guessed
once globally. Apple's threadgroup-size guidance centers on
`threadExecutionWidth` and `maxTotalThreadsPerThreadgroup`; the maximum
threads per threadgroup can differ between pipeline states because register and
threadgroup-memory use differ by kernel. For nonuniform dispatches, Metal can
cover grids whose dimensions are not multiples of the threadgroup size. For
manual `dispatchThreadgroups`, the grid must be rounded up and kernels must
guard out-of-bounds work.

For this repo, that means Q4_K/Q6_K kernels should choose threadgroup geometry
per kernel family: decode matvec, small-batch prefill, larger prefill GEMM,
attention, norm, and logits do not have the same occupancy, memory, or
reduction behavior. A single "one thread per output element" matmul shape is
useful for parity, but it is not a performance design.

Command-buffer overhead matters for per-token decode. Apple's command-buffer
best practice is to submit the fewest command buffers possible without starving
the GPU. More frequent submission can keep the GPU busy when the CPU is behind,
but it can also create CPU stalls through CPU-GPU synchronization. For LLM
decode, each token has many small dependent kernels, so synchronous per-op
submission is especially expensive. The target design should encode larger
work chunks and read back only at phase boundaries where CPU sampling truly
needs data.

Persistent objects matter. Apple explicitly recommends creating device,
command queue, pipeline states, and resource storage early, then reusing them.
Pipeline creation and resource allocation should not happen inside the compute
loop. For this repo, model load should build a reusable backend model context:
persistent weight buffers, KV buffers, scratch buffers, pipeline handles, and
prefill/decode schedules.

Storage mode guidance favors GPU-private resources for large, static,
GPU-only data, populated through a one-time copy from CPU-visible memory. Small
frequently changed CPU/GPU data can use shared storage. On Apple Silicon, the
physical memory is unified, but the API-level guidance still matters because
storage modes affect access permissions, placement, and driver optimizations.
For Gemma 2 9B Q4, the default performance posture should be:

- model weights: persistent model-resident backend buffers;
- KV cache: backend-resident, with CPU reference parity first;
- per-token inputs and sampled logits: small CPU-visible transfer surfaces;
- scratch: reusable backend-owned buffers sized during planning.

The MSL kernel design implication is memory-first. A Gemma Q4 decode step is
dominated by reading large packed weight matrices and KV data. Competitive
kernels should avoid expanding packed weights, keep reads coalesced, reuse
activation values across rows, use simdgroup reductions/matrix operations where
they fit, and keep intermediate data in registers or threadgroup memory only
when the access pattern pays for it. Threadgroup memory is not automatically a
win; bad staging can increase synchronization and scattered access cost.

The practical benchmark posture should be strict: measure prefill and decode
separately, include checksum/parity gates, and compare against a current
`llama.cpp` Metal baseline for the same prompt, context, batch size, thread
settings, and model file. Published Apple guidance is directionally useful, but
the MacBook Air M4 is fanless and memory-bandwidth-sensitive, so sustained
local measurements must drive final kernel choices.

## Gemma 2 Architecture And Tokenizer Findings

Gemma 2 is a decoder-only transformer, but the Gemma 2 9B path is not a
minor variant of the current Llama path. The local GGUF metadata and current
primary/reference sources agree on the important 9B dimensions:

- hidden size: `3584`;
- layers: `42`;
- context length: `8192`;
- attention heads: `16`;
- KV heads: `8`;
- head dimension: `256`;
- local sliding window: `4096`;
- attention logit softcap: `50`;
- final logit softcap: `30`;
- tied token/output embeddings.

There is one naming mismatch to keep straight. The Gemma 2 report lists a 9B
"feedforward dim" of `28672`, while this GGUF has
`gemma2.feed_forward_length = 14336` and per-layer tensors
`ffn_gate.weight` and `ffn_up.weight` shaped `[3584, 14336]`. For this file,
`14336` is the per-branch FFN intermediate size; the GeGLU pair is
`2 * 14336 = 28672`.

The block execution order should be implemented as Gemma-specific semantics,
not stretched through the existing Llama block:

1. Look up token embeddings and apply Gemma's input embedding scale
   `sqrt(hidden_size)`.
2. For each layer, save the residual, RMS-normalize with `attn_norm`, run
   self-attention, apply `post_attention_norm`, then add the residual.
3. Save the new residual, RMS-normalize with `ffn_norm`, run the gated FFN,
   apply `post_ffw_norm`, then add the residual.
4. Apply `output_norm`.
5. Project to vocabulary logits using the tied token embedding matrix because
   this local file has no separate `output.weight`.
6. Apply final logit softcapping:
   `logits = tanh(logits / final_logit_softcapping) * final_logit_softcapping`.

Gemma RMSNorm is subtly different at source-model level: Google/HF code uses
learned norm parameters as an offset, effectively multiplying by
`1 + weight` after RMS normalization. GGUF conversion may store the effective
runtime multiplier instead. The local norm tensors are F32 and their sampled
values are not near zero, so the CPU path should treat the GGUF tensor values
as the multiplicative weights used by the file, then verify against a tiny
golden Gemma output before assuming the converter convention is correct.
Use epsilon from `gemma2.attention.layer_norm_rms_epsilon`, which is about
`1e-6`.

Attention details that matter for correctness:

- Gemma 2 alternates local sliding-window attention and global attention every
  other layer. Current Transformers defaults use zero-based layer `0` as
  sliding and layer `1` as full, repeating. The local GGUF has only the
  architecture-wide `gemma2.attention.sliding_window = 4096`, not an explicit
  per-layer table, so the engine should encode the Gemma 2 layer pattern in
  the Gemma 2 model path and prove it with a golden fixture.
- GQA repeats the `8` KV heads across the `16` query heads, so the repeat
  factor is `2`.
- Q projection output width is `16 * 256 = 4096`; K and V projection output
  widths are `8 * 256 = 2048`; attention output projects from `4096` back to
  hidden size `3584`.
- RoPE applies to Q and K using the `256`-wide head dimension and absolute
  positions, including decode positions after cached tokens.
- Attention score scaling is `1 / sqrt(256)`, from Gemma's
  `query_pre_attn_scalar = 256`, not `1 / sqrt(hidden_size / head_count)`.
- Attention logit softcapping is applied to scaled QK scores before masking
  and softmax:
  `scores = tanh(scores / attn_logit_softcapping) * attn_logit_softcapping`.
- Sliding layers must restrict causal attention to the latest `4096` tokens;
  full layers may attend over the full causal context up to `8192`.

The FFN is a GeGLU-style gated MLP:

- `gate = ffn_gate(x)`;
- `up = ffn_up(x)`;
- activation is the approximate GELU/tanh variant used by Gemma;
- `down = ffn_down(activation(gate) * up)`;
- `post_ffw_norm` is applied to the FFN output before adding the residual.

Tokenizer and chat-template details are also Gemma-specific even though the
local GGUF records `tokenizer.ggml.model = "llama"`:

- Gemma 2 uses a SentencePiece tokenizer with preserved whitespace, split
  digits, and byte fallback.
- The local GGUF has `tokenizer.ggml.tokens` length `256000` and
  `token_embd.weight` shape `[3584, 256000]`. The Gemma 2 report references a
  256k-family vocabulary, and some published tables say `256128`; this runtime
  must trust the local GGUF tensor and tokenizer arrays.
- Special token IDs in the local file are: PAD `0`, EOS `1`, BOS `2`, UNK `3`;
  `<mask>` is token `4`; `<start_of_turn>` is token `106`; `<end_of_turn>` is
  token `107`.
- `tokenizer.ggml.add_bos_token = true`,
  `tokenizer.ggml.add_eos_token = false`, and
  `tokenizer.ggml.add_space_prefix = false`.
- Gemma 2 IT chat formatting supports only `user` and `model` roles. System
  turns are rejected; system-like instructions should be folded into the first
  user turn.
- The local `tokenizer.chat_template` prepends BOS, requires alternating
  `user` and `assistant` roles, maps `assistant` to `model`, emits each turn as
  `<start_of_turn>{role}\n{trimmed content}<end_of_turn>\n`, and appends
  `<start_of_turn>model\n` when `add_generation_prompt` is true.

Relevant local GGUF metadata keys for the first Gemma 2 loader are:

| Key | Local value |
| --- | --- |
| `general.architecture` | `gemma2` |
| `general.name` | `gemma-2-9b-it` |
| `general.file_type` | `15` |
| `general.quantization_version` | `2` |
| `gemma2.block_count` | `42` |
| `gemma2.context_length` | `8192` |
| `gemma2.embedding_length` | `3584` |
| `gemma2.feed_forward_length` | `14336` |
| `gemma2.attention.head_count` | `16` |
| `gemma2.attention.head_count_kv` | `8` |
| `gemma2.attention.key_length` | `256` |
| `gemma2.attention.value_length` | `256` |
| `gemma2.attention.sliding_window` | `4096` |
| `gemma2.attention.layer_norm_rms_epsilon` | about `1e-6` |
| `gemma2.attn_logit_softcapping` | `50` |
| `gemma2.final_logit_softcapping` | `30` |

Relevant local tensor names are:

| Tensor | Required semantics |
| --- | --- |
| `token_embd.weight` | input embedding and tied output projection |
| `output_norm.weight` | final RMSNorm weight |
| `blk.N.attn_norm.weight` | pre-attention RMSNorm |
| `blk.N.post_attention_norm.weight` | post-attention RMSNorm before residual add |
| `blk.N.ffn_norm.weight` | pre-FFN RMSNorm |
| `blk.N.post_ffw_norm.weight` | post-FFN RMSNorm before residual add |
| `blk.N.attn_q.weight` | Q projection, output width `4096` |
| `blk.N.attn_k.weight` | K projection, output width `2048` |
| `blk.N.attn_v.weight` | V projection, output width `2048` |
| `blk.N.attn_output.weight` | attention output projection, input width `4096` |
| `blk.N.ffn_gate.weight` | FFN gate projection |
| `blk.N.ffn_up.weight` | FFN up projection |
| `blk.N.ffn_down.weight` | FFN down projection |

What the repo must implement for CPU correctness before Metal performance
work:

- Add a focused `gemma2` architecture/config path beside the existing Llama
  path. Parse the `gemma2.*` keys above and keep the local file's
  `feed_forward_length`, vocab size, and tensor directory authoritative.
- Add Q4_K and Q6_K Tensor Core layout support first, because the local Gemma
  file cannot even load tensor payloads without those type IDs.
- Bind Gemma 2 weights by their local tensor names, including the four norm
  tensors per block and tied output through `token_embd.weight`.
- Implement a slow, deterministic CPU reference path for packed Q4_K/Q6_K
  matvec, Gemma RMSNorm, embedding scale, RoPE over `head_dim = 256`, GQA,
  alternating sliding/full causal masks, attention softcap, GeGLU FFN,
  post-norm residual order, output norm, tied logits, final softcap, and Gemma
  IT prompt formatting.
- Add tokenizer tests that cover the local special IDs, BOS/EOS behavior, role
  rejection/alternation, and the exact generation prompt suffix.
- Only after the CPU path matches golden Gemma fixtures should Metal work begin.
  The first Metal target should accelerate the same packed-weight semantics,
  not introduce a different execution model.

## Target Architecture Proposal

The target should be a focused Gemma 2 Q4 runtime beside the current
Llama/SmolLM2 path. It should not become a broad graph framework yet. The
shared boundaries are GGUF metadata, tensor layout, tokenizer data, backend
buffers, Metal kernels, KV cache, sampling, and benchmark reporting. Gemma 2
architecture semantics should live in a Gemma-owned engine path.

End-to-end shape:

1. The CLI opens the GGUF file and passes a file-backed byte source into
   `model_format`.
2. `model_format` validates the GGUF header, metadata, tensor directory, tensor
   alignment, and per-tensor payload sizes without copying every tensor into
   owned `TensorStorage`.
3. The engine chooses `ModelArchitecture::gemma2`, parses a
   `Gemma2ModelConfig`, and binds a `Gemma2ModelWeights` view over validated
   tensor payloads.
4. The tokenizer layer loads the existing GGUF tokenizer arrays and selects a
   Gemma IT chat formatter from `general.architecture`, special-token metadata,
   and `tokenizer.chat_template`.
5. A Metal Gemma runtime plan uploads packed weights into persistent backend
   buffers, allocates backend-resident KV/scratch/layer-state buffers, creates
   reusable pipeline state, and exposes separate prefill and decode schedules.
6. Generation uses CPU only for prompt construction, token sampling, text
   decoding, small control data, and correctness/reference checks. The hot
   model path keeps weights, KV, scratch, and layer intermediates on Metal.

Loading should avoid the current large duplicate-copy pattern. The target file
owner can be an mmap-backed or otherwise file-backed byte source whose lifetime
outlives model binding and backend upload. Tensor directory entries should
become validated payload views: `offset`, `nbytes`, `shape`, `dtype`, row
layout, and alignment. CPU reference tests may copy tiny rows or blocks, but
production Gemma loading should not eagerly copy all tensors into host
`TensorStorage` before uploading them again to Metal. The Metal path should
copy packed payload views directly into backend model buffers, then release any
temporary staging objects that are not needed for tokenizer or metadata access.

Gemma metadata should sit beside, not inside, the current Llama config. Add a
small `Gemma2ModelConfig` with the local keys already proven in this file:
layer count, context length, hidden size, FFN branch size, Q/K/V head counts
and dimensions, sliding-window length, norm epsilon, attention softcap, final
softcap, tied-output policy, and vocabulary size from the tokenizer/tensor
directory. The binder should require the local Gemma tensor names listed above,
including `post_attention_norm` and `post_ffw_norm`, and should bind
`token_embd.weight` as both embedding and output projection when no
`output.weight` exists. Unsupported architectures should still fail explicitly;
the Llama path should remain boring and unchanged.

Tokenizer support should also be side-by-side. The tokenizer model metadata may
still say `tokenizer.ggml.model = "llama"`, but Gemma prompt formatting should
be selected from `general.architecture = "gemma2"` and the local
`tokenizer.chat_template`. The first Gemma formatter should cover only the IT
template observed in this file: BOS handling, user/model alternation, rejection
of system turns, trimming turn content, `<start_of_turn>`/`<end_of_turn>`, and
the generation prompt suffix. It should not try to become a general Jinja
runtime.

Q4_K and Q6_K should be represented as first-class tensor layouts:

- `DType::q4_k` for GGUF type ID `12`;
- `DType::q6_k` for GGUF type ID `14`;
- block size `256` values for both;
- row-byte validation based on exact packed block sizes, `144` bytes for Q4_K
  and `210` bytes for Q6_K;
- byte-level block views with accessors for scales, mins, bit planes, and
  quants rather than relying on unsafe host struct packing;
- per-tensor dtype dispatch during binding and backend upload.

The model file's `general.file_type` should remain descriptive only. Execution
must trust the tensor directory because this Gemma file mixes Q4_K and Q6_K
matrices inside the same quantization policy.

Packed quantized Metal matmul should enter the backend as an owned kernel
family, not as whole-model dequantization. The first kernels should target the
Gemma projection shapes and consume packed Q4_K/Q6_K buffers directly, with
unpacking fused into dot products and F32 accumulation until parity says a lower
precision accumulator is acceptable. Decode matvec, small-batch prefill, and
larger prefill GEMM should be separate kernel shapes because their reuse,
threadgroup memory, and dispatch geometry are different. Backend op tests can
exercise these kernels directly, but production generation should call them
through a Gemma runtime plan that controls residency, scratch, and command
encoding.

The model-resident Metal plan should own:

- packed weights for all Gemma matrices, plus F16/F32 embedding and norm
  tensors;
- a backend-resident KV cache shaped for Gemma GQA, `head_dim = 256`,
  alternating sliding/full layers, and the configured context limit;
- reusable scratch for normalized hidden states, Q/K/V, attention workspace,
  FFN intermediates, logits, masks or mask parameters, and reductions;
- per-layer state handles so a layer can be encoded without rebuilding buffer
  bindings or reallocating scratch;
- small CPU-visible transfer buffers for input token IDs, position/control
  data, and final logits needed by sampling.

Prefill and decode should be planned separately. Prefill consumes many prompt
tokens, populates KV, may use batch-oriented projection and attention kernels,
and defines time-to-first-token. Decode consumes one new token at a time, uses
cached K/V with sliding/full attention rules, keeps the layer chain resident,
and reads back only the logits needed for sampling. Mixing both through one
operation-level API will hide the wrong costs and make benchmark numbers hard
to trust.

CPU/reference scaffolding still needed:

- metadata and tensor-directory tests for `gemma2` and mixed Q4_K/Q6_K files;
- tokenizer/chat-template tests for the local Gemma IT format;
- byte-level Q4_K/Q6_K layout and dequantization fixtures;
- tiny row matvec fixtures for Q4_K and Q6_K;
- Gemma RMSNorm, embedding scale, RoPE, GQA head repeat, alternating
  sliding/full masks, attention softcap, GeGLU, post-norm residual order, tied
  logits, and final softcap tests;
- small golden comparisons against a reference runtime or captured logits.

That scaffolding should stay tiny and deterministic. It is not a full Gemma 2
9B CPU runtime and should not drive product performance work.

Before claiming performance, the generation path must report comparable
end-to-end metrics for the real command path:

- model path, backend, architecture, quantization mix, prompt token count,
  generated token count, seed/sampling mode, and context length;
- load time and model-resident upload/plan time;
- time to first token;
- prompt eval tokens/s;
- decode tokens/s after prefill;
- total tokens/s including prefill and decode;
- peak/resident memory if cheaply available;
- checksum, logit digest, or other parity marker for the measured run;
- the exact `llama.cpp` baseline command and settings when making a comparison.

## Implementation Roadmap

First implementation slices, in order:

1. Add layout-only Q4_K/Q6_K support in the GGUF/tensor boundary. The repo
   should be able to inspect the Gemma tensor directory, validate payload sizes,
   and expose safe block views without claiming execution.
2. Add `gemma2` metadata parsing and weight-binding skeletons beside Llama.
   Bind all required local tensor names and return explicit unsupported
   execution errors where kernels are not ready.
3. Add Gemma tokenizer/chat-template support for the observed IT template,
   including role validation and generation-prompt behavior.
4. Add tiny CPU/reference fixtures for Q4_K/Q6_K dequantization, row matvec,
   Gemma norms, softcaps, masks, and tied logits. Keep this as correctness
   scaffolding only.
5. Add a file-backed loading path that can hand validated payload views to
   backend upload without eager whole-model tensor copies.
6. Add a Metal Gemma model plan that uploads packed weights and allocates
   resident KV/scratch/layer state, initially with inventory and unavailable
   device tests.
7. Add packed Q4_K/Q6_K Metal matvec parity kernels for representative Gemma
   projection shapes.
8. Add separate Gemma prefill and decode schedules on top of the resident plan.
9. Add end-to-end `local-gguf-generate --bench` metrics for Gemma Q4 and only
   then compare against a current `llama.cpp` Metal baseline.

Do not spend these slices optimizing the old CPU/Llama path unless the change
directly proves one of the Gemma blockers above.

## First Slice Recommendation

Start with layout-only Q4_K/Q6_K support. This removes the first hard blocker:
the local Gemma file currently fails before metadata, binding, tokenizer, or
backend planning can be exercised. The slice should teach the GGUF/tensor
boundary to recognize type IDs `12` and `14`, calculate exact row and payload
sizes, expose safe byte-level block views, and update inspector/tests so the
Gemma tensor directory is visible. It should not add generation, CPU 9B
execution, Metal kernels, or broad architecture refactors.

## Sources

Local repo sources inspected in Phase 1:

- `AGENTS.md`
- `README.md`
- `docs/STATUS.md`
- `docs/DECISIONS.md`
- `docs/GRAPH.md`
- `docs/MODEL_POLICY.md`
- `docs/ARCHITECTURE.md`
- `docs/adr/0001-build-own-engine.md`
- `docs/adr/0002-cpu-first-metal-second.md`
- `docs/adr/0004-performance-first-cpp-metal-toolchain.md`
- `docs/adr/0005-q8-0-first-quantized-block-format.md`
- `docs/adr/0007-target-llama-smollm2-for-first-inference.md`
- `docs/adr/0008-tokenizer-metadata-first.md`
- `docs/adr/0009-f32-append-only-kv-cache.md`
- `docs/adr/0010-specialize-metal-kv-decode.md`
- `docs/adr/0011-opt-in-generate-backend-selection.md`
- `src/model_format/*`
- `include/local_gguf_engine/model_format/*`
- `src/engine/model.cpp`
- `src/engine/llama_model.cpp`
- `src/engine/llama_layer.cpp`
- `src/engine/kv_cache.cpp`
- `src/tokenizer/tokenizer.cpp`
- `src/tensor/*`
- `src/tensor/ops/*`
- `src/tensor/backends/*`
- `src/frontends/cli/generate.cpp`
- `src/frontends/cli/gguf_inspect.cpp`
- `benchmarks/*`
- `tests/*`
- `Makefile`

Local probe commands/results from Phase 2:

- `ls -lh models`
- `make MODE=release gguf-inspect`
- `build/release/gguf-inspect models/gemma-2-9b-Q4.gguf`
- `build/release/local-gguf-generate models/gemma-2-9b-Q4.gguf "Hello" 1 --greedy`
- read-only GGUF metadata/directory probe for `models/gemma-2-9b-Q4.gguf`

Online sources inspected in Phase 3:

- llama.cpp upstream snapshot:
  `ggml-org/llama.cpp@75f460ac289e61eb3c2bb63c9487794a1ed514d1`
  (current master fetched 2026-06-20).
- llama.cpp Metal source layout:
  <https://github.com/ggml-org/llama.cpp/tree/75f460ac289e61eb3c2bb63c9487794a1ed514d1/ggml/src/ggml-metal>
- llama.cpp Metal graph/context scheduling:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/ggml/src/ggml-metal/ggml-metal-context.m>
- llama.cpp Metal op dispatch and matmul/flash-attention selection:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/ggml/src/ggml-metal/ggml-metal-ops.cpp>
- llama.cpp Metal kernels and Q4_K/Q6_K kernel registrations:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/ggml/src/ggml-metal/ggml-metal.metal>
- ggml tensor type IDs and ggml type/file-type enums:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/ggml/include/ggml.h>
- llama.cpp file-type labels including `LLAMA_FTYPE_MOSTLY_Q4_K_M`:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/include/llama.h>
- ggml quantization block structs:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/ggml/src/ggml-common.h>
- ggml quantization/dequantization reference code:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/ggml/src/ggml-quants.c>
- ggml type traits, row sizing, and quantized type metadata:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/ggml/src/ggml.c>
- llama.cpp Gemma 2 model hparams/tensors/graph:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/src/models/gemma2.cpp>
- llama.cpp architecture metadata keys and tensor names:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/src/llama-arch.cpp>
- llama.cpp graph attention, SWA mask, KV context, and softcap path:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/src/llama-graph.cpp>
- llama.cpp SWA and softcap hparams:
  <https://github.com/ggml-org/llama.cpp/blob/75f460ac289e61eb3c2bb63c9487794a1ed514d1/src/llama-hparams.h>
- Apple Metal threadgroup/grid sizing:
  <https://developer.apple.com/documentation/metal/calculating-threadgroup-and-grid-sizes>
- Apple Metal Best Practices, command buffers:
  <https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/CommandBuffers.html>
- Apple Metal Best Practices, persistent objects:
  <https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/PersistentObjects.html>
- Apple Metal Best Practices, resource options/storage modes:
  <https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/ResourceOptions.html>
- Apple Metal Best Practices, pipeline creation:
  <https://developer.apple.com/library/archive/documentation/3DDrawing/Conceptual/MTLBestPracticesGuide/Pipelines.html>
- Metal Shading Language Specification:
  <https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf>

Online and local sources inspected in Phase 4:

- Gemma 2 technical report:
  <https://arxiv.org/abs/2408.00118>
- Google AI for Developers, Gemma 2 model card:
  <https://ai.google.dev/gemma/docs/core/model_card_2>
- Google AI for Developers, Gemma formatting and system instructions:
  <https://ai.google.dev/gemma/docs/core/prompt-structure>
- Google-owned Hugging Face model card for `google/gemma-2-9b-it`:
  <https://huggingface.co/google/gemma-2-9b-it>
- Google DeepMind Gemma library transformer code:
  <https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_transformer.py>
- Google DeepMind Gemma library config and RMSNorm code:
  <https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_config.py>
  <https://github.com/google-deepmind/gemma/blob/main/gemma/gm/nn/_layers.py>
- Google DeepMind Gemma library tokenizer code:
  <https://github.com/google-deepmind/gemma/blob/main/gemma/gm/text/_tokenizer.py>
- Hugging Face Transformers current Gemma 2 config and model code:
  <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/configuration_gemma2.py>
  <https://github.com/huggingface/transformers/blob/main/src/transformers/models/gemma2/modular_gemma2.py>
- GGUF specification:
  <https://github.com/ggml-org/ggml/blob/master/docs/gguf.md>
- Current llama.cpp GGUF/Gemma mapping references:
  <https://github.com/ggml-org/llama.cpp/blob/master/gguf-py/gguf/constants.py>
  <https://github.com/ggml-org/llama.cpp/blob/master/src/llama-arch.cpp>
- Read-only local GGUF header/directory probes for
  `models/gemma-2-9b-Q4.gguf`, covering metadata keys, tokenizer template,
  tensor names, tensor shapes, tensor type IDs, and sampled norm tensor values.
