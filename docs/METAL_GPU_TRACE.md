# Production Decode GPU Trace

Build the release executable, then capture one warmed production decode command
buffer to a new path:

```sh
make MODE=release generate
scripts/capture_decode_gpu_trace.py \
  --model models/gemma-2-9b-Q4.gguf \
  --output /tmp/gemma2-decode.gputrace
```

The script enables Metal capture, starts the ordinary chat runtime (including
its normal startup warm-up), and arms a one-shot capture for the next command
buffer labeled `Gemma2 resident decode`. Prefill is not captured. Capture starts
before that decode command buffer is created and stops after it completes; the
rest of generation is ordinary uncaptured execution. Existing output files are
never overwritten.

Allow substantial free disk space. The Stage 0 target-model capture was about
8.5 GB because a trace document snapshots the resident resources referenced by
the command buffer even though only one command buffer is captured. Remove the
artifact after inspection when it is no longer needed.

Open the resulting document with the full Xcode GPU profiler and inspect the
debug groups for embedding, per-layer attention projections, attention/KV,
attention output, FFN gate/up/GELU, FFN down, final norm/output, and selection.
Profiler occupancy, limiter, bandwidth, cache, ALU, and register statistics may
use replayed counter passes and must not be reported as canonical timing.

The Apple M4 probed for Stage 0 exposes only the runtime `timestamp` /
`GPUTimestamp` counter set. Stage-boundary sampling is supported, while draw,
dispatch, tile, and blit boundary sampling are not. Detailed limiter statistics
therefore come from Xcode's trace profiler, not runtime counter buffers. If only
Command Line Tools are installed, capture creation can still be verified but
the trace cannot be inspected locally until full Xcode is installed.
