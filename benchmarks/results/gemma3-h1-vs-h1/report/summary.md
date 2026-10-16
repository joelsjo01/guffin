# Release chat run b4e116d0-2943-4696-8263-2208d131fe02

Single-pass actual-chat observations. Frontend-completed conversations contribute to the rates; known limited, interrupted and incomplete conversations remain listed below. Unknown stop reasons are not assumed to be EOS.

## Paired workload comparisons

Each row compares the same jointly complete case IDs within one model, mode and suite. Coverage is jointly complete / planned for both engines; cases completed by only one engine do not enter these rates or charts.

| Suite | Mode | Configurations (Guffin / llama.cpp) | Paired cases | Guffin t/s | llama.cpp t/s | Rate ratio |
| --- | --- | --- | --- | --- | --- | --- |
| core | speculative | guffin-speculative / llama-speculative | 32 / 32 | 18.53 | 13.26 | 1.40× |
| supplementary | speculative | guffin-speculative / llama-speculative | 6 / 6 | 16.99 | 12.94 | 1.31× |

## guffin-speculative

Engine: `guffin`; mode: `speculative`; model SHA-256: `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5`.

Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.

| Suite | Complete / planned | Generation t/s | Selected tokens | Median internal TTFT ms | Peak footprint GiB |
| --- | --- | --- | --- | --- | --- |
| core | 32 / 32 | 18.53 | 46860 | 592.95 | 9.15 |
| supplementary | 6 / 6 | 16.99 | 5234 | 645.99 | 9.15 |

| Category | Turns | Generation t/s | Prompt t/s | Median internal TTFT ms |
| --- | --- | --- | --- | --- |
| coding | 10 | 19.18 | 93.81 | 602.09 |
| conversation | 2 | 18.41 | 52.13 | 364.49 |
| document | 2 | 14.87 | 96.88 | 24065.72 |
| extraction | 8 | 19.68 | 94.41 | 1200.33 |
| humanities | 8 | 17.60 | 73.59 | 502.51 |
| math | 8 | 21.46 | 64.81 | 595.17 |
| multilingual | 4 | 17.76 | 81.68 | 617.95 |
| planning | 6 | 17.75 | 70.78 | 656.33 |
| reasoning | 8 | 19.67 | 79.68 | 609.28 |
| roleplay | 8 | 17.64 | 73.41 | 601.85 |
| stem | 8 | 18.06 | 69.33 | 595.60 |
| writing | 8 | 18.62 | 68.55 | 499.64 |

| Turn position | Turns | Generation t/s | Median internal TTFT ms |
| --- | --- | --- | --- |
| first | 38 | 18.64 | 614.09 |
| follow-up | 42 | 18.06 | 508.28 |

## llama-speculative

Engine: `llama`; mode: `speculative`; model SHA-256: `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5`.

Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.

| Suite | Complete / planned | Generation t/s | Selected tokens | Median internal TTFT ms | Peak footprint GiB |
| --- | --- | --- | --- | --- | --- |
| core | 32 / 32 | 13.26 | 48112 | 931.52 | 4.47 |
| supplementary | 6 / 6 | 12.94 | 4934 | 1253.03 | 7.05 |

| Category | Turns | Generation t/s | Prompt t/s | Median internal TTFT ms |
| --- | --- | --- | --- | --- |
| coding | 10 | 14.23 | 108.46 | 1124.98 |
| conversation | 2 | 14.10 | 57.53 | 799.54 |
| document | 2 | 12.49 | 118.24 | 19723.27 |
| extraction | 8 | 15.23 | 100.20 | 1544.45 |
| humanities | 8 | 12.43 | 57.34 | 835.70 |
| math | 8 | 15.35 | 108.83 | 2008.68 |
| multilingual | 4 | 12.72 | 97.08 | 1756.58 |
| planning | 6 | 12.85 | 87.97 | 2069.59 |
| reasoning | 8 | 14.43 | 85.85 | 1151.79 |
| roleplay | 8 | 12.66 | 96.80 | 976.53 |
| stem | 8 | 12.99 | 87.35 | 826.92 |
| writing | 8 | 12.51 | 99.17 | 1116.86 |

| Turn position | Turns | Generation t/s | Median internal TTFT ms |
| --- | --- | --- | --- |
| first | 38 | 13.45 | 813.16 |
| follow-up | 42 | 12.99 | 1323.21 |

## Cases

| Case | Suite | Category | Configuration | Status | Stop reasons | Turns | Generation t/s | Selected tokens | Wall reply seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guffin-document-review | supplementary | document | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 14.87 | 1136 | 124.45 |
| mt-bench-141 | core | stem | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.78 | 2423 | 136.99 |
| mt-bench-143 | core | stem | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.28 | 2318 | 128.02 |
| guffin-code-review | supplementary | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.51 | 1240 | 92.43 |
| mt-bench-96 | core | roleplay | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.79 | 1589 | 131.43 |
| mt-bench-106 | core | reasoning | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.94 | 393 | 21.14 |
| mt-bench-151 | core | humanities | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.74 | 2214 | 175.14 |
| mt-bench-105 | core | reasoning | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 20.11 | 738 | 39.48 |
| guffin-code-review | supplementary | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 13.92 | 993 | 89.76 |
| mt-bench-100 | core | roleplay | llama-speculative | complete | unknown: 2 | 2 / 2 | 11.98 | 2088 | 175.74 |
| mt-bench-106 | core | reasoning | llama-speculative | complete | unknown: 2 | 2 / 2 | 13.86 | 472 | 36.97 |
| mt-bench-116 | core | math | llama-speculative | complete | unknown: 2 | 2 / 2 | 15.01 | 921 | 66.55 |
| mt-bench-87 | core | writing | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.74 | 1150 | 98.56 |
| mt-bench-101 | core | reasoning | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.60 | 167 | 15.19 |
| mt-bench-114 | core | math | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 21.45 | 1692 | 80.05 |
| mt-bench-90 | core | writing | llama-speculative | complete | unknown: 2 | 2 / 2 | 14.25 | 604 | 47.59 |
| guffin-swedish-chat | supplementary | multilingual | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.47 | 478 | 28.59 |
| mt-bench-157 | core | humanities | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.54 | 2712 | 157.41 |
| guffin-chinese-chat | supplementary | multilingual | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.04 | 518 | 29.91 |
| mt-bench-131 | core | extraction | llama-speculative | complete | unknown: 2 | 2 / 2 | 15.63 | 114 | 9.77 |
| mt-bench-154 | core | humanities | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.44 | 2849 | 230.70 |
| mt-bench-133 | core | extraction | llama-speculative | complete | unknown: 2 | 2 / 2 | 15.65 | 158 | 14.53 |
| mt-bench-154 | core | humanities | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.49 | 2927 | 168.41 |
| mt-bench-149 | core | stem | llama-speculative | complete | unknown: 2 | 2 / 2 | 13.01 | 2377 | 184.24 |
| mt-bench-145 | core | stem | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.81 | 1820 | 150.87 |
| mt-bench-134 | core | extraction | llama-speculative | complete | unknown: 2 | 2 / 2 | 14.62 | 250 | 20.08 |
| mt-bench-114 | core | math | llama-speculative | complete | unknown: 2 | 2 / 2 | 15.53 | 1838 | 126.14 |
| mt-bench-131 | core | extraction | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 20.47 | 114 | 8.02 |
| guffin-document-review | supplementary | document | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.49 | 1143 | 130.91 |
| mt-bench-140 | core | extraction | llama-speculative | complete | unknown: 2 | 2 / 2 | 15.54 | 176 | 15.55 |
| mt-bench-134 | core | extraction | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.60 | 251 | 15.57 |
| mt-bench-151 | core | humanities | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.95 | 2177 | 122.25 |
| mt-bench-90 | core | writing | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 20.29 | 1041 | 52.96 |
| mt-bench-97 | core | roleplay | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.10 | 2258 | 126.00 |
| mt-bench-87 | core | writing | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.59 | 1200 | 69.19 |
| mt-bench-81 | core | writing | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.00 | 2934 | 245.86 |
| mt-bench-143 | core | stem | llama-speculative | complete | unknown: 2 | 2 / 2 | 13.14 | 2431 | 186.75 |
| mt-bench-95 | core | roleplay | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.51 | 368 | 22.59 |
| mt-bench-82 | core | writing | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.71 | 505 | 28.01 |
| mt-bench-111 | core | math | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 21.34 | 1111 | 53.11 |
| guffin-swedish-chat | supplementary | multilingual | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.55 | 459 | 40.15 |
| mt-bench-149 | core | stem | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.22 | 2352 | 130.17 |
| mt-bench-100 | core | roleplay | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.18 | 2065 | 121.44 |
| mt-bench-95 | core | roleplay | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.80 | 461 | 40.15 |
| mt-bench-82 | core | writing | llama-speculative | complete | unknown: 2 | 2 / 2 | 13.18 | 597 | 47.27 |
| guffin-short-chat | supplementary | conversation | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.41 | 52 | 3.53 |
| mt-bench-155 | core | humanities | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.53 | 3055 | 175.14 |
| mt-bench-124 | core | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 14.34 | 1258 | 97.54 |
| mt-bench-97 | core | roleplay | llama-speculative | complete | unknown: 2 | 2 / 2 | 13.29 | 2081 | 158.46 |
| mt-bench-155 | core | humanities | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.40 | 3113 | 252.37 |
| mt-bench-123 | core | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.56 | 2144 | 110.65 |
| guffin-short-chat | supplementary | conversation | llama-speculative | complete | unknown: 2 | 2 / 2 | 14.10 | 52 | 5.16 |
| mt-bench-111 | core | math | llama-speculative | complete | unknown: 2 | 2 / 2 | 15.38 | 1318 | 93.99 |
| mt-bench-96 | core | roleplay | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.65 | 1718 | 98.81 |
| mt-bench-140 | core | extraction | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.69 | 175 | 12.80 |
| mt-bench-121 | core | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 14.26 | 3303 | 232.94 |
| mt-bench-101 | core | reasoning | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.10 | 160 | 9.57 |
| mt-bench-116 | core | math | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 21.71 | 662 | 31.30 |
| mt-bench-124 | core | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.95 | 1270 | 71.41 |
| mt-bench-126 | core | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 14.18 | 2229 | 158.78 |
| mt-bench-121 | core | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.75 | 3185 | 162.21 |
| mt-bench-81 | core | writing | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.50 | 2815 | 153.09 |
| mt-bench-119 | core | math | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 21.46 | 610 | 30.03 |
| mt-bench-126 | core | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.18 | 2195 | 115.49 |
| guffin-sustained-design | supplementary | planning | guffin-speculative | complete | stop_token: 6 | 6 / 6 | 17.75 | 1810 | 105.98 |
| mt-bench-141 | core | stem | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.95 | 2421 | 188.24 |
| mt-bench-133 | core | extraction | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.24 | 156 | 12.07 |
| mt-bench-109 | core | reasoning | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.77 | 350 | 19.75 |
| mt-bench-123 | core | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 14.31 | 2139 | 157.30 |
| mt-bench-157 | core | humanities | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.21 | 2782 | 230.73 |
| guffin-chinese-chat | supplementary | multilingual | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.84 | 635 | 52.82 |
| mt-bench-105 | core | reasoning | llama-speculative | complete | unknown: 2 | 2 / 2 | 15.05 | 1330 | 91.16 |
| mt-bench-109 | core | reasoning | llama-speculative | complete | unknown: 2 | 2 / 2 | 13.81 | 269 | 21.90 |
| mt-bench-145 | core | stem | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.97 | 1720 | 96.65 |
| mt-bench-119 | core | math | llama-speculative | complete | unknown: 2 | 2 / 2 | 15.26 | 659 | 47.10 |
| guffin-sustained-design | supplementary | planning | llama-speculative | complete | unknown: 6 | 6 / 6 | 12.85 | 1652 | 140.76 |

## Per-case generation rate ratios

Guffin / llama.cpp; above 1 means higher measured generation rate. Modes and source-model hashes match, but natural outputs can differ.

| Case | Mode | Configurations | Rate ratio |
| --- | --- | --- | --- |
| mt-bench-100 | speculative | guffin-speculative / llama-speculative | 1.43× |
| mt-bench-101 | speculative | guffin-speculative / llama-speculative | 1.52× |
| mt-bench-105 | speculative | guffin-speculative / llama-speculative | 1.34× |
| mt-bench-106 | speculative | guffin-speculative / llama-speculative | 1.44× |
| mt-bench-109 | speculative | guffin-speculative / llama-speculative | 1.36× |
| mt-bench-111 | speculative | guffin-speculative / llama-speculative | 1.39× |
| mt-bench-114 | speculative | guffin-speculative / llama-speculative | 1.38× |
| mt-bench-116 | speculative | guffin-speculative / llama-speculative | 1.45× |
| mt-bench-119 | speculative | guffin-speculative / llama-speculative | 1.41× |
| mt-bench-121 | speculative | guffin-speculative / llama-speculative | 1.38× |
| mt-bench-123 | speculative | guffin-speculative / llama-speculative | 1.37× |
| mt-bench-124 | speculative | guffin-speculative / llama-speculative | 1.32× |
| mt-bench-126 | speculative | guffin-speculative / llama-speculative | 1.35× |
| mt-bench-131 | speculative | guffin-speculative / llama-speculative | 1.31× |
| mt-bench-133 | speculative | guffin-speculative / llama-speculative | 1.23× |
| mt-bench-134 | speculative | guffin-speculative / llama-speculative | 1.34× |
| mt-bench-140 | speculative | guffin-speculative / llama-speculative | 1.27× |
| mt-bench-141 | speculative | guffin-speculative / llama-speculative | 1.37× |
| mt-bench-143 | speculative | guffin-speculative / llama-speculative | 1.39× |
| mt-bench-145 | speculative | guffin-speculative / llama-speculative | 1.40× |
| mt-bench-149 | speculative | guffin-speculative / llama-speculative | 1.40× |
| mt-bench-151 | speculative | guffin-speculative / llama-speculative | 1.41× |
| mt-bench-154 | speculative | guffin-speculative / llama-speculative | 1.41× |
| mt-bench-155 | speculative | guffin-speculative / llama-speculative | 1.41× |
| mt-bench-157 | speculative | guffin-speculative / llama-speculative | 1.44× |
| mt-bench-81 | speculative | guffin-speculative / llama-speculative | 1.54× |
| mt-bench-82 | speculative | guffin-speculative / llama-speculative | 1.42× |
| mt-bench-87 | speculative | guffin-speculative / llama-speculative | 1.38× |
| mt-bench-90 | speculative | guffin-speculative / llama-speculative | 1.42× |
| mt-bench-95 | speculative | guffin-speculative / llama-speculative | 1.37× |
| mt-bench-96 | speculative | guffin-speculative / llama-speculative | 1.38× |
| mt-bench-97 | speculative | guffin-speculative / llama-speculative | 1.36× |
| guffin-chinese-chat | speculative | guffin-speculative / llama-speculative | 1.40× |
| guffin-code-review | speculative | guffin-speculative / llama-speculative | 1.26× |
| guffin-document-review | speculative | guffin-speculative / llama-speculative | 1.19× |
| guffin-short-chat | speculative | guffin-speculative / llama-speculative | 1.31× |
| guffin-sustained-design | speculative | guffin-speculative / llama-speculative | 1.38× |
| guffin-swedish-chat | speculative | guffin-speculative / llama-speculative | 1.39× |

## Incomplete or unattempted conversations

None.

## Measurement boundaries

- One pass per planned conversation; no repetitions, confidence intervals or stability claim.
- Natural answers and follow-up histories can differ. These are workload observations, not identical-work or quality-equivalence results.
- Generation rate uses summed generation_steps / summed generation_ns. Selected tokens are reported separately; proposed and rejected draft tokens are excluded.
- Internal TTFT and external wall-turn latency have separate clocks. Startup, cooldowns and time waiting for the next prompt are excluded from generation rates.
- Physical footprint uses the macOS native lifetime maximum at the last successful observation. RSS is the maximum sampled resident size. Unavailable measurements and collection errors remain explicit.
- Frontend completion does not establish EOS. Unknown stop reasons stay unknown; known limited or interrupted replies are excluded from complete-conversation aggregates.
- Headline comparisons and charts use only jointly complete cases planned for both engines with matching model hash, mode and suite. Independent configuration aggregates retain their own completed workloads and coverage.
- The core MT-Bench workload and project supplements have separate suite summaries; their combined rates are not a core-workload headline.
- Gemma 2 comparisons must account for Guffin's Q6-down to Q4 and F16-head to Q8 numerical policies; the same GGUF hash does not establish identical executed weights.

Charts: [generation](generation.svg), [internal TTFT](ttft.svg), [memory](memory.svg). Raw turn rows: [CSV](summary.csv); complete summary: [JSON](summary.json).
