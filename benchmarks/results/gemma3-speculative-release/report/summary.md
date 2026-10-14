# Release chat run 4901ee27-74cf-46cc-ae27-be827e033b3b

Single-pass actual-chat observations. Frontend-completed conversations contribute to the rates; known limited, interrupted and incomplete conversations remain listed below. Unknown stop reasons are not assumed to be EOS.

## Paired workload comparisons

Each row compares the same jointly complete case IDs within one model, mode and suite. Coverage is jointly complete / planned for both engines; cases completed by only one engine do not enter these rates or charts.

| Suite | Mode | Configurations (Guffin / llama.cpp) | Paired cases | Guffin t/s | llama.cpp t/s | Rate ratio |
| --- | --- | --- | --- | --- | --- | --- |
| core | speculative | guffin-speculative / llama-speculative | 32 / 32 | 18.43 | 8.43 | 2.19× |
| supplementary | speculative | guffin-speculative / llama-speculative | 6 / 6 | 16.63 | 7.81 | 2.13× |

## guffin-speculative

Engine: `guffin`; mode: `speculative`; model SHA-256: `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5`.

Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.

| Suite | Complete / planned | Generation t/s | Selected tokens | Median internal TTFT ms | Peak footprint GiB |
| --- | --- | --- | --- | --- | --- |
| core | 32 / 32 | 18.43 | 47143 | 597.94 | 9.15 |
| supplementary | 6 / 6 | 16.63 | 5195 | 664.81 | 9.16 |

| Category | Turns | Generation t/s | Prompt t/s | Median internal TTFT ms |
| --- | --- | --- | --- | --- |
| coding | 10 | 19.47 | 93.77 | 600.55 |
| conversation | 2 | 18.27 | 51.49 | 368.98 |
| document | 2 | 14.52 | 96.10 | 24260.82 |
| extraction | 8 | 19.67 | 94.48 | 1199.46 |
| humanities | 8 | 17.18 | 73.42 | 508.59 |
| math | 8 | 21.13 | 64.40 | 600.94 |
| multilingual | 4 | 17.36 | 79.31 | 639.71 |
| planning | 6 | 17.77 | 70.50 | 659.23 |
| reasoning | 8 | 19.73 | 78.62 | 609.08 |
| roleplay | 8 | 17.67 | 73.40 | 602.56 |
| stem | 8 | 17.96 | 69.40 | 598.36 |
| writing | 8 | 17.73 | 68.14 | 508.76 |

| Turn position | Turns | Generation t/s | Median internal TTFT ms |
| --- | --- | --- | --- |
| first | 38 | 18.45 | 613.45 |
| follow-up | 42 | 18.00 | 520.22 |

## llama-speculative

Engine: `llama`; mode: `speculative`; model SHA-256: `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5`.

Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.

| Suite | Complete / planned | Generation t/s | Selected tokens | Median internal TTFT ms | Peak footprint GiB |
| --- | --- | --- | --- | --- | --- |
| core | 32 / 32 | 8.43 | 45550 | 1020.54 | 4.47 |
| supplementary | 6 / 6 | 7.81 | 4920 | 1369.21 | 7.10 |

| Category | Turns | Generation t/s | Prompt t/s | Median internal TTFT ms |
| --- | --- | --- | --- | --- |
| coding | 10 | 9.77 | 106.75 | 1148.51 |
| conversation | 2 | 9.26 | 57.54 | 799.48 |
| document | 2 | 7.38 | 115.86 | 20128.26 |
| extraction | 8 | 11.37 | 102.31 | 1535.65 |
| humanities | 8 | 7.46 | 59.33 | 803.78 |
| math | 8 | 11.75 | 104.84 | 1495.96 |
| multilingual | 4 | 7.36 | 96.59 | 1765.21 |
| planning | 6 | 7.74 | 86.62 | 2106.58 |
| reasoning | 8 | 9.60 | 86.12 | 1150.81 |
| roleplay | 8 | 7.52 | 102.08 | 1041.32 |
| stem | 8 | 8.28 | 87.69 | 803.86 |
| writing | 8 | 7.63 | 97.15 | 1140.71 |

| Turn position | Turns | Generation t/s | Median internal TTFT ms |
| --- | --- | --- | --- |
| first | 38 | 8.62 | 807.73 |
| follow-up | 42 | 8.09 | 1394.82 |

## Cases

| Case | Suite | Category | Configuration | Status | Stop reasons | Turns | Generation t/s | Selected tokens | Wall reply seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| guffin-document-review | supplementary | document | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 14.52 | 1130 | 126.44 |
| mt-bench-141 | core | stem | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.59 | 2422 | 138.44 |
| mt-bench-143 | core | stem | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.26 | 2510 | 138.70 |
| guffin-code-review | supplementary | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 16.67 | 1186 | 92.70 |
| mt-bench-96 | core | roleplay | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.50 | 1599 | 221.17 |
| mt-bench-106 | core | reasoning | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.50 | 529 | 28.66 |
| mt-bench-151 | core | humanities | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.97 | 2079 | 262.11 |
| mt-bench-105 | core | reasoning | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 20.77 | 1075 | 54.59 |
| guffin-code-review | supplementary | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 8.78 | 1162 | 151.48 |
| mt-bench-100 | core | roleplay | llama-speculative | complete | unknown: 2 | 2 / 2 | 6.69 | 1903 | 291.15 |
| mt-bench-106 | core | reasoning | llama-speculative | complete | unknown: 2 | 2 / 2 | 9.13 | 469 | 53.97 |
| mt-bench-116 | core | math | llama-speculative | complete | unknown: 2 | 2 / 2 | 11.27 | 694 | 65.21 |
| mt-bench-87 | core | writing | llama-speculative | complete | unknown: 2 | 2 / 2 | 6.93 | 1015 | 153.67 |
| mt-bench-101 | core | reasoning | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.66 | 157 | 22.30 |
| mt-bench-114 | core | math | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 21.22 | 1699 | 81.24 |
| mt-bench-90 | core | writing | llama-speculative | complete | unknown: 2 | 2 / 2 | 9.81 | 686 | 75.20 |
| guffin-swedish-chat | supplementary | multilingual | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 16.72 | 446 | 27.86 |
| mt-bench-157 | core | humanities | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.35 | 2332 | 137.10 |
| guffin-chinese-chat | supplementary | multilingual | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.80 | 712 | 41.26 |
| mt-bench-131 | core | extraction | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.18 | 120 | 12.28 |
| mt-bench-154 | core | humanities | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.66 | 2953 | 386.68 |
| mt-bench-133 | core | extraction | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.13 | 158 | 17.38 |
| mt-bench-154 | core | humanities | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.03 | 3263 | 192.78 |
| mt-bench-149 | core | stem | llama-speculative | complete | unknown: 2 | 2 / 2 | 8.70 | 2316 | 267.60 |
| mt-bench-145 | core | stem | llama-speculative | complete | unknown: 2 | 2 / 2 | 8.18 | 1835 | 232.64 |
| mt-bench-134 | core | extraction | llama-speculative | complete | unknown: 2 | 2 / 2 | 10.36 | 260 | 27.99 |
| mt-bench-114 | core | math | llama-speculative | complete | unknown: 2 | 2 / 2 | 11.08 | 1498 | 141.04 |
| mt-bench-131 | core | extraction | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 20.48 | 114 | 8.01 |
| guffin-document-review | supplementary | document | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.38 | 980 | 172.86 |
| mt-bench-140 | core | extraction | llama-speculative | complete | unknown: 2 | 2 / 2 | 11.93 | 167 | 18.16 |
| mt-bench-134 | core | extraction | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.66 | 250 | 15.47 |
| mt-bench-151 | core | humanities | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.66 | 2317 | 132.27 |
| mt-bench-90 | core | writing | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.85 | 964 | 50.21 |
| mt-bench-97 | core | roleplay | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.17 | 2225 | 123.89 |
| mt-bench-87 | core | writing | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.66 | 1069 | 61.56 |
| mt-bench-81 | core | writing | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.32 | 3015 | 413.35 |
| mt-bench-143 | core | stem | llama-speculative | complete | unknown: 2 | 2 / 2 | 8.57 | 2513 | 294.72 |
| mt-bench-95 | core | roleplay | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.62 | 431 | 25.95 |
| mt-bench-82 | core | writing | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.90 | 544 | 29.80 |
| mt-bench-111 | core | math | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 21.03 | 2149 | 103.34 |
| guffin-swedish-chat | supplementary | multilingual | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.00 | 438 | 66.17 |
| mt-bench-149 | core | stem | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.11 | 2246 | 125.11 |
| mt-bench-100 | core | roleplay | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 16.96 | 1418 | 84.70 |
| mt-bench-95 | core | roleplay | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.51 | 140 | 20.95 |
| mt-bench-82 | core | writing | llama-speculative | complete | unknown: 2 | 2 / 2 | 8.88 | 580 | 67.19 |
| guffin-short-chat | supplementary | conversation | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.27 | 53 | 3.61 |
| mt-bench-155 | core | humanities | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 16.83 | 2859 | 170.86 |
| mt-bench-124 | core | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 9.04 | 1341 | 158.20 |
| mt-bench-97 | core | roleplay | llama-speculative | complete | unknown: 2 | 2 / 2 | 8.43 | 2245 | 268.05 |
| mt-bench-155 | core | humanities | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.14 | 3038 | 426.80 |
| mt-bench-123 | core | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 20.42 | 1260 | 62.82 |
| guffin-short-chat | supplementary | conversation | llama-speculative | complete | unknown: 2 | 2 / 2 | 9.26 | 52 | 7.01 |
| mt-bench-111 | core | math | llama-speculative | complete | unknown: 2 | 2 / 2 | 12.65 | 1618 | 135.30 |
| mt-bench-96 | core | roleplay | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.66 | 1578 | 90.81 |
| mt-bench-140 | core | extraction | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.91 | 184 | 13.23 |
| mt-bench-121 | core | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 10.37 | 2713 | 262.95 |
| mt-bench-101 | core | reasoning | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.44 | 198 | 12.46 |
| mt-bench-116 | core | math | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 21.08 | 922 | 44.66 |
| mt-bench-124 | core | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.65 | 1199 | 65.40 |
| mt-bench-126 | core | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 9.71 | 1777 | 184.38 |
| mt-bench-121 | core | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 20.04 | 3122 | 156.67 |
| mt-bench-81 | core | writing | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 16.95 | 2827 | 167.85 |
| mt-bench-119 | core | math | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 21.37 | 565 | 27.97 |
| mt-bench-126 | core | coding | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 19.79 | 2545 | 129.54 |
| guffin-sustained-design | supplementary | planning | guffin-speculative | complete | stop_token: 6 | 6 / 6 | 17.77 | 1668 | 97.97 |
| mt-bench-141 | core | stem | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.77 | 2739 | 353.53 |
| mt-bench-133 | core | extraction | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.88 | 158 | 12.26 |
| mt-bench-109 | core | reasoning | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 18.40 | 309 | 17.82 |
| mt-bench-123 | core | coding | llama-speculative | complete | unknown: 2 | 2 / 2 | 10.36 | 1716 | 174.69 |
| mt-bench-157 | core | humanities | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.25 | 2390 | 332.41 |
| guffin-chinese-chat | supplementary | multilingual | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.62 | 647 | 88.17 |
| mt-bench-105 | core | reasoning | llama-speculative | complete | unknown: 2 | 2 / 2 | 10.74 | 1084 | 103.52 |
| mt-bench-109 | core | reasoning | llama-speculative | complete | unknown: 2 | 2 / 2 | 7.85 | 239 | 32.51 |
| mt-bench-145 | core | stem | guffin-speculative | complete | stop_token: 2 | 2 / 2 | 17.89 | 1860 | 104.81 |
| mt-bench-119 | core | math | llama-speculative | complete | unknown: 2 | 2 / 2 | 11.92 | 493 | 44.19 |
| guffin-sustained-design | supplementary | planning | llama-speculative | complete | unknown: 6 | 6 / 6 | 7.74 | 1641 | 224.23 |

## Per-case generation rate ratios

Guffin / llama.cpp; above 1 means higher measured generation rate. Modes and source-model hashes match, but natural outputs can differ.

| Case | Mode | Configurations | Rate ratio |
| --- | --- | --- | --- |
| mt-bench-100 | speculative | guffin-speculative / llama-speculative | 2.53× |
| mt-bench-101 | speculative | guffin-speculative / llama-speculative | 2.28× |
| mt-bench-105 | speculative | guffin-speculative / llama-speculative | 1.93× |
| mt-bench-106 | speculative | guffin-speculative / llama-speculative | 2.14× |
| mt-bench-109 | speculative | guffin-speculative / llama-speculative | 2.34× |
| mt-bench-111 | speculative | guffin-speculative / llama-speculative | 1.66× |
| mt-bench-114 | speculative | guffin-speculative / llama-speculative | 1.92× |
| mt-bench-116 | speculative | guffin-speculative / llama-speculative | 1.87× |
| mt-bench-119 | speculative | guffin-speculative / llama-speculative | 1.79× |
| mt-bench-121 | speculative | guffin-speculative / llama-speculative | 1.93× |
| mt-bench-123 | speculative | guffin-speculative / llama-speculative | 1.97× |
| mt-bench-124 | speculative | guffin-speculative / llama-speculative | 2.17× |
| mt-bench-126 | speculative | guffin-speculative / llama-speculative | 2.04× |
| mt-bench-131 | speculative | guffin-speculative / llama-speculative | 1.68× |
| mt-bench-133 | speculative | guffin-speculative / llama-speculative | 1.56× |
| mt-bench-134 | speculative | guffin-speculative / llama-speculative | 1.90× |
| mt-bench-140 | speculative | guffin-speculative / llama-speculative | 1.67× |
| mt-bench-141 | speculative | guffin-speculative / llama-speculative | 2.26× |
| mt-bench-143 | speculative | guffin-speculative / llama-speculative | 2.13× |
| mt-bench-145 | speculative | guffin-speculative / llama-speculative | 2.19× |
| mt-bench-149 | speculative | guffin-speculative / llama-speculative | 2.08× |
| mt-bench-151 | speculative | guffin-speculative / llama-speculative | 2.22× |
| mt-bench-154 | speculative | guffin-speculative / llama-speculative | 2.22× |
| mt-bench-155 | speculative | guffin-speculative / llama-speculative | 2.36× |
| mt-bench-157 | speculative | guffin-speculative / llama-speculative | 2.39× |
| mt-bench-81 | speculative | guffin-speculative / llama-speculative | 2.32× |
| mt-bench-82 | speculative | guffin-speculative / llama-speculative | 2.13× |
| mt-bench-87 | speculative | guffin-speculative / llama-speculative | 2.55× |
| mt-bench-90 | speculative | guffin-speculative / llama-speculative | 2.02× |
| mt-bench-95 | speculative | guffin-speculative / llama-speculative | 2.35× |
| mt-bench-96 | speculative | guffin-speculative / llama-speculative | 2.36× |
| mt-bench-97 | speculative | guffin-speculative / llama-speculative | 2.16× |
| guffin-chinese-chat | speculative | guffin-speculative / llama-speculative | 2.34× |
| guffin-code-review | speculative | guffin-speculative / llama-speculative | 1.90× |
| guffin-document-review | speculative | guffin-speculative / llama-speculative | 1.97× |
| guffin-short-chat | speculative | guffin-speculative / llama-speculative | 1.97× |
| guffin-sustained-design | speculative | guffin-speculative / llama-speculative | 2.30× |
| guffin-swedish-chat | speculative | guffin-speculative / llama-speculative | 2.39× |

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
