# Release chat run 37131a8f-b572-4536-b08a-441ec90df018

Single-pass actual-chat observations. Frontend-completed conversations contribute to the rates; known limited, interrupted and incomplete conversations remain listed below. Unknown stop reasons are not assumed to be EOS.

## Paired workload comparisons

Each row compares the same jointly complete case IDs within one model, mode and suite. Coverage is jointly complete / planned for both engines; cases completed by only one engine do not enter these rates or charts.

| Suite | Mode | Configurations (Guffin / llama.cpp) | Paired cases | Guffin t/s | llama.cpp t/s | Rate ratio |
| --- | --- | --- | --- | --- | --- | --- |
| core | default | guffin-default / llama-default | 32 / 32 | 12.57 | 12.40 | 1.01× |
| supplementary | default | guffin-default / llama-default | 6 / 6 | 12.62 | 12.51 | 1.01× |

## guffin-default

Engine: `guffin`; mode: `default`; model SHA-256: `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5`.

Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.

| Suite | Complete / planned | Generation t/s | Selected tokens | Median internal TTFT ms | Peak footprint GiB |
| --- | --- | --- | --- | --- | --- |
| core | 32 / 32 | 12.57 | 46022 | 597.63 | 8.62 |
| supplementary | 6 / 6 | 12.62 | 5057 | 646.10 | 8.62 |

| Category | Turns | Generation t/s | Prompt t/s | Median internal TTFT ms |
| --- | --- | --- | --- | --- |
| coding | 10 | 12.72 | 93.59 | 601.53 |
| conversation | 2 | 13.09 | 51.67 | 367.73 |
| document | 2 | 12.21 | 96.13 | 24252.90 |
| extraction | 8 | 12.93 | 94.46 | 1199.71 |
| humanities | 8 | 11.90 | 74.09 | 508.96 |
| math | 8 | 12.90 | 64.99 | 600.04 |
| multilingual | 4 | 12.89 | 81.55 | 621.40 |
| planning | 6 | 12.76 | 71.25 | 655.75 |
| reasoning | 8 | 12.90 | 78.51 | 610.08 |
| roleplay | 8 | 12.78 | 73.47 | 603.39 |
| stem | 8 | 12.72 | 69.46 | 594.88 |
| writing | 8 | 12.99 | 68.65 | 497.53 |

| Turn position | Turns | Generation t/s | Median internal TTFT ms |
| --- | --- | --- | --- |
| first | 38 | 12.61 | 614.75 |
| follow-up | 42 | 12.54 | 522.78 |

## llama-default

Engine: `llama`; mode: `default`; model SHA-256: `7bb69bff3f48a7b642355d64a90e481182a7794707b3133890646b1efa778ff5`.

Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.

| Suite | Complete / planned | Generation t/s | Selected tokens | Median internal TTFT ms | Peak footprint GiB |
| --- | --- | --- | --- | --- | --- |
| core | 32 / 32 | 12.40 | 47489 | 966.35 | 5.06 |
| supplementary | 6 / 6 | 12.51 | 5033 | 1252.12 | 7.26 |

| Category | Turns | Generation t/s | Prompt t/s | Median internal TTFT ms |
| --- | --- | --- | --- | --- |
| coding | 10 | 12.60 | 109.15 | 1073.14 |
| conversation | 2 | 12.96 | 57.52 | 799.71 |
| document | 2 | 12.02 | 117.27 | 19886.51 |
| extraction | 8 | 12.98 | 102.54 | 1536.73 |
| humanities | 8 | 11.44 | 100.29 | 799.68 |
| math | 8 | 12.82 | 94.98 | 939.70 |
| multilingual | 4 | 12.83 | 95.23 | 1622.73 |
| planning | 6 | 12.70 | 89.32 | 2084.54 |
| reasoning | 8 | 12.89 | 87.25 | 1134.90 |
| roleplay | 8 | 12.74 | 105.55 | 1051.12 |
| stem | 8 | 12.64 | 87.35 | 809.62 |
| writing | 8 | 12.68 | 99.10 | 1087.73 |

| Turn position | Turns | Generation t/s | Median internal TTFT ms |
| --- | --- | --- | --- |
| first | 38 | 12.61 | 811.27 |
| follow-up | 42 | 12.20 | 1325.97 |

## Cases

| Case | Suite | Category | Configuration | Status | Stop reasons | Turns | Generation t/s | Selected tokens | Wall reply seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mt-bench-116 | core | math | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.96 | 740 | 57.99 |
| mt-bench-143 | core | stem | llama-default | complete | unknown: 2 | 2 / 2 | 12.71 | 2199 | 174.74 |
| mt-bench-95 | core | roleplay | guffin-default | complete | stop_token: 2 | 2 / 2 | 13.00 | 356 | 28.89 |
| mt-bench-154 | core | humanities | llama-default | complete | unknown: 2 | 2 / 2 | 11.14 | 3263 | 294.42 |
| mt-bench-97 | core | roleplay | llama-default | complete | unknown: 2 | 2 / 2 | 12.67 | 2234 | 178.13 |
| guffin-document-review | supplementary | document | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.21 | 988 | 129.45 |
| mt-bench-140 | core | extraction | llama-default | complete | unknown: 2 | 2 / 2 | 12.90 | 167 | 17.06 |
| mt-bench-145 | core | stem | llama-default | complete | unknown: 2 | 2 / 2 | 12.70 | 1820 | 151.53 |
| mt-bench-82 | core | writing | llama-default | complete | unknown: 2 | 2 / 2 | 12.90 | 580 | 46.90 |
| guffin-chinese-chat | supplementary | multilingual | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.94 | 729 | 57.57 |
| mt-bench-114 | core | math | llama-default | complete | unknown: 2 | 2 / 2 | 12.74 | 1481 | 121.43 |
| guffin-short-chat | supplementary | conversation | guffin-default | complete | stop_token: 2 | 2 / 2 | 13.09 | 52 | 4.70 |
| mt-bench-101 | core | reasoning | llama-default | complete | unknown: 2 | 2 / 2 | 12.95 | 174 | 15.36 |
| mt-bench-101 | core | reasoning | guffin-default | complete | stop_token: 2 | 2 / 2 | 13.04 | 153 | 12.96 |
| mt-bench-124 | core | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.79 | 1313 | 107.02 |
| mt-bench-119 | core | math | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.93 | 824 | 65.27 |
| mt-bench-82 | core | writing | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.98 | 520 | 41.09 |
| mt-bench-106 | core | reasoning | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.92 | 704 | 55.99 |
| mt-bench-143 | core | stem | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.70 | 2347 | 186.02 |
| mt-bench-154 | core | humanities | guffin-default | complete | stop_token: 2 | 2 / 2 | 11.08 | 3351 | 303.51 |
| mt-bench-149 | core | stem | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.67 | 2320 | 184.23 |
| mt-bench-81 | core | writing | guffin-default | complete | stop_token: 2 | 2 / 2 | 13.05 | 2451 | 188.79 |
| mt-bench-133 | core | extraction | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.90 | 156 | 16.07 |
| mt-bench-95 | core | roleplay | llama-default | complete | unknown: 2 | 2 / 2 | 12.96 | 162 | 14.93 |
| mt-bench-123 | core | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.77 | 2030 | 159.96 |
| mt-bench-87 | core | writing | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.88 | 1086 | 85.35 |
| mt-bench-126 | core | coding | llama-default | complete | unknown: 2 | 2 / 2 | 12.63 | 1988 | 158.82 |
| mt-bench-105 | core | reasoning | llama-default | complete | unknown: 2 | 2 / 2 | 12.86 | 1075 | 86.22 |
| mt-bench-114 | core | math | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.88 | 1194 | 93.89 |
| mt-bench-134 | core | extraction | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.94 | 242 | 21.54 |
| guffin-sustained-design | supplementary | planning | guffin-default | complete | stop_token: 6 | 6 / 6 | 12.76 | 1595 | 128.86 |
| guffin-sustained-design | supplementary | planning | llama-default | complete | unknown: 6 | 6 / 6 | 12.70 | 1639 | 141.30 |
| mt-bench-116 | core | math | llama-default | complete | unknown: 2 | 2 / 2 | 12.96 | 896 | 73.19 |
| mt-bench-100 | core | roleplay | llama-default | complete | unknown: 2 | 2 / 2 | 12.75 | 2060 | 169.24 |
| mt-bench-141 | core | stem | llama-default | complete | unknown: 2 | 2 / 2 | 12.68 | 2390 | 189.78 |
| mt-bench-140 | core | extraction | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.91 | 168 | 16.92 |
| mt-bench-134 | core | extraction | llama-default | complete | unknown: 2 | 2 / 2 | 13.03 | 259 | 22.76 |
| mt-bench-124 | core | coding | llama-default | complete | unknown: 2 | 2 / 2 | 12.76 | 1467 | 125.69 |
| mt-bench-157 | core | humanities | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.82 | 2736 | 216.09 |
| mt-bench-157 | core | humanities | llama-default | complete | unknown: 2 | 2 / 2 | 12.33 | 2395 | 196.95 |
| mt-bench-123 | core | coding | llama-default | complete | unknown: 2 | 2 / 2 | 12.64 | 2402 | 198.26 |
| mt-bench-90 | core | writing | llama-default | complete | unknown: 2 | 2 / 2 | 13.08 | 635 | 53.82 |
| mt-bench-111 | core | math | llama-default | complete | unknown: 2 | 2 / 2 | 12.80 | 1787 | 141.07 |
| guffin-chinese-chat | supplementary | multilingual | llama-default | complete | unknown: 2 | 2 / 2 | 12.81 | 645 | 53.44 |
| mt-bench-97 | core | roleplay | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.74 | 2204 | 174.39 |
| mt-bench-119 | core | math | llama-default | complete | unknown: 2 | 2 / 2 | 12.84 | 619 | 51.30 |
| mt-bench-121 | core | coding | llama-default | complete | unknown: 2 | 2 / 2 | 12.55 | 3060 | 245.06 |
| mt-bench-111 | core | math | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.87 | 1154 | 90.69 |
| mt-bench-109 | core | reasoning | guffin-default | complete | stop_token: 2 | 2 / 2 | 13.02 | 385 | 30.61 |
| mt-bench-155 | core | humanities | llama-default | complete | unknown: 2 | 2 / 2 | 10.35 | 2721 | 277.04 |
| mt-bench-133 | core | extraction | llama-default | complete | unknown: 2 | 2 / 2 | 13.02 | 158 | 16.48 |
| guffin-short-chat | supplementary | conversation | llama-default | complete | unknown: 2 | 2 / 2 | 12.96 | 52 | 5.48 |
| guffin-swedish-chat | supplementary | multilingual | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.81 | 443 | 35.84 |
| mt-bench-106 | core | reasoning | llama-default | complete | unknown: 2 | 2 / 2 | 12.98 | 395 | 32.82 |
| mt-bench-90 | core | writing | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.93 | 696 | 55.49 |
| mt-bench-149 | core | stem | llama-default | complete | unknown: 2 | 2 / 2 | 12.50 | 2309 | 186.23 |
| mt-bench-87 | core | writing | llama-default | complete | unknown: 2 | 2 / 2 | 12.90 | 1357 | 114.31 |
| mt-bench-155 | core | humanities | guffin-default | complete | stop_token: 2 | 2 / 2 | 11.58 | 3099 | 268.70 |
| guffin-swedish-chat | supplementary | multilingual | llama-default | complete | unknown: 2 | 2 / 2 | 12.86 | 437 | 37.38 |
| mt-bench-141 | core | stem | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.74 | 2239 | 176.47 |
| mt-bench-145 | core | stem | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.78 | 1734 | 136.66 |
| mt-bench-81 | core | writing | llama-default | complete | unknown: 2 | 2 / 2 | 12.48 | 3253 | 262.07 |
| mt-bench-105 | core | reasoning | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.83 | 1032 | 83.30 |
| mt-bench-96 | core | roleplay | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.78 | 1729 | 136.76 |
| mt-bench-96 | core | roleplay | llama-default | complete | unknown: 2 | 2 / 2 | 12.82 | 1568 | 130.24 |
| mt-bench-126 | core | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.77 | 2064 | 162.65 |
| mt-bench-131 | core | extraction | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.98 | 114 | 11.17 |
| mt-bench-109 | core | reasoning | llama-default | complete | unknown: 2 | 2 / 2 | 12.87 | 350 | 29.83 |
| guffin-code-review | supplementary | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.50 | 1250 | 121.69 |
| mt-bench-131 | core | extraction | llama-default | complete | unknown: 2 | 2 / 2 | 12.89 | 114 | 11.26 |
| guffin-code-review | supplementary | coding | llama-default | complete | unknown: 2 | 2 / 2 | 12.43 | 1095 | 106.42 |
| guffin-document-review | supplementary | document | llama-default | complete | unknown: 2 | 2 / 2 | 12.02 | 1165 | 136.61 |
| mt-bench-151 | core | humanities | llama-default | complete | unknown: 2 | 2 / 2 | 12.63 | 2151 | 171.52 |
| mt-bench-151 | core | humanities | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.72 | 2137 | 169.10 |
| mt-bench-121 | core | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.70 | 2979 | 235.58 |
| mt-bench-100 | core | roleplay | guffin-default | complete | stop_token: 2 | 2 / 2 | 12.79 | 1765 | 139.20 |

## Per-case generation rate ratios

Guffin / llama.cpp; above 1 means higher measured generation rate. Modes and source-model hashes match, but natural outputs can differ.

| Case | Mode | Configurations | Rate ratio |
| --- | --- | --- | --- |
| mt-bench-100 | default | guffin-default / llama-default | 1.00× |
| mt-bench-101 | default | guffin-default / llama-default | 1.01× |
| mt-bench-105 | default | guffin-default / llama-default | 1.00× |
| mt-bench-106 | default | guffin-default / llama-default | 1.00× |
| mt-bench-109 | default | guffin-default / llama-default | 1.01× |
| mt-bench-111 | default | guffin-default / llama-default | 1.01× |
| mt-bench-114 | default | guffin-default / llama-default | 1.01× |
| mt-bench-116 | default | guffin-default / llama-default | 1.00× |
| mt-bench-119 | default | guffin-default / llama-default | 1.01× |
| mt-bench-121 | default | guffin-default / llama-default | 1.01× |
| mt-bench-123 | default | guffin-default / llama-default | 1.01× |
| mt-bench-124 | default | guffin-default / llama-default | 1.00× |
| mt-bench-126 | default | guffin-default / llama-default | 1.01× |
| mt-bench-131 | default | guffin-default / llama-default | 1.01× |
| mt-bench-133 | default | guffin-default / llama-default | 0.99× |
| mt-bench-134 | default | guffin-default / llama-default | 0.99× |
| mt-bench-140 | default | guffin-default / llama-default | 1.00× |
| mt-bench-141 | default | guffin-default / llama-default | 1.00× |
| mt-bench-143 | default | guffin-default / llama-default | 1.00× |
| mt-bench-145 | default | guffin-default / llama-default | 1.01× |
| mt-bench-149 | default | guffin-default / llama-default | 1.01× |
| mt-bench-151 | default | guffin-default / llama-default | 1.01× |
| mt-bench-154 | default | guffin-default / llama-default | 1.00× |
| mt-bench-155 | default | guffin-default / llama-default | 1.12× |
| mt-bench-157 | default | guffin-default / llama-default | 1.04× |
| mt-bench-81 | default | guffin-default / llama-default | 1.05× |
| mt-bench-82 | default | guffin-default / llama-default | 1.01× |
| mt-bench-87 | default | guffin-default / llama-default | 1.00× |
| mt-bench-90 | default | guffin-default / llama-default | 0.99× |
| mt-bench-95 | default | guffin-default / llama-default | 1.00× |
| mt-bench-96 | default | guffin-default / llama-default | 1.00× |
| mt-bench-97 | default | guffin-default / llama-default | 1.01× |
| guffin-chinese-chat | default | guffin-default / llama-default | 1.01× |
| guffin-code-review | default | guffin-default / llama-default | 1.01× |
| guffin-document-review | default | guffin-default / llama-default | 1.02× |
| guffin-short-chat | default | guffin-default / llama-default | 1.01× |
| guffin-sustained-design | default | guffin-default / llama-default | 1.00× |
| guffin-swedish-chat | default | guffin-default / llama-default | 1.00× |

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
