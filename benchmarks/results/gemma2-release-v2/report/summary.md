# Release chat run 6f371fa1-9501-42c7-b4d4-86708ee01051

Single-pass actual-chat observations. Frontend-completed conversations contribute to the rates; known limited, interrupted and incomplete conversations remain listed below. Unknown stop reasons are not assumed to be EOS.

## Paired workload comparisons

Each row compares the same jointly complete case IDs within one model, mode and suite. Coverage is jointly complete / planned for both engines; cases completed by only one engine do not enter these rates or charts.

| Suite | Mode | Configurations (Guffin / llama.cpp) | Paired cases | Guffin t/s | llama.cpp t/s | Rate ratio |
| --- | --- | --- | --- | --- | --- | --- |
| core | default | guffin-default / llama-default | 32 / 32 | 16.76 | 13.63 | 1.23× |
| supplementary | default | guffin-default / llama-default | 6 / 6 | 15.40 | 12.64 | 1.22× |

## guffin-default

Engine: `guffin`; mode: `default`; model SHA-256: `0452e54ba9b6c5b2e466fbd7f28c089fefb7c3dceddcc2b75becab6b16cae523`.

Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.

| Suite | Complete / planned | Generation t/s | Selected tokens | Median internal TTFT ms | Peak footprint GiB |
| --- | --- | --- | --- | --- | --- |
| core | 32 / 32 | 16.76 | 21729 | 399.82 | 9.15 |
| supplementary | 6 / 6 | 15.40 | 4226 | 443.12 | 9.17 |

| Category | Turns | Generation t/s | Prompt t/s | Median internal TTFT ms |
| --- | --- | --- | --- | --- |
| coding | 10 | 16.80 | 141.96 | 402.19 |
| conversation | 2 | 16.40 | 79.14 | 240.08 |
| document | 2 | 13.35 | 133.50 | 17464.22 |
| extraction | 8 | 17.37 | 152.33 | 722.08 |
| humanities | 8 | 16.65 | 125.20 | 253.04 |
| math | 8 | 17.38 | 101.25 | 402.50 |
| multilingual | 4 | 16.37 | 116.21 | 416.79 |
| planning | 6 | 16.52 | 102.73 | 450.08 |
| reasoning | 8 | 16.93 | 130.32 | 407.89 |
| roleplay | 8 | 16.16 | 121.47 | 335.79 |
| stem | 8 | 16.58 | 114.46 | 328.55 |
| writing | 8 | 16.57 | 115.24 | 256.09 |

| Turn position | Turns | Generation t/s | Median internal TTFT ms |
| --- | --- | --- | --- |
| first | 38 | 16.45 | 409.90 |
| follow-up | 42 | 16.61 | 257.79 |

## llama-default

Engine: `llama`; mode: `default`; model SHA-256: `0452e54ba9b6c5b2e466fbd7f28c089fefb7c3dceddcc2b75becab6b16cae523`.

Independent completed workload; these raw configuration statistics can cover different case IDs and are not cross-engine comparison aggregates.

| Suite | Complete / planned | Generation t/s | Selected tokens | Median internal TTFT ms | Peak footprint GiB |
| --- | --- | --- | --- | --- | --- |
| core | 32 / 32 | 13.63 | 20167 | 1179.88 | 2.58 |
| supplementary | 6 / 6 | 12.64 | 3822 | 1867.92 | 5.49 |

| Category | Turns | Generation t/s | Prompt t/s | Median internal TTFT ms |
| --- | --- | --- | --- | --- |
| coding | 10 | 13.27 | 143.47 | 3686.53 |
| conversation | 2 | 13.74 | 68.55 | 649.18 |
| document | 2 | 11.26 | 139.92 | 16667.10 |
| extraction | 8 | 13.93 | 133.00 | 1475.17 |
| humanities | 8 | 13.49 | 137.93 | 2086.54 |
| math | 8 | 13.64 | 119.67 | 1103.53 |
| multilingual | 4 | 13.69 | 119.59 | 1202.11 |
| planning | 6 | 13.29 | 123.65 | 2067.40 |
| reasoning | 8 | 13.62 | 115.00 | 1012.41 |
| roleplay | 8 | 13.53 | 129.27 | 926.64 |
| stem | 8 | 13.76 | 133.10 | 1593.30 |
| writing | 8 | 13.93 | 130.10 | 1087.19 |

| Turn position | Turns | Generation t/s | Median internal TTFT ms |
| --- | --- | --- | --- |
| first | 38 | 13.62 | 654.43 |
| follow-up | 42 | 13.29 | 2326.86 |

## Cases

| Case | Suite | Category | Configuration | Status | Stop reasons | Turns | Generation t/s | Selected tokens | Wall reply seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mt-bench-116 | core | math | guffin-default | complete | stop_token: 2 | 2 / 2 | 17.64 | 515 | 29.83 |
| mt-bench-143 | core | stem | llama-default | complete | unknown: 2 | 2 / 2 | 13.91 | 958 | 72.40 |
| mt-bench-95 | core | roleplay | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.14 | 68 | 5.22 |
| mt-bench-154 | core | humanities | llama-default | complete | unknown: 2 | 2 / 2 | 13.34 | 1431 | 113.33 |
| mt-bench-97 | core | roleplay | llama-default | complete | unknown: 2 | 2 / 2 | 13.55 | 906 | 71.10 |
| guffin-document-review | supplementary | document | guffin-default | complete | stop_token: 2 | 2 / 2 | 13.35 | 1097 | 117.12 |
| mt-bench-140 | core | extraction | llama-default | complete | unknown: 2 | 2 / 2 | 14.09 | 439 | 34.87 |
| mt-bench-145 | core | stem | llama-default | complete | unknown: 2 | 2 / 2 | 13.59 | 606 | 47.65 |
| mt-bench-82 | core | writing | llama-default | complete | unknown: 2 | 2 / 2 | 14.24 | 316 | 23.87 |
| guffin-chinese-chat | supplementary | multilingual | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.44 | 449 | 28.13 |
| mt-bench-114 | core | math | llama-default | complete | unknown: 2 | 2 / 2 | 13.52 | 567 | 44.03 |
| guffin-short-chat | supplementary | conversation | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.40 | 33 | 2.48 |
| mt-bench-101 | core | reasoning | llama-default | complete | unknown: 2 | 2 / 2 | 13.96 | 87 | 7.57 |
| mt-bench-101 | core | reasoning | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.58 | 64 | 4.67 |
| mt-bench-124 | core | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 17.89 | 1078 | 63.09 |
| mt-bench-119 | core | math | guffin-default | complete | stop_token: 2 | 2 / 2 | 18.14 | 448 | 25.72 |
| mt-bench-82 | core | writing | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.47 | 358 | 22.38 |
| mt-bench-106 | core | reasoning | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.82 | 201 | 12.78 |
| mt-bench-143 | core | stem | guffin-default | complete | stop_token: 2 | 2 / 2 | 17.23 | 1051 | 61.83 |
| mt-bench-154 | core | humanities | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.77 | 1495 | 89.84 |
| mt-bench-149 | core | stem | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.22 | 1029 | 64.10 |
| mt-bench-81 | core | writing | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.62 | 1086 | 65.81 |
| mt-bench-133 | core | extraction | guffin-default | complete | stop_token: 2 | 2 / 2 | 17.01 | 194 | 13.97 |
| mt-bench-95 | core | roleplay | llama-default | complete | unknown: 2 | 2 / 2 | 13.70 | 103 | 9.24 |
| mt-bench-123 | core | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.35 | 750 | 46.50 |
| mt-bench-87 | core | writing | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.12 | 742 | 46.67 |
| mt-bench-126 | core | coding | llama-default | complete | unknown: 2 | 2 / 2 | 13.48 | 1471 | 115.74 |
| mt-bench-105 | core | reasoning | llama-default | complete | unknown: 2 | 2 / 2 | 13.47 | 349 | 29.73 |
| mt-bench-114 | core | math | guffin-default | complete | stop_token: 2 | 2 / 2 | 17.28 | 657 | 38.68 |
| mt-bench-134 | core | extraction | guffin-default | complete | stop_token: 2 | 2 / 2 | 18.39 | 263 | 16.08 |
| guffin-sustained-design | supplementary | planning | guffin-default | complete | stop_token: 6 | 6 / 6 | 16.52 | 1586 | 98.71 |
| guffin-sustained-design | supplementary | planning | llama-default | complete | unknown: 6 | 6 / 6 | 13.29 | 1339 | 111.76 |
| mt-bench-116 | core | math | llama-default | complete | unknown: 2 | 2 / 2 | 13.63 | 438 | 34.79 |
| mt-bench-100 | core | roleplay | llama-default | complete | unknown: 2 | 2 / 2 | 13.47 | 697 | 54.81 |
| mt-bench-141 | core | stem | llama-default | complete | unknown: 2 | 2 / 2 | 13.97 | 873 | 66.36 |
| mt-bench-140 | core | extraction | guffin-default | complete | stop_token: 2 | 2 / 2 | 17.09 | 319 | 21.02 |
| mt-bench-134 | core | extraction | llama-default | complete | unknown: 2 | 2 / 2 | 14.07 | 257 | 20.55 |
| mt-bench-124 | core | coding | llama-default | complete | unknown: 2 | 2 / 2 | 13.37 | 667 | 55.26 |
| mt-bench-157 | core | humanities | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.76 | 996 | 61.05 |
| mt-bench-157 | core | humanities | llama-default | complete | unknown: 2 | 2 / 2 | 13.41 | 782 | 62.33 |
| mt-bench-123 | core | coding | llama-default | complete | unknown: 2 | 2 / 2 | 13.72 | 881 | 68.46 |
| mt-bench-90 | core | writing | llama-default | complete | unknown: 2 | 2 / 2 | 13.85 | 449 | 35.84 |
| mt-bench-111 | core | math | llama-default | complete | unknown: 2 | 2 / 2 | 13.79 | 607 | 46.85 |
| guffin-chinese-chat | supplementary | multilingual | llama-default | complete | unknown: 2 | 2 / 2 | 13.59 | 401 | 32.14 |
| mt-bench-97 | core | roleplay | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.21 | 898 | 56.23 |
| mt-bench-119 | core | math | llama-default | complete | unknown: 2 | 2 / 2 | 13.57 | 437 | 34.27 |
| mt-bench-121 | core | coding | llama-default | complete | unknown: 2 | 2 / 2 | 13.31 | 1255 | 99.13 |
| mt-bench-111 | core | math | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.62 | 479 | 29.48 |
| mt-bench-109 | core | reasoning | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.36 | 222 | 14.21 |
| mt-bench-155 | core | humanities | llama-default | complete | unknown: 2 | 2 / 2 | 13.58 | 1004 | 77.95 |
| mt-bench-133 | core | extraction | llama-default | complete | unknown: 2 | 2 / 2 | 13.44 | 152 | 14.76 |
| guffin-short-chat | supplementary | conversation | llama-default | complete | unknown: 2 | 2 / 2 | 13.74 | 49 | 4.73 |
| guffin-swedish-chat | supplementary | multilingual | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.27 | 307 | 19.71 |
| mt-bench-106 | core | reasoning | llama-default | complete | unknown: 2 | 2 / 2 | 13.67 | 211 | 17.33 |
| mt-bench-90 | core | writing | guffin-default | complete | stop_token: 2 | 2 / 2 | 17.34 | 443 | 26.56 |
| mt-bench-149 | core | stem | llama-default | complete | unknown: 2 | 2 / 2 | 13.48 | 800 | 63.59 |
| mt-bench-87 | core | writing | llama-default | complete | unknown: 2 | 2 / 2 | 13.52 | 741 | 59.63 |
| mt-bench-155 | core | humanities | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.62 | 1231 | 74.55 |
| guffin-swedish-chat | supplementary | multilingual | llama-default | complete | unknown: 2 | 2 / 2 | 13.83 | 298 | 23.81 |
| mt-bench-141 | core | stem | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.67 | 884 | 53.49 |
| mt-bench-145 | core | stem | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.06 | 689 | 43.47 |
| mt-bench-81 | core | writing | llama-default | complete | unknown: 2 | 2 / 2 | 14.18 | 1033 | 76.97 |
| mt-bench-105 | core | reasoning | guffin-default | complete | stop_token: 2 | 2 / 2 | 17.40 | 386 | 23.81 |
| mt-bench-96 | core | roleplay | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.14 | 478 | 30.46 |
| mt-bench-96 | core | roleplay | llama-default | complete | unknown: 2 | 2 / 2 | 13.56 | 551 | 43.31 |
| mt-bench-126 | core | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.96 | 1616 | 95.96 |
| mt-bench-131 | core | extraction | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.98 | 231 | 15.02 |
| mt-bench-109 | core | reasoning | llama-default | complete | unknown: 2 | 2 / 2 | 13.69 | 179 | 14.80 |
| guffin-code-review | supplementary | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 15.69 | 754 | 62.45 |
| mt-bench-131 | core | extraction | llama-default | complete | unknown: 2 | 2 / 2 | 13.67 | 98 | 9.07 |
| guffin-code-review | supplementary | coding | llama-default | complete | unknown: 2 | 2 / 2 | 12.41 | 863 | 86.20 |
| guffin-document-review | supplementary | document | llama-default | complete | unknown: 2 | 2 / 2 | 11.26 | 872 | 110.69 |
| mt-bench-151 | core | humanities | llama-default | complete | unknown: 2 | 2 / 2 | 13.73 | 822 | 63.71 |
| mt-bench-151 | core | humanities | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.38 | 901 | 55.65 |
| mt-bench-121 | core | coding | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.73 | 1253 | 75.56 |
| mt-bench-100 | core | roleplay | guffin-default | complete | stop_token: 2 | 2 / 2 | 16.11 | 704 | 44.36 |

## Per-case generation rate ratios

Guffin / llama.cpp; above 1 means higher measured generation rate. Modes and source-model hashes match, but natural outputs can differ.

| Case | Mode | Configurations | Rate ratio |
| --- | --- | --- | --- |
| mt-bench-100 | default | guffin-default / llama-default | 1.20× |
| mt-bench-101 | default | guffin-default / llama-default | 1.19× |
| mt-bench-105 | default | guffin-default / llama-default | 1.29× |
| mt-bench-106 | default | guffin-default / llama-default | 1.23× |
| mt-bench-109 | default | guffin-default / llama-default | 1.20× |
| mt-bench-111 | default | guffin-default / llama-default | 1.20× |
| mt-bench-114 | default | guffin-default / llama-default | 1.28× |
| mt-bench-116 | default | guffin-default / llama-default | 1.29× |
| mt-bench-119 | default | guffin-default / llama-default | 1.34× |
| mt-bench-121 | default | guffin-default / llama-default | 1.26× |
| mt-bench-123 | default | guffin-default / llama-default | 1.19× |
| mt-bench-124 | default | guffin-default / llama-default | 1.34× |
| mt-bench-126 | default | guffin-default / llama-default | 1.26× |
| mt-bench-131 | default | guffin-default / llama-default | 1.24× |
| mt-bench-133 | default | guffin-default / llama-default | 1.27× |
| mt-bench-134 | default | guffin-default / llama-default | 1.31× |
| mt-bench-140 | default | guffin-default / llama-default | 1.21× |
| mt-bench-141 | default | guffin-default / llama-default | 1.19× |
| mt-bench-143 | default | guffin-default / llama-default | 1.24× |
| mt-bench-145 | default | guffin-default / llama-default | 1.18× |
| mt-bench-149 | default | guffin-default / llama-default | 1.20× |
| mt-bench-151 | default | guffin-default / llama-default | 1.19× |
| mt-bench-154 | default | guffin-default / llama-default | 1.26× |
| mt-bench-155 | default | guffin-default / llama-default | 1.22× |
| mt-bench-157 | default | guffin-default / llama-default | 1.25× |
| mt-bench-81 | default | guffin-default / llama-default | 1.17× |
| mt-bench-82 | default | guffin-default / llama-default | 1.16× |
| mt-bench-87 | default | guffin-default / llama-default | 1.19× |
| mt-bench-90 | default | guffin-default / llama-default | 1.25× |
| mt-bench-95 | default | guffin-default / llama-default | 1.18× |
| mt-bench-96 | default | guffin-default / llama-default | 1.19× |
| mt-bench-97 | default | guffin-default / llama-default | 1.20× |
| guffin-chinese-chat | default | guffin-default / llama-default | 1.21× |
| guffin-code-review | default | guffin-default / llama-default | 1.26× |
| guffin-document-review | default | guffin-default / llama-default | 1.19× |
| guffin-short-chat | default | guffin-default / llama-default | 1.19× |
| guffin-sustained-design | default | guffin-default / llama-default | 1.24× |
| guffin-swedish-chat | default | guffin-default / llama-default | 1.18× |

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
