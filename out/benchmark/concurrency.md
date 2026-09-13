# Control latency while the planner thinks

CPU: 13th Gen Intel(R) Core(TM) i5-13600KF, 6 P-cores (12 threads) + 8 E-cores (8 threads) (Windows GetLogicalProcessorInformationEx) · OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3 · control step = policy + classifier, paced at 25 Hz, budget 40 ms · 60 s per scenario

| | scenario | control step p50 / p95 / p99 (ms) | steps over 40 ms | policy / classifier p50 (ms) | planner s per call (median, n) | P-core / E-core utilisation |
|---|---|---|---|---|---|---|
| a | control alone, default | 45.48 / 70.18 / 76.56 | 79.72% | 36.38 / 8.12 | – | 93.1% / 11.9% |
| a' | control alone, P-cores | 43.74 / 51.67 / 58.78 | 76.39% | 35.53 / 8.17 | – | 93.3% / 13.8% |
| b | control + planner, both default | 140.58 / 288.25 / 437.9 | 100.0% | 86.69 / 19.77 | 8.23 (6) | 88.9% / 16.2% |
| c | control P-cores + planner E-cores | 62.06 / 93.78 / 148.58 | 96.9% | 48.38 / 9.93 | 8.33 (6) | 99.8% / 43.2% |
| d | as c, control hyper-threading on | 36.91 / 41.09 / 43.67 | 9.99% | 29.91 / 7.12 | 9.13 (7) | 96.6% / 30.1% |
| e | as c, CPU pinning on | 29.69 / 40.23 / 57.78 | 5.22% | 23.92 / 5.58 | 12.23 (4) | 94.3% / 34.6% |

Placement (tenplaces.cores): control {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1, "ENABLE_HYPER_THREADING": false, "ENABLE_CPU_PINNING": false, "SCHEDULING_CORE_TYPE": "PCORE_ONLY", "INFERENCE_NUM_THREADS": 6}; planner {"SCHEDULING_CORE_TYPE": "ECORE_ONLY", "INFERENCE_NUM_THREADS": 8, "ENABLE_CPU_PINNING": false}
