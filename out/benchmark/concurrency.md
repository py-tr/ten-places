# Control latency while the planner thinks

CPU: 13th Gen Intel(R) Core(TM) i5-13600KF, 6 P-cores (12 threads) + 8 E-cores (8 threads) (Windows GetLogicalProcessorInformationEx) · OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3 · control step = policy + classifier, paced at 25 Hz, budget 40 ms · 60 s per scenario

| | scenario | control step p50 / p95 / p99 (ms) | steps over 40 ms | policy / classifier p50 (ms) | planner s per call (median, n) | P-core / E-core utilisation |
|---|---|---|---|---|---|---|
| a | control alone, default | 24.46 / 32.64 / 35.62 | 0.13% | 19.45 / 4.77 | – | 6.2% / 0.4% |
| a' | control alone, P-cores | 24.35 / 30.6 / 34.89 | 0.07% | 19.45 / 4.73 | – | 6.0% / 0.4% |
| b | control + planner, both default | 52.84 / 63.51 / 70.29 | 97.26% | 42.1 / 9.99 | 4.18 (14) | 26.1% / 0.6% |
| c | control P-cores + planner E-cores | 30.29 / 43.44 / 54.34 | 11.06% | 24.14 / 5.85 | 5.55 (9) | 8.3% / 43.2% |
| d | as c, control hyper-threading on | 29.01 / 34.11 / 36.19 | 0.0% | 23.2 / 5.69 | 5.52 (12) | 6.8% / 42.2% |
| e | as c, CPU pinning on | 29.31 / 42.6 / 55.44 | 5.99% | 23.31 / 5.73 | 5.86 (8) | 5.4% / 34.9% |

Placement (tenplaces.cores): control {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1, "ENABLE_HYPER_THREADING": false, "ENABLE_CPU_PINNING": false, "SCHEDULING_CORE_TYPE": "PCORE_ONLY", "INFERENCE_NUM_THREADS": 6}; planner {"SCHEDULING_CORE_TYPE": "ECORE_ONLY", "INFERENCE_NUM_THREADS": 8, "ENABLE_CPU_PINNING": false}
