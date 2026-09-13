# Control latency while the planner thinks (DRY RUN: synthetic models)

CPU: 13th Gen Intel(R) Core(TM) i5-13600KF, 6 P-cores (12 threads) + 8 E-cores (8 threads) (Windows GetLogicalProcessorInformationEx) · OpenVINO 2026.3.1-22476-759c5a6ab8c-releases/2026/3 · control step = policy + classifier, paced at 25 Hz, budget 40 ms · 30 s per scenario

Warning: the CPU was 22% busy before the run; rerun on an idle machine.

| | scenario | control step p50 / p95 / p99 (ms) | steps over 40 ms | policy / classifier p50 (ms) | planner s per call (median, n) | P-core / E-core utilisation |
|---|---|---|---|---|---|---|
| a | control alone, default | 2.46 / 6.12 / 15.91 | 0.0% | 1.9 / 0.55 | – | 10.5% / 6.3% |
| a' | control alone, P-cores | 1.99 / 4.06 / 5.47 | 0.0% | 1.5 / 0.45 | – | 7.9% / 6.0% |
| b | control + planner, both default | 4.99 / 7.9 / 10.84 | 0.13% | 3.63 / 1.15 | 0.49 (63) | 14.5% / 11.8% |
| c | control P-cores + planner E-cores | 2.77 / 4.96 / 6.69 | 0.0% | 1.93 / 0.77 | 0.55 (54) | 11.4% / 19.0% |
| d | as c, control hyper-threading on | 3.11 / 6.05 / 9.91 | 0.0% | 2.14 / 0.92 | 0.76 (40) | 26.5% / 36.6% |
| e | as c, CPU pinning on | 7.61 / 55.83 / 103.1 | 6.96% | 5.52 / 1.49 | 1.45 (22) | 24.7% / 23.5% |

Placement (tenplaces.cores): control {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1, "ENABLE_HYPER_THREADING": false, "ENABLE_CPU_PINNING": false, "SCHEDULING_CORE_TYPE": "PCORE_ONLY", "INFERENCE_NUM_THREADS": 6}; planner {"SCHEDULING_CORE_TYPE": "ECORE_ONLY", "INFERENCE_NUM_THREADS": 8, "ENABLE_CPU_PINNING": false}
