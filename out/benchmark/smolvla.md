# SmolVLA on the CPU (Intel64 Family 6 Model 183 Stepping 1, GenuineIntel)

`models/smolvla_base`, 450M parameters, 3 cameras [3, 256, 256], 10 flow-matching steps, a chunk of 50 actions per pass. PyTorch 2.11.0+cu128 FP32, 14 threads, random inputs (timing only).

| per pass, median | p90 | passes |
|---|---|---|
| 6622.8 ms | 7821.9 ms | 10 |
