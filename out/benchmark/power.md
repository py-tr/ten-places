# Power per inference — 13th Gen Intel(R) Core(TM) i5-13600KF

Meter: HWiNFO64 `CPU Package Power [W]` (CPU package, not wall power). Checkpoint: `out\train\skills_ctx\cup\checkpoints\015000\pretrained_model`.

| phase | CPU busy | mean package power | inferences/s | energy per inference | above idle |
|---|---|---|---|---|---|
| idle | 0% | 18.0 W (30 samples) | – | – mJ | – mJ |
| pytorch_fp32 | 64% | 108.9 W (30 samples) | 26 | 4193 mJ | 3499 mJ |
| openvino_fp32 | 8% | 109.3 W (30 samples) | 53 | 2075 mJ | 1732 mJ |
| openvino_int8w | 9% | 106.6 W (30 samples) | 62 | 1716 mJ | 1426 mJ |
| openvino_int8w_25hz | 3% | 63.3 W (30 samples) | 25 | 2532 mJ | 1811 mJ |
| openvino_int8w_25hz_pcores_pinned | 2% | 56.0 W (30 samples) | 25 | 2242 mJ | 1521 mJ |
