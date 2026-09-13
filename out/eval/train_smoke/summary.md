# Evaluation — train_smoke

Held-out seeds 0–1.

## Checkpoints (PyTorch, GPU, execute 50 actions per chunk)

| step | success | 95% CI |
|---|---|---|
| 40 | 0/2 | (0.0, 0.658) |

## Variants on step 40

| variant | success | 95% CI | policy ms/step (mean) |
|---|---|---|---|
| torch_exec50 | 0/2 | (0.0, 0.658) | 1.9 |
| torch_exec10 | 0/2 | (0.0, 0.658) | 3.4 |
| torch_ensemble | 0/2 | (0.0, 0.658) | 21.1 |
| ov_fp32_exec50 | 0/2 | (0.0, 0.658) | 3.2 |
| ov_int8_exec50 | 0/2 | (0.0, 0.658) | 2.2 |
| ov_int8_exec10 | 0/2 | (0.0, 0.658) | 4.6 |
