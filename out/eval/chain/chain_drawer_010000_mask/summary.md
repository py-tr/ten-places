# chain_drawer_010000_mask: seeds 100-119 (20), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'drawer': 'out/train/chain_t1_drawer/drawer/checkpoints/010000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 12/20** (Wilson 95% (0.387, 0.781)), mean steps 4.20

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 18/20 | 0 | 0 | 0 |
| spoon | 16/20 | 17 | 0 | 0 |
| plate | 18/20 | 17 | 0 | 0 |
| fork | 12/20 | 12 | 0 | 0 |
| cup | 20/20 | 20 | 0 | 0 |

first failed step: none 12, drawer 2, fork 2, spoon 2, plate 2

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 101 | PASS | 5 | 8.24 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 102 | PASS | 5 | 7.47 | ok b300 | ok c331 | ok c161 | ok c301 | ok c171 | 0 |
| 103 | PASS | 5 | 7.71 | ok b300 | ok c331 | ok c151 | ok c311 | ok c171 | 0 |
| 104 | fail | 2 | 5.8 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 105 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 106 | fail | 4 | 8.22 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 107 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 108 | PASS | 5 | 7.72 | ok b300 | ok c321 | ok c161 | ok c321 | ok c191 | 0 |
| 109 | fail | 2 | 5.71 | X b300 p0.00 | X b450 p0.00 | ok b210 | X b500 p0.01 | ok c151 | 0 |
| 110 | PASS | 5 | 9.18 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 111 | PASS | 5 | 8.56 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | fail | 3 | 6.36 | ok b300 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 113 | fail | 3 | 8.35 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c151 | 0 |
| 114 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 115 | PASS | 5 | 9.04 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 116 | fail | 3 | 8.34 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c151 | 0 |
| 117 | fail | 4 | 8.92 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 118 | fail | 3 | 7.1 | ok b300 | X c331 p0.00 | ok c161 | X b500 p0.00 | ok c151 | 0 |
| 119 | PASS | 5 | 8.52 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
