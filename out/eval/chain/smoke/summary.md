# smoke: seeds 100-102 (3), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True}

**full tables 1/3** (Wilson 95% (0.061, 0.792)), mean steps 3.67

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 2/3 | 0 | 0 | 0 |
| spoon | 1/3 | 1 | 0 | 0 |
| plate | 3/3 | 3 | 0 | 0 |
| fork | 2/3 | 2 | 0 | 0 |
| cup | 3/3 | 3 | 0 | 0 |

first failed step: none 1, drawer 1, spoon 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.05 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 101 | fail | 2 | 5.27 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 102 | fail | 4 | 9.95 | ok b300 | X b450 p0.00 | ok c161 | ok c301 | ok c181 | 0 |
