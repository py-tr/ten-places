# cupsz_cand_retry: seeds 100-149 (50), backend torch/cuda

levers: retry=['plate', 'fork', 'cup'] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'cup': 'out/train/cup_sizes/cup/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 22/50** (Wilson 95% (0.312, 0.577)), mean steps 4.26

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 48/50 | 0 | 0 | 0 |
| spoon | 44/50 | 45 | 0 | 0 |
| plate | 48/50 | 48 | 3 | 1 |
| fork | 42/50 | 46 | 10 | 2 |
| cup | 31/50 | 34 | 20 | 1 |

first failed step: none 22, cup 17, fork 4, spoon 4, drawer 2, plate 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 7.87 | ok b300 | ok c311 | ok c161 | ok c381 | ok c161 | 1 |
| 101 | fail | 4 | 8.4 | ok b300 | ok c321 | ok c161 | ok c321 | X b230 p0.00 | 1 |
| 102 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 103 | PASS | 5 | 8.02 | ok b300 | ok c311 | ok c151 | ok c311 | ok c221 | 0 |
| 104 | fail | 4 | 8.05 | ok b300 | ok c321 | ok c161 | ok c321 | X b230 p0.00 | 1 |
| 105 | fail | 4 | 8.63 | ok b300 | ok c311 | ok c161 | ok c311 | X c71 p1.00 | 1 |
| 106 | fail | 4 | 8.18 | ok b300 | ok c321 | ok c151 | X c81 p0.00 | ok c221 | 1 |
| 107 | fail | 4 | 8.71 | ok b300 | ok c311 | ok c161 | X c361 p0.00 | ok c161 | 1 |
| 108 | PASS | 5 | 7.61 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 109 | fail | 2 | 5.92 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c191 | 1 |
| 110 | fail | 4 | 8.91 | ok b300 | ok c311 | ok c161 | X c391 p0.01 | ok c131 | 1 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 112 | fail | 4 | 8.07 | ok b300 | ok c311 | ok c151 | ok c321 | X b230 p0.00 | 1 |
| 113 | fail | 4 | 8.52 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.00 | 1 |
| 114 | PASS | 5 | 8.77 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 115 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 116 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | fail | 4 | 8.75 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 118 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | fail | 4 | 8.58 | ok b300 | X c321 p1.00 | ok c151 | ok c311 | ok c181 | 0 |
| 120 | PASS | 5 | 8.57 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 121 | fail | 2 | 7.34 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 2 |
| 122 | PASS | 5 | 8.49 | ok b300 | ok c311 | ok c151 | ok c321 | ok c221 | 0 |
| 123 | fail | 3 | 7.95 | ok b300 | ok c321 | ok c161 | X b500 p0.00 | X b230 p0.00 | 2 |
| 124 | PASS | 5 | 8.33 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c191 | 0 |
| 126 | fail | 4 | 8.51 | ok b300 | ok c311 | ok c161 | ok c301 | X c71 p1.00 | 1 |
| 127 | fail | 4 | 8.21 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 128 | fail | 4 | 8.07 | ok b300 | ok c321 | ok c161 | ok c321 | X b230 p0.01 | 1 |
| 129 | fail | 4 | 8.65 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 130 | fail | 4 | 8.27 | ok b300 | ok c321 | ok c151 | ok c321 | X b230 p0.00 | 2 |
| 131 | fail | 2 | 5.89 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 1 |
| 132 | fail | 4 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | fail | 4 | 8.91 | ok b300 | ok c321 | ok c151 | ok c311 | X b230 p0.00 | 1 |
| 135 | PASS | 5 | 8.69 | ok b300 | ok c321 | ok c151 | ok c311 | ok c71 | 1 |
| 136 | fail | 4 | 8.62 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c151 | 1 |
| 137 | fail | 4 | 8.34 | ok b300 | ok c311 | ok c161 | ok c331 | X b230 p0.00 | 1 |
| 138 | fail | 4 | 8.17 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.00 | 1 |
| 139 | fail | 3 | 8.8 | ok b300 | ok c311 | X b210 p0.00 | X c341 p0.93 | ok c161 | 1 |
| 140 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | fail | 4 | 8.58 | ok b300 | ok c321 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 142 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | PASS | 5 | 8.44 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 144 | PASS | 5 | 8.87 | ok b300 | ok c321 | ok c171 | ok c311 | ok c171 | 0 |
| 145 | PASS | 5 | 9.0 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 146 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 147 | PASS | 5 | 9.07 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 148 | fail | 3 | 8.1 | ok b300 | X b450 p0.10 | ok c151 | ok c321 | X c71 p0.87 | 2 |
| 149 | fail | 4 | 8.62 | ok b300 | ok c321 | ok c151 | ok c311 | X b230 p0.00 | 1 |
