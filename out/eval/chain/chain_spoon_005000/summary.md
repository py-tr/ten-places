# chain_spoon_005000: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/chain_t1_spoon/spoon/checkpoints/005000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 12/50** (Wilson 95% (0.143, 0.374)), mean steps 3.60

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 46/50 | 0 | 0 | 0 |
| spoon | 20/50 | 31 | 0 | 0 |
| plate | 45/50 | 45 | 0 | 0 |
| fork | 24/50 | 27 | 0 | 0 |
| cup | 45/50 | 45 | 0 | 0 |

first failed step: spoon 29, none 12, drawer 4, fork 2, plate 2, cup 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | fail | 3 | 8.22 | ok b300 | X b450 p0.01 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 101 | fail | 3 | 5.25 | X b300 p0.00 | ok b450 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 102 | fail | 4 | 9.95 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c201 | 0 |
| 103 | fail | 4 | 8.45 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 104 | fail | 4 | 7.68 | ok b300 | X b450 p0.00 | ok c161 | ok c341 | ok c171 | 0 |
| 105 | fail | 3 | 8.8 | ok b300 | X c331 p0.20 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 106 | fail | 3 | 7.84 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 107 | fail | 4 | 9.04 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 0 |
| 108 | fail | 3 | 8.79 | ok b300 | X c301 p0.82 | ok c171 | X b500 p0.00 | ok c181 | 0 |
| 109 | fail | 2 | 7.19 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 110 | PASS | 5 | 9.34 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 111 | fail | 4 | 8.66 | ok b300 | X b450 p0.00 | ok c161 | ok c301 | ok c181 | 0 |
| 112 | fail | 3 | 8.63 | ok b300 | X b450 p0.00 | ok c151 | X c321 p0.00 | ok c181 | 0 |
| 113 | fail | 4 | 8.55 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | fail | 1 | 3.25 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c241 | 0 |
| 115 | fail | 3 | 8.9 | ok b300 | X c331 p0.02 | ok c151 | X b500 p0.00 | ok c191 | 0 |
| 116 | PASS | 5 | 8.57 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | fail | 4 | 9.17 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 8.94 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | fail | 3 | 8.95 | ok b300 | X c311 p0.98 | ok c151 | X c331 p1.00 | ok c171 | 0 |
| 120 | fail | 4 | 6.74 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 121 | PASS | 5 | 8.85 | ok b300 | ok c341 | ok c171 | ok c311 | ok c181 | 0 |
| 122 | fail | 2 | 6.51 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 123 | fail | 3 | 5.47 | X b300 p0.00 | ok c331 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 124 | fail | 3 | 8.27 | ok b300 | X b450 p0.01 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 125 | fail | 3 | 8.81 | ok b300 | X c331 p0.76 | ok c161 | X b500 p0.00 | ok c191 | 0 |
| 126 | fail | 3 | 8.65 | ok b300 | X b450 p0.02 | ok c161 | X c331 p0.00 | ok c171 | 0 |
| 127 | fail | 4 | 8.41 | ok b300 | X b450 p0.00 | ok c161 | ok c321 | ok c251 | 0 |
| 128 | fail | 4 | 8.28 | ok b300 | X b450 p0.02 | ok c161 | ok c301 | ok c161 | 0 |
| 129 | fail | 2 | 9.05 | ok b300 | X c331 p0.01 | ok c151 | X b500 p0.00 | X b230 p0.00 | 0 |
| 130 | fail | 2 | 8.55 | ok b300 | X c301 p1.00 | ok c151 | X b500 p0.00 | X b230 p0.00 | 0 |
| 131 | fail | 3 | 5.67 | X b300 p0.00 | ok c331 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 132 | fail | 4 | 8.75 | ok b300 | ok c321 | ok c161 | ok c311 | X b230 p0.00 | 0 |
| 133 | fail | 3 | 9.08 | ok b300 | X c331 p0.07 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 134 | fail | 4 | 9.22 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c171 | 0 |
| 135 | fail | 4 | 8.72 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c161 | 0 |
| 136 | PASS | 5 | 8.77 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 137 | fail | 2 | 6.21 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 138 | PASS | 5 | 8.55 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 3 | 8.9 | ok b300 | X c331 p0.03 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 140 | fail | 3 | 9.17 | ok b300 | X c301 p1.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 141 | fail | 2 | 9.06 | ok b300 | X c331 p0.17 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 142 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | PASS | 5 | 8.1 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 144 | PASS | 5 | 8.93 | ok b300 | ok c321 | ok c171 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.65 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 146 | fail | 3 | 9.09 | ok b300 | X c351 p0.01 | ok c161 | X b500 p0.00 | ok c181 | 0 |
| 147 | fail | 4 | 9.7 | ok b300 | X b450 p0.00 | ok c161 | ok c301 | ok c161 | 0 |
| 148 | PASS | 5 | 8.0 | ok b300 | ok c311 | ok c151 | ok c321 | ok c161 | 0 |
| 149 | PASS | 5 | 8.1 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
