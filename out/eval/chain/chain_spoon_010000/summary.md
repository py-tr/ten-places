# chain_spoon_010000: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/chain_t1_spoon/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 14/50** (Wilson 95% (0.175, 0.417)), mean steps 3.74

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 45/50 | 0 | 0 | 0 |
| spoon | 22/50 | 27 | 0 | 0 |
| plate | 44/50 | 44 | 0 | 0 |
| fork | 29/50 | 28 | 0 | 0 |
| cup | 47/50 | 47 | 0 | 0 |

first failed step: spoon 28, none 14, drawer 5, plate 2, cup 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.02 | ok b300 | ok c311 | ok c161 | ok b500 | ok c161 | 0 |
| 101 | fail | 3 | 5.13 | X b300 p0.00 | ok b450 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 102 | fail | 4 | 9.99 | ok b300 | ok c311 | ok c161 | ok c291 | X b230 p0.00 | 0 |
| 103 | fail | 3 | 8.45 | ok b300 | X b450 p0.01 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 104 | fail | 3 | 7.11 | ok b300 | X b450 p0.00 | ok c171 | X b500 p0.00 | ok c171 | 0 |
| 105 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | fail | 3 | 7.75 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c191 | 0 |
| 107 | fail | 2 | 9.05 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 108 | fail | 4 | 8.99 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 0 |
| 109 | fail | 3 | 8.15 | ok b300 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 110 | PASS | 5 | 9.34 | ok b300 | ok c321 | ok c161 | ok c301 | ok c161 | 0 |
| 111 | fail | 3 | 8.65 | ok b300 | X c331 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 112 | fail | 4 | 8.65 | ok b300 | X b450 p0.00 | ok c161 | ok c321 | ok c181 | 0 |
| 113 | fail | 4 | 8.54 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c171 | 0 |
| 114 | fail | 2 | 4.76 | X b300 p0.01 | ok c351 | X b210 p0.00 | X b500 p0.00 | ok c251 | 0 |
| 115 | fail | 3 | 8.89 | ok b300 | X c331 p0.03 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 116 | fail | 3 | 8.49 | ok b300 | X b450 p0.05 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 117 | PASS | 5 | 9.17 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 118 | PASS | 5 | 9.02 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | fail | 4 | 8.98 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c171 | 0 |
| 120 | fail | 3 | 6.53 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c221 | 0 |
| 121 | PASS | 5 | 9.0 | ok b300 | ok c321 | ok c171 | ok c321 | ok c181 | 0 |
| 122 | fail | 3 | 6.97 | ok b300 | X c331 p0.00 | ok c151 | X c371 p0.00 | ok c161 | 0 |
| 123 | fail | 3 | 5.47 | X b300 p0.00 | ok c321 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 124 | PASS | 5 | 8.31 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 125 | PASS | 5 | 8.81 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | fail | 4 | 8.66 | ok b300 | X b450 p0.00 | ok c161 | ok c301 | ok c171 | 0 |
| 127 | fail | 4 | 8.4 | ok b300 | X b450 p0.00 | ok c171 | ok c321 | ok c211 | 0 |
| 128 | fail | 3 | 5.75 | X b300 p0.00 | ok c341 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 129 | fail | 2 | 9.05 | ok b300 | X c331 p0.33 | ok c151 | X b500 p0.00 | X b230 p0.00 | 0 |
| 130 | fail | 1 | 6.19 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 131 | fail | 3 | 5.55 | X b300 p0.00 | ok c321 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 132 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c311 | ok c191 | 0 |
| 133 | PASS | 5 | 9.09 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | fail | 4 | 9.42 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 0 |
| 135 | fail | 4 | 8.72 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c161 | 0 |
| 136 | PASS | 5 | 8.79 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 137 | fail | 3 | 6.12 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 138 | fail | 3 | 8.56 | ok b300 | X c331 p0.02 | ok c151 | X b500 p0.00 | ok c161 | 0 |
| 139 | fail | 4 | 8.94 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 140 | fail | 4 | 9.16 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c161 | 0 |
| 141 | fail | 3 | 9.06 | ok b300 | X b450 p0.00 | X b210 p0.00 | ok b500 | ok c161 | 0 |
| 142 | PASS | 5 | 8.47 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | fail | 3 | 6.05 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c191 | 0 |
| 144 | PASS | 5 | 8.92 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | fail | 3 | 8.66 | ok b300 | X c301 p1.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 146 | fail | 4 | 9.05 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c171 | 0 |
| 147 | fail | 4 | 9.7 | ok b300 | X b450 p0.00 | ok c171 | ok c311 | ok c161 | 0 |
| 148 | PASS | 5 | 7.85 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 149 | fail | 4 | 8.19 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c171 | 0 |
