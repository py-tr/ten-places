# chain_drawer_007500: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t1_drawer/drawer/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 5/50** (Wilson 95% (0.043, 0.214)), mean steps 1.80

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 7/50 | 0 | 0 | 0 |
| spoon | 5/50 | 5 | 0 | 0 |
| plate | 28/50 | 28 | 0 | 0 |
| fork | 5/50 | 13 | 0 | 0 |
| cup | 45/50 | 46 | 0 | 0 |

first failed step: drawer 43, none 5, spoon 2

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | fail | 1 | 1.4 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 101 | fail | 2 | 0.41 | X b300 p0.00 | X b450 p0.00 | ok c171 | X b500 p0.00 | ok c181 | 0 |
| 102 | fail | 3 | 6.41 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 103 | fail | 2 | 5.22 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 104 | fail | 1 | 0.52 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 0 |
| 105 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 106 | fail | 0 | 5.32 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X c151 p0.71 | 0 |
| 107 | PASS | 5 | 7.96 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 108 | PASS | 5 | 7.61 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 109 | fail | 2 | 0.86 | X b300 p0.00 | X b450 p0.00 | ok c181 | X c371 p0.04 | ok c161 | 0 |
| 110 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.05 | ok c171 | 0 |
| 111 | fail | 3 | 5.93 | ok b300 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c191 | 0 |
| 112 | fail | 2 | 0.95 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c81 p0.00 | ok c171 | 0 |
| 113 | fail | 1 | 4.09 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 114 | fail | 1 | 1.28 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c251 | 0 |
| 115 | fail | 1 | 0.52 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 116 | fail | 1 | 0.64 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c201 | 0 |
| 117 | fail | 1 | 1.23 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 118 | fail | 2 | 1.0 | X b300 p0.00 | X b450 p0.00 | ok c171 | X b500 p0.00 | ok c161 | 0 |
| 119 | fail | 2 | 0.38 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c191 | 0 |
| 120 | fail | 2 | 4.81 | X b300 p0.00 | X b450 p0.00 | ok c151 | X c321 p0.04 | ok c161 | 0 |
| 121 | fail | 1 | 0.25 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c191 | 0 |
| 122 | fail | 0 | 5.17 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 123 | fail | 1 | 0.17 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c191 | 0 |
| 124 | fail | 1 | 0.38 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 125 | fail | 2 | 1.29 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c191 | 0 |
| 126 | fail | 2 | 4.23 | X b300 p0.00 | X b450 p0.00 | ok c151 | X c311 p0.98 | ok c171 | 0 |
| 127 | fail | 2 | 5.5 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c191 | 0 |
| 128 | fail | 2 | 0.67 | X b300 p0.01 | X b450 p0.00 | ok c201 | X b500 p0.00 | ok c171 | 0 |
| 129 | fail | 2 | 4.27 | X b300 p0.00 | X b450 p0.00 | ok c151 | X c301 p0.96 | ok c191 | 0 |
| 130 | fail | 1 | 4.6 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c221 | 0 |
| 131 | fail | 2 | 0.47 | X b300 p0.00 | X b450 p0.00 | ok c171 | X c351 p0.67 | ok c151 | 0 |
| 132 | fail | 2 | 4.29 | X b300 p0.00 | X b450 p0.00 | ok c171 | X c321 p0.99 | ok c181 | 0 |
| 133 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c191 | X b500 p0.00 | ok c181 | 0 |
| 134 | fail | 2 | 0.01 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c181 | 0 |
| 135 | PASS | 5 | 7.47 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 136 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 137 | fail | 2 | 0.62 | X b300 p0.02 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 138 | fail | 1 | 1.93 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 139 | fail | 1 | 1.04 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 140 | fail | 1 | 1.21 | X b300 p0.00 | X b450 p0.00 | ok b210 | X c321 p1.00 | X b230 p0.00 | 0 |
| 141 | fail | 1 | -0.01 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 142 | fail | 2 | 4.53 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 143 | fail | 1 | 0.65 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 0 |
| 144 | fail | 2 | 1.15 | X b300 p0.00 | X b450 p0.00 | ok c181 | X b500 p0.00 | ok c171 | 0 |
| 145 | fail | 0 | 2.79 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 146 | PASS | 5 | 8.61 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 147 | fail | 1 | 5.98 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 148 | fail | 0 | -0.01 | X b300 p0.00 | X b450 p0.00 | X c191 p1.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 149 | fail | 2 | 0.02 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 0 |
