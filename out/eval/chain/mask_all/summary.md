# mask_all: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 34/50** (Wilson 95% (0.542, 0.792)), mean steps 4.40

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 48/50 | 0 | 0 | 0 |
| spoon | 45/50 | 45 | 0 | 0 |
| plate | 43/50 | 42 | 0 | 0 |
| fork | 38/50 | 42 | 0 | 0 |
| cup | 46/50 | 47 | 0 | 0 |

first failed step: none 34, fork 5, plate 5, spoon 3, drawer 2, cup 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.22 | ok b300 | ok c311 | ok c161 | ok c331 | ok c171 | 0 |
| 101 | PASS | 5 | 8.31 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 102 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 103 | PASS | 5 | 8.02 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 104 | PASS | 5 | 7.91 | ok b300 | ok c311 | ok c161 | ok c321 | ok c181 | 0 |
| 105 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 106 | fail | 4 | 8.1 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 107 | fail | 4 | 8.71 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 108 | PASS | 5 | 7.6 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 109 | fail | 1 | 6.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 110 | fail | 4 | 8.92 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 112 | PASS | 5 | 8.12 | ok b300 | ok c311 | ok c161 | ok c321 | ok c171 | 0 |
| 113 | PASS | 5 | 8.52 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | PASS | 5 | 8.77 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 115 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 116 | fail | 3 | 8.31 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c151 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 119 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 120 | PASS | 5 | 8.56 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 121 | PASS | 5 | 7.73 | ok b300 | ok c311 | ok b210 | ok c321 | ok c201 | 0 |
| 122 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 123 | fail | 4 | 7.95 | ok b300 | ok c321 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 124 | PASS | 5 | 8.29 | ok b300 | ok c321 | ok c151 | ok c311 | ok c151 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | PASS | 5 | 8.51 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 127 | PASS | 5 | 8.16 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 128 | PASS | 5 | 8.07 | ok b300 | ok c321 | ok c161 | ok c311 | ok c141 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 130 | fail | 3 | 8.29 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c221 | 0 |
| 131 | fail | 2 | 5.84 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c151 | 0 |
| 132 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c171 | ok c311 | ok c181 | 0 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 135 | PASS | 5 | 8.68 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.63 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 137 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c161 | ok c341 | ok c161 | 0 |
| 138 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 139 | fail | 4 | 8.78 | ok b300 | ok c311 | ok b210 | X c391 p0.27 | ok c191 | 0 |
| 140 | fail | 2 | 8.66 | ok b300 | ok c311 | X b210 p0.00 | X c321 p0.05 | X b230 p0.00 | 0 |
| 141 | fail | 4 | 8.58 | ok b300 | ok c321 | ok c161 | ok c321 | X b230 p0.00 | 0 |
| 142 | fail | 4 | 8.19 | ok b300 | ok c311 | X b210 p0.00 | ok c311 | ok c151 | 0 |
| 143 | fail | 2 | 8.44 | ok b300 | ok c311 | X b210 p0.00 | X c341 p0.45 | X c71 p0.01 | 0 |
| 144 | PASS | 5 | 8.86 | ok b300 | ok c321 | ok c171 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 9.0 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 146 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 148 | fail | 4 | 8.07 | ok b300 | X b450 p0.00 | ok c161 | ok c321 | ok c241 | 0 |
| 149 | fail | 1 | 8.66 | ok b300 | X b450 p0.00 | X c161 p1.00 | X c271 p0.85 | X b230 p0.00 | 0 |
