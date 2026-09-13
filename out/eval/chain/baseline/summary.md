# baseline: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True}

**full tables 30/50** (Wilson 95% (0.462, 0.724)), mean steps 4.14

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 44/50 | 0 | 0 | 0 |
| spoon | 36/50 | 37 | 0 | 0 |
| plate | 43/50 | 44 | 0 | 0 |
| fork | 37/50 | 39 | 0 | 0 |
| cup | 47/50 | 47 | 0 | 0 |

first failed step: none 30, spoon 8, drawer 6, plate 3, fork 2, cup 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.02 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 101 | fail | 2 | 5.1 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 102 | fail | 4 | 9.92 | ok b300 | X b450 p0.00 | ok c161 | ok c301 | ok c181 | 0 |
| 103 | PASS | 5 | 8.51 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 104 | fail | 4 | 7.89 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c171 | 0 |
| 105 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | fail | 4 | 7.7 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 107 | PASS | 5 | 9.04 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 108 | PASS | 5 | 8.79 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 109 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c161 | ok c321 | ok c151 | 0 |
| 110 | PASS | 5 | 9.34 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 111 | fail | 4 | 8.68 | ok b300 | ok c321 | ok c161 | X c351 p0.00 | ok c181 | 0 |
| 112 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | fail | 2 | 5.15 | X b300 p0.02 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 115 | PASS | 5 | 8.79 | ok b300 | ok c321 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 9.16 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 9.03 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.99 | ok b300 | ok c321 | ok c151 | ok c321 | ok c171 | 0 |
| 120 | fail | 2 | 6.82 | ok b300 | X c341 p0.01 | X b210 p0.00 | X b500 p0.00 | ok c191 | 0 |
| 121 | PASS | 5 | 9.0 | ok b300 | ok c321 | ok c171 | ok c321 | ok c181 | 0 |
| 122 | fail | 0 | 5.59 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 123 | fail | 2 | 5.24 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 124 | PASS | 5 | 8.32 | ok b300 | ok c321 | ok c151 | ok c311 | ok c151 | 0 |
| 125 | PASS | 5 | 8.81 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 127 | PASS | 5 | 8.51 | ok b300 | ok c311 | ok c161 | ok c311 | ok c201 | 0 |
| 128 | fail | 4 | 9.65 | ok b300 | X b450 p0.00 | ok c161 | ok c301 | ok c161 | 0 |
| 129 | PASS | 5 | 9.05 | ok b300 | ok c311 | ok c151 | ok c301 | ok c211 | 0 |
| 130 | fail | 2 | 6.21 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c211 | 0 |
| 131 | fail | 2 | 5.66 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 132 | fail | 4 | 8.74 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 0 |
| 133 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 9.22 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 135 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.77 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 137 | fail | 2 | 5.99 | X b300 p0.01 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 138 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 3 | 8.92 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 140 | PASS | 5 | 9.11 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | PASS | 5 | 9.06 | ok b300 | ok c311 | ok c161 | ok c311 | ok c211 | 0 |
| 142 | PASS | 5 | 8.49 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | fail | 1 | 6.0 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 144 | PASS | 5 | 8.94 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 145 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 146 | PASS | 5 | 9.04 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 147 | fail | 3 | 9.7 | ok b300 | ok c311 | X b210 p0.00 | X c331 p0.05 | ok c161 | 0 |
| 148 | fail | 4 | 7.94 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 149 | fail | 4 | 8.16 | ok b300 | ok c311 | X c151 p1.00 | ok c321 | ok c151 | 0 |
