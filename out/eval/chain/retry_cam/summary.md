# retry_cam: seeds 100-149 (50), backend torch/cuda

levers: retry=['drawer', 'spoon', 'fork'] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=60 retry_camera_end=True

**full tables 31/50** (Wilson 95% (0.482, 0.741)), mean steps 4.26

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 44/50 | 1 | 12 | 1 |
| spoon | 38/50 | 38 | 12 | 0 |
| plate | 47/50 | 47 | 0 | 0 |
| fork | 39/50 | 46 | 12 | 3 |
| cup | 45/50 | 45 | 0 | 0 |

first failed step: none 31, drawer 6, spoon 6, cup 3, plate 3, fork 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.15 | ok b300 | ok c311 | ok c161 | ok c381 | ok c161 | 1 |
| 101 | fail | 2 | 5.86 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c371 p0.27 | ok c171 | 3 |
| 102 | PASS | 5 | 9.74 | ok b300 | ok c331 | ok c151 | ok c301 | ok c191 | 0 |
| 103 | PASS | 5 | 8.51 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 104 | fail | 4 | 7.08 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c171 | 2 |
| 105 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | PASS | 5 | 7.77 | ok b300 | ok c311 | ok c151 | ok c321 | ok c211 | 2 |
| 107 | PASS | 5 | 9.04 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 108 | PASS | 5 | 8.97 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 109 | PASS | 5 | 8.16 | ok b300 | ok c311 | ok c161 | ok c341 | ok c161 | 0 |
| 110 | PASS | 5 | 9.34 | ok b300 | ok c311 | ok c161 | ok c341 | ok c161 | 1 |
| 111 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | fail | 4 | 8.68 | ok b300 | ok c311 | ok c151 | ok c321 | X b230 p0.00 | 0 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | fail | 2 | 7.45 | ok c141 | ok c341 | X b210 p0.00 | X c431 p0.36 | X b230 p0.00 | 2 |
| 115 | PASS | 5 | 8.78 | ok b300 | ok c321 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.5 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 9.17 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 9.04 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.99 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 120 | fail | 2 | 4.87 | X b300 p0.00 | X b450 p0.00 | ok c151 | X c351 p0.00 | ok c171 | 3 |
| 121 | PASS | 5 | 8.92 | ok b300 | ok c321 | ok c171 | ok c311 | ok c181 | 0 |
| 122 | fail | 4 | 6.86 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c181 | 3 |
| 123 | fail | 2 | 5.45 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 3 |
| 124 | PASS | 5 | 8.31 | ok b300 | ok c321 | ok c151 | ok c311 | ok c151 | 0 |
| 125 | PASS | 5 | 8.81 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 127 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 128 | fail | 2 | 5.8 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c151 | 3 |
| 129 | fail | 4 | 9.04 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 0 |
| 130 | fail | 3 | 6.93 | ok b300 | X b450 p0.00 | ok c151 | X c371 p0.47 | ok c191 | 3 |
| 131 | fail | 2 | 5.69 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 3 |
| 132 | fail | 4 | 8.74 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 0 |
| 133 | PASS | 5 | 9.07 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 9.42 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 135 | PASS | 5 | 8.76 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 136 | fail | 4 | 8.77 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c171 | 1 |
| 137 | fail | 2 | 5.6 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c341 p0.63 | ok c161 | 2 |
| 138 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 3 | 8.9 | ok b300 | ok c311 | X b210 p0.00 | X c321 p0.88 | ok c151 | 0 |
| 140 | PASS | 5 | 9.17 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 141 | PASS | 5 | 9.05 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 142 | PASS | 5 | 8.46 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | fail | 3 | 7.21 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 3 |
| 144 | PASS | 5 | 8.95 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 146 | PASS | 5 | 9.04 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | fail | 3 | 9.7 | ok b300 | ok c311 | ok b210 | X c331 p0.72 | X b230 p0.00 | 0 |
| 148 | fail | 4 | 7.99 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c171 | 1 |
| 149 | fail | 4 | 8.19 | ok b300 | ok c311 | X c151 p1.00 | ok c331 | ok c161 | 0 |
