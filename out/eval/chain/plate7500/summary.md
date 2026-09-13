# plate7500: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 31/50** (Wilson 95% (0.482, 0.741)), mean steps 4.18

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 44/50 | 0 | 0 | 0 |
| spoon | 37/50 | 38 | 0 | 0 |
| plate | 44/50 | 46 | 0 | 0 |
| fork | 36/50 | 40 | 0 | 0 |
| cup | 48/50 | 48 | 0 | 0 |

first failed step: none 31, spoon 7, drawer 6, plate 3, fork 3

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.04 | ok b300 | ok c311 | ok c161 | ok c331 | ok c161 | 0 |
| 101 | fail | 2 | 5.62 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c181 | 0 |
| 102 | fail | 3 | 9.7 | ok b300 | ok c311 | X c151 p1.00 | ok c311 | X b230 p0.00 | 0 |
| 103 | PASS | 5 | 8.44 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 104 | fail | 4 | 7.54 | ok b300 | X b450 p0.01 | ok c151 | ok c341 | ok c181 | 0 |
| 105 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | fail | 4 | 7.87 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 107 | PASS | 5 | 9.04 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 108 | PASS | 5 | 8.79 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 109 | PASS | 5 | 8.21 | ok b300 | ok c321 | ok c161 | ok c341 | ok c151 | 0 |
| 110 | PASS | 5 | 9.34 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 111 | fail | 4 | 8.65 | ok b300 | ok c321 | ok c151 | X c351 p0.00 | ok c181 | 0 |
| 112 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c141 | ok c311 | ok c171 | 0 |
| 114 | fail | 1 | 1.96 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c221 | 0 |
| 115 | PASS | 5 | 8.78 | ok b300 | ok c321 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.6 | ok b300 | ok c311 | ok c141 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 9.17 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 118 | PASS | 5 | 9.1 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 9.0 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 120 | fail | 3 | 6.47 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 121 | PASS | 5 | 8.89 | ok b300 | ok c321 | ok c171 | ok c311 | ok c181 | 0 |
| 122 | fail | 3 | 6.77 | ok b300 | X c331 p0.00 | ok c141 | X b500 p0.00 | ok c181 | 0 |
| 123 | fail | 1 | 5.38 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | X b230 p0.00 | 0 |
| 124 | PASS | 5 | 8.32 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 125 | PASS | 5 | 8.81 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 127 | PASS | 5 | 8.4 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 128 | fail | 2 | 5.72 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 129 | PASS | 5 | 9.05 | ok b300 | ok c311 | ok c151 | ok c301 | ok c211 | 0 |
| 130 | fail | 2 | 6.17 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c221 | 0 |
| 131 | fail | 2 | 5.65 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 132 | PASS | 5 | 8.72 | ok b300 | ok c311 | ok c161 | ok c311 | ok c201 | 0 |
| 133 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 9.42 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 135 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.76 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 137 | fail | 2 | 5.68 | X b300 p0.01 | X b450 p0.00 | ok c151 | X c311 p0.82 | ok c171 | 0 |
| 138 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 3 | 8.93 | ok b300 | ok c311 | X b210 p0.00 | X c341 p0.20 | ok c171 | 0 |
| 140 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 141 | PASS | 5 | 9.07 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 142 | fail | 4 | 8.47 | ok b300 | ok c311 | ok c151 | X c301 p0.00 | ok c161 | 0 |
| 143 | fail | 2 | 6.11 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 0 |
| 144 | PASS | 5 | 8.94 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 146 | PASS | 5 | 9.04 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.71 | ok b300 | ok c311 | ok c151 | ok c291 | ok c151 | 0 |
| 148 | fail | 4 | 7.95 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 149 | fail | 4 | 8.14 | ok b300 | ok c311 | X c151 p1.00 | ok c321 | ok c161 | 0 |
