# home_hold12: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=12 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 32/50** (Wilson 95% (0.501, 0.759)), mean steps 4.20

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 43/50 | 0 | 0 | 0 |
| spoon | 38/50 | 38 | 0 | 0 |
| plate | 46/50 | 47 | 0 | 0 |
| fork | 37/50 | 40 | 0 | 0 |
| cup | 46/50 | 46 | 0 | 0 |

first failed step: none 32, drawer 7, spoon 5, fork 3, plate 2, cup 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.14 | ok b300 | ok c311 | ok c161 | ok c331 | ok c151 | 0 |
| 101 | fail | 3 | 6.58 | ok b300 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 102 | fail | 4 | 9.73 | ok b300 | X b450 p0.00 | ok c161 | ok c301 | ok c181 | 0 |
| 103 | PASS | 5 | 8.44 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 104 | fail | 3 | 7.0 | ok b300 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c181 | 0 |
| 105 | PASS | 5 | 8.81 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 106 | fail | 4 | 7.89 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 107 | PASS | 5 | 9.04 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 108 | PASS | 5 | 8.99 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 109 | PASS | 5 | 8.19 | ok b300 | ok c321 | ok c161 | ok c341 | ok c161 | 0 |
| 110 | PASS | 5 | 9.34 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 111 | fail | 4 | 8.65 | ok b300 | ok c321 | ok c161 | X c351 p0.00 | ok c181 | 0 |
| 112 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | fail | 0 | 3.3 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 115 | PASS | 5 | 8.8 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 116 | PASS | 5 | 8.54 | ok b300 | ok c311 | ok c151 | ok c311 | ok c151 | 0 |
| 117 | PASS | 5 | 9.16 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 9.11 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 119 | PASS | 5 | 8.99 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 120 | fail | 2 | 4.76 | X b300 p0.00 | X b450 p0.00 | ok c151 | X c321 p0.01 | ok c161 | 0 |
| 121 | PASS | 5 | 8.85 | ok b300 | ok c321 | ok c171 | ok c321 | ok c181 | 0 |
| 122 | fail | 4 | 7.29 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 123 | fail | 2 | 5.47 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 124 | PASS | 5 | 8.32 | ok b300 | ok c321 | ok c151 | ok c311 | ok c151 | 0 |
| 125 | PASS | 5 | 8.82 | ok b300 | ok c321 | ok c161 | ok c311 | ok c191 | 0 |
| 126 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 127 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c161 | ok c311 | ok c201 | 0 |
| 128 | fail | 2 | 5.74 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.01 | ok c141 | 0 |
| 129 | fail | 4 | 9.06 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 0 |
| 130 | fail | 1 | 5.94 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c251 | 0 |
| 131 | fail | 2 | 5.63 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c151 | 0 |
| 132 | PASS | 5 | 8.73 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 133 | PASS | 5 | 9.09 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 134 | PASS | 5 | 9.42 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 135 | PASS | 5 | 8.74 | ok b300 | ok c321 | ok c151 | ok c311 | ok c151 | 0 |
| 136 | fail | 4 | 8.77 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c171 | 0 |
| 137 | fail | 2 | 5.87 | X b300 p0.01 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 138 | PASS | 5 | 8.58 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 3 | 8.91 | ok b300 | ok c311 | X c191 p0.97 | ok c301 | X b230 p0.00 | 0 |
| 140 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 141 | PASS | 5 | 9.06 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 142 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | PASS | 5 | 8.1 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 144 | PASS | 5 | 8.93 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 146 | PASS | 5 | 9.09 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | fail | 2 | 9.71 | ok b300 | ok c311 | X b210 p0.00 | X c351 p0.96 | X b230 p0.00 | 0 |
| 148 | fail | 4 | 8.03 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c161 | 0 |
| 149 | PASS | 5 | 8.21 | ok b300 | ok c311 | ok c151 | ok c301 | ok c151 | 0 |
