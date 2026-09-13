# drawer_exec50: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'n_action_steps': 50}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 34/50** (Wilson 95% (0.542, 0.792)), mean steps 4.20

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 43/50 | 0 | 0 | 0 |
| spoon | 37/50 | 37 | 0 | 0 |
| plate | 43/50 | 43 | 0 | 0 |
| fork | 39/50 | 43 | 0 | 0 |
| cup | 48/50 | 48 | 0 | 0 |

first failed step: none 34, drawer 7, spoon 6, plate 2, cup 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.95 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 101 | PASS | 5 | 9.06 | ok b300 | ok c321 | ok c161 | ok c331 | ok c171 | 0 |
| 102 | PASS | 5 | 8.33 | ok b300 | ok c321 | ok c161 | ok c301 | ok c161 | 0 |
| 103 | PASS | 5 | 8.83 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 104 | fail | 1 | 3.26 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 105 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | fail | 3 | 6.34 | ok b300 | X b450 p0.00 | ok c151 | X c311 p0.88 | ok c181 | 0 |
| 107 | PASS | 5 | 9.06 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 108 | fail | 2 | 2.23 | X b300 p0.00 | X b450 p0.00 | ok c181 | X b500 p0.00 | ok c171 | 0 |
| 109 | PASS | 5 | 8.49 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 110 | PASS | 5 | 9.2 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 111 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | PASS | 5 | 8.76 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 113 | fail | 1 | 1.07 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 114 | PASS | 5 | 8.69 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 115 | PASS | 5 | 8.96 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 116 | PASS | 5 | 8.26 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 9.19 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | fail | 2 | 5.79 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c311 p0.01 | ok c161 | 0 |
| 119 | PASS | 5 | 8.56 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 120 | PASS | 5 | 9.0 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 121 | PASS | 5 | 7.61 | ok b300 | ok c311 | ok c181 | ok c321 | ok c191 | 0 |
| 122 | fail | 2 | 6.06 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c191 | 0 |
| 123 | fail | 4 | 10.0 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c161 | 0 |
| 124 | fail | 1 | 6.04 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 125 | PASS | 5 | 9.06 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 126 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 127 | PASS | 5 | 8.69 | ok b300 | ok c321 | ok c161 | ok c311 | ok c231 | 0 |
| 128 | fail | 4 | 9.77 | ok b300 | X b450 p0.00 | ok c161 | ok c331 | ok c171 | 0 |
| 129 | PASS | 5 | 8.95 | ok b300 | ok c311 | ok c151 | ok c301 | ok c221 | 0 |
| 130 | fail | 0 | 5.01 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 131 | fail | 2 | 2.53 | X b300 p0.00 | X b450 p0.00 | ok c171 | X b500 p0.00 | ok c171 | 0 |
| 132 | fail | 4 | 8.69 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 0 |
| 133 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 9.49 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 135 | PASS | 5 | 8.82 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 136 | fail | 4 | 8.92 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c171 | 0 |
| 137 | fail | 3 | 6.21 | ok b300 | X b450 p0.00 | ok c161 | X c331 p0.78 | ok c161 | 0 |
| 138 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 4 | 9.07 | ok b300 | ok c311 | X b210 p0.00 | ok c311 | ok c171 | 0 |
| 140 | PASS | 5 | 9.06 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | PASS | 5 | 9.0 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 142 | PASS | 5 | 8.29 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | PASS | 5 | 7.77 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 144 | PASS | 5 | 9.14 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 146 | PASS | 5 | 9.22 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | fail | 3 | 9.67 | ok b300 | ok c311 | X b210 p0.00 | X c321 p0.59 | ok c161 | 0 |
| 148 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 149 | PASS | 5 | 8.55 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
