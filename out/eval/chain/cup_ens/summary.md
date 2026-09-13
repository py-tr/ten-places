# cup_ens: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'temporal_coeff': 0.01}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 31/50** (Wilson 95% (0.482, 0.741)), mean steps 4.06

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 44/50 | 0 | 0 | 0 |
| spoon | 37/50 | 38 | 0 | 0 |
| plate | 47/50 | 46 | 0 | 0 |
| fork | 35/50 | 38 | 0 | 0 |
| cup | 40/50 | 40 | 0 | 0 |

first failed step: none 31, spoon 7, drawer 6, fork 3, plate 2, cup 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 101 | fail | 1 | 5.47 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | X b230 p0.00 | 0 |
| 102 | PASS | 5 | 9.66 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 103 | PASS | 5 | 8.51 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 104 | fail | 4 | 7.85 | ok b300 | X b450 p0.01 | ok c161 | ok c321 | ok c161 | 0 |
| 105 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 106 | fail | 3 | 7.99 | ok b300 | ok c321 | ok c151 | X b500 p0.00 | X b230 p0.00 | 0 |
| 107 | PASS | 5 | 9.04 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 108 | PASS | 5 | 8.76 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 109 | fail | 2 | 7.34 | ok b300 | X b450 p0.00 | ok c161 | X b500 p0.00 | X b230 p0.00 | 0 |
| 110 | PASS | 5 | 9.34 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 111 | fail | 4 | 8.65 | ok b300 | ok c321 | ok c161 | X c351 p0.00 | ok c161 | 0 |
| 112 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 114 | fail | 0 | 2.85 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 115 | fail | 2 | 7.15 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | X b230 p0.00 | 0 |
| 116 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 9.17 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 8.94 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 119 | PASS | 5 | 9.0 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 120 | fail | 3 | 6.23 | ok b300 | X b450 p0.00 | ok c151 | X c311 p0.50 | ok c161 | 0 |
| 121 | PASS | 5 | 8.86 | ok b300 | ok c321 | ok c171 | ok c321 | ok c161 | 0 |
| 122 | fail | 2 | 6.93 | ok b300 | X c331 p0.00 | ok c151 | X b500 p0.00 | X b230 p0.00 | 0 |
| 123 | fail | 2 | 5.47 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c191 | 0 |
| 124 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 125 | PASS | 5 | 8.82 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 126 | fail | 3 | 8.64 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | X b230 p0.00 | 0 |
| 127 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 128 | PASS | 5 | 8.3 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 129 | PASS | 5 | 9.05 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 130 | fail | 1 | 5.88 | X b300 p0.00 | X b450 p0.00 | ok b210 | X b500 p0.00 | X b230 p0.00 | 0 |
| 131 | fail | 2 | 5.69 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 132 | PASS | 5 | 8.72 | ok b300 | ok c311 | ok c161 | ok c311 | ok c251 | 0 |
| 133 | PASS | 5 | 9.07 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 134 | PASS | 5 | 9.42 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 135 | PASS | 5 | 8.73 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 136 | fail | 4 | 8.8 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c161 | 0 |
| 137 | fail | 2 | 5.88 | X b300 p0.01 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 138 | fail | 4 | 8.6 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.00 | 0 |
| 139 | fail | 2 | 8.91 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 140 | PASS | 5 | 9.16 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 141 | PASS | 5 | 9.06 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 142 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | PASS | 5 | 8.07 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 144 | PASS | 5 | 8.89 | ok b300 | ok c321 | ok c161 | ok c341 | ok c161 | 0 |
| 145 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 146 | PASS | 5 | 9.05 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 147 | fail | 3 | 9.7 | ok b300 | ok c311 | X b210 p0.00 | X c331 p0.19 | ok c171 | 0 |
| 148 | fail | 4 | 7.96 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c161 | 0 |
| 149 | PASS | 5 | 8.09 | ok b300 | ok c311 | ok c151 | ok c311 | ok c151 | 0 |
