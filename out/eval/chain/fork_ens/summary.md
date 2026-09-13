# fork_ens: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'temporal_coeff': 0.01}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 30/50** (Wilson 95% (0.462, 0.724)), mean steps 4.20

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 46/50 | 0 | 0 | 0 |
| spoon | 36/50 | 38 | 0 | 0 |
| plate | 45/50 | 45 | 0 | 0 |
| fork | 35/50 | 37 | 0 | 0 |
| cup | 48/50 | 47 | 0 | 0 |

first failed step: none 30, spoon 10, fork 5, drawer 4, plate 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 101 | fail | 2 | 5.84 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 102 | fail | 4 | 9.68 | ok b300 | ok c331 | ok c151 | X b500 p0.00 | ok c201 | 0 |
| 103 | PASS | 5 | 8.54 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 104 | fail | 4 | 7.64 | ok b300 | X b450 p0.01 | ok c161 | ok c311 | ok c171 | 0 |
| 105 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | fail | 4 | 7.92 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 107 | PASS | 5 | 9.04 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 108 | PASS | 5 | 8.98 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 109 | PASS | 5 | 8.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 110 | fail | 4 | 9.34 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 111 | fail | 4 | 8.64 | ok b300 | ok c321 | ok c161 | X c351 p0.00 | ok c181 | 0 |
| 112 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | fail | 0 | 2.81 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 115 | PASS | 5 | 7.83 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 116 | PASS | 5 | 8.57 | ok b300 | ok c311 | ok c141 | ok c301 | ok c161 | 0 |
| 117 | PASS | 5 | 9.16 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 9.09 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.96 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 120 | fail | 3 | 6.33 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 121 | PASS | 5 | 8.92 | ok b300 | ok c321 | ok c171 | ok c311 | ok c181 | 0 |
| 122 | fail | 3 | 7.05 | ok b300 | X c331 p0.00 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 123 | fail | 2 | 5.22 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c161 | 0 |
| 124 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c151 | ok c301 | ok c151 | 0 |
| 125 | PASS | 5 | 8.82 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | fail | 4 | 8.65 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 127 | PASS | 5 | 8.44 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 128 | fail | 4 | 9.43 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c161 | 0 |
| 129 | PASS | 5 | 9.05 | ok b300 | ok c311 | ok c151 | ok c301 | ok c251 | 0 |
| 130 | fail | 2 | 6.14 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok b230 | 0 |
| 131 | fail | 2 | 5.71 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 132 | PASS | 5 | 8.72 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 133 | PASS | 5 | 9.09 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 9.42 | ok b300 | ok c321 | ok c151 | ok c321 | ok c181 | 0 |
| 135 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.77 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 137 | fail | 3 | 6.14 | ok b300 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 138 | PASS | 5 | 8.58 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 139 | fail | 4 | 8.92 | ok b300 | ok c311 | X b210 p0.00 | ok c311 | ok c161 | 0 |
| 140 | PASS | 5 | 9.15 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 141 | PASS | 5 | 9.07 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 142 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | fail | 2 | 6.02 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c191 | 0 |
| 144 | PASS | 5 | 8.87 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 146 | PASS | 5 | 9.05 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 147 | fail | 1 | 9.71 | ok b300 | X c311 p1.00 | X b210 p0.00 | X c351 p1.00 | X b230 p0.00 | 0 |
| 148 | fail | 4 | 7.94 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 149 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c301 | ok c151 | 0 |
