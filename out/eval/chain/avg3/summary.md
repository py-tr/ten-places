# avg3: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={'spoon': ['out/train/cutlery_t1/spoon/checkpoints/007500/pretrained_model', 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'], 'fork': ['out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'out/train/cutlery_t1/fork/checkpoints/010000/pretrained_model'], 'plate': ['out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'out/train/plate_t1/plate/checkpoints/007500/pretrained_model']} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 28/50** (Wilson 95% (0.423, 0.688)), mean steps 4.14

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 45/50 | 0 | 0 | 0 |
| spoon | 36/50 | 37 | 0 | 0 |
| plate | 46/50 | 46 | 0 | 0 |
| fork | 37/50 | 37 | 0 | 0 |
| cup | 43/50 | 44 | 0 | 0 |

first failed step: none 28, spoon 9, drawer 5, cup 4, fork 3, plate 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 7.85 | ok b300 | ok c311 | ok c161 | ok c331 | ok c151 | 0 |
| 101 | fail | 2 | 5.22 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 102 | PASS | 5 | 9.57 | ok b300 | ok c311 | ok c151 | ok c301 | ok c231 | 0 |
| 103 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 104 | fail | 4 | 8.01 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c171 | 0 |
| 105 | PASS | 5 | 8.8 | ok b300 | ok c301 | ok c161 | ok c311 | ok c151 | 0 |
| 106 | fail | 4 | 7.88 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 107 | PASS | 5 | 9.04 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 108 | PASS | 5 | 8.76 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 109 | PASS | 5 | 8.14 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 110 | fail | 4 | 9.34 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 111 | PASS | 5 | 8.65 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 112 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c321 | ok c151 | ok c301 | ok c161 | 0 |
| 114 | fail | 0 | 5.45 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 115 | PASS | 5 | 8.86 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | fail | 3 | 8.45 | ok b300 | ok c311 | X b210 p0.00 | ok c311 | X c81 p0.09 | 0 |
| 117 | PASS | 5 | 9.15 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 9.06 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 119 | PASS | 5 | 9.0 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 120 | fail | 3 | 6.48 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 121 | PASS | 5 | 8.85 | ok b300 | ok c311 | ok c171 | ok c321 | ok c181 | 0 |
| 122 | fail | 3 | 6.26 | ok b300 | X c331 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 123 | fail | 2 | 5.47 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 124 | PASS | 5 | 8.34 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 125 | PASS | 5 | 8.81 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 126 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 127 | fail | 4 | 8.54 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.00 | 0 |
| 128 | fail | 3 | 8.32 | ok b300 | X b450 p0.03 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 129 | fail | 4 | 9.06 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 0 |
| 130 | fail | 1 | 6.35 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 131 | fail | 2 | 5.68 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 0 |
| 132 | fail | 4 | 8.73 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 0 |
| 133 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 134 | PASS | 5 | 9.42 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 135 | PASS | 5 | 8.74 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 136 | fail | 4 | 8.78 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c171 | 0 |
| 137 | fail | 2 | 5.88 | X b300 p0.01 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 138 | PASS | 5 | 8.58 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 4 | 8.91 | ok b300 | X b450 p0.00 | ok c141 | ok c301 | ok c181 | 0 |
| 140 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | fail | 4 | 9.06 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.00 | 0 |
| 142 | fail | 4 | 8.45 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c161 | 0 |
| 143 | fail | 2 | 6.16 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 0 |
| 144 | PASS | 5 | 8.86 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 146 | PASS | 5 | 9.06 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 147 | PASS | 5 | 9.7 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 148 | fail | 4 | 7.94 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 149 | PASS | 5 | 8.15 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
