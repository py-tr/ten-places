# cupsz_base_retry: seeds 100-149 (50), backend torch/cuda

levers: retry=['plate', 'fork', 'cup'] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 42/50** (Wilson 95% (0.715, 0.917)), mean steps 4.70

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 48/50 | 0 | 0 | 0 |
| spoon | 47/50 | 47 | 0 | 0 |
| plate | 47/50 | 47 | 3 | 0 |
| fork | 46/50 | 46 | 5 | 1 |
| cup | 47/50 | 47 | 6 | 3 |

first failed step: none 42, cup 2, drawer 2, plate 2, fork 1, spoon 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c161 | ok b500 | ok c161 | 1 |
| 101 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 102 | fail | 4 | 8.64 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 103 | PASS | 5 | 8.02 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 104 | PASS | 5 | 8.04 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 105 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c151 | ok c321 | ok c181 | 0 |
| 107 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 108 | PASS | 5 | 7.59 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 109 | fail | 1 | 5.9 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c151 | 2 |
| 110 | PASS | 5 | 8.92 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 1 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 112 | PASS | 5 | 8.13 | ok b300 | ok c311 | ok c151 | ok c321 | ok c181 | 0 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | PASS | 5 | 8.78 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 115 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.27 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 8.56 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.58 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 120 | PASS | 5 | 8.57 | ok b300 | ok c321 | ok c151 | ok c321 | ok c171 | 0 |
| 121 | fail | 4 | 7.45 | ok b300 | ok c311 | X b210 p0.00 | ok c321 | ok c181 | 1 |
| 122 | PASS | 5 | 8.5 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 123 | fail | 4 | 7.95 | ok b300 | ok c321 | ok c161 | X b500 p0.00 | ok c171 | 2 |
| 124 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 127 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c161 | ok c311 | ok c231 | 1 |
| 128 | PASS | 5 | 7.99 | ok b300 | ok c311 | ok c161 | ok c321 | ok c151 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c211 | 1 |
| 130 | PASS | 5 | 8.24 | ok b300 | ok c311 | ok c151 | ok c321 | ok c201 | 0 |
| 131 | fail | 2 | 5.78 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 1 |
| 132 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 135 | PASS | 5 | 8.69 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 136 | fail | 4 | 8.63 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c171 | 0 |
| 137 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 138 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 2 | 8.79 | ok b300 | ok c311 | X b210 p0.00 | X c331 p0.54 | X b230 p0.01 | 2 |
| 140 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | fail | 4 | 8.59 | ok b300 | ok c321 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 142 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | PASS | 5 | 8.46 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 144 | PASS | 5 | 8.87 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.99 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 146 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.07 | ok b300 | ok c321 | ok c151 | ok c301 | ok c151 | 0 |
| 148 | PASS | 5 | 8.38 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 149 | PASS | 5 | 8.54 | ok b300 | ok c311 | ok c151 | ok c311 | ok c151 | 0 |
