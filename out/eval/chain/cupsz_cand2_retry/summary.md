# cupsz_cand2_retry: seeds 100-149 (50), backend torch/cuda

levers: retry=['plate', 'fork', 'cup'] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'cup': 'out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 43/50** (Wilson 95% (0.738, 0.93)), mean steps 4.76

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 48/50 | 0 | 0 | 0 |
| spoon | 46/50 | 46 | 0 | 0 |
| plate | 49/50 | 49 | 2 | 1 |
| fork | 46/50 | 49 | 7 | 2 |
| cup | 49/50 | 49 | 2 | 1 |

first failed step: none 43, drawer 2, spoon 2, fork 1, plate 1, cup 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.01 | ok b300 | ok c311 | ok c161 | ok c371 | ok c161 | 1 |
| 101 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 102 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 103 | PASS | 5 | 8.02 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 104 | PASS | 5 | 8.02 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 105 | PASS | 5 | 8.56 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | PASS | 5 | 7.53 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 1 |
| 107 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 108 | PASS | 5 | 7.6 | ok b300 | ok c321 | ok c161 | ok c321 | ok c191 | 0 |
| 109 | fail | 2 | 5.91 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c301 p0.61 | ok c161 | 1 |
| 110 | fail | 4 | 8.92 | ok b300 | ok c311 | ok c161 | X c341 p0.09 | ok c171 | 1 |
| 111 | PASS | 5 | 8.6 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | PASS | 5 | 8.08 | ok b300 | ok c311 | ok c161 | ok c321 | ok c171 | 0 |
| 113 | PASS | 5 | 8.51 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | PASS | 5 | 8.74 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 115 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.25 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 118 | PASS | 5 | 8.56 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 119 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 120 | PASS | 5 | 8.56 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 121 | PASS | 5 | 7.61 | ok b300 | ok c311 | ok c181 | ok c321 | ok c181 | 1 |
| 122 | PASS | 5 | 8.49 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 123 | PASS | 5 | 7.95 | ok b300 | ok c321 | ok c161 | ok c361 | ok c161 | 1 |
| 124 | PASS | 5 | 8.29 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | PASS | 5 | 8.51 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 127 | PASS | 5 | 8.2 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 128 | PASS | 5 | 7.91 | ok b300 | ok c311 | ok c161 | ok c321 | ok c171 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c241 | 0 |
| 130 | PASS | 5 | 8.29 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 1 |
| 131 | fail | 2 | 5.82 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 1 |
| 132 | PASS | 5 | 8.6 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c251 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 135 | PASS | 5 | 8.66 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 136 | fail | 4 | 8.63 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c161 | 0 |
| 137 | PASS | 5 | 8.31 | ok b300 | ok c321 | ok c161 | ok c331 | ok c171 | 0 |
| 138 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 139 | fail | 3 | 8.8 | ok b300 | ok c311 | X b210 p0.00 | X c371 p0.14 | ok c161 | 2 |
| 140 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | PASS | 5 | 8.63 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 142 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | fail | 4 | 8.45 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.04 | 1 |
| 144 | PASS | 5 | 8.87 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 145 | PASS | 5 | 8.99 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 146 | PASS | 5 | 8.73 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c151 | ok c311 | ok c151 | 0 |
| 148 | fail | 4 | 8.05 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c181 | 0 |
| 149 | PASS | 5 | 8.54 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
