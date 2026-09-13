# retry_all: seeds 100-149 (50), backend torch/cuda

levers: retry=['drawer', 'spoon', 'plate', 'fork', 'cup'] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True}

**full tables 33/50** (Wilson 95% (0.522, 0.776)), mean steps 4.38

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 45/50 | 0 | 9 | 0 |
| spoon | 40/50 | 39 | 11 | 0 |
| plate | 47/50 | 46 | 7 | 3 |
| fork | 40/50 | 43 | 12 | 2 |
| cup | 47/50 | 47 | 4 | 1 |

first failed step: none 33, drawer 5, spoon 5, fork 3, cup 2, plate 2

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c161 | ok c381 | ok c161 | 1 |
| 101 | fail | 2 | 5.65 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 3 |
| 102 | PASS | 5 | 9.69 | ok b300 | ok c321 | ok c151 | ok c301 | ok c71 | 1 |
| 103 | PASS | 5 | 8.53 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 104 | PASS | 5 | 7.68 | ok b300 | ok c311 | ok c161 | ok c321 | ok c171 | 1 |
| 105 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | fail | 4 | 7.85 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c181 | 1 |
| 107 | PASS | 5 | 9.05 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 108 | PASS | 5 | 8.97 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 109 | PASS | 5 | 8.14 | ok b300 | ok c321 | ok c161 | ok c341 | ok c151 | 0 |
| 110 | PASS | 5 | 9.34 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 1 |
| 111 | fail | 4 | 8.66 | ok b300 | ok b450 | ok c161 | X c341 p0.00 | ok c181 | 2 |
| 112 | fail | 4 | 8.61 | ok b300 | ok c311 | ok c151 | ok c321 | X b230 p0.01 | 1 |
| 113 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 114 | fail | 2 | 3.52 | X b300 p0.03 | X b450 p0.00 | ok c181 | X b500 p0.00 | ok c171 | 4 |
| 115 | PASS | 5 | 8.82 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | fail | 4 | 8.46 | ok b300 | ok c311 | X b210 p0.00 | ok c311 | ok c151 | 1 |
| 117 | PASS | 5 | 9.16 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 119 | PASS | 5 | 8.98 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 120 | fail | 3 | 6.37 | ok b300 | X b450 p0.00 | ok c151 | X c311 p0.96 | ok c161 | 2 |
| 121 | PASS | 5 | 8.86 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 122 | fail | 4 | 6.16 | ok b300 | X b450 p0.00 | ok c161 | ok c331 | ok c171 | 4 |
| 123 | fail | 1 | 5.23 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | X b230 p0.00 | 4 |
| 124 | PASS | 5 | 8.33 | ok b300 | ok c321 | ok c151 | ok c311 | ok c151 | 0 |
| 125 | PASS | 5 | 8.81 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 127 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 128 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 129 | PASS | 5 | 9.05 | ok b300 | ok c311 | ok c151 | ok c301 | ok c201 | 0 |
| 130 | fail | 4 | 8.63 | ok b300 | ok c311 | ok b210 | X c381 p0.67 | ok c221 | 1 |
| 131 | fail | 2 | 5.63 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 3 |
| 132 | fail | 4 | 8.75 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 1 |
| 133 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 134 | PASS | 5 | 9.42 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 135 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.73 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c171 | 1 |
| 137 | fail | 2 | 5.5 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c381 p0.02 | ok c161 | 3 |
| 138 | PASS | 5 | 8.56 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 4 | 8.92 | ok b300 | ok c311 | X b210 p0.00 | ok b500 | ok c161 | 2 |
| 140 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | PASS | 5 | 9.06 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 142 | PASS | 5 | 8.46 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 143 | fail | 2 | 7.22 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 4 |
| 144 | PASS | 5 | 8.94 | ok b300 | ok c321 | ok c171 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 146 | PASS | 5 | 9.09 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.7 | ok b300 | ok c311 | ok c161 | ok c301 | ok c151 | 1 |
| 148 | fail | 4 | 7.85 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 1 |
| 149 | PASS | 5 | 8.24 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
