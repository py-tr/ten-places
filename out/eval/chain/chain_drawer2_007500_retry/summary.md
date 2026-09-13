# chain_drawer2_007500_retry: seeds 100-149 (50), backend torch/cuda

levers: retry=['drawer'] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 43/50** (Wilson 95% (0.738, 0.93)), mean steps 4.84

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 50/50 | 2 | 2 | 1 |
| spoon | 49/50 | 49 | 0 | 0 |
| plate | 49/50 | 49 | 0 | 0 |
| fork | 47/50 | 47 | 0 | 0 |
| cup | 47/50 | 47 | 0 | 0 |

first failed step: none 43, cup 3, fork 2, spoon 1, plate 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 7.86 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 101 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 102 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 103 | PASS | 5 | 8.01 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 104 | PASS | 5 | 8.04 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 105 | PASS | 5 | 8.57 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 106 | fail | 4 | 8.28 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c181 | 0 |
| 107 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 108 | PASS | 5 | 7.59 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 109 | PASS | 5 | 8.67 | ok c101 | ok c311 | ok c161 | ok c321 | ok c151 | 1 |
| 110 | PASS | 5 | 8.93 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | fail | 4 | 8.06 | ok b300 | ok c311 | ok c151 | ok c321 | X b230 p0.00 | 0 |
| 113 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | PASS | 5 | 8.74 | ok b300 | ok c321 | ok c151 | ok c321 | ok c181 | 0 |
| 115 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.25 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 118 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c151 | ok c321 | ok c171 | 0 |
| 120 | PASS | 5 | 8.57 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 121 | PASS | 5 | 7.64 | ok b300 | ok c311 | ok c171 | ok c321 | ok c181 | 0 |
| 122 | PASS | 5 | 8.5 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 123 | fail | 4 | 7.95 | ok b300 | ok c321 | ok c161 | X b500 p0.00 | ok c191 | 0 |
| 124 | PASS | 5 | 8.3 | ok b300 | ok c321 | ok c151 | ok c301 | ok c151 | 0 |
| 125 | PASS | 5 | 8.68 | ok b300 | ok c321 | ok c161 | ok c311 | ok c191 | 0 |
| 126 | PASS | 5 | 8.52 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 127 | PASS | 5 | 8.2 | ok b300 | ok c311 | ok c161 | ok c311 | ok c231 | 0 |
| 128 | PASS | 5 | 7.87 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c191 | 0 |
| 130 | fail | 4 | 8.22 | ok b300 | ok c321 | ok c151 | ok c321 | X b230 p0.00 | 0 |
| 131 | PASS | 5 | 8.72 | ok c101 | ok c311 | ok c161 | ok c311 | ok c161 | 1 |
| 132 | fail | 4 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 0 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 8.91 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 135 | PASS | 5 | 8.69 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.64 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c181 | 0 |
| 137 | PASS | 5 | 8.36 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 138 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 3 | 8.77 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 140 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 142 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | PASS | 5 | 8.44 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 144 | PASS | 5 | 8.87 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.81 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 146 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 148 | PASS | 5 | 8.38 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 149 | PASS | 5 | 8.56 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
