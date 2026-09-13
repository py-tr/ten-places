# chain_drawer_002500: seeds 100-149 (50), backend torch/cuda

levers: retry=[] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t1_drawer/drawer/checkpoints/002500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 0/50** (Wilson 95% (0.0, 0.071)), mean steps 1.24

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 0/50 | 0 | 0 | 0 |
| spoon | 0/50 | 0 | 0 | 0 |
| plate | 23/50 | 23 | 0 | 0 |
| fork | 0/50 | 11 | 0 | 0 |
| cup | 39/50 | 39 | 0 | 0 |

first failed step: drawer 50

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | fail | 1 | 0.0 | X b300 p0.02 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 101 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c181 | X b500 p0.00 | ok c181 | 0 |
| 102 | fail | 2 | 0.44 | X b300 p0.00 | X b450 p0.00 | ok c181 | X b500 p0.00 | ok c171 | 0 |
| 103 | fail | 2 | 0.17 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c361 p0.14 | ok c181 | 0 |
| 104 | fail | 1 | -0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 105 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 106 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c201 | 0 |
| 107 | fail | 2 | 0.82 | X b300 p0.00 | X b450 p0.00 | ok c191 | X b500 p0.00 | ok c191 | 0 |
| 108 | fail | 0 | 0.73 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 109 | fail | 2 | 0.11 | X b300 p0.00 | X b450 p0.00 | ok c181 | X c351 p0.02 | ok c161 | 0 |
| 110 | fail | 1 | 0.01 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 111 | fail | 2 | 0.87 | X b300 p0.00 | X b450 p0.00 | ok c171 | X c361 p0.05 | ok c171 | 0 |
| 112 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c81 p0.00 | ok c171 | 0 |
| 113 | fail | 1 | 0.71 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 114 | fail | 0 | 0.31 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 115 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 0 |
| 116 | fail | 1 | 0.02 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c151 | 0 |
| 117 | fail | 1 | 0.85 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 118 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c201 | X b500 p0.00 | ok c221 | 0 |
| 119 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c181 | 0 |
| 120 | fail | 0 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 121 | fail | 1 | 0.54 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c211 | 0 |
| 122 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 123 | fail | 0 | 0.12 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 124 | fail | 1 | -0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c161 | 0 |
| 125 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c171 | X c361 p0.01 | ok c191 | 0 |
| 126 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c371 p0.00 | ok c151 | 0 |
| 127 | fail | 1 | -0.05 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c211 | 0 |
| 128 | fail | 1 | -0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c151 | 0 |
| 129 | fail | 2 | 0.45 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c361 p0.01 | ok c191 | 0 |
| 130 | fail | 0 | 0.83 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 131 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c181 | X c351 p0.85 | ok c161 | 0 |
| 132 | fail | 1 | 0.14 | X b300 p0.00 | X b450 p0.00 | ok c211 | X c71 p0.01 | X b230 p0.00 | 0 |
| 133 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c191 | X b500 p0.00 | X b230 p0.00 | 0 |
| 134 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c181 | 0 |
| 135 | fail | 0 | 0.02 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 136 | fail | 1 | -0.01 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c151 | 0 |
| 137 | fail | 2 | 0.79 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 0 |
| 138 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 139 | fail | 1 | 0.44 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c151 | 0 |
| 140 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c81 p0.01 | ok c171 | 0 |
| 141 | fail | 0 | 0.13 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 0 |
| 142 | fail | 2 | 0.01 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c171 | 0 |
| 143 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 0 |
| 144 | fail | 1 | -0.0 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 0 |
| 145 | fail | 1 | 0.71 | X b300 p0.01 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c201 | 0 |
| 146 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | X b230 p0.00 | 0 |
| 147 | fail | 2 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c161 | X c371 p0.29 | ok c171 | 0 |
| 148 | fail | 1 | 0.0 | X b300 p0.00 | X b450 p0.00 | ok c191 | X b500 p0.00 | X b230 p0.00 | 0 |
| 149 | fail | 2 | 0.32 | X b300 p0.00 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c151 | 0 |
