# cup_stall2_gateA: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] third=[] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_stall2/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/fork_cont/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t4/plate/checkpoints/007500/pretrained_model', 'spoon': 'out/train/spoon_cont2/spoon/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 14/100** (Wilson 95% (0.085, 0.221)), mean steps 4.01

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 99/100 | 8 | 9 | 2 |
| spoon | 95/100 | 96 | 0 | 0 |
| plate | 95/100 | 95 | 6 | 1 |
| fork | 95/100 | 94 | 7 | 1 |
| cup | 17/100 | 29 | 88 | 8 |

first failed step: cup 74, none 14, plate 5, spoon 4, fork 2, drawer 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | fail | 4 | 7.9 | ok b300 | ok c301 | ok c161 | ok b500 | X b230 p0.00 | 2 |
| 101 | PASS | 5 | 8.33 | ok b300 | ok c311 | ok c161 | ok c311 | ok c201 | 1 |
| 102 | fail | 4 | 8.66 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 103 | PASS | 5 | 8.02 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 104 | fail | 4 | 8.55 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 105 | fail | 4 | 8.64 | ok b300 | ok c301 | ok c161 | ok c311 | X b230 p0.01 | 1 |
| 106 | fail | 4 | 8.28 | ok b300 | ok c311 | ok c161 | ok c291 | X b230 p0.00 | 1 |
| 107 | fail | 4 | 8.71 | ok b300 | ok b450 | ok c161 | ok c291 | X b230 p0.00 | 1 |
| 108 | fail | 4 | 7.6 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 1 |
| 109 | fail | 4 | 8.67 | ok c101 | ok c311 | ok c151 | ok c311 | X b230 p0.01 | 2 |
| 110 | fail | 4 | 8.92 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.99 | 0 |
| 111 | fail | 4 | 8.6 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.07 | 1 |
| 112 | fail | 4 | 8.09 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 113 | fail | 4 | 8.54 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.01 | 1 |
| 114 | fail | 4 | 8.76 | ok b300 | ok c311 | ok c161 | ok c301 | X c71 p1.00 | 1 |
| 115 | fail | 4 | 8.72 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 116 | fail | 4 | 8.24 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.01 | 1 |
| 117 | fail | 4 | 8.75 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c221 | 1 |
| 118 | PASS | 5 | 8.56 | ok b300 | ok c301 | ok c161 | ok c311 | ok c251 | 1 |
| 119 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c151 | ok c301 | ok c231 | 0 |
| 120 | fail | 4 | 8.57 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 121 | fail | 4 | 8.01 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 1 |
| 122 | PASS | 5 | 8.49 | ok b300 | ok c311 | ok c151 | ok c311 | ok c71 | 1 |
| 123 | fail | 4 | 7.95 | ok b300 | ok c321 | ok c161 | ok c301 | X c201 p0.41 | 1 |
| 124 | PASS | 5 | 8.27 | ok b300 | ok c311 | ok c151 | ok c301 | ok c71 | 1 |
| 125 | fail | 4 | 8.67 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.09 | 1 |
| 126 | fail | 2 | 8.51 | ok b300 | ok c301 | X b210 p0.00 | X b500 p0.00 | X b230 p0.02 | 3 |
| 127 | fail | 4 | 8.15 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.02 | 1 |
| 128 | fail | 4 | 7.9 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.34 | 1 |
| 129 | fail | 4 | 8.65 | ok b300 | ok c301 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 130 | fail | 4 | 8.24 | ok b300 | ok c311 | ok c151 | ok c311 | X c91 p0.00 | 1 |
| 131 | fail | 4 | 8.73 | ok c101 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 2 |
| 132 | fail | 4 | 8.62 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 133 | fail | 4 | 9.14 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c311 | ok c161 | ok c301 | ok c231 | 0 |
| 135 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c161 | ok c301 | ok b230 | 1 |
| 136 | fail | 3 | 8.63 | ok b300 | X c311 p1.00 | ok c161 | ok c301 | X c71 p1.00 | 1 |
| 137 | fail | 4 | 8.33 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 1 |
| 138 | fail | 4 | 8.17 | ok b300 | ok c311 | ok c151 | ok c301 | X c151 p0.21 | 1 |
| 139 | fail | 4 | 8.78 | ok b300 | ok c301 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 140 | fail | 4 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.03 | 1 |
| 141 | fail | 4 | 8.59 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 142 | fail | 4 | 8.16 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 143 | fail | 4 | 8.47 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.01 | 1 |
| 144 | fail | 3 | 8.86 | ok b300 | ok c311 | ok b210 | X b500 p0.00 | X b230 p0.00 | 3 |
| 145 | fail | 4 | 9.0 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 146 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c411 | ok c101 | 1 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c151 | ok c301 | ok c201 | 0 |
| 148 | fail | 4 | 8.38 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.00 | 1 |
| 149 | fail | 4 | 8.59 | ok b300 | ok c311 | ok c161 | ok c301 | X c71 p0.96 | 1 |
| 150 | fail | 4 | 8.42 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 151 | fail | 4 | 8.57 | ok c101 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 2 |
| 152 | fail | 4 | 8.35 | ok b300 | ok c301 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 153 | fail | 4 | 8.91 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 154 | fail | 4 | 7.27 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 1 |
| 155 | fail | 4 | 8.78 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.05 | 1 |
| 156 | fail | 4 | 8.15 | ok b300 | ok c311 | ok c161 | ok c301 | X c71 p0.99 | 1 |
| 157 | fail | 4 | 8.73 | ok b300 | ok c301 | ok c151 | ok c301 | X c71 p1.00 | 1 |
| 158 | fail | 4 | 8.21 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 159 | fail | 4 | 9.25 | ok b300 | ok c301 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 160 | fail | 3 | 8.88 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | X b230 p0.01 | 2 |
| 161 | fail | 4 | 8.72 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.01 | 1 |
| 162 | fail | 4 | 8.3 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 163 | fail | 4 | 8.75 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.01 | 1 |
| 164 | fail | 4 | 8.79 | ok b300 | ok c301 | ok c151 | ok c301 | X b230 p0.01 | 1 |
| 165 | fail | 4 | 7.94 | ok c101 | ok c311 | ok c161 | ok c311 | X b230 p0.84 | 2 |
| 166 | fail | 4 | 9.03 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.01 | 1 |
| 167 | fail | 4 | 7.89 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.00 | 1 |
| 168 | fail | 2 | 8.6 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 3 |
| 169 | fail | 4 | 8.67 | ok b300 | ok c311 | ok c161 | ok c301 | X c71 p0.07 | 1 |
| 170 | fail | 4 | 8.38 | ok b300 | ok c301 | ok c151 | ok c311 | X b230 p0.00 | 1 |
| 171 | fail | 2 | 9.17 | ok c91 | ok c301 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 4 |
| 172 | fail | 3 | 9.13 | ok c101 | ok c301 | X b210 p0.00 | ok c311 | X b230 p0.01 | 3 |
| 173 | fail | 4 | 7.99 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.00 | 1 |
| 174 | PASS | 5 | 8.7 | ok b300 | ok c301 | ok c161 | ok c301 | ok c171 | 0 |
| 175 | fail | 3 | 5.48 | X b300 p0.00 | X b450 p0.00 | ok c161 | ok c301 | ok c251 | 2 |
| 176 | PASS | 5 | 8.72 | ok b300 | ok c311 | ok c161 | ok c311 | ok c71 | 1 |
| 177 | fail | 3 | 9.02 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | X c221 p1.00 | 0 |
| 178 | fail | 4 | 8.87 | ok b300 | ok c301 | ok c161 | ok c301 | X b230 p0.18 | 1 |
| 179 | fail | 3 | 8.23 | ok b300 | ok c311 | X c161 p1.00 | ok c281 | X b230 p0.28 | 1 |
| 180 | fail | 4 | 8.53 | ok b300 | ok c411 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 181 | fail | 4 | 8.38 | ok b300 | X c311 p1.00 | ok c161 | ok c301 | ok c201 | 1 |
| 182 | fail | 4 | 8.47 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 183 | fail | 4 | 8.28 | ok b300 | ok c301 | ok c161 | ok c301 | X b230 p0.23 | 1 |
| 184 | fail | 4 | 8.71 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 185 | fail | 4 | 8.76 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.01 | 1 |
| 186 | fail | 4 | 8.42 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 187 | fail | 4 | 7.51 | ok c91 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 2 |
| 188 | fail | 4 | 8.52 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 189 | fail | 4 | 7.99 | ok b300 | ok c311 | ok c151 | ok c301 | X c131 p0.54 | 1 |
| 190 | fail | 4 | 8.99 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 191 | PASS | 5 | 8.45 | ok c91 | ok c311 | ok c161 | ok c311 | ok b230 | 1 |
| 192 | fail | 4 | 8.33 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 193 | fail | 4 | 8.85 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.13 | 1 |
| 194 | fail | 4 | 9.13 | ok b300 | ok c311 | ok c161 | ok c301 | X c201 p0.54 | 0 |
| 195 | fail | 4 | 8.35 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 196 | fail | 4 | 8.5 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 197 | fail | 4 | 8.89 | ok b300 | ok c311 | ok c161 | ok c311 | X c71 p0.88 | 1 |
| 198 | fail | 4 | 7.88 | ok b300 | ok c311 | ok c151 | ok c491 | X c71 p0.99 | 1 |
| 199 | PASS | 5 | 8.68 | ok b300 | ok c411 | ok c151 | ok c331 | ok c191 | 0 |
