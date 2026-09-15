# plate_sizes_gateC: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] third=[] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/fork_cont/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_sizes/plate/checkpoints/007500/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 60/100** (Wilson 95% (0.502, 0.691)), mean steps 4.18

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 99/100 | 5 | 8 | 2 |
| spoon | 90/100 | 91 | 0 | 0 |
| plate | 72/100 | 70 | 39 | 9 |
| fork | 70/100 | 80 | 21 | 0 |
| cup | 87/100 | 92 | 11 | 0 |

first failed step: none 60, plate 25, spoon 9, cup 3, fork 2, drawer 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | fail | 2 | 7.97 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | X b230 p0.13 | 3 |
| 101 | PASS | 5 | 8.36 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 102 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 103 | PASS | 5 | 8.01 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 104 | PASS | 5 | 8.03 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 1 |
| 105 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | PASS | 5 | 8.16 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 107 | fail | 3 | 8.7 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c171 | 2 |
| 108 | PASS | 5 | 7.59 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 109 | PASS | 5 | 8.66 | ok c101 | ok c311 | ok c151 | ok c311 | ok c161 | 1 |
| 110 | PASS | 5 | 8.92 | ok b300 | ok c311 | ok c151 | ok c401 | ok c171 | 0 |
| 111 | fail | 3 | 8.61 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 112 | PASS | 5 | 8.07 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 113 | PASS | 5 | 8.52 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 114 | fail | 2 | 8.78 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 3 |
| 115 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.25 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 118 | fail | 2 | 8.58 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | X c161 p1.00 | 2 |
| 119 | fail | 2 | 8.59 | ok b300 | ok c321 | X b210 p0.00 | X c341 p0.97 | X b230 p0.01 | 2 |
| 120 | fail | 4 | 8.56 | ok b300 | ok c321 | ok b210 | X c351 p1.00 | ok c191 | 1 |
| 121 | fail | 4 | 7.48 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c241 | 0 |
| 122 | PASS | 5 | 8.49 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 123 | PASS | 5 | 7.95 | ok b300 | ok c321 | ok c151 | ok c301 | ok c161 | 0 |
| 124 | PASS | 5 | 8.35 | ok b300 | ok c321 | ok c151 | ok c301 | ok c161 | 0 |
| 125 | fail | 2 | 8.67 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | X c201 p1.00 | 2 |
| 126 | fail | 3 | 8.51 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c151 | 2 |
| 127 | fail | 4 | 8.18 | ok b300 | ok c311 | ok b210 | X b500 p0.00 | ok c201 | 2 |
| 128 | PASS | 5 | 7.97 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c191 | 0 |
| 130 | PASS | 5 | 8.24 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 131 | PASS | 5 | 8.71 | ok c101 | ok c311 | ok c161 | ok c311 | ok c161 | 1 |
| 132 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c301 | ok c241 | 0 |
| 133 | fail | 3 | 9.14 | ok b300 | ok c311 | X b210 p0.00 | X c71 p0.32 | ok c241 | 2 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 135 | fail | 3 | 8.69 | ok b300 | ok c321 | X b210 p0.00 | X c371 p0.62 | ok c161 | 1 |
| 136 | fail | 4 | 8.63 | ok b300 | X b450 p0.00 | ok c141 | ok c301 | ok c161 | 0 |
| 137 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 138 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 139 | PASS | 5 | 8.83 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 140 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 141 | PASS | 5 | 8.58 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 142 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 143 | PASS | 5 | 8.45 | ok b300 | ok c311 | ok c211 | ok c301 | ok c231 | 1 |
| 144 | fail | 3 | 8.87 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c181 | 2 |
| 145 | PASS | 5 | 8.99 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 1 |
| 146 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 1 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c151 | ok c301 | ok c151 | 0 |
| 148 | fail | 3 | 8.12 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | X c71 p0.99 | 1 |
| 149 | fail | 3 | 8.53 | ok b300 | ok c321 | X b210 p0.00 | X c351 p0.77 | ok c161 | 1 |
| 150 | fail | 3 | 8.45 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c171 | 2 |
| 151 | fail | 4 | 8.57 | ok c101 | ok c311 | ok c151 | ok c301 | X b230 p0.01 | 3 |
| 152 | fail | 4 | 8.39 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c181 | 0 |
| 153 | PASS | 5 | 8.91 | ok b300 | ok c321 | ok c161 | ok c301 | ok c211 | 0 |
| 154 | fail | 4 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | X c181 p0.00 | 1 |
| 155 | PASS | 5 | 8.76 | ok b300 | ok c311 | ok c161 | ok c501 | ok c171 | 0 |
| 156 | fail | 4 | 8.16 | ok b300 | X b450 p0.01 | ok c151 | ok c301 | ok c181 | 0 |
| 157 | fail | 3 | 8.71 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 158 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 1 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 1 |
| 160 | fail | 1 | 8.87 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X b230 p0.49 | 3 |
| 161 | PASS | 5 | 8.96 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 162 | fail | 3 | 8.27 | ok b300 | ok c321 | X b210 p0.00 | X c331 p0.71 | ok c201 | 1 |
| 163 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 164 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 165 | fail | 1 | 7.4 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X c71 p1.00 | 4 |
| 166 | PASS | 5 | 9.33 | ok b300 | ok c311 | ok c151 | ok c301 | ok c231 | 0 |
| 167 | PASS | 5 | 7.58 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 168 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 169 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 170 | fail | 3 | 8.39 | ok b300 | ok c311 | X b210 p0.00 | X c331 p0.82 | ok c181 | 1 |
| 171 | fail | 3 | 9.17 | ok c91 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c181 | 3 |
| 172 | PASS | 5 | 9.13 | ok c101 | ok c311 | ok c151 | ok c311 | ok c171 | 1 |
| 173 | PASS | 5 | 8.03 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 174 | fail | 3 | 8.7 | ok b300 | ok c311 | X b210 p0.00 | X c311 p0.94 | ok c171 | 1 |
| 175 | PASS | 5 | 7.7 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 176 | fail | 4 | 8.72 | ok b300 | ok c321 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 177 | fail | 2 | 9.03 | ok b300 | X c331 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 2 |
| 178 | PASS | 5 | 8.87 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 1 |
| 179 | fail | 2 | 8.25 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 3 |
| 180 | fail | 2 | 8.31 | ok b300 | ok c301 | X b210 p0.00 | X b500 p0.00 | X b230 p0.00 | 3 |
| 181 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c161 | ok c301 | ok c201 | 0 |
| 182 | fail | 3 | 8.46 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c171 | 2 |
| 183 | PASS | 5 | 8.29 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 184 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 185 | fail | 3 | 8.67 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c191 | 2 |
| 186 | PASS | 5 | 8.42 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 187 | fail | 2 | 5.9 | X b300 p0.00 | X b450 p0.00 | ok c221 | X c391 p1.00 | ok c171 | 2 |
| 188 | PASS | 5 | 8.52 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 189 | fail | 3 | 7.6 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 190 | PASS | 5 | 8.91 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 191 | fail | 4 | 7.07 | ok b300 | X b450 p0.01 | ok c171 | ok c401 | ok c191 | 1 |
| 192 | fail | 4 | 8.26 | ok b300 | ok c311 | X b210 p0.00 | ok c301 | ok c171 | 1 |
| 193 | PASS | 5 | 8.85 | ok b300 | ok c321 | ok c161 | ok c511 | ok c171 | 0 |
| 194 | PASS | 5 | 9.09 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 195 | PASS | 5 | 8.35 | ok b300 | ok c321 | ok c151 | ok c301 | ok c191 | 0 |
| 196 | PASS | 5 | 8.51 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 197 | PASS | 5 | 8.88 | ok b300 | ok c331 | ok c151 | ok c301 | ok c161 | 0 |
| 198 | fail | 3 | 7.9 | ok b300 | ok c321 | X b210 p0.00 | X c341 p0.81 | ok c161 | 1 |
| 199 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
