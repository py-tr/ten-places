# lever6_third_attempt: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] third=['plate', 'fork', 'cup'] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 72/100** (Wilson 95% (0.625, 0.799)), mean steps 4.60

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 100/100 | 9 | 9 | 1 |
| spoon | 92/100 | 93 | 0 | 0 |
| plate | 89/100 | 91 | 11 | 2 |
| fork | 84/100 | 95 | 17 | 0 |
| cup | 95/100 | 98 | 5 | 0 |

first failed step: none 72, plate 9, spoon 8, fork 7, cup 4

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.26 | ok b300 | ok c311 | ok c161 | ok c381 | ok c161 | 2 |
| 101 | PASS | 5 | 8.37 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 102 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 103 | PASS | 5 | 8.01 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 104 | PASS | 5 | 8.03 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 105 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 106 | fail | 4 | 8.26 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c181 | 2 |
| 107 | fail | 4 | 8.71 | ok b300 | ok c311 | ok c161 | X c361 p0.00 | ok c191 | 2 |
| 108 | PASS | 5 | 7.64 | ok b300 | ok c321 | ok c161 | ok c311 | ok c191 | 0 |
| 109 | PASS | 5 | 8.68 | ok c101 | ok c311 | ok c161 | ok c311 | ok c161 | 1 |
| 110 | fail | 4 | 8.92 | ok b300 | ok c311 | ok c161 | X c391 p0.01 | ok c171 | 2 |
| 111 | PASS | 5 | 8.6 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | PASS | 5 | 8.1 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 113 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | PASS | 5 | 8.78 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 115 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.21 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 118 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 119 | fail | 4 | 8.59 | ok b300 | X b450 p0.32 | ok c151 | ok c321 | ok c151 | 0 |
| 120 | PASS | 5 | 8.56 | ok b300 | ok c321 | ok c151 | ok c321 | ok c171 | 0 |
| 121 | PASS | 5 | 7.64 | ok b300 | ok c311 | ok c171 | ok c321 | ok c221 | 0 |
| 122 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 123 | fail | 4 | 7.95 | ok b300 | ok c311 | ok c161 | X c381 p0.20 | ok c181 | 2 |
| 124 | PASS | 5 | 8.29 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 126 | PASS | 5 | 8.52 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 127 | PASS | 5 | 8.21 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 128 | PASS | 5 | 7.91 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c201 | 0 |
| 130 | PASS | 5 | 8.26 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 1 |
| 131 | PASS | 5 | 8.73 | ok c101 | ok c311 | ok c161 | ok c311 | ok c161 | 1 |
| 132 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c221 | 0 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c211 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 135 | PASS | 5 | 8.69 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.63 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c161 | 0 |
| 137 | PASS | 5 | 8.31 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 138 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 139 | fail | 3 | 8.78 | ok b300 | ok c311 | X b210 p0.00 | X c381 p0.70 | ok c161 | 3 |
| 140 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 141 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 142 | PASS | 5 | 8.16 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | fail | 4 | 8.44 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.02 | 2 |
| 144 | PASS | 5 | 8.87 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 8.99 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 146 | PASS | 5 | 8.7 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.07 | ok b300 | ok c311 | ok c151 | ok c301 | ok c151 | 0 |
| 148 | fail | 4 | 8.28 | ok b300 | ok c311 | ok c151 | ok c321 | X c211 p0.05 | 2 |
| 149 | PASS | 5 | 8.54 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 150 | PASS | 5 | 8.43 | ok b300 | ok c311 | ok c171 | ok c311 | ok c181 | 0 |
| 151 | fail | 4 | 8.57 | ok c101 | ok c311 | ok c151 | ok c311 | X b230 p0.01 | 3 |
| 152 | fail | 4 | 8.33 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c191 | 0 |
| 153 | PASS | 5 | 8.91 | ok b300 | ok c321 | ok c161 | ok c301 | ok c201 | 0 |
| 154 | fail | 4 | 7.72 | ok b300 | ok c311 | ok c161 | ok c311 | X c71 p0.57 | 2 |
| 155 | fail | 4 | 8.77 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c171 | 2 |
| 156 | fail | 4 | 8.16 | ok b300 | X b450 p0.01 | ok c151 | ok c311 | ok c171 | 0 |
| 157 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 158 | fail | 3 | 8.23 | ok b300 | ok c311 | X c161 p1.00 | X c321 p0.96 | ok c201 | 1 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 160 | fail | 2 | 8.88 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.06 | ok c181 | 4 |
| 161 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 162 | fail | 3 | 8.28 | ok b300 | ok c311 | X b210 p0.00 | X c381 p0.74 | ok c191 | 4 |
| 163 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 164 | PASS | 5 | 8.79 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 165 | PASS | 5 | 8.29 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 166 | PASS | 5 | 8.89 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 167 | PASS | 5 | 7.62 | ok c81 | ok c311 | ok c161 | ok c321 | ok c171 | 1 |
| 168 | fail | 3 | 8.58 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c181 | 4 |
| 169 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 170 | PASS | 5 | 8.39 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 171 | fail | 1 | 9.18 | ok c91 | X c311 p1.00 | X b210 p0.00 | X c381 p0.97 | X c71 p0.83 | 7 |
| 172 | fail | 3 | 9.12 | ok c101 | ok c311 | X b210 p0.00 | X c381 p0.05 | ok c171 | 5 |
| 173 | PASS | 5 | 7.98 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 174 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c171 | ok c311 | ok c231 | 0 |
| 175 | fail | 3 | 6.83 | ok c101 | X b450 p0.00 | ok c151 | X c311 p0.00 | ok c171 | 3 |
| 176 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 177 | fail | 4 | 9.03 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 0 |
| 178 | PASS | 5 | 8.86 | ok b300 | ok c311 | ok c161 | ok c321 | ok c181 | 0 |
| 179 | fail | 4 | 8.22 | ok b300 | ok c311 | X c161 p1.00 | ok c281 | ok c171 | 0 |
| 180 | PASS | 5 | 8.44 | ok b300 | ok c301 | ok c161 | ok c311 | ok c191 | 0 |
| 181 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 182 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 183 | fail | 4 | 8.33 | ok b300 | ok c311 | ok c161 | X c361 p0.00 | ok c181 | 2 |
| 184 | fail | 4 | 8.71 | ok b300 | ok c311 | X b210 p0.00 | ok c321 | ok c171 | 2 |
| 185 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 1 |
| 186 | PASS | 5 | 8.42 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 187 | PASS | 5 | 8.73 | ok c121 | ok c311 | ok c151 | ok c311 | ok c171 | 1 |
| 188 | PASS | 5 | 8.52 | ok b300 | ok c321 | ok c151 | ok c301 | ok c181 | 0 |
| 189 | fail | 3 | 7.67 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 4 |
| 190 | PASS | 5 | 8.98 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 191 | PASS | 5 | 8.54 | ok c91 | ok c321 | ok c161 | ok c311 | ok c181 | 1 |
| 192 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 193 | PASS | 5 | 8.85 | ok b300 | ok c321 | ok c151 | ok c331 | ok c171 | 0 |
| 194 | PASS | 5 | 9.11 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 195 | PASS | 5 | 8.35 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 196 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 197 | fail | 4 | 8.88 | ok b300 | ok c331 | X b210 p0.00 | ok c311 | ok c171 | 2 |
| 198 | fail | 4 | 7.94 | ok b300 | ok c321 | ok c151 | X c361 p0.01 | ok c151 | 2 |
| 199 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
