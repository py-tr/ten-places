# lever4_drawer_repull: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 72/100** (Wilson 95% (0.625, 0.799)), mean steps 4.62

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 99/100 | 5 | 7 | 2 |
| spoon | 92/100 | 91 | 0 | 0 |
| plate | 90/100 | 93 | 11 | 3 |
| fork | 84/100 | 90 | 16 | 1 |
| cup | 97/100 | 99 | 5 | 2 |

first failed step: none 72, plate 9, fork 8, spoon 7, cup 3, drawer 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 7.85 | ok b300 | ok c311 | ok c161 | ok b500 | ok c161 | 1 |
| 101 | PASS | 5 | 8.36 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 102 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 103 | PASS | 5 | 8.01 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 104 | PASS | 5 | 8.04 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 105 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 106 | fail | 4 | 8.23 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c171 | 1 |
| 107 | fail | 4 | 8.7 | ok b300 | ok c311 | ok c161 | X c371 p0.00 | ok c181 | 1 |
| 108 | PASS | 5 | 7.59 | ok b300 | ok c321 | ok c161 | ok c321 | ok c191 | 0 |
| 109 | PASS | 5 | 8.66 | ok c101 | ok c311 | ok c161 | ok c311 | ok c161 | 1 |
| 110 | PASS | 5 | 8.92 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | PASS | 5 | 8.11 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 113 | PASS | 5 | 8.54 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 114 | PASS | 5 | 8.74 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 115 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c151 | ok c301 | ok c191 | 0 |
| 116 | PASS | 5 | 8.25 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 118 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 120 | PASS | 5 | 8.57 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 121 | fail | 4 | 7.65 | ok b300 | ok c311 | ok c161 | ok c321 | X c71 p1.00 | 1 |
| 122 | PASS | 5 | 8.5 | ok b300 | ok c311 | ok c151 | ok c321 | ok c181 | 0 |
| 123 | fail | 4 | 7.95 | ok b300 | ok c321 | ok c161 | X b500 p0.00 | ok c171 | 1 |
| 124 | PASS | 5 | 8.37 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | fail | 4 | 8.51 | ok b300 | ok c311 | ok c161 | X c371 p0.00 | ok c171 | 1 |
| 127 | PASS | 5 | 8.16 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 128 | PASS | 5 | 8.04 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 130 | PASS | 5 | 8.25 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 1 |
| 131 | PASS | 5 | 8.76 | ok c101 | ok c311 | ok c161 | ok c311 | ok c171 | 1 |
| 132 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c221 | 0 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c151 | ok c321 | ok c171 | 0 |
| 135 | PASS | 5 | 8.69 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.63 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c161 | 0 |
| 137 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c161 | ok c331 | ok c171 | 0 |
| 138 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 139 | fail | 3 | 8.77 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c71 | 3 |
| 140 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 142 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 143 | PASS | 5 | 8.45 | ok b300 | ok c311 | ok c151 | ok c311 | ok c211 | 1 |
| 144 | PASS | 5 | 8.87 | ok b300 | ok c321 | ok c171 | ok c311 | ok c171 | 0 |
| 145 | PASS | 5 | 8.83 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 146 | PASS | 5 | 8.71 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c161 | ok c301 | ok c151 | 0 |
| 148 | fail | 4 | 8.04 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c221 | 0 |
| 149 | PASS | 5 | 8.55 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 150 | PASS | 5 | 8.34 | ok b300 | ok c311 | ok c161 | ok c321 | ok c181 | 0 |
| 151 | fail | 4 | 8.59 | ok c101 | ok c311 | ok c151 | ok c311 | X c71 p0.99 | 2 |
| 152 | fail | 4 | 8.4 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 0 |
| 153 | PASS | 5 | 8.91 | ok b300 | ok c321 | ok c161 | ok c301 | ok c201 | 0 |
| 154 | fail | 4 | 8.7 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 155 | fail | 4 | 8.77 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c191 | 1 |
| 156 | fail | 4 | 8.16 | ok b300 | X b450 p0.01 | ok c151 | ok c301 | ok c171 | 0 |
| 157 | PASS | 5 | 8.72 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 158 | fail | 4 | 8.35 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c171 | 1 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 160 | fail | 2 | 8.88 | ok b300 | X b450 p0.00 | X b210 p0.00 | X c371 p0.28 | ok c171 | 2 |
| 161 | PASS | 5 | 8.72 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 162 | fail | 4 | 8.27 | ok b300 | ok c321 | X c81 p0.78 | ok c301 | ok c181 | 1 |
| 163 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 164 | PASS | 5 | 8.79 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 165 | fail | 4 | 7.69 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 0 |
| 166 | PASS | 5 | 9.27 | ok b300 | ok c311 | ok c151 | ok c311 | ok c211 | 0 |
| 167 | fail | 3 | 7.89 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c181 | 2 |
| 168 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 1 |
| 169 | fail | 3 | 8.67 | ok b300 | ok c311 | X c151 p1.00 | X c281 p1.00 | ok c151 | 0 |
| 170 | PASS | 5 | 8.39 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 171 | fail | 3 | 9.18 | ok c91 | ok c311 | X b210 p0.00 | X c381 p0.99 | ok c171 | 3 |
| 172 | fail | 3 | 9.11 | ok c101 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c171 | 3 |
| 173 | PASS | 5 | 7.97 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 174 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c171 | ok c311 | ok c191 | 0 |
| 175 | PASS | 5 | 7.43 | ok b300 | ok c311 | ok c151 | ok c321 | ok c161 | 1 |
| 176 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c151 | ok c321 | ok c181 | 0 |
| 177 | fail | 4 | 9.03 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 0 |
| 178 | PASS | 5 | 8.87 | ok b300 | ok c311 | ok c161 | ok c321 | ok c181 | 0 |
| 179 | fail | 4 | 8.24 | ok b300 | ok c311 | X c161 p1.00 | ok c281 | ok c161 | 0 |
| 180 | PASS | 5 | 8.51 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 181 | PASS | 5 | 8.45 | ok b300 | ok c311 | ok c151 | ok c311 | ok c201 | 0 |
| 182 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 183 | PASS | 5 | 8.29 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 1 |
| 184 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c231 | ok c311 | ok c171 | 0 |
| 185 | fail | 4 | 8.68 | ok b300 | ok c311 | X b210 p0.00 | ok b500 | ok c151 | 2 |
| 186 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 187 | PASS | 5 | 7.5 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 188 | PASS | 5 | 8.51 | ok b300 | ok c321 | ok c151 | ok c301 | ok c181 | 0 |
| 189 | fail | 3 | 7.87 | ok b300 | ok c311 | X b210 p0.00 | X c291 p1.00 | ok c161 | 1 |
| 190 | fail | 4 | 8.96 | ok b300 | ok b450 | ok c151 | X c331 p1.00 | ok c161 | 0 |
| 191 | fail | 2 | 6.49 | X b300 p0.01 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 2 |
| 192 | PASS | 5 | 8.33 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 193 | PASS | 5 | 8.85 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 194 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 195 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 196 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 197 | PASS | 5 | 8.88 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 1 |
| 198 | fail | 4 | 7.95 | ok b300 | ok c321 | ok c151 | X c341 p1.00 | ok c171 | 1 |
| 199 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
