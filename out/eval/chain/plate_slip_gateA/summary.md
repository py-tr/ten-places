# plate_slip_gateA: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] third=[] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t4/plate/checkpoints/007500/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 76/100** (Wilson 95% (0.668, 0.833)), mean steps 4.67

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 100/100 | 6 | 8 | 2 |
| spoon | 91/100 | 92 | 0 | 0 |
| plate | 94/100 | 94 | 6 | 0 |
| fork | 86/100 | 95 | 14 | 1 |
| cup | 96/100 | 98 | 7 | 3 |

first failed step: none 76, spoon 9, fork 6, plate 6, cup 3

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 7.9 | ok b300 | ok c311 | ok c161 | ok c321 | ok c161 | 0 |
| 101 | PASS | 5 | 8.38 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 102 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 103 | PASS | 5 | 8.0 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 104 | fail | 4 | 7.1 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c161 | 0 |
| 105 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 106 | fail | 3 | 7.32 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c181 | 2 |
| 107 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 108 | PASS | 5 | 7.61 | ok b300 | ok c321 | ok c161 | ok c321 | ok c191 | 0 |
| 109 | PASS | 5 | 8.67 | ok c101 | ok c311 | ok c151 | ok c311 | ok c161 | 1 |
| 110 | fail | 4 | 8.92 | ok b300 | ok c311 | ok c151 | X c341 p0.01 | ok c171 | 1 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | PASS | 5 | 8.09 | ok b300 | ok c311 | ok c161 | ok c321 | ok c171 | 0 |
| 113 | fail | 4 | 8.54 | ok b300 | X c321 p1.00 | ok c151 | ok c311 | ok c181 | 0 |
| 114 | PASS | 5 | 8.77 | ok b300 | ok c321 | ok c161 | ok c321 | ok c171 | 0 |
| 115 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.25 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | fail | 4 | 8.75 | ok b300 | ok c311 | ok c161 | X c351 p0.06 | ok c161 | 1 |
| 118 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.58 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 120 | PASS | 5 | 8.57 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 0 |
| 121 | PASS | 5 | 8.62 | ok c101 | ok c321 | ok c161 | ok c321 | ok c201 | 2 |
| 122 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 123 | PASS | 5 | 7.95 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 1 |
| 124 | PASS | 5 | 8.31 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 126 | fail | 3 | 8.53 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 127 | PASS | 5 | 8.15 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 128 | PASS | 5 | 7.93 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c161 | ok c301 | ok c211 | 0 |
| 130 | PASS | 5 | 8.22 | ok b300 | ok c321 | ok c151 | ok c321 | ok c191 | 0 |
| 131 | PASS | 5 | 8.71 | ok c101 | ok c311 | ok c161 | ok c311 | ok c161 | 1 |
| 132 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c311 | ok c221 | 0 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c211 | 0 |
| 134 | PASS | 5 | 8.91 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 135 | PASS | 5 | 8.69 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.63 | ok b300 | X b450 p0.00 | ok c151 | ok c321 | ok c161 | 0 |
| 137 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c161 | ok c321 | ok c171 | 0 |
| 138 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 139 | PASS | 5 | 8.79 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 140 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 142 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 143 | fail | 4 | 8.44 | ok b300 | ok c311 | ok c151 | ok c311 | X c181 p0.00 | 1 |
| 144 | fail | 2 | 8.87 | ok b300 | ok c321 | X b210 p0.00 | X c141 p0.22 | X c71 p0.33 | 3 |
| 145 | PASS | 5 | 8.99 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 146 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.07 | ok b300 | ok c311 | ok c151 | ok c311 | ok c151 | 0 |
| 148 | PASS | 5 | 8.38 | ok b300 | ok c321 | ok c151 | ok c311 | ok c181 | 1 |
| 149 | PASS | 5 | 8.53 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 150 | PASS | 5 | 8.42 | ok b300 | ok c311 | ok c151 | ok c321 | ok c181 | 0 |
| 151 | fail | 4 | 8.56 | ok c101 | ok c311 | ok c151 | ok c301 | X b230 p0.16 | 2 |
| 152 | fail | 4 | 8.39 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c181 | 0 |
| 153 | PASS | 5 | 8.91 | ok b300 | ok c321 | ok c161 | ok c301 | ok c201 | 0 |
| 154 | PASS | 5 | 8.81 | ok b300 | ok c311 | ok c161 | ok c311 | ok c201 | 1 |
| 155 | fail | 4 | 8.76 | ok b300 | ok c311 | ok c151 | X c351 p0.06 | ok c171 | 1 |
| 156 | fail | 4 | 8.16 | ok b300 | X b450 p0.01 | ok c161 | ok c331 | ok c171 | 0 |
| 157 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 158 | fail | 4 | 8.28 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c181 | 1 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 160 | fail | 3 | 8.88 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c191 | 1 |
| 161 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 162 | fail | 4 | 8.28 | ok b300 | ok c321 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 163 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 164 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 165 | PASS | 5 | 8.28 | ok b300 | ok c311 | ok c171 | ok c311 | ok c191 | 0 |
| 166 | PASS | 5 | 7.91 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 167 | PASS | 5 | 8.14 | ok b300 | ok c321 | ok c151 | ok c321 | ok c181 | 0 |
| 168 | fail | 3 | 8.59 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c171 | 2 |
| 169 | fail | 3 | 8.66 | ok b300 | ok c311 | X c151 p1.00 | X c281 p1.00 | ok c161 | 0 |
| 170 | PASS | 5 | 8.38 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 171 | fail | 3 | 9.17 | ok c91 | ok c311 | X b210 p0.00 | X c381 p0.19 | ok c171 | 3 |
| 172 | PASS | 5 | 9.12 | ok c101 | ok c311 | ok c151 | ok c311 | ok c161 | 1 |
| 173 | PASS | 5 | 7.96 | ok b300 | ok c321 | ok c151 | ok c311 | ok c161 | 0 |
| 174 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 175 | PASS | 5 | 8.01 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 176 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c321 | ok c221 | 0 |
| 177 | fail | 4 | 9.03 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c171 | 0 |
| 178 | PASS | 5 | 8.87 | ok b300 | ok c311 | ok c161 | ok c321 | ok c181 | 0 |
| 179 | PASS | 5 | 8.21 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 180 | PASS | 5 | 8.52 | ok b300 | ok c301 | ok c161 | ok c311 | ok c191 | 0 |
| 181 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c161 | ok c301 | ok c201 | 0 |
| 182 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 183 | fail | 4 | 8.28 | ok b300 | ok c311 | ok c161 | X c361 p0.00 | ok c171 | 1 |
| 184 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 185 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 186 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 187 | PASS | 5 | 7.66 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 188 | PASS | 5 | 8.55 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 189 | PASS | 5 | 8.11 | ok b300 | ok c321 | ok b210 | ok c311 | ok c161 | 1 |
| 190 | PASS | 5 | 8.2 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 191 | fail | 3 | 7.01 | ok b300 | X b450 p0.00 | ok c161 | X c361 p0.01 | ok c171 | 2 |
| 192 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 193 | PASS | 5 | 8.85 | ok b300 | ok c321 | ok c161 | ok c331 | ok c171 | 0 |
| 194 | PASS | 5 | 9.11 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 195 | PASS | 5 | 8.35 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 196 | PASS | 5 | 8.56 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 197 | fail | 4 | 8.88 | ok b300 | ok c331 | X b210 p0.00 | ok c311 | ok c171 | 1 |
| 198 | fail | 4 | 7.92 | ok b300 | ok c321 | ok c151 | X c341 p0.98 | ok c151 | 1 |
| 199 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
