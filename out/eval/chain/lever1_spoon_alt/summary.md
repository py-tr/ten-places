# lever1_spoon_alt: seeds 100-199 (100), backend torch/cuda

levers: retry=['spoon', 'plate', 'fork', 'cup'] retry_exec={'spoon': 'exec50'} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=False drawer_retry_release=0.0

**full tables 70/100** (Wilson 95% (0.604, 0.781)), mean steps 4.47

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 97/100 | 0 | 0 | 0 |
| spoon | 86/100 | 87 | 13 | 0 |
| plate | 90/100 | 90 | 14 | 4 |
| fork | 82/100 | 86 | 22 | 2 |
| cup | 92/100 | 96 | 9 | 2 |

first failed step: none 70, spoon 11, plate 7, fork 6, cup 3, drawer 3

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c161 | ok c371 | ok c161 | 1 |
| 101 | PASS | 5 | 8.36 | ok b300 | ok c321 | ok c161 | ok c321 | ok c161 | 0 |
| 102 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 103 | PASS | 5 | 8.02 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 104 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c161 | ok c321 | ok c181 | 0 |
| 105 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | PASS | 5 | 8.24 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 1 |
| 107 | fail | 4 | 8.71 | ok b300 | ok c311 | ok c161 | X c371 p0.00 | ok c181 | 1 |
| 108 | PASS | 5 | 7.6 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 109 | fail | 2 | 6.03 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 3 |
| 110 | fail | 4 | 8.92 | ok b300 | ok c311 | ok c161 | X c381 p0.01 | ok c161 | 1 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 112 | PASS | 5 | 8.08 | ok b300 | ok c311 | ok c151 | ok c321 | ok c181 | 0 |
| 113 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 114 | PASS | 5 | 8.77 | ok b300 | ok c321 | ok c161 | ok c321 | ok c181 | 0 |
| 115 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.29 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 118 | PASS | 5 | 8.56 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.58 | ok b300 | ok c321 | ok c151 | ok c321 | ok c171 | 0 |
| 120 | PASS | 5 | 8.56 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 121 | PASS | 5 | 7.7 | ok b300 | ok c311 | ok c171 | ok c321 | ok c171 | 0 |
| 122 | PASS | 5 | 8.5 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 123 | fail | 3 | 7.94 | ok b300 | ok c321 | ok c161 | X b500 p0.00 | X b230 p0.00 | 2 |
| 124 | PASS | 5 | 8.33 | ok b300 | ok c321 | ok c151 | ok c311 | ok c151 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c191 | 0 |
| 126 | PASS | 5 | 8.52 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 127 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 128 | PASS | 5 | 8.06 | ok b300 | ok c311 | ok c161 | ok c321 | ok c151 | 0 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c201 | 0 |
| 130 | fail | 4 | 8.24 | ok b300 | ok c321 | ok c151 | ok c321 | X b230 p0.03 | 1 |
| 131 | fail | 2 | 5.82 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 2 |
| 132 | fail | 4 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c151 | ok c321 | ok c181 | 0 |
| 135 | PASS | 5 | 8.68 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.64 | ok b300 | X b450 p0.00 | ok c161 | ok b500 | ok c151 | 3 |
| 137 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c161 | ok c341 | ok c161 | 0 |
| 138 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 139 | fail | 2 | 8.77 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.01 | X c81 p0.01 | 3 |
| 140 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 141 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 142 | PASS | 5 | 8.18 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 143 | PASS | 5 | 8.46 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 144 | PASS | 5 | 8.87 | ok b300 | ok c321 | ok c171 | ok c311 | ok c161 | 0 |
| 145 | PASS | 5 | 9.0 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 146 | PASS | 5 | 8.71 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 148 | fail | 4 | 8.08 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c181 | 1 |
| 149 | PASS | 5 | 8.5 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 150 | fail | 3 | 8.36 | ok b300 | ok c311 | X b210 p0.00 | ok c321 | X c71 p0.71 | 2 |
| 151 | fail | 4 | 6.8 | ok b300 | X b450 p0.02 | ok c151 | ok c301 | ok c71 | 3 |
| 152 | fail | 4 | 8.36 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 1 |
| 153 | fail | 4 | 8.91 | ok b300 | ok c321 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 154 | PASS | 5 | 7.72 | ok b300 | ok c311 | ok c161 | ok c321 | ok c191 | 0 |
| 155 | fail | 4 | 8.77 | ok b300 | ok c311 | ok c151 | X c361 p0.08 | ok c161 | 1 |
| 156 | fail | 4 | 8.16 | ok b300 | X b450 p0.00 | ok c151 | ok c371 | ok c191 | 2 |
| 157 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c301 | ok c151 | 0 |
| 158 | fail | 4 | 8.26 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c161 | 1 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 160 | fail | 3 | 8.88 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c161 | 2 |
| 161 | PASS | 5 | 8.73 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 162 | fail | 3 | 8.27 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c181 | 2 |
| 163 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 164 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c151 | ok c301 | ok c191 | 0 |
| 165 | PASS | 5 | 8.3 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 166 | PASS | 5 | 7.91 | ok b300 | ok c311 | ok c151 | ok c321 | ok c161 | 0 |
| 167 | PASS | 5 | 7.85 | ok b300 | ok c311 | ok c151 | ok c321 | ok c161 | 1 |
| 168 | fail | 3 | 8.59 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c181 | 2 |
| 169 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 170 | PASS | 5 | 8.39 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 1 |
| 171 | fail | 4 | 7.18 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c171 | 1 |
| 172 | fail | 2 | 6.11 | ok b300 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c171 | 3 |
| 173 | PASS | 5 | 7.94 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 174 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c171 | ok c311 | ok c161 | 0 |
| 175 | PASS | 5 | 7.82 | ok b300 | ok c311 | ok c151 | ok c321 | ok c171 | 0 |
| 176 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c151 | ok c321 | ok c181 | 0 |
| 177 | fail | 4 | 9.03 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 1 |
| 178 | PASS | 5 | 8.87 | ok b300 | ok c311 | ok c161 | ok c321 | ok c181 | 0 |
| 179 | PASS | 5 | 8.24 | ok b300 | ok c311 | ok c161 | ok c301 | ok c151 | 0 |
| 180 | PASS | 5 | 8.54 | ok b300 | ok c311 | ok c161 | ok c311 | ok c201 | 0 |
| 181 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 182 | PASS | 5 | 8.46 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 183 | fail | 3 | 8.28 | ok b300 | X c311 p1.00 | ok c161 | X b500 p0.00 | ok c151 | 1 |
| 184 | fail | 4 | 8.71 | ok b300 | ok c311 | X b210 p0.00 | ok c311 | ok c161 | 1 |
| 185 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c171 | ok c301 | ok c181 | 1 |
| 186 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 187 | fail | 0 | 4.97 | X b300 p0.00 | X b450 p0.00 | X b210 p0.00 | X b500 p0.00 | X c71 p0.75 | 4 |
| 188 | PASS | 5 | 8.55 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 189 | fail | 3 | 7.3 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 190 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 191 | fail | 2 | 6.47 | X b300 p0.00 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c161 | 2 |
| 192 | PASS | 5 | 8.33 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 193 | PASS | 5 | 8.81 | ok b300 | ok c321 | ok c151 | ok c331 | ok c151 | 0 |
| 194 | PASS | 5 | 9.12 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 1 |
| 195 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 196 | PASS | 5 | 8.55 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 197 | fail | 2 | 8.88 | ok b300 | ok c331 | X b210 p0.00 | X c341 p0.81 | X c151 p0.99 | 1 |
| 198 | fail | 4 | 7.97 | ok b300 | ok c321 | ok c151 | X c341 p0.99 | ok c171 | 1 |
| 199 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
