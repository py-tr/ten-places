# fork_ens_probe: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] third=[] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'temporal_coeff': 0.01}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t4/plate/checkpoints/007500/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 73/100** (Wilson 95% (0.636, 0.807)), mean steps 4.63

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 100/100 | 8 | 12 | 2 |
| spoon | 90/100 | 93 | 0 | 0 |
| plate | 95/100 | 94 | 6 | 0 |
| fork | 82/100 | 94 | 20 | 3 |
| cup | 96/100 | 99 | 6 | 2 |

first failed step: none 73, spoon 10, fork 9, plate 5, cup 3

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.19 | ok b300 | ok c311 | ok c161 | ok c391 | ok c161 | 1 |
| 101 | PASS | 5 | 8.34 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 102 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 103 | PASS | 5 | 8.03 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 104 | PASS | 5 | 8.04 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 105 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 106 | PASS | 5 | 7.64 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 2 |
| 107 | fail | 4 | 8.71 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c191 | 1 |
| 108 | PASS | 5 | 7.6 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 109 | PASS | 5 | 8.66 | ok c101 | ok c311 | ok c151 | ok c311 | ok c161 | 1 |
| 110 | PASS | 5 | 8.93 | ok b300 | ok c311 | ok c151 | ok c291 | ok c171 | 1 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 112 | PASS | 5 | 8.11 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 113 | fail | 4 | 8.54 | ok b300 | X c321 p1.00 | ok c151 | ok c311 | ok c181 | 0 |
| 114 | PASS | 5 | 8.77 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 115 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.27 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 118 | fail | 4 | 8.56 | ok b300 | ok c311 | ok c161 | X c361 p0.00 | ok c181 | 1 |
| 119 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 120 | PASS | 5 | 8.57 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 121 | PASS | 5 | 8.62 | ok c101 | ok c321 | ok c161 | ok c311 | ok c71 | 2 |
| 122 | PASS | 5 | 8.5 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 123 | fail | 4 | 7.95 | ok b300 | ok c321 | ok c161 | X b500 p0.10 | ok c161 | 1 |
| 124 | PASS | 5 | 8.29 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 126 | fail | 3 | 8.51 | ok b300 | ok c311 | X b210 p0.00 | X c381 p0.79 | ok c171 | 2 |
| 127 | PASS | 5 | 8.21 | ok b300 | ok c311 | ok c161 | ok c311 | ok c191 | 0 |
| 128 | PASS | 5 | 8.49 | ok c101 | ok c311 | ok c161 | ok c311 | ok c171 | 1 |
| 129 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 130 | PASS | 5 | 8.24 | ok b300 | ok c321 | ok c151 | ok c311 | ok c201 | 0 |
| 131 | PASS | 5 | 8.78 | ok c101 | ok c311 | ok c161 | ok c301 | ok c161 | 1 |
| 132 | fail | 4 | 8.61 | ok b300 | ok c311 | ok c151 | ok c301 | X c81 p0.87 | 1 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 135 | PASS | 5 | 8.7 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 136 | fail | 4 | 8.62 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c161 | 0 |
| 137 | PASS | 5 | 8.31 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 138 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 139 | PASS | 5 | 8.77 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 140 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 141 | PASS | 5 | 8.63 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 142 | PASS | 5 | 8.21 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 143 | fail | 4 | 8.39 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 144 | fail | 3 | 8.87 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.01 | ok c171 | 2 |
| 145 | PASS | 5 | 8.83 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 146 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 147 | PASS | 5 | 9.07 | ok b300 | ok c311 | ok c161 | ok c301 | ok c151 | 0 |
| 148 | fail | 3 | 8.23 | ok b300 | X c361 p0.23 | ok c161 | X b500 p0.00 | ok c181 | 1 |
| 149 | fail | 3 | 8.67 | ok b300 | X b450 p0.00 | ok c161 | X c371 p0.14 | ok c161 | 1 |
| 150 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 151 | PASS | 5 | 8.5 | ok c91 | ok c311 | ok c151 | ok c301 | ok c191 | 2 |
| 152 | fail | 4 | 8.37 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c191 | 0 |
| 153 | PASS | 5 | 8.91 | ok b300 | ok c311 | ok c161 | ok c291 | ok c221 | 0 |
| 154 | fail | 4 | 8.6 | ok b300 | ok c311 | ok c161 | ok c311 | X c71 p1.00 | 1 |
| 155 | fail | 4 | 8.77 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | ok c171 | 1 |
| 156 | fail | 3 | 8.16 | ok b300 | X b450 p0.01 | ok c161 | X b500 p0.00 | ok c171 | 1 |
| 157 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 158 | fail | 4 | 8.27 | ok b300 | ok c311 | ok c161 | X c521 p0.03 | ok c181 | 1 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 160 | fail | 4 | 8.71 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c171 | 0 |
| 161 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 162 | PASS | 5 | 8.27 | ok b300 | ok c321 | ok c151 | ok c301 | ok c201 | 0 |
| 163 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 164 | fail | 4 | 8.8 | ok b300 | ok c311 | ok c151 | X c361 p0.10 | ok c191 | 1 |
| 165 | fail | 4 | 7.48 | ok b300 | X b450 p0.01 | ok c161 | ok c311 | ok c191 | 1 |
| 166 | PASS | 5 | 7.89 | ok b300 | ok c311 | ok c151 | ok c301 | ok c191 | 0 |
| 167 | PASS | 5 | 7.65 | ok b300 | ok c311 | ok c151 | ok c321 | ok c181 | 0 |
| 168 | fail | 3 | 8.58 | ok b300 | ok c321 | X b210 p0.00 | X c421 p0.29 | ok c171 | 2 |
| 169 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 170 | PASS | 5 | 8.38 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 171 | fail | 2 | 9.18 | ok c91 | ok c311 | X b210 p0.00 | X c351 p0.77 | X c71 p0.55 | 3 |
| 172 | PASS | 5 | 9.12 | ok c101 | ok c311 | ok b210 | ok c311 | ok c161 | 2 |
| 173 | PASS | 5 | 7.93 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 174 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c161 | ok c301 | ok c211 | 0 |
| 175 | PASS | 5 | 7.28 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 1 |
| 176 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 177 | fail | 4 | 9.03 | ok b300 | X c331 p0.07 | ok c161 | ok c311 | ok c181 | 0 |
| 178 | PASS | 5 | 8.87 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 179 | PASS | 5 | 8.31 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 180 | fail | 4 | 8.44 | ok b300 | ok c301 | ok c161 | X c351 p1.00 | ok c181 | 1 |
| 181 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c161 | ok c301 | ok c201 | 0 |
| 182 | PASS | 5 | 8.46 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 183 | fail | 4 | 8.29 | ok b300 | ok c311 | ok c161 | X c411 p0.01 | ok c181 | 1 |
| 184 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 185 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 186 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 187 | PASS | 5 | 7.58 | ok c91 | ok c311 | ok c151 | ok c311 | ok c171 | 1 |
| 188 | PASS | 5 | 8.51 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 189 | fail | 3 | 7.59 | ok b300 | ok c311 | X b210 p0.00 | X c321 p0.64 | ok c161 | 1 |
| 190 | PASS | 5 | 8.72 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 191 | fail | 3 | 7.15 | ok b300 | X b450 p0.00 | ok c161 | X c351 p0.01 | ok c181 | 2 |
| 192 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 193 | PASS | 5 | 8.85 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 1 |
| 194 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 195 | PASS | 5 | 8.35 | ok b300 | ok c321 | ok c161 | ok c311 | ok c181 | 0 |
| 196 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 197 | PASS | 5 | 8.88 | ok b300 | ok c331 | ok c161 | ok c311 | ok c171 | 0 |
| 198 | fail | 4 | 7.92 | ok b300 | ok c321 | ok c151 | X c361 p0.67 | ok c171 | 1 |
| 199 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c151 | ok c301 | ok c191 | 0 |
