# cup_stall_gateA: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] third=[] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_stall/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/fork_cont/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t4/plate/checkpoints/007500/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 42/100** (Wilson 95% (0.328, 0.518)), mean steps 4.26

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 100/100 | 8 | 8 | 2 |
| spoon | 93/100 | 96 | 0 | 0 |
| plate | 95/100 | 95 | 5 | 0 |
| fork | 93/100 | 92 | 9 | 0 |
| cup | 45/100 | 78 | 32 | 3 |

first failed step: cup 45, none 42, spoon 7, plate 4, fork 2

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.26 | ok b300 | ok c311 | ok c161 | ok b500 | ok c241 | 1 |
| 101 | fail | 4 | 8.39 | ok b300 | ok c321 | ok c161 | ok c311 | X b230 p0.02 | 1 |
| 102 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 103 | fail | 4 | 8.02 | ok b300 | ok c311 | ok c161 | ok c301 | X c151 p1.00 | 0 |
| 104 | fail | 4 | 8.04 | ok b300 | ok c321 | ok c161 | ok c311 | X c81 p0.97 | 1 |
| 105 | PASS | 5 | 8.57 | ok b300 | ok c311 | ok c161 | ok c311 | ok c141 | 0 |
| 106 | PASS | 5 | 8.09 | ok b300 | ok c311 | ok c161 | ok c401 | ok c191 | 0 |
| 107 | fail | 4 | 8.71 | ok b300 | ok c311 | ok c161 | ok c301 | X c71 p1.00 | 1 |
| 108 | PASS | 5 | 7.58 | ok b300 | ok c321 | ok c161 | ok c311 | ok c151 | 0 |
| 109 | fail | 4 | 8.66 | ok c101 | ok c311 | ok c151 | ok c311 | X c161 p1.00 | 1 |
| 110 | PASS | 5 | 8.92 | ok b300 | ok c311 | ok c151 | ok c401 | ok c181 | 0 |
| 111 | fail | 4 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | X c151 p1.00 | 0 |
| 112 | PASS | 5 | 8.08 | ok b300 | ok c311 | ok c161 | ok c311 | ok c151 | 0 |
| 113 | fail | 4 | 8.52 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.00 | 1 |
| 114 | fail | 4 | 8.77 | ok b300 | ok c321 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 115 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c301 | ok c71 | 1 |
| 116 | fail | 4 | 8.25 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 117 | fail | 4 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 118 | PASS | 5 | 8.58 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 119 | PASS | 5 | 8.58 | ok b300 | ok c321 | ok c151 | ok c301 | ok c151 | 0 |
| 120 | PASS | 5 | 8.57 | ok b300 | ok c321 | ok c151 | ok c301 | ok c191 | 0 |
| 121 | fail | 4 | 7.63 | ok b300 | ok c311 | ok c151 | ok c311 | X b230 p0.01 | 1 |
| 122 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 123 | fail | 4 | 7.95 | ok b300 | ok c321 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 124 | fail | 4 | 8.33 | ok b300 | ok c321 | ok c151 | ok c301 | X b230 p0.34 | 1 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c301 | ok c141 | 0 |
| 126 | fail | 2 | 8.52 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | X b230 p0.01 | 3 |
| 127 | fail | 4 | 8.17 | ok b300 | ok c311 | ok c161 | ok c311 | X c151 p1.00 | 0 |
| 128 | fail | 4 | 7.96 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 129 | fail | 4 | 8.65 | ok b300 | ok c311 | ok c161 | ok c301 | X c171 p1.00 | 0 |
| 130 | PASS | 5 | 8.3 | ok b300 | ok c321 | ok c151 | ok c311 | ok c141 | 0 |
| 131 | fail | 4 | 8.72 | ok c101 | ok c311 | ok c161 | ok c311 | X c161 p1.00 | 1 |
| 132 | fail | 4 | 8.61 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 133 | fail | 4 | 9.13 | ok b300 | ok c311 | ok c161 | ok c311 | X c161 p1.00 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c161 | ok c301 | ok c161 | 0 |
| 135 | fail | 4 | 8.69 | ok b300 | ok c321 | ok c161 | ok c301 | X c161 p1.00 | 1 |
| 136 | fail | 3 | 8.62 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 137 | fail | 4 | 8.35 | ok b300 | ok c321 | ok c161 | ok c311 | X b230 p0.01 | 1 |
| 138 | fail | 4 | 8.19 | ok b300 | ok c311 | ok c151 | ok c301 | X c71 p1.00 | 1 |
| 139 | PASS | 5 | 8.78 | ok b300 | ok c311 | ok c151 | ok c301 | ok c201 | 0 |
| 140 | fail | 4 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | X c161 p1.00 | 0 |
| 141 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c161 | ok c301 | ok c191 | 0 |
| 142 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c301 | ok c211 | 0 |
| 143 | PASS | 5 | 8.45 | ok b300 | ok c311 | ok c151 | ok c301 | ok c151 | 0 |
| 144 | fail | 3 | 8.87 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c151 | 2 |
| 145 | fail | 4 | 9.0 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 146 | PASS | 5 | 8.7 | ok b300 | ok c321 | ok c161 | ok c301 | ok c151 | 0 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c151 | ok c291 | ok c151 | 0 |
| 148 | fail | 2 | 8.14 | ok b300 | X c361 p0.16 | ok c161 | X c71 p0.57 | X b230 p0.00 | 2 |
| 149 | PASS | 5 | 8.58 | ok b300 | ok c321 | ok c161 | ok c301 | ok c201 | 0 |
| 150 | fail | 4 | 8.32 | ok b300 | ok c311 | ok c161 | ok c311 | X c141 p1.00 | 0 |
| 151 | fail | 4 | 8.58 | ok c101 | ok c311 | ok c151 | ok c301 | X c171 p1.00 | 1 |
| 152 | fail | 3 | 8.35 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | X c231 p0.99 | 0 |
| 153 | fail | 4 | 8.91 | ok b300 | ok c321 | ok c161 | ok c301 | X c71 p0.97 | 1 |
| 154 | fail | 4 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | X c151 p1.00 | 0 |
| 155 | fail | 3 | 8.76 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | X b230 p0.00 | 2 |
| 156 | fail | 3 | 8.16 | ok b300 | X b450 p0.01 | ok c161 | ok c311 | X c151 p1.00 | 0 |
| 157 | PASS | 5 | 8.73 | ok b300 | ok c311 | ok c151 | ok c301 | ok b230 | 0 |
| 158 | fail | 4 | 8.22 | ok b300 | ok c311 | ok c161 | ok c301 | X c151 p1.00 | 0 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c161 | ok c301 | ok c231 | 0 |
| 160 | fail | 4 | 8.69 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c71 | 1 |
| 161 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 162 | PASS | 5 | 8.32 | ok b300 | ok c321 | ok c151 | ok c301 | ok c161 | 0 |
| 163 | fail | 4 | 8.75 | ok b300 | ok c321 | ok c151 | ok c301 | X c161 p1.00 | 0 |
| 164 | PASS | 5 | 8.79 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 165 | fail | 4 | 8.28 | ok b300 | ok c311 | ok c171 | ok c311 | X c161 p1.00 | 0 |
| 166 | PASS | 5 | 9.11 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 167 | PASS | 5 | 8.09 | ok b300 | ok c321 | ok c151 | ok c301 | ok c181 | 0 |
| 168 | fail | 3 | 8.59 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c171 | 2 |
| 169 | fail | 4 | 8.64 | ok b300 | ok c311 | ok c151 | ok c301 | X c211 p1.00 | 0 |
| 170 | fail | 4 | 8.39 | ok b300 | ok c311 | ok c151 | ok c311 | X c171 p1.00 | 0 |
| 171 | fail | 1 | 7.23 | ok b300 | X c331 p0.00 | X b210 p0.00 | X b500 p0.00 | X c71 p0.98 | 3 |
| 172 | fail | 4 | 9.12 | ok c101 | ok c311 | ok c151 | ok c311 | X c71 p1.00 | 2 |
| 173 | PASS | 5 | 7.94 | ok b300 | ok c321 | ok c151 | ok c311 | ok c121 | 0 |
| 174 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 175 | PASS | 5 | 7.59 | ok c91 | ok c311 | ok c151 | ok c311 | ok c141 | 1 |
| 176 | fail | 4 | 8.72 | ok b300 | ok c321 | ok c161 | ok c311 | X c151 p0.65 | 1 |
| 177 | fail | 3 | 9.03 | ok b300 | X c331 p0.00 | ok c161 | ok c311 | X c181 p1.00 | 0 |
| 178 | PASS | 5 | 8.87 | ok b300 | ok c311 | ok c161 | ok c301 | ok c151 | 0 |
| 179 | fail | 4 | 8.32 | ok b300 | ok c311 | ok c161 | ok c301 | X c141 p1.00 | 0 |
| 180 | fail | 3 | 8.45 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | X b230 p0.00 | 2 |
| 181 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c161 | ok c301 | ok c231 | 0 |
| 182 | PASS | 5 | 8.48 | ok b300 | ok c311 | ok c151 | ok c301 | ok c71 | 1 |
| 183 | fail | 4 | 8.28 | ok b300 | ok c311 | ok c161 | ok c311 | X c161 p1.00 | 0 |
| 184 | fail | 4 | 8.71 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 185 | fail | 4 | 8.83 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 186 | PASS | 5 | 8.42 | ok b300 | ok c311 | ok c161 | ok c311 | ok c231 | 0 |
| 187 | fail | 4 | 7.66 | ok c91 | ok c311 | ok c151 | ok c311 | X c131 p1.00 | 1 |
| 188 | PASS | 5 | 8.54 | ok b300 | ok c321 | ok c161 | ok c301 | ok c241 | 0 |
| 189 | fail | 3 | 7.69 | ok b300 | ok c311 | X b210 p0.00 | ok b500 | X b230 p0.01 | 3 |
| 190 | fail | 4 | 8.24 | ok b300 | ok c311 | ok c151 | ok c301 | X c161 p1.00 | 0 |
| 191 | fail | 4 | 8.48 | ok c91 | ok c321 | ok c161 | ok c311 | X c161 p1.00 | 1 |
| 192 | PASS | 5 | 8.33 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 193 | PASS | 5 | 8.85 | ok b300 | ok c321 | ok c161 | ok c311 | ok c201 | 0 |
| 194 | PASS | 5 | 9.13 | ok b300 | ok c311 | ok c151 | ok c301 | ok c151 | 0 |
| 195 | fail | 4 | 8.35 | ok b300 | ok c321 | ok c161 | ok c301 | X c151 p1.00 | 0 |
| 196 | PASS | 5 | 9.01 | ok c91 | ok c311 | ok c151 | ok c311 | ok c141 | 1 |
| 197 | fail | 4 | 8.88 | ok b300 | ok c331 | ok c161 | ok c301 | X b230 p0.02 | 1 |
| 198 | PASS | 5 | 7.94 | ok b300 | ok c321 | ok c151 | ok c491 | ok c141 | 0 |
| 199 | fail | 4 | 8.66 | ok b300 | ok c311 | ok c151 | ok c311 | X c141 p1.00 | 0 |
