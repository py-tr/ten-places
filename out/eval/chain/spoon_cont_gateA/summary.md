# spoon_cont_gateA: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] third=[] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/fork_cont/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t4/plate/checkpoints/007500/pretrained_model', 'spoon': 'out/train/spoon_cont/spoon/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 81/100** (Wilson 95% (0.722, 0.875)), mean steps 4.70

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 99/100 | 6 | 9 | 2 |
| spoon | 94/100 | 94 | 0 | 0 |
| plate | 93/100 | 95 | 6 | 1 |
| fork | 90/100 | 92 | 11 | 1 |
| cup | 94/100 | 96 | 7 | 1 |

first failed step: none 81, plate 7, spoon 5, fork 3, cup 3, drawer 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 8.05 | ok b300 | ok c301 | ok c161 | ok b500 | ok c161 | 1 |
| 101 | PASS | 5 | 8.37 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 102 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 103 | PASS | 5 | 8.03 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 104 | PASS | 5 | 8.04 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 105 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | fail | 4 | 7.47 | ok b300 | ok c311 | ok c161 | X b500 p0.00 | ok c181 | 2 |
| 107 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 108 | PASS | 5 | 7.6 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 109 | PASS | 5 | 8.67 | ok c101 | ok c311 | ok c151 | ok c311 | ok c161 | 1 |
| 110 | PASS | 5 | 8.92 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 111 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 112 | PASS | 5 | 8.12 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 113 | fail | 3 | 8.53 | ok b300 | X b450 p0.00 | ok c151 | X b500 p0.00 | ok c161 | 1 |
| 114 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 115 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 116 | PASS | 5 | 8.27 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 118 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c161 | ok c311 | ok c161 | 0 |
| 119 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 120 | PASS | 5 | 8.56 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 121 | PASS | 5 | 7.57 | ok b300 | ok c311 | ok c171 | ok c311 | ok c211 | 0 |
| 122 | PASS | 5 | 8.49 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 123 | PASS | 5 | 7.95 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 124 | PASS | 5 | 8.33 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 126 | fail | 3 | 8.52 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 127 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 128 | PASS | 5 | 7.98 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 129 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 130 | PASS | 5 | 8.24 | ok b300 | ok c311 | ok c151 | ok c311 | ok c191 | 0 |
| 131 | PASS | 5 | 8.71 | ok c101 | ok c311 | ok c161 | ok c311 | ok c171 | 1 |
| 132 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 134 | PASS | 5 | 8.91 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 135 | PASS | 5 | 8.69 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 136 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 137 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 138 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 139 | PASS | 5 | 8.77 | ok b300 | ok c301 | ok c151 | ok c301 | ok c171 | 0 |
| 140 | PASS | 5 | 8.66 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 141 | PASS | 5 | 8.59 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 142 | PASS | 5 | 8.16 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 143 | fail | 4 | 8.46 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.00 | 1 |
| 144 | fail | 2 | 8.87 | ok b300 | ok c311 | X b210 p0.00 | X c141 p0.07 | X c71 p0.74 | 3 |
| 145 | PASS | 5 | 8.83 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 146 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 147 | PASS | 5 | 9.08 | ok b300 | ok c311 | ok c161 | ok c301 | ok c151 | 0 |
| 148 | fail | 4 | 8.09 | ok b300 | ok c311 | ok c161 | ok c311 | X b230 p0.00 | 1 |
| 149 | fail | 4 | 8.66 | ok b300 | ok c311 | X c161 p1.00 | ok c271 | ok c171 | 0 |
| 150 | PASS | 5 | 8.34 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 151 | fail | 4 | 8.57 | ok c101 | ok c311 | ok c151 | ok c301 | X b230 p0.05 | 2 |
| 152 | fail | 4 | 8.35 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c181 | 0 |
| 153 | PASS | 5 | 8.91 | ok b300 | ok c311 | ok c161 | ok c301 | ok c201 | 0 |
| 154 | PASS | 5 | 7.85 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 1 |
| 155 | PASS | 5 | 8.76 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 156 | fail | 4 | 8.16 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c171 | 0 |
| 157 | PASS | 5 | 8.77 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 158 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 160 | fail | 4 | 8.87 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c191 | 0 |
| 161 | PASS | 5 | 8.64 | ok b300 | ok c301 | ok c161 | ok c311 | ok c171 | 0 |
| 162 | fail | 3 | 8.28 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | X b230 p0.00 | 2 |
| 163 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 164 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 165 | PASS | 5 | 8.28 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 166 | PASS | 5 | 9.16 | ok b300 | ok c311 | ok c151 | ok c301 | ok c211 | 0 |
| 167 | PASS | 5 | 7.63 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 168 | fail | 3 | 8.58 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 169 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 170 | fail | 3 | 8.39 | ok b300 | ok c311 | ok c151 | X b500 p0.00 | X c151 p0.99 | 2 |
| 171 | fail | 3 | 9.17 | ok c91 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c171 | 3 |
| 172 | PASS | 5 | 9.12 | ok c101 | ok c311 | ok c151 | ok c311 | ok c171 | 1 |
| 173 | PASS | 5 | 7.94 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 174 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 175 | PASS | 5 | 7.57 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 1 |
| 176 | PASS | 5 | 8.72 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 177 | fail | 3 | 9.02 | ok b300 | X b450 p0.01 | ok c161 | X c81 p0.06 | ok c181 | 1 |
| 178 | PASS | 5 | 8.87 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 179 | fail | 4 | 8.24 | ok b300 | ok c311 | X c161 p1.00 | ok c281 | ok c151 | 0 |
| 180 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c161 | ok c401 | ok c191 | 1 |
| 181 | PASS | 5 | 8.39 | ok b300 | ok c311 | ok c161 | ok c301 | ok c231 | 0 |
| 182 | PASS | 5 | 8.45 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 183 | PASS | 5 | 8.29 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 184 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 185 | PASS | 5 | 8.82 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 186 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 187 | PASS | 5 | 7.86 | ok c201 | ok c311 | ok c151 | ok c301 | ok c171 | 1 |
| 188 | PASS | 5 | 8.54 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 189 | fail | 4 | 7.94 | ok b300 | ok c311 | X b210 p0.00 | ok c311 | ok c161 | 1 |
| 190 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 191 | fail | 2 | 5.45 | X b300 p0.01 | X b450 p0.00 | ok c161 | X c311 p0.98 | ok c181 | 2 |
| 192 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 193 | PASS | 5 | 8.85 | ok b300 | ok c311 | ok c161 | ok c501 | ok c171 | 0 |
| 194 | PASS | 5 | 9.11 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 195 | PASS | 5 | 8.35 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 196 | PASS | 5 | 8.54 | ok b300 | ok c301 | ok c151 | ok c311 | ok c171 | 0 |
| 197 | PASS | 5 | 8.88 | ok b300 | ok c321 | ok c161 | ok c311 | ok c161 | 0 |
| 198 | PASS | 5 | 7.94 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 199 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
