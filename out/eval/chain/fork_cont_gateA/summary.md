# fork_cont_gateA: seeds 100-199 (100), backend torch/cuda

levers: retry=['drawer', 'plate', 'fork', 'cup'] third=[] retry_exec={} exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'cup': 'out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model', 'drawer': 'out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model', 'fork': 'out/train/fork_cont/fork/checkpoints/007500/pretrained_model', 'plate': 'out/train/plate_t4/plate/checkpoints/007500/pretrained_model', 'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=0 retry_camera_end=True drawer_retry_release=0.0

**full tables 81/100** (Wilson 95% (0.722, 0.875)), mean steps 4.69

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 99/100 | 8 | 10 | 2 |
| spoon | 90/100 | 92 | 0 | 0 |
| plate | 94/100 | 96 | 4 | 0 |
| fork | 95/100 | 94 | 6 | 0 |
| cup | 91/100 | 98 | 9 | 1 |

first failed step: none 81, spoon 9, cup 5, plate 4, drawer 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 100 | PASS | 5 | 7.96 | ok b300 | ok c311 | ok c161 | ok b500 | ok c161 | 1 |
| 101 | PASS | 5 | 8.37 | ok b300 | ok c321 | ok c161 | ok c311 | ok c171 | 0 |
| 102 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 103 | PASS | 5 | 8.02 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 104 | PASS | 5 | 7.83 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 105 | PASS | 5 | 8.64 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 106 | PASS | 5 | 7.99 | ok b300 | ok c311 | ok c161 | ok c401 | ok c171 | 0 |
| 107 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c161 | ok c301 | ok c201 | 0 |
| 108 | PASS | 5 | 7.6 | ok b300 | ok c321 | ok c161 | ok c311 | ok c191 | 0 |
| 109 | PASS | 5 | 8.68 | ok c101 | ok c321 | ok c151 | ok c311 | ok c161 | 1 |
| 110 | PASS | 5 | 8.92 | ok b300 | ok c311 | ok c151 | ok c401 | ok c171 | 0 |
| 111 | PASS | 5 | 8.61 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 112 | PASS | 5 | 8.11 | ok b300 | ok c311 | ok c161 | ok c311 | ok c201 | 0 |
| 113 | PASS | 5 | 8.53 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 114 | PASS | 5 | 8.77 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 115 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 116 | PASS | 5 | 8.24 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 117 | PASS | 5 | 8.75 | ok b300 | ok c311 | ok c161 | ok c301 | ok c161 | 0 |
| 118 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 119 | PASS | 5 | 8.59 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 120 | PASS | 5 | 8.56 | ok b300 | ok c321 | ok c151 | ok c301 | ok c181 | 0 |
| 121 | fail | 4 | 8.61 | ok c101 | ok c321 | ok c161 | ok c311 | X c171 p0.37 | 2 |
| 122 | PASS | 5 | 8.62 | ok b300 | ok c311 | ok c151 | ok c311 | ok c171 | 0 |
| 123 | PASS | 5 | 7.95 | ok b300 | ok c321 | ok c161 | ok c301 | ok c161 | 0 |
| 124 | PASS | 5 | 8.32 | ok b300 | ok c321 | ok c151 | ok c301 | ok c161 | 0 |
| 125 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 126 | fail | 3 | 8.52 | ok b300 | ok c311 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 127 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 128 | PASS | 5 | 7.92 | ok b300 | ok c321 | ok c161 | ok c301 | ok c161 | 0 |
| 129 | fail | 4 | 8.66 | ok b300 | ok c311 | ok c161 | ok c301 | X b230 p0.01 | 1 |
| 130 | PASS | 5 | 8.26 | ok b300 | ok c321 | ok c151 | ok c311 | ok c171 | 0 |
| 131 | PASS | 5 | 8.71 | ok c101 | ok c311 | ok c161 | ok c311 | ok c171 | 1 |
| 132 | fail | 3 | 8.6 | ok b300 | ok c311 | X c151 p1.00 | ok c301 | X c71 p0.97 | 1 |
| 133 | PASS | 5 | 9.14 | ok b300 | ok c311 | ok c161 | ok c311 | ok c211 | 0 |
| 134 | PASS | 5 | 8.9 | ok b300 | ok c321 | ok c161 | ok c301 | ok c191 | 0 |
| 135 | PASS | 5 | 8.67 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 136 | fail | 4 | 8.62 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c171 | 0 |
| 137 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 138 | PASS | 5 | 8.16 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 139 | PASS | 5 | 8.81 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 140 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | ok c171 | 0 |
| 141 | PASS | 5 | 8.63 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 142 | PASS | 5 | 8.17 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 143 | fail | 4 | 8.43 | ok b300 | ok c311 | ok c151 | ok c301 | X b230 p0.00 | 1 |
| 144 | fail | 2 | 8.87 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | X c161 p0.92 | 3 |
| 145 | PASS | 5 | 8.99 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 146 | PASS | 5 | 8.73 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 147 | PASS | 5 | 9.07 | ok b300 | ok c321 | ok c151 | ok c301 | ok c151 | 0 |
| 148 | PASS | 5 | 8.35 | ok b300 | ok c321 | ok c151 | ok c311 | ok c191 | 1 |
| 149 | fail | 1 | 8.62 | ok b300 | X c311 p1.00 | X c161 p1.00 | X c271 p0.96 | X c171 p1.00 | 0 |
| 150 | PASS | 5 | 8.35 | ok b300 | ok c311 | ok c171 | ok c311 | ok c181 | 0 |
| 151 | fail | 4 | 8.58 | ok c101 | ok c311 | ok c151 | ok c301 | X c71 p0.99 | 2 |
| 152 | fail | 4 | 8.42 | ok b300 | X b450 p0.00 | ok c151 | ok c311 | ok c181 | 0 |
| 153 | PASS | 5 | 8.91 | ok b300 | ok c321 | ok c161 | ok c301 | ok c191 | 0 |
| 154 | fail | 3 | 7.64 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | X c71 p0.00 | 1 |
| 155 | PASS | 5 | 8.77 | ok b300 | ok c311 | ok c151 | ok c411 | ok c171 | 0 |
| 156 | fail | 3 | 8.15 | X b300 p1.00 | X b450 p0.00 | ok c161 | ok b500 | ok c181 | 1 |
| 157 | PASS | 5 | 8.65 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 158 | PASS | 5 | 8.24 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 159 | PASS | 5 | 9.25 | ok b300 | ok c311 | ok c161 | ok c301 | ok c171 | 0 |
| 160 | fail | 4 | 8.68 | ok b300 | X b450 p0.00 | ok c151 | ok c301 | ok c181 | 0 |
| 161 | PASS | 5 | 8.96 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 162 | fail | 4 | 8.28 | ok b300 | ok c311 | ok c151 | ok c301 | X c71 p1.00 | 1 |
| 163 | PASS | 5 | 8.75 | ok b300 | ok c321 | ok c151 | ok c301 | ok c171 | 0 |
| 164 | PASS | 5 | 8.8 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 165 | fail | 4 | 7.5 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 1 |
| 166 | PASS | 5 | 7.97 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 167 | PASS | 5 | 7.47 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 168 | fail | 3 | 8.59 | ok b300 | ok c321 | X b210 p0.00 | X b500 p0.00 | ok c161 | 2 |
| 169 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 170 | PASS | 5 | 8.38 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 171 | fail | 2 | 7.21 | ok b300 | X c331 p0.00 | X b210 p0.00 | X b500 p0.00 | ok c181 | 2 |
| 172 | PASS | 5 | 9.12 | ok c101 | ok c311 | ok c201 | ok c311 | ok c161 | 1 |
| 173 | PASS | 5 | 8.0 | ok b300 | ok c311 | ok c151 | ok c311 | ok c161 | 0 |
| 174 | PASS | 5 | 8.7 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 175 | PASS | 5 | 7.54 | ok c281 | ok c311 | ok c151 | ok c311 | ok c171 | 1 |
| 176 | PASS | 5 | 8.72 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 177 | fail | 4 | 9.03 | ok b300 | X b450 p0.00 | ok c161 | ok c311 | ok c181 | 0 |
| 178 | PASS | 5 | 8.86 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 179 | PASS | 5 | 8.32 | ok b300 | ok c311 | ok c161 | ok c301 | ok c181 | 0 |
| 180 | PASS | 5 | 8.54 | ok b300 | ok c301 | ok c161 | ok c301 | ok c201 | 0 |
| 181 | PASS | 5 | 8.41 | ok b300 | ok c311 | ok c161 | ok c301 | ok c191 | 0 |
| 182 | PASS | 5 | 8.47 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 183 | PASS | 5 | 8.24 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 184 | PASS | 5 | 8.71 | ok b300 | ok c311 | ok c161 | ok c311 | ok c171 | 0 |
| 185 | PASS | 5 | 8.68 | ok b300 | ok c311 | ok c151 | ok c301 | ok c161 | 0 |
| 186 | PASS | 5 | 8.42 | ok b300 | ok c311 | ok c161 | ok c311 | ok c181 | 0 |
| 187 | PASS | 5 | 8.2 | ok c91 | ok c311 | ok c151 | ok c301 | ok c161 | 1 |
| 188 | PASS | 5 | 8.54 | ok b300 | ok c321 | ok c161 | ok c301 | ok c171 | 0 |
| 189 | PASS | 5 | 8.1 | ok b300 | ok c321 | ok c151 | ok c301 | ok c161 | 0 |
| 190 | PASS | 5 | 8.08 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
| 191 | fail | 4 | 6.65 | ok b300 | X b450 p0.00 | ok c161 | ok c501 | ok c171 | 1 |
| 192 | PASS | 5 | 8.32 | ok b300 | ok c321 | ok c161 | ok c301 | ok c181 | 0 |
| 193 | PASS | 5 | 8.85 | ok b300 | ok c321 | ok c161 | ok c331 | ok c171 | 0 |
| 194 | PASS | 5 | 9.12 | ok b300 | ok c311 | ok c151 | ok c301 | ok c181 | 0 |
| 195 | PASS | 5 | 8.35 | ok b300 | ok c321 | ok c161 | ok c301 | ok c161 | 0 |
| 196 | PASS | 5 | 9.01 | ok c91 | ok c311 | ok c151 | ok c311 | ok c171 | 1 |
| 197 | PASS | 5 | 8.87 | ok b300 | ok c331 | ok c161 | ok c311 | ok c161 | 0 |
| 198 | PASS | 5 | 7.87 | ok b300 | ok c311 | ok c151 | ok c391 | ok c181 | 0 |
| 199 | PASS | 5 | 8.67 | ok b300 | ok c311 | ok c151 | ok c311 | ok c181 | 0 |
