# v8 (plate_t4) vs v7 (plate_t1)

| row | v7 | v8 | +/- | p | v8 per step (drawer/spoon/plate/fork/cup) | v7 per step | new losses |
|---|---|---|---|---|---|---|---|
| agent 0-49 | 39/50 | 42/50 | +7 / -4 | 0.549 | 50/47/50/48/47 | 50/46/48/44/48 | 19 fork, 20 cup, 24 spoon, 25 cup |
| agent 200-249 | 42/50 | 39/50 | +4 / -7 | 0.549 | 49/46/47/44/48 | 50/49/44/47/50 | 205 cup, 212 spoon, 220 spoon, 223 drawer_open, 230 plate, 240 fork, 243 cup |
| fixed OV 0-49 | 40/50 | 44/50 | +6 / -2 | 0.289 | 50/47/49/49/48 | 50/46/47/45/47 | 15 plate, 27 fork |
| fixed torch 0-49 | 41/50 | 43/50 | +4 / -2 | 0.688 | 50/47/50/47/49 | 50/48/45/45/49 | 41 cup, 46 spoon |
| fixed OV 200-249 | 40/50 | 42/50 | +5 / -3 | 0.727 | 50/47/49/47/47 | 49/46/44/46/45 | 209 fork, 216 cup, 243 cup |
| ranges x1.5 (fixed) | 43/50 | 42/50 | +3 / -4 | 1.000 | 49/46/46/45/49 | 49/46/47/45/49 | 215 plate, 229 plate, 230 plate, 235 spoon |
| ranges x2.0 (fixed) | 35/50 | 38/50 | +7 / -4 | 0.549 | 47/46/44/41/48 | 48/45/44/39/46 | 206 fork, 215 plate, 237 drawer_open, 247 plate |
| ranges x2.0 (agent) | 36/50 | 37/50 | +6 / -5 | 1.000 | 48/47/44/39/47 | 48/44/47/39/49 | 206 fork, 215 plate, 229 fork, 238 plate, 242 plate |
| sizes +-10% (fixed) | 28/50 | 30/50 | +6 / -4 | 0.754 | 50/46/43/39/41 | 49/48/41/37/38 | 236 fork, 239 plate, 244 spoon, 245 spoon |
| sizes +-20% (fixed) | 23/50 | 24/50 | +8 / -7 | 1.000 | 50/46/32/34/36 | 50/47/40/37/31 | 200 spoon, 203 plate, 207 plate, 211 cup, 230 plate, 239 plate, 246 plate |
| cup only +-20% (fixed) | 25/50 | 32/50 | +10 / -3 | 0.092 | 49/47/46/46/35 | 48/45/41/41/34 | 203 cup, 215 cup, 227 cup |
| sizes +-20% (agent) | 23/50 | 25/50 | +9 / -7 | 0.804 | 50/44/35/36/34 | 50/48/34/37/33 | 200 plate, 218 cup, 226 cup, 230 plate, 240 spoon, 244 plate, 246 plate |

Placement error medians (cm, agent rows, placed objects): v7 -> v8

- agent 0-49: spoon 0.31 -> 0.30; plate 0.40 -> 0.41; fork 0.78 -> 0.76; cup 0.26 -> 0.40
- agent 200-249: spoon 0.37 -> 0.36; plate 0.36 -> 0.41; fork 0.85 -> 0.75; cup 0.40 -> 0.32
- ranges x2.0 (agent): spoon 0.35 -> 0.35; plate 0.51 -> 0.43; fork 0.90 -> 0.72; cup 0.42 -> 0.37
- sizes +-20% (agent): spoon 0.52 -> 0.55; plate 0.69 -> 0.64; fork 0.77 -> 0.64; cup 0.47 -> 0.45
