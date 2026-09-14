# First plan: E-cores vs every core

Qwen3-VL-4B INT4, OpenVINO GenAI, one top-camera image + command -> JSON plan; 10 demo commands; same plan 10/10. Medians:

| Placement | Plan (wall) | Time to first token | Per output token | Tokens/s | Input / output tokens |
|---|---|---|---|---|---|
| E-cores (while the arms move) | 20.4 s | 11.31 s | 282 ms | 3.5 | 452 / 34 |
| Every core (arms still) | 7.3 s | 3.45 s | 119 ms | 8.4 | 452 / 34 |

| seed | command | E-cores | all cores | TTFT E / all | same plan |
|---|---|---|---|---|---|
| 0 | Set the table. | 25.3 s | 10.3 s | 14.04 / 5.31 s | yes |
| 1 | Could you set everything out for dinner? | 23.2 s | 8.4 s | 11.44 / 3.65 s | yes |
| 2 | No fork today, set the rest. | 21.4 s | 7.6 s | 10.64 / 3.56 s | yes |
| 3 | I'm having soup tonight. | 16.8 s | 6.0 s | 11.09 / 3.39 s | yes |
| 4 | Just the plate and the cup. | 16.0 s | 5.3 s | 10.63 / 3.10 s | yes |
| 5 | Set the table. | 22.3 s | 8.5 s | 11.68 / 3.06 s | yes |
| 6 | Set the table and light a candle. | 18.1 s | 6.9 s | 11.17 / 3.52 s | yes |
| 7 | Just my cup, thanks. | 15.6 s | 5.0 s | 11.68 / 3.13 s | yes |
| 8 | Put out the spoon and the plate. | 19.3 s | 7.1 s | 10.85 / 3.81 s | yes |
| 9 | Set the table. | 24.6 s | 7.8 s | 12.12 / 3.19 s | yes |
