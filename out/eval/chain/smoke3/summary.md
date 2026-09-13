# smoke3: seeds 101-101 (1), backend torch/cuda

levers: retry=['drawer', 'spoon'] exec={'spoon': {'temporal_coeff': 0.01}, 'drawer': {'temporal_coeff': 0.01}, 'plate': {'temporal_coeff': 0.01}, 'fork': {'n_action_steps': 50}, 'cup': {'n_action_steps': 10}} ckpt={'spoon': 'out/train/cutlery_t1/spoon/checkpoints/010000/pretrained_model', 'plate': 'out/train/plate_t1/plate/checkpoints/005000/pretrained_model', 'fork': 'out/train/cutlery_t1/fork/checkpoints/007500/pretrained_model'} avg={} budgets={} settle=30 home_frames=20 home_hold=0 release=0.35 threshold=0.5 camera_ends={'drawer': False, 'spoon': True, 'plate': True, 'fork': True, 'cup': True} home_until=60 retry_camera_end=False drawer_retry_release=1.0

**full tables 0/1** (Wilson 95% (0.0, 0.793)), mean steps 3.00

| step | ok | ended by camera | retries | recovered by retry |
|---|---|---|---|---|
| drawer | 1/1 | 0 | 1 | 0 |
| spoon | 0/1 | 0 | 1 | 0 |
| plate | 1/1 | 1 | 0 | 0 |
| fork | 0/1 | 0 | 0 | 0 |
| cup | 1/1 | 1 | 0 | 0 |

first failed step: spoon 1

| seed | ok | steps | drawer cm @spoon start | drawer | spoon | plate | fork | cup | retries |
|---|---|---|---|---|---|---|---|---|---|
| 101 | fail | 3 | 6.36 | ok b300 | X b450 p0.00 | ok c161 | X b500 p0.00 | ok c171 | 2 |
