# Ten Places

Two simulated SO-101 arms set a dinner table in MuJoCo from a spoken or typed command: open the drawer, hand the
spoon and the fork from one arm to the other, move the plate, set the cup. A small vision-language model plans the
steps, learned ACT policies drive both arms from the cameras, and a camera classifier checks every step — all on an
Intel CPU with OpenVINO.

**86 of 100 held-out tables set completely** (two sets of 50 randomised tables, frozen configuration, each run once).
Every grasp is friction contact with a ~17 N gripper — no weld or attach constraint anywhere — and at run time the
robot sees only its cameras and its joint angles, never the simulator's object poses. It is a hierarchical VLA: a
vision-language model plans from the top camera and the command, and small ACT policies act from three cameras at
25 Hz. The skills themselves are not language-conditioned — a single policy given the skill as an input learned to
ignore it (`docs/findings.md`) — so language reaches the arms through the plan.

Built for the Intel online challenge *Bimanual VLA Manipulation with Multi-Modal Reasoning* (AI Infra Summit
Hackathon 2026). Every number below comes from the script or result file named next to it. Evaluation seeds 0–49 are
never used for training; choices are made on tuning seeds 100–149 (two early ones, made on seeds 0–29, were re-checked
there and held — `docs/findings.md`).

## How it works

```
"Set the table, but skip the cup."          (typed, or spoken → Speechmatics real-time STT)
        │   first look: the camera classifier reads the table; steps already done are not planned again
        ▼
Qwen3-VL-4B INT4 on OpenVINO GenAI  ── top-camera image + command → JSON plan (schema-constrained)
        │
        ▼
Verifier  ── adds physical prerequisites (cutlery needs the drawer open; the plate starts on the
             fork's spot), drops duplicates, fixes the order; every correction is shown
        │
        ▼
One ACT policy per skill (3 cameras + joints → 12 joint targets at 25 Hz), OpenVINO INT8 weights
        │   between skills: grippers released, arms home; after each: camera classifier (ResNet18, OpenVINO, 5–6 ms)
        │   → done / retry / re-queue the unfinished steps; finished steps are re-checked before the next one
        ▼
MuJoCo: two SO-101 arms, randomised dinner table
```

**Why a VLM above small learned policies.** Language and scene understanding live in the vision-language model;
the visuomotor policies that move the arms are small ACT models (one of the candidate policies the challenge names),
trained here in MuJoCo with LeRobot. The split is a latency decision: at 16 ms per forward pass on the CPU (OpenVINO
INT8 weights), a policy can run every 40 ms control step and blend overlapping action chunks, which keeps it
closed-loop through contact. On the 50 tuning tables the spoon hand-off fails when the policy re-plans every 10
actions (4/50) and completes with ensembling (43/50), as it does executing whole 50-action chunks (44/50); on the
drawer ensembling is ahead (20/20 against 18/20 and 15/20, seeds 100–119). A large end-to-end VLA predicts open-loop chunks — for scale,
Intel's π0.5 reference takes 294 ms per inference with stock PyTorch on a Core Ultra X7 358H at 40 W
([Intel](https://docs.openedgeplatform.intel.com/2026.1/OEP-articles/publications/optimizing-pi0.5-lva-model.html)).
Measured on this i5: LeRobot's SmolVLA (450M parameters) takes 6.6 s per 50-action chunk with stock PyTorch —
against 40 ms for our ACT policy in PyTorch and 16 ms with OpenVINO INT8 weights (`scripts/bench_smolvla.py`,
`out/benchmark/smolvla.md`; timing only: the public base checkpoint, not trained on this task, not exported to
OpenVINO).
The planner runs at the speed of a conversation (seconds), the policies at the speed of contact (25 Hz).

The person can keep talking while the robot works: "stop" halts at once; "skip the fork" or "oh, and the cup too"
changes the plan after the current step, verified like any plan. The robot answers with Speechmatics text-to-speech.

## Results

**Full table, 50 held-out randomised tables** (seeds 0–49, run once with every selection frozen beforehand;
`out/eval/final5/`, `out/eval/agent_table/ov_w8_report5.json`):

| Run | Full tables (95% CI) | Mean steps of 5 | Drawer | Spoon | Plate | Fork | Cup |
|---|---|---|---|---|---|---|---|
| **Full agent on OpenVINO** (what the robot runs: re-checks, repairs, re-queues failed steps) | **43/50 (74–93%)** | 4.76 | 50 | 47 | 47 | 45 | 49 |
| Fixed five-step sequence (plate, fork, cup retried once), OpenVINO INT8 weights | 41/50 (69–90%) | 4.74 | 50 | 46 | 47 | 46 | 48 |
| Fixed five-step sequence, PyTorch reference (CUDA GPU) | 39/50 (65–87%) | 4.62 | 49 | 46 | 45 | 43 | 48 |

Task success does not depend on inference speed: the simulation waits for each action, so the GPU reference and the
CPU rows differ only in the numbers the networks compute.

OpenVINO INT8 and PyTorch are indistinguishable per seed (5 tables differ one way, 3 the other; McNemar p = 0.73).
The configuration was chosen on 50 separate tuning tables (seeds 100–149), where it set 41/50 with PyTorch.

**A second held-out set, and tables outside the training ranges** — 50 fresh seeds (200–249) never used before, the
same frozen configuration, each run once (`out/eval/agent_table/ov_w8_fresh200-249.json`, `out/eval/stress_*/`):

| Run on seeds 200–249 | Full tables (95% CI) |
|---|---|
| **Full agent on OpenVINO** | **43/50 (74–93%)** |
| Fixed five-step sequence, OpenVINO | 33/50 (52–78%) |
| Fixed sequence, friction, masses, light and colours widened ×1.5 beyond the training ranges | 41/50 (69–90%) |

The agent replicates its 43/50; here its re-checks, retries and re-queued steps add 10 tables and lose none (McNemar
p = 0.002). Widening the ranges cost nothing measurable (paired with the normal ranges: 14 tables better, 6 worse,
p = 0.12). Over both held-out sets the full agent sets 86 of 100 tables. The submission video shows the first 10
seeds as a grid with pass/fail per seed. (The first look, on in `run_agent.py`, is off in these rows; on fresh tables
it skips nothing — `docs/findings.md`.)

**Placement accuracy**, full agent over the 100 held-out tables (the simulator's measurement): median error spoon
0.33 cm, plate 0.41 cm, cup 0.44 cm, fork 0.78 cm; every placed object within the 2.5 cm tolerance (largest 2.45 cm);
184 of 200 hand-offs completed.

| Component | Result | Evidence |
|---|---|---|
| Demonstration runs: 10 tables, 10 requests fixed before recording (2 spoken, 1 changed mid-run by a scripted sentence), full agent on OpenVINO | 9/10 done exactly as asked; seed 4's plate missed four times, each miss caught by the camera | `scripts/score_demo.py`, `out/video/demo/summary.md` |
| Planner: unseen commands → correct verified plan | 8/8 on the set written before it was scored, incl. naming what no skill can do ("dim the lights"); 10/10, 5/5 and 4/6 on the three sets used while writing the prompts | `scripts/eval_planner.py` |
| First look: steps already done are skipped | fresh tables: none read as done (50/50); half-set tables: read exactly 50/50; agent on 50 fresh tables: skipped nothing | `scripts/eval_initial_state.py`, `eval_agent_table.py --look-first` |
| A VLA baseline: SmolVLA (450M) fine-tuned on the same 100 cup demonstrations, same 30 tuning tables, all seven starts | 63/210 cups placed (ACT on the same data: 167/210; the deployed ACT cup: 198/210); 6.6 s per chunk on this CPU against 16 ms | `scripts/train_smolvla.py`, `out/eval/context/cup_smolvla005000_*` |
| Mid-run spoken changes understood | 8/10 on sentences written before the run | `scripts/eval_amend.py --set fresh` |
| Camera classifier on learned-policy states | false "drawer done" 3/363, false "spoon done" 1/671 | `docs/findings.md` |
| Scripted demonstrator (training data) | 60/60 full tables, 72/72 verified subset plans | `make spike-table` |

How the system got from 0 to 43 of 50 tables — every change, what it measured, and what did not work —
is in [`docs/findings.md`](docs/findings.md).

## OpenVINO on an Intel Core i5-13600KF

**Policy latency**, the five deployed skill policies, idle machine, batch 1 (`scripts/benchmark.py`,
`out/benchmark/<skill>_<step>.md`):

| Variant | Latency median (p95) | Throughput | IR size |
|---|---|---|---|
| PyTorch FP32 eager (CPU, its default 14 threads) | 39–45 ms (45–48) | – | – |
| OpenVINO FP32 (drawer, cup) | 18.4–19.5 ms (21–22) | 85–92 inf/s | 130.5 MB |
| **OpenVINO INT8 weights** (deployed) | **15.5–15.8 ms** (17–20) | 109–112 inf/s | 33.0 MB |
| … E-cores only | 37–66 ms | – | – |

**Control and planner at the same time** — policy with temporal ensembling + camera check at 25 Hz (40 ms budget)
while the 4B planner generates, 60 s per placement (`scripts/bench_concurrency.py`, `out/benchmark/concurrency.md`):

| Placement | Control step p50 / p99 | Steps over 40 ms | Planner answer (median) |
|---|---|---|---|
| Control alone, default scheduling | 24 / 36 ms | 0.1% | – |
| Control + planner, default scheduling | 53 / 70 ms | 97% | 4.2 s |
| P-cores for control, E-cores for the planner | 30 / 54 ms | 11% | 5.5 s |
| … + hyper-threading on for control | **29 / 36 ms** | **0%** | 5.5 s |
| … + pinned threads (`--cores split`, `tenplaces/cores.py`) | 29 / 55 ms | 6% | 5.9 s |

Left to the OS, the planner makes almost every control step late. Splitting the cores fixes that at the cost of a
slower planner (4.2 → 5.5 s). The robot ships the pinned placement (6% late); hyper-threading on measured 0% in this
run and is not shipped.

**Power and energy** — CPU package power logged by HWiNFO64 (package, not wall power), cup policy, 60 s per phase
(`scripts/power_bench.py`, `out/benchmark/power.md`):

| Phase | Package power | Inferences/s | Energy per inference, above idle |
|---|---|---|---|
| Idle | 18.0 W | – | – |
| PyTorch FP32, back to back | 108.9 W | 26 | 3.50 J |
| OpenVINO FP32, back to back | 109.3 W | 53 | 1.73 J |
| **OpenVINO INT8 weights, back to back** | 106.6 W | 62 | **1.43 J** |
| OpenVINO INT8 weights at 25 Hz, default scheduling | 63.3 W | 25 | 1.81 J |
| … at 25 Hz, P-cores, pinned | 56.0 W | 25 | 1.52 J |

What the optimisation buys:
- **Precision chosen by task success, not output error.** INT8 weights keep full-table success (41 vs 39 of 50);
  INT8 activations in the transformer cost it (hand-off checkpoint: 7/20 against 13/20 for FP32 and 14/20 for INT8
  weights on the same seeds, `docs/findings.md`), so they are not shipped. INT4 weights (23 MB instead of 33) keep
  four skills but break the cup — its actions drift ~0.05 rad — so the full table fails every time (0/50 against
  41/50 on the tuning seeds): the ladder stops at INT8 weights.
- **Latency spent on quality.** At 16 ms the policy can run every control step with temporal ensembling: the drawer
  20/20 against 18/20 open-loop and 15/20 re-planning every 10 actions (seeds 100–119); the spoon hand-off 43/50
  against 4/50 re-planning every 10 actions (open-loop whole chunks: 44/50, seeds 100–149).
- **Hybrid-core placement for concurrent workloads.** Real-time control on the P-cores, the VLM on the E-cores:
  the arms keep 25 Hz while the planner thinks. The first plan comes while the arms are still, so it runs on
  every core: median 7.1 s against 14.3 s on the E-cores, same plans (10 demo commands,
  `scripts/bench_planner_placement.py`). Everything asked while the arms move — the check for impossible parts
  ("light a candle"), spoken changes — stays on the E-cores.
- **Energy.** INT8 weights use 2.4× less energy per inference than PyTorch at its default 14 threads on the same
  CPU (1.43 vs 3.50 J above idle); at the robot's 25 Hz, P-core placement draws 7 W less than default scheduling.
- Every model that runs on the machine runs on OpenVINO: the policies (INT8 weights), the planner (Qwen3-VL-4B INT4,
  OpenVINO GenAI) and the camera classifier (ResNet18, 5–6 ms); speech is Speechmatics' cloud service. The benchmark
  and evaluation scripts take `--device` and `benchmark.py` lists the machine's OpenVINO devices; the hybrid-core
  placement is CPU-only, and the Core Ultra iGPU/NPU paths are untested here.
- **Model cache, measured rather than assumed.** OpenVINO's `CACHE_DIR` gets each skill policy ready in 0.16–0.17 s
  instead of 0.83–0.86 s (5×) — but the 4B planner loads slower from its 3 GB cache (8.8 s) than it compiles from
  its IR (4.7 s). The robot does not use the cache yet: the policies would save ~3.4 s per launch, the planner would
  lose ~4 s (`scripts/bench_compile_cache.py`, `out/benchmark/compile_cache.md`).

## The scene

Built from scratch with `mujoco.MjSpec` from the official SO-101 model (Apache-2.0); every prop is a MuJoCo
primitive.
- Arm A (x = −0.30 m) faces arm B (x = +0.30 m). **The hand-offs are forced by reach:** with the fingers pointing
  down neither arm can grasp past the table's midline, so cutlery that starts in A's drawer and must end on B's side
  has to change hands.
- The drawer is a tray under a fixed lid; the cutlery is only reachable once arm A has pulled it open.
- The stock finger collision meshes fill the gap between the jaws; they are replaced by box pads, with gripper force
  limited to a realistic ~17 N.
- Randomised per seed, uniformly (`tenplaces/scene_table.py`, `sample()`; positions in the table frame; object sizes
  are fixed):

  | What | Range |
  |---|---|
  | Drawer (closed tray centre) | x −15 to 0 mm, y 70 to 100 mm |
  | Spoon and fork in the tray | ±5 mm across; along: spoon 0 to 4 mm, fork 32 to 36 mm |
  | Plate start | x 150 to 190 mm, y 70 to 100 mm |
  | Cup start | x 190 to 205 mm, y −150 to −135 mm |
  | Placemat (the plate's target) | x 110 to 125 mm, y −15 to 15 mm |
  | Friction, every contact | 0.7 to 1.3 |
  | Mass: cutlery / plate / cup | 25–60 g / 80–160 g / 40–90 g |
  | Light | direction tilted up to ±0.5 in x and y from straight down; diffuse 0.4 to 0.9 |
  | Colours | table RGB 0.2–0.8 per channel, floor 0.1–0.5 |

## Training

- A scripted controller with privileged state (IK for the 5-DOF arm, closed-loop drawer pull) generates the
  demonstrations; it is never used at run time.
- One LeRobot ACT policy per skill, fine-tuned on demonstrations from every start a verified plan can produce, on
  layouts weighted toward the hard cases, and on takeover episodes (the learned policy starts, the scripted
  controller finishes).
- The camera classifier is trained on scripted runs labelled by the simulator, including drawer pulls that stop
  short, so "drawer done" means open far enough for the cutlery.

## Run it

```
make third-party      # the official SO-101 model (TheRobotStudio/SO-ARM100) at the pinned commit
pip install -r requirements-lock.txt
make models HF_SKILLS_REPO=<user>/<repo>   # trained skills + classifier, and the OpenVINO planner (~4.4 GB)
make test             # 176 tests
make watch SEED=3                                        # scripted controller, live 3D viewer
make watch-agent CMD="just the plate and the cup" SEED=3 # VLM plan + learned policies, live
make agent CMD="set the table, but skip the cup" SEED=3  # rendered to out/video/ with the plan panel
make bench                                               # OpenVINO benchmark of the five deployed policies
make report                                              # the 50 held-out tables (hours; PyTorch row needs CUDA)
make grid                                                # score the 10 demo runs, tile them into one video
```

`requirements-lock.txt` pins the exact environment every number was produced with (Python 3.13, openvino 2026.3.1,
openvino-genai 2026.3.1, nncf 3.3.0, mujoco 3.13.0, lerobot 0.6.1; torch is only needed for training).

Voice (set `SPEECHMATICS_API_KEY`):

```
python scripts/run_agent.py --mic --listen --speak --seed 3 --video out/video/voice_s3.mp4
```

Speak the command; the arms start as soon as the transcript is final. With `--listen` the microphone stays open:
"stop" halts at once (matched locally, no model call); anything else goes to the planner, is verified, and takes
over when the current step ends, so an object is never dropped mid-air. `--speak` lets the robot say its plan and
the reason for any correction ("I'll open the drawer first — the spoon is inside") and what it cannot do. Speech runs
on worker threads, so the control loop never waits for the network. The hybrid-core placement above is on by
default (`--cores default` turns it off).

## Limitations

- 7 of 50 held-out tables are not set completely; the losses are spread over the spoon hand-off (3), the plate (2)
  and the fork (2).
- The first look skips what is already done, but a table half-set out of the order the skills were trained in (the
  plate already out before the spoon) can make a later skill fail; the verifier flags such orders.
- A plate knocked off its mat is noticed 27 times out of 75 and put back twice by the deployed robot. An experimental
  classifier and plate fine-tune reach 77/80 noticed and 30/80 put back — below the bar set before the attempt (60/80),
  so they are not shipped (`docs/findings.md`).
- Learned rollouts are repeatable only up to rendering (a new OpenGL context can shift a few pixels by one intensity
  level), so results are reported over 50 seeds with confidence intervals.
- Measured on a desktop Intel CPU without iGPU or NPU (no Core Ultra was available); on a Core Ultra,
  `scripts/benchmark.py --device GPU` or `NPU` is the path to try, untested here.

## Layout

```
tenplaces/scene.py, scene_table.py   procedural MJCF: arms, pads, table, props, cameras
tenplaces/ik.py, control.py          fingers-down IK for the 5-DOF SO-101; two-arm motion (demonstrator only)
tenplaces/oracle/                    scripted privileged-state controllers (demonstrations only)
tenplaces/env.py, env_table.py       25 Hz episode environments, demo recording, hand-over between skills
tenplaces/planner.py                 VLM planner (OpenVINO GenAI) + symbolic verifier
tenplaces/state_classifier.py        camera-only task-state classifier (OpenVINO)
tenplaces/agent.py                   command → plan → per-skill policy → check → retry / re-queue
tenplaces/voice.py, listen.py, speak.py   Speechmatics speech-to-text, live listening, text-to-speech
tenplaces/evaluate*.py, grader*.py   closed-loop evaluation and success predicates
tenplaces/lerobot_policy.py, skill_policies.py, ov_backend.py   policies and their OpenVINO backends
tenplaces/cores.py                   hybrid-core placement for control and planner
tenplaces/demo_video.py              demo renderer with the live plan panel
```

| Challenge deliverable | Where |
|---|---|
| Reproducible repository | this repo, `Makefile` |
| MuJoCo simulation with randomisation and evaluation config | `tenplaces/scene_table.py` (`sample()`), `scripts/final_report.py` |
| Intel inference benchmark | `scripts/benchmark.py`, `scripts/bench_concurrency.py` |
| Video across 10 randomised seeds | `scripts/final_report.py --videos 10` + `scripts/make_grid_video.py` |
| Technical README / architecture | this file, `docs/findings.md` |

## Licence

MIT. SO-ARM100 assets: Apache-2.0 (TheRobotStudio). Qwen3-VL-4B: Apache-2.0.
