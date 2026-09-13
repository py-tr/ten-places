# Ten Places

Two simulated SO-101 arms set a dinner table in MuJoCo from a spoken or typed command: open the drawer, hand the
spoon and the fork from one arm to the other, move the plate, set the cup. A small vision-language model plans the
steps, learned ACT policies drive both arms from the cameras, and a camera classifier checks every step — all on an
Intel CPU with OpenVINO.

Built for the Intel online challenge *Bimanual VLA Manipulation with Multi-Modal Reasoning* (AI Infra Summit
Hackathon 2026). Every number below comes from the script or result file named next to it; evaluation seeds 0–49
are never used for training or for choosing anything.

## How it works

```
"Set the table, but skip the cup."          (typed, or spoken → Speechmatics real-time STT)
        │
        ▼
Qwen3-VL-4B INT4 on OpenVINO GenAI  ── top-camera image + command → JSON plan (schema-constrained)
        │
        ▼
Verifier  ── adds physical prerequisites (cutlery needs the drawer open; the plate starts on the
             fork's spot), drops duplicates, fixes the order; every correction is shown
        │
        ▼
One ACT policy per skill (3 cameras + joints → 12 joint targets at 25 Hz), OpenVINO INT8 weights
        │   between skills: grippers released, arms home; after each: camera classifier (ResNet18, OpenVINO, 8 ms)
        │   → done / retry / re-plan; finished steps are re-checked before the next one
        ▼
MuJoCo: two SO-101 arms, randomised dinner table
```

**Why a VLM above small learned policies.** Language and scene understanding live in the vision-language model;
the visuomotor policies that move the arms are small ACT models (one of the candidate policies the challenge names),
trained here in MuJoCo with LeRobot. The split is a latency decision: at 16 ms per forward pass on the CPU (OpenVINO
INT8 weights), a policy can run every 40 ms control step and blend overlapping action chunks, which is what lets the
spoon hand-off complete (3/10 → 10/10 without it). A large end-to-end VLA predicts open-loop chunks — for scale,
Intel's π0.5 reference takes 294 ms per inference with stock PyTorch on a Core Ultra X7 358H at 40 W
([Intel](https://docs.openedgeplatform.intel.com/2026.1/OEP-articles/publications/optimizing-pi0.5-lva-model.html)).
The planner runs at the speed of a conversation (seconds), the policies at the speed of contact (25 Hz).

The person can keep talking while the robot works: "stop" halts at once; "skip the fork" or "oh, and the cup too"
changes the plan after the current step, verified like any plan. The robot answers with Speechmatics text-to-speech.

## Results

**Full table, 50 held-out randomised tables** (seeds 0–49, run once with every selection frozen beforehand;
`out/eval/final5/`, `out/eval/agent_table/ov_w8_report5.json`):

| Run | Full tables (95% CI) | Mean steps of 5 | Drawer | Spoon | Plate | Fork | Cup |
|---|---|---|---|---|---|---|---|
| **Full agent on OpenVINO** (what the robot runs: re-checks, retries, re-plans) | **43/50 (74–93%)** | 4.76 | 50 | 47 | 47 | 45 | 49 |
| Fixed five-step sequence, OpenVINO INT8 weights | 41/50 (69–90%) | 4.74 | 50 | 46 | 47 | 46 | 48 |
| Fixed five-step sequence, PyTorch reference | 39/50 (65–87%) | 4.62 | 49 | 46 | 45 | 43 | 48 |

OpenVINO INT8 and PyTorch are indistinguishable per seed (5 tables differ one way, 3 the other; McNemar p = 0.73).
The configuration was chosen on 50 separate tuning tables (seeds 100–149), where it set 41/50 with PyTorch. The
submission video shows the first 10 seeds as a grid with pass/fail per seed.

| Component | Result | Evidence |
|---|---|---|
| Planner: unseen commands → correct verified plan | 10/10; 8/8 on a later set, incl. naming what no skill can do ("dim the lights") | `scripts/eval_planner.py` |
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
| PyTorch FP32 eager (CPU) | 39–45 ms (45–48) | – | – |
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
slower planner (4.2 → 5.5 s). Hyper-threading and pinning trade places between runs (an earlier run: 10% vs 5% late);
the split itself helps in every run.

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
  weights on the same seeds, `docs/findings.md`), so they are not shipped.
- **Latency spent on quality.** At 16 ms the policy can run every control step with temporal ensembling, which is
  what lets the spoon hand-off complete (3/10 → 10/10).
- **Hybrid-core placement for concurrent workloads.** Real-time control on the P-cores, the VLM on the E-cores:
  the arms keep 25 Hz while the planner thinks (6–8 s for a plan and its check).
- **Energy.** INT8 weights use 2.5× less energy per inference than PyTorch on the same CPU (1.43 vs 3.50 J above
  idle); at the robot's 25 Hz, P-core placement draws 7 W less than default scheduling.
- Every model in the loop runs on OpenVINO: the policies (INT8 weights), the planner (Qwen3-VL-4B INT4, OpenVINO
  GenAI) and the camera classifier (ResNet18, 7.8 ms). Every script takes `--device`, so the same code targets the
  CPU, iGPU or NPU of a Core Ultra.

## The scene

Built from scratch with `mujoco.MjSpec` from the official SO-101 model (Apache-2.0); every prop is a MuJoCo
primitive.
- Arm A (x = −0.30 m) faces arm B (x = +0.30 m). **The hand-offs are forced by reach:** with the fingers pointing
  down neither arm can grasp past the table's midline, so cutlery that starts in A's drawer and must end on B's side
  has to change hands.
- The drawer is a tray under a fixed lid; the cutlery is only reachable once arm A has pulled it open.
- The stock finger collision meshes fill the gap between the jaws; they are replaced by box pads, with gripper force
  limited to a realistic ~17 N.
- Randomised per seed: drawer, cutlery, plate, cup and placemat positions, object sizes and masses, friction, light
  direction and intensity, table and floor colour.

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
make test             # 152 tests
make watch SEED=3                                        # scripted controller, live 3D viewer
make watch-agent CMD="just the plate and the cup" SEED=3 # VLM plan + learned policies, live
make agent CMD="set the table, but skip the cup" SEED=3  # rendered to out/video/ with the plan panel
make bench                                               # OpenVINO benchmark
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
- The robot does not yet repair a table knocked by someone else: the camera misses a plate slid off its mat, and the
  plate policy never learned to re-place one (`docs/findings.md`).
- Learned rollouts are repeatable only up to rendering (a new OpenGL context can shift a few pixels by one intensity
  level), so results are reported over 50 seeds with confidence intervals.
- Benchmarked on a desktop Intel CPU without iGPU or NPU; the Core Ultra iGPU/NPU paths are supported by `--device`
  but not measured here.

## Layout

```
tenplaces/scene.py, scene_table.py   procedural MJCF: arms, pads, table, props, cameras
tenplaces/ik.py, control.py          fingers-down IK for the 5-DOF SO-101; two-arm motion (demonstrator only)
tenplaces/oracle/                    scripted privileged-state controllers (demonstrations only)
tenplaces/env.py, env_table.py       25 Hz episode environments, demo recording, hand-over between skills
tenplaces/planner.py                 VLM planner (OpenVINO GenAI) + symbolic verifier
tenplaces/state_classifier.py        camera-only task-state classifier (OpenVINO)
tenplaces/agent.py                   command → plan → per-skill policy → check → retry / re-plan
tenplaces/voice.py, listen.py, speak.py   Speechmatics speech-to-text, live listening, text-to-speech
tenplaces/evaluate*.py, grader*.py   closed-loop evaluation and success predicates
tenplaces/lerobot_policy.py, skill_policies.py, ov_backend.py   policies and their OpenVINO backends
tenplaces/cores.py                   hybrid-core placement for control and planner
tenplaces/demo_video.py              demo renderer with the live plan panel
```

| Challenge deliverable | Where |
|---|---|
| Reproducible repository | this repo, `Makefile` |
| MuJoCo simulation with randomisation and evaluation config | `tenplaces/scene_table.py`, `tenplaces/randomize.py`, `scripts/final_report.py` |
| Intel inference benchmark | `scripts/benchmark.py`, `scripts/bench_concurrency.py` |
| Video across 10 randomised seeds | `scripts/final_report.py --videos 10` + `scripts/make_grid_video.py` |
| Technical README / architecture | this file, `docs/findings.md` |

## Licence

MIT. SO-ARM100 assets: Apache-2.0 (TheRobotStudio). Qwen3-VL-4B: Apache-2.0.
