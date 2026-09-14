# Findings

Engineering record behind the README: every change tried, its measurement, what did not work.
Seeds: tuning 100–199 and 300–399; held-out 0–49 and 200–249, run only with every selection frozen beforehand;
demonstrations 1000+. The script or result file is named next to each number.

## Architecture

- **One policy per skill, not one skill-conditioned policy.** A single ACT with a skill one-hot ignored it (same
  scene, each of the five one-hots → action chunks differ by ≤ 0.1) and inferred the phase from the cameras — not
  steerable by a planner. It also froze at the hand-off, where the demonstrations pause and the policy has no sense of
  time. One skill-conditioned ACT: 1/10 full tables, 2.3/5 steps (`scripts/eval_table_checkpoints.py`).
- **Context shift.** Skills trained only on full tables fail when a subset command skips a step (the cup, with the
  plate not moved). Fix: demonstrations from every start a verified plan can produce. Tuning seeds 100–129, 30 tables
  per start: plate not moved (3 starts) 54/90 → 80/90; plate moved (4 starts) 113/120 → 118/120
  (`scripts/eval_skill_context.py --seeds 100 130`, `out/eval/context/cup_*_seeds100-129.md`). First chosen on seeds
  0–9 (15/30 → 29/30), inside the held-out set; confirmed here on tuning seeds.
- **Temporal ensembling.** The spoon hand-off froze at the pause before arm A lets go when the policy re-planned
  every 10 actions. A forward pass every control step, blending overlapping chunks, carries the release through; no
  retraining. Tuning seeds 100–149, spoon: 4/50 every 10 actions, 43/50 ensembling, 44/50 whole 50-action chunks
  open-loop — the fix is not re-planning mid-hand-off. Drawer, seeds 100–119: 15/20 every 10, 18/20 whole chunks,
  20/20 ensembling (`scripts/eval_skill_variants.py`). One network call per 40 ms step: OpenVINO INT8 weights (16 ms)
  fits; PyTorch on the same CPU (39–45 ms) does not. SmolVLA (450M): 6.6 s per 50-action chunk on this CPU, PyTorch
  (`scripts/bench_smolvla.py`, timing only).
- **VLA baseline.** SmolVLA (450M; expert trained, vision encoder frozen, bf16), 5k steps on the same 100 full-table
  cup demonstrations the first ACT cup learned from (`scripts/train_smolvla.py`); all seven starts, tuning seeds
  100–129, same evaluation, 10 actions per chunk:

  | Cup, 30 tables per start | Plate already moved (4 starts) | Plate not moved (3 starts) | All |
  |---|---|---|---|
  | SmolVLA, 5k steps | 44/120 | 19/90 | 63/210 |
  | ACT, same data (skills_v1, 20k steps) | 113/120 | 54/90 | 167/210 |
  | ACT, plus demonstrations from every start (skills_ctx) | 118/120 | 80/90 | 198/210 |

  LeRobot's recipe trains SmolVLA 20k steps (2.5k: 2/21 placed; 5k: 63/210) — compared at a quarter of that budget.
  One SmolVLA chunk: 6.6 s (PyTorch, this CPU); ACT: 16 ms (OpenVINO).

## From skills alone to the chain

Each skill ~90–100% when started by the scripted controller; the chain of learned skills: 0 of 10 tables.

| Change | Result | Script |
|---|---|---|
| Per-skill ACT, each step alone from a realistic start (20k steps each), 10 held-out seeds | drawer 10/10, cup 10/10, plate 4/10, spoon 0/10, fork 0/10 | `scripts/watch_skill_evals.py` |
| Plate: demos weighted to hard layouts + takeover demos (learned policy starts, scripted controller finishes), 10k-step fine-tune, seeds 100–119 | 14/20 → **20/20** | `scripts/eval_skill_checkpoints.py` |
| Spoon vs drawer opening (8 / 9 / 10 cm), before → after demos with varied openings | 4/10, 10/10, 0/10 → 10/10, 10/10, 2/10 | `scripts/eval_skill_checkpoints.py --drawer-open` |
| Fork cut off by its time budget → longer caps, seeds 100–109 | fork 0 → 3/10 | `scripts/eval_table_budgets.py` |
| Grippers released, arms home between skills (every demonstration's start state), seeds 100–119 | 4/20 → **10/20** full tables, 2.70 → 4.05 of 5 steps | `scripts/eval_table_home.py` |
| Drawer runs its whole budget instead of stopping at the camera's "done" (≥ 6 cm; the cutlery needs ~7.4), seeds 100–119 | spoon after the learned drawer 12/20 → 16/20; 10/20 → 15/20 full tables | `scripts/eval_table_home.py` |
| Classifier v3: drawer "done" only when open enough for the cutlery, trained with stalled pulls; learned-policy states, seeds 120–139 | false "drawer done" 61 → 3 of 363 checks, false "spoon done" 10 → 1 of 671; agent 14/20 → 16/20 (seeds 100–119) | `make state-data state-clf` |
| How the drawer step ends, classifier v3, seeds 100–149 | budget: PyTorch 30/50, OpenVINO 31/50; camera: 30/50, 29/50 — no difference | `TENPLACES_CAMERA_ENDS=drawer=1 scripts/eval_table_home.py` |

Held-out seeds 0–49, each row run once with frozen selections (`scripts/final_report.py`, `scripts/eval_agent_table.py`):

| Configuration | PyTorch | OpenVINO INT8 weights | Full agent |
|---|---|---|---|
| Per-skill policies, no hand-over between skills | 2/50 | 5/50 | – |
| + release and home between skills | 23/50 | 22/50 | 24/50 (PyTorch) |
| + drawer runs its budget | 26/50 | 24/50 | 26/50 (PyTorch) |
| + classifier v3 | 24/50 | 23/50 | 26/50 (PyTorch) |
| + drawer fine-tuned on genuinely stalled pulls | 39/50 | 41/50 | 43/50 (OpenVINO) |
| + cup trained on varied sizes (re-run of both held-out sets) | 38/50 | 42/50 | 43/50 (OpenVINO) |
| + camera-ended drawer re-pull (final; re-run of both held-out sets) | 41/50 | 40/50 | **39/50 (OpenVINO)** |

- Rows 3–4 were chosen on 20 tuning seeds and do not show on the held-out seeds: vs the row above, PyTorch +7 / −4
  tables (McNemar p = 0.55). The OpenVINO drawer moved the wrong way, 48 → 42 → 39 of 50 (vs release-and-home: 0
  gained, 9 lost), not reproduced on the 50 tuning seeds (44 vs 44). 20 tuning seeds are too few to choose on; later
  decisions use 50 or more.
- Row 5, chosen on 50 tuning seeds (41/50), held: vs the classifier-v3 row, PyTorch +15 / −0 (p < 0.001),
  OpenVINO +20 / −2 (p < 0.001), full agent 26 → 43 (+21 / −4).
- Row 6: the one change is the cup (below); the rest is run-to-run noise (± a few tables, see Repeatability).
- Row 7: the drawer re-pull (below). Vs row 6: PyTorch +6 / −3, OpenVINO +3 / −5, agent +3 / −7 — within noise; on
  seeds 200–249 the fixed sequence 30 → 40/50 (+11 / −1, p = 0.006).

**Execution settings re-checked on tuning seeds.** Spoon, plate, fork and cup had their execution setting chosen on
seeds 20–29 — inside today's held-out set. Re-run on 100–149 with the deployed checkpoints
(`eval_skill_variants.py --deployed --seeds 100 150`); rule fixed before the last results: keep the current setting
unless another is better by a paired McNemar test at p < 0.05.

| Skill | every 10 actions | whole 50-action chunks | ensembling | current | verdict |
|---|---|---|---|---|---|
| spoon | 4/50 | 44/50 | 43/50 | ensembling | keep (+3 / −2, p = 1.00) |
| plate | 14/50 | 40/50 | 45/50 | ensembling | keep (best) |
| fork | 42/50 | 49/50 | 49/50 | whole chunks | keep (tied best) |
| cup | 49/50 | 50/50 | 50/50 | every 10 | keep (+1 / −0, p = 1.00) |

Every setting holds. `--select` refuses seeds below 100.

**Drawer demos from genuinely stalled pulls.** 100 new drawer demos (`scripts/record_chain_demos.py`): 60 start
where a learned pull stalled (release + home, then the scripted controller re-grips and finishes), 40 plain pulls
from a closed drawer; fine-tuned from the deployed drawer.
- With the demo set's own statistics: the drawer froze in one pose on every table (0/50). Arm B never moves in drawer
  demos, so several of its joint spreads were float32 noise (~0); MEAN_STD normalisation divides the policy's own
  1e-5–1e-4 rad arm-B deviation by ~1e-8. Reproduced offline: a 1e-4 rad nudge on one arm-B joint sends the predicted
  arm-A motion to the frozen pose.
- Same data, the parent's statistics (`scripts/use_parent_stats.py`; `train_skills.py` refuses a demo set with a
  near-zero spread), 7.5k steps, seeds 100–149: drawer 44 → 48/50, spoon 36 → 45, **full tables 30 → 41/50**
  (Wilson 69–90%); with a drawer retry 43/50. The spoon's losses were stalled drawers.
- Spoon fine-tune on 190 new demos (after the learned drawer, openings 7.2–10.2 cm): worse (20–22/50); not used.
- Idle joints fed their training mean (`LeRobotPolicy(mask_idle_std=1e-4)`, `eval_table_chain.py --mask-idle`): the
  broken first attempt's drawer 0/50 → 18/20. On top of the new drawer: 34/50 vs 41/50; off.

**Drawer re-pull.** Tuning seeds 100–199, spoon first attempt by the drawer opening at its start: < 7.4 cm 0/7,
≥ 7.4 cm 88/93 (`out/eval/chain/lever1_base/rows.json`). Seven short drawers (5.9–7.0 cm, friction 1.03–1.13)
pass the grader's 6 cm and are graded as spoon losses. The fixed sequence never retried the drawer; the agent's
re-queued drawer ran its whole budget from a half-open tray. Fix: a failed drawer check triggers a second pull that
ends at the camera's "done" (`RETRY["drawer"]`, `RETRY_CAMERA_END`, `evaluate_table.ends_on_camera`).
- Gate written first: ≥ 3 of 100 tables completed after the second pull (seeds 100–199). Result: 3 (109, 131, 175;
  175 short in this run only). Re-pulled on 7: camera-ended at 8.6–9.2 cm on five, budget-ended at 7.4 and 7.1 cm on
  two; spoon after it 6/7. Run 72/100 vs 70/100 (+9 / −7).
- A wide-open drawer read as "not done": once in 100 (7.57 cm, p = 0.28); the re-pull moved it 0 mm. Every pull
  ≥ 7.6 cm read p ≥ 0.80.
- Shipped. Held-out re-run: fixed sequence, seeds 200–249, 30 → 40/50 (+11 / −1, p = 0.006); seeds 0–49 42 → 40/50
  (+3 / −5). Full agent 39 → 42 (+7 / −4) and 43 → 39 (+3 / −7); both sets 82 → 81/100.
- Agent, tuning seeds 100–149 (reported, not gating; rows list the retried skills): 40/50 vs 41/50 without; the drawer
  re-pulled on 2 tables (100, 109), both completed. Held-out sizes ±20% row: re-pulled on 8 tables, spoon ok after it
  on 7.

**Plate without the spoon step.** Demo seed 4 ("Just the plate and the cup.") lost the plate four times. The plate
after the drawer alone and after drawer + spoon (scripted prefix, seeds 100–129, `eval_skill_context.py --deployed`):
30/30 each. Seed 4 is a hard plate table — fails from both starts, and in every run on seeds 0–49; not tuned on.

**Where the chain lost (before the drawer fix).** Fork, mostly downstream: of its 25 failures (PyTorch, seeds 0–49,
release + home), 15 on seeds where the spoon failed, 4 where the plate failed (the plate starts on the fork's spot),
6 its own. Drawer pull stalls at 5–6.5 cm, mostly on high-friction tables (opening vs friction: Spearman −0.58).
Started from a drawer set 5 / 6 / 6.5 cm open, the drawer policy reaches ≥ 7.4 cm on 18 / 19 / 20 of 20; inside the
chain a retry recovered 0 of 9 real stalls (`scripts/eval_table_chain.py --retry`, seeds 100–149): a tray dragged
there under high friction, arm still on the handle, is not the same start as a tray set there.

## Second held-out set, robustness, object sizes

Fresh seeds 200–249, untouched by any step of the work; frozen configuration, each run once (`final_report.py`
refuses the tuning range; `scene_table.sample(stress, shape)`).

- Before the cup change: full agent 43/50 (74–93%); both held-out sets 86/100. Fixed sequence 33/50 (52–78%);
  agent vs fixed +10 / −0 (p = 0.002).
- Size-trained cup, re-run: full agent 39/50 (65–87%), fixed sequence 30/50 (46–72%); agent vs
  fixed +11 / −2 (p = 0.02). Both held-out sets: 82/100. Vs before, per seed: 0–49 +3 / −3 (lost 11 plate, 25 cup,
  46 spoon), 200–249 +1 / −5 (lost 207, 239 plate; 228, 245 spoon; 243 cup; p = 0.22). Two of the eight lost tables
  are cup misses (25, 243): cup 97/100 against 99/100, within the noise of a step at 97–99%. The other six are steps
  the change does not run.
- Final (drawer re-pull on, re-run): full agent 42/50 (72–92%), fixed sequence 40/50 (67–89%); agent vs fixed
  +5 / −3 (p = 0.73). Both held-out sets: 81/100.
- Robustness rows, previous cup, paired with the fixed sequence at the training ranges (33/50):
  friction, the three masses, light and colours widened ×1.5 about their centres (placements unchanged): 41/50
  (+14 / −6, p = 0.12); ×2.0: 33/50 (+10 / −10, p = 1.0).
- Object sizes (`TENPLACES_SHAPE`: plate radius, cup radius and height ×(1 ± shape); cutlery length ±min(shape, 10%)
  for the 16 cm tray; everything else identical to the paired table), previous cup: ±10% 25/50 (+4 / −12,
  p = 0.08); ±20% 17/50 (+2 / −18, p < 0.001); cup size only ±20% (`TENPLACES_SHAPE_ONLY=cup`) 18/50 (+2 / −17,
  p < 0.001), cup placed 23/50 — the cup carries nearly the whole loss. At ±20%, cups placed by the size of the cup
  (tables the trained sizes set): within ±5% 9/11, 5–10% off 9/12, > 10% smaller 1/11, > 10% larger 4/13. Plate:
  fails > 10% smaller (0/5), holds larger (12/13). Longer cutlery (+5–10%) costs the fork (5/9).
- Grader size-proof: a cup or plate set exactly on its target passes on all 50 tables at ±20% and at trained sizes.
- Robustness rows, size-trained cup before the re-pull, paired with the fixed sequence at the training ranges and
  sizes (30/50): ×1.5 40/50
  (+17 / −7, p = 0.06); ×2.0 38/50 (+15 / −7, p = 0.13); sizes ±10% 27/50 (+4 / −7, p = 0.55); ±20% 26/50 (+4 / −8,
  p = 0.39), cup placed 40/50 (previous cup 25/50); cup size only ±20% 22/50 (+3 / −11, p = 0.06), cup placed 34/50
  (previous 23/50). Same cup sizes in the last two rows, cup placed 40 vs 34: run-to-run noise at this size. Gap
  smaller, not closed.
- Robustness rows, final (re-pull on), paired with the same run at the training ranges and sizes. Fixed sequence
  (40/50): ×1.5 43/50 (+9 / −6, p = 0.61), ×2.0 35/50 (+7 / −12, p = 0.36), sizes ±10% 28/50 (+2 / −14,
  p = 0.004), ±20% 23/50 (+2 / −19, p < 0.001), cup size only ±20% 25/50 (+0 / −15, p < 0.001). Full agent
  (42/50): ×2.0 36/50 (+6 / −12, p = 0.24), sizes ±20% 23/50 (+1 / −20, p < 0.001). Against a 40/50 baseline the
  size loss is clear; the earlier "±20% not significant" sat on a 30/50 baseline.

**Cup trained on varied sizes.** Gates written before any result, all on tuning seeds. The deployed cup's misses at
other sizes: all "never picked up", none a false "done" from the camera.
- Scripted controller first: its fixed 5.8 cm approach hit the rim of larger cups (×1.2: 3/20 placed). Approach
  raised by the extra rim height for cups above the trained size (×1.0 unchanged): 20/20 at every scale ×0.75–1.25.
- Attempt 1: 100 demonstrations, cup ×0.77–1.25 (40 at the trained size); fine-tuned 7.5k steps, augmentation,
  AMP, lr 2e-5. Worse at every size: cup alone, trained size 37/50 vs 48/50; ±20% cup sizes 24/50 vs 29/50; fixed
  sequence 22/50 vs 42/50. Not shipped.
- Attempt 2, one variable against attempt 1's loss at the trained size: + the deployed cup's own 80 demonstrations
  (180); the deployed cup's recipe (no augmentation, no AMP, default lr); its statistics; final checkpoint only.
  Trained size 48/50 vs 48/50; fixed sequence 43/50 vs 42/50 (cup 49 vs 47); ±20% cup sizes 35/50 vs 29/50
  (+12 / −6, p = 0.24) — gate p < 0.05 missed.
- Confirmation, written before it ran: the same checkpoint on 150 untouched tuning seeds (150–199, 300–399), cup
  alone, ±20% cup sizes: 99 vs 80 (+31 / −12, p = 0.005). Passed; with the other two gates holding, it replaced the
  deployed cup. The first 50-seed test not pooled in.

**Lost tables, final re-run** (full agent, OpenVINO, `out/eval/agent_table/ov_w8_report7.json`,
`ov_w8_fresh200-249_v7.json`; a table counts at its first step, in task order, still undone at the end):

| Step | Tables | Seeds | Final distance from target |
|---|---|---|---|
| plate | 8 | 11, 204, 239, 247 · 12, 207, 229 · 233 | 16.6–20.8 cm: not carried · 3.6–4.2 cm off the mat · 0.9 cm: within the distance, failed another check (flat, upright or released) |
| spoon | 5 | 1, 36, 37, 47, 214 | 22.6–25.3 cm: not picked up |
| fork | 4 | 4, 18 · 44, 222 | 16.7–19.3 cm: not carried · 2.9–3.6 cm, just outside the 2.5 cm tolerance |
| cup | 2 | 41 · 45 | 5.2 cm · 1.4 cm: within the distance, failed another check |

No table lost at the drawer. Largest single losses: the plate (8), then the spoon (5).

Mechanisms, tuning seeds 100–199 (fixed sequence, first attempts, recorded start/end states, `out/eval/chain/lever1_base/rows.json`):
- Spoon: drawer at the spoon's start < 7.4 cm 0/7 placed, ≥ 7.4 cm 88/93. The seven short drawers (5.9–7.0 cm,
  friction 1.03–1.13) pass the grader's 6 cm and are graded as spoon losses; the other five misses are grasps that move
  the spoon ≤ 7 mm (spoon far toward arm A, drawer at the low end of its y range).
- Plate, 13 failures: 8 grasped and never lifted (held, moved < 3 cm, budget out); 3 not carried; 1 carried 7 cm and
  still held; 1 upside down.
- Fork, 22 failures: short drawer 5; dropped or flipped in transit 10 (5 upside down); still held when the budget ran
  out 4 (hand-off stall at the release); not carried 3.

## Planner

- **Plan first, check later.** Plan and impossible-part check both before moving: 18.7 s median to the first
  motion. Now the plan alone starts the arms (13.5 s on the E-cores, seed 3, "I want to eat soup."); the text-only
  check answers 4.5 s later on the planner's worker thread.
- **First plan on every core.** Arms still until the first plan, so a second planner pipeline on every core
  (`VLMPlanner(idle_config={})`) makes it; the E-core pipeline keeps what runs while the arms move. 10 demo commands:
  median 14.3 → 7.1 s, same plan 10/10 (`scripts/bench_planner_placement.py`); full agent with the control models
  loaded, seed 3: 9.0 s instead of 13.5 s. Cost: a second copy of the model in memory (~3 GB).
- **A dangling "and" emptied the plan.** "Set the table and light a candle." → the 4B model returned no steps; the
  re-plan without the impossible part used "Set the table and .", also read as incomplete — nothing done. Found on
  demo seed 6's pre-registered command; the re-plan drops the dangling join ("Set the table."): dev set 4/5 → 5/5,
  other sets unchanged; seed 6 sets the full table and names the candle (`scripts/eval_planner.py`,
  `tests/test_planner.py`).
- **Czech commands.** Six commands, same prompt: plans right 6/6; the impossible-part check refused 4 of 6 it should
  have accepted — 2/6 end to end. Not shipped.

## First look

Before planning, the camera classifier reads the table once; steps seen done with probability ≥ 0.95 go to the
planner as done (`run_command(look_first=0.95)`), said aloud, shown on the panel, re-checked like any done step.
Gates, tuning seeds 100–149 (`scripts/eval_initial_state.py`): fresh tables — highest probability for any step
0.003 (v3), none read as done; tables half-set by the scripted controller (ten prefixes) — 50/50 read exactly,
lowest probability for a done step 0.990. Agent, 50 fresh tuning tables, with and without: nothing skipped (41/50
without, 39/50 with; the 10 differing tables, 4 one way, 6 the other, are run-to-run noise; p = 0.75). On by default
in `run_agent.py`; off in the reported evaluations.

Half-set tables outside what the skills learned: tuning seed 120, drawer and plate already done — both seen
(p = 1.0), planner proposed spoon, fork, cup; the spoon failed twice (every spoon demonstration had the plate still at
its start). The verifier flags the order ("'spoon' after 'plate' is outside the order tested collision-free"). The
extra demonstration half-sets the table in the trained order (drawer and spoon done).

## OpenVINO precision

One hand-off checkpoint (40k steps), 20 held-out seeds, task success re-measured in closed loop per precision
(`scripts/int8_study.py`, `out/int8_study/040000/int8_study.json`; latencies measured while training ran):

| Variant | Latency | Success (Wilson 95%) |
|---|---|---|
| OpenVINO FP32 | 33.7 ms | 13/20 (0.43–0.82) |
| INT8 weights only | 28.6 ms | 14/20 (0.48–0.86) |
| INT8 image encoder (weights + activations), transformer float | 20.2 ms | 12/20 (0.39–0.78) |
| INT8 everything (weights + activations) | 13.8 ms | 7/20 (0.18–0.57) |

Quantising the transformer's activations costs task success; INT8 weights keep it.

**INT4 weights** (`nncf` INT4_ASYM, groups of 64; the three input projections not a multiple of 64 stay INT8):
23 MB instead of 33 MB per policy. Full agent, tuning seeds 100–149, the five policies of that time in INT4: 0/50
full tables vs 41/50 INT8 — drawer (50), spoon (44), plate (45), fork (48) hold; the cup fails every table. On the
same half-set tables the cup's first actions differ from INT8's by up to 0.05 rad, a steady drift in arm B's shoulder
pan and wrist — enough to miss a small cup and its 2.5 cm target (`eval_agent_table.py --backend ov-w4`). No faster:
one benchmark run, INT4 16.8–21.3 ms vs INT8 weights 16.8–19.3 ms (`out/benchmark/opt_out_0914/`). The 75k hand-off
checkpoint: 19/20 on both FP32 and INT8 weights (`scripts/eval_checkpoints.py`). Deployed: INT8 weights (15.5–16 ms,
idle machine).

## Disturbance repair (one attempt, not shipped)

Every finished step is re-checked before the next one and at the end; a step that no longer holds goes back to the
front of the queue. Plate pushed 7 cm right after placing, four directions, tuning seeds 120–139, OpenVINO
(`scripts/eval_agent_table.py --push after-plate:plate:DX:DY`):

| System | Knock noticed | Plate put back | put back: +x / −x / +y / −y |
|---|---|---|---|
| Deployed (classifier v3, plate_t1) | 27/75 | 2/75 | 2 / 0 / 0 / 0 |
| Attempt (classifier v4, plate_t2) | 77/80 | 30/80 | 8 / 0 / 8 / 14 |

- Detection: classifier v4 adds 200 scripted runs with a placed plate or cup knocked 4–9 cm (seeds 4400–4599,
  `record_state_data.py --slide-frac`). Scripted placement knocked 7 cm: v4 notices 20/20 in every direction (v3: 18,
  19, 20, 8; `scripts/eval_slid_detection.py`); first look as clean as v3 (0/50 fresh tables wrongly "done", 50/50
  half-done tables exact).
- Repair: the plate fine-tuned 7.5k steps on its own statistics with 100 scripted demonstrations of putting a
  knocked plate back (`record_context_demos.py --displace plate`). The scripted controller failed 69 of 169 random
  knocks, mostly toward arm A (−x) — the direction the learned plate never recovers.
- Gates written before any result: plate alone ≥ 29/30 from both starts (30/30, 30/30); no regression on the 50
  tuning tables (41/50; v4 alone 40/50); ≥ 60 of 80 knocked plates put back — 30/80, missed. Neither classifier nor
  plate deployed. A knock toward arm A defeats the scripted controller that writes the demonstrations, and a knock
  moves more than the plate: a repair skill needs a controller that re-grasps from there first.

## Repeatability and timing

- MuJoCo replays identical actions bit-for-bit and the policies are deterministic, but a fresh OpenGL context can
  render a few pixels one intensity level differently; the closed loop amplifies it (seed 128: drawer 6.40 vs
  6.30 cm). Two agent runs of the identical system, seeds 100–149: 41 and 39 tables, 10 of 50 differ. Two
  fixed-sequence runs, seeds 100–199, differing only where a spoon retry succeeded (none did): both 70/100, 16 differ,
  8 each way. A two-table difference between runs is not a result; paired McNemar tests over 50 seeds are.
- **Failed spoon retried with the other controller** (gate written first: ≥ 2 tables rescued, seeds 100–199): the
  second attempt runs the whole 50-action chunk open-loop instead of ensembling (`skill_policies.RETRY_EXEC`,
  `eval_table_chain.py --retry-exec spoon=exec50`). 13 retried, 0 placed: a spoon that fails once fails again from
  the state it leaves, under either controller (same-controller retries earlier: 0/23). Off.
- **Windows power throttling.** A process started from a console without a visible window can be classed as
  background and throttled (EcoQoS): 2026-09-14 benchmarks 4–8× slower on PyTorch, ~1.3× on OpenVINO, CPU otherwise
  idle and at full clock. One process: 14-thread PyTorch matmul 227 GFLOP/s as launched, 646 GFLOP/s after opting out.
  The robot and the benchmark scripts opt out at start (`tenplaces.cores.no_power_throttling`). Success rates
  unaffected (the simulation waits for every action). README latency table: the clean 2026-09-13 run; a re-run with
  the opt-out (`out/benchmark/opt_out_0914/`) matches it on the P-cores within ~10%; the E-cores-only rows and the
  E-core planner 1.5–2× slower that day, unexplained.
