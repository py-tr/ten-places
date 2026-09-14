# Findings: how the system got here

The README shows the final system. This page is the engineering record behind it: every change that was tried,
what it measured, and what did not work. Tuning and comparisons use seeds 100–149; the reporting seeds 0–49 are run
only with selections frozen beforehand. Every number comes from the script named next to it.

## Design decisions that changed the architecture

- **One policy per skill, not one skill-conditioned policy.** A single ACT conditioned on a skill one-hot learned to
  ignore it (same scene with each of the five one-hots → action chunks differ by ≤ 0.1) and inferred the phase from
  the cameras, so a planner could not steer it; it also froze at the hand-off, where the demos pause briefly and the
  policy has no sense of time. One skill-conditioned ACT: 1/10 full tables, 2.3/5 steps (`scripts/eval_table_checkpoints.py`).
- **Context shift.** Skills trained only on full tables fail when a subset command skips a step: the cup fails when
  the plate has not been moved. Demonstrations from every start a verified plan can produce fixed it. On tuning seeds
  100–129, 30 tables per start: with the plate not moved (3 starts) 54/90 → 80/90, with it moved (4 starts)
  113/120 → 118/120 (`scripts/eval_skill_context.py --seeds 100 130`, `out/eval/context/cup_*_seeds100-129.md`).
  The deployed cup was first chosen on seeds 0–9 (15/30 → 29/30), inside the reporting set; this re-check on the
  tuning seeds confirms the choice.
- **Temporal ensembling, made affordable by OpenVINO.** The spoon hand-off froze at the pause before arm A lets go
  when the policy re-planned every 10 actions. A forward pass every control step, blending overlapping action chunks,
  carries the release through, no retraining. Tuning seeds 100–149, spoon: 4/50 re-planning every 10 actions, 43/50
  with ensembling — and 44/50 executing whole 50-action chunks open-loop, so for the spoon the fix is not re-planning
  mid-hand-off, not ensembling as such. Drawer, seeds 100–119: 15/20 every 10, 18/20 whole chunks, 20/20 ensembling
  (`scripts/eval_skill_variants.py`). (First measured on seeds 0–9: spoon 3/10 → 10/10.) This needs one network call per 40 ms step: OpenVINO INT8 weights (16 ms) fits,
  PyTorch on the same CPU (39–45 ms) does not. An end-to-end VLA is out of reach here: LeRobot's SmolVLA (450M
  parameters) takes 6.6 s per 50-action chunk on this CPU in PyTorch (`scripts/bench_smolvla.py`, timing only).

- **A VLA baseline, measured.** SmolVLA (450M parameters; expert trained, vision encoder frozen, bf16) fine-tuned
  5k steps on the same 100 full-table cup demonstrations the first ACT cup learned from (`scripts/train_smolvla.py`),
  scored from all seven starts on tuning seeds 100–129 with the same evaluation and chunking (10 actions per chunk):

  | Cup, 30 tables per start | Plate already moved (4 starts) | Plate not moved (3 starts) | All |
  |---|---|---|---|
  | SmolVLA, 5k steps | 44/120 | 19/90 | 63/210 |
  | ACT, same data (skills_v1, 20k steps) | 113/120 | 54/90 | 167/210 |
  | ACT, deployed (plus demonstrations from every start) | 118/120 | 80/90 | 198/210 |

  LeRobot's own recipe trains SmolVLA 20k steps; at 2.5k it placed 2 of 21, at 5k 63 of 210, so it was still
  learning — the comparison is at a quarter of that budget. On this CPU one SmolVLA chunk takes 6.6 s in PyTorch
  against 16 ms for ACT with OpenVINO. Both numbers point the same way: the language lives in the VLM planner, and
  small policies act.

## Making skills that work alone work in a chain

Each skill scored ~90–100% started by the scripted controller, yet the chain of learned skills set 0 of 10 tables.

| Change | Result | Script |
|---|---|---|
| Per-skill ACT, each step alone from a realistic start (20k steps each), 10 held-out seeds | drawer 10/10, cup 10/10, plate 4/10, spoon 0/10, fork 0/10 | `scripts/watch_skill_evals.py` |
| Plate: demos weighted to hard layouts + takeover demos (learned policy starts, scripted controller finishes), 10k-step fine-tune, seeds 100–119 | 14/20 → **20/20** | `scripts/eval_skill_checkpoints.py` |
| Spoon vs drawer opening (8 / 9 / 10 cm), before → after demos with varied openings | 4/10, 10/10, 0/10 → 10/10, 10/10, 2/10 | `scripts/eval_skill_checkpoints.py --drawer-open` |
| Fork cut off by its time budget (seen on video) → longer caps, seeds 100–109 | fork 0 → 3/10 | `scripts/eval_table_budgets.py` |
| Release the grippers and return home between skills (the state every demonstration starts from), seeds 100–119 | 4/20 → **10/20** full tables, 2.70 → 4.05 of 5 steps | `scripts/eval_table_home.py` |
| Drawer runs its whole budget instead of stopping at the camera's "done" (≥ 6 cm; the cutlery needs ~7.4), seeds 100–119 | spoon after the learned drawer 12/20 → 16/20; 10/20 → 15/20 full tables | `scripts/eval_table_home.py` |
| Classifier v3: drawer "done" only when open enough for the cutlery, trained with stalled pulls; learned-policy states, seeds 120–139 | false "drawer done" 61 → 3 of 363 checks, false "spoon done" 10 → 1 of 671; agent 14/20 → 16/20 (seeds 100–119) | `make state-data state-clf` |
| How the drawer step ends, with classifier v3, 50 seeds (100–149) | budget: PyTorch 30/50, OpenVINO 31/50; camera: 30/50, 29/50 — no difference | `TENPLACES_CAMERA_ENDS=drawer=1 scripts/eval_table_home.py` |

Reporting seeds 0–49, each run once with frozen selections (`scripts/final_report.py`, `scripts/eval_agent_table.py`):

| Configuration | PyTorch | OpenVINO INT8 weights | Full agent |
|---|---|---|---|
| Per-skill policies, no hand-over between skills | 2/50 | 5/50 | – |
| + release and home between skills | 23/50 | 22/50 | 24/50 (PyTorch) |
| + drawer runs its budget | 26/50 | 24/50 | 26/50 (PyTorch) |
| + classifier v3 | 24/50 | 23/50 | 26/50 (PyTorch) |
| + drawer fine-tuned on genuinely stalled pulls (final, below) | 39/50 | 41/50 | **43/50 (OpenVINO)** |

The last two changes were chosen on 20 tuning seeds and do not show up on the reporting seeds: paired against the
row above, full-table success is within noise (PyTorch +7 / −4 tables, McNemar p = 0.55). One step moved the wrong
way: the OpenVINO drawer, 48 → 42 → 39 of 50 (vs the release-and-home row: 0 gained, 9 lost), which does not
reproduce on the 50 tuning seeds (44 vs 44). Lesson: 20 tuning seeds are too few to choose on — later decisions use 50.

The final row was chosen that way (41/50 on the tuning seeds) and holds on the reporting seeds. Paired against the
classifier-v3 row: PyTorch +15 / −0 tables (McNemar p < 0.001), OpenVINO +20 / −2 (p < 0.001), full agent 26 → 43
(+21 / −4, PyTorch then, OpenVINO now). The agent's re-checks and retries add 3 tables and lose 1 over the fixed
sequence (p = 0.62). Its 7 remaining losses: spoon 3, plate 2, fork 2.

**Execution settings re-checked on the tuning seeds.** Spoon, plate, fork and cup had been given their execution
setting on seeds 20–29 — a tuning split chosen when only seeds 0–9 were reported, inside today's reporting set. Re-run on seeds 100–149 with the deployed checkpoints (`eval_skill_variants.py --deployed
--seeds 100 150`), with the rule fixed before the last results were in: keep the current setting unless another is
better by a paired McNemar test at p < 0.05.

| Skill | every 10 actions | whole 50-action chunks | ensembling | current | verdict |
|---|---|---|---|---|---|
| spoon | 4/50 | 44/50 | 43/50 | ensembling | keep (+3 / −2, p = 1.00) |
| plate | 14/50 | 40/50 | 45/50 | ensembling | keep (best) |
| fork | 42/50 | 49/50 | 49/50 | whole chunks | keep (tied best) |
| cup | 49/50 | 50/50 | 50/50 | every 10 | keep (+1 / −0, p = 1.00) |

Every setting holds, so the reported numbers stand. `--select` now refuses seeds below 100.

**Drawer demos from genuinely stalled pulls (the fix that worked).** 100 new drawer demos
(`scripts/record_chain_demos.py`): 60 start where a learned pull really stalled (release + home, then the scripted
controller re-grips and finishes), 40 are plain pulls from a closed drawer; fine-tuned from the deployed drawer.
- First attempt, with the demo set's own statistics: the drawer froze in one pose on every table (0/50). Arm B never
  moves in drawer demos, so several of its joint spreads were float32 rounding noise (~0) while the policy still
  issued tiny arm-B commands; LeRobot's MEAN_STD normalisation divides the resulting 1e-5–1e-4 rad deviation by ~1e-8.
  Reproduced offline: a 1e-4 rad nudge on one arm-B joint sends the predicted arm-A motion to the frozen pose.
- Same data, fine-tuned with the parent's statistics (`scripts/use_parent_stats.py`; `train_skills.py` now refuses a
  demo set with a near-zero spread), 7.5k steps, seeds 100–149: drawer 44 → 48/50, spoon 36 → 45, **full tables
  30 → 41/50** (Wilson 69–90%); with a drawer retry 43/50. The spoon's losses were stalled drawers all along.
- A spoon fine-tune on 190 new demos (after the learned drawer, plus drawer openings of 7.2–10.2 cm) made the spoon
  worse (20–22/50) and was not used.
- Confirmation from the other side: feeding the broken first attempt's idle joints their training mean
  (`LeRobotPolicy(mask_idle_std=1e-4)`, `eval_table_chain.py --mask-idle`) brought its drawer back from 0/50 to 18/20.
  Masking the deployed skills on top of the new drawer did not help (34/50 vs 41/50), so it stays off.

**Plate without the spoon step.** Demo seed 4 ("Just the plate and the cup.") lost the plate four times. The
deployed plate started after the drawer alone and after drawer + spoon (scripted prefix, seeds 100–129,
`eval_skill_context.py --deployed`): 30/30 each, so that start is not a weakness. Seed 4 is a hard table for the
plate — it fails there from both starts even with the scripted drawer, and in every report-#5 run — and, being a
reporting seed, it is not tuned on.

**Where the chain still loses.** Fork, the weakest step, is mostly downstream: of its 25 failures (PyTorch, seeds
0–49, release + home), 15 are on seeds where the spoon failed and 4 where the plate failed (the plate starts on the
fork's spot); 6 are the fork's own. The drawer pull sometimes stalls at 5–6.5 cm, mostly on high-friction tables
(opening vs friction: Spearman −0.58). Started from a drawer set 5 / 6 / 6.5 cm open, the drawer policy reaches
≥ 7.4 cm on 18 / 19 / 20 of 20 — but inside the chain a retry recovered 0 of 9 real stalls (`scripts/eval_table_chain.py
--retry`, seeds 100–149): a tray dragged there under high friction, with the arm still hung on the handle, is not the
same start as a tray set there.

## A second held-out set, and tables outside the training ranges

After the report, 50 fresh seeds (200–249) that no step of the work had touched, with the same frozen configuration,
each run once (`final_report.py` refuses the tuning range and allows 200+ for this; `scene_table.sample(stress)`):
- Full agent on OpenVINO: 43/50 (74–93%) — the reporting seeds' 43/50 again. Over both held-out sets: 86/100.
- Fixed five-step sequence on OpenVINO: 33/50 (52–78%), against 41/50 on seeds 0–49 — a different, harder-looking
  set, and part of the gap is run-to-run noise (below). Here the agent adds 10 tables and loses none over the fixed
  sequence (McNemar p = 0.002; on seeds 0–49: +3 / −1). The re-checks and re-queued steps earn their place on the
  harder set.
- Robustness outside the training ranges: friction, the three masses, light and colours widened ×1.5 about their
  centres (placements unchanged, so the tables pair seed by seed): 41/50 against 33/50 at the normal ranges, 14 tables
  better and 6 worse (p = 0.12) — no measurable loss. The classifier and the policies both see the widened scenes. Widened ×2.0: 33/50, 10
  tables better and 10 worse (p = 1.0).
- Object sizes, which training never varied (`TENPLACES_SHAPE`: plate radius, cup radius and height ×(1 ± shape),
  cutlery length ±min(shape, 10%) for the 16 cm tray; everything else identical to the paired table): ±10%
  25/50 (4 better, 12 worse, p = 0.08), ±20% 17/50 (2 better, 18 worse, p < 0.001), against 33/50 at the trained
  sizes. The cup carries most of it — at ±20% it misses on 24 tables where the trained sizes placed it. Of those
  tables, by the cup's size: within ±5% 9/11 placed, 5–10% off 9/12, more than 10% smaller 1/11, more than 10%
  larger 4/13. The plate fails when more than 10% smaller (0/5) and holds when larger (12/13); longer cutlery
  (+5–10%) costs the fork (5/9). The grader is not the cause: a cup or a plate set exactly on its target passes on
  all 50 tables at ±20%, as at the trained sizes. Each skill learned one size of each object. With only the cup's
  size varied ±20% (`TENPLACES_SHAPE_ONLY=cup`, the same cups as in the every-object row): 18/50 (2 better, 17 worse,
  p < 0.001), the cup placed 23/50 — nearly the whole loss of the every-object row (17/50).
- One attempt to teach the cup other sizes, gates written before any result (tuning seeds 100–149): 100 scripted
  demonstrations with the cup ×0.77–1.25 (40 at the trained size), the deployed cup fine-tuned 7.5k steps on its own
  statistics. It got worse at every size — cup alone at the trained size 37/50 against 48/50, at ±20% cup sizes 24/50
  against 29/50, the fixed sequence 22/50 full tables against 42/50 — so it is not shipped. Before it could
  demonstrate the larger cups at all, the scripted controller needed a higher approach (×1.2: 3/20 placed, then
  20/20). The deployed cup's misses at other sizes are all "never picked up", none a false "done" from the camera.
  Untested: the fine-tune used the knock attempt's recipe (augmentation, AMP, lr 2e-5), not the deployed cup's own.

Where the 14 lost tables of the 100 went (full agent, OpenVINO, `out/eval/agent_table/ov_w8_report5.json` and
`ov_w8_fresh200-249.json`; a table counts at its first step, in task order, that is still undone at the end):

| Step | Tables | Seeds | Final distance from target |
|---|---|---|---|
| spoon | 6 | 24, 36, 37, 214, 220, 223 | 22–25 cm: the spoon never leaves the tray, even after the re-queue |
| plate | 4 | 4, 12, 204, 226 | 7–10 cm |
| fork | 4 | 18, 19, 240 · 233 | 17–20 cm (not carried) · 2.8 cm (placed just outside the 2.5 cm tolerance) |

No table is lost at the drawer, and the cup fails only on a table already lost (seed 4). Picking cutlery out of the
tray is the largest single loss: 9 of the 14.

## Planner

- **Plan first, check later.** Asking for the plan and then for what no skill can do, both before moving, took
  18.7 s median to the first motion. Now the plan alone starts the arms (13.5 s on the E-cores, seed 3, "I want to
  eat soup.") and the text-only check answers 4.5 s later on the planner's worker thread.
- **First plan on every core.** The arms are still until the first plan arrives, so a second planner pipeline on
  every core (`VLMPlanner(idle_config={})`) makes it; the E-core pipeline keeps what runs while the arms move.
  10 demo commands on their own tables: median 14.3 → 7.1 s, same plan 10/10 (`scripts/bench_planner_placement.py`);
  in the full agent, with the control models loaded, seed 3's plan took 9.0 s instead of 13.5 s. Cost: a second
  copy of the model in memory (~3 GB).
- **A dangling "and" emptied the plan.** "Set the table and light a candle." makes the 4B model return no steps;
  the re-plan without the impossible part used "Set the table and .", which it also read as incomplete, so the
  robot did nothing. Found on demo seed 6's pre-registered command; the re-plan now drops the dangling join
  ("Set the table."): dev set 4/5 → 5/5, the other sets unchanged, and seed 6 sets the full table and names the
  candle (`scripts/eval_planner.py`, `tests/test_planner.py`).
- **Czech commands: plans yes, refusals no.** Six Czech commands, the same prompt: the plan was right 6/6, but
  the check for what no skill can do refused 4 of the 6 commands it should have accepted — 2/6 end to end. Not
  shipped; the demo stays in English.

## The first look: what is already done is not planned again

Before planning, the camera classifier reads the table once; steps it sees done with probability ≥ 0.95 are passed
to the planner as done (`run_command(look_first=0.95)`), said aloud and shown on the panel; the re-check guards them
like any done step. Gates on tuning seeds 100–149 (`scripts/eval_initial_state.py`): fresh tables — highest
probability for any step 0.003 (v3), none read as done; tables half-set by the scripted controller (ten prefixes) —
50/50 read exactly, lowest probability for a done step 0.990. The agent on the 50 fresh tuning tables with and
without it: the look skipped nothing on any table (41/50 without, 39/50 with — the 10 tables that differ, 4 one way
and 6 the other, are run-to-run noise, since nothing the robot did changed; McNemar p = 0.75). On by default in
`run_agent.py`; the reported evaluations run without it.

A half-set table can still be outside what the skills learned. Tuning seed 120 with the drawer and the plate already
done: the look saw both (p = 1.0), the planner proposed only spoon, fork and cup — and the spoon failed twice, because
every spoon demonstration had the plate still at its start; a plate already on its mat is a scene the spoon policy has
never seen (the verifier flags the order: "'spoon' after 'plate' is outside the order tested collision-free"). The
extra demonstration therefore half-sets the table in the trained order (drawer and spoon done).

## OpenVINO precision study

One hand-off checkpoint (40k steps), 20 held-out seeds, task success re-measured in closed loop for every precision
(`scripts/int8_study.py`, `out/int8_study/040000/int8_study.json`; latencies measured while training ran on the same
machine):

| Variant | Latency | Success (Wilson 95%) |
|---|---|---|
| OpenVINO FP32 | 33.7 ms | 13/20 (0.43–0.82) |
| INT8 weights only | 28.6 ms | 14/20 (0.48–0.86) |
| INT8 image encoder (weights + activations), transformer float | 20.2 ms | 12/20 (0.39–0.78) |
| INT8 everything (weights + activations) | 13.8 ms | 7/20 (0.18–0.57) |

Quantising the transformer's activations is what costs task success; INT8 weights keep it.

**The fourth rung, INT4 weights** (`nncf` INT4_ASYM, groups of 64; the three input projections whose size is not a
multiple of 64 stay INT8): 23 MB instead of 33 MB per policy. The full agent on tuning seeds 100–149 with the five
deployed policies in INT4: 0/50 full tables against 41/50 with INT8 weights — the drawer (50), spoon (44), plate (45)
and fork (48) hold, the cup fails on every table. On the same half-set tables the cup's first actions differ from
INT8's by up to 0.05 rad, a steady drift in arm B's shoulder pan and wrist rather than noise — enough to miss a small
cup and its 2.5 cm target (`eval_agent_table.py --backend ov-w4`). Nor is it faster here: in one benchmark run INT4 took 16.8–21.3 ms per
inference against 16.8–19.3 ms for INT8 weights (`out/benchmark/opt_out_0914/`). Task success, not model size, decides where the
ladder stops. (The later 75k hand-off
checkpoint scored 19/20 on both FP32 and INT8 weights, `scripts/eval_checkpoints.py`.) The deployed table policies
run INT8 weights (15.5–16 ms on an idle machine, README).

## Disturbance repair (one attempt, measured, not shipped)

The agent re-checks every finished step before the next one and at the end; a step that no longer holds goes back to
the front of the queue. Measured with the real agent: the plate pushed 7 cm right after it is placed, in each of four
directions, tuning seeds 120–139, OpenVINO (`scripts/eval_agent_table.py --push after-plate:plate:DX:DY`):

| System | Knock noticed | Plate put back | put back: +x / −x / +y / −y |
|---|---|---|---|
| Deployed (classifier v3, plate_t1) | 27/75 | 2/75 | 2 / 0 / 0 / 0 |
| Attempt (classifier v4, plate_t2) | 77/80 | 30/80 | 8 / 0 / 8 / 14 |

- Detection: classifier v4 adds 200 scripted runs in which a placed plate or cup is knocked 4–9 cm (seeds 4400–4599,
  `record_state_data.py --slide-frac`); v3 had only seen a plate off its target before it was placed. On a scripted
  placement knocked 7 cm, v4 notices 20/20 in every direction (v3: 18, 19, 20, 8; `scripts/eval_slid_detection.py`)
  and reads the first look as cleanly as v3 (0/50 fresh tables wrongly "done", 50/50 half-done tables exact).
- Repair: the deployed plate fine-tuned 7.5k steps on its own statistics with 100 demonstrations of the scripted
  controller putting a knocked plate back (`record_context_demos.py --displace plate`). The scripted controller
  itself failed 69 of the 169 random knocks tried, mostly toward arm A (−x) — the direction the learned plate never
  recovers.
- Gates written down before any result, one attempt: the plate alone ≥ 29/30 from both starts (30/30, 30/30); no
  regression on the 50 tuning tables (41/50, as deployed; v4 alone 40/50); ≥ 60 of 80 knocked plates put back —
  30/80 missed it. So neither the classifier nor the plate is deployed. More demonstrations alone would not close
  it: a knock toward arm A defeats even the scripted controller that writes them, and a knock moves more than the
  plate (the fork and cup targets beside it), so a repair skill needs a controller that re-grasps from there first.

## Repeatability

MuJoCo replays identical actions bit-for-bit, and the policies are deterministic (no sampling at inference), but a
fresh OpenGL context can render a few pixels one intensity level differently; the closed loop amplifies that
(seed 128: drawer 6.40 vs 6.30 cm). Seed-level comparisons between runs are therefore partly noise, and results are
reported over 50 seeds with Wilson intervals. How much: two agent runs of the identical system on seeds 100–149 set
41 and 39 tables and disagree on 10 of the 50 (the first-look gate above) — a difference of two tables between runs
is not a result; paired comparisons with McNemar tests are.

Timing is a separate matter. Started from a console with no visible window, a process can be classed as background
by Windows 11 and throttled (EcoQoS): on 2026-09-14 the benchmarks came out 4–8× slower on PyTorch and ~1.3× on
OpenVINO, with the CPU otherwise idle and at full clock. In one process, a 14-thread PyTorch matmul ran 227 GFLOP/s
as launched and 646 GFLOP/s after opting out. The robot and the benchmark scripts now opt out at start
(`tenplaces.cores.no_power_throttling`). Success rates are unaffected — the simulation waits for every action. The
README's latency table is the clean run of 2026-09-13; a re-run with the opt-out (`out/benchmark/opt_out_0914/`)
matches it on the P-cores within about 10%, while the E-cores-only rows and the E-core planner ran about 1.5–2×
slower that day, not explained.
