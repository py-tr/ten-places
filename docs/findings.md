# Findings: how the system got here

The README shows the final system. This page is the engineering record behind it: every change that was tried,
what it measured, and what did not work. Tuning and comparisons use seeds 100–149; the reporting seeds 0–49 are run
only with selections frozen beforehand. Every number comes from the script named next to it.

## Design decisions that changed the architecture

- **One policy per skill, not one skill-conditioned policy.** A single ACT conditioned on a skill one-hot learned to
  ignore it (same scene with each of the five one-hots → action chunks differ by ≤ 0.1) and inferred the phase from
  the cameras, so a planner could not steer it; it also froze at the hand-off, where the demos pause briefly and the
  policy has no sense of time. One skill-conditioned ACT: 1/10 full tables, 2.3/5 steps (`scripts/eval_table_checkpoints.py`).
- **Context shift.** Skills trained only on full tables fail when a subset command skips a step: the cup scored 10/10
  when the plate had been moved and 5/10 when not. Demonstrations from every start a verified plan can produce fixed
  it: cup 15/30 → 29/30 with the plate not moved, 40/40 when it had been (`scripts/eval_skill_context.py`).
- **Temporal ensembling, made affordable by OpenVINO.** The spoon hand-off froze at the pause before arm A lets go
  (3/10 re-planning every 10 actions, 4/10 with 1.7× more time). A forward pass every control step, blending
  overlapping action chunks, carries the release through: 10/10, no retraining. Drawer: 15/20 → 20/20
  (`scripts/eval_skill_variants.py`). This needs one network call per 40 ms step: OpenVINO INT8 weights (17 ms) fits,
  PyTorch on the same CPU (42–47 ms) does not.

## Making skills that work alone work in a chain

Each skill scored ~90–100% started by the scripted controller, yet the chain of learned skills set 0 of 10 tables.

| Change | Result | Script |
|---|---|---|
| Per-skill ACT, each step alone from a realistic start (20k steps each), 10 held-out seeds | drawer 10/10, cup 10/10, plate 4/10, spoon 0/10, fork 0/10 | `scripts/watch_skill_evals.py` |
| Plate: demos weighted to hard layouts + takeover demos (learned policy starts, scripted controller finishes), 10k-step fine-tune, seeds 100–119 | 14/20 → **20/20** | `scripts/eval_skill_checkpoints.py` |
| Spoon vs drawer opening (8 / 9 / 10 cm), before → after demos with varied openings | 5/10, 10/10, 0/10 → 10/10, 10/10, 2/10 | `scripts/eval_skill_checkpoints.py --drawer-open` |
| Fork cut off by its time budget (seen on video) → longer caps, seeds 100–109 | fork 0 → 3/10 | `scripts/eval_table_budgets.py` |
| Release the grippers and return home between skills (the state every demonstration starts from), seeds 100–119 | 4/20 → **10/20** full tables, 2.70 → 4.05 of 5 steps | `scripts/eval_table_home.py` |
| Drawer runs its whole budget instead of stopping at the camera's "done" (≥ 6 cm; the cutlery needs ~7.4), seeds 100–119 | spoon after the learned drawer 12/20 → 16/20; 10/20 → 15/20 full tables | `scripts/eval_table_home.py` |
| Classifier v3: drawer "done" only when open enough for the cutlery, trained with stalled pulls; learned-policy states, seeds 120–139 | false "drawer done" 61 → 3 of 363 checks, false "spoon done" 10 → 1 of 671; agent 14/20 → 16/20 (seeds 100–119) | `make state-data state-clf` |
| How the drawer step ends, with classifier v3, 50 seeds (100–149) | budget: PyTorch 30/50, OpenVINO 31/50; camera: 30/50, 29/50 — no difference | `TENPLACES_CAMERA_ENDS=drawer=1 scripts/eval_table_home.py` |

Reporting seeds 0–49, each run once with frozen selections (`scripts/final_report.py`, `scripts/eval_agent_table.py`):

| Configuration | PyTorch | OpenVINO INT8 weights | Full agent |
|---|---|---|---|
| Per-skill policies, no hand-over between skills | 2/50 | 5/50 | – |
| + release and home between skills | 23/50 | 22/50 | 24/50 |
| + drawer runs its budget | 26/50 | 24/50 | 26/50 |
| + classifier v3 (final) | 24/50 | 23/50 | 26/50 |

The last two changes were chosen on 20 tuning seeds and do not show up on the reporting seeds: paired against the
row above, full-table success is within noise (PyTorch +7 / −4 tables, McNemar p = 0.55). One step moved the wrong
way: the OpenVINO drawer, 48 → 42 → 39 of 50 (vs the release-and-home row: 0 gained, 9 lost), which does not
reproduce on the 50 tuning seeds (44 vs 44). Lesson: 20 tuning seeds are too few to choose on — later decisions use 50.

**Where the chain still loses.** Fork, the weakest step, is mostly downstream: of its 25 failures (PyTorch, seeds
0–49, release + home), 15 are on seeds where the spoon failed and 4 where the plate failed (the plate starts on the
fork's spot); 6 are the fork's own. The drawer pull sometimes stalls at 5–6.5 cm, mostly on high-friction tables
(opening vs friction: Spearman −0.58). Started from a drawer set 5 / 6 / 6.5 cm open, the drawer policy reaches
≥ 7.4 cm on 18 / 19 / 20 of 20 — but inside the chain a retry recovered 0 of 9 real stalls (`scripts/eval_table_chain.py
--retry`, seeds 100–149): a tray dragged there under high friction, with the arm still hung on the handle, is not the
same start as a tray set there.

## OpenVINO precision study

Hand-off checkpoint, 20 held-out seeds, task success re-measured in closed loop for every precision
(`scripts/int8_study.py`; latencies measured while training ran on the same machine):

| Variant | Latency | Success |
|---|---|---|
| OpenVINO FP32 | 34.6 ms | 19/20 |
| INT8 weights only | 29.5 ms | 19/20 |
| INT8 image encoder, transformer float | 29.1 ms | 17/20 |
| INT8 everything (weights + activations) | 13.8 ms | 7/20 |

Quantising the transformer's activations is what costs task success; INT8 weights keep it. The deployed table
policies run INT8 weights (16–17 ms on an idle machine, README).

## Disturbance repair (measured, not working yet)

The agent re-checks every finished step before the next one and at the end; a step that no longer holds goes back to
the front of the queue. With the learned system this does not yet recover a knocked table:
- Plate slid 7 cm off its mat after a scripted placement (seeds 120–139): the camera notices only a slide back toward
  the plate's start (20/20); toward the spoon, and both along x, 0/20. The classifier learned "the plate has left its
  start", never having seen a plate knocked off the mat. The slide toward the spoon also knocks the spoon off (19/20).
- Learned plate policy re-placing a displaced plate: 0/44 (`scripts/eval_agent_table.py --push "after-plate:plate:0:-0.07"`).

The fix is data: classifier runs with objects slid after placement, and plate demonstrations from displaced starts.

## Repeatability

MuJoCo replays identical actions bit-for-bit, and the policies are deterministic (no sampling at inference), but a
fresh OpenGL context can render a few pixels one intensity level differently; the closed loop amplifies that
(seed 128: drawer 6.40 vs 6.30 cm). Seed-level comparisons between runs are therefore partly noise, and results are
reported over 50 seeds with Wilson intervals.
