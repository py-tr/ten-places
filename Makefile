PY ?= python
DATA ?= data/handoff_v1
RUN ?= out/train/act_handoff_v1
STEPS ?= 20000
CKPT ?= $(RUN)/checkpoints/075000/pretrained_model

SKILL_RUNS ?= out/train/skills_v1 out/train/skills_v2 out/train/skills_ctx
CMD ?= set the table
SEED ?= 0
.DEFAULT_GOAL := test

# The official SO-101 model (Apache-2.0), pinned to the commit every result here was produced with.
SO_ARM_COMMIT ?= eecbe3e0a9ebb23e25ad7b2759b03884c6660903
third-party:
	git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100
	git -C third_party/SO-ARM100 checkout $(SO_ARM_COMMIT)

.PHONY: third-party models scene spike spike-table demos table-demos train spike-ov bench bench-handoff skills \
	skills-v2 report state-data state-clf eval-planner agent grid eval-context ctx-demos skills-ctx voice recover

# Trained weights for a clean clone: the skills and the classifier from the project's model repo (made with
# scripts/upload_models.py), the planner from Intel's OpenVINO repo. make models HF_SKILLS_REPO=<user>/<repo>
HF_SKILLS_REPO ?=
models:
	$(PY) scripts/download_models.py --skills-repo "$(HF_SKILLS_REPO)"

test:
	$(PY) -m pytest -q

scene:
	$(PY) scripts/render_scene.py --seeds 0 1 2

spike:
	$(PY) scripts/spike_handoff.py --seeds 50 --video 2

spike-table:
	$(PY) scripts/spike_table.py --seeds 60 --start 1000 --video 2

demos:
	$(PY) scripts/record_demos.py --episodes 200 --start 2000 --root $(DATA) --overwrite

table-demos:
	$(PY) scripts/record_table_demos.py --episodes 100 --start 3000 --root data/table_v1 --overwrite
	$(PY) scripts/add_skill_feature.py --root data/table_v1 --out data/table_v1_skill
	$(PY) scripts/build_cache.py --root data/table_v1_skill --cache data/table_v1_skill_cache --repo-id local/tenplaces_table --verify

# ACT: 50-step action chunks (2 s at 25 Hz), batch 32; checkpoints every 5k steps.
train:
	TENPLACES_CACHE=$(DATA)_cache $(PY) scripts/train.py --dataset.repo_id=local/tenplaces_handoff --dataset.root=$(DATA) \
		--policy.type=act --policy.device=cuda --policy.push_to_hub=false \
		--policy.chunk_size=50 --policy.n_action_steps=50 \
		--batch_size=32 --steps=$(STEPS) --save_freq=5000 --log_freq=200 --num_workers=8 \
		--output_dir=$(RUN)

spike-ov:
	$(PY) scripts/spike_act_openvino.py --root $(DATA) --checkpoint $(CKPT)

# Intel deliverable 3: the five deployed skill policies (out/eval/selected_checkpoints.json, plus the cup), one report
# each in out/benchmark/<skill>_<step>.md, with the machine's OpenVINO devices listed at the top.
DEPLOYED ?= out/train/chain_t2_drawer/drawer/checkpoints/007500/pretrained_model \
    out/train/spoon_cont2/spoon/checkpoints/007500/pretrained_model \
    out/train/plate_t4/plate/checkpoints/007500/pretrained_model \
    out/train/fork_cont/fork/checkpoints/007500/pretrained_model \
    out/train/cup_sizes2/cup/checkpoints/007500/pretrained_model
bench:
	for ck in $(DEPLOYED); do $(PY) scripts/benchmark.py --checkpoint $$ck || exit 1; done

# The first single hand-off policy (before the table task), kept for the precision study in docs/findings.md.
bench-handoff:
	$(PY) scripts/benchmark.py --checkpoint $(CKPT)

# One ACT per skill (20k steps each), then spoon/fork/plate fine-tuned +40k from those weights.
skills:
	$(PY) scripts/train_skills.py --out out/train/skills_v1 --steps 20000

skills-v2:
	$(PY) scripts/train_skills.py --out out/train/skills_v2 --skills spoon fork plate \
		--init-from out/train/skills_v1 --init-step 20000 --steps 40000

# The reported numbers: the 50 held-out tables (seeds 0-49) with the frozen selections, as report 5 ran them — the
# fixed sequence on PyTorch (needs CUDA) and OpenVINO INT8 weights, then the full agent on OpenVINO.
# Tuning seeds instead: make report REPORT_SEEDS="100 150".
REPORT_SEEDS ?= 0 50
report:
	$(PY) scripts/final_report.py --seeds $(REPORT_SEEDS) --rows torch ov_w8 --videos 10 --workers 4 --home-frames 20 --out out/eval/final
	$(PY) scripts/eval_agent_table.py --seeds $(REPORT_SEEDS) --report --backend ov-w8 --workers 4 --name ov_w8_report

# The second held-out set (fresh seeds 200-249) and the robustness rows on it: the full agent, the fixed sequence at
# the training ranges, with friction/masses/light/colours widened x1.5, and with object sizes +-10% and +-20%.
report-fresh:
	$(PY) scripts/eval_agent_table.py --seeds 200 250 --backend ov-w8 --workers 4 --name ov_w8_fresh200-249
	env TENPLACES_STRESS=1.0 $(PY) scripts/final_report.py --seeds 200 250 --rows ov_w8 --videos 0 --workers 4 --out out/eval/stress_1.0
	env TENPLACES_STRESS=1.5 $(PY) scripts/final_report.py --seeds 200 250 --rows ov_w8 --videos 0 --workers 4 --out out/eval/stress_1.5
	env TENPLACES_SHAPE=0.1 $(PY) scripts/final_report.py --seeds 200 250 --rows ov_w8 --videos 0 --workers 4 --out out/eval/shape_0.1
	env TENPLACES_SHAPE=0.2 $(PY) scripts/final_report.py --seeds 200 250 --rows ov_w8 --videos 0 --workers 4 --out out/eval/shape_0.2
	env TENPLACES_SHAPE=0.2 TENPLACES_SHAPE_ONLY=cup $(PY) scripts/final_report.py --seeds 200 250 --rows ov_w8 --videos 0 \
		--workers 4 --out out/eval/shape_cup_0.2

# Context shift: each skill from every start a verified subset plan can give it, then demos from those starts.
eval-context:
	$(PY) scripts/eval_skill_context.py --skill cup --runs out/train/skills_v1 --seeds 100 130 --name cup_v1_seeds100-129
	$(PY) scripts/eval_skill_context.py --skill cup --runs out/train/skills_v1 out/train/skills_ctx --seeds 100 130 \
		--name cup_ctx_seeds100-129

ctx-demos:
	$(PY) scripts/record_context_demos.py --skills cup --episodes 80 --root data/table_ctx_cup
	$(PY) scripts/record_context_demos.py --skills plate fork --episodes 60 60 --start 6000 --root data/table_ctx_pf

skills-ctx:
	$(PY) scripts/train_skills.py --data data/table_ctx_cup --cache "" --out out/train/skills_ctx --skills cup \
		--init-from out/train/skills_v1 --init-step 20000 --steps 15000 --drop-optimizer
	$(PY) scripts/train_skills.py --data data/table_ctx_pf --cache "" --out out/train/skills_ctx --skills plate fork \
		--init-from out/train/skills_v2 --init-step 40000 --steps 15000 --drop-optimizer

# Camera task-state classifier: simulator-labelled random subset runs -> ResNet18 -> OpenVINO IR.
# v3 (default): the drawer counts as done only once open far enough for the cutlery (7.4 cm), and a quarter of the
# runs pull it short, so a stalled pull reads "not done" and the agent retries it. (v2: --runs 300 --out data/state_v1,
# drawer label = the grader's 6 cm, trained into models/state_classifier_v2.)
state-data:
	$(PY) scripts/record_state_data.py --runs 400 --start 4000 --out data/state_v3 --short-frac 0.25 --drawer-enough 0.074

state-clf:
	$(PY) scripts/train_state_classifier.py --labelled data/state_v3 --out models/state_classifier_v3 --epochs 3

eval-planner:
	$(PY) scripts/eval_planner.py
	$(PY) scripts/eval_amend.py --set fresh
	$(PY) scripts/eval_amend.py --set heldout
	$(PY) scripts/eval_amend.py --set dev

# make recover SEED=3 — knock the plate off its mat mid-run. A known failure with the learned system (README "Disturbances"): the camera
# misses it unless it slides back toward its start, and the plate policy never re-placed a displaced plate (0/44).
recover:
	$(PY) scripts/run_agent.py --command "set the table" --seed $(SEED) --push "after-plate:plate:0:-0.07" \
		--video out/video/recover_s$(SEED).mp4

# Speak the command, keep talking while it works (needs SPEECHMATICS_API_KEY). make voice SEED=3
voice:
	$(PY) scripts/run_agent.py --mic --listen --seed $(SEED) --video out/video/voice_s$(SEED).mp4

# make agent CMD="set the table, but skip the cup" SEED=3
agent:
	$(PY) scripts/run_agent.py --command "$(CMD)" --seed $(SEED) --runs $(SKILL_RUNS) --video out/video/agent_s$(SEED).mp4

# Live 3D viewer. make watch SEED=3 (scripted) | make watch-agent CMD="just the plate and the cup" SEED=3
watch:
	$(PY) scripts/watch_live.py --oracle --seed $(SEED)

watch-agent:
	$(PY) scripts/watch_live.py --command "$(CMD)" --seed $(SEED)

# Intel deliverable 4: the 10 pre-registered demonstration runs (configs/demo_seeds.json, out/video/demo/seed<N>.*)
# scored against their commands and tiled into one video with PASS/FAIL stamps.
grid:
	$(PY) scripts/score_demo.py
	$(PY) scripts/make_grid_video.py --dir out/video/demo --label demo --out out/video/grid_demo.mp4 \
		--caption "requests done exactly as asked"
