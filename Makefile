PY ?= python
DATA ?= data/handoff_v1
RUN ?= out/train/act_handoff_v1
STEPS ?= 20000
CKPT ?= $(RUN)/checkpoints/075000/pretrained_model

SKILL_RUNS ?= out/train/skills_v1 out/train/skills_v2 out/train/skills_ctx2
CMD ?= set the table
SEED ?= 0
.DEFAULT_GOAL := test

# The official SO-101 model (Apache-2.0), pinned to the commit every result here was produced with.
SO_ARM_COMMIT ?= eecbe3e0a9ebb23e25ad7b2759b03884c6660903
third-party:
	git clone https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100
	git -C third_party/SO-ARM100 checkout $(SO_ARM_COMMIT)

.PHONY: third-party scene spike spike-table demos table-demos train spike-ov bench skills skills-v2 eval-skills \
	state-data state-clf eval-planner agent grid eval-context ctx-demos skills-ctx voice recover

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

bench:
	$(PY) scripts/benchmark.py --checkpoint $(CKPT)

# One ACT per skill (20k steps each), then spoon/fork/plate fine-tuned +40k from those weights.
skills:
	$(PY) scripts/train_skills.py --out out/train/skills_v1 --steps 20000

skills-v2:
	$(PY) scripts/train_skills.py --out out/train/skills_v2 --skills spoon fork plate \
		--init-from out/train/skills_v1 --init-step 20000 --steps 40000

# Full table on held-out seeds: sequencer + camera classifier, PyTorch and OpenVINO rows.
eval-skills:
	$(PY) scripts/eval_skills.py --runs $(SKILL_RUNS) --name skills_mixed --seeds 0 10 --videos 10

# Context shift: each skill from every start a verified subset plan can give it, then demos from those starts.
eval-context:
	$(PY) scripts/eval_skill_context.py --skill cup --runs out/train/skills_v1
	$(PY) scripts/eval_skill_context.py --skill cup --runs out/train/skills_v1 out/train/skills_ctx --name cup_ctx

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

# Knock the plate off mid-run; the camera re-check puts it back. make recover SEED=3
# Knock the plate off its mat mid-run. A known failure with the learned system (README "Disturbances"): the camera
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

# Intel deliverable 4: the 10 held-out seeds from eval-skills tiled into one video with PASS/FAIL stamps.
grid:
	$(PY) scripts/make_grid_video.py --dir out/eval/skills_mixed --label torch_clf --out out/video/grid_10_seeds.mp4
