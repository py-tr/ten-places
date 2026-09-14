"""Control-step latency while the VLM planner thinks, and which cores do the work (Intel deliverable 3).

One control step at 25 Hz is one ACT policy inference (temporal ensembling: the policy runs every step) plus
one camera-classifier inference, with a 40 ms budget. When the person speaks, the planner (Qwen3-VL-4B INT4)
generates for seconds on the same CPU. Per scenario this measures the control step p50/p95/p99, the share of
steps over budget, the planner's latency per call, and the mean utilisation of P-core vs E-core logical CPUs:
  a  control alone, default scheduling          a' control alone on P-cores (tenplaces.cores)
  b  control + planner generating nonstop, both default scheduling
  c  control on P-cores + planner on E-cores    d  as c, but control with hyper-threading on
  e  as c, with OpenVINO CPU pinning on (off by default in tenplaces.cores)
Control steps are paced at 25 Hz as in the agent; observations are one seeded TableEpisode frame.

    python scripts/bench_concurrency.py                 # real models (~3 GB planner), on an idle CPU
    python scripts/bench_concurrency.py --quick         # shorter windows, no scenario d
    python scripts/bench_concurrency.py --dry-run       # tiny synthetic OpenVINO models: checks the logic only
"""
import argparse
import gc
import json
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np
import openvino as ov
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces import cores  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402

BUDGET_MS = 40.0


class SyntheticControl:
    """Stand-in policy + classifier: small conv nets on the real image sizes (not our models)."""

    def __init__(self):
        import openvino.opset13 as ops

        def net(shape, ch, layers):
            x = ops.parameter(list(shape), np.float32)
            y = x
            for i in range(layers):
                w = np.random.default_rng(i).standard_normal((ch, y.get_output_partial_shape(0)[1].get_length(), 3, 3))
                y = ops.relu(ops.convolution(y, ops.constant(w.astype(np.float32) * 0.1), [2 if i < 3 else 1] * 2,
                                             [1, 1], [1, 1], [1, 1]))
            return ov.Model([ops.reduce_mean(y, [2, 3], keep_dims=False)], [x])

        self.models = {"policy": net((1, 3, 144, 192), 64, 6), "classifier": net((1, 3, 144, 192), 32, 4)}
        self.feed = [np.random.default_rng(0).random((1, 3, 144, 192), np.float32)]
        self.core = ov.Core()

    def configure(self, cfg):
        self.compiled = {k: self.core.compile_model(m, "CPU", {"PERFORMANCE_HINT": "LATENCY", **cfg})
                         for k, m in self.models.items()}
        return cores.effective(self.compiled["policy"])

    def step(self):
        t0 = time.perf_counter()
        self.compiled["policy"](self.feed)
        t1 = time.perf_counter()
        self.compiled["classifier"](self.feed)
        return 1000 * (t1 - t0), 1000 * (time.perf_counter() - t1)


class SyntheticPlanner:
    """Stand-in planner: one call = 40 inferences of a stack of 2048x2048 matmuls (memory-bound, like decoding)."""

    def __init__(self, cfg):
        import openvino.opset13 as ops

        x = ops.parameter([1, 2048], np.float32)
        y = x
        for i in range(12):
            y = ops.tanh(ops.matmul(y, ops.constant(np.full((2048, 2048), 1e-4 * (i + 1), np.float32)), False, False))
        self.compiled = ov.Core().compile_model(ov.Model([y], [x]), "CPU", cfg)
        self.feed = [np.ones((1, 2048), np.float32)]

    def call(self):
        for _ in range(40):
            self.compiled(self.feed)

    def close(self):
        del self.compiled


class RealControl:
    """The deployed control path: LeRobotPolicy (ov-<precision>, temporal ensembling) + the camera classifier."""

    def __init__(self, args, obs):
        from tenplaces.lerobot_policy import LeRobotPolicy
        from tenplaces.ov_backend import OVACT

        self.OVACT, self.args, self.obs = OVACT, args, obs
        self.policy = LeRobotPolicy(args.checkpoint, backend=f"ov-{args.precision}", temporal_coeff=args.temporal_coeff)
        self.ir = ov.Core().read_model(Path(args.checkpoint) / "openvino" / f"act_{args.precision}.xml")

    def configure(self, cfg):
        from tenplaces.state_classifier import OVStateClassifier

        compiled = ov.Core().compile_model(self.ir, "CPU", {"PERFORMANCE_HINT": "LATENCY", **cfg})
        self.policy.policy.model = self.OVACT(compiled)
        self.policy.reset()
        self.clf = OVStateClassifier(self.args.classifier, ov_config=cfg)
        return cores.effective(compiled)

    def step(self):
        t0 = time.perf_counter()
        self.policy.select_action(self.obs)
        t1 = time.perf_counter()
        self.clf.probs(self.obs["images"]["top"])
        return 1000 * (t1 - t0), 1000 * (time.perf_counter() - t1)


class RealPlanner:
    def __init__(self, args, cfg, image):
        from tenplaces.planner import VLMPlanner

        self.planner, self.image = VLMPlanner(args.planner, ov_config=cfg or None), image

    def call(self):  # a mid-run change, as the agent sends it
        self.planner.amend("Set the table.", "Please skip the fork.", self.image, ["drawer"], "spoon",
                           ["plate", "fork", "cup"])

    def close(self):
        del self.planner


def observation(seed):
    import mujoco

    from tenplaces.agent import PLANNER_HW, SKILL_INDEX, SKILL_TEXT, planner_image
    from tenplaces.env_table import SKILLS, TableEpisode

    ep = TableEpisode(seed, render=True)
    obs = ep.observation()
    onehot = np.zeros(len(SKILLS), np.float32)
    onehot[SKILL_INDEX["spoon"]] = 1.0
    obs.update(task=SKILL_TEXT["spoon"], env_state=onehot, skill="spoon")
    rend = mujoco.Renderer(ep.m, *PLANNER_HW)
    image = planner_image(ep, rend)
    rend.close()
    ep.close()
    return obs, image


def pct(a, q):
    return round(float(np.percentile(a, q)), 2) if len(a) else None


def run_scenario(control, ctl_cfg, planner, seconds, topo):
    effective = control.configure(ctl_cfg)
    for _ in range(25):  # warm-up
        control.step()
    stop, calls, errors = threading.Event(), [], []

    def think():
        try:
            while not stop.is_set():
                t = time.perf_counter()
                planner.call()
                calls.append(time.perf_counter() - t)
        except Exception as e:  # re-raised on the main thread
            errors.append(e)

    worker = threading.Thread(target=think, daemon=True) if planner else None
    if worker:
        worker.start()
        time.sleep(1.0)  # let generation get going before measuring
    psutil.cpu_percent(percpu=True)
    pol, clf, tick, t_end = [], [], time.perf_counter(), time.perf_counter() + seconds
    while time.perf_counter() < t_end:
        p, c = control.step()
        pol.append(p)
        clf.append(c)
        tick += 1.0 / 25
        now = time.perf_counter()
        tick = max(tick, now)  # behind schedule: start the next step now, no catch-up burst
        time.sleep(tick - now)
    util = psutil.cpu_percent(percpu=True)
    stop.set()
    if worker:
        worker.join()
    if errors:
        raise errors[0]
    step = np.add(pol, clf)
    mean = lambda idx: round(float(np.mean([util[i] for i in idx if i < len(util)])), 1) if idx else None  # noqa: E731
    return {"control_effective": effective, "steps": len(step), "step_ms_p50": pct(step, 50), "step_ms_p95": pct(step, 95),
            "step_ms_p99": pct(step, 99), "step_ms_max": round(float(step.max()), 2),
            "over_budget_pct": round(100 * float(np.mean(step > BUDGET_MS)), 2),
            "policy_ms_p50": pct(pol, 50), "classifier_ms_p50": pct(clf, 50),
            "planner_calls": len(calls), "planner_s_median": round(float(np.median(calls)), 2) if calls else None,
            "planner_s_max": round(float(max(calls)), 2) if calls else None,
            "util_pcore_pct": mean(topo.p_logical), "util_ecore_pct": mean(topo.e_logical)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="out/train/skills_v2/spoon/checkpoints/040000/pretrained_model")
    ap.add_argument("--precision", default="w8", help="IR in <checkpoint>/openvino/act_<precision>.xml")
    ap.add_argument("--temporal-coeff", type=float, default=0.01)
    ap.add_argument("--classifier", default="models/state_classifier_v3/state_classifier.xml")
    ap.add_argument("--planner", default="models/Qwen3-VL-4B-Instruct-int4-ov")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seconds", type=float, default=60.0, help="measurement window per scenario")
    ap.add_argument("--quick", action="store_true", help="20 s windows, no hyper-threading scenario")
    ap.add_argument("--dry-run", action="store_true", help="tiny synthetic models instead of ours")
    args = ap.parse_args()
    import torch

    torch.set_num_threads(1)  # as scripts/run_agent.py: pre/post-processing only; OpenVINO owns the control threads
    from tenplaces.cores import no_power_throttling

    no_power_throttling()
    seconds = min(args.seconds, 20.0) if args.quick else args.seconds
    topo = cores.topology()
    # The defaults are the pinned placement (scenario e); a', c, d are measured against its unpinned form.
    pinned, plan_pinned = cores.control_config(topo), cores.planner_config(topo)
    ctl = {**pinned, "ENABLE_CPU_PINNING": False}
    plan = {**plan_pinned, "ENABLE_CPU_PINNING": False} if topo.hybrid else plan_pinned
    ctl_ht = {**ctl, "ENABLE_HYPER_THREADING": True}
    if topo.hybrid:
        ctl_ht["INFERENCE_NUM_THREADS"] = len(topo.p_logical)
    scenarios = [("a", "control alone, default", {}, None), ("a'", "control alone, P-cores", ctl, None),
                 ("b", "control + planner, both default", {}, {}), ("c", "control P-cores + planner E-cores", ctl, plan)]
    if not args.quick:
        scenarios.append(("d", "as c, control hyper-threading on", ctl_ht, plan))
    # tenplaces.cores leaves threads unpinned (Windows' own default); this row keeps checking that choice.
    scenarios.append(("e", "as c, CPU pinning on", {**ctl, "ENABLE_CPU_PINNING": True},
                      {**plan, "ENABLE_CPU_PINNING": True} if topo.hybrid else plan))

    busy = psutil.cpu_percent(interval=1.0)  # other load before we start: these are wall-clock numbers
    if args.dry_run:
        control, make_planner = SyntheticControl(), lambda cfg: SyntheticPlanner(cfg)
    else:
        obs, image = observation(args.seed)
        control, make_planner = RealControl(args, obs), lambda cfg: RealPlanner(args, cfg, image)

    rows, planner, planner_cfg = [], None, None
    for key, label, c_cfg, p_cfg in scenarios:
        if planner is not None and p_cfg != planner_cfg:  # one planner in memory at a time
            planner.close()
            planner = None
            gc.collect()
        if p_cfg is not None and planner is None:
            print(f"[{key}] loading planner with {p_cfg or 'default properties'}", flush=True)
            planner, planner_cfg = make_planner(p_cfg), p_cfg
        r = run_scenario(control, c_cfg, planner if p_cfg is not None else None, seconds, topo)
        rows.append({"scenario": key, "label": label, "control_config": c_cfg, "planner_config": p_cfg, **r})
        print(json.dumps(rows[-1]), flush=True)
    if planner is not None:
        planner.close()

    import openvino_genai as og

    env = {"cpu": ov.Core().get_property("CPU", "FULL_DEVICE_NAME"), "topology": topo.describe(),
           "openvino": ov.__version__, "openvino_genai": og.__version__, "python": platform.python_version(),
           "os": platform.platform(), "cpu_busy_before_pct": busy, "seconds_per_scenario": seconds,
           "models": "SYNTHETIC stand-ins (--dry-run)" if args.dry_run else
           {"policy": f"{args.checkpoint} ov-{args.precision}, temporal ensembling {args.temporal_coeff}",
            "classifier": args.classifier, "planner": args.planner}}
    name = "concurrency_dryrun" if args.dry_run else "concurrency"
    out = OUT / "benchmark"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{name}.json").write_text(json.dumps({"env": env, "budget_ms": BUDGET_MS, "rows": rows}, indent=1))
    lines = [f"# Control latency while the planner thinks{' (DRY RUN: synthetic models)' if args.dry_run else ''}", "",
             f"CPU: {env['cpu']}, {env['topology']} · OpenVINO {env['openvino']} · control step = policy + classifier, "
             f"paced at 25 Hz, budget {BUDGET_MS:.0f} ms · {seconds:.0f} s per scenario", ""]
    if busy > 10:
        lines += [f"Warning: the CPU was {busy:.0f}% busy before the run; rerun on an idle machine.", ""]
    lines += ["| | scenario | control step p50 / p95 / p99 (ms) | steps over 40 ms | policy / classifier p50 (ms) "
              "| planner s per call (median, n) | P-core / E-core utilisation |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        plan_s = f"{r['planner_s_median']} ({r['planner_calls']})" if r["planner_calls"] else "–"
        lines.append(f"| {r['scenario']} | {r['label']} | {r['step_ms_p50']} / {r['step_ms_p95']} / {r['step_ms_p99']} "
                     f"| {r['over_budget_pct']}% | {r['policy_ms_p50']} / {r['classifier_ms_p50']} | {plan_s} "
                     f"| {r['util_pcore_pct']}% / {r['util_ecore_pct']}% |")
    lines += ["", "Placement (tenplaces.cores): control " + json.dumps(ctl) + "; planner " + json.dumps(plan)]
    (out / f"{name}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
