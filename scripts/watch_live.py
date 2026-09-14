"""Watch the robots live in MuJoCo's 3D viewer (drag to orbit, scroll to zoom, double-click to track a body).

    python scripts/watch_live.py --oracle --seed 3                       # scripted controller, full table
    python scripts/watch_live.py --oracle --plan drawer cup --seed 3     # scripted controller, any plan
    python scripts/watch_live.py --command "set the table" --seed 3      # the agent: VLM plan + learned policies

The agent mode prints every decision (plan, verifier corrections, camera checks) as it happens. --speed 0.5
plays at half speed; the window can be closed at any time.
"""
import argparse
import sys
import time
from pathlib import Path

import mujoco.viewer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.env_table import TableEpisode  # noqa: E402


class LiveView:
    """Opens the passive viewer on first use and keeps simulated time in step with wall time."""

    def __init__(self, speed: float = 1.0):
        self.speed, self.handle, self.t0 = speed, None, None

    def sync(self, m, d):
        if self.handle is None:
            self.handle = mujoco.viewer.launch_passive(m, d)
            self.t0 = time.perf_counter() - d.time / self.speed
        if not self.handle.is_running():
            raise KeyboardInterrupt("viewer closed")
        self.handle.sync()
        ahead = d.time / self.speed - (time.perf_counter() - self.t0)
        if ahead > 0:
            time.sleep(ahead)

    def close(self):
        if self.handle is not None:
            self.handle.close()


def run_oracle(seed: int, plan, view: LiveView):
    from tenplaces.grader_table import grade_table
    from tenplaces.oracle import table

    ep = None

    def on_step(m, d):  # every physics step; the viewer only needs ~50 Hz
        if ep is not None and ep.ctl.steps % 10 == 0:
            view.sync(m, d)

    ep = TableEpisode(seed, render=False, on_step=on_step)
    if plan:
        table.run_plan(ep.ctl, ep.params, plan)
    else:
        table.run(ep.ctl, ep.params)
    g = grade_table(ep.m, ep.d, ep.params)
    print("graded:", {k: v for k, v in g.items() if isinstance(v, bool)})
    ep.ctl.hold(2.0)


def run_agent(seed: int, command: str, runs, backend: str, view: LiveView):
    from tenplaces.agent import run_command
    from tenplaces.planner import VLMPlanner
    from tenplaces.skill_policies import SkillPolicies

    kw = dict(device="cuda") if backend == "torch" else dict(backend=backend, calib_cache="data/table_v1_skill_cache")
    policy = SkillPolicies(runs, n_action_steps=10, **kw)
    planner = VLMPlanner()
    _, grade = run_command(policy, planner, command, seed, on_frame=lambda ep, obs: view.sync(ep.m, ep.d))
    print("graded:", {k: v for k, v in grade.items() if isinstance(v, bool)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--oracle", action="store_true", help="scripted controller instead of the learned agent")
    ap.add_argument("--plan", nargs="*", default=None, help="oracle mode: skills to run (default: full table)")
    ap.add_argument("--command", default="set the table")
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx",
                                                  "out/train/skills_ctx2"])
    ap.add_argument("--backend", default="ov-w8", choices=["torch", "ov-fp32", "ov-w8"],
                    help="ov-w8 (default) is what the robot runs and needs no NVIDIA GPU; torch runs on CUDA")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    import torch

    torch.set_num_threads(1)  # as scripts/run_agent.py: pre/post-processing only; OpenVINO owns the control threads
    from tenplaces.cores import no_power_throttling

    no_power_throttling()
    runs = [r for r in args.runs if Path(r).is_dir()]
    view = LiveView(args.speed)
    try:
        if args.oracle:
            run_oracle(args.seed, args.plan, view)
        else:
            run_agent(args.seed, args.command, runs, args.backend, view)
        input("done — press Enter to close the viewer")
    except (KeyboardInterrupt, EOFError):  # window closed, or no console to press Enter in
        pass
    finally:
        view.close()


if __name__ == "__main__":
    main()
