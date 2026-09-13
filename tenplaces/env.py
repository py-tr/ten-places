"""Episode environment shared by demo recording and policy evaluation.

Control runs at FPS: every control step holds a 12-D joint-position target (6 per arm, gripper last)
for SUBSTEPS physics steps. Observations are camera images plus the 12 measured joint positions.
"""
import mujoco
import numpy as np

from . import randomize, scene
from .control import GRIP_OPEN, Bimanual
from .ik import ARM_JOINTS
from .oracle import handoff

FPS = 25
CAMERAS = ("top", "a_wrist", "b_wrist")
IMAGE_HW = (192, 256)
TASK = "Hand the spoon from arm A to arm B and place it on the placemat."
JOINTS = [p + j for p in ("a_", "b_") for j in (*ARM_JOINTS, "gripper")]


class Episode:
    def __init__(self, seed: int, render: bool = True, on_step=None):
        self.params = randomize.sample(seed)
        _, self.m, self.d = scene.compile_scene(self.params)
        self.substeps = int(round(1.0 / FPS / self.m.opt.timestep))
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in JOINTS])
        self.act = np.array([self.m.actuator(j).id for j in JOINTS])
        self.renderer = mujoco.Renderer(self.m, *IMAGE_HW) if render else None
        self.ctl = Bimanual(self.m, self.d, on_step=on_step)
        for p in ("a_", "b_"):
            self.ctl.set_now(p, handoff.HOME, GRIP_OPEN)
        mujoco.mj_forward(self.m, self.d)
        self.ctl.hold(0.3)

    def state(self) -> np.ndarray:
        return self.d.qpos[self.qadr].astype(np.float32)

    def command(self) -> np.ndarray:
        return self.d.ctrl[self.act].astype(np.float32)

    def images(self) -> dict:
        out = {}
        for cam in CAMERAS:
            self.renderer.update_scene(self.d, camera=cam)
            out[cam] = self.renderer.render().copy()
        return out

    def observation(self) -> dict:
        obs = {"state": self.state()}
        if self.renderer is not None:
            obs["images"] = self.images()
        return obs

    def step(self, action: np.ndarray):
        """Apply a 12-D joint target for one control period (policy evaluation)."""
        self.d.ctrl[self.act] = action
        for _ in range(self.substeps):
            self.ctl.step()
        return self.observation()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def record_oracle(seed: int):
    """Run the scripted oracle and sample (observation, action) at FPS.

    The action label for frame t is the joint target commanded at frame t+1, i.e. where the oracle is
    driving the arms next. Returns (frames, result) where result carries the grader verdict.
    """
    from . import grader
    from .control import IKFailure

    frames, cmds = [], []
    ep = None

    def on_step(m, d):
        if ep is not None and ep.ctl.steps % ep.substeps == 0:
            frames.append(ep.observation())
            cmds.append(ep.command())

    ep = Episode(seed, render=True, on_step=on_step)
    ep.ctl.steps = 0  # align sampling with the start of the scripted motion
    phases = []
    try:
        handoff.run(ep.ctl, ep.params.target_xy, phases)
        result = grader.grade_handoff(ep.m, ep.d, ep.params.target_xy, trace={})
    except IKFailure as e:
        result = {"success": False, "failure": "ik_unreachable", "detail": str(e)}
    ep.close()
    actions = cmds[1:] + cmds[-1:]
    for f, a in zip(frames, actions):
        f["action"] = a
    result["phases"] = [(name, step // ep.substeps) for name, step in phases]
    return frames, result
