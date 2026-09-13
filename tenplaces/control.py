"""Joint-space motion executor for the two arms (position actuators, min-jerk interpolation)."""
import mujoco
import numpy as np

from .ik import ARM_JOINTS, ArmIK

GRIP_OPEN = 1.0
GRIP_CLOSED = -0.15  # below the contact angle, so the position servo squeezes (force-limited)


class Bimanual:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, on_step=None):
        self.m, self.d = model, data
        self.ik = {p: ArmIK(model, p) for p in ("a_", "b_")}
        self.act = {p: np.array([model.actuator(p + j).id for j in ARM_JOINTS]) for p in ("a_", "b_")}
        self.grip_act = {p: model.actuator(p + "gripper").id for p in ("a_", "b_")}
        self.on_step = on_step  # callback(model, data) after every physics step, e.g. video/recording
        self.steps = 0

    def q(self, p):
        return self.d.ctrl[self.act[p]].copy()

    def solve(self, p, pos, jaw_axis_xy, max_err=3e-3):
        q, pe, re = self.ik[p].solve_robust(self.d.qpos, pos, ArmIK.down_rotation(jaw_axis_xy), q_hint=self.q(p))
        if pe > max_err or re > 0.05:
            raise IKFailure(f"{p} unreachable: pos err {pe * 1000:.1f} mm, rot err {re:.3f} rad at {np.round(pos, 3)}")
        return q

    def set_now(self, p, q, grip=None):
        """Teleport an arm (reset only)."""
        self.d.qpos[self.ik[p].qadr] = q
        self.d.ctrl[self.act[p]] = q
        if grip is not None:
            self.d.ctrl[self.grip_act[p]] = grip
            self.d.qpos[self.m.joint(p + "gripper").qposadr[0]] = grip

    def move(self, targets: dict, grips: dict | None = None, duration=1.0):
        """Interpolate the given arms to joint targets (and gripper commands) over `duration` seconds."""
        grips = grips or {}
        start = {p: self.q(p) for p in targets}
        gstart = {p: self.d.ctrl[self.grip_act[p]] for p in grips}
        n = max(1, int(duration / self.m.opt.timestep))
        for i in range(1, n + 1):
            s = i / n
            s = 10 * s**3 - 15 * s**4 + 6 * s**5  # min-jerk
            for p, q in targets.items():
                self.d.ctrl[self.act[p]] = start[p] + s * (q - start[p])
            for p, g in grips.items():
                self.d.ctrl[self.grip_act[p]] = gstart[p] + s * (g - gstart[p])
            self.step()

    def hold(self, duration):
        for _ in range(int(duration / self.m.opt.timestep)):
            self.step()

    def step(self):
        mujoco.mj_step(self.m, self.d)
        self.steps += 1
        if self.on_step is not None:
            self.on_step(self.m, self.d)


class IKFailure(RuntimeError):
    pass
