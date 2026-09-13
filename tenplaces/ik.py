"""Damped-least-squares IK for one SO-101 grip site (position + full orientation).

The SO-101 is 5-DOF, but its links move in one vertical plane: a fingers-down target costs one
constraint and wrist_roll supplies yaw, so "grip point + fingers down + jaw axis" is exactly solvable
wherever the point is in reach.
"""
import mujoco
import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


class ArmIK:
    def __init__(self, model: mujoco.MjModel, prefix: str):
        self.m = model
        self.d = mujoco.MjData(model)
        self.prefix = prefix
        self.site = model.site(prefix + "grip").id
        joints = [model.joint(prefix + j) for j in ARM_JOINTS]
        self.qadr = np.array([j.qposadr[0] for j in joints])
        self.dadr = np.array([j.dofadr[0] for j in joints])
        self.lo = np.array([j.range[0] for j in joints])
        self.hi = np.array([j.range[1] for j in joints])

    @staticmethod
    def down_rotation(jaw_axis_xy) -> np.ndarray:
        """Target site rotation: fingers point down (site z = world +z), jaws open along `jaw_axis_xy`."""
        x = np.array([jaw_axis_xy[0], jaw_axis_xy[1], 0.0])
        x /= np.linalg.norm(x)
        z = np.array([0.0, 0.0, 1.0])
        return np.stack([x, np.cross(z, x), z], axis=1)

    def solve(self, qpos_full: np.ndarray, target_pos, target_rot, q_init=None,
              iters=200, rot_weight=0.35, damping=1e-3, tol=1e-4):
        """Return (q[5], position_error_m, rotation_error_rad). Other joints are held at qpos_full."""
        d = self.d
        d.qpos[:] = qpos_full
        q = (np.array(q_init) if q_init is not None else qpos_full[self.qadr]).astype(float).copy()
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        target_pos = np.asarray(target_pos, float)
        pos_err = rot_err = np.inf
        for _ in range(iters):
            d.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.m, d)
            mujoco.mj_comPos(self.m, d)
            p = d.site_xpos[self.site]
            R = d.site_xmat[self.site].reshape(3, 3)
            e_p = target_pos - p
            e_r = 0.5 * sum(np.cross(R[:, i], target_rot[:, i]) for i in range(3))
            pos_err, rot_err = np.linalg.norm(e_p), np.linalg.norm(e_r)
            if pos_err < tol and rot_err < 10 * tol:
                break
            mujoco.mj_jacSite(self.m, d, jacp, jacr, self.site)
            J = np.vstack([jacp[:, self.dadr], rot_weight * jacr[:, self.dadr]])
            e = np.concatenate([e_p, rot_weight * e_r])
            dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        return q, pos_err, rot_err

    def solve_robust(self, qpos_full, target_pos, target_rot, q_hint=None, **kw):
        """Try the hint first, then a few fixed restarts; return the best solution."""
        starts = [q_hint] if q_hint is not None else []
        starts += [qpos_full[self.qadr], np.array([0, -0.5, 0.8, 1.2, 0]), np.array([0, 0.3, 0.3, 1.4, 0]),
                   np.array([0, -1.0, 1.4, 0.6, 1.5]), np.array([0, -1.0, 1.4, 0.6, -1.5])]
        best = None
        for s in starts:
            q, pe, re = self.solve(qpos_full, target_pos, target_rot, q_init=s, **kw)
            if best is None or pe + 0.05 * re < best[1] + 0.05 * best[2]:
                best = (q, pe, re)
            if pe < 1e-3 and re < 1e-2:
                break
        return best
