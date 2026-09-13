"""Per-sub-task success predicates for the dinner-table layout, from simulator state (evaluation only)."""
import numpy as np

from .grader import touching

POS_TOL = 0.025
DRAWER_OPEN_MIN = 0.06
PADS = ("a_pad", "b_pad")


def _flat(m, d, body, z_max):
    return d.xpos[m.body(body).id][2] < z_max


def _upright(m, d, body):
    return d.xmat[m.body(body).id].reshape(3, 3)[2, 2] > 0.95


def _near(m, d, body, target_xy):
    return float(np.linalg.norm(d.xpos[m.body(body).id][:2] - np.asarray(target_xy)))


def grade_table(m, d, params) -> dict:
    t = params.targets()
    out = {"drawer_open": bool(-d.qpos[m.joint("drawer_slide").qposadr[0]] >= DRAWER_OPEN_MIN)}
    for name in ("spoon", "fork"):
        R = d.xmat[m.body(name).id].reshape(3, 3)
        aligned = abs(R[0, 0]) > 0.9
        err = _near(m, d, name, t[name])
        out[name] = bool(err < POS_TOL and _flat(m, d, name, 0.02) and aligned and not touching(m, d, name, PADS))
        out[f"{name}_err_m"] = round(err, 4)
    for name, z_max in (("plate", 0.005), ("cup", 0.005)):
        err = _near(m, d, name, t[name])
        out[name] = bool(err < POS_TOL and _flat(m, d, name, z_max) and _upright(m, d, name)
                         and not touching(m, d, name, PADS))
        out[f"{name}_err_m"] = round(err, 4)
    steps = ("drawer_open", "spoon", "fork", "plate", "cup")
    out["subtasks_done"] = sum(out[s] for s in steps)
    out["success"] = out["subtasks_done"] == len(steps)
    out["failed"] = [s for s in steps if not out[s]]
    return out
