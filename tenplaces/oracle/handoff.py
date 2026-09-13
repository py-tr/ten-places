"""Scripted privileged-state oracle for the spoon hand-off (A picks -> A/B exchange -> B places).

Used only to generate demonstrations and to label failure states; it reads true object poses from
the simulator, which the learned policy never sees.
"""
import numpy as np

from ..control import GRIP_CLOSED, GRIP_OPEN, Bimanual

GRASP_Z = 0.013  # grip-site height when grasping the spoon from the table (pads clear the table)
EXCHANGE_Z = 0.045  # spoon height during the exchange (fingers-down reach ends ~0.1 m up)
EXCHANGE_HOVER = 0.03  # approach/retreat height above the exchange, kept inside reach
HOVER = 0.06
HOME = np.array([0.0, -1.4, 1.3, 1.1, 0.0])  # folded, clear of the shared workspace


def _spoon_axis(d, m):
    R = d.xmat[m.body("spoon").id].reshape(3, 3)
    return R[:, 0][:2] / np.linalg.norm(R[:, 0][:2])


def _perp(axis_xy):
    return np.array([-axis_xy[1], axis_xy[0]])


def run(ctl: Bimanual, target_xy, phases: list | None = None):
    """Execute the hand-off. Appends (phase_name, sim_step) to `phases` as each phase starts."""
    m, d = ctl.m, ctl.d
    mark = (lambda name: phases.append((name, ctl.steps))) if phases is not None else (lambda name: None)

    # 1. A picks its end of the spoon from the table.
    mark("a_reach")
    axis = _spoon_axis(d, m)
    grip_a = d.site("spoon_grip_a").xpos.copy()
    jaw = _perp(axis)
    pre = ctl.solve("a_", [grip_a[0], grip_a[1], GRASP_Z + HOVER], jaw)
    ctl.move({"a_": pre}, {"a_": GRIP_OPEN}, 1.2)
    ctl.move({"a_": ctl.solve("a_", [grip_a[0], grip_a[1], GRASP_Z], jaw)}, duration=0.6)
    mark("a_grasp")
    ctl.move({}, {"a_": GRIP_CLOSED}, 0.4)
    ctl.hold(0.2)

    # 2. A carries the spoon to the exchange pose: spoon centred on the midline, aligned with x.
    mark("a_carry")
    ex_axis = np.array([1.0, 0.0])
    ex_centre = np.array([0.0, float(np.clip(grip_a[1], -0.05, 0.05))])
    lift = ctl.solve("a_", [grip_a[0], grip_a[1], EXCHANGE_Z], jaw)
    ctl.move({"a_": lift}, duration=0.6)
    a_ex = ctl.solve("a_", [ex_centre[0] - 0.045, ex_centre[1], EXCHANGE_Z], _perp(ex_axis))
    ctl.move({"a_": a_ex}, duration=1.0)
    ctl.hold(0.2)

    # 3. B takes the other end (grip point read from the spoon's actual pose), then A lets go.
    mark("b_reach")
    grip_b = d.site("spoon_grip_b").xpos.copy()
    b_axis = _perp(_spoon_axis(d, m))
    ctl.move({"b_": ctl.solve("b_", [grip_b[0], grip_b[1], grip_b[2] + EXCHANGE_HOVER], b_axis)}, {"b_": GRIP_OPEN}, 1.2)
    ctl.move({"b_": ctl.solve("b_", [grip_b[0], grip_b[1], grip_b[2] + 0.001], b_axis)}, duration=0.6)
    mark("b_grasp")
    ctl.move({}, {"b_": GRIP_CLOSED}, 0.4)
    ctl.hold(0.2)
    mark("a_release")
    ctl.move({}, {"a_": GRIP_OPEN}, 0.3)
    ctl.move({"a_": ctl.solve("a_", [ex_centre[0] - 0.045, ex_centre[1], EXCHANGE_Z + EXCHANGE_HOVER], _perp(ex_axis))}, duration=0.5)
    ctl.move({"a_": HOME}, duration=1.0)

    # 4. B sets the spoon on the placemat and retreats.
    mark("b_place")
    target = np.asarray(target_xy, float)
    place_grip = target + 0.045 * ex_axis
    ctl.move({"b_": ctl.solve("b_", [place_grip[0], place_grip[1], EXCHANGE_Z], _perp(ex_axis))}, duration=1.0)
    ctl.move({"b_": ctl.solve("b_", [place_grip[0], place_grip[1], GRASP_Z + 0.002], _perp(ex_axis))}, duration=0.7)
    ctl.move({}, {"b_": GRIP_OPEN}, 0.3)
    mark("b_retreat")
    ctl.move({"b_": ctl.solve("b_", [place_grip[0], place_grip[1], GRASP_Z + HOVER], _perp(ex_axis))}, duration=0.5)
    ctl.move({"b_": HOME}, duration=1.0)
    ctl.hold(0.5)
    mark("done")
