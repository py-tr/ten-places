"""Scripted privileged-state oracle for the full dinner-table workflow (scene_table layout).

Skills (each also usable alone, for per-skill demos):
  open_drawer   A grips the pull post and slides the tray out toward itself
  cutlery(name) A picks the utensil (lying across the open tray), turns it to point along +x while
                carrying it to the midline, B takes the far end and sets it on its spot
  plate         B pinches the plate's rim wall, moves it onto the placemat
  cup           B pinches the cup's rim wall, sets it on its spot
Lessons from the hand-off spike are built in: grip points keep ~3 mm clearance to the fixed jaw,
approach heights stay inside fingers-down reach, pads never press into the table, and every carry is
a straight line through IK waypoints (a joint-space move between equal-height poses dips mid-way).
"""
import numpy as np

from ..control import GRIP_CLOSED, Bimanual
from .handoff import HOME

PARTIAL_OPEN = 0.35     # jaw gap ~4 cm: rims and the drawer post
CUTLERY_OPEN = 0.13     # small opening: the moving pad must clear the neighbouring fork / lid edge
CUTLERY_GRIP_Z = 0.017  # utensil centre 0.010 on the tray floor; pads end 5 mm above the floor
CARRY_Z = 0.065         # clears the drawer lid (top at 0.038) with the utensil hanging below the pads
EXCHANGE_Y = -0.065     # hand-off point: B's fixed pad stays ~2 cm clear of the drawer lid (starts at y ~ -0.02)
EXCHANGE_Z = 0.045      # hand off low: B's approach above it must stay inside fingers-down reach
HANDLE_GRIP_Z = 0.022
RIM_INSET = 0.008       # grip site sits this far beyond the wall centre, toward the moving jaw


def _axis_xy(d, m, body):
    R = d.xmat[m.body(body).id].reshape(3, 3)
    v = R[:, 0][:2]
    return v / np.linalg.norm(v)


def _perp(v):
    return np.array([-v[1], v[0]])


def _rot(v, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def _signed_angle(a, b):
    return float(np.arctan2(a[0] * b[1] - a[1] * b[0], np.dot(a, b)))


def _line(ctl, arm, start, end, jaw, n=5, duration=1.5, jaw_end=None):
    """Move the grip site along a straight line through n IK waypoints, optionally turning the jaw."""
    turn = 0.0 if jaw_end is None else _signed_angle(jaw, jaw_end)
    for i in range(1, n + 1):
        f = i / n
        p = np.asarray(start) + (np.asarray(end) - np.asarray(start)) * f
        ctl.move({arm: ctl.solve(arm, p, _rot(jaw, turn * f))}, duration=duration / n)


def open_drawer(ctl: Bimanual, open_dist: float, mark=lambda name: None):
    d = ctl.d
    mark("open_drawer")
    # Jaws close along x with the moving pad on the outer (-x) side and the rigid fixed pad in the gap
    # between post and tray: pulling toward A, the fixed pad pushes the post.
    jaw = np.array([-1.0, 0.0])
    # The post cannot move sideways, so the fixed pad must start against it (1.5 mm clearance) for both
    # pads to squeeze; the usual clearance for loose objects would leave a one-pad grasp.
    slide = ctl.m.joint("drawer_slide").qposadr[0]
    end_x = d.site("drawer_grip").xpos[0] + 0.0035 * jaw[0] - open_dist  # where the handle must end up
    # Closed loop: the tray lags the gripper during a pull (it opened 6-7 cm of a commanded 9), so check
    # the tray position and re-grip for the remainder, at most twice.
    for _ in range(3):
        if -d.qpos[slide] >= open_dist - 0.005:
            break
        h = d.site("drawer_grip").xpos.copy()
        h[:2] += 0.0035 * jaw
        ctl.move({"a_": ctl.solve("a_", [h[0], h[1], HANDLE_GRIP_Z + 0.05], jaw)}, {"a_": PARTIAL_OPEN}, 1.0)
        ctl.move({"a_": ctl.solve("a_", [h[0], h[1], HANDLE_GRIP_Z], jaw)}, duration=0.6)
        ctl.move({}, {"a_": GRIP_CLOSED}, 0.4)
        ctl.hold(0.2)
        _line(ctl, "a_", [h[0], h[1], HANDLE_GRIP_Z], [end_x, h[1], HANDLE_GRIP_Z], jaw, n=6, duration=1.8)
        ctl.move({}, {"a_": PARTIAL_OPEN}, 0.3)
        ctl.move({"a_": ctl.solve("a_", [end_x, h[1], HANDLE_GRIP_Z + 0.05], jaw)}, duration=0.5)


def cutlery(ctl: Bimanual, name: str, target_xy, mark=lambda name: None):
    m, d = ctl.m, ctl.d
    mark(f"{name}_pick")
    axis = _axis_xy(d, m, name)  # ~ +y while lying in the tray
    grip_a = d.site(f"{name}_grip_a").xpos.copy()
    # Jaw toward +x: arm A cannot point its jaw back toward its own base (jaw -x is unreachable at every
    # height here), so the moving jaw opens toward +x with a small opening (CUTLERY_OPEN).
    jaw = _perp(axis)
    if jaw[0] < 0:
        jaw = -jaw
    ctl.move({"a_": ctl.solve("a_", [grip_a[0], grip_a[1], CARRY_Z], jaw)}, {"a_": CUTLERY_OPEN}, 1.0)
    ctl.move({"a_": ctl.solve("a_", [grip_a[0], grip_a[1], CUTLERY_GRIP_Z], jaw)}, duration=0.7)
    ctl.move({}, {"a_": GRIP_CLOSED}, 0.4)
    ctl.hold(0.2)

    mark(f"{name}_carry")
    ctl.move({"a_": ctl.solve("a_", [grip_a[0], grip_a[1], CARRY_Z], jaw)}, duration=0.7)
    # Turn the utensil so it points along +x (grip_b toward B): the jaw turns by the same angle.
    ex_axis = np.array([1.0, 0.0])
    ex_jaw = _rot(jaw, _signed_angle(_axis_xy(d, m, name), ex_axis))
    _line(ctl, "a_", [grip_a[0], grip_a[1], CARRY_Z], [-0.035, EXCHANGE_Y, CARRY_Z], jaw, n=10, duration=1.6,
          jaw_end=ex_jaw)
    ctl.move({"a_": ctl.solve("a_", [-0.035, EXCHANGE_Y, EXCHANGE_Z], ex_jaw)}, duration=0.5)
    ctl.hold(0.2)

    mark(f"{name}_exchange")
    grip_b = d.site(f"{name}_grip_b").xpos.copy()
    # B's moving jaw opens toward -y, away from the drawer lid (toward +y it lands on the lid).
    b_jaw = _perp(_axis_xy(d, m, name))
    if b_jaw[1] > 0:
        b_jaw = -b_jaw
    ctl.move({"b_": ctl.solve("b_", [grip_b[0], grip_b[1], grip_b[2] + 0.025], b_jaw)}, {"b_": CUTLERY_OPEN}, 1.0)
    ctl.move({"b_": ctl.solve("b_", [grip_b[0], grip_b[1], grip_b[2] + 0.001], b_jaw)}, duration=0.6)
    ctl.move({}, {"b_": GRIP_CLOSED}, 0.4)
    ctl.hold(0.2)
    ctl.move({}, {"a_": PARTIAL_OPEN}, 0.3)
    # Retreat only 1.5 cm: at y=-0.065 fingers-down reach ends between 6.5 and 7.5 cm.
    ctl.move({"a_": ctl.solve("a_", [-0.035, EXCHANGE_Y, EXCHANGE_Z + 0.015], ex_jaw)}, duration=0.4)
    ctl.move({"a_": HOME}, duration=0.9)

    mark(f"{name}_place")
    tgt = np.asarray(target_xy, float) + 0.035 * ex_axis
    held = d.site("b_grip").xpos.copy()
    _line(ctl, "b_", [held[0], held[1], CARRY_Z - 0.01], [tgt[0], tgt[1], CARRY_Z - 0.01], b_jaw, n=8, duration=1.2)
    ctl.move({"b_": ctl.solve("b_", [tgt[0], tgt[1], 0.015], b_jaw)}, duration=0.7)
    ctl.move({}, {"b_": PARTIAL_OPEN}, 0.3)
    ctl.move({"b_": ctl.solve("b_", [tgt[0], tgt[1], 0.05], b_jaw)}, duration=0.5)


def _rim_move(ctl: Bimanual, body: str, site: str, grip_z: float, target_xy, carry_z: float, mark, hover=0.025,
              via_xy=None):
    """B pinches a vertical rim wall (fixed pad inside, moving pad outside) and moves the object."""
    m, d = ctl.m, ctl.d
    mark(f"{body}_pick")
    centre = d.xpos[m.body(body).id][:2].copy()
    wall = d.site(site).xpos[:2].copy()
    out = (wall - centre) / np.linalg.norm(wall - centre)  # outward normal at the grasp point
    grip = wall + RIM_INSET * out
    jaw = out  # moving jaw opens outward, fixed pad goes inside the rim
    ctl.move({"b_": ctl.solve("b_", [grip[0], grip[1], grip_z + hover], jaw)}, {"b_": PARTIAL_OPEN}, 1.0)
    ctl.move({"b_": ctl.solve("b_", [grip[0], grip[1], grip_z], jaw)}, duration=0.7)
    ctl.move({}, {"b_": GRIP_CLOSED}, 0.4)
    ctl.hold(0.25)
    mark(f"{body}_carry")
    ctl.move({"b_": ctl.solve("b_", [grip[0], grip[1], carry_z], jaw)}, duration=0.7)
    goal = np.asarray(target_xy, float) + (grip - centre)
    start = [grip[0], grip[1], carry_z]
    if via_xy is not None:  # detour, e.g. to keep the object away from arm B's own links
        via = np.asarray(via_xy, float) + (grip - centre)
        _line(ctl, "b_", start, [via[0], via[1], carry_z], jaw, n=6, duration=0.9)
        start = [via[0], via[1], carry_z]
    _line(ctl, "b_", start, [goal[0], goal[1], carry_z], jaw, n=8, duration=1.2)
    ctl.move({"b_": ctl.solve("b_", [goal[0], goal[1], grip_z + 0.002], jaw)}, duration=0.7)
    ctl.move({}, {"b_": PARTIAL_OPEN}, 0.3)
    ctl.move({"b_": ctl.solve("b_", [goal[0], goal[1], grip_z + hover], jaw)}, duration=0.5)


def plate(ctl: Bimanual, target_xy, mark=lambda name: None):
    _rim_move(ctl, "plate", "plate_grip", grip_z=0.019, target_xy=target_xy, carry_z=0.06, mark=mark)


def cup(ctl: Bimanual, target_xy, mark=lambda name: None, via_xy=None, scale: float = 1.0):
    # Approach 5.8 cm (pads clear the 4.1 cm rim), carry 5.8 cm: the cup bottom clears the plate rim by
    # ~6 mm, and both stay inside B's fingers-down reach near its base. A taller cup (scale > 1) raises the
    # approach by its extra rim height, so the pads still clear it; the grasp and the carry are the same distance
    # above the cup's bottom at any size, so they stay.
    _rim_move(ctl, "cup", "cup_grip", grip_z=0.030, target_xy=target_xy, carry_z=0.058, mark=mark,
              hover=0.028 + 0.035 * max(0.0, scale - 1.0), via_xy=via_xy)


def run_plan(ctl: Bimanual, params, steps, phases: list | None = None):
    """Execute any verified plan (a subset / order of drawer, spoon, plate, fork, cup)."""
    mark = (lambda name: phases.append((name, ctl.steps))) if phases is not None else (lambda name: None)
    t = params.targets()
    for skill in steps:
        if skill == "drawer":
            open_drawer(ctl, params.drawer_open, mark)
            ctl.move({"a_": HOME}, duration=0.8)
        elif skill in ("spoon", "fork"):
            cutlery(ctl, skill, t[skill], mark)
        elif skill == "plate":
            plate(ctl, t["plate"], mark)
        elif skill == "cup":
            cup(ctl, t["cup"], mark, scale=params.cup_scale)
        else:
            raise ValueError(f"unknown skill {skill}")
        ctl.move({"a_": HOME, "b_": HOME}, duration=0.8)
    ctl.hold(0.5)
    mark("done")


def run(ctl: Bimanual, params, phases: list | None = None):
    """Full workflow: drawer -> spoon -> plate -> fork -> cup (the plate starts on the fork's spot)."""
    mark = (lambda name: phases.append((name, ctl.steps))) if phases is not None else (lambda name: None)
    t = params.targets()
    open_drawer(ctl, params.drawer_open, mark)
    ctl.move({"a_": HOME}, duration=0.8)
    cutlery(ctl, "spoon", t["spoon"], mark)
    ctl.move({"b_": HOME}, duration=0.8)
    plate(ctl, t["plate"], mark)
    ctl.move({"b_": HOME}, duration=0.8)
    cutlery(ctl, "fork", t["fork"], mark)
    ctl.move({"b_": HOME}, duration=0.8)
    cup(ctl, t["cup"], mark, scale=params.cup_scale)
    ctl.move({"a_": HOME, "b_": HOME}, duration=1.0)
    ctl.hold(0.5)
    mark("done")
