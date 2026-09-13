"""Full dinner-table layout (separate from scene.py so the hand-off dataset stays reproducible).

Workflow the layout is built for (arms can only grasp fingers-down on their own half of the table):
  1. A pulls the cutlery drawer: a tray sliding out from under a fixed lid. The cutlery lies across the
     tray (along y) under the lid, so it is only reachable once the drawer is open.
  2. Spoon A -> B, B sets it right of the placemat.   3. B moves the plate onto the placemat (this
     frees the fork's spot).   4. Fork A -> B, B sets it left of the placemat.   5. B sets the cup above
     the spoon at the top corner. Arm B's half is crowded: start spots and targets never overlap in that order.
The plate has a raised rim wall and the cup a thin wall: fingers-down pads straddle a wall, the
standard grasp for a rigid parallel gripper. Every prop is a MuJoCo primitive.
"""
from dataclasses import asdict, dataclass

import mujoco
import numpy as np

from .scene import BASE_X, _lookat_quat, _prepare_arm, _yaw_quat

HALF_THICK = 0.006  # cutlery half thickness


@dataclass
class TableParams:
    seed: int
    drawer_xy: tuple = (-0.008, 0.09)      # tray centre when closed (open handle ~0.1 m from arm A's base)
    drawer_open: float = 0.09              # how far it must slide toward A (-x)
    spoon_offset: tuple = (0.002, 0.0)     # cutlery centre in the tray frame; both lie along y under the lid
    fork_offset: tuple = (0.034, 0.0)
    plate_xy: tuple = (0.17, 0.085)        # plate start (B side)
    cup_xy: tuple = (0.195, -0.14)         # cup start (B side, clear of the spoon's spot)
    mat_xy: tuple = (0.117, 0.0)           # placemat centre (plate target); the plate must land clear of the lid
    friction: float = 1.0
    cutlery_mass: float = 0.04
    plate_mass: float = 0.12
    cup_mass: float = 0.06
    light_dir: tuple = (0.0, 0.0, -1.0)
    light_diffuse: float = 0.7
    table_rgb: tuple = (0.55, 0.42, 0.30)
    floor_rgb: tuple = (0.25, 0.28, 0.32)

    def targets(self) -> dict:
        mx, my = self.mat_xy
        # The fork's spot must end clear of the drawer lid (which spans the +y side up to x ~0.05); the
        # spoon's spot sits 4 cm toward A so the cup fits at the spoon side's top corner, clear of the
        # spoon, the plate and arm B's shoulder.
        return {"plate": (mx, my), "spoon": (mx - 0.04, my - 0.085), "fork": (mx + 0.005, my + 0.085),
                "cup": (mx + 0.075, my - 0.075)}

    def to_dict(self):
        return asdict(self)


def sample(seed: int, nominal: bool = False) -> TableParams:
    if nominal:
        return TableParams(seed=seed)
    rng = np.random.default_rng(seed)
    u = rng.uniform
    return TableParams(
        seed=seed,
        drawer_xy=(u(-0.015, 0.0), u(0.07, 0.10)),  # the open handle must stay >= ~0.1 m from arm A's base
        # Under the lid when closed, and >= 3.6 cm clear of its edge once pulled 0.09.
        spoon_offset=(u(0.0, 0.004), u(-0.005, 0.005)),
        fork_offset=(u(0.032, 0.036), u(-0.005, 0.005)),  # >= 2 cm from the lid edge once open, 5 mm from the spoon's pad
        plate_xy=(u(0.15, 0.19), u(0.07, 0.10)),
        cup_xy=(u(0.19, 0.205), u(-0.15, -0.135)),
        mat_xy=(u(0.11, 0.125), u(-0.015, 0.015)),  # plate edge >= 6 mm clear of the lid's far end
        friction=u(0.7, 1.3),
        cutlery_mass=u(0.025, 0.06),
        plate_mass=u(0.08, 0.16),
        cup_mass=u(0.04, 0.09),
        light_dir=tuple(np.array([u(-0.5, 0.5), u(-0.5, 0.5), -1.0])),
        light_diffuse=u(0.4, 0.9),
        table_rgb=tuple(u(0.2, 0.8, 3)),
        floor_rgb=tuple(u(0.1, 0.5, 3)),
    )


def _ring(body, name, radius, height, wall, z0, n, mass, rgba, friction):
    """A vertical ring wall from n box segments, bottom at z0 in the body frame."""
    seg = 2 * radius * np.sin(np.pi / n) * 1.05
    for i in range(n):
        a = 2 * np.pi * i / n
        body.add_geom(name=f"{name}_wall{i}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[wall / 2, seg / 2, height / 2],
                      pos=[radius * np.cos(a), radius * np.sin(a), z0 + height / 2], quat=_yaw_quat(a),
                      mass=mass / n, rgba=rgba, friction=[friction, 0.005, 0.0001], condim=4)


def build(p: TableParams) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "tenplaces_table"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720
    fr = [p.friction, 0.005, 0.0001]

    wb = spec.worldbody
    wb.add_light(name="sun", pos=[0, 0, 1.5], dir=[float(v) for v in p.light_dir],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[p.light_diffuse] * 3, castshadow=True)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75], rgba=[*p.floor_rgb, 1])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.48, 0.32, 0.02], pos=[0, 0, -0.02],
                rgba=[*p.table_rgb, 1], friction=fr)
    for prefix, x, yaw in (("a_", -BASE_X, 0.0), ("b_", BASE_X, np.pi)):
        frame = wb.add_frame(pos=[x, 0, 0], quat=_yaw_quat(yaw))
        frame.attach_body(_prepare_arm(p.friction).body("base"), prefix, "")

    # Place-setting markers (visual only): placemat for the plate, outlines for spoon / fork / cup.
    t = p.targets()
    wb.add_geom(name="placemat", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.075, 0.13, 0.0005], pos=[*t["plate"], 0.0005],
                rgba=[0.25, 0.55, 0.35, 1], contype=0, conaffinity=0)
    for key, size in (("spoon", [0.085, 0.012]), ("fork", [0.085, 0.012])):
        wb.add_geom(name=f"{key}_spot", type=mujoco.mjtGeom.mjGEOM_BOX, size=[*size, 0.0006], pos=[*t[key], 0.0011],
                    rgba=[0.9, 0.9, 0.8, 1], contype=0, conaffinity=0)
    wb.add_geom(name="cup_spot", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.0006, 0], pos=[*t["cup"], 0.0011],
                rgba=[0.9, 0.9, 0.8, 1], contype=0, conaffinity=0)

    # Drawer: fixed lid on posts; a tray on a slide joint underneath; a pull post at the tray's -x end.
    dx, dy = p.drawer_xy
    tray_half = (0.075, 0.08)  # the cutlery (13.5 cm) lies across the tray
    lid_z = 0.034
    wood = [0.45, 0.30, 0.18, 1]
    # The lid covers the back of the tray, where the cutlery lies when the drawer is closed.
    lid_x0, lid_x1 = dx - 0.012, dx + 0.052  # covers the cutlery only; ends short of the placemat area
    lid_c, lid_len = (lid_x0 + lid_x1) / 2, (lid_x1 - lid_x0) / 2
    wb.add_geom(name="lid", type=mujoco.mjtGeom.mjGEOM_BOX, size=[lid_len, tray_half[1] + 0.012, 0.004],
                pos=[lid_c, dy, lid_z], rgba=wood, friction=fr)
    for sy in (-1, 1):
        wb.add_geom(name=f"lid_post{sy}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[lid_len, 0.004, lid_z / 2],
                    pos=[lid_c, dy + sy * (tray_half[1] + 0.008), lid_z / 2], rgba=wood, friction=fr)
    tray = wb.add_body(name="drawer", pos=[dx, dy, 0.0])
    tray.add_joint(name="drawer_slide", type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[1, 0, 0], range=[-0.10, 0.0],
                   damping=2.0, frictionloss=0.05)
    tray.add_geom(name="tray_floor", type=mujoco.mjtGeom.mjGEOM_BOX, size=[tray_half[0], tray_half[1], 0.002],
                  pos=[0, 0, 0.002], mass=0.05, rgba=wood, friction=fr)
    for sx in (-1, 1):
        tray.add_geom(name=f"tray_end{sx}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.002, tray_half[1], 0.009],
                      pos=[sx * tray_half[0], 0, 0.011], mass=0.01, rgba=wood, friction=fr)
    for sy in (-1, 1):
        tray.add_geom(name=f"tray_side{sy}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[tray_half[0], 0.002, 0.009],
                      pos=[0, sy * tray_half[1], 0.011], mass=0.01, rgba=wood, friction=fr)
    # Pull post: 16 mm thick along x (= the pad gap at gripper angle 0), gripped with the jaws closing
    # along x. A 12 mm gap to the tray's end wall leaves room for the fixed pad, which pushes the post
    # when A pulls toward itself (no reliance on friction).
    tray.add_geom(name="drawer_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.008, 0.012, 0.016],
                  pos=[-tray_half[0] - 0.022, 0, 0.016], mass=0.01, rgba=[0.2, 0.2, 0.22, 1], friction=fr)
    tray.add_geom(name="drawer_handle_bar", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.012, 0.004, 0.004],
                  pos=[-tray_half[0] - 0.010, 0, 0.004], mass=0.005, rgba=[0.2, 0.2, 0.22, 1], friction=fr)
    tray.add_site(name="drawer_grip", pos=[-tray_half[0] - 0.022, 0, 0.022], size=[0.004, 0, 0], group=4)

    # Cutlery lying across the tray (axis along +y): grip_a is the end nearer the table centre. A turns
    # each utensil to point along +x before the hand-off (B takes grip_b).
    for name, off, head in (("spoon", p.spoon_offset, "bowl"), ("fork", p.fork_offset, "tines")):
        b = wb.add_body(name=name, pos=[dx + off[0], dy + off[1], 0.004 + HALF_THICK], quat=_yaw_quat(np.pi / 2))
        b.add_freejoint(name=f"{name}_free")
        kw = dict(friction=fr, condim=4, rgba=[0.8, 0.8, 0.85, 1])
        b.add_geom(name=f"{name}_handle", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.055, 0.008, HALF_THICK],
                   mass=0.8 * p.cutlery_mass, **kw)
        if head == "bowl":
            b.add_geom(name=f"{name}_head", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID, size=[0.016, 0.012, HALF_THICK],
                       pos=[0.066, 0, 0], mass=0.2 * p.cutlery_mass, **kw)
        else:
            b.add_geom(name=f"{name}_head", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.014, 0.011, HALF_THICK * 0.7],
                       pos=[0.066, 0, -0.002], mass=0.2 * p.cutlery_mass, **kw)
        b.add_site(name=f"{name}_grip_a", pos=[-0.035, 0, 0], size=[0.003, 0, 0], group=4)
        b.add_site(name=f"{name}_grip_b", pos=[0.035, 0, 0], size=[0.003, 0, 0], group=4)

    # Plate: disc with a raised rim wall (pads straddle the wall).
    plate = wb.add_body(name="plate", pos=[*p.plate_xy, 0.0])
    plate.add_freejoint(name="plate_free")
    white = [0.95, 0.95, 0.93, 1]
    plate.add_geom(name="plate_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.052, 0.003, 0], pos=[0, 0, 0.003],
                   mass=0.6 * p.plate_mass, rgba=white, friction=fr, condim=4)
    _ring(plate, "plate", 0.050, 0.016, 0.004, 0.006, 16, 0.4 * p.plate_mass, white, p.friction)
    # Grasp point on the +y side of the rim: reachable at both the start and the placemat (the +x rim
    # is not reachable at the placemat, too close to arm B's base).
    plate.add_site(name="plate_grip", pos=[0, 0.050, 0.019], size=[0.003, 0, 0], group=4)

    # Cup: thin-walled cylinder (pads straddle the wall at the rim).
    cup = wb.add_body(name="cup", pos=[*p.cup_xy, 0.0])
    cup.add_freejoint(name="cup_free")
    blue = [0.35, 0.5, 0.8, 1]
    cup.add_geom(name="cup_base", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.024, 0.003, 0], pos=[0, 0, 0.003],
                 mass=0.5 * p.cup_mass, rgba=blue, friction=fr, condim=4)
    # Espresso-cup height (3.5 cm): arm B's fingers-down reach near its base ends ~7 cm up, and the cup
    # must clear the plate rim while carried past it.
    _ring(cup, "cup", 0.023, 0.035, 0.004, 0.006, 12, 0.5 * p.cup_mass, blue, p.friction)
    # -x rim: B's jaw then points away from its own base (36/40 start/target poses reachable vs ~29/40
    # for the side rims); a +x rim would point B's jaw back at its base, which the wrist cannot reach.
    cup.add_site(name="cup_grip", pos=[-0.023, 0, 0.036], size=[0.003, 0, 0], group=4)

    wb.add_camera(name="top", pos=[0, -0.42, 0.62], quat=_lookat_quat([0, -0.42, 0.62], [0, 0.0, 0.0]), fovy=55)
    wb.add_camera(name="front", pos=[0.0, -0.75, 0.35], quat=_lookat_quat([0.0, -0.75, 0.35], [0, 0, 0.05]), fovy=45)
    return spec


def compile_scene(p: TableParams):
    spec = build(p)
    model = spec.compile()
    return spec, model, mujoco.MjData(model)
