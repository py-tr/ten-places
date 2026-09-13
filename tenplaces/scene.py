"""Procedural MuJoCo scene: two SO-101 arms facing each other across a dinner table.

Built from scratch with MjSpec from the official SO-ARM100 assets (Apache-2.0). All props are MuJoCo
primitives, so there are no third-party mesh licences and contacts stay stable.

Frame conventions
- World: table top at z=0, +x points from arm A to arm B.
- Arm A ("a_") sits at x=-BASE_X facing +x; arm B ("b_") at x=+BASE_X facing -x.
- Each gripper body frame has the fingers along -z and the moving jaw opening along +x. The site
  "<p>grip" is the point between the jaw pads where a grasped object's centre sits.
"""
import mujoco
import numpy as np

from .paths import SO101_XML
from .randomize import SceneParams

BASE_X = 0.30
SPOON_HALF_THICK = 0.006

# Jaw pads, measured from the SO-ARM100 meshes (gripper body frame): fixed jaw inner face at
# x=-0.008, moving jaw inner face at x=+0.008 when the gripper joint is 0.
PAD_HALF = np.array([0.004, 0.008, 0.012])  # thick enough that a squeezing pad cannot tunnel into a handle
GRIPPER_FORCE = 1.2  # N·m on the jaw hinge (~17 N at the pads); the stock 3.35 drives pads through thin objects
PAD_Z = -0.095
GRIP_SITE = np.array([0.005, 0.0, PAD_Z])  # ~4-5 mm clearance to the fixed jaw for typical spoon widths
FINGER_BODIES = ("gripper", "moving_jaw_so101_v1")


def _quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def _quat_rotate(q, v):
    out = np.zeros(3)
    mujoco.mju_rotVecQuat(out, np.asarray(v, float), np.asarray(q, float))
    return out


def _yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def _lookat_quat(pos, target, up=(0, 0, 1)):
    """Camera quat: MuJoCo cameras look along -z with +y up."""
    fwd = np.asarray(target, float) - np.asarray(pos, float)
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, up)
    right /= np.linalg.norm(right)
    cam_up = np.cross(right, fwd)
    mat = np.stack([right, cam_up, -fwd], axis=1).reshape(-1)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, mat)
    return q


def _prepare_arm(friction: float) -> mujoco.MjSpec:
    """Load one SO-101 and replace the finger meshes' convex-hull collisions with box pads."""
    arm = mujoco.MjSpec.from_file(str(SO101_XML))
    for g in arm.geoms:
        if g.parent.name in FINGER_BODIES and g.group == 3:
            g.contype = 0
            g.conaffinity = 0

    arm.actuator("gripper").forcerange = [-GRIPPER_FORCE, GRIPPER_FORCE]
    pad_kw = dict(type=mujoco.mjtGeom.mjGEOM_BOX, size=PAD_HALF, friction=[1.5 * friction, 0.02, 0.002],
                  condim=4, solref=[0.004, 1], solimp=[0.95, 0.99, 0.001, 0.5, 2],
                  rgba=[0.15, 0.15, 0.15, 1], group=3)
    gripper = arm.body("gripper")
    gripper.add_geom(name="pad_fixed", pos=[-0.008 - PAD_HALF[0], 0, PAD_Z], **pad_kw)
    gripper.add_site(name="grip", pos=GRIP_SITE, size=[0.004, 0, 0], rgba=[1, 0, 0, 0.5], group=4)

    # The moving pad lives in the jaw's body frame; place it using the jaw pose at joint angle 0.
    jaw = arm.body("moving_jaw_so101_v1")
    p_j, q_j = np.array(jaw.pos), np.array(jaw.quat)
    centre_g = np.array([0.008 + PAD_HALF[0], 0, PAD_Z])
    jaw.add_geom(name="pad_moving", pos=_quat_rotate(_quat_conj(q_j), centre_g - p_j), quat=_quat_conj(q_j), **pad_kw)
    gripper.add_camera(name="wrist", pos=[0.0, 0.045, -0.02], quat=_lookat_quat([0, 0.045, -0.02], [0, 0, -0.12], up=(1, 0, 0)), fovy=75)
    return arm


def build(params: SceneParams) -> mujoco.MjSpec:
    spec = mujoco.MjSpec()
    spec.modelname = "tenplaces"
    spec.option.timestep = 0.002
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10
    spec.visual.headlight.ambient = [0.25, 0.25, 0.25]
    spec.visual.headlight.diffuse = [0.3, 0.3, 0.3]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    wb = spec.worldbody
    wb.add_light(name="sun", pos=[0, 0, 1.5], dir=[float(v) for v in params.light_dir],
                 type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[params.light_diffuse] * 3, castshadow=True)
    wb.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.75],
                rgba=[*params.floor_rgb, 1])
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.48, 0.32, 0.02], pos=[0, 0, -0.02],
                rgba=[*params.table_rgb, 1], friction=[params.friction, 0.005, 0.0001])

    for prefix, x, yaw in (("a_", -BASE_X, 0.0), ("b_", BASE_X, np.pi)):
        frame = wb.add_frame(pos=[x, 0, 0], quat=_yaw_quat(yaw))
        frame.attach_body(_prepare_arm(params.friction).body("base"), prefix, "")

    # Placemat: where arm B must set the spoon down. Visual only.
    wb.add_geom(name="placemat", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.035, 0.0005, 0],
                pos=[*params.target_xy, 0.0005], rgba=[0.2, 0.7, 0.3, 1], contype=0, conaffinity=0)

    # Spoon: long handle so the two grippers can hold opposite ends, bowl at the +x end.
    spoon = wb.add_body(name="spoon", pos=[*params.spoon_xy, SPOON_HALF_THICK], quat=_yaw_quat(params.spoon_yaw))
    spoon.add_freejoint(name="spoon_free")
    half_len = params.spoon_length / 2
    obj_kw = dict(friction=[params.friction, 0.005, 0.0001], condim=4, rgba=[0.8, 0.8, 0.85, 1])
    spoon.add_geom(name="spoon_handle", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[half_len, params.spoon_width / 2, SPOON_HALF_THICK], mass=0.8 * params.spoon_mass, **obj_kw)
    spoon.add_geom(name="spoon_bowl", type=mujoco.mjtGeom.mjGEOM_ELLIPSOID,
                   size=[0.022, 0.016, SPOON_HALF_THICK], pos=[half_len + 0.012, 0, 0], mass=0.2 * params.spoon_mass, **obj_kw)
    spoon.add_site(name="spoon_grip_a", pos=[-0.045, 0, 0], size=[0.003, 0, 0], group=4)
    spoon.add_site(name="spoon_grip_b", pos=[0.045, 0, 0], size=[0.003, 0, 0], group=4)

    wb.add_camera(name="top", pos=[0, -0.42, 0.62], quat=_lookat_quat([0, -0.42, 0.62], [0, 0.0, 0.0]), fovy=55)
    wb.add_camera(name="front", pos=[0.0, -0.75, 0.35], quat=_lookat_quat([0.0, -0.75, 0.35], [0, 0, 0.05]), fovy=45)
    return spec


def compile_scene(params: SceneParams):
    spec = build(params)
    model = spec.compile()
    data = mujoco.MjData(model)
    return spec, model, data
