"""Episode environment for the full dinner-table layout (scene_table), shared by recording and evaluation.

Same control/observation contract as env.py (12-D joint targets at FPS, camera images + 12 joint
positions). Oracle runs are split into skill segments, each with its own instruction, so the same
recording can train one policy per skill or one policy for the whole workflow.
"""
import mujoco
import numpy as np

from . import scene_table
from .control import GRIP_OPEN, Bimanual
from .env import CAMERAS, FPS, JOINTS
from .oracle import table
from .oracle.handoff import HOME

IMAGE_HW = (144, 192)  # smaller than the hand-off scene: full-table demos are ~4x longer

# Skill segments in execution order: (skill, first phase mark of the segment, instruction).
SKILLS = [
    ("drawer", "open_drawer", "Open the cutlery drawer."),
    ("spoon", "spoon_pick", "Hand the spoon from arm A to arm B and place it right of the placemat."),
    ("plate", "plate_pick", "Move the plate onto the placemat."),
    ("fork", "fork_pick", "Hand the fork from arm A to arm B and place it left of the placemat."),
    ("cup", "cup_pick", "Set the cup at the top right of the place setting."),
]
FULL_TASK = "Set the dinner table: open the drawer, place the spoon, the plate, the fork and the cup."


class TableEpisode:
    def __init__(self, seed: int, render: bool = True, on_step=None, image_hw=IMAGE_HW, params=None):
        self.params = params if params is not None else scene_table.sample(seed)  # params: a table set by hand
        _, self.m, self.d = scene_table.compile_scene(self.params)
        self.substeps = int(round(1.0 / FPS / self.m.opt.timestep))
        self.qadr = np.array([self.m.joint(j).qposadr[0] for j in JOINTS])
        self.act = np.array([self.m.actuator(j).id for j in JOINTS])
        self.renderer = mujoco.Renderer(self.m, *image_hw) if render else None
        self.ctl = Bimanual(self.m, self.d, on_step=on_step)
        for p in ("a_", "b_"):
            self.ctl.set_now(p, HOME, GRIP_OPEN)
        mujoco.mj_forward(self.m, self.d)
        self.ctl.hold(0.5)

    def state(self):
        return self.d.qpos[self.qadr].astype(np.float32)

    def command(self):
        return self.d.ctrl[self.act].astype(np.float32)

    def observation(self):
        obs = {"state": self.state()}
        if self.renderer is not None:
            obs["images"] = {}
            for cam in CAMERAS:
                self.renderer.update_scene(self.d, camera=cam)
                obs["images"][cam] = self.renderer.render().copy()
        return obs

    def step(self, action):
        self.d.ctrl[self.act] = action
        for _ in range(self.substeps):
            self.ctl.step()
        return self.observation()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()


def record_table_oracle(seed: int, image_hw=IMAGE_HW):
    """Run the full-table oracle, sampling (observation, action) at FPS.

    Returns (frames, segments, result): segments is [(skill, instruction, first_frame, end_frame)] and
    result is the grade_table verdict. As in env.record_oracle, the action label for frame t is the
    joint target commanded at frame t+1.
    """
    from .control import IKFailure
    from .grader_table import grade_table

    frames, cmds = [], []
    ep = None

    def on_step(m, d):
        if ep is not None and ep.ctl.steps % ep.substeps == 0:
            frames.append(ep.observation())
            cmds.append(ep.command())

    ep = TableEpisode(seed, render=True, on_step=on_step, image_hw=image_hw)
    ep.ctl.steps = 0
    phases, error = [], None
    try:
        table.run(ep.ctl, ep.params, phases)
    except IKFailure as e:
        error = str(e)
    result = grade_table(ep.m, ep.d, ep.params)
    result["error"] = error
    ep.close()
    actions = cmds[1:] + cmds[-1:]
    for f, a in zip(frames, actions):
        f["action"] = a
    starts = {name: step // ep.substeps for name, step in phases}
    bounds = [starts.get(mark) for _, mark, _ in SKILLS] + [len(frames)]
    segments = []
    for i, (skill, _, text) in enumerate(SKILLS):
        if bounds[i] is None or bounds[i + 1] is None:
            continue
        segments.append((skill, text, bounds[i], min(bounds[i + 1], len(frames))))
    return frames, segments, result


def displace(m, d, body: str, dx: float, dy: float):
    """Move a free body by (dx, dy) m at once — a knock, for training data and probes — with its velocity zeroed.
    (At run time a push is a Slide in tenplaces.agent: a velocity over a quarter second.)"""
    import mujoco

    jnt = m.body_jntadr[m.body(body).id]
    adr, dof = m.jnt_qposadr[jnt], m.jnt_dofadr[jnt]
    d.qpos[adr:adr + 2] += (dx, dy)
    d.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(m, d)


def record_skill_oracle(seed: int, skill: str, before=(), image_hw=IMAGE_HW, policy=None, policy_frames: int = 0,
                        drawer_open: float | None = None, displace_body: str | None = None, displace_xy=(0.0, 0.0),
                        cup_scale: float | None = None):
    """Scripted run of the skills in `before` (not recorded), then `skill` alone, recorded at FPS.

    Both arms are at home between skills (oracle.table.run_plan), so the recording starts from the same pose
    as a policy started by the sequencer. Returns (frames, result); action labels as in record_table_oracle.

    Takeover (HG-DAgger-style): with `policy`, the learned policy first runs the skill for `policy_frames`
    control steps (not recorded), then the scripted controller takes over from wherever the policy left the arms
    — its moves start from the current joint targets and its grasps from the object's current pose — and only
    that continuation is recorded: demonstrations of recovering from the policy's own approach errors. If the
    policy has already disturbed the object beyond what the script handles, the grade fails and the caller
    drops the episode.

    displace_body: after the prefix (which placed it), knock that object by `displace_xy` m and let it settle,
    unrecorded — the recorded skill then puts a displaced object back (disturbance-repair demonstrations).
    cup_scale: this table with the cup that size (radius and height; scene_table.TableParams.cup_scale).
    """
    from .control import IKFailure
    from .grader_table import grade_table

    frames, cmds, recording = [], [], [False]
    ep = None

    def on_step(m, d):
        if recording[0] and ep.ctl.steps % ep.substeps == 0:
            frames.append(ep.observation())
            cmds.append(ep.command())

    params = None
    if cup_scale is not None:
        params = scene_table.sample(seed)
        params.cup_scale = float(cup_scale)
    ep = TableEpisode(seed, render=True, on_step=on_step, image_hw=image_hw, params=params)
    if drawer_open is not None:  # the scripted pull reads it at run time (oracle.table.open_drawer)
        ep.params.drawer_open = drawer_open
    error = None
    try:
        if before:
            table.run_plan(ep.ctl, ep.params, list(before))
        if displace_body is not None:
            displace(ep.m, ep.d, displace_body, *displace_xy)
            ep.ctl.hold(0.5)  # settle; not recorded
        if policy is not None and policy_frames > 0:
            onehot = np.zeros(len(SKILLS), dtype=np.float32)
            onehot[[s for s, _, _ in SKILLS].index(skill)] = 1.0
            text = dict((s, t) for s, _, t in SKILLS)[skill]
            policy.reset()
            obs = ep.observation()
            for _ in range(policy_frames):
                obs["task"], obs["env_state"], obs["skill"] = text, onehot, skill
                obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
        ep.ctl.steps = 0
        recording[0] = True
        table.run_plan(ep.ctl, ep.params, [skill])
    except IKFailure as e:
        error = str(e)
    result = grade_table(ep.m, ep.d, ep.params)
    result["error"] = error
    ep.close()
    actions = cmds[1:] + cmds[-1:]
    for f, a in zip(frames, actions):
        f["action"] = a
    return frames, result


RELEASED = 0.35  # the gripper command every demonstrated skill starts from once that arm has let go (oracle runs)


def go_home(ep: TableEpisode, frames: int = 20, on_frame=None, release_frames: int = 8, until: int = 60,
            tol: float = 0.05):
    """Between chained skills: put both arms in the state every demonstration starts a skill from — first open
    each gripper to at least RELEASED for `release_frames` (lets go of anything still held, e.g. the drawer handle
    the camera judged "done" too early; an open gripper stays open), then a smooth joint-space interpolation of
    the joint targets to the home pose over `frames` control steps. Then, if a joint is still more than `tol` rad
    from home, keep commanding home for up to `until` frames, opening both grippers fully half-way — an arm hung up
    on the drawer handle (tuning seed 104: shoulder pan 0.78 rad off, the next skill started from there). Returns
    at once when the arms are already home. Proprioception only — no object state. Returns the last observation."""
    obs = None
    cmd = ep.command().astype(np.float64)
    cmd[5], cmd[11] = max(cmd[5], RELEASED), max(cmd[11], RELEASED)
    for _ in range(release_frames):
        obs = ep.step(cmd)
        if on_frame is not None:
            on_frame(ep, obs)
    start = cmd.copy()
    goal = start.copy()
    goal[0:5], goal[6:11] = HOME, HOME
    for i in range(1, frames + 1):
        s = 0.5 - 0.5 * np.cos(np.pi * i / frames)
        obs = ep.step(start + (goal - start) * s)
        if on_frame is not None:
            on_frame(ep, obs)
    for i in range(until):
        q = ep.state()
        if max(np.abs(q[0:5] - HOME).max(), np.abs(q[6:11] - HOME).max()) < tol:
            break
        if i == until // 2:  # not converging: whatever holds the arm, let go of it
            goal[5], goal[11] = GRIP_OPEN, GRIP_OPEN
        obs = ep.step(goal)
        if on_frame is not None:
            on_frame(ep, obs)
    return obs
