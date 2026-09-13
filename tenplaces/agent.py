"""The full loop: a natural-language command -> VLM plan (verified) -> one policy per skill, with a visual
completion check after each skill, one retry, then re-planning from what the camera shows.

With a voice source (tenplaces.listen), the person can keep talking while the robot works: "stop" ends the
run at once; anything else goes to the planner's amend() on a worker thread, so the arms keep moving while
the VLM thinks (seconds on the CPU). The new plan takes over when the current step ends (a step is never
abandoned mid-air); if the step ends first, the arms hold still until the answer arrives. A sentence heard
while an amendment is pending is amended on top of it. After the last step it keeps listening for `linger_s`
seconds, so "oh, and the cup too" still works.

Things also change without being asked: before each new step and once at the end, every finished step is
re-checked with the camera classifier (8 ms each); one that no longer holds — the plate knocked off the mat —
goes back to the front of the queue (at most `max_repairs` times per step). `disturb` slides objects by a
given distance mid-run, to show and test exactly that.

Every decision (plan, verifier corrections, speech heard, checks, retries, repairs) is logged so the demo
video can show the plan panel next to the robot.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import mujoco
import numpy as np

from .env import FPS
from .env_table import SKILLS, TableEpisode, go_home
from .evaluate_table import CAMERA_ENDS, DEFAULT_BUDGETS, RETRY, SETTLE
from .grader_table import grade_table
from .listen import STOP
from .planner import verify

SKILL_INDEX = {s: i for i, (s, _, _) in enumerate(SKILLS)}
SKILL_NAMES = [s for s, _, _ in SKILLS]
SKILL_TEXT = {s: t for s, _, t in SKILLS}
PLANNER_HW = (336, 448)  # the planner sees a larger top-camera image than the policy
CONFIRM_FRAMES = 10  # a finished step that looks undone is looked at again 0.4 s later before it is redone


def parse_push(text: str):
    """'T:BODY:DX:DY[:DUR]' -> a `disturb` entry: slide BODY by (DX, DY) m over DUR s (default 0.25) at simulated
    time T, or 1 s after a step is confirmed when T is 'after-<skill>' (e.g. 'after-plate:plate:0:-0.07')."""
    t, body, dx, dy, *dur = text.split(":")
    when = t if t.startswith("after-") else float(t)
    return when, body, (float(dx), float(dy)), float(dur[0]) if dur else 0.25


def planner_image(ep: TableEpisode, renderer) -> np.ndarray:
    renderer.update_scene(ep.d, camera="top")
    return renderer.render().copy()


def run_skill(ep: TableEpisode, policy, skill: str, obs: dict, budget: int, on_frame=None, checker=None,
              check_every=10, min_frames=40, settle_frames=30, interrupt=None, early_end: bool = True):
    """Run one skill's policy until the camera checker says done (then settle) or the budget runs out.
    early_end=False: no mid-skill checks — the policy runs the whole budget and the camera checks once at the end
    (evaluate_table.CAMERA_ENDS).

    Returns (obs, done, check_ms); done/check_ms are None when no checker is given or interrupt() fired.
    """
    policy.reset()
    onehot = np.zeros(len(SKILLS), dtype=np.float32)
    onehot[SKILL_INDEX[skill]] = 1.0

    def step(obs):
        obs["task"], obs["env_state"], obs["skill"] = SKILL_TEXT[skill], onehot, skill
        obs = ep.step(np.asarray(policy.select_action(obs), dtype=np.float64))
        if on_frame is not None:
            on_frame(ep, obs)
        return obs

    ok, ms = None, None
    for i in range(budget):
        obs = step(obs)
        if interrupt is not None and interrupt():
            return obs, None, None
        if checker is not None and early_end and i >= min_frames and i % check_every == 0:
            ok, ms = checker(skill, obs["images"]["top"])
            if ok:
                # "Done" is seen as soon as the object is in place, often while it is still held: let the
                # policy finish the release and retreat that ends every demo.
                for _ in range(settle_frames):
                    obs = step(obs)
                return obs, True, ms
    if checker is not None:
        ok, ms = checker(skill, obs["images"]["top"])
    return obs, ok, ms


def default_checker(planner, classifier_xml="models/state_classifier_v3/state_classifier.xml", ov_config=None):
    """Completion checks: the camera classifier when it has been trained, else the VLM's yes/no answer.
    ov_config: OpenVINO properties for the classifier (tenplaces.cores.control_config())."""
    from pathlib import Path

    if Path(classifier_xml).exists():
        from .state_classifier import OVStateClassifier

        clf = OVStateClassifier(classifier_xml, ov_config=ov_config)

        def check(skill, top_small):
            t = time.perf_counter()
            done = clf.is_done(skill, top_small)
            return done, 1000 * (time.perf_counter() - t)

        return check, "classifier"
    return None, "vlm"


class Amender:
    """planner.amend() on one worker thread. Jobs run one at a time; a job submitted without `remaining`
    amends the result of the job before it, so two quick sentences both count."""

    def __init__(self, planner, command: str):
        self.planner, self.command = planner, command
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="amend")
        self.last = None

    def submit(self, heard, image, done, current, remaining=None):
        prev = self.last

        def work():
            rem = prev.result()["steps"] if remaining is None else remaining
            return self.planner.amend(self.command, heard, image, done, current, rem)

        self.last = self.pool.submit(work)
        return self.last

    def close(self):
        """Drop jobs not started; wait for the running one, so the planner is free for the next command."""
        self.pool.shutdown(wait=True, cancel_futures=True)


class Slide:
    """Slide a free body by (dx, dy) metres over `duration` seconds, starting at simulated time t0: its
    horizontal velocity is driven for the window, then stopped. Distance-based rather than force-based,
    because friction and mass are randomised per seed (a 5 N shove launched the plate off the table on one
    seed and moved it 7 cm on another). It still collides with whatever is in the way."""

    def __init__(self, ep, t0: float, body: str, delta, duration: float = 0.25):
        self.t0, self.body, self.duration = t0, body, duration
        self.delta = np.asarray(delta, float)
        self.dof = ep.m.jnt_dofadr[ep.m.body_jntadr[ep.m.body(body).id]]
        self.state = "waiting"

    def update(self, ep) -> bool:
        """Apply for the current frame; True on the frame the slide starts."""
        t = ep.d.time
        if self.state == "waiting" and t >= self.t0:
            self.state = "sliding"
            ep.d.qvel[self.dof:self.dof + 2] = self.delta / self.duration
            return True
        if self.state == "sliding":
            if t < self.t0 + self.duration:
                ep.d.qvel[self.dof:self.dof + 2] = self.delta / self.duration
            else:
                ep.d.qvel[self.dof:self.dof + 3] = 0.0
                self.state = "done"
        return False


def run_command(policy, planner, command: str, seed: int, budgets=None, max_attempts: int = 2,
                max_replans: int = 1, on_frame=None, log=print, checker=None, on_event=None, voice=None,
                linger_s: float = 4.0, recheck: bool = True, max_repairs: int = 1, disturb=(),
                async_amend: bool = True, home_frames: int = 20):
    """Execute a command on one seeded scene. Returns (events, grade).

    checker(skill, policy_top_image) -> (done, ms); defaults to the camera classifier if trained, else the VLM.
    on_frame(ep, obs) runs after every control step; on_event(event_dict) after every decision.
    voice: a tenplaces.listen source polled every control step (None: no speech during the run).
    disturb: [(sim_time_s, body, (dx, dy) metres, duration_s)] slides of free bodies (plate, cup, spoon, fork).
    async_amend: amend on a worker thread while the arms keep moving (False: the simulation waits for amend()).
    """
    budgets = budgets or DEFAULT_BUDGETS
    if checker is None:
        checker, _ = default_checker(planner)
    ep = TableEpisode(seed, render=True)
    prend = mujoco.Renderer(ep.m, *PLANNER_HW)
    obs = ep.observation()
    events, done, repairs = [], [], {}
    said = {"stop": False, "queue": None, "current": None, "hold_t0": None}
    latest = {"obs": obs}
    amender = Amender(planner, command) if voice is not None and async_amend else None
    pending = []  # amendments submitted and not yet applied, oldest first
    # A slide's time is simulated seconds or "after-<skill>": 1 s after that step is confirmed, so a demo push
    # lands on the placed plate (an absolute time hit the plate before it was picked, in the first full run).
    slides = [Slide(ep, t0, body, delta, dur) for t0, body, delta, dur in disturb if not isinstance(t0, str)]
    deferred = [(t0[len("after-"):], body, delta, dur) for t0, body, delta, dur in disturb if isinstance(t0, str)]
    started = []  # slides that began since the last control step (announced by frame())
    if disturb:  # every physics step: friction undoes a velocity set only once per 40 ms control step
        ep.ctl.on_step = lambda m, d: started.extend(s for s in slides if s.update(ep))

    def event(kind, **kw):
        kw.update(kind=kind, t=round(ep.d.time, 2))
        events.append(kw)
        if on_event is not None:
            on_event(kw)
        log(f"[{kw['t']:6.2f}s] {kind}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if k not in ("kind", "t")))

    def listen():
        if voice is None or said["stop"]:
            return
        for text in voice.poll(ep.d.time):
            event("heard", text=text)
            if STOP.search(text):
                said["stop"] = True
                event("stop", heard=text)
                if pending:  # the arms stop now; a change to what comes next no longer applies
                    for j in pending:
                        j["future"].cancel()
                    event("discarded", heard=[j["heard"] for j in pending])
                    pending.clear()
                return
            remaining = said["queue"] if said["queue"] is not None else queue
            if amender is None:
                announce_unsupported(wait=True)  # the VLM answers one call at a time
                a = planner.amend(command, text, planner_image(ep, prend), done, said["current"], remaining)
                event("amend", heard=text, proposed=a["proposed"], steps=a["steps"], corrections=a["corrections"],
                      after=said["current"], ms=round(a["ms"]))
                said["queue"] = a["steps"]
                continue
            # The renderer is not thread-safe: the image is taken here; only the VLM call runs on the worker.
            future = amender.submit(text, planner_image(ep, prend), list(done), said["current"],
                                    None if pending else list(remaining))
            pending.append({"heard": text, "current": said["current"], "future": future, "thinking": False})

    def collect():
        """Announce the amendment the worker is on; take finished ones, oldest first."""
        while pending:
            j, f = pending[0], pending[0]["future"]
            finished = f.done()
            if not j["thinking"] and (finished or f.running()):
                j["thinking"] = True
                event("thinking", heard=j["heard"])
            if not finished:
                return
            pending.pop(0)
            a = f.result()
            waited = ep.d.time - said["hold_t0"] if said["hold_t0"] is not None else 0.0
            event("amend", heard=j["heard"], proposed=a["proposed"], steps=a["steps"], corrections=a["corrections"],
                  after=j["current"], ms=round(a["ms"]), waited_s=round(waited, 2))
            said["queue"] = a["steps"]

    later = {"unsupported": None, "own_pool": None}  # the cannot_do answer still on its way (see the plan below)

    def announce_unsupported(wait: bool):
        """Say what no skill does once the background cannot_do answers; wait=True before any other planner call on
        this thread (the VLM answers one call at a time) and at the end, so the answer is never lost."""
        job = later["unsupported"]
        if job is None or not (wait or job.done()):
            return
        later["unsupported"] = None
        try:
            items, ms = job.result()
        except Exception as e:  # the plan stands; only the apology is lost
            log(f"cannot_do failed: {e}")
            return
        event("unsupported", items=items, ms=round(ms))

    def frame(ep_, obs_):
        latest["obs"] = obs_
        while started:
            s = started.pop(0)
            event("pushed", body=s.body, delta_m=s.delta.round(3).tolist(), duration_s=s.duration)
        if on_frame is not None:
            on_frame(ep_, obs_)
        listen()
        collect()
        announce_unsupported(wait=False)

    def hold_until_amended(obs):
        """At a step boundary: keep the arms where they are, still listening, until pending amendments arrive.
        Paced at the control rate, as a real controller would hold, and so the wait shows at its true length."""
        if not pending:
            return obs
        hold, said["hold_t0"], tick = ep.command(), ep.d.time, time.perf_counter()
        while pending and not said["stop"]:
            obs = ep.step(hold)
            frame(ep, obs)
            tick += 1.0 / FPS
            time.sleep(max(0.0, tick - time.perf_counter()))
        said["hold_t0"] = None
        return obs

    def sweep():
        """Re-check finished steps; the ones that no longer hold are redone first (canonical order). A "not done"
        must survive a second look CONFIRM_FRAMES later with the arms held still: one bad frame (an arm retreating
        over the spoon) is not a knocked-over object — the first smoke run redid a spoon that was in place."""
        if not recheck or checker is None:
            return []
        broken = []
        for s in list(done):
            ok, ms = checker(s, latest["obs"]["images"]["top"])
            if not ok:
                hold = ep.command()
                for _ in range(CONFIRM_FRAMES):
                    frame(ep, ep.step(hold))
                ok, ms = checker(s, latest["obs"]["images"]["top"])
            if not ok:
                repairs[s] = repairs.get(s, 0) + 1
                done.remove(s)
                if repairs[s] > max_repairs:
                    event("gave_up", skill=s, repairs=repairs[s] - 1)
                    continue
                event("regressed", skill=s, ms=round(ms, 1))
                broken.append(s)
        return sorted(broken, key=SKILL_NAMES.index)

    # The planner answers one call at a time: plan/replan/is_done below only run with no amendment pending.
    # plan() first, so the arms start as soon as there is a plan. What the command asks for that no skill does
    # ("light a candle": cannot_do, text only — it never needs the image) is asked meanwhile on the planner's one
    # worker thread and said when the answer arrives. Only an empty plan waits for it: plan_checked then re-plans
    # without the impossible part ("set the table and light a candle"). Asking both in a row before moving took
    # 18.7 s median to the first motion (out/planner/eval_planner_checked.json).
    # The arms are still until this plan arrives: a planner with an all-core pipeline (VLMPlanner idle_config)
    # uses it here, and its core-split pipeline for everything asked while the arms move.
    image = planner_image(ep, prend)
    idle = {"idle": True} if getattr(planner, "idle_pipe", None) is not None else {}
    plan = planner.plan(command, image, done, **idle)
    if hasattr(planner, "cannot_do"):
        if not plan["steps"] and hasattr(planner, "plan_checked"):
            plan = planner.plan_checked(command, image, done, **idle)
        else:
            pool = amender.pool if amender is not None else ThreadPoolExecutor(max_workers=1, thread_name_prefix="cannot")
            later["own_pool"] = None if amender is not None else pool
            later["unsupported"] = pool.submit(planner.cannot_do, command)
    event("plan", command=command, proposed=plan["proposed"], steps=plan["steps"], corrections=plan["corrections"],
          unsupported=plan.get("unsupported", []), ms=round(plan["ms"]))
    queue, replans = list(plan["steps"]), 0
    wanted = list(plan["steps"])  # what the person currently wants: the plan, then every spoken change
    started_any = False  # the first skill starts from the episode's initial pose; later ones after go_home
    while not said["stop"]:
        broken = sweep()
        queue = broken + [s for s in queue if s not in broken]
        if not queue:  # keep listening a little after the last step: "oh, and the cup too"
            hold = ep.command()
            for _ in range(int(linger_s * 25) if voice is not None else 0):
                obs = ep.step(hold)
                frame(ep, obs)
                if said["queue"] or said["stop"]:
                    break
            obs = hold_until_amended(obs)
            if said["queue"]:
                queue, said["queue"] = [s for s in said["queue"] if s not in done], None
                wanted = done + queue
                continue
            if sweep_again := sweep():  # a push during the linger
                queue = sweep_again
                continue
            break
        skill = queue.pop(0)
        said["current"] = skill
        ok = False
        for attempt in range(1, (max_attempts if RETRY[skill] else 1) + 1):  # drawer/spoon retries never recovered
            if home_frames and started_any:
                # Every demonstration starts a skill with the grippers released and both arms home; the camera can
                # call a step done while an arm still holds the drawer handle. Tuning seeds 100-119: 4/20 -> 10/20
                # full tables (scripts/eval_table_home.py).
                obs = go_home(ep, home_frames, on_frame=frame)
            started_any = True
            event("skill_start", skill=skill, attempt=attempt)
            obs, ok, ms = run_skill(ep, policy, skill, obs, budgets[skill], frame, checker,
                                    interrupt=lambda: said["stop"], early_end=CAMERA_ENDS[skill],
                                    settle_frames=SETTLE[skill])
            if said["stop"]:
                break
            if checker is None:
                obs = hold_until_amended(obs)
                if said["stop"]:
                    break
                announce_unsupported(wait=True)  # the VLM answers one call at a time
                ok, ms = planner.is_done(skill, planner_image(ep, prend))
            event("check", skill=skill, done=ok, ms=round(ms))
            if ok:
                done.append(skill)
                for x in [x for x in deferred if x[0] == skill]:  # pushes tied to this step: 1 s from now
                    deferred.remove(x)
                    slides.append(Slide(ep, ep.d.time + 1.0, *x[1:]))
                break
        said["current"] = None
        if said["stop"]:
            break
        obs = hold_until_amended(obs)
        if said["stop"]:
            break
        if said["queue"] is not None:  # a spoken change replaces what was left, now that the step has ended
            queue, said["queue"] = [s for s in said["queue"] if s not in done], None
            wanted = done + ([skill] if not ok else []) + [s for s in queue if s != skill]  # a failed step stays wanted
            if ok:
                continue
        if not ok and replans < max_replans:
            # What is wanted is known (the plan plus any spoken change); only the execution failed. Retry the rest
            # from the current state — asking the planner again would re-read the original command and bring back
            # a spoken "skip the fork" (it did, in the first full run).
            replans += 1
            steps, notes = verify([s for s in wanted if s not in done], done)
            event("replan", reason=f"{skill} not confirmed", steps=steps, corrections=notes, ms=0)
            queue = steps
    announce_unsupported(wait=True)  # a short plan can end before the answer: still say it
    grade = grade_table(ep.m, ep.d, ep.params)
    event("finished", done=done, stopped=said["stop"],
          graded=[s for s in ("drawer_open", "spoon", "plate", "fork", "cup") if grade[s]])
    if amender is not None:
        amender.close()
    if later["own_pool"] is not None:
        later["own_pool"].shutdown(wait=True)
    prend.close()
    ep.close()
    return events, grade
