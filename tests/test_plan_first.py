"""Plan first, then ask what no skill does while the arms already move: the arms do not wait for cannot_do, its
answer is announced as its own event, the planner never answers two calls at once, and an empty plan still waits
for it (plan_checked re-plans without the impossible part). No model is loaded."""
import threading
import time

from test_agent_voice import ScriptedPlanner, StillPolicy, instant_check
from test_intent import IMAGE, ScriptedVLM

from tenplaces.agent import run_command


class CheckingPlanner(ScriptedPlanner):
    """plan() as scripted; cannot_do() answers `unsupported`. Records whether two planner calls ever overlapped."""

    def __init__(self, steps, unsupported=("light a candle",), wait_for=None):
        super().__init__(steps)
        self.unsupported, self.wait_for = list(unsupported), wait_for
        self.lock, self.active, self.overlap, self.saw_motion = threading.Lock(), 0, False, None

    def _call(self, fn):
        with self.lock:
            self.active += 1
            self.overlap |= self.active > 1
        try:
            return fn()
        finally:
            with self.lock:
                self.active -= 1

    def plan(self, command, image, done=()):
        return self._call(lambda: ScriptedPlanner.plan(self, command, image, done))

    def cannot_do(self, command, image=None):
        def answer():
            if self.wait_for is not None:  # only answers once the arms have moved: proves they did not wait
                self.saw_motion = self.wait_for.wait(timeout=10.0)
            time.sleep(0.05)
            return list(self.unsupported), 50.0
        return self._call(answer)

    def plan_checked(self, command, image, done=()):
        p = self.plan(command, image, done)
        items, ms = self.cannot_do(command)
        p.update(unsupported=items)
        return p


class MovingPolicy(StillPolicy):
    def __init__(self, moved):
        self.moved = moved

    def select_action(self, obs):
        self.moved.set()
        return super().select_action(obs)


def kinds(events):
    return [e["kind"] for e in events]


def test_arms_start_before_the_unsupported_answer():
    moved = threading.Event()
    planner = CheckingPlanner(["cup"], wait_for=moved)
    events, _ = run_command(MovingPolicy(moved), planner, "Only the cup, and light a candle.", seed=0,
                            checker=instant_check, log=lambda *_: None)
    assert planner.saw_motion is True  # cannot_do was still pending while the policy already ran
    unsupported = [e for e in events if e["kind"] == "unsupported"]
    assert len(unsupported) == 1 and unsupported[0]["items"] == ["light a candle"]
    assert kinds(events).index("plan") < kinds(events).index("unsupported") < kinds(events).index("finished")
    assert not planner.overlap


def test_empty_plan_waits_for_the_check():
    planner = CheckingPlanner([])
    events, _ = run_command(StillPolicy(), planner, "Light a candle.", seed=0, checker=instant_check,
                            log=lambda *_: None)
    plan = next(e for e in events if e["kind"] == "plan")
    assert plan["unsupported"] == ["light a candle"] and "unsupported" not in kinds(events)
    assert not planner.overlap


def test_only_the_first_plan_uses_the_idle_pipeline():
    class IdlePlanner(CheckingPlanner):
        idle_pipe = object()  # VLMPlanner(idle_config=...) has one

        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.calls = []

        def plan(self, command, image, done=(), idle=False):
            self.calls.append(("plan", idle))
            return super().plan(command, image, done)

        def cannot_do(self, command, image=None, idle=False):
            self.calls.append(("cannot_do", idle))
            return super().cannot_do(command, image)

    planner = IdlePlanner(["cup"])
    run_command(StillPolicy(), planner, "Only the cup, and light a candle.", seed=0, checker=instant_check,
                log=lambda *_: None)
    assert planner.calls == [("plan", True), ("cannot_do", False)]  # cannot_do runs while the arms move


def test_cannot_do_sends_no_image():
    class RecordingVLM(ScriptedVLM):
        def _ask(self, prompt, image, schema, max_new_tokens=120, idle=False):
            self.images = getattr(self, "images", []) + [image]
            return super()._ask(prompt, image, schema, max_new_tokens, idle)

    vlm = RecordingVLM([{"unsupported": ["light a candle"]}])
    assert vlm.cannot_do("Set the table and light a candle.", IMAGE)[0] == ["light a candle"]
    assert vlm.images == [None]
