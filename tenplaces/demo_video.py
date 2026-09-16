"""Demo video: the robot (front camera), the policy's cameras, and a live plan panel.

    rec = DemoRecorder("out/video/demo.mp4", voice=voice)
    run_command(policy, planner, command, seed, on_frame=rec.on_frame, on_event=rec.on_event, voice=voice)
    rec.close()

The panel shows everything needed to follow the reasoning: the command, the VLM's proposed plan, the
verifier's corrections, the current skill, each completion check, what was said while the robot worked
(with the words appearing as they are spoken), disturbances and repairs, and the OpenVINO timings.
"""
import imageio.v2 as iio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

MAIN_HW = (540, 960)
PANEL_W = 480
SMALL_HW = (180, 240)


def _font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


class DemoRecorder:
    def __init__(self, path: str, fps: int = 25, title: str = "Ten Places · two SO-101 arms · OpenVINO on Intel CPU",
                 hold_s: float = 2.0, voice=None):
        self.writer = iio.get_writer(path, fps=fps, macro_block_size=8)
        self.title = title
        self.renderer = None
        self.voice = voice
        self.fps, self.frames, self.t_start = fps, 0, 0.0
        self._last, self.hold_frames = None, int(hold_s * fps)
        self.state = {"command": "", "proposed": [], "steps": [], "corrections": [], "current": None,
                      "checks": [], "planner_ms": None, "heard": [], "notes": [], "unsupported": [], "said": "",
                      "replanned_from": None, "seen": []}
        self.f_big, self.f, self.f_small = _font(26), _font(20), _font(16)

    @property
    def duration_s(self) -> float:
        """Length of the video so far; one frame per 40 ms control step, so this is simulated time. The first
        frame shows sim time t_start + 1/fps (the scene starts after a settling period, not at 0)."""
        return self.frames / self.fps

    def _done(self):
        last = {}
        for k, ok in self.state["checks"]:  # a step is done if its latest check says so (re-checks included)
            last[k] = ok
        return {k for k, ok in last.items() if ok}

    def on_event(self, e):
        s = self.state
        if e.get("said"):  # added by tenplaces.speak.SpeakEvents when the robot says something
            s["said"] = e["said"]
        if e["kind"] in ("plan", "replan"):
            s["command"] = e.get("command", s["command"])
            s["proposed"] = e.get("proposed", s["proposed"])
            s["steps"], s["corrections"], s["planner_ms"] = e["steps"], e["corrections"], e.get("ms")
            s["unsupported"] = e.get("unsupported", s["unsupported"])
            s["replanned_from"] = e.get("replanned_from") if e["kind"] == "plan" else s["replanned_from"]
        elif e["kind"] == "unsupported":  # the check that runs while the arms already move
            s["unsupported"] = e.get("items", [])
        elif e["kind"] == "seen_done":  # the first look: already done before the robot started
            s["seen"] = e.get("steps", [])
        elif e["kind"] == "amend":
            done = self._done()
            kept = [x for x in s["steps"] if x in done or x == e.get("after")]
            s["steps"] = kept + [x for x in e["steps"] if x not in kept]
            s["proposed"], s["corrections"], s["planner_ms"] = e["proposed"], e["corrections"], e.get("ms")
            if e.get("waited_s"):
                s["notes"].append(f"arms held {e['waited_s']:.1f} s for the new plan")
        elif e["kind"] == "heard":
            s["heard"].append(e["text"])
        elif e["kind"] == "thinking":
            s["notes"].append(f'planner thinking about "{e.get("heard", "")}" — the robot keeps working')
        elif e["kind"] == "discarded":
            s["notes"].append("pending change dropped (stop)")
        elif e["kind"] == "pushed":
            s["notes"].append(f"disturbance: {e['body']} pushed")
        elif e["kind"] == "regressed":
            s["checks"].append((e["skill"], False))
            s["notes"].append(f"camera: {e['skill']} no longer in place → redo")
        elif e["kind"] == "gave_up":
            s["notes"].append(f"{e['skill']}: stopped after {e['repairs']} repair attempt(s)")
        elif e["kind"] == "stop":
            s["current"] = "stopped by voice"
        elif e["kind"] == "skill_start":
            s["current"] = f"{e['skill']} (attempt {e['attempt']})"
        elif e["kind"] == "check":
            s["checks"].append((e["skill"], e["done"]))
        elif e["kind"] == "finished":
            s["current"] = s["current"] if s["current"] == "stopped by voice" else "finished"
            # Hold the end state with the final panel, long enough for the last spoken line (the TTS voice
            # speaks 13-16 characters per second, measured).
            hold = max(self.hold_frames, int((0.5 + len(e.get("said", "")) / 13) * self.fps))
            if self.renderer is not None and self._last is not None:
                for _ in range(hold):
                    self.on_frame(*self._last)

    def on_frame(self, ep, obs):
        self._last = (ep, obs)
        if self.frames == 0:
            self.t_start = ep.d.time - 1.0 / self.fps
        if self.renderer is None:
            self.renderer = mujoco.Renderer(ep.m, *MAIN_HW)
        self.renderer.update_scene(ep.d, camera="front")
        main = self.renderer.render()
        small = [np.asarray(Image.fromarray(obs["images"][c]).resize(SMALL_HW[::-1])) for c in ("top", "a_wrist", "b_wrist")]
        strip = np.concatenate(small, axis=1)
        h = MAIN_HW[0] + SMALL_HW[0]
        canvas = Image.new("RGB", (MAIN_HW[1] + PANEL_W, h), (18, 20, 24))
        canvas.paste(Image.fromarray(main), (0, 0))
        canvas.paste(Image.fromarray(strip), (0, MAIN_HW[0]))
        self._panel(ImageDraw.Draw(canvas), MAIN_HW[1] + 16, ep.d.time)
        self.writer.append_data(np.asarray(canvas))
        self.frames += 1

    def _panel(self, draw, x, t):
        s, y = self.state, 14
        head = (140, 200, 255)
        status_y = MAIN_HW[0] + SMALL_HW[0] - 30
        # What the robot said last sits in a fixed slot above the status line; the sections above stop
        # drawing rather than run into it.
        robot = _wrap(f'Robot: "{s["said"]}"', 46) if s["said"] else []
        if len(robot) > 4:
            robot = robot[:3] + [robot[3] + " …"]
        limit = status_y - 8 - 20 * len(robot)

        def text(y, msg, font, fill, dy):
            if y + dy <= limit:
                draw.text((x, y), msg, font=font, fill=fill)
            return y + dy

        y = text(y, self.title, self.f_small, (170, 170, 180), 30)
        y = text(y, "Command", self.f_small, head, 22)
        for line in _wrap(f'"{s["command"]}"', 38):
            y = text(y, line, self.f, (255, 255, 255), 26)
        if s["seen"]:
            for line in _wrap("camera, first look — already done: " + ", ".join(s["seen"]), 46):
                y = text(y, line, self.f_small, (120, 220, 120), 20)
        y += 8
        y = text(y, f"VLM proposed ({s['planner_ms'] or 0:.0f} ms, OpenVINO INT4)", self.f_small, head, 22)
        y = text(y, " → ".join(s["proposed"]) or "–", self.f, (230, 230, 230), 30)
        y = text(y, "Verified plan", self.f_small, head, 22)
        done = self._done()
        for step in s["steps"]:
            # Plain-text markers: Arial has no check-mark or play glyphs on Windows.
            active = step not in done and bool(s["current"]) and s["current"].startswith(step)
            mark = "[done]" if step in done else ("[now] " if active else "[ ]   ")
            colour = (120, 220, 120) if step in done else (255, 220, 90) if active else (200, 200, 200)
            y = text(y, f"{mark} {step}", self.f, colour, 26)
        if s["unsupported"]:
            for line in _wrap("no skill for: " + ", ".join(s["unsupported"]), 46):
                y = text(y, line, self.f_small, (255, 130, 130), 20)
        if s["replanned_from"]:  # the whole command gave no plan; this one is from the part the skills can do
            for line in _wrap(f'planned again from: "{s["replanned_from"]}"', 46):
                y = text(y, line, self.f_small, (255, 130, 130), 20)
        for c in s["corrections"][:3]:
            for line in _wrap("verifier: " + c, 46):
                y = text(y, line, self.f_small, (255, 170, 120), 20)
        for n in s["notes"][-2:]:
            for line in _wrap(n, 46):
                y = text(y, line, self.f_small, (255, 200, 120), 20)
        y += 8
        live = getattr(self.voice, "partial", "") if self.voice is not None else ""
        if s["heard"] or live:
            # A scripted sentence (run_agent.py --say) is not speech: say so rather than credit Speechmatics.
            scripted = type(self.voice).__name__ == "ScriptedVoice"
            y = text(y, "Said while working (scripted, fixed time)" if scripted else "Heard while working (Speechmatics)",
                     self.f_small, head, 22)
            for said in s["heard"][-2:]:
                for line in _wrap(f'"{said}"', 42):
                    y = text(y, line, self.f_small, (255, 255, 255), 20)
            if live:
                for line in _wrap(f"… {live}", 42):
                    y = text(y, line, self.f_small, (170, 170, 180), 20)
            y += 6
        if s["checks"]:
            y = text(y, "Camera checks (OpenVINO classifier)", self.f_small, head, 22)
            for skill, ok in s["checks"][-4:]:
                y = text(y, f"{skill}: {'done' if ok else 'not done → retry'}", self.f_small,
                         (120, 220, 120) if ok else (255, 120, 120), 20)
        y = limit + 4
        for i, line in enumerate(robot):  # "Robot:" in the heading colour, the words in lilac
            dx = 0
            if i == 0:
                draw.text((x, y), "Robot:", font=self.f_small, fill=head)
                dx, line = draw.textlength("Robot: ", font=self.f_small), line[len("Robot: "):]
            draw.text((x + dx, y), line, font=self.f_small, fill=(215, 185, 255)); y += 20
        status = f"t = {t:5.1f} s" + (f"   ·   {s['current']}" if s["current"] in ("finished", "stopped by voice") else "")
        draw.text((x, status_y), status, font=self.f_small, fill=(170, 170, 180))

    def close(self):
        self.writer.close()
        if self.renderer is not None:
            self.renderer.close()


def _wrap(text, width):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + ([cur] if cur else [])
