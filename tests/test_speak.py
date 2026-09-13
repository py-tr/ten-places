"""The robot's voice without the network: phrases for agent events, a non-blocking speaker with a stub
synthesizer, the simulated-time audio track, muxing into the video, the disk cache, and one agent run with
the stand-ins of test_agent_voice. One real TTS call runs only when SPEECHMATICS_API_KEY is set."""
import io
import re
import subprocess
import time

import numpy as np
import pytest
import soundfile as sf

from tenplaces import speak
from tenplaces.agent import run_command
from tenplaces.listen import STOP, ScriptedVoice
from tenplaces.planner import verify
from tenplaces.speak import EchoGuard, Speaker, SpeakEvents, mux, phrase_for, synthesize

RATE = 16000


def tone(seconds=0.5, amp=1000):
    return np.full(int(seconds * RATE), amp, np.int16), RATE


def slow_synth(text, voice):
    time.sleep(0.2)
    return tone(0.1)


def test_plan_says_the_steps_why_and_what_it_cannot_do():
    e = {"kind": "plan", "t": 0.0, "steps": ["drawer", "spoon"], "unsupported": ["pour water"],
         "corrections": ["added 'drawer' before 'spoon': the spoon is inside the closed drawer"]}
    assert phrase_for(e) == ("I'll open the drawer, then place the spoon. The drawer comes first — the spoon is "
                             "inside. Sorry, I can't pour water.")
    assert phrase_for({"kind": "plan", "steps": [], "corrections": []}) == "Alright — nothing to do."
    assert phrase_for({"kind": "plan", "steps": [], "corrections": [], "unsupported": ["fold the napkin"]}) == \
        "Sorry, I can't fold the napkin."


def test_spoken_changes_are_confirmed_as_differences():
    st = {}
    phrase_for({"kind": "plan", "steps": ["drawer", "spoon", "plate", "fork", "cup"], "corrections": []}, st)
    phrase_for({"kind": "check", "skill": "drawer", "done": True}, st)
    assert phrase_for({"kind": "amend", "after": "spoon", "steps": ["plate", "cup"], "corrections": []}, st) == \
        "OK — skipping the fork."
    assert phrase_for({"kind": "amend", "after": None, "steps": ["plate", "cup", "fork"], "corrections": []}, st) == \
        "OK — adding the fork."
    assert phrase_for({"kind": "amend", "after": "plate", "steps": [], "corrections": []}, st) == \
        "OK — nothing more after this."


def test_routine_events_are_silent_and_the_end_is_honest():
    st = {}
    phrase_for({"kind": "plan", "steps": ["drawer", "plate", "fork"], "corrections": []}, st)
    for e in ({"kind": "skill_start", "skill": "drawer", "attempt": 1}, {"kind": "heard", "text": "hm"},
              {"kind": "check", "skill": "drawer", "done": True}, {"kind": "check", "skill": "plate", "done": True},
              {"kind": "pushed", "body": "plate"}):
        assert phrase_for(e, st) is None
    assert phrase_for({"kind": "skill_start", "skill": "fork", "attempt": 2}, st) == "Trying the fork again."
    assert phrase_for({"kind": "regressed", "skill": "plate"}, st) == "The plate moved — putting it back."
    assert phrase_for({"kind": "finished", "done": ["drawer"], "stopped": False}, st) == \
        "Done — but the plate and fork aren't in place."
    assert phrase_for({"kind": "finished", "done": ["drawer", "plate", "fork"], "stopped": False}, st) == \
        "The table is set."
    assert phrase_for({"kind": "stop"}) == "Stopping."
    assert phrase_for({"kind": "finished", "done": [], "stopped": True}) is None
    assert phrase_for({"kind": "thinking", "heard": "Skip the fork."}) == "One moment."
    assert phrase_for({"kind": "thinking", "heard": "Stop!"}) is None


def test_the_robot_never_says_the_stop_word():
    # The microphone may hear the robot: nothing it says may read as "stop" to tenplaces.listen.
    phrases = [phrase_for(e) for e in (
        {"kind": "stop"}, {"kind": "gave_up", "skill": "cup", "repairs": 1}, {"kind": "regressed", "skill": "drawer"},
        {"kind": "replan", "reason": "fork not confirmed", "steps": ["fork"], "corrections": []})]
    assert all(p and not STOP.search(p) for p in phrases + list(speak.COMMON))


def test_speaker_never_blocks_and_records_sim_time():
    sp = Speaker(play=False, synth=slow_synth)
    t = time.perf_counter()
    for i in range(5):
        sp.say(f"line {i}", sim_time=0.4 * i)
    assert time.perf_counter() - t < 0.05  # five 0.2 s syntheses queued, none waited for
    sp.close()
    assert [x["t"] for x in sp.lines] == pytest.approx([0.0, 0.4, 0.8, 1.2, 1.6]) and sp.latest == "line 4"


def test_a_backlog_drops_stale_lines_from_playback_but_not_from_the_record():
    played = []

    class Recording(Speaker):
        def _play(self, samples, rate, text):
            played.append(text)

    sp = Recording(play=True, synth=slow_synth, max_pending=1)
    for i in range(4):
        sp.say(f"line {i}")
    sp.close()
    assert played[-1] == "line 3" and len(played) <= 2 and len(sp.lines) == 4


def test_track_is_aligned_to_simulated_time(tmp_path):
    sp = Speaker(play=False, synth=lambda text, voice: tone(0.5))
    for t, text in ((1.0, "a"), (1.2, "b"), (1.3, "c"), (9.0, "after the end")):
        sp.say(text, sim_time=t)
    sp.close()
    placed = sp.write_track(tmp_path / "t.wav", duration_s=4.0)
    assert placed == [(1.0, "a"), (1.65, "b"), (2.3, "c")]  # overlaps start after the previous line + 0.15 s
    x, rate = sf.read(str(tmp_path / "t.wav"), dtype="int16")
    onsets = np.flatnonzero(np.diff((x != 0).astype(int)) == 1) + 1
    assert rate == RATE and len(x) == 4 * RATE and list(onsets) == [16000, 26400, 36800]


def test_mux_gives_the_video_an_audio_stream(tmp_path):
    import imageio.v2 as iio
    import imageio_ffmpeg

    video = tmp_path / "v.mp4"
    w = iio.get_writer(video, fps=25, macro_block_size=8)
    for i in range(50):
        w.append_data(np.full((64, 64, 3), 4 * i, np.uint8))
    w.close()
    sf.write(str(tmp_path / "a.wav"), tone(2.0)[0], RATE)
    mux(video, tmp_path / "a.wav", video)  # in place
    info = subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-i", str(video)],
                          capture_output=True, text=True).stderr
    assert re.search(r"Stream #0:\d.*Video: h264", info) and re.search(r"Stream #0:\d.*Audio: aac", info)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["a.wav", "v.mp4"]


def test_a_cached_phrase_needs_no_request(tmp_path, monkeypatch):
    buf, calls = io.BytesIO(), []
    sf.write(buf, tone(0.3)[0], RATE, format="WAV", subtype="PCM_16")
    monkeypatch.setattr(speak, "_fetch", lambda text, voice, **kw: calls.append((voice, text)) or buf.getvalue())
    a = synthesize("Hello.", "sarah", cache_dir=tmp_path)
    b = synthesize("Hello.", "sarah", cache_dir=tmp_path)
    synthesize("Hello.", "theo", cache_dir=tmp_path)
    assert calls == [("sarah", "Hello."), ("theo", "Hello.")]
    assert a[1] == RATE and a[0].dtype == np.int16 and np.array_equal(a[0], b[0])


def test_speak_events_chains_and_records_what_was_said():
    sp, got = Speaker(play=False, synth=lambda text, voice: tone(0.1)), []
    on_event = SpeakEvents(sp, forward=got.append)
    on_event({"kind": "stop", "t": 3.2, "heard": "Stop."})
    on_event({"kind": "check", "t": 3.3, "skill": "cup", "done": True})
    sp.close()
    assert [e["kind"] for e in got] == ["stop", "check"] and got[0]["said"] == "Stopping." and "said" not in got[1]
    assert sp.lines == [{"t": 3.2, "text": "Stopping.", "cut": True}]


def test_the_robot_does_not_obey_its_own_voice():
    sp = Speaker(play=False, synth=lambda text, voice: tone(0.1))
    now = time.perf_counter()
    sp.playing.append([now - 1.0, now - 0.5, "The plate moved — putting it back."])
    guard = EchoGuard(ScriptedVoice([(0.0, "The plate moved putting it back."), (0.0, "Stop!")]), sp)
    assert guard.poll(1.0) == ["Stop!"] and guard.ignored == ["The plate moved putting it back."]
    sp.close()


class StillPolicy:
    def reset(self):
        pass

    def select_action(self, obs):
        return obs["state"].astype(np.float64)


class ScriptedPlanner:
    def __init__(self, steps, amendments):
        self.steps, self.amendments = steps, amendments

    def plan(self, command, image, done=()):
        steps, notes = verify(self.steps, done)
        return {"proposed": list(self.steps), "steps": steps, "corrections": notes, "ms": 0.0}

    def amend(self, command, heard, image, done=(), current=None, remaining=()):
        steps, notes = verify(self.amendments[heard], list(done) + ([current] if current else []))
        return {"proposed": self.amendments[heard], "steps": steps, "corrections": notes, "ms": 0.0}


def test_an_agent_run_talks_back():
    sp = Speaker(play=False, synth=lambda text, voice: tone(0.2))
    events, _ = run_command(StillPolicy(), ScriptedPlanner(["spoon", "cup"], {"Skip the cup.": ["spoon"]}),
                            "spoon and cup", seed=0, checker=lambda skill, image: (True, 0.0),
                            voice=ScriptedVoice([(1.5, "Skip the cup.")]), linger_s=0.5,
                            on_event=SpeakEvents(sp), log=lambda *_: None)
    sp.close()
    said = [e for e in events if "said" in e]
    assert [(x["t"], x["text"]) for x in sp.lines] == [(e["t"], e["said"]) for e in said]
    assert [(e["kind"], e["said"]) for e in said if e["kind"] != "thinking"] == [
        ("plan", "I'll open the drawer, then place the spoon and cup. The drawer comes first — the spoon is inside."),
        ("amend", "OK — skipping the cup."), ("finished", "The table is set.")]
    # The answer to what was heard cuts off the plan sentence; nothing else interrupts.
    assert [x["cut"] for x in sp.lines][1] and not sp.lines[0]["cut"] and not sp.lines[-1]["cut"]


def test_an_interrupting_line_cuts_the_one_before(tmp_path):
    sp = Speaker(play=False, synth=lambda text, voice: tone(2.0))
    sp.say("a long plan", sim_time=0.5)
    sp.say("OK", sim_time=1.5, interrupt=True)
    sp.close()
    assert sp.write_track(tmp_path / "t.wav", duration_s=4.0, start_s=0.5) == [(0.0, "a long plan"), (1.0, "OK")]
    x, _ = sf.read(str(tmp_path / "t.wav"), dtype="int16")
    assert x[: RATE].all() and x[RATE: 3 * RATE].all() and not x[3 * RATE:].any()


def test_real_speechmatics_tts(tmp_path):
    from tenplaces.voice import _require_key

    try:
        _require_key()
    except RuntimeError:
        pytest.skip("SPEECHMATICS_API_KEY not set")
    import requests

    try:
        samples, rate = synthesize("Stopping.", "sarah", cache_dir=tmp_path)
    except (requests.ConnectionError, requests.Timeout):
        pytest.skip("no network")
    assert rate == RATE and 0.3 < len(samples) / rate < 3 and np.abs(samples).max() > 1000
