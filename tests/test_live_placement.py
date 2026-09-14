"""Placing what was said while the robot worked (tenplaces.listen.live_placements): each heard sentence goes to
the simulated time it was said, through the computer's clock logged against simulated time."""
from tenplaces.listen import live_placements, sim_time_of

CLOCK = [(1000.0, 10.0), (1002.0, 11.0), (1004.0, 13.0)]  # the simulation ran at half speed, then real time


def test_clock_interpolates_and_clamps():
    assert sim_time_of(1001.0, CLOCK) == 10.5
    assert sim_time_of(1003.0, CLOCK) == 12.0
    assert sim_time_of(990.0, CLOCK) == 10.0 and sim_time_of(1010.0, CLOCK) == 13.0


def test_heard_sentences_are_placed_where_they_were_said():
    live = {"t0_wall": 990.0, "clock": CLOCK, "utterances": [
        {"text": "The plate moved — putting it back.", "audio_start": 9.0, "audio_end": 10.0, "wall": 1001.0},
        {"text": "Skip the fork.", "audio_start": 11.0, "audio_end": 12.5, "wall": 1003.5},
        {"text": "No span.", "audio_start": None, "audio_end": None, "wall": 1003.9}]}
    events = [{"kind": "skill_start", "t": 10.0}, {"kind": "heard", "text": "Skip the fork.", "t": 12.8},
              {"kind": "heard", "text": "No span.", "t": 12.9}]
    (p,) = live_placements(events, live)  # the robot's own voice (no heard event) and a sentence without timings
    assert p["text"] == "Skip the fork."
    assert (p["sim_start"], p["sim_end"]) == (10.5, 11.5)  # wall 1001.0 and 1002.5
    assert p["sim_end"] <= p["heard_t"]


def test_repeated_sentences_pair_in_order():
    live = {"t0_wall": 1000.0, "clock": CLOCK, "utterances": [
        {"text": "Stop.", "audio_start": 0.5, "audio_end": 1.0, "wall": 1001.5},
        {"text": "Stop.", "audio_start": 3.0, "audio_end": 3.5, "wall": 1003.9}]}
    events = [{"kind": "heard", "text": "Stop.", "t": 11.0}, {"kind": "heard", "text": "Stop.", "t": 13.0}]
    first, second = live_placements(events, live)
    assert first["audio_start"] == 0.5 and second["audio_start"] == 3.0
