"""Run the whole agent on one scene: command (typed, spoken live, or a recording) -> VLM plan ->
per-skill policies -> camera checks; optionally keep listening while the robot works, and let it talk back.

    python scripts/run_agent.py --command "set the table, but skip the cup" --seed 3 --video out/video/agent_s3.mp4
    python scripts/run_agent.py --mic --listen --speak --seed 3 --video out/video/voice_s3.mp4  # SPEECHMATICS_API_KEY
    python scripts/run_agent.py --command "set the table" --say "8:Skip the fork." --seed 3  # scripted interjection
    python scripts/run_agent.py --audio command.wav --seed 3

--listen keeps the microphone open for the whole run: "stop" ends it at once, anything else changes the plan
after the current step (the arms keep moving while the planner thinks). --say T:TEXT injects a sentence at
simulated time T (repeatable; no microphone). --speak: the robot talks back; with --video its voice becomes the
audio track. --cores split: control (policy + camera check) on the P-cores, the VLM planner on the E-cores.
Runs dirs: for each skill the last dir that has it wins. Per-skill execution settings come from
out/eval/exec_settings.json. Events and the grade go to <video>.json or out/agent/.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tenplaces.agent import default_checker, run_command  # noqa: E402
from tenplaces.paths import OUT  # noqa: E402
from tenplaces.planner import VLMPlanner  # noqa: E402
from tenplaces.skill_policies import SkillPolicies  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--command")
    src.add_argument("--mic", action="store_true", help="speak the command (Speechmatics real-time STT)")
    src.add_argument("--audio", help="recorded command (wav/flac/ogg) streamed through Speechmatics")
    talk = ap.add_mutually_exclusive_group()
    talk.add_argument("--listen", action="store_true", help="keep listening while the robot works")
    talk.add_argument("--say", action="append", default=[], metavar="T:TEXT",
                      help="say TEXT at simulated time T seconds (repeatable)")
    ap.add_argument("--linger", type=float, default=4.0, help="seconds to keep listening after the last step")
    ap.add_argument("--speak", action="store_true",
                    help="the robot talks back (Speechmatics TTS); with --video its voice becomes the audio track")
    ap.add_argument("--tts-voice", default="sarah", choices=["sarah", "theo", "megan", "jack"])
    ap.add_argument("--silent", action="store_true", help="with --speak: audio track only, nothing through the speakers")
    ap.add_argument("--push", action="append", default=[], metavar="T:BODY:DX:DY[:DUR]",
                    help="slide BODY (plate, cup, spoon, fork) by DX, DY metres over DUR s (default 0.25) at "
                         "simulated time T, e.g. 20:plate:0:-0.07; finished steps are re-checked and redone")
    ap.add_argument("--look-first", type=float, default=0.95, metavar="P",
                    help="first look: steps the camera classifier already sees done with probability >= P are skipped "
                         "(on fresh tuning tables it skipped nothing: docs/findings.md)")
    ap.add_argument("--no-look-first", action="store_true", help="plan without the first look")
    ap.add_argument("--prepared", nargs="*", default=[], metavar="SKILL",
                    help="steps the scripted controller does before the robot starts (a table someone half-set)")
    ap.add_argument("--cores", default="split", choices=["default", "split"],
                    help="split: policy + camera check on P-cores, VLM planner on E-cores (tenplaces.cores)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", nargs="+", default=["out/train/skills_v1", "out/train/skills_v2", "out/train/skills_ctx",
                                                  "out/train/skills_ctx2"])
    ap.add_argument("--backend", default="ov-w8", choices=["torch", "ov-fp32", "ov-w8", "ov-a8w8_backbone"])
    ap.add_argument("--planner", default="models/Qwen3-VL-4B-Instruct-int4-ov")
    ap.add_argument("--calib-cache", default="data/table_v1_skill_cache")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()
    import torch

    # The policy's pre/post-processing is 3 small images and a 12-D vector; torch's default (14 OpenMP threads over
    # P- and E-cores on the i5-13600KF) competes with OpenVINO's control threads. One thread; OpenVINO does the work.
    torch.set_num_threads(1)
    from tenplaces.cores import no_power_throttling

    no_power_throttling()  # Windows may otherwise class a console-started robot as background and slow it down

    # Load the models first, so a spoken command is acted on as soon as it is transcribed.
    control_cfg = planner_cfg = None
    if args.cores == "split":
        from tenplaces.cores import control_config, planner_config

        control_cfg, planner_cfg = control_config(), planner_config()
        print(f"[cores] control {control_cfg}\n[cores] planner {planner_cfg}", flush=True)
    runs = [r for r in args.runs if Path(r).is_dir()]
    kw = (dict(device="cuda") if args.backend == "torch"
          else dict(backend=args.backend, calib_cache=args.calib_cache, ov_config=control_cfg))
    policy = SkillPolicies(runs, n_action_steps=10, **kw)
    for skill, ck in policy.sources.items():
        print(f"policy {skill}: {ck}", flush=True)
    # With the core split, a second planner pipeline on every core makes the first plan (the arms are still then).
    planner = VLMPlanner(args.planner, ov_config=planner_cfg, idle_config={} if args.cores == "split" else None)
    checker, _ = default_checker(planner, ov_config=control_cfg)

    command, spoken = args.command, None
    if args.mic or args.audio:
        from tenplaces.voice import transcribe_file, transcribe_mic

        t = time.perf_counter()
        show = lambda text: print(f"  … {text}", flush=True)  # noqa: E731 - partial transcripts as they arrive
        # With --video the microphone recording is kept next to it (<video>.command.wav, for scripts/voice_over.py).
        mic_wav = Path(args.video).with_suffix(".command.wav") if args.mic and args.video else None
        command, partials, ms = (transcribe_mic(on_partial=show, save_to=mic_wav) if args.mic
                                 else transcribe_file(args.audio, on_partial=show))
        if not command:
            sys.exit("no speech recognised")
        spoken = {"source": "mic" if args.mic else str(args.audio), "audio": str(mic_wav or args.audio), "partials": len(partials),
                  "ms_final_after_speech_end": round(ms) if ms == ms else None,
                  "s_total": round(time.perf_counter() - t, 2)}
        print(f"speechmatics: {command!r} (final {ms:.0f} ms after the speech ended)", flush=True)

    voice = live = None
    if args.listen:
        from tenplaces.listen import LiveVoice

        voice = live = LiveVoice()
    elif args.say:
        from tenplaces.listen import ScriptedVoice

        voice = ScriptedVoice([(float(s.split(":", 1)[0]), s.split(":", 1)[1]) for s in args.say])

    speaker = None
    if args.speak:  # the robot talks back; EchoGuard keeps the open mic from hearing the robot's own voice
        from tenplaces.speak import EchoGuard, Speaker

        speaker = Speaker(voice=args.tts_voice, play=not args.silent)
        speaker.prewarm()
        if args.listen:
            voice = EchoGuard(voice, speaker)

    rec = None
    if args.video:
        from tenplaces.demo_video import DemoRecorder

        Path(args.video).parent.mkdir(parents=True, exist_ok=True)
        rec = DemoRecorder(args.video, voice=voice)
    on_event = rec.on_event if rec else None
    if speaker is not None:
        from tenplaces.speak import SpeakEvents

        on_event = SpeakEvents(speaker, forward=on_event)
    try:
        from tenplaces.agent import parse_push

        pushes = [parse_push(p) for p in args.push]  # "after-plate:...": 1 s after the plate is confirmed
        events, grade = run_command(policy, planner, command, args.seed, checker=checker, voice=voice,
                                    linger_s=args.linger, disturb=pushes,
                                    look_first=None if args.no_look_first else args.look_first,
                                    prepare=args.prepared, on_frame=rec.on_frame if rec else None, on_event=on_event)
    finally:
        if rec:
            rec.close()
        if voice is not None:
            voice.close()
        live_info = None
        if live is not None and args.video:  # the whole microphone recording, for scripts/voice_over.py
            live_info = live.save(Path(args.video).with_suffix(".live.wav"))
            print(f"[voice] kept {live_info['audio_s']:.1f} s of microphone audio over {live_info['wall_s']:.1f} s of "
                  f"wall time, {len(live_info['utterances'])} sentences", flush=True)
        if speaker is not None:
            speaker.close()
            if rec is not None:  # the robot's voice as the video's audio track, aligned to simulated time
                from tenplaces.speak import mux

                wav = Path(args.video).with_suffix(".wav")
                speaker.write_track(wav, rec.duration_s, start_s=rec.t_start)
                mux(args.video, wav, args.video)
    out = Path(args.video).with_suffix(".json") if args.video else OUT / "agent" / f"seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"command": command, "voice": spoken, "seed": args.seed, "backend": args.backend,
                               "look_first": args.look_first, "prepared": args.prepared,
                               "video_t_start": rec.t_start if rec else None,  # sim time of the video's first frame
                               "live_audio": live_info and live_info["wav"],
                               "cores": args.cores, "sources": policy.sources, "exec_settings": policy.exec_settings,
                               "events": events, "grade": grade}, indent=1,
                              default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
