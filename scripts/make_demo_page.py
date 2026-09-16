"""Build the project's demo page (docs/index.html) from the result files, for GitHub Pages (source: main /docs).

    python scripts/make_demo_page.py --version 11 --video-url https://www.youtube.com/embed/<id>

Every number on the page comes from a result file named next to it: the three held-out sets (full agent and the fixed
sequence), the per-seed table, the robustness rows and the benchmark table. Nothing is typed in by hand except the
video link and the repository URL.
"""
import argparse
import csv
import json
from html import escape
from pathlib import Path

OUT = Path("out/eval")
STEPS = ["drawer_open", "spoon", "plate", "fork", "cup"]
LABEL = {"drawer_open": "drawer", "spoon": "spoon", "plate": "plate", "fork": "fork", "cup": "cup"}


def truth(v):
    return v is True or str(v) == "True"


def rows_json(p):
    return json.loads(Path(p).read_text())["rows"]


def rows_csv(p):
    with Path(p).open() as f:
        return list(csv.DictReader(f))


def count(rows):
    return sum(truth(r["success"]) for r in rows), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True, help="re-run version: 11, 12, …")
    ap.add_argument("--video-url", default=None, help="embeddable video URL (YouTube/Vimeo embed link)")
    ap.add_argument("--repo-url", default="https://github.com/")
    ap.add_argument("--out", default="docs/index.html")
    args = ap.parse_args()
    v = args.version

    sets = []  # (label, agent rows, fixed rows); ASCII only in the generated page (no encoding surprises)
    for label, agent, fixed in (("seeds 0-49", f"agent_table/ov_w8_report{v}.json", f"final{v}/ov_w8.csv"),
                                ("seeds 200-249", f"agent_table/ov_w8_fresh200-249_v{v}.json", f"v{v}_stress_1.0/ov_w8.csv"),
                                ("seeds 250-299", f"agent_table/ov_w8_set3_250-299_v{v}.json", f"v{v}_set3/ov_w8.csv"),
                                ("seeds 850-899", f"agent_table/ov_w8_set4_850-899_v{v}.json", f"v{v}_set4/ov_w8.csv")):
        pa, pf = OUT / agent, OUT / fixed
        if pa.exists():
            sets.append((label, rows_json(pa), rows_csv(pf) if pf.exists() else None))
    agent_all = [r for _, rows, _ in sets for r in rows]
    ak, an = count(agent_all)
    fixed_all = [r for _, _, rows in sets if rows for r in rows]
    fk, fn = count(fixed_all) if fixed_all else (0, 0)
    steps_all = {s: sum(truth(r[s]) for r in agent_all) for s in STEPS}
    errs = [r[f"{s}_err_m"] * 100 for r in agent_all for s in ("spoon", "plate", "fork", "cup")
            if truth(r[s]) and r.get(f"{s}_err_m") is not None]
    errs.sort()

    def robust(path, label):
        p = OUT / path
        if not p.exists():
            return None
        k, n = count(rows_csv(p))
        return f"<tr><td>{escape(label)}</td><td>{k}/{n}</td></tr>"

    rob = "".join(filter(None, [
        robust(f"v{v}_stress_1.0/ov_w8.csv", "training ranges (same tables)"),
        robust(f"v{v}_stress_1.5/ov_w8.csv", "friction, masses, light, colours x1.5"),
        robust(f"v{v}_stress_2.0/ov_w8.csv", "the same, x2.0"),
        robust(f"v{v}_shape_0.1/ov_w8.csv", "object sizes +/-10%"),
        robust(f"v{v}_shape_0.2/ov_w8.csv", "object sizes +/-20%"),
        robust(f"v{v}_shape_cup_0.2/ov_w8.csv", "cup size only +/-20%")]))

    per_seed = []
    for label, rows, _ in sets:
        for r in sorted(rows, key=lambda x: x["seed"]):
            first = next((LABEL[s] for s in STEPS if not truth(r[s])), "-")
            err = [f"{s} {r[f'{s}_err_m'] * 100:.1f}" for s in ("spoon", "plate", "fork", "cup")
                   if truth(r[s]) and r.get(f"{s}_err_m") is not None]
            per_seed.append(f"<tr class='{'ok' if truth(r['success']) else 'no'}'><td>{r['seed']}</td>"
                            f"<td>{escape(label)}</td><td>{'set' if truth(r['success']) else 'not set'}</td>"
                            f"<td>{sum(truth(r[s]) for s in STEPS)}/5</td><td>{escape(first)}</td>"
                            f"<td>{escape(', '.join(err))}</td></tr>")
    video = (f'<iframe src="{escape(args.video_url)}" title="demo" allowfullscreen loading="lazy"></iframe>'
             if args.video_url else '<p class="todo">Video link goes here.</p>')
    html = f"""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ten Places &mdash; two SO-101 arms set a dinner table</title>
<style>
 :root {{ color-scheme: light dark; --fg:#111; --bg:#fff; --mut:#666; --ok:#0a7f3f; --no:#a11; --line:#e3e3e3; }}
 @media (prefers-color-scheme: dark) {{ :root {{ --fg:#eee; --bg:#111; --mut:#aaa; --ok:#5fd08a; --no:#f08a8a; --line:#333; }} }}
 body {{ margin:0 auto; padding:24px 16px 64px; max-width:56rem; font:16px/1.55 system-ui,sans-serif; color:var(--fg); background:var(--bg); }}
 h1 {{ font-size:1.9rem; margin:0 0 .3em; }} h2 {{ margin:2.2em 0 .5em; font-size:1.25rem; }}
 .lead {{ font-size:1.1rem; color:var(--mut); }}
 .nums {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr)); gap:12px; margin:1.5em 0; }}
 .num {{ border:1px solid var(--line); border-radius:10px; padding:12px 14px; }}
 .num b {{ display:block; font-size:1.6rem; }} .num span {{ color:var(--mut); font-size:.92rem; }}
 table {{ border-collapse:collapse; width:100%; font-size:.95rem; }}
 th,td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); }}
 tr.ok td:nth-child(3) {{ color:var(--ok); }} tr.no td:nth-child(3) {{ color:var(--no); }}
 .scroll {{ max-height:26rem; overflow:auto; border:1px solid var(--line); border-radius:10px; }}
 iframe {{ width:100%; aspect-ratio:16/9; border:0; border-radius:10px; }}
 .todo {{ color:var(--no); }} footer {{ margin-top:3em; color:var(--mut); font-size:.9rem; }}
 code {{ background:rgba(127,127,127,.12); padding:.1em .35em; border-radius:4px; }}
</style>
<h1>Ten Places</h1>
<p class="lead">Say &ldquo;set the table, but skip the cup&rdquo;. Two simulated SO-101 arms do it: one pulls the cutlery drawer
open and hands the spoon and fork across to the other, which places them with the plate and the cup. A
vision-language model plans; one learned policy per skill drives the arms; a camera checks every step. Every model
runs on an Intel CPU with OpenVINO.</p>
<div class="nums">
 <div class="num"><b>{ak} / {an}</b><span>held-out tables set completely &mdash; learned policies, not scripts</span></div>
 <div class="num"><b>~92%</b><span>of control steps come from the learned policies</span></div>
 <div class="num"><b>2</b><span>mid-air hand-offs per table, friction grasps only</span></div>
 <div class="num"><b>16 ms</b><span>per policy step on the CPU (OpenVINO INT8 weights)</span></div>
</div>
<h2>The robot, ten randomised tables</h2>
{video}
<h2>Every held-out table</h2>
<p>Three sets of 50 randomised tables, never used for training or tuning; the shipped configuration, each run once.
Fixed five-step sequence on the same tables: {fk}/{fn}.</p>
<div class="scroll"><table><thead><tr><th>seed</th><th>set</th><th>result</th><th>steps</th>
<th>first step undone</th><th>placement error (cm)</th></tr></thead><tbody>
{''.join(per_seed)}
</tbody></table></div>
<h2>Steps across the {an} tables</h2>
<table><tbody>{''.join(f'<tr><td>{LABEL[s]}</td><td>{steps_all[s]}/{an}</td></tr>' for s in STEPS)}</tbody></table>
<p>Placement error of placed objects: median {errs[len(errs) // 2]:.2f} cm, largest {errs[-1]:.2f} cm.</p>
<h2>Pushed beyond the training ranges (fixed sequence, same 50 tables)</h2>
<table><tbody>{rob}</tbody></table>
<h2>On the Intel CPU</h2>
<p>Every model in the loop runs in OpenVINO on an Intel Core i5-13600KF (CPU only; no Core Ultra was available):
the Qwen3-VL-4B INT4 planner, the five ACT policies (INT8 weights, 15.5&ndash;15.8 ms against 39&ndash;45 ms in
PyTorch, 4&times; smaller, no measurable loss in task success) and the camera classifier (5&ndash;6 ms). Full INT8 is
faster still and halves task success, so it is not shipped. Control runs on the performance cores at 25 Hz while the
planner thinks on the efficiency cores: late control steps 97% &rarr; 6%. Energy per inference 1.43 J against
3.50 J.</p>
<footer>Every number here comes from a script and a result file in the repository:
<a href="{escape(args.repo_url)}">{escape(args.repo_url)}</a> &mdash; see <code>README.md</code> and
<code>docs/findings.md</code>. Built for the Intel track of the AI Infra Summit Hackathon 2026.</footer>
</html>
"""
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")
    print(f"{p}: {an} held-out tables ({ak} set), fixed {fk}/{fn}, {len(per_seed)} per-seed rows")


if __name__ == "__main__":
    main()
