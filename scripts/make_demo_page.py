"""Build the project's landing page (docs/index.html) from the result files, for GitHub Pages (source: main /docs).

    python scripts/make_demo_page.py --version 11

Every number on the page is read out of a file in the repository at build time:

  held-out sets, full agent   out/eval/agent_table/*.json
  held-out sets, fixed seq.   out/eval/final<v>/ov_w8.csv, out/eval/v<v>_{set3,set4,stress_1.0}/ov_w8.csv
  robustness rows             out/eval/v<v>_{stress,shape}_*/ov_w8.csv
  demonstration runs          out/video/demo_final/summary.md, demo_final_extra/summary.md,
                              demo_live/take2/summary.md
  policy latency              out/benchmark/{drawer,spoon,plate,fork,cup}_*.md
  control under load          out/benchmark/concurrency.md
  energy per inference        out/benchmark/power.md
  first plan, core placement  out/benchmark/planner_placement.md

A handful of figures live in the prose of docs/findings.md and cannot be parsed from a table. They are collected in
PROSE below, each with the file and section it was read from, and nowhere else.

The page is written as ASCII only (non-ASCII has come out as mojibake before); the writer asserts it. The stills are
copied into docs/ because GitHub Pages serves that folder and out/ is git-ignored.
"""
import argparse
import csv
import json
import re
import shutil
from html import escape
from pathlib import Path

EVAL = Path("out/eval")
BENCH = Path("out/benchmark")
VIDEO = Path("out/video")
STEPS = ["drawer_open", "spoon", "plate", "fork", "cup"]
LABEL = {"drawer_open": "drawer", "spoon": "spoon", "plate": "plate", "fork": "fork", "cup": "cup"}
PLACED = ["spoon", "plate", "fork", "cup"]
SKILL_BENCH = ["drawer", "spoon", "plate", "fork", "cup"]
STILLS = VIDEO / "stills"
IMAGES = ["hero_web.jpg", "handoff_web.jpg", "grid_web.jpg"]

# Figures that exist only in the prose of docs/findings.md / README.md. Nothing else on the page is typed in.
PROSE = {
    # findings.md, "Repeatability and timing" -> "Who moves the arms"
    "policy_share": "92%",
    # findings.md, "OpenVINO precision" (int8_study.py, one hand-off checkpoint, 20 held-out seeds)
    "int8_all": "7/20", "int8_w": "14/20", "fp32": "13/20",
    # findings.md, "OpenVINO precision" -> INT4 weights (eval_agent_table.py --backend ov-w4, tuning seeds 100-149)
    "int4_cup": "0/50", "int4_int8": "41/50", "int4_mb": "23 MB", "int8_mb": "33 MB",
    # findings.md, "The hand-off is forced, not chosen" (IK over held-out seeds 200-249)
    "reach": "0 of 50",
    # README.md, "The scene" / "Results"
    "grip_n": "~17 N", "tol_cm": "2.5 cm",
}

# Attempts that were gated before they ran and missed their bar: docs/findings.md, sections
# "Not shipped, 2026-09-15" and "Disturbance repair (one attempt, not shipped)".
NOT_SHIPPED = [
    ("Cup, 100 more size demonstrations", "full tables at the trained size &ge; 47 of 50", "45 of 50",
     "Not shipped, 2026-09-15"),
    ("Spoon trained on its own failed grasps", "spoon first attempt &ge; 95 of 100", "94 of 100",
     "Not shipped, 2026-09-15"),
    ("Putting a knocked plate back (classifier v4 + plate fine-tune)", "&ge; 60 of 80 knocked plates put back",
     "30 of 80", "Disturbance repair"),
]

ASCII = [("→", "&rarr;"), ("—", "&mdash;"), ("–", "&ndash;"), ("“", "&ldquo;"),
         ("”", "&rdquo;"), ("‘", "&lsquo;"), ("’", "&rsquo;"), ("×", "&times;"),
         ("±", "&plusmn;"), ("≥", "&ge;"), ("≤", "&le;"), ("·", "&middot;"),
         (" ", " "), ("…", "...")]


def esc(s):
    """HTML-escape, then replace the non-ASCII characters the result files use with entities."""
    s = escape(str(s))
    for ch, ent in ASCII:
        s = s.replace(ch, ent)
    return s


def truth(v):
    return v is True or str(v) == "True"


def rows_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))["rows"]


def rows_csv(p):
    with Path(p).open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def count(rows):
    return sum(truth(r["success"]) for r in rows), len(rows)


def md_rows(path):
    """The rows of every pipe table in a markdown file, as lists of cells; separator rows dropped, headers kept."""
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") and c for c in cells):
            continue
        out.append(cells)
    return out


def bench_latency():
    """Median latency and IR size of the PyTorch and INT8-weights rows of each deployed skill's benchmark table.

    Columns: variant | precision | device | latency median / p95 | throughput | IR size | task success. Each file
    holds a second, narrower table (a core-scheduling sweep); only the two variant rows are read.
    """
    torch_ms, w8_ms, ir = [], [], set()
    for skill in SKILL_BENCH:
        for p in sorted(BENCH.glob(f"{skill}_*.md")):
            for cells in md_rows(p):
                if len(cells) < 6 or cells[0] not in ("pytorch", "w8"):
                    continue
                med = float(cells[3].split("/")[0].strip())
                if cells[0] == "pytorch":
                    torch_ms.append(med)
                else:
                    w8_ms.append(med)
                    ir.add(cells[5].strip())
    if not torch_ms or not w8_ms:
        raise SystemExit(f"no policy benchmark rows under {BENCH}")
    return torch_ms, w8_ms, ir


def concurrency():
    """scenario -> (p50/p99 ms, share of control steps over the 40 ms budget, planner seconds)."""
    out = {}
    for cells in md_rows(BENCH / "concurrency.md")[1:]:
        lat = [x.strip() for x in cells[2].split("/")]
        planner = cells[5].split("(")[0].strip()
        out[cells[0]] = (f"{lat[0]} / {lat[2]}", cells[3], planner + " s" if planner[:1].isdigit() else planner)
    return out


def energy():
    """phase -> (mean package power, energy per inference above idle in J).

    Columns: phase | CPU busy | mean package power | inferences/s | energy per inference | above idle.
    """
    out = {}
    for cells in md_rows(BENCH / "power.md"):
        if len(cells) < 6 or cells[0] == "phase":
            continue
        j = cells[5].replace("mJ", "").strip()
        out[cells[0]] = (cells[2].split("(")[0].strip(),
                         f"{float(j) / 1000:.2f} J" if j.replace(".", "", 1).isdigit() else "&ndash;")
    return out


def demo_sets():
    """The demonstration tables: (heading count, source path, body rows) per recorded set."""
    sets = []
    for path in (VIDEO / "demo_final/summary.md", VIDEO / "demo_final_extra/summary.md",
                 VIDEO / "demo_live/take2/summary.md"):
        if not path.exists():
            continue
        head = Path(path).read_text(encoding="utf-8").splitlines()[0]
        m = re.search(r"(\d+)\s*/\s*(\d+)", head)
        rows = md_rows(path)
        sets.append((int(m.group(1)), int(m.group(2)), path.as_posix(), rows[0], rows[1:]))
    return sets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="11", help="re-run version: 11, 12, ...")
    ap.add_argument("--repo-url", default="https://github.com/py-tr/ten-places")
    ap.add_argument("--video-url", default=None, help="embeddable video URL; omitted if not given")
    ap.add_argument("--out", default="docs/index.html")
    args = ap.parse_args()
    v = args.version
    docs = Path(args.out).parent

    # ---- held-out sets: full agent (json) and the fixed five-step sequence (csv), same tables -------------------
    sets = []
    for label, agent, fixed in (("0&ndash;49", f"agent_table/ov_w8_report{v}.json", f"final{v}/ov_w8.csv"),
                                ("200&ndash;249", f"agent_table/ov_w8_fresh200-249_v{v}.json", f"v{v}_stress_1.0/ov_w8.csv"),
                                ("250&ndash;299", f"agent_table/ov_w8_set3_250-299_v{v}.json", f"v{v}_set3/ov_w8.csv"),
                                ("850&ndash;899", f"agent_table/ov_w8_set4_850-899_v{v}.json", f"v{v}_set4/ov_w8.csv")):
        pa, pf = EVAL / agent, EVAL / fixed
        if pa.exists():
            sets.append((label, rows_json(pa), rows_csv(pf) if pf.exists() else None))
    if not sets:
        raise SystemExit(f"no held-out result files for version {v} under {EVAL}")

    agent_all = [r for _, rows, _ in sets for r in rows]
    ak, an = count(agent_all)
    fixed_all = [r for _, _, rows in sets if rows for r in rows]
    fk, fn = count(fixed_all) if fixed_all else (0, 0)
    steps_all = {s: sum(truth(r[s]) for r in agent_all) for s in STEPS}
    cutlery = sum(truth(r[s]) for r in agent_all for s in ("spoon", "fork"))

    # placement error per object, in cm, over the placed objects of every held-out table
    err = {}
    for s in PLACED:
        e = sorted(r[f"{s}_err_m"] * 100 for r in agent_all
                   if truth(r[s]) and r.get(f"{s}_err_m") is not None)
        err[s] = (e[len(e) // 2], e[-1], len(e))
    worst = max(v2[1] for v2 in err.values())

    # a lost table counts at its first step, in task order, still undone at the end
    lost = {s: [r["seed"] for r in agent_all if not truth(r["success"])
                and next((x for x in STEPS if not truth(r[x])), None) == s] for s in STEPS}

    set_rows = "".join(
        f"<tr><td>seeds {lab}</td><td><b>{count(rows)[0]}/{count(rows)[1]}</b></td>"
        f"<td>{f'{count(fx)[0]}/{count(fx)[1]}' if fx else '&ndash;'}</td></tr>"
        for lab, rows, fx in sets)

    per_seed = []
    for label, rows, _ in sets:
        for r in sorted(rows, key=lambda x: x["seed"]):
            first = next((LABEL[s] for s in STEPS if not truth(r[s])), "&ndash;")
            e = ", ".join(f"{s} {r[f'{s}_err_m'] * 100:.1f}" for s in PLACED
                          if truth(r[s]) and r.get(f"{s}_err_m") is not None)
            ok = truth(r["success"])
            per_seed.append(
                f"<tr class='{'ok' if ok else 'no'}'><td>{r['seed']}</td><td>{label}</td>"
                f"<td>{'set' if ok else 'incomplete'}</td><td>{sum(truth(r[s]) for s in STEPS)}/5</td>"
                f"<td>{first}</td><td>{e}</td></tr>")

    # ---- robustness: the fixed sequence and the full agent, pushed past the training ranges --------------------
    def csv_count(path):
        p = EVAL / path
        return count(rows_csv(p)) if p.exists() else None

    def json_count(path):
        p = EVAL / path
        return count(rows_json(p)) if p.exists() else None

    base_f = csv_count(f"v{v}_stress_1.0/ov_w8.csv")
    base_a = json_count(f"agent_table/ov_w8_fresh200-249_v{v}.json")
    rob = [("Training ranges and sizes &mdash; the baseline these are paired against", base_f, base_a),
           ("Friction, masses, light, colours widened &times;1.5", csv_count(f"v{v}_stress_1.5/ov_w8.csv"), None),
           ("The same, widened &times;2.0", csv_count(f"v{v}_stress_2.0/ov_w8.csv"),
            json_count(f"agent_table/ov_w8_fresh_stress2.0_v{v}.json")),
           ("Plate and cup sizes &plusmn;10%, cutlery length &plusmn;10%", csv_count(f"v{v}_shape_0.1/ov_w8.csv"), None),
           ("Plate and cup sizes &plusmn;20%, cutlery length &plusmn;10%", csv_count(f"v{v}_shape_0.2/ov_w8.csv"),
            json_count(f"agent_table/ov_w8_fresh_shape0.2_v{v}.json")),
           ("Cup size only &plusmn;20%", csv_count(f"v{v}_shape_cup_0.2/ov_w8.csv"), None)]
    rob_rows = "".join(
        f"<tr><td>{lab}</td><td>{f'{f[0]}/{f[1]}' if f else '&ndash;'}</td>"
        f"<td>{f'{a[0]}/{a[1]}' if a else '&ndash;'}</td></tr>" for lab, f, a in rob if f or a)

    # ---- demonstration runs ------------------------------------------------------------------------------------
    dsets = demo_sets()
    dk = sum(k for k, _, _, _, _ in dsets)
    dn = sum(n for _, n, _, _, _ in dsets)
    # The headline is the pre-registered set on its own; the later runs are counted beside it, never folded in.
    d0k, d0n = (dsets[0][0], dsets[0][1]) if dsets else (0, 0)
    extras = [f"{k}/{n}" for k, n, _, _, _ in dsets[1:]]
    half_set = extras[0] if extras else "&ndash;"
    live_take = extras[1] if len(extras) > 1 else "&ndash;"
    head = dsets[0][3] if dsets else []
    demo_head = "".join(f"<th>{esc(c)}</th>" for c in head)
    demo_body = ""
    for _, _, src, _, rows in dsets:
        for cells in rows:
            cls = "ok" if cells[-1].strip() == "PASS" else "no"
            demo_body += f"<tr class='{cls}'>" + "".join(f"<td>{esc(c)}</td>" for c in cells) + "</tr>"

    # ---- the Intel CPU -----------------------------------------------------------------------------------------
    torch_ms, w8_ms, ir = bench_latency()
    cc, en = concurrency(), energy()
    plan_head = Path(BENCH / "planner_placement.md").read_text(encoding="utf-8").splitlines()[2]
    plan_e = re.search(r"Median ([\d.]+) s on the E-cores, ([\d.]+) s on every core", plan_head)

    lat_rows = "".join([
        f"<tr><td>PyTorch FP32 eager, same CPU</td><td>{min(torch_ms):.1f}&ndash;{max(torch_ms):.1f} ms</td>"
        f"<td>&ndash;</td></tr>",
        f"<tr class='ok'><td><b>OpenVINO INT8 weights &mdash; deployed</b></td>"
        f"<td><b>{min(w8_ms):.1f}&ndash;{max(w8_ms):.1f} ms</b></td><td>{esc(sorted(ir)[0])}</td></tr>"])

    conc_rows = ""
    for key, lab in (("a", "Control alone"),
                     ("b", "Control + planner, default scheduling"),
                     ("c", "Control on the P-cores, planner on the E-cores"),
                     ("e", "The same, threads pinned &mdash; deployed")):
        if key not in cc:
            continue
        cls = " class='ok'" if key == "e" else ""
        conc_rows += (f"<tr{cls}><td>{lab}</td><td>{esc(cc[key][0])} ms</td>"
                      f"<td>{esc(cc[key][1])}</td><td>{esc(cc[key][2])}</td></tr>")

    gate_rows = "".join(f"<tr><td>{lab}</td><td>{bar}</td><td class='no'>{got}</td></tr>"
                        for lab, bar, got, _ in NOT_SHIPPED)

    imgs = [n for n in IMAGES if (STILLS / n).exists()]
    for n in imgs:
        docs.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(STILLS / n, docs / n)
    has = set(imgs)

    hero = ("""<figure class="shot">
<img src="hero_web.jpg" width="1440" height="720" alt="Two arms at a fully set table; the side panel lists the
verified plan with all five steps done and the line: no skill for, light a candle.">
<figcaption>Demonstration seed 6. The command was &ldquo;Set the table and light a candle.&rdquo; &mdash; the five
steps it can do are done, and the candle is named as something no skill does.</figcaption></figure>"""
            if "hero_web.jpg" in has else "")
    grid = ("""<figure class="shot wide">
<img src="grid_web.jpg" width="2400" height="480" alt="Ten demonstration runs tiled in two rows, each stamped PASS
except seed 2, which is stamped wrong refusal.">
<figcaption>All ten pre-registered runs, pass or fail, each with its own randomised table.</figcaption></figure>"""
            if "grid_web.jpg" in has else "")
    handoff = ("""<figure class="shot narrow">
<img src="handoff_web.jpg" width="960" height="720" alt="Wrist camera view of a gripper closing on the spoon as it
is handed across.">
<figcaption>Arm B's wrist camera during a hand-off &mdash; one of the three views the policy sees. At run time there
are no simulator object poses, only cameras and joint angles.</figcaption></figure>"""
               if "handoff_web.jpg" in has else "")
    video = (f'<figure class="shot"><iframe src="{esc(args.video_url)}" title="Demonstration runs" '
             f'allowfullscreen loading="lazy"></iframe></figure>' if args.video_url else "")

    repo = esc(args.repo_url)
    css = """
:root {
  color-scheme: light dark;
  --bg:#f1f4f6; --fg:#16212b; --mut:#4a5c6b; --faint:#71838f; --line:#d4dce1; --line2:#bfcad1;
  --card:#fff; --sunk:#e7ecef;
  --teal:#1f7381; --tealsoft:#e2eff1; --amber:#b4671a; --ambersoft:#fbeedd;
  --ok:#3d7a52; --no:#a3202b; --accent:#1f7381; --shade:rgba(20,22,28,.04);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg:#10161c; --fg:#e6edf3; --mut:#9db0be; --faint:#748796; --line:#26323c; --line2:#33424e;
    --card:#18212a; --sunk:#131b22;
    --teal:#5fb6c4; --tealsoft:#152b31; --amber:#e0913a; --ambersoft:#2e2213;
    --ok:#6fbc87; --no:#e07a6b; --accent:#5fb6c4; --shade:rgba(255,255,255,.04);
  }
}
* { box-sizing:border-box; }
body {
  margin:0; padding:0 0 80px; background:var(--bg); color:var(--fg);
  font:16px/1.6 "IBM Plex Sans",system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  -webkit-text-size-adjust:100%;
}
.wrap { max-width:60rem; margin:0 auto; padding:0 20px; }
header.top { background:var(--card); border-bottom:1px solid var(--line); margin-bottom:34px; }
header.top .wrap { padding-top:38px; padding-bottom:30px; }
.eyebrow { font-family:"IBM Plex Mono",ui-monospace,Consolas,monospace; font-size:.72rem;
  text-transform:uppercase; letter-spacing:.11em; color:var(--teal); margin-bottom:12px; }
h1 { font-family:Archivo,"IBM Plex Sans",sans-serif; font-size:clamp(2rem,5vw,2.9rem); line-height:1.08;
  margin:0 0 .2em; letter-spacing:-.03em; font-weight:700; }
h2 { font-family:Archivo,"IBM Plex Sans",sans-serif; font-size:1.34rem; margin:3.2rem 0 .7rem;
  letter-spacing:-.02em; font-weight:700; display:flex; align-items:center; gap:.6rem; }
h2::before { content:""; width:.55rem; height:1.5rem; border-radius:2px; background:var(--teal); flex:none; }
h3 { font-family:Archivo,"IBM Plex Sans",sans-serif; font-size:1.02rem; margin:1.9rem 0 .4rem; font-weight:600; }
p { margin:.7em 0; }
.sub { color:var(--mut); font-size:1.05rem; margin:0 0 1.4em; }
.lead { font-size:1.08rem; }
.mut { color:var(--mut); }
.note { color:var(--mut); font-size:.9rem; }
a { color:var(--accent); }
.hero-nums { display:grid; grid-template-columns:repeat(auto-fit,minmax(11rem,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:10px; overflow:hidden; margin-top:26px; }
.hn { background:var(--card); padding:15px 17px; }
.hn b { display:block; font-family:Archivo,sans-serif; font-size:1.85rem; line-height:1.1;
  letter-spacing:-.03em; font-variant-numeric:tabular-nums; color:var(--teal); }
.hn span { display:block; color:var(--mut); font-size:.81rem; margin-top:.3em; }
.nums { display:grid; grid-template-columns:repeat(auto-fit,minmax(10.5rem,1fr)); gap:1px;
  background:var(--line); border:1px solid var(--line); border-radius:10px; overflow:hidden; margin:1.6em 0; }
.num { padding:15px 17px; background:var(--card); }
.num b { display:block; font-family:Archivo,sans-serif; font-size:1.6rem; line-height:1.15;
  letter-spacing:-.025em; font-variant-numeric:tabular-nums; }
.num span { display:block; color:var(--mut); font-size:.84rem; margin-top:.35em; }
th { background:var(--sunk); }
tbody tr.ok td:first-child { box-shadow:inset 3px 0 0 var(--ok); }
tbody tr.no td:first-child { box-shadow:inset 3px 0 0 var(--no); }
tbody tr.no { background:var(--ambersoft); }
summary { font-weight:600; color:var(--teal); }
blockquote { border-left-color:var(--amber); background:var(--ambersoft); }
figure.shot { margin:1.4em 0; }
figure.shot img, figure.shot iframe {
  display:block; width:100%; height:auto; max-width:100%; border:1px solid var(--line); border-radius:12px;
  background:var(--shade);
}
figure.shot iframe { aspect-ratio:16/9; border:0; }
figure.narrow { max-width:26rem; }
figure.wide { overflow-x:auto; }
figure.wide img { min-width:44rem; }
figcaption { color:var(--mut); font-size:.88rem; margin-top:.55em; }
.tw { overflow-x:auto; margin:1em 0; }
table { border-collapse:collapse; width:100%; font-size:.94rem; }
caption { text-align:left; color:var(--mut); font-size:.88rem; padding-bottom:.5em; }
th, td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); vertical-align:top; }
th { font-weight:600; color:var(--mut); font-size:.82rem; text-transform:uppercase; letter-spacing:.04em;
     white-space:nowrap; }
tbody tr.ok td:last-child { color:var(--ok); }
tbody tr.no td:last-child { color:var(--no); }
td.no { color:var(--no); }
details { border:1px solid var(--line); border-radius:12px; padding:0 14px; margin:1em 0; background:var(--card); }
details[open] { padding-bottom:10px; }
summary { cursor:pointer; padding:12px 0; font-size:.94rem; }
details .tw { max-height:26rem; overflow:auto; }
code { background:var(--shade); padding:.12em .38em; border-radius:5px; font-size:.9em; }
ul { padding-left:1.15rem; } li { margin:.35em 0; }
blockquote { margin:1em 0; padding:.8em 1em; border-left:3px solid var(--line); background:var(--card);
             border-radius:0 10px 10px 0; }
blockquote p { margin:0; }
footer { margin-top:4rem; padding-top:1.2rem; border-top:1px solid var(--line); color:var(--mut); font-size:.9rem; }
@media (max-width:30rem) { body { padding:24px 16px 56px; } h1 { font-size:1.6rem; } figure.wide img { min-width:34rem; } }
"""

    html = f"""<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ten Places &mdash; two SO-101 arms set a dinner table</title>
<meta name="description" content="A spoken or typed command becomes a set dinner table in MuJoCo: two SO-101 arms,
a vision-language planner over five learned ACT policies, every model on an Intel CPU with OpenVINO.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@600;700&amp;family=IBM+Plex+Mono:wght@400;500&amp;family=IBM+Plex+Sans:wght@400;500;600&amp;display=swap">
<style>{css}</style>

<header class="top"><div class="wrap">
<div class="eyebrow">Intel track &middot; AI Infra Summit Hackathon 2026</div>
<h1>&ldquo;Set the table.&rdquo;</h1>
<p class="sub">Say it out loud or type it. Two SO-101 arms work out what you meant, hand the cutlery between them,
and check their own work. Every model runs on one Intel CPU.</p>
<div class="hero-nums">
<div class="hn"><b>{ak}/{an}</b><span>tables it had never seen, set completely</span></div>
<div class="hn"><b>{d0k}/{d0n}</b><span>pre-registered demonstrations done as asked</span></div>
<div class="hn"><b>{min(w8_ms):.0f} ms</b><span>per policy step, OpenVINO INT8</span></div>
<div class="hn"><b>{cutlery}/{len(agent_all) * 2}</b><span>spoon and fork placed after a hand-off</span></div>
</div>
</div></header>

<div class="wrap">

<p class="lead">Say or type <em>&ldquo;set the table&rdquo;</em>, or <em>&ldquo;no fork today, set the
rest&rdquo;</em>. In MuJoCo, arm A pulls the cutlery drawer open and hands the spoon and the fork across to arm B,
which places them with the plate and the cup. A vision-language model reads the command and the top camera and
writes a plan; a verifier repairs it; five learned ACT policies &mdash; one per skill &mdash; drive both arms from
three cameras at 25&nbsp;Hz; a camera classifier checks every step and re-queues what did not hold. Grasps are
friction contact only ({PROSE['grip_n']} gripper), and at run time the robot sees cameras and joint angles, never
simulator object poses.</p>

{hero}

<div class="nums">
 <div class="num"><b>{ak} / {an}</b><span>held-out tables set completely, four sets of 50</span></div>
 <div class="num"><b>{PROSE['policy_share']}</b><span>of control steps come from the learned policies</span></div>
 <div class="num"><b>{worst:.2f} cm</b><span>worst placement error over {an} tables, inside a 2.5 cm tolerance</span></div>
 <div class="num"><b>{min(w8_ms):.1f}&ndash;{max(w8_ms):.1f} ms</b><span>per policy step on the CPU, OpenVINO INT8
 weights</span></div>
</div>

<h2>Held-out tables</h2>
<p>Four sets of 50 randomised tables, never used for training or tuning, every selection frozen before the run and
each table run once. The full agent plans, checks after each skill, retries and re-queues; the fixed five-step
sequence runs the same policies in a fixed order on the same tables.</p>
<div class="tw"><table>
<thead><tr><th>Held-out set</th><th>Full agent</th><th>Fixed sequence</th></tr></thead>
<tbody>{set_rows}
<tr><td><b>All {an} tables</b></td><td><b>{ak}/{an}</b></td><td><b>{fk}/{fn}</b></td></tr></tbody></table></div>

<h3>Which steps, and where it lost</h3>
<div class="tw"><table>
<thead><tr><th>Step</th><th>Done, of {an}</th><th>Placement error, median</th><th>Largest</th></tr></thead><tbody>
{''.join(f"<tr><td>{LABEL[s]}</td><td>{steps_all[s]}/{an}</td>"
         f"<td>{f'{err[s][0]:.2f} cm' if s in err else '&ndash;'}</td>"
         f"<td>{f'{err[s][1]:.2f} cm' if s in err else '&ndash;'}</td></tr>" for s in STEPS)}
</tbody></table></div>
<p>Every placed object landed inside the {PROSE['tol_cm']} tolerance; the largest error over all {an} tables was
{worst:.2f} cm. The spoon and the fork are each handed from one arm to the other, and were placed on {cutlery} of
{2 * an} attempts.</p>
<p>{an - ak} tables were not completed. Counting each at its first step still undone at the end:
{', '.join(f'{LABEL[s]} {len(lost[s])}' for s in STEPS if lost[s])}. Never the first step to fail:
{', '.join(LABEL[s] for s in STEPS if not lost[s])}.</p>

<details><summary>Every one of the {an} held-out tables</summary>
<div class="tw"><table><thead><tr><th>Seed</th><th>Set</th><th>Result</th><th>Steps</th>
<th>First step undone</th><th>Placement error (cm)</th></tr></thead>
<tbody>{''.join(per_seed)}</tbody></table></div></details>

<h2>Demonstration runs</h2>
<p><b>{d0k} of {d0n}</b> pre-registered runs did exactly what was asked. The ten tables and their ten requests were
fixed in <code>configs/demo_seeds.json</code> before any of them was recorded &mdash; two spoken, one changed
mid-run by a scripted sentence &mdash; and each was then run once by the full agent on OpenVINO. Two further runs,
recorded the same way and listed with them below: a table already half set ({half_set}), and a live take with the
plan changed by voice while the arms were moving ({live_take}).</p>
{grid}
{video}
<div class="tw"><table><thead><tr>{demo_head}</tr></thead><tbody>{demo_body}</tbody></table></div>

<blockquote><p><b>The one miss is a refusal, not a motion.</b> On seed 2, &ldquo;No fork today, set the rest&rdquo;,
the plan was right (drawer, spoon, plate, cup), all four steps were carried out, and the table ended exactly as
asked. The impossible-part check, which runs separately to name anything no skill can do, read the phrase
&ldquo;set the rest&rdquo; as a thing it cannot do and announced it. The scorer counts a wrong refusal as a failed
run.</p></blockquote>

{handoff}

<h2>Where it degrades</h2>
<p>The same 50 held-out tables (seeds 200&ndash;249) with the scene pushed past what the policies were trained on.
Each row is paired against the first: same tables, one thing changed.</p>
<div class="tw"><table>
<thead><tr><th>Condition</th><th>Fixed sequence</th><th>Full agent</th></tr></thead>
<tbody>{rob_rows}</tbody></table></div>
<p>Widening friction, masses, lighting and colours is survivable. Object sizes are not: the plate and the cutlery
were each trained at a single size, and at &plusmn;20% a clear majority of the loss is the plate and the cup being
missed outright. Two further size fine-tunes lost the trained size and were not shipped. The grader itself is
size-proof &mdash; an object set on its target passes on all 50 tables at &plusmn;20%.</p>

<h2>On the Intel CPU</h2>
<p>An Intel Core i5-13600KF desktop CPU, no discrete GPU and no Core Ultra iGPU or NPU. Everything that runs the
robot runs there: the Qwen3-VL-4B INT4 planner on OpenVINO GenAI, the five ACT policies on OpenVINO INT8 weights,
and the ResNet18 camera classifier. Training and one PyTorch reference row used a GPU; nothing at run time does.</p>

<h3>The latency that buys closed-loop control</h3>
<div class="tw"><table><caption>Median over the five deployed skill policies, idle machine, batch 1.</caption>
<thead><tr><th>Variant</th><th>Latency, median</th><th>IR size</th></tr></thead>
<tbody>{lat_rows}</tbody></table></div>
<p>At {min(w8_ms):.1f}&ndash;{max(w8_ms):.1f} ms a forward pass fits inside every 40 ms control step, so the policy
re-runs each step and blends overlapping action chunks. That is what makes the hand-off work: re-planning every ten
actions instead froze the spoon at the pause before arm A lets go.</p>

<h3>The rungs that were rejected</h3>
<ul>
<li><b>INT8 everywhere</b> is faster and halves task success. Quantising the transformer's activations as well as
its weights: {PROSE['int8_all']} against {PROSE['fp32']} for FP32 and {PROSE['int8_w']} for INT8 weights, on the
same held-out seeds. Not shipped.</li>
<li><b>INT4 weights</b> ({PROSE['int4_mb']} per policy instead of {PROSE['int8_mb']}) break the cup. Four skills
hold; the cup's first actions drift by up to 0.05 rad and it fails every table &mdash; {PROSE['int4_cup']} against
{PROSE['int4_int8']} for INT8 weights. It was no faster than INT8 on this CPU in the same run. Not shipped.</li>
</ul>
<p>The ladder stops at INT8 weights, chosen on task success rather than on output error.</p>

<h3>Two models, one CPU, at the same time</h3>
<p>The policy has a 40 ms budget per control step while the 4B planner is generating.</p>
<div class="tw"><table>
<thead><tr><th>Scheduling</th><th>Control step p50 / p99</th><th>Steps over 40 ms</th>
<th>Planner answer, median</th></tr></thead><tbody>{conc_rows}</tbody></table></div>
<p>Left to the default scheduler the planner makes almost every control step late. Putting control on the P-cores
and the planner on the E-cores fixes it, and costs the planner about a second. The first plan is different: the arms
are still, so it runs on every core &mdash; a median of {plan_e.group(2) if plan_e else '?'} s against
{plan_e.group(1) if plan_e else '?'} s on the E-cores alone, same plan on all ten demonstration commands.</p>
<p class="note">Energy, CPU package power above idle: {en['pytorch_fp32'][1]} per inference in PyTorch,
{en['openvino_int8w'][1]} on OpenVINO INT8 weights.</p>

<h2>How it was built</h2>
<p>Each skill on its own reached about 90&ndash;100%. Chained together, the first run set <b>0 of 10</b> tables. The
loop that closed the gap was the same every time.</p>
<ul>
<li><b>Let it fail, then watch where.</b> Run one policy alone over thousands of tables and keep the states it
fails in &mdash; the plate's grasp slipping off the rim wall, the fork stalled mid hand-off with one arm still
holding it.</li>
<li><b>Take over from the failure, do not restart.</b> The scripted controller continues from the exact state the
policy left, and that continuation becomes training data. Restarting instead is what killed the cup version of this:
it learned to let go of a good grasp, and collapsed.</li>
<li><b>Write the gate before the result.</b> Every candidate had a numeric bar, set on tuning seeds before it ran.
Held-out sets were run once, with every selection already frozen.</li>
</ul>
<p>Three attempts met that bar and were not shipped:</p>
<div class="tw"><table>
<thead><tr><th>Attempt</th><th>Bar, written first</th><th>Result</th></tr></thead>
<tbody>{gate_rows}</tbody></table></div>
<p>The hand-off itself is not a design flourish: arm A sits at x&nbsp;=&nbsp;&minus;0.30 m and arm B at
+0.30 m, and over 50 held-out tables inverse kinematics reaches the other arm's targets on {PROSE['reach']}. Neither
arm can do any part of the other's work, so cutlery from A's drawer changes hands mid-air.</p>

<h2>The repository</h2>
<ul>
<li><a href="{repo}">{repo}</a> &mdash; code, configs and every result file quoted here.</li>
<li><a href="{repo}#readme">README</a> &mdash; architecture, the full results tables, the OpenVINO measurements.</li>
<li><a href="{repo}/blob/main/docs/findings.md">docs/findings.md</a> &mdash; the engineering record: every change
tried, what it measured, and what did not work.</li>
</ul>

<footer>Every number on this page is read at build time out of a result file in the repository by
<code>scripts/make_demo_page.py</code>; none is typed in. Built for the Intel track of the AI Infra Summit
Hackathon 2026.</footer>
</div>
</html>
"""
    if not html.isascii():
        bad = sorted({c for c in html if not c.isascii()})
        where = [html[max(0, html.index(c) - 40):html.index(c) + 40].encode("ascii", "backslashreplace").decode()
                 for c in bad[:3]]
        raise SystemExit("non-ASCII in the generated page: "
                         + ", ".join(f"U+{ord(c):04X}" for c in bad) + " (add them to ASCII)\n"
                         + "\n".join(where))
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="ascii")
    kb = p.stat().st_size / 1024 + sum((docs / n).stat().st_size for n in imgs) / 1024
    print(f"{p}: agent {ak}/{an}, fixed {fk}/{fn}, {len(per_seed)} per-seed rows, demos {dk}/{dn}, "
          f"{len(imgs)} images, {kb:.0f} kB total")


if __name__ == "__main__":
    main()
