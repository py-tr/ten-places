"""scripts/power_bench.py: HWiNFO CSV parsing and energy per inference, on a synthetic log."""
import importlib.util
from datetime import datetime, timedelta
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "power_bench.py"


def _module():
    spec = importlib.util.spec_from_file_location("power_bench", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_log(path: Path, t0: datetime, watts_per_second, date_fmt="%d.%m.%Y"):
    header = '"Date","Time","Core VIDs (avg) [V]","CPU Package Power [W]","CPU Package [°C]",'
    lines = [header]
    for i, w in enumerate(watts_per_second):
        for half in (0, 500):
            t = t0 + timedelta(seconds=i, milliseconds=half)
            lines.append(f'{t.strftime(date_fmt)},{t.strftime("%H:%M:%S")}.{half:03d},1.100,{w:.3f},55,')
    lines.append(header)  # HWiNFO repeats the header at the end
    path.write_text("\n".join(lines) + "\n", encoding="cp1252")


def test_energy_per_inference_total_and_above_idle(tmp_path):
    mod = _module()
    t0 = datetime(2026, 9, 13, 14, 0, 0)
    _write_log(tmp_path / "log.csv", t0, [10.0] * 10 + [40.0] * 10)
    stamps, watts, col = mod.read_hwinfo(tmp_path / "log.csv")
    assert col == "CPU Package Power [W]" and len(watts) == 40
    phases = {"phases": [
        {"phase": "idle", "start": (t0 + timedelta(seconds=1)).isoformat(),
         "end": (t0 + timedelta(seconds=9)).isoformat(), "inferences": 0},
        {"phase": "openvino_int8w", "start": (t0 + timedelta(seconds=11)).isoformat(),
         "end": (t0 + timedelta(seconds=19)).isoformat(), "inferences": 800},
    ]}
    rows = {r["phase"]: r for r in mod.analyze(phases, stamps, watts)}
    assert rows["idle"]["mean_w"] == 10.0 and rows["idle"]["mj_per_inference"] is None
    r = rows["openvino_int8w"]
    assert r["mean_w"] == 40.0 and r["inferences_per_s"] == 100.0
    assert abs(r["mj_per_inference"] - 400.0) < 1e-9  # 40 W x 8 s / 800 = 0.4 J
    assert abs(r["mj_per_inference_above_idle"] - 300.0) < 1e-9  # (40 - 10) W x 8 s / 800


def test_us_date_format(tmp_path):
    mod = _module()
    _write_log(tmp_path / "log.csv", datetime(2026, 9, 13, 9, 0, 0), [12.5] * 3, date_fmt="%m/%d/%Y")
    stamps, watts, _ = mod.read_hwinfo(tmp_path / "log.csv")
    assert stamps[0] == datetime(2026, 9, 13, 9, 0, 0) and watts == [12.5] * 6
