"""Which cores run what, on a hybrid Intel CPU (performance + efficiency cores).

Two workloads share the CPU during a run: real-time control (ACT policy + camera classifier, one batch-1
request every 40 ms control step) and the VLM planner (seconds per answer, and it can wait). Left to the OS,
the planner's threads land on the performance cores and the control step misses its budget. So control gets
the performance cores (latency hint, one stream, hyper-threading off) and the planner gets the efficiency
cores, both with their threads pinned. Measured with the real models on an idle i5-13600KF while the planner
generates (scripts/bench_concurrency.py): default scheduling 141 ms control step p50, 100% of steps over the
40 ms budget; P/E split 62 ms, 97%; + hyper-threading on 37 ms, 10%; + pinning (control and planner) 30 ms,
5% — the default here. (A synthetic-model dry run had suggested the opposite about pinning.)

    policy = LeRobotPolicy(ck, backend="ov-w8", ov_config=control_config())
    planner = VLMPlanner(ov_config=planner_config())

OpenVINO does not report how many cores of each type there are, so topology() asks the OS (Windows:
GetLogicalProcessorInformationEx, per-core EfficiencyClass; Linux: /sys/devices/cpu_core|cpu_atom). On a
CPU with one core type, or when the OS cannot tell, both roles fall back to ANY_CORE.
"""
import ctypes
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

CONTROL, PLANNER = "control", "planner"
EFFECTIVE_KEYS = ("PERFORMANCE_HINT", "SCHEDULING_CORE_TYPE", "INFERENCE_NUM_THREADS", "NUM_STREAMS",
                  "ENABLE_HYPER_THREADING", "ENABLE_CPU_PINNING")


@dataclass(frozen=True)
class Topology:
    p_cores: int  # physical cores of the fastest class (every core on a single-type CPU)
    e_cores: int  # physical cores of the slower classes (0 on a single-type CPU)
    p_logical: tuple = ()  # logical processor numbers on performance cores (both hyper-threads)
    e_logical: tuple = ()
    source: str = "unknown"

    @property
    def hybrid(self) -> bool:
        return self.p_cores > 0 and self.e_cores > 0

    def describe(self) -> str:
        if not self.p_cores:
            return f"{os.cpu_count()} logical processors, core types unknown ({self.source})"
        if not self.hybrid:
            return f"{self.p_cores} cores ({len(self.p_logical)} threads), one core type ({self.source})"
        return (f"{self.p_cores} P-cores ({len(self.p_logical)} threads) + {self.e_cores} E-cores "
                f"({len(self.e_logical)} threads) ({self.source})")


def _windows() -> Topology:
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    size = ctypes.c_ulong(0)
    k32.GetLogicalProcessorInformationEx(0, None, ctypes.byref(size))  # 0 = RelationProcessorCore
    buf = ctypes.create_string_buffer(size.value)
    if not k32.GetLogicalProcessorInformationEx(0, buf, ctypes.byref(size)):
        raise OSError(ctypes.get_last_error())
    raw, off, cores = buf.raw, 0, []
    while off < size.value:
        rec_size = int.from_bytes(raw[off + 4:off + 8], "little")
        eff_class = raw[off + 9]  # PROCESSOR_RELATIONSHIP.EfficiencyClass: higher = faster core
        groups = int.from_bytes(raw[off + 30:off + 32], "little")
        logical = []
        for g in range(groups):  # GROUP_AFFINITY {KAFFINITY Mask; WORD Group; WORD Reserved[3]} from offset 32
            ga = off + 32 + 16 * g
            mask = int.from_bytes(raw[ga:ga + 8], "little")
            group = int.from_bytes(raw[ga + 8:ga + 10], "little")
            logical += [64 * group + b for b in range(64) if mask >> b & 1]
        cores.append((eff_class, logical))
        off += rec_size
    return _split(cores, "Windows GetLogicalProcessorInformationEx")


def _linux() -> Topology:
    def cpus(path):
        out = []
        for part in Path(path).read_text().strip().split(","):
            a, _, b = part.partition("-")
            out += range(int(a), int(b or a) + 1)
        return out

    def physical(logical):  # one entry per physical core: its set of hyper-thread siblings
        return {Path(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list").read_text() for c in logical}

    p = cpus("/sys/devices/cpu_core/cpus")
    e = cpus("/sys/devices/cpu_atom/cpus") if Path("/sys/devices/cpu_atom/cpus").exists() else []
    return Topology(len(physical(p)), len(physical(e)), tuple(p), tuple(e), "Linux /sys/devices/cpu_core")


def _split(cores, source) -> Topology:
    best = max(c for c, _ in cores)
    p = [lp for c, lps in cores if c == best for lp in lps]
    e = [lp for c, lps in cores if c != best for lp in lps]
    return Topology(sum(c == best for c, _ in cores), sum(c != best for c, _ in cores),
                    tuple(sorted(p)), tuple(sorted(e)), source)


@lru_cache(maxsize=1)
def topology() -> Topology:
    try:
        if sys.platform == "win32":
            return _windows()
        if Path("/sys/devices/cpu_core/cpus").exists():
            return _linux()
    except (OSError, ValueError, AttributeError):
        pass
    return Topology(0, 0)


def control_config(topo: Topology | None = None, threads: int | None = None) -> dict:
    """OpenVINO CPU properties for the 25 Hz control models (policy, classifier)."""
    topo = topo or topology()
    # Pinning on: with the planner busy on the E-cores it cut the control step to 29.7 ms p50 and 5.2% of steps
    # over the 40 ms budget (62.1 ms / 96.9% unpinned), real models, idle i5-13600KF (bench_concurrency, e vs c).
    cfg = {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 1, "ENABLE_HYPER_THREADING": False,
           "ENABLE_CPU_PINNING": True, "SCHEDULING_CORE_TYPE": "PCORE_ONLY" if topo.hybrid else "ANY_CORE"}
    if threads or topo.hybrid:
        cfg["INFERENCE_NUM_THREADS"] = threads or topo.p_cores
    return cfg


def planner_config(topo: Topology | None = None, threads: int | None = None) -> dict:
    """OpenVINO CPU properties for the VLM planner (openvino_genai pipelines take them as keyword arguments)."""
    topo = topo or topology()
    if not topo.hybrid:
        return {"SCHEDULING_CORE_TYPE": "ANY_CORE", **({"INFERENCE_NUM_THREADS": threads} if threads else {})}
    return {"SCHEDULING_CORE_TYPE": "ECORE_ONLY", "INFERENCE_NUM_THREADS": threads or topo.e_cores,
            "ENABLE_CPU_PINNING": True}  # pinned as in bench_concurrency scenario e (see the module docstring)


def config(role: str, topo: Topology | None = None, threads: int | None = None) -> dict:
    return {CONTROL: control_config, PLANNER: planner_config}[role](topo, threads)


def effective(compiled) -> dict:
    """What a compiled model actually runs with (the plugin may adjust what was asked)."""
    out = {}
    for k in EFFECTIVE_KEYS:
        try:
            out[k] = str(compiled.get_property(k))
        except RuntimeError:
            out[k] = None
    return out
