"""CPU placement: P-cores for control, E-cores for the planner, ANY_CORE on a single-type CPU; the properties
compile on this machine's OpenVINO and reach the models through the ov_config plumbing (tiny synthetic IRs)."""
from types import SimpleNamespace

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from tenplaces import cores
from tenplaces.ov_backend import compile_act
from tenplaces.state_classifier import N_SKILLS, OVStateClassifier

HYBRID = cores.Topology(6, 8, tuple(range(12)), tuple(range(12, 20)), "test")
UNIFORM = cores.Topology(8, 0, tuple(range(16)), (), "test")
UNKNOWN = cores.Topology(0, 0)


def tiny_model(shape=(1, 3, 32, 32), out=N_SKILLS):
    x = ops.parameter(list(shape), np.float32, name="x")
    w = ops.constant(np.full((out, shape[1], 3, 3), 0.01, np.float32))
    y = ops.reduce_mean(ops.convolution(x, w, [2, 2], [1, 1], [1, 1], [1, 1]), [2, 3], keep_dims=False)
    return ov.Model([y], [x], "tiny")


def test_hybrid_split():
    ctl, plan = cores.control_config(HYBRID), cores.planner_config(HYBRID)
    assert ctl["SCHEDULING_CORE_TYPE"] == "PCORE_ONLY" and ctl["INFERENCE_NUM_THREADS"] == 6
    assert ctl["PERFORMANCE_HINT"] == "LATENCY" and ctl["ENABLE_HYPER_THREADING"] is False
    assert plan["SCHEDULING_CORE_TYPE"] == "ECORE_ONLY" and plan["INFERENCE_NUM_THREADS"] == 8
    assert cores.config(cores.CONTROL, HYBRID) == ctl and cores.config(cores.PLANNER, HYBRID) == plan
    assert cores.control_config(HYBRID, threads=4)["INFERENCE_NUM_THREADS"] == 4


@pytest.mark.parametrize("topo", [UNIFORM, UNKNOWN])
def test_single_core_type_falls_back_to_any_core(topo):
    for role in (cores.CONTROL, cores.PLANNER):
        cfg = cores.config(role, topo)
        assert cfg["SCHEDULING_CORE_TYPE"] == "ANY_CORE" and "INFERENCE_NUM_THREADS" not in cfg
    assert not topo.hybrid and isinstance(topo.describe(), str)


def test_detected_topology_is_consistent():
    t = cores.topology()
    assert not set(t.p_logical) & set(t.e_logical)
    assert t.p_cores <= len(t.p_logical) and t.e_cores <= len(t.e_logical)
    assert t.describe()


@pytest.mark.parametrize("role", [cores.CONTROL, cores.PLANNER])
def test_openvino_applies_the_properties(role):
    cfg = cores.config(role)
    compiled = ov.Core().compile_model(tiny_model(), "CPU", cfg)
    compiled([np.zeros((1, 3, 32, 32), np.float32)])
    got = cores.effective(compiled)
    assert got["SCHEDULING_CORE_TYPE"].endswith(cfg["SCHEDULING_CORE_TYPE"])
    if "INFERENCE_NUM_THREADS" in cfg:
        assert got["INFERENCE_NUM_THREADS"] == str(cfg["INFERENCE_NUM_THREADS"])


def test_ov_config_reaches_the_classifier_and_the_policy(tmp_path):
    ov.save_model(tiny_model(), tmp_path / "clf.xml")
    cfg = cores.control_config()
    clf = OVStateClassifier(tmp_path / "clf.xml", ov_config=cfg)
    assert clf.probs(np.zeros((32, 32, 3), np.uint8)).shape == (N_SKILLS,)
    assert cores.effective(clf.compiled)["SCHEDULING_CORE_TYPE"].endswith(cfg["SCHEDULING_CORE_TYPE"])
    assert cores.effective(OVStateClassifier(tmp_path / "clf.xml").compiled)["SCHEDULING_CORE_TYPE"].endswith("ANY_CORE")

    ov.save_model(tiny_model(), tmp_path / "openvino" / "act_fp32.xml")  # compile_act reuses a cached IR
    policy = SimpleNamespace(config=SimpleNamespace(image_features={}, env_state_feature=None))
    act = compile_act(policy, None, "fp32", None, tmp_path / "openvino", ov_config=cfg)
    assert cores.effective(act.compiled)["SCHEDULING_CORE_TYPE"].endswith(cfg["SCHEDULING_CORE_TYPE"])
