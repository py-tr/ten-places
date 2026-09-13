"""Success predicates and failure taxonomy, computed from simulator state. Used for evaluation only."""
import numpy as np

PLACE_TOL = 0.03  # spoon centre within 3 cm of the placemat centre


def spoon_state(m, d):
    return d.xpos[m.body("spoon").id].copy()


def touching(m, d, body_a: str, geom_prefixes: tuple) -> bool:
    bid = m.body(body_a).id
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = c.geom1, c.geom2
        for ga, gb in ((g1, g2), (g2, g1)):
            if m.geom_bodyid[ga] == bid and (m.geom(gb).name or "").startswith(geom_prefixes):
                return True
    return False


def grade_handoff_episode(m, d, target_xy, stats: dict) -> dict:
    """Grade a closed-loop episode using what happened during it (`stats` from the evaluator):
    a_lifted / b_lifted = that arm's pads touched the spoon while it was off the table."""
    result = grade_handoff(m, d, target_xy, trace={})
    if not result["success"] and result["failure"] != "stuck_in_gripper":
        if not stats["a_lifted"]:
            result["failure"] = "a_grasp_miss"
        elif not stats["b_lifted"]:
            result["failure"] = "handoff_miss"
        elif result["failure"] != "not_on_table":
            result["failure"] = "place_off_target"
    result.update({k: stats[k] for k in ("a_lifted", "b_lifted")}, spoon_z_max=round(stats["z_max"], 3))
    return result


def grade_handoff(m, d, target_xy, trace: dict) -> dict:
    """trace: phase -> spoon z at the end of that phase (filled by the harness)."""
    p = spoon_state(m, d)
    dist = float(np.linalg.norm(p[:2] - np.asarray(target_xy)))
    held = touching(m, d, "spoon", ("a_pad", "b_pad"))
    on_table = p[2] < 0.02
    ok = dist < PLACE_TOL and on_table and not held
    if ok:
        failure = None
    elif trace.get("a_carry_z", 1) < 0.03:
        failure = "a_grasp_miss"
    elif trace.get("a_release_z", 1) < 0.03:
        failure = "handoff_drop"
    elif held:
        failure = "stuck_in_gripper"
    elif not on_table:
        failure = "not_on_table"
    else:
        failure = "place_off_target"
    return {"success": bool(ok), "failure": failure, "place_err_m": dist, "spoon_final": p.round(4).tolist()}
