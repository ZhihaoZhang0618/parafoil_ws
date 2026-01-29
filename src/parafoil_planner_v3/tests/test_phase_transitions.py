import numpy as np

from parafoil_planner_v3.guidance.phase_manager import PhaseConfig, PhaseManager
from parafoil_planner_v3.types import GuidancePhase, State, Target, Wind


def _state_at(p_ned, altitude_m, t=0.0):
    p = np.array([p_ned[0], p_ned[1], -altitude_m], dtype=float)
    return State(p_I=p, v_I=np.array([4.0, 0.0, 1.0]), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=t)


def test_cruise_to_approach_and_flare_and_landed():
    cfg = PhaseConfig(approach_entry_distance=180.0, glide_ratio_nominal=3.0, approach_altitude_extra=5.0, flare_altitude=10.0, flare_distance=30.0)
    pm = PhaseManager(cfg)

    target = Target(p_I=np.array([0.0, 0.0, 0.0]))
    wind = Wind(v_I=np.array([0.0, 2.0, 0.0]))

    # Start in cruise, far away
    s0 = _state_at([300.0, 0.0], altitude_m=80.0, t=0.0)
    tr = pm.update(s0, target, wind)
    assert not tr.triggered
    assert pm.current_phase == GuidancePhase.CRUISE

    # Close enough and low enough to enter approach
    s1 = _state_at([100.0, 0.0], altitude_m=30.0, t=1.0)
    tr = pm.update(s1, target, wind)
    assert tr.triggered
    assert tr.to_phase == GuidancePhase.APPROACH
    assert pm.current_phase == GuidancePhase.APPROACH

    # Trigger flare by altitude
    s2 = _state_at([20.0, 0.0], altitude_m=8.0, t=2.0)
    tr = pm.update(s2, target, wind)
    assert tr.triggered
    assert tr.to_phase == GuidancePhase.FLARE
    assert pm.current_phase == GuidancePhase.FLARE

    # Landed when down >= 0
    s3 = State(p_I=np.array([0.0, 0.0, 0.1]), v_I=np.zeros(3), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=3.0)
    tr = pm.update(s3, target, wind)
    assert tr.triggered
    assert tr.to_phase == GuidancePhase.LANDED
    assert pm.current_phase == GuidancePhase.LANDED


def test_abort_trigger_and_landed():
    cfg = PhaseConfig(
        approach_entry_distance=180.0,
        glide_ratio_nominal=3.0,
        approach_altitude_extra=5.0,
        flare_altitude=10.0,
        flare_distance=30.0,
        abort_min_altitude_m=3.0,
        abort_min_altitude_distance_m=80.0,
        abort_max_glide_ratio=8.0,
        abort_max_time_s=100.0,
        abort_max_phase_time_s=80.0,
    )
    pm = PhaseManager(cfg)
    target = Target(p_I=np.array([0.0, 0.0, 0.0]))
    wind = Wind(v_I=np.array([0.0, 2.0, 0.0]))

    # Low altitude but far from target triggers abort
    s0 = _state_at([200.0, 0.0], altitude_m=2.5, t=10.0)
    tr = pm.update(s0, target, wind)
    assert tr.triggered
    assert tr.to_phase == GuidancePhase.ABORT
    assert pm.current_phase == GuidancePhase.ABORT

    # Landed after abort
    s1 = State(p_I=np.array([0.0, 0.0, 0.1]), v_I=np.zeros(3), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=12.0)
    tr = pm.update(s1, target, wind)
    assert tr.triggered
    assert tr.to_phase == GuidancePhase.LANDED
