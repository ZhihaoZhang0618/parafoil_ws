import numpy as np

from parafoil_planner_v3.guidance.approach_guidance import ApproachGuidance, ApproachGuidanceConfig
from parafoil_planner_v3.types import State, Target, Wind


def _state_at(p_ned, altitude_m, t=0.0):
    p = np.array([p_ned[0], p_ned[1], -altitude_m], dtype=float)
    return State(p_I=p, v_I=np.array([4.0, 0.0, 0.0]), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=t)


def test_wind_correction_aimpoint_shifts_upwind():
    cfg = ApproachGuidanceConfig(wind_correction_gain=1.0, wind_max_drift_m=100.0)
    guidance = ApproachGuidance(cfg)
    state = _state_at([0.0, 0.0], altitude_m=30.0, t=0.0)
    target = Target(p_I=np.array([100.0, 0.0, 0.0]))
    # Wind pushes to +E (east). Aimpoint should shift to negative east (west) to compensate.
    wind = Wind(v_I=np.array([0.0, 2.0, 0.0]))

    aim = guidance.compute_aimpoint(state, target, wind, brake=0.3)
    assert aim[1] < 0.0
