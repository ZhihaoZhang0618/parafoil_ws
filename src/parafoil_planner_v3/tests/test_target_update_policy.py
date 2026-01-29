import numpy as np

from parafoil_planner_v3.landing_site_selector import LandingSiteSelection
from parafoil_planner_v3.target_update_policy import TargetUpdatePolicy, TargetUpdatePolicyConfig, UpdateReason
from parafoil_planner_v3.types import GuidancePhase, Target


def _make_selection(x: float, score: float, margin: float = 1.0) -> LandingSiteSelection:
    return LandingSiteSelection(
        target=Target(p_I=np.array([x, 0.0, 0.0], dtype=float)),
        score=float(score),
        risk=0.0,
        distance_to_desired_m=0.0,
        reach_margin_mps=float(margin),
        time_to_land_s=10.0,
        reason="ok",
    )


def test_hysteresis_blocks_small_change():
    cfg = TargetUpdatePolicyConfig(score_hysteresis=0.5, dist_hysteresis_m=10.0)
    policy = TargetUpdatePolicy(cfg)
    sel_a = _make_selection(0.0, score=1.0)
    sel_b = _make_selection(1.0, score=0.8)

    target, reason = policy.update(sel_a, GuidancePhase.CRUISE, current_time=0.0, current_margin=1.0)
    assert reason == UpdateReason.INITIAL
    assert np.allclose(target.position_xy, sel_a.target.position_xy)

    target, reason = policy.update(sel_b, GuidancePhase.CRUISE, current_time=1.0, current_margin=1.0)
    assert reason == UpdateReason.CRUISE_HYSTERESIS
    assert np.allclose(target.position_xy, sel_a.target.position_xy)


def test_hysteresis_allows_significant_change():
    cfg = TargetUpdatePolicyConfig(score_hysteresis=0.5, dist_hysteresis_m=10.0)
    policy = TargetUpdatePolicy(cfg)
    sel_a = _make_selection(0.0, score=1.0)
    sel_b = _make_selection(2.0, score=0.1)

    policy.update(sel_a, GuidancePhase.CRUISE, current_time=0.0, current_margin=1.0)
    target, reason = policy.update(sel_b, GuidancePhase.CRUISE, current_time=1.0, current_margin=1.0)
    assert reason == UpdateReason.CRUISE_UPDATE
    assert np.allclose(target.position_xy, sel_b.target.position_xy)


def test_flare_lock():
    cfg = TargetUpdatePolicyConfig(flare_lock=True)
    policy = TargetUpdatePolicy(cfg)
    sel_a = _make_selection(0.0, score=1.0)
    sel_b = _make_selection(5.0, score=0.0)

    policy.update(sel_a, GuidancePhase.CRUISE, current_time=0.0, current_margin=1.0)
    target, reason = policy.update(sel_b, GuidancePhase.FLARE, current_time=1.0, current_margin=1.0)
    assert reason == UpdateReason.FLARE_LOCKED
    assert np.allclose(target.position_xy, sel_a.target.position_xy)


def test_approach_emergency_only():
    cfg = TargetUpdatePolicyConfig(
        approach_allow_update="emergency_only",
        score_hysteresis=0.5,
        approach_significant_factor=2.0,
    )
    policy = TargetUpdatePolicy(cfg)
    sel_a = _make_selection(0.0, score=1.0)
    sel_b = _make_selection(3.0, score=0.6)
    sel_c = _make_selection(4.0, score=-0.1)

    policy.update(sel_a, GuidancePhase.CRUISE, current_time=0.0, current_margin=1.0)
    target, reason = policy.update(sel_b, GuidancePhase.APPROACH, current_time=1.0, current_margin=1.0)
    assert reason == UpdateReason.APPROACH_HYSTERESIS
    assert np.allclose(target.position_xy, sel_a.target.position_xy)

    target, reason = policy.update(sel_c, GuidancePhase.APPROACH, current_time=2.0, current_margin=1.0)
    assert reason == UpdateReason.APPROACH_SIGNIFICANT
    assert np.allclose(target.position_xy, sel_c.target.position_xy)


def test_emergency_reselect_and_cooldown():
    cfg = TargetUpdatePolicyConfig(emergency_margin_mps=-0.5, emergency_cooldown_s=2.0)
    policy = TargetUpdatePolicy(cfg)
    sel_a = _make_selection(0.0, score=1.0)
    sel_b = _make_selection(5.0, score=0.2)
    sel_c = _make_selection(10.0, score=0.1)

    policy.update(sel_a, GuidancePhase.CRUISE, current_time=0.0, current_margin=1.0)
    target, reason = policy.update(sel_b, GuidancePhase.CRUISE, current_time=1.0, current_margin=-1.0)
    assert reason == UpdateReason.EMERGENCY_RESELECT
    assert np.allclose(target.position_xy, sel_b.target.position_xy)

    target, reason = policy.update(sel_c, GuidancePhase.CRUISE, current_time=2.0, current_margin=-1.0)
    assert reason == UpdateReason.EMERGENCY_COOLDOWN
    assert np.allclose(target.position_xy, sel_b.target.position_xy)
