import pytest

from parafoil_planner_v3.guidance.flare_guidance import FlareGuidance, FlareGuidanceConfig


def test_flare_brake_ramp_is_bounded_and_saturates():
    cfg = FlareGuidanceConfig(flare_initial_brake=0.6, flare_max_brake=1.0, flare_ramp_time=1.0)
    flare = FlareGuidance(config=cfg)

    b0 = flare._ramp_up_brake(0.0)
    bmid = flare._ramp_up_brake(0.5)
    b1 = flare._ramp_up_brake(1.0)
    b2 = flare._ramp_up_brake(2.0)

    assert b0 == pytest.approx(0.6)
    assert 0.6 <= bmid <= 1.0
    assert b1 == pytest.approx(1.0)
    assert b2 == pytest.approx(1.0)
    assert b0 <= bmid <= b1 <= b2

