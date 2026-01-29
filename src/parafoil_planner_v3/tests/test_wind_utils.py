import numpy as np
import pytest

from parafoil_planner_v3.utils.wind_utils import (
    WindConvention,
    WindInputFrame,
    clip_wind_xy,
    enu_to_ned,
    frame_from_frame_id,
    to_ned_wind_to,
)


def test_enu_to_ned_vector_mapping():
    # ENU [E, N, U] -> NED [N, E, D]
    v_enu = np.array([1.0, 2.0, 3.0])
    v_ned = enu_to_ned(v_enu)
    assert np.allclose(v_ned, np.array([2.0, 1.0, -3.0]))


def test_to_ned_wind_to_handles_frame_and_convention():
    v_enu_wind_to = np.array([1.0, 2.0, 0.0])
    v_ned = to_ned_wind_to(v_enu_wind_to, input_frame=WindInputFrame.ENU, convention=WindConvention.TO)
    assert np.allclose(v_ned, np.array([2.0, 1.0, -0.0]))

    v_enu_wind_from = np.array([1.0, 2.0, 0.0])
    v_ned = to_ned_wind_to(v_enu_wind_from, input_frame=WindInputFrame.ENU, convention=WindConvention.FROM)
    assert np.allclose(v_ned, np.array([-2.0, -1.0, 0.0]))


def test_frame_from_frame_id_best_effort():
    assert frame_from_frame_id("ned") == WindInputFrame.NED
    assert frame_from_frame_id("ENU") == WindInputFrame.ENU
    assert frame_from_frame_id("world") == WindInputFrame.ENU
    assert frame_from_frame_id("") is None


def test_clip_wind_xy_only_scales_xy():
    v = np.array([6.0, 8.0, 3.0])  # xy speed = 10
    out, clipped = clip_wind_xy(v, max_speed_mps=5.0)
    assert clipped
    assert np.allclose(out[2], 3.0)
    assert float(np.linalg.norm(out[:2])) == pytest.approx(5.0, abs=1e-9)
