import numpy as np
import pytest

from parafoil_planner_v3.trajectory_library.scenario_features import compute_scenario_features
from parafoil_planner_v3.types import State, Target, Wind


def test_compute_scenario_features_relative_wind_sign():
    # Target due North => bearing = 0 (atan2(E, N))
    state = State(p_I=np.array([0.0, 0.0, -10.0]), v_I=np.zeros(3), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=0.0)
    target = Target(p_I=np.array([10.0, 0.0, 0.0]))

    # Wind-to East => wind_angle = +pi/2, relative_wind = +pi/2
    wind = Wind(v_I=np.array([0.0, 1.0, 0.0]))
    feats = compute_scenario_features(state, target, wind)
    assert float(feats[4]) == pytest.approx(np.pi / 2.0, abs=1e-9)

    # Wind-to West => wind_angle = -pi/2, relative_wind = -pi/2
    wind = Wind(v_I=np.array([0.0, -1.0, 0.0]))
    feats = compute_scenario_features(state, target, wind)
    assert float(feats[4]) == pytest.approx(-np.pi / 2.0, abs=1e-9)
