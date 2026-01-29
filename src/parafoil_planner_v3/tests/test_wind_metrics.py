import numpy as np

from parafoil_planner_v3.trajectory_library.wind_metrics import compute_wind_drift_metrics
from parafoil_planner_v3.types import Scenario, State, Trajectory, Waypoint


def test_compute_wind_drift_metrics_backward_fraction():
    # Target direction = East (bearing=+90deg); wind-to also East.
    scenario = Scenario(
        wind_speed=10.0,
        wind_direction_deg=90.0,
        initial_altitude_m=50.0,
        target_distance_m=100.0,
        target_bearing_deg=90.0,
    )
    wind_I = np.array([0.0, 10.0, 0.0], dtype=float)

    # Ground velocity is +2 m/s East; air-relative velocity is -8 m/s West -> backward drift.
    v_I = np.array([0.0, 2.0, 0.0], dtype=float)
    s0 = State(p_I=np.array([-10.0, 0.0, -10.0]), v_I=v_I, q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=0.0)
    s1 = State(p_I=np.array([-10.0, 2.0, -9.0]), v_I=v_I, q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=1.0)
    s2 = State(p_I=np.array([-10.0, 4.0, -8.0]), v_I=v_I, q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=2.0)

    traj = Trajectory(
        waypoints=[Waypoint(t=0.0, state=s0), Waypoint(t=1.0, state=s1), Waypoint(t=2.0, state=s2)],
        controls=[],
    )

    m = compute_wind_drift_metrics(traj, scenario=scenario, wind_I=wind_I)
    assert float(m["backward_drift_fraction"]) == 1.0
    assert float(m["v_g_along_to_target_mean_mps"]) == 2.0
    assert float(m["v_g_along_wind_mean_mps"]) == 2.0

