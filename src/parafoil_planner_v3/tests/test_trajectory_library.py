import numpy as np

from parafoil_planner_v3.trajectory_library.library_generator import ScenarioConfig, TrajectoryLibraryGenerator
from parafoil_planner_v3.trajectory_library.library_manager import LibraryTrajectory, TrajectoryLibrary
from parafoil_planner_v3.trajectory_library.scenario_features import compute_scenario_features
from parafoil_planner_v3.types import Control, Scenario, State, Target, Trajectory, TrajectoryType, Waypoint, Wind


def _dummy_traj() -> Trajectory:
    s = State(p_I=np.zeros(3), v_I=np.zeros(3), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=0.0)
    return Trajectory(waypoints=[Waypoint(0.0, s)], controls=[Control(0.0, 0.0)], trajectory_type=TrajectoryType.DIRECT)


def test_library_build_and_query_knn():
    trajs = [
        LibraryTrajectory(
            scenario=Scenario(2.0, 90.0, 80.0, 120.0, 0.0),
            trajectory_type=TrajectoryType.DIRECT,
            trajectory=_dummy_traj(),
            cost=1.0,
        ),
        LibraryTrajectory(
            scenario=Scenario(4.0, 180.0, 50.0, 150.0, 90.0),
            trajectory_type=TrajectoryType.DIRECT,
            trajectory=_dummy_traj(),
            cost=2.0,
        ),
    ]
    lib = TrajectoryLibrary(trajs)
    lib.build_index()

    state = State(p_I=np.array([120.0, 0.0, -80.0]), v_I=np.zeros(3), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=0.0)
    target = Target(p_I=np.zeros(3))
    wind = Wind(v_I=np.array([0.0, 2.0, 0.0]))
    feats = compute_scenario_features(state, target, wind)
    dist, idx = lib.query_knn(feats, k=1)
    assert int(np.atleast_1d(idx)[0]) in (0, 1)


def test_template_library_generation_saves_and_loads(tmp_path):
    scenario_cfg = ScenarioConfig(
        wind_speeds=[0.0],
        wind_directions_deg=[0.0],
        initial_altitudes_m=[80.0],
        target_distances_m=[120.0],
        target_bearings_deg=[0.0],
    )
    out = tmp_path / "library.pkl"

    gen = TrajectoryLibraryGenerator()
    lib = gen.generate_library(scenario_cfg, str(out))
    assert out.exists()
    assert len(lib) == 4

    loaded = TrajectoryLibrary.load(str(out))
    assert len(loaded) == 4

    # Query should return a valid index.
    state = State(p_I=np.array([-120.0, 0.0, -80.0]), v_I=np.zeros(3), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=0.0)
    target = Target(p_I=np.zeros(3))
    wind = Wind(v_I=np.zeros(3))
    feats = compute_scenario_features(state, target, wind)
    dist, idx = loaded.query_knn(feats, k=1)
    assert int(np.atleast_1d(idx)[0]) in range(len(loaded))

    entry = loaded[int(np.atleast_1d(idx)[0])]
    traj_meta = entry.trajectory.metadata or {}
    assert "duration_s" in traj_meta
    assert "altitude_loss_m" in traj_meta
    assert "max_bank_deg" in traj_meta
    assert "path_length_m" in traj_meta
    assert "turn_total_rad" in traj_meta
    assert entry.metadata is not None
    assert "trajectory_metrics" in entry.metadata
