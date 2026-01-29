import numpy as np

from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics
from parafoil_planner_v3.offline.e2e import Scenario as E2EScenario
from parafoil_planner_v3.offline.e2e import simulate_one
from parafoil_planner_v3.planner_core import PlannerConfig, PlannerCore
from parafoil_planner_v3.trajectory_library.library_manager import LibraryTrajectory, TrajectoryLibrary
from parafoil_planner_v3.types import Control, Scenario, State, Target, Trajectory, TrajectoryType, Waypoint, Wind


def test_planner_core_library_mode_returns_adapted_trajectory():
    dyn = SixDOFDynamics()

    # Library with one simple trajectory in target-centered coordinates.
    # Scenario: distance=120m bearing=0deg (state->target) -> start at (-120,0) in target-centered NED.
    s0 = State(
        p_I=np.array([-120.0, 0.0, -80.0]),
        v_I=np.array([3.0, 0.0, 0.0]),
        q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
        w_B=np.zeros(3),
        t=0.0,
    )
    lib_traj = Trajectory(
        waypoints=[Waypoint(0.0, s0)],
        controls=[Control(0.2, 0.2)],
        trajectory_type=TrajectoryType.DIRECT,
    )
    lib = TrajectoryLibrary(
        [
            LibraryTrajectory(
                scenario=Scenario(2.0, 90.0, 80.0, 120.0, 0.0),
                trajectory_type=TrajectoryType.DIRECT,
                trajectory=lib_traj,
                cost=0.0,
            )
        ]
    )
    lib.build_index()

    cfg = PlannerConfig(use_library=True, enable_gpm_fine_tuning=False, Vh_min=0.0)
    planner = PlannerCore(dynamics=dyn, config=cfg, library=lib)

    state = State(p_I=np.array([10.0, -5.0, -80.0]), v_I=np.array([4.0, 0.0, 1.0]), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=0.0)
    target = Target(p_I=np.array([0.0, 0.0, 0.0]))
    wind = Wind(v_I=np.array([0.0, 2.0, 0.0]))

    traj, info = planner.plan(state, target, wind)
    assert info.success
    assert traj.waypoints
    assert np.allclose(traj.waypoints[0].state.position_xy, state.position_xy)


def test_library_selection_respects_no_fly_circle():
    dyn = SixDOFDynamics()

    def _make_traj(points: list[tuple[float, float]]) -> Trajectory:
        waypoints = []
        controls = []
        for i, (x, y) in enumerate(points):
            s = State(
                p_I=np.array([x, y, -80.0]),
                v_I=np.array([3.0, 0.0, 0.0]),
                q_IB=np.array([1.0, 0.0, 0.0, 0.0]),
                w_B=np.zeros(3),
                t=float(i),
            )
            waypoints.append(Waypoint(t=float(i), state=s))
            controls.append(Control(0.2, 0.2))
        return Trajectory(waypoints=waypoints, controls=controls, trajectory_type=TrajectoryType.DIRECT)

    # Candidate A passes through (-60,0) which we will mark as no-fly.
    traj_a = _make_traj([(-120.0, 0.0), (-60.0, 0.0), (0.0, 0.0)])
    traj_b = _make_traj([(-120.0, 0.0), (-60.0, 30.0), (0.0, 0.0)])
    traj_b = Trajectory(waypoints=traj_b.waypoints, controls=traj_b.controls, trajectory_type=TrajectoryType.S_TURN)

    scenario = Scenario(0.0, 0.0, 80.0, 120.0, 0.0)
    lib = TrajectoryLibrary(
        [
            LibraryTrajectory(scenario=scenario, trajectory_type=TrajectoryType.DIRECT, trajectory=traj_a),
            LibraryTrajectory(scenario=scenario, trajectory_type=TrajectoryType.S_TURN, trajectory=traj_b),
        ]
    )
    lib.build_index()

    cfg = PlannerConfig(
        use_library=True,
        enable_gpm_fine_tuning=False,
        k_neighbors=2,
        no_fly_circles=(( -60.0, 0.0, 5.0, 0.0),),
    )
    planner = PlannerCore(dynamics=dyn, config=cfg, library=lib)

    state = State(p_I=np.array([-120.0, 0.0, -80.0]), v_I=np.array([3.0, 0.0, 0.0]), q_IB=np.array([1.0, 0.0, 0.0, 0.0]), w_B=np.zeros(3), t=0.0)
    target = Target(p_I=np.array([0.0, 0.0, 0.0]))
    wind = Wind(v_I=np.zeros(3))

    traj, info = planner.plan(state, target, wind)
    assert info.success
    assert traj.trajectory_type == TrajectoryType.S_TURN


def test_offline_full_mission_no_wind_simplified_reaches_target():
    s = E2EScenario(altitude_m=80.0, distance_m=120.0, bearing_deg=0.0, wind_speed_mps=0.0, wind_direction_deg=0.0)
    out = simulate_one(s, dynamics_mode="simplified", record_history=False)
    m = out["metrics"]
    assert m["final_phase"] == "landed"
    assert float(m["landing_error_m"]) < 10.0


def test_offline_full_mission_with_crosswind_simplified_runs():
    # Crosswind case where the baseline controller still converges reliably.
    s = E2EScenario(altitude_m=80.0, distance_m=120.0, bearing_deg=0.0, wind_speed_mps=2.0, wind_direction_deg=105.0)
    out = simulate_one(s, dynamics_mode="simplified", record_history=False)
    m = out["metrics"]
    assert m["final_phase"] == "landed"
    assert float(m["landing_error_m"]) < 10.0


def test_flare_touchdown_brake_reduces_touchdown_vertical_speed_6dof():
    # Keep altitude low so this test stays fast, but still exercises the 6-DOF flare closed loop.
    s = E2EScenario(altitude_m=8.0, distance_m=10.0, bearing_deg=0.0, wind_speed_mps=0.0, wind_direction_deg=0.0)
    base = simulate_one(
        s,
        dynamics_mode="6dof",
        record_history=False,
        max_time_s=30.0,
        flare_touchdown_altitude_m=-1.0,
        flare_ramp_time_s=0.1,
        flare_mode="touchdown_brake",
    )
    flare = simulate_one(
        s,
        dynamics_mode="6dof",
        record_history=False,
        max_time_s=30.0,
        flare_touchdown_altitude_m=0.2,
        flare_ramp_time_s=0.1,
        flare_mode="touchdown_brake",
    )

    v_base = float(base["metrics"].get("touchdown_vertical_velocity_mps", base["metrics"]["vertical_velocity_mps"]))
    v_flare = float(flare["metrics"].get("touchdown_vertical_velocity_mps", flare["metrics"]["vertical_velocity_mps"]))
    assert v_flare < v_base - 0.02


def test_full_mission_with_wind_and_flare_6dof():
    s = E2EScenario(altitude_m=40.0, distance_m=80.0, bearing_deg=0.0, wind_speed_mps=2.0, wind_direction_deg=90.0)
    base = simulate_one(
        s,
        dynamics_mode="6dof",
        record_history=False,
        max_time_s=140.0,
        flare_touchdown_altitude_m=-1.0,
        flare_ramp_time_s=0.1,
        flare_mode="touchdown_brake",
    )
    flare = simulate_one(
        s,
        dynamics_mode="6dof",
        record_history=False,
        max_time_s=140.0,
        flare_touchdown_altitude_m=0.2,
        flare_ramp_time_s=0.1,
        flare_mode="touchdown_brake",
    )

    assert base["metrics"]["final_phase"] == "landed"
    assert flare["metrics"]["final_phase"] == "landed"
    assert float(flare["metrics"]["landing_error_m"]) < 10.0

    v_base = float(base["metrics"].get("touchdown_vertical_velocity_mps", base["metrics"]["vertical_velocity_mps"]))
    v_flare = float(flare["metrics"].get("touchdown_vertical_velocity_mps", flare["metrics"]["vertical_velocity_mps"]))
    assert v_flare < v_base - 0.05
