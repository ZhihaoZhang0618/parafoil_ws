import numpy as np

from parafoil_planner_v3.environment import NoFlyCircle, NoFlyPolygon
from parafoil_planner_v3.landing_site_selector import (
    LandingSiteSelector,
    LandingSiteSelectorConfig,
    ReachabilityConfig,
    RiskGrid,
    RiskLayer,
    RiskMapAggregator,
)
from parafoil_planner_v3.types import State, Target, Wind


def _make_state(altitude_m: float = 50.0) -> State:
    return State(
        p_I=np.array([0.0, 0.0, -altitude_m], dtype=float),
        v_I=np.zeros(3, dtype=float),
        q_IB=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        w_B=np.zeros(3, dtype=float),
        t=0.0,
    )


def test_risk_grid_npz_interpolation(tmp_path):
    risk = np.array([[0.0, 1.0], [2.0, 3.0]], dtype=float)
    path = tmp_path / "risk_grid.npz"
    np.savez(path, risk_map=risk, origin_n=0.0, origin_e=0.0, resolution_m=1.0)

    grid = RiskGrid.from_file(path, oob_value=0.9)
    assert np.isclose(grid.risk_at(0.0, 0.0), 0.0)
    assert np.isclose(grid.risk_at(1.0, 0.0), 2.0)
    assert np.isclose(grid.risk_at(0.5, 0.5), 1.5)
    assert np.isclose(grid.risk_at(5.0, 5.0), 0.9)


def test_risk_map_aggregator_sum():
    grid_a = RiskGrid(origin_n=0.0, origin_e=0.0, resolution_m=1.0, risk_map=np.array([[0.2]]))
    grid_b = RiskGrid(origin_n=0.0, origin_e=0.0, resolution_m=1.0, risk_map=np.array([[0.5]]))
    agg = RiskMapAggregator(
        layers=[
            RiskLayer(name="a", weight=2.0, grid=grid_a),
            RiskLayer(name="b", weight=1.0, grid=grid_b),
        ],
        clip_min=0.0,
        clip_max=1.0,
    )
    total, details = agg.risk(0.0, 0.0)
    assert np.isclose(total, 2.0 * 0.2 + 1.0 * 0.5)
    assert details["a"] == 0.2
    assert details["b"] == 0.5


def test_reachability_margin():
    cfg = LandingSiteSelectorConfig(
        enabled=True,
        grid_resolution_m=10.0,
        search_radius_m=40.0,
        max_candidates=200,
        random_seed=0,
        w_risk=0.0,
        w_distance=1.0,
        w_reach_margin=0.0,
        nofly_buffer_m=0.0,
        nofly_weight=0.0,
        snap_to_terrain=False,
        reachability=ReachabilityConfig(brake=0.2, min_time_s=2.0, max_time_s=200.0, wind_margin_mps=0.2),
    )
    selector = LandingSiteSelector(cfg)
    state = _make_state(altitude_m=50.0)
    tgo = selector._time_to_land(state, terrain=None, north_m=0.0, east_m=0.0)

    wind_calm = Wind(v_I=np.array([0.0, 0.0, 0.0], dtype=float))
    ok, margin, _ = selector._reachable(state, wind_calm, np.array([40.0, 0.0], dtype=float), tgo)
    assert ok
    assert margin > 0.0

    wind_strong = Wind(v_I=np.array([6.0, 0.0, 0.0], dtype=float))
    ok, margin, _ = selector._reachable(state, wind_strong, np.array([40.0, 0.0], dtype=float), tgo)
    assert not ok
    assert margin < 0.0


def test_selection_prefers_desired_when_safe():
    cfg = LandingSiteSelectorConfig(
        enabled=True,
        grid_resolution_m=10.0,
        search_radius_m=40.0,
        max_candidates=200,
        random_seed=0,
        w_risk=0.0,
        w_distance=1.0,
        w_reach_margin=0.0,
        nofly_buffer_m=0.0,
        nofly_weight=0.0,
        snap_to_terrain=False,
        reachability=ReachabilityConfig(brake=0.2, min_time_s=2.0, max_time_s=200.0, wind_margin_mps=0.2),
    )
    selector = LandingSiteSelector(cfg)
    state = _make_state(altitude_m=50.0)
    desired = Target(p_I=np.array([20.0, 0.0, 0.0], dtype=float))
    wind = Wind(v_I=np.array([0.0, 0.0, 0.0], dtype=float))

    selection = selector.select(state, desired, wind, terrain=None, no_fly_circles=[], no_fly_polygons=[])
    assert selection.reason == "ok"
    assert np.allclose(selection.target.position_xy, desired.position_xy)
    assert np.isclose(selection.distance_to_desired_m, 0.0)


def test_nofly_hard_violation_blocks_all_candidates():
    cfg = LandingSiteSelectorConfig(
        enabled=True,
        grid_resolution_m=10.0,
        search_radius_m=20.0,
        max_candidates=200,
        random_seed=0,
        w_risk=0.0,
        w_distance=1.0,
        w_reach_margin=0.0,
        nofly_buffer_m=0.0,
        nofly_weight=1.0,
        snap_to_terrain=False,
        reachability=ReachabilityConfig(brake=0.2, min_time_s=2.0, max_time_s=200.0, wind_margin_mps=0.2),
    )
    selector = LandingSiteSelector(cfg)
    state = _make_state(altitude_m=50.0)
    desired = Target(p_I=np.array([10.0, 0.0, 0.0], dtype=float))
    wind = Wind(v_I=np.array([0.0, 0.0, 0.0], dtype=float))

    nofly = [NoFlyCircle(center_n=0.0, center_e=0.0, radius_m=50.0, clearance_m=0.0)]
    selection = selector.select(state, desired, wind, terrain=None, no_fly_circles=nofly, no_fly_polygons=[])
    assert selection.reason == "no_reachable_candidate"


def test_nofly_soft_penalty_pushes_away():
    cfg = LandingSiteSelectorConfig(
        enabled=True,
        grid_resolution_m=5.0,
        search_radius_m=25.0,
        max_candidates=500,
        random_seed=0,
        w_risk=1.0,
        w_distance=0.1,
        w_reach_margin=0.0,
        nofly_buffer_m=10.0,
        nofly_weight=10.0,
        snap_to_terrain=False,
        reachability=ReachabilityConfig(brake=0.2, min_time_s=2.0, max_time_s=200.0, wind_margin_mps=0.2),
    )
    selector = LandingSiteSelector(cfg)
    state = _make_state(altitude_m=50.0)
    desired = Target(p_I=np.array([10.0, 0.0, 0.0], dtype=float))
    wind = Wind(v_I=np.array([0.0, 0.0, 0.0], dtype=float))

    nofly = [NoFlyCircle(center_n=0.0, center_e=0.0, radius_m=5.0, clearance_m=0.0)]
    selection = selector.select(state, desired, wind, terrain=None, no_fly_circles=nofly, no_fly_polygons=[])
    assert selection.reason == "ok"
    assert selection.distance_to_desired_m > 1.0


def test_nofly_polygon_hard_violation():
    cfg = LandingSiteSelectorConfig(
        enabled=True,
        grid_resolution_m=10.0,
        search_radius_m=20.0,
        max_candidates=200,
        random_seed=0,
        w_risk=0.0,
        w_distance=1.0,
        w_reach_margin=0.0,
        nofly_buffer_m=0.0,
        nofly_weight=1.0,
        snap_to_terrain=False,
        reachability=ReachabilityConfig(brake=0.2, min_time_s=2.0, max_time_s=200.0, wind_margin_mps=0.2),
    )
    selector = LandingSiteSelector(cfg)
    state = _make_state(altitude_m=50.0)
    desired = Target(p_I=np.array([5.0, 5.0, 0.0], dtype=float))
    wind = Wind(v_I=np.array([0.0, 0.0, 0.0], dtype=float))

    poly = NoFlyPolygon(vertices=np.array([[-50.0, -50.0], [50.0, -50.0], [50.0, 50.0], [-50.0, 50.0]]))
    selection = selector.select(state, desired, wind, terrain=None, no_fly_circles=[], no_fly_polygons=[poly])
    assert selection.reason == "no_reachable_candidate"


def test_desired_target_unreachable_wind_reason():
    cfg = LandingSiteSelectorConfig(
        enabled=True,
        grid_resolution_m=10.0,
        search_radius_m=60.0,
        max_candidates=400,
        random_seed=0,
        w_risk=0.0,
        w_distance=1.0,
        w_reach_margin=0.0,
        w_energy=0.0,
        nofly_buffer_m=0.0,
        nofly_weight=0.0,
        snap_to_terrain=False,
        min_progress_mps=0.0,
        reachability=ReachabilityConfig(brake=0.2, min_time_s=2.0, max_time_s=200.0, wind_margin_mps=0.2),
    )
    selector = LandingSiteSelector(cfg)
    state = _make_state(altitude_m=80.0)
    desired = Target(p_I=np.array([60.0, 0.0, 0.0], dtype=float))
    v_air, _ = selector._polar.interpolate(cfg.reachability.brake)
    wind = Wind(v_I=np.array([-float(v_air) - 0.5, 0.0, 0.0], dtype=float))

    selection = selector.select(state, desired, wind, terrain=None, no_fly_circles=[], no_fly_polygons=[])
    assert selection.reason == "unreachable_wind"
    assert selection.metadata.get("desired_v_g_along_max_mps", 1.0) <= cfg.min_progress_mps + 1e-6
