from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.environment import (
    FlatTerrain,
    GridTerrain,
    NoFlyCircle,
    NoFlyPolygon,
    PlaneTerrain,
    TerrainModel,
    load_no_fly_polygons,
)
from parafoil_planner_v3.landing_site_selector import LandingSiteSelection, LandingSiteSelector
from parafoil_planner_v3.dynamics.parafoil_6dof import SixDOFDynamics
from parafoil_planner_v3.optimization import GPMCollocation, GPMSolver, SolverInfo
from parafoil_planner_v3.optimization.solver_interface import SolverConfig
from parafoil_planner_v3.trajectory_library.library_manager import LibraryTrajectory, TrajectoryLibrary
from parafoil_planner_v3.trajectory_library.scenario_features import compute_scenario_features
from parafoil_planner_v3.trajectory_library.trajectory_adapter import adapt_trajectory
from parafoil_planner_v3.types import Control, State, Target, Trajectory, TrajectoryType, Waypoint, Wind
from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy, wrap_pi


@dataclass(frozen=True)
class PlannerConfig:
    # GPM options
    gpm_num_nodes: int = 20
    gpm_scheme: str = "LGL"
    tf_guess: float = 30.0
    enable_warm_start: bool = True
    max_constraint_violation_for_accept: float = 0.5

    # Solver options / weights (mapped into SolverConfig)
    solver_method: str = "SLSQP"
    solver_maxiter: int = 300
    solver_ftol: float = 1e-6
    solver_max_solve_time_s: float = 0.0

    w_terminal_pos: float = 50.0
    w_terminal_vel: float = 1.0
    w_running_u: float = 0.05
    w_running_du: float = 0.01
    w_running_w: float = 0.0
    w_running_wdot: float = 0.0
    w_running_energy: float = 0.0

    tf_min: float = 1.0
    tf_max: float = 120.0
    delta_rate_max: float = 3.0
    Vh_min: float = 0.5
    Vh_max: float = 8.0
    roll_max_deg: float = 60.0
    yaw_rate_max_deg: float = 120.0

    # Path constraints (optional)
    enforce_terminal_upwind_heading: bool = False
    terminal_heading_tol_deg: float = 5.0

    terrain_type: str = "flat"  # flat|plane
    terrain_height0_m: float = 0.0
    terrain_slope_n: float = 0.0
    terrain_slope_e: float = 0.0
    terrain_clearance_m: float = 0.0
    terrain_file: str = ""

    # Each entry: (center_n, center_e, radius_m, clearance_m)
    no_fly_circles: tuple[tuple[float, float, float, float], ...] = ()
    # Polygon constraints (optional)
    no_fly_polygons: tuple[NoFlyPolygon, ...] = ()
    no_fly_polygons_file: str = ""

    # Library options
    use_library: bool = False
    k_neighbors: int = 5
    library_coarse_k_neighbors: int = 5
    library_fine_k_neighbors: int = 5
    library_require_coarse_match: bool = True
    library_fallback_to_coarse: bool = True
    enable_gpm_fine_tuning: bool = True
    fine_tuning_trigger_m: float = 10.0
    library_terminal_pos_tol_m: float = 20.0
    library_cost_w_terminal_pos: float = 1.0
    library_cost_w_heading: float = 0.2
    library_cost_w_control: float = 0.05
    library_cost_w_time: float = 0.02
    library_cost_w_path_length: float = 0.0
    # Library safety gates (0 disables a check)
    library_max_feature_distance: float = 0.0
    library_max_altitude_error_m: float = 0.0
    library_max_distance_error_m: float = 0.0
    library_max_bearing_error_deg: float = 0.0
    library_max_wind_speed_error_mps: float = 0.0
    library_max_rel_wind_error_deg: float = 0.0
    library_min_scale_xy: float = 0.0
    library_max_scale_xy: float = 0.0
    library_min_scale_z: float = 0.0
    library_max_scale_z: float = 0.0
    # Skip library matching when wind makes the target line fundamentally untrackable.
    # This is a "sanity gate" to avoid trusting KNN+adaptation in extreme wind.
    library_skip_if_unreachable_wind: bool = True
    library_min_track_ground_speed_mps: float = 0.2

    # Fallback trajectory generation
    fallback_dt: float = 1.0
    fallback_brake: float = 0.2
    fallback_wind_correction_gain: float = 1.0
    fallback_min_ground_speed: float = 0.5

    # Headwind / aimpoint planning
    headwind_enable: bool = True
    headwind_wind_ratio_trigger: float = 0.8
    headwind_max_aim_offset_m: float = 200.0
    headwind_min_progress_mps: float = 0.0
    headwind_aimpoint_nofly_projection: bool = True


class PlannerCore:
    def __init__(
        self,
        dynamics: SixDOFDynamics,
        config: Optional[PlannerConfig] = None,
        library: Optional[TrajectoryLibrary] = None,
        library_coarse: Optional[TrajectoryLibrary] = None,
        library_fine: Optional[TrajectoryLibrary] = None,
        landing_site_selector: LandingSiteSelector | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.config = config or PlannerConfig()
        self._polar = PolarTable()
        self.library = library
        self.library_coarse = library_coarse
        self.library_fine = library_fine or library
        self.landing_site_selector = landing_site_selector
        self.last_site_selection: LandingSiteSelection | None = None
        self.last_aimpoint_target: Target | None = None
        self._terrain_cache: TerrainModel | None = None
        self._terrain_cache_key: str | None = None

        self.gpm = GPMCollocation(N=int(self.config.gpm_num_nodes), scheme=str(self.config.gpm_scheme))
        solver_cfg = SolverConfig(
            method=str(self.config.solver_method),
            maxiter=int(self.config.solver_maxiter),
            ftol=float(self.config.solver_ftol),
            max_solve_time_s=float(self.config.solver_max_solve_time_s),
            w_terminal_pos=float(self.config.w_terminal_pos),
            w_terminal_vel=float(self.config.w_terminal_vel),
            w_running_u=float(self.config.w_running_u),
            w_running_du=float(self.config.w_running_du),
            w_running_w=float(self.config.w_running_w),
            w_running_wdot=float(self.config.w_running_wdot),
            w_running_energy=float(self.config.w_running_energy),
            tf_min=float(self.config.tf_min),
            tf_max=float(self.config.tf_max),
            delta_rate_max=float(self.config.delta_rate_max),
            Vh_min=float(self.config.Vh_min),
            Vh_max=float(self.config.Vh_max),
            roll_max_rad=float(np.deg2rad(self.config.roll_max_deg)),
            yaw_rate_max=float(np.deg2rad(self.config.yaw_rate_max_deg)),
            terminal_heading_tol_rad=float(np.deg2rad(self.config.terminal_heading_tol_deg)),
        )
        # placeholder f; updated per-plan to capture wind.
        self.solver = GPMSolver(f=lambda x, u, t: np.zeros_like(x), gpm=self.gpm, config=solver_cfg)

    def _wind_trackability_diag(self, state: State, target: Target, wind: Wind) -> dict:
        """
        Best-effort diagnostic for "can we make progress toward target given wind?".

        Uses still-air max airspeed (brake=0) as an optimistic bound.
        """
        rel = target.position_xy - state.position_xy
        dist = float(np.linalg.norm(rel))
        wind_xy = np.asarray(wind.v_I[:2], dtype=float)

        V_air_max, _ = self._polar.interpolate(0.0)
        if dist < 1e-6:
            return {
                "distance_m": 0.0,
                "V_air_max_mps": float(V_air_max),
                "wind_speed_mps": float(np.linalg.norm(wind_xy)),
                "headwind_mps": 0.0,
                "crosswind_mps": 0.0,
                "cross_ok": True,
                "v_track_max_mps": float(V_air_max),
            }

        dir_hat = rel / dist
        head = float(np.dot(wind_xy, dir_hat))  # >0 tailwind (pushes toward target)
        cross_vec = wind_xy - head * dir_hat
        cross = float(np.linalg.norm(cross_vec))

        if cross >= float(V_air_max):
            # Can't cancel crosswind to stay on a desired track; reaching an exact target becomes ill-conditioned.
            return {
                "distance_m": float(dist),
                "V_air_max_mps": float(V_air_max),
                "wind_speed_mps": float(np.linalg.norm(wind_xy)),
                "headwind_mps": float(head),
                "crosswind_mps": float(cross),
                "cross_ok": False,
                "v_track_max_mps": float("nan"),
            }

        v_track = float(np.sqrt(max(float(V_air_max) * float(V_air_max) - cross * cross, 0.0)) + head)
        return {
            "distance_m": float(dist),
            "V_air_max_mps": float(V_air_max),
            "wind_speed_mps": float(np.linalg.norm(wind_xy)),
            "headwind_mps": float(head),
            "crosswind_mps": float(cross),
            "cross_ok": True,
            "v_track_max_mps": float(v_track),
        }

    def _load_terrain(self) -> TerrainModel | None:
        terrain_type = str(self.config.terrain_type).strip().lower()
        if terrain_type == "plane":
            return PlaneTerrain(
                height0_m=float(self.config.terrain_height0_m),
                slope_n=float(self.config.terrain_slope_n),
                slope_e=float(self.config.terrain_slope_e),
            )
        if terrain_type == "flat":
            if float(self.config.terrain_clearance_m) > 0.0 or float(self.config.terrain_height0_m) != 0.0:
                return FlatTerrain(height0_m=float(self.config.terrain_height0_m))
            return None
        if terrain_type == "grid":
            path = str(self.config.terrain_file).strip()
            if not path:
                raise ValueError("terrain_type=grid requires terrain_file")
            if self._terrain_cache_key != path or self._terrain_cache is None:
                self._terrain_cache = GridTerrain.from_file(path)
                self._terrain_cache_key = path
            return self._terrain_cache
        return None

    def _build_no_fly(self) -> tuple[list[NoFlyCircle], list[NoFlyPolygon]]:
        circles = [
            NoFlyCircle(center_n=float(c[0]), center_e=float(c[1]), radius_m=float(c[2]), clearance_m=float(c[3]))
            for c in (self.config.no_fly_circles or ())
        ]
        polygons: list[NoFlyPolygon] = []
        for poly in (self.config.no_fly_polygons or ()):
            if isinstance(poly, NoFlyPolygon):
                polygons.append(poly)
            else:
                polygons.append(NoFlyPolygon(vertices=np.asarray(poly, dtype=float)))
        if str(self.config.no_fly_polygons_file).strip():
            polygons.extend(load_no_fly_polygons(str(self.config.no_fly_polygons_file)))
        return circles, polygons

    def _terminal_heading_hat(self, wind: Wind) -> Optional[np.ndarray]:
        if not bool(self.config.enforce_terminal_upwind_heading):
            return None
        w_xy = wind.v_I[:2]
        n = float(np.linalg.norm(w_xy))
        if n <= 1e-6:
            return None
        return (-w_xy / n).astype(float)

    def _terminal_heading_error(self, state: State, wind: Wind) -> float:
        hat = self._terminal_heading_hat(wind)
        if hat is None:
            return 0.0
        v_xy = state.v_I[:2]
        speed = float(np.linalg.norm(v_xy))
        if speed > 0.3:
            heading = float(np.arctan2(v_xy[1], v_xy[0]))
        else:
            _, _, yaw = quat_to_rpy(state.q_IB)
            heading = float(yaw)
        desired = float(np.arctan2(hat[1], hat[0]))
        return float(abs(wrap_pi(heading - desired)))

    def _estimate_time_to_land(
        self,
        state: State,
        target_xy: np.ndarray,
        terrain: TerrainModel | None,
        brake: float,
    ) -> tuple[float, float, float]:
        v_air, sink = self._polar.interpolate(float(brake))
        terrain_h = float(terrain.height_m(float(target_xy[0]), float(target_xy[1]))) if terrain is not None else 0.0
        h_agl = float(state.altitude - terrain_h - float(self.config.terrain_clearance_m))
        if h_agl <= 1e-6 or sink <= 1e-6:
            return -1.0, float(v_air), float(sink)
        return float(h_agl / sink), float(v_air), float(sink)

    @staticmethod
    def _ground_speed_along_max(state: State, target_xy: np.ndarray, wind: Wind, v_air: float) -> float:
        rel = np.asarray(target_xy, dtype=float) - state.position_xy
        dist = float(np.linalg.norm(rel))
        if dist <= 1e-6:
            return float(v_air)
        else:
            d_hat = rel / dist
        wind_xy = np.asarray(wind.v_I[:2], dtype=float)
        return float(v_air) + float(np.dot(wind_xy, d_hat))

    @staticmethod
    def _min_signed_distance_to_nofly(
        north_m: float,
        east_m: float,
        no_fly_circles: list[NoFlyCircle],
        no_fly_polygons: list[NoFlyPolygon],
    ) -> float:
        min_dist = float("inf")
        for zone in no_fly_circles:
            min_dist = min(min_dist, float(zone.signed_distance_m(north_m, east_m)))
        for zone in no_fly_polygons:
            min_dist = min(min_dist, float(zone.signed_distance_m(north_m, east_m)))
        return float(min_dist)

    def _nofly_buffer_m(self) -> float:
        if self.landing_site_selector is None:
            return 0.0
        return float(self.landing_site_selector.config.nofly_buffer_m)

    def _project_out_of_nofly(
        self,
        aim_xy: np.ndarray,
        wind_hat: np.ndarray,
        no_fly_circles: list[NoFlyCircle],
        no_fly_polygons: list[NoFlyPolygon],
    ) -> tuple[np.ndarray, bool]:
        d0 = self._min_signed_distance_to_nofly(float(aim_xy[0]), float(aim_xy[1]), no_fly_circles, no_fly_polygons)
        if d0 >= 0.0:
            return aim_xy, True
        step = float(max(self._nofly_buffer_m(), 2.0))
        max_extra = float(max(self.config.headwind_max_aim_offset_m, 0.0) + 2.0 * step)
        traveled = 0.0
        prev = aim_xy.copy()
        prev_d = float(d0)
        current = aim_xy.copy()
        while traveled <= max_extra:
            current = current - wind_hat * step
            traveled += step
            d = self._min_signed_distance_to_nofly(float(current[0]), float(current[1]), no_fly_circles, no_fly_polygons)
            if d >= 0.0:
                if prev_d < 0.0 and (prev_d - d) != 0.0:
                    t = prev_d / (prev_d - d)
                    projected = prev + (current - prev) * t
                else:
                    projected = current
                return projected.astype(float), True
            prev = current
            prev_d = float(d)
        return current.astype(float), False

    def _compute_aimpoint_target(
        self,
        state: State,
        touchdown_target: Target,
        wind: Wind,
        terrain: TerrainModel | None,
        no_fly_circles: list[NoFlyCircle],
        no_fly_polygons: list[NoFlyPolygon],
        selection: LandingSiteSelection | None,
    ) -> tuple[Target | None, dict]:
        meta: dict = {"aimpoint_used": False}
        if not bool(self.config.headwind_enable):
            meta["aimpoint_reason"] = "disabled"
            return None, meta

        wind_xy = np.asarray(wind.v_I[:2], dtype=float)
        wind_speed = float(np.linalg.norm(wind_xy))

        v_air = float("nan")
        sink = float("nan")
        if selection is not None and selection.metadata:
            v_air = float(selection.metadata.get("touchdown_v_air_mps", float("nan")))
            sink = float(selection.metadata.get("touchdown_sink_mps", float("nan")))

        if not np.isfinite(v_air) or not np.isfinite(sink):
            if self.landing_site_selector is not None:
                brake = float(self.landing_site_selector.config.reachability.brake)
            else:
                brake = float(self.config.fallback_brake)
            _, v_air, sink = self._estimate_time_to_land(state, touchdown_target.position_xy, terrain, brake)
            v_air = float(v_air)
            sink = float(sink)

        tgo = -1.0
        if selection is not None and float(selection.time_to_land_s) > 0.0:
            tgo = float(selection.time_to_land_s)
        else:
            if self.landing_site_selector is not None:
                brake = float(self.landing_site_selector.config.reachability.brake)
            else:
                brake = float(self.config.fallback_brake)
            tgo, _, _ = self._estimate_time_to_land(state, touchdown_target.position_xy, terrain, brake)

        meta.update(
            {
                "aimpoint_tgo_s": float(tgo),
                "aimpoint_v_air_mps": float(v_air),
                "aimpoint_sink_mps": float(sink),
                "aimpoint_wind_speed_mps": float(wind_speed),
            }
        )

        if tgo <= 0.0:
            meta["aimpoint_reason"] = "invalid_tgo"
            return None, meta

        drift_mag = float(wind_speed * tgo)
        max_offset = float(max(self.config.headwind_max_aim_offset_m, 0.0))
        offset_mag = drift_mag
        if max_offset > 1e-6:
            offset_mag = float(min(drift_mag, max_offset))

        wind_hat = None
        if wind_speed > 1e-6:
            wind_hat = wind_xy / wind_speed
            aim_xy = touchdown_target.position_xy - wind_hat * offset_mag
        else:
            aim_xy = touchdown_target.position_xy.copy()
            offset_mag = 0.0

        meta.update(
            {
                "aimpoint_offset_m": float(offset_mag),
                "aimpoint_drift_m": float(drift_mag),
                "aimpoint_n": float(aim_xy[0]),
                "aimpoint_e": float(aim_xy[1]),
            }
        )

        if no_fly_circles or no_fly_polygons:
            signed_dist = self._min_signed_distance_to_nofly(float(aim_xy[0]), float(aim_xy[1]), no_fly_circles, no_fly_polygons)
            meta["aimpoint_nofly_signed_dist_m"] = float(signed_dist)
            if signed_dist < 0.0:
                meta["aimpoint_in_nofly"] = True
                if bool(self.config.headwind_aimpoint_nofly_projection) and wind_hat is not None:
                    projected_xy, ok = self._project_out_of_nofly(aim_xy, wind_hat, no_fly_circles, no_fly_polygons)
                    meta["aimpoint_projected"] = True
                    meta["aimpoint_projection_ok"] = bool(ok)
                    aim_xy = projected_xy if ok else aim_xy
                    if not ok:
                        meta["aimpoint_reason"] = "nofly_projection_failed"
                        return None, meta
                    meta["aimpoint_n"] = float(aim_xy[0])
                    meta["aimpoint_e"] = float(aim_xy[1])
                else:
                    meta["aimpoint_reason"] = "aimpoint_in_nofly"
                    return None, meta
            else:
                meta["aimpoint_in_nofly"] = False

        meta["aimpoint_offset_m"] = float(np.linalg.norm(aim_xy - touchdown_target.position_xy))

        v_g_along_max = self._ground_speed_along_max(state, aim_xy, wind, v_air)
        meta["aimpoint_v_g_along_max_mps"] = float(v_g_along_max)

        wind_ratio = float(wind_speed / max(v_air, 1e-6))
        meta["headwind_wind_ratio"] = float(wind_ratio)

        trigger = float(self.config.headwind_wind_ratio_trigger)
        use_aimpoint = bool(wind_ratio >= trigger) if trigger > 0.0 else True
        if selection is not None and str(selection.reason) == "unreachable_wind":
            use_aimpoint = True
            meta["aimpoint_trigger"] = "unreachable_wind"
        else:
            meta["aimpoint_trigger"] = "wind_ratio" if use_aimpoint else "below_trigger"

        if v_g_along_max <= float(self.config.headwind_min_progress_mps):
            use_aimpoint = False
            meta["aimpoint_blocked_reason"] = "unreachable_wind"

        meta["aimpoint_used"] = bool(use_aimpoint)
        return Target(p_I=np.array([float(aim_xy[0]), float(aim_xy[1]), float(touchdown_target.p_I[2])], dtype=float)), meta

    def _check_trajectory_feasible(
        self,
        traj: Trajectory,
        target: Target,
        wind: Wind,
        terrain: TerrainModel | None,
        no_fly_circles: list[NoFlyCircle],
        no_fly_polygons: list[NoFlyPolygon],
    ) -> tuple[bool, str]:
        if not traj.waypoints:
            return False, "empty"
        # Terminal position tolerance
        final = traj.waypoints[-1].state
        terminal_err = float(np.linalg.norm(final.position_xy - target.position_xy))
        if terminal_err > float(self.config.library_terminal_pos_tol_m):
            return False, f"terminal_error>{terminal_err:.2f}"

        # Check path constraints
        for wp in traj.waypoints:
            st = wp.state
            v_xy = st.v_I[:2]
            Vh = float(np.linalg.norm(v_xy))
            if Vh < float(self.config.Vh_min) or Vh > float(self.config.Vh_max):
                return False, "Vh_out_of_bounds"
            roll, _, _ = quat_to_rpy(st.q_IB)
            if abs(float(roll)) > float(np.deg2rad(self.config.roll_max_deg)):
                return False, "roll_out_of_bounds"
            yaw_rate = float(st.w_B[2])
            if abs(yaw_rate) > float(np.deg2rad(self.config.yaw_rate_max_deg)):
                return False, "yaw_rate_out_of_bounds"

            if terrain is not None:
                terrain_h = float(terrain.height_m(float(st.p_I[0]), float(st.p_I[1])))
                altitude = float(st.altitude)
                if altitude < terrain_h + float(self.config.terrain_clearance_m):
                    return False, "terrain_violation"

            for zone in no_fly_circles:
                if zone.signed_distance_m(float(st.p_I[0]), float(st.p_I[1])) < 0.0:
                    return False, "no_fly_circle"

            for zone in no_fly_polygons:
                if zone.signed_distance_m(float(st.p_I[0]), float(st.p_I[1])) < 0.0:
                    return False, "no_fly_polygon"

        # Control bounds / rate
        controls = traj.controls or []
        if controls:
            rate_limit = float(self.config.delta_rate_max)
            meta = traj.metadata or {}
            # For adapted library trajectories, keep control-rate feasibility consistent with
            # the original library time scale (undo time scaling).
            if meta.get("adapted") and meta.get("scale_t") is not None:
                try:
                    scale_t = float(meta.get("scale_t"))
                except (TypeError, ValueError):
                    scale_t = 1.0
                if scale_t > 1e-6:
                    rate_limit = rate_limit / scale_t
            last = controls[0]
            for k in range(len(controls)):
                ctrl = controls[k]
                if not (0.0 <= ctrl.delta_L <= 1.0 and 0.0 <= ctrl.delta_R <= 1.0):
                    return False, "control_bounds"
                if k > 0:
                    t0 = traj.waypoints[k - 1].t if k - 1 < len(traj.waypoints) else traj.waypoints[-1].t
                    t1 = traj.waypoints[k].t if k < len(traj.waypoints) else traj.waypoints[-1].t
                    dt = float(max(t1 - t0, 0.0))
                    if dt > 1e-6:
                        rate_l = abs((ctrl.delta_L - last.delta_L) / dt)
                        rate_r = abs((ctrl.delta_R - last.delta_R) / dt)
                        if rate_l > rate_limit or rate_r > rate_limit:
                            return False, "control_rate"
                last = ctrl

        # Terminal heading constraint (if enabled)
        if self.config.enforce_terminal_upwind_heading:
            heading_err = self._terminal_heading_error(final, wind)
            if heading_err > float(np.deg2rad(self.config.terminal_heading_tol_deg)):
                return False, "terminal_heading"

        return True, "ok"

    def _evaluate_library_cost(self, traj: Trajectory, target: Target, wind: Wind) -> float:
        if not traj.waypoints:
            return float("inf")
        final = traj.waypoints[-1].state
        terminal_err = float(np.linalg.norm(final.position_xy - target.position_xy))
        heading_err = float(self._terminal_heading_error(final, wind))
        # Control effort
        effort = 0.0
        if traj.controls:
            effort = float(np.mean([c.delta_L * c.delta_L + c.delta_R * c.delta_R for c in traj.controls]))
        # Path length
        length = 0.0
        pts = [wp.state.position_xy for wp in traj.waypoints]
        for i in range(1, len(pts)):
            length += float(np.linalg.norm(pts[i] - pts[i - 1]))
        duration = float(traj.waypoints[-1].t - traj.waypoints[0].t)

        return float(
            self.config.library_cost_w_terminal_pos * terminal_err
            + self.config.library_cost_w_heading * heading_err
            + self.config.library_cost_w_control * effort
            + self.config.library_cost_w_time * duration
            + self.config.library_cost_w_path_length * length
        )

    def _library_match_ok(
        self,
        lib: LibraryTrajectory,
        features: np.ndarray,
        knn_distance: Optional[float],
    ) -> tuple[bool, str]:
        if knn_distance is not None and float(self.config.library_max_feature_distance) > 0.0:
            if float(knn_distance) > float(self.config.library_max_feature_distance):
                return False, "feature_distance"

        alt_err = abs(float(features[0]) - float(lib.scenario.initial_altitude_m))
        if float(self.config.library_max_altitude_error_m) > 0.0 and alt_err > float(self.config.library_max_altitude_error_m):
            return False, "altitude_mismatch"

        dist_err = abs(float(features[1]) - float(lib.scenario.target_distance_m))
        if float(self.config.library_max_distance_error_m) > 0.0 and dist_err > float(self.config.library_max_distance_error_m):
            return False, "distance_mismatch"

        bearing = float(features[2])
        bearing_lib = float(np.deg2rad(lib.scenario.target_bearing_deg))
        bearing_err = abs(float(wrap_pi(bearing - bearing_lib)))
        if float(self.config.library_max_bearing_error_deg) > 0.0:
            if bearing_err > float(np.deg2rad(self.config.library_max_bearing_error_deg)):
                return False, "bearing_mismatch"

        wind_speed = float(features[3])
        wind_err = abs(wind_speed - float(lib.scenario.wind_speed))
        if float(self.config.library_max_wind_speed_error_mps) > 0.0 and wind_err > float(self.config.library_max_wind_speed_error_mps):
            return False, "wind_speed_mismatch"

        rel_wind = float(features[4])
        wind_dir_lib = float(np.deg2rad(lib.scenario.wind_direction_deg))
        rel_wind_lib = float(wrap_pi(wind_dir_lib - bearing_lib))
        rel_wind_err = abs(float(wrap_pi(rel_wind - rel_wind_lib)))
        if float(self.config.library_max_rel_wind_error_deg) > 0.0:
            if rel_wind_err > float(np.deg2rad(self.config.library_max_rel_wind_error_deg)):
                return False, "rel_wind_mismatch"

        return True, "ok"

    def _library_scale_ok(self, traj: Trajectory) -> tuple[bool, str]:
        meta = traj.metadata or {}
        scale_xy = meta.get("scale_xy")
        scale_z = meta.get("scale_z")
        if scale_xy is not None:
            if float(self.config.library_min_scale_xy) > 0.0 and float(scale_xy) < float(self.config.library_min_scale_xy):
                return False, "scale_xy_min"
            if float(self.config.library_max_scale_xy) > 0.0 and float(scale_xy) > float(self.config.library_max_scale_xy):
                return False, "scale_xy_max"
        if scale_z is not None:
            if float(self.config.library_min_scale_z) > 0.0 and float(scale_z) < float(self.config.library_min_scale_z):
                return False, "scale_z_min"
            if float(self.config.library_max_scale_z) > 0.0 and float(scale_z) > float(self.config.library_max_scale_z):
                return False, "scale_z_max"
        return True, "ok"

    def _match_library(
        self,
        library: TrajectoryLibrary,
        features: np.ndarray,
        planning_target: Target,
        state: State,
        wind: Wind,
        k: int,
        stage: str,
    ) -> tuple[Trajectory | None, dict]:
        dist, idx = library.query_knn(features, k=max(int(k), 1))
        idx_arr = np.atleast_1d(idx)
        dist_arr = np.atleast_1d(dist)

        terrain = self._load_terrain()
        no_fly, no_fly_polygons = self._build_no_fly()

        best_traj: Trajectory | None = None
        best_cost = float("inf")
        best_idx = None
        best_reason = "no_candidate"
        best_dist = None
        best_term_err = float("inf")

        for i, cand_idx in enumerate(idx_arr.tolist()):
            lib = library[int(cand_idx)]
            adapted = adapt_trajectory(lib, state, planning_target, wind)
            knn_dist = float(dist_arr[i]) if i < len(dist_arr) else None
            match_ok, match_reason = self._library_match_ok(lib, features, knn_dist)
            scale_ok, scale_reason = self._library_scale_ok(adapted)
            if not match_ok:
                feasible, reason = False, f"match:{match_reason}"
            elif not scale_ok:
                feasible, reason = False, f"scale:{scale_reason}"
            else:
                feasible, reason = self._check_trajectory_feasible(
                    adapted,
                    planning_target,
                    wind,
                    terrain,
                    no_fly,
                    no_fly_polygons,
                )
            cost = self._evaluate_library_cost(adapted, planning_target, wind)
            term_err = float(np.linalg.norm(adapted.waypoints[-1].state.position_xy - planning_target.position_xy)) if adapted.waypoints else float("inf")

            meta = {
                "library_stage": stage,
                "library_candidate_index": int(cand_idx),
                "library_knn_distance": knn_dist,
                "library_match_ok": bool(match_ok),
                "library_scale_ok": bool(scale_ok),
                "library_feasible": bool(feasible),
                "library_feasible_reason": str(reason),
                "library_cost": float(cost),
            }
            if adapted.metadata:
                adapted.metadata.update(meta)
            else:
                adapted.metadata = meta

            if feasible and cost < best_cost:
                best_traj = adapted
                best_cost = float(cost)
                best_idx = int(cand_idx)
                best_reason = reason
                best_dist = float(dist_arr[i]) if i < len(dist_arr) else None
                best_term_err = float(term_err)

        return best_traj, {
            "stage": stage,
            "best_idx": best_idx,
            "best_dist": best_dist,
            "best_cost": best_cost,
            "best_reason": best_reason,
            "best_term_err": best_term_err,
        }

    def _solve_gpm(self, state: State, target: Target, wind: Wind) -> Tuple[Trajectory, SolverInfo]:
        # Capture wind in dynamics closure
        wind_I = wind.v_I.copy()

        def f(x: np.ndarray, u: np.ndarray, t: float) -> np.ndarray:
            return self.dynamics.f_vector(x, u, t, wind_I=wind_I)

        self.solver.f = f

        x0 = state.to_vector()
        tf_guess = float(self.config.tf_guess)
        dist = float(np.linalg.norm(target.position_xy - state.position_xy))
        Vh = max(state.speed_horizontal, 1.0)
        tf_guess = max(tf_guess, dist / Vh)

        warm = self.solver.last_solution_z if self.config.enable_warm_start else None
        terrain = self._load_terrain()
        no_fly, no_fly_polygons = self._build_no_fly()
        terminal_heading_hat_xy = self._terminal_heading_hat(wind)

        traj, info = self.solver.solve(
            x0=x0,
            p_target=target.p_I,
            tf_guess=tf_guess,
            warm_start=warm,
            terrain=terrain,
            terrain_clearance_m=float(self.config.terrain_clearance_m),
            no_fly_circles=no_fly,
            no_fly_polygons=no_fly_polygons,
            terminal_heading_hat_xy=terminal_heading_hat_xy,
        )
        return traj, info

    def _fallback_direct(self, state: State, target: Target, wind: Wind | None = None) -> Trajectory:
        # Straight-line interpolation in NED, keeping attitude/omega constant.
        p0 = state.p_I
        p1 = target.p_I.copy()
        if wind is not None:
            wind_xy = wind.v_I[:2]
            d_xy = target.position_xy - state.position_xy
            dist = float(np.linalg.norm(d_xy))
            v_air, _ = PolarTable().interpolate(float(self.config.fallback_brake))
            if dist > 1e-6:
                dir_hat = d_xy / dist
                v_ground = float(v_air + np.dot(wind_xy, dir_hat))
                v_ground = max(v_ground, float(self.config.fallback_min_ground_speed))
                t_go = dist / max(v_ground, 1e-3)
            else:
                t_go = 0.0
            drift = wind_xy * float(t_go) * float(self.config.fallback_wind_correction_gain)
            p1[0] = float(target.position_xy[0] - drift[0])
            p1[1] = float(target.position_xy[1] - drift[1])
        dist = float(np.linalg.norm((p1 - p0)[:2]))
        Vh = max(state.speed_horizontal, 1.0)
        tf = max(5.0, dist / Vh)
        dt = float(max(self.config.fallback_dt, 0.2))
        times = np.arange(0.0, tf + 1e-6, dt, dtype=float)

        waypoints = []
        controls = []
        for t in times:
            a = 0.0 if tf <= 1e-9 else float(t / tf)
            p = (1.0 - a) * p0 + a * p1
            s = State(p_I=p, v_I=state.v_I, q_IB=state.q_IB, w_B=state.w_B, t=float(state.t + t))
            waypoints.append(Waypoint(t=float(state.t + t), state=s))
            controls.append(Control(self.config.fallback_brake, self.config.fallback_brake))

        return Trajectory(
            waypoints=waypoints,
            controls=controls,
            trajectory_type=TrajectoryType.DIRECT,
            metadata={"fallback": True},
        )

    def plan(
        self,
        state: State,
        target: Target,
        wind: Wind,
        landing_site_selection: LandingSiteSelection | None = None,
    ) -> Tuple[Trajectory, SolverInfo]:
        gpm_attempted = False
        desired_target = target
        planning_target = target
        self.last_site_selection = None
        self.last_aimpoint_target = None

        terrain = self._load_terrain()
        no_fly, no_fly_polygons = self._build_no_fly()

        selection: LandingSiteSelection | None = None
        if landing_site_selection is not None:
            selection = landing_site_selection
        elif self.landing_site_selector is not None:
            try:
                selection = self.landing_site_selector.select(
                    state=state,
                    desired_target=desired_target,
                    wind=wind,
                    terrain=terrain,
                    no_fly_circles=no_fly,
                    no_fly_polygons=no_fly_polygons,
                )
            except Exception:  # pragma: no cover
                selection = None

        if selection is not None:
            self.last_site_selection = selection
            planning_target = selection.target

        touchdown_target = planning_target
        if bool(self.config.headwind_enable):
            aim_target, aim_meta = self._compute_aimpoint_target(
                state=state,
                touchdown_target=touchdown_target,
                wind=wind,
                terrain=terrain,
                no_fly_circles=no_fly,
                no_fly_polygons=no_fly_polygons,
                selection=selection,
            )
            if selection is not None and selection.metadata is not None:
                selection.metadata.update(aim_meta)
            self.last_aimpoint_target = aim_target
            if aim_target is not None and bool(aim_meta.get("aimpoint_used", False)):
                planning_target = aim_target
        # 1) Try library match (optional)
        skip_library_reason: str | None = None
        track_diag = self._wind_trackability_diag(state, planning_target, wind)
        if bool(self.config.library_skip_if_unreachable_wind):
            if not bool(track_diag.get("cross_ok", True)):
                skip_library_reason = "wind_crosswind_exceeds_V_air_max"
            else:
                v_track_max = float(track_diag.get("v_track_max_mps", 0.0))
                if v_track_max <= float(self.config.library_min_track_ground_speed_mps):
                    skip_library_reason = "wind_no_progress_to_target"

        skip_suffix = ""
        if skip_library_reason is not None:
            skip_suffix = (
                f" skip_library={skip_library_reason}"
                f" V_air_max={float(track_diag.get('V_air_max_mps', float('nan'))):.2f}"
                f" wind={float(track_diag.get('wind_speed_mps', float('nan'))):.2f}"
                f" head={float(track_diag.get('headwind_mps', float('nan'))):.2f}"
                f" cross={float(track_diag.get('crosswind_mps', float('nan'))):.2f}"
                f" v_track_max={float(track_diag.get('v_track_max_mps', float('nan'))):.2f}"
            )

        if self.config.use_library and skip_library_reason is None and (
            (self.library_fine is not None and len(self.library_fine) > 0)
            or (self.library_coarse is not None and len(self.library_coarse) > 0)
        ):
            feats = compute_scenario_features(state, planning_target, wind)
            coarse_best = None
            coarse_meta = {}
            coarse_ok = False

            if self.library_coarse is not None and len(self.library_coarse) > 0:
                coarse_best, coarse_meta = self._match_library(
                    self.library_coarse,
                    feats,
                    planning_target,
                    state,
                    wind,
                    k=self.config.library_coarse_k_neighbors or self.config.k_neighbors,
                    stage="coarse",
                )
                coarse_ok = coarse_best is not None

            require_coarse = bool(self.config.library_require_coarse_match) and self.library_coarse is not None
            if require_coarse and not coarse_ok:
                # Coarse gate failed; skip fine stage and fall back to GPM.
                pass
            else:
                fine_best = None
                fine_meta = {}
                if self.library_fine is not None and len(self.library_fine) > 0:
                    fine_best, fine_meta = self._match_library(
                        self.library_fine,
                        feats,
                        planning_target,
                        state,
                        wind,
                        k=self.config.library_fine_k_neighbors or self.config.k_neighbors,
                        stage="fine",
                    )
                if fine_best is not None:
                    best_traj = fine_best
                    best_meta = fine_meta
                else:
                    best_traj = coarse_best if (self.config.library_fallback_to_coarse and coarse_best is not None) else None
                    best_meta = coarse_meta

                if best_traj is not None:
                    stage = best_meta.get("stage", "library")
                    best_idx = best_meta.get("best_idx")
                    best_dist = best_meta.get("best_dist")
                    best_cost = best_meta.get("best_cost", float("inf"))
                    best_reason = best_meta.get("best_reason", "ok")
                    best_term_err = best_meta.get("best_term_err", float("inf"))

                    # Optionally fine-tune if mismatch is large.
                    if self.config.enable_gpm_fine_tuning and best_term_err > float(self.config.fine_tuning_trigger_m):
                        try:
                            traj, info = self._solve_gpm(state, planning_target, wind)
                            gpm_attempted = True
                            if info.success and info.max_violation <= float(self.config.max_constraint_violation_for_accept):
                                if skip_library_reason is not None:
                                    info = SolverInfo(
                                        success=info.success,
                                        status=info.status,
                                        message=f"{info.message}{skip_suffix}",
                                        iterations=info.iterations,
                                        cost=info.cost,
                                        solve_time=info.solve_time,
                                        max_violation=info.max_violation,
                                        terminal_error_m=info.terminal_error_m,
                                    )
                                return traj, info
                        except Exception as e:  # pragma: no cover
                            info = SolverInfo(
                                success=False,
                                status=-1,
                                message=f"exception:{e}",
                                iterations=-1,
                                cost=float("inf"),
                                solve_time=0.0,
                                max_violation=float("inf"),
                                terminal_error_m=float("inf"),
                            )

                    info = SolverInfo(
                        success=True,
                        status=0,
                        message=f"library:{stage}:{best_traj.trajectory_type.value} idx={best_idx} dist={best_dist} cost={best_cost:.3g} reason={best_reason}",
                        iterations=0,
                        cost=float(best_cost),
                        solve_time=0.0,
                        max_violation=0.0,
                        terminal_error_m=float(best_term_err),
                    )
                    return best_traj, info

            # No feasible library candidate → try GPM if enabled.
            if self.config.enable_gpm_fine_tuning:
                try:
                    traj, info = self._solve_gpm(state, planning_target, wind)
                    gpm_attempted = True
                    if info.success and info.max_violation <= float(self.config.max_constraint_violation_for_accept):
                        return traj, info
                except Exception as e:  # pragma: no cover
                    info = SolverInfo(
                        success=False,
                        status=-1,
                        message=f"exception:{e}",
                        iterations=-1,
                        cost=float("inf"),
                        solve_time=0.0,
                        max_violation=float("inf"),
                        terminal_error_m=float("inf"),
                    )

        # 2) Solve online with GPM
        if not gpm_attempted:
            try:
                traj, info = self._solve_gpm(state, planning_target, wind)
                if info.success and info.max_violation <= float(self.config.max_constraint_violation_for_accept):
                    if skip_library_reason is not None:
                        info = SolverInfo(
                            success=info.success,
                            status=info.status,
                            message=f"{info.message}{skip_suffix}",
                            iterations=info.iterations,
                            cost=info.cost,
                            solve_time=info.solve_time,
                            max_violation=info.max_violation,
                            terminal_error_m=info.terminal_error_m,
                        )
                    return traj, info
            except Exception as e:  # pragma: no cover
                info = SolverInfo(
                    success=False,
                    status=-1,
                    message=f"exception:{e}",
                    iterations=-1,
                    cost=float("inf"),
                    solve_time=0.0,
                    max_violation=float("inf"),
                    terminal_error_m=float("inf"),
                )
                if skip_library_reason is not None:
                    info = SolverInfo(
                        success=info.success,
                        status=info.status,
                        message=f"{info.message}{skip_suffix}",
                        iterations=info.iterations,
                        cost=info.cost,
                        solve_time=info.solve_time,
                        max_violation=info.max_violation,
                        terminal_error_m=info.terminal_error_m,
                    )
                return self._fallback_direct(state, planning_target, wind), info

        # 3) Fallback
        if skip_library_reason is not None:
            info = SolverInfo(
                success=info.success,
                status=info.status,
                message=f"{info.message}{skip_suffix}",
                iterations=info.iterations,
                cost=info.cost,
                solve_time=info.solve_time,
                max_violation=info.max_violation,
                terminal_error_m=info.terminal_error_m,
            )
        return self._fallback_direct(state, planning_target, wind), info
