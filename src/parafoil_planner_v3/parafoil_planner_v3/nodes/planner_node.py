from __future__ import annotations

from dataclasses import dataclass
import pathlib
from typing import Optional

import json
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Vector3Stamped
from geometry_msgs.msg import Point
from nav_msgs.msg import Odometry, Path
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import Marker, MarkerArray

from parafoil_msgs.srv import SetTarget

from parafoil_dynamics.integrators import parse_integrator_type
from parafoil_dynamics.math3d import quat_from_euler
from parafoil_dynamics.params import Params

from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.dynamics.parafoil_6dof import IntegratorConfig, SixDOFDynamics
from parafoil_planner_v3.environment import NoFlyPolygon
from parafoil_planner_v3.landing_site_selector import (
    LandingSiteSelector,
    LandingSiteSelectorConfig,
    ReachabilityConfig,
    RiskGrid,
    RiskLayer,
    RiskMapAggregator,
)
from parafoil_planner_v3.planner_core import PlannerConfig, PlannerCore
from parafoil_planner_v3.target_update_policy import TargetUpdatePolicy, TargetUpdatePolicyConfig, UpdateReason
from parafoil_planner_v3.trajectory_library.library_manager import TrajectoryLibrary
from parafoil_planner_v3.types import GuidancePhase, State, Target, Wind
from parafoil_planner_v3.utils.wind_utils import (
    WindConvention,
    WindInputFrame,
    clip_wind_xy,
    frame_from_frame_id,
    parse_wind_convention,
    parse_wind_input_frame,
    to_ned_wind_to,
)


@dataclass(frozen=True)
class NoiseConfig:
    position_sigma: np.ndarray
    velocity_sigma: np.ndarray
    attitude_sigma: np.ndarray
    angular_rate_sigma: np.ndarray


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _quat_enu_to_ned(q_enu) -> np.ndarray:
    # Inverse of sim_node conversion:
    # q_enu.w = q_ned.w
    # q_enu.x = q_ned.y
    # q_enu.y = q_ned.x
    # q_enu.z = -q_ned.z
    return np.array([q_enu.w, q_enu.y, q_enu.x, -q_enu.z], dtype=float)


def _quat_ned_to_enu(q_ned: np.ndarray):
    q_ned = np.asarray(q_ned, dtype=float).reshape(4)
    # Same mapping used by parafoil_simulator_ros for visualization.
    # q_enu = [w, y, x, -z]
    return float(q_ned[0]), float(q_ned[2]), float(q_ned[1]), float(-q_ned[3])


def _ned_to_enu_xyz(p_ned: np.ndarray) -> tuple[float, float, float]:
    p = np.asarray(p_ned, dtype=float).reshape(3)
    return float(p[1]), float(p[0]), float(-p[2])


class PlannerNode(Node):
    def __init__(self) -> None:
        super().__init__("parafoil_planner_v3")

        # ------------------------------------------------------------------
        # Parameters
        # ------------------------------------------------------------------
        self.declare_parameter("planner_rate_hz", 1.0)
        self.declare_parameter("use_library", False)
        self.declare_parameter("library_path", "")
        self.declare_parameter("library.k_neighbors", 5)
        self.declare_parameter("library.enable_gpm_fine_tuning", True)
        self.declare_parameter("library.fine_tuning_trigger_m", 10.0)
        self.declare_parameter("library.terminal_pos_tol_m", 20.0)
        self.declare_parameter("library.cost_w_terminal_pos", 1.0)
        self.declare_parameter("library.cost_w_heading", 0.2)
        self.declare_parameter("library.cost_w_control", 0.05)
        self.declare_parameter("library.cost_w_time", 0.02)
        self.declare_parameter("library.cost_w_path_length", 0.0)
        self.declare_parameter("library.coarse_path", "")
        self.declare_parameter("library.fine_path", "")
        self.declare_parameter("library.coarse_k_neighbors", 5)
        self.declare_parameter("library.fine_k_neighbors", 5)
        self.declare_parameter("library.require_coarse_match", True)
        self.declare_parameter("library.fallback_to_coarse", True)
        # Library safety gates (0 disables a check)
        self.declare_parameter("library.max_feature_distance", 0.0)
        self.declare_parameter("library.max_altitude_error_m", 0.0)
        self.declare_parameter("library.max_distance_error_m", 0.0)
        self.declare_parameter("library.max_bearing_error_deg", 0.0)
        self.declare_parameter("library.max_wind_speed_error_mps", 0.0)
        self.declare_parameter("library.max_rel_wind_error_deg", 0.0)
        self.declare_parameter("library.min_scale_xy", 0.0)
        self.declare_parameter("library.max_scale_xy", 0.0)
        self.declare_parameter("library.min_scale_z", 0.0)
        self.declare_parameter("library.max_scale_z", 0.0)
        self.declare_parameter("library.skip_if_unreachable_wind", True)
        self.declare_parameter("library.min_track_ground_speed_mps", 0.2)

        self.declare_parameter("gpm.num_nodes", 20)
        self.declare_parameter("gpm.scheme", "LGL")
        self.declare_parameter("gpm.tf_guess", 30.0)
        self.declare_parameter("gpm.max_constraint_violation_for_accept", 0.5)

        # GPM solver config (see config/gpm_params.yaml)
        self.declare_parameter("solver.method", "SLSQP")
        self.declare_parameter("solver.maxiter", 300)
        self.declare_parameter("solver.ftol", 1.0e-6)
        self.declare_parameter("solver.max_solve_time_s", 0.0)

        self.declare_parameter("weights.w_terminal_pos", 50.0)
        self.declare_parameter("weights.w_terminal_vel", 1.0)
        self.declare_parameter("weights.w_running_u", 0.05)
        self.declare_parameter("weights.w_running_du", 0.01)
        self.declare_parameter("weights.w_running_w", 0.0)
        self.declare_parameter("weights.w_running_wdot", 0.0)
        self.declare_parameter("weights.w_running_energy", 0.0)

        self.declare_parameter("fallback.dt", 1.0)
        self.declare_parameter("fallback.brake", 0.2)
        self.declare_parameter("fallback.wind_correction_gain", 0.5)
        self.declare_parameter("fallback.min_ground_speed", 0.5)

        self.declare_parameter("constraints.tf_min", 1.0)
        self.declare_parameter("constraints.tf_max", 120.0)
        self.declare_parameter("constraints.delta_rate_max", 3.0)
        self.declare_parameter("constraints.Vh_min", 0.5)
        self.declare_parameter("constraints.Vh_max", 8.0)
        self.declare_parameter("constraints.roll_max_deg", 60.0)
        self.declare_parameter("constraints.yaw_rate_max_deg", 120.0)

        self.declare_parameter("constraints.terminal_upwind_heading", False)
        self.declare_parameter("constraints.terminal_heading_tol_deg", 5.0)
        self.declare_parameter("constraints.terrain_type", "flat")
        self.declare_parameter("constraints.terrain_height0_m", 0.0)
        self.declare_parameter("constraints.terrain_slope_n", 0.0)
        self.declare_parameter("constraints.terrain_slope_e", 0.0)
        self.declare_parameter("constraints.terrain_clearance_m", 0.0)
        self.declare_parameter("constraints.terrain_file", "")
        # JSON-encoded list of circles: [[center_n, center_e, radius_m, clearance_m], ...]
        self.declare_parameter("constraints.no_fly_circles", "[]")
        # JSON-encoded list of polygons: [[[n,e],[n,e],...], ...] or [{"vertices":[...], "clearance_m":1.0}, ...]
        self.declare_parameter("constraints.no_fly_polygons", "[]")
        # Optional file (yaml/json/geojson) with polygons
        self.declare_parameter("constraints.no_fly_polygons_file", "")

        self.declare_parameter("noise.position_sigma", [0.5, 0.5, 0.3])
        self.declare_parameter("noise.velocity_sigma", [0.1, 0.1, 0.05])
        self.declare_parameter("noise.attitude_sigma", [0.01, 0.01, 0.02])
        self.declare_parameter("noise.angular_rate_sigma", [0.005, 0.005, 0.01])
        self.declare_parameter("noise.seed", 0)

        self.declare_parameter("target.position_ned", [0.0, 0.0, 0.0])
        self.declare_parameter("target.auto_mode", "manual")  # manual|current|reach_center|safety
        self.declare_parameter("target.update_policy.enabled", True)
        self.declare_parameter("target.update_policy.enable_hysteresis", True)
        self.declare_parameter("target.update_policy.score_hysteresis", 0.5)
        self.declare_parameter("target.update_policy.dist_hysteresis_m", 20.0)
        self.declare_parameter("target.update_policy.cruise_allow_update", True)
        self.declare_parameter("target.update_policy.approach_allow_update", "emergency_only")
        self.declare_parameter("target.update_policy.flare_lock", True)
        self.declare_parameter("target.update_policy.emergency_margin_mps", -0.5)
        self.declare_parameter("target.update_policy.emergency_cooldown_s", 2.0)
        self.declare_parameter("target.update_policy.approach_significant_factor", 2.0)
        self.declare_parameter("target.update_policy.smooth_transition", False)
        self.declare_parameter("target.update_policy.smooth_rate_mps", 5.0)

        self.declare_parameter("dynamics_params_yaml", "")
        self.declare_parameter("integrator.method", "rk4")
        self.declare_parameter("integrator.dt_max", 0.01)

        self.declare_parameter("wind.use_topic", True)
        self.declare_parameter("wind.topic", "/wind_estimate")
        self.declare_parameter("wind.default_ned", [0.0, 2.0, 0.0])
        # /wind_estimate conversion -> internal standard: NED + wind-to
        self.declare_parameter("wind.input_frame", "ned")  # ned|enu|auto
        self.declare_parameter("wind.convention", "to")  # to|from
        self.declare_parameter("wind.timeout_s", 2.0)
        self.declare_parameter("wind.max_speed_mps", 0.0)  # 0 disables clipping

        # Safety-first landing site selection
        self.declare_parameter("safety.enable", False)
        self.declare_parameter("safety.selector.grid_resolution_m", 25.0)
        self.declare_parameter("safety.selector.search_radius_m", 0.0)
        self.declare_parameter("safety.selector.max_candidates", 800)
        self.declare_parameter("safety.selector.random_seed", 0)
        self.declare_parameter("safety.selector.w_risk", 5.0)
        self.declare_parameter("safety.selector.w_distance", 1.0)
        self.declare_parameter("safety.selector.w_reach_margin", 1.0)
        self.declare_parameter("safety.selector.w_energy", 0.5)
        self.declare_parameter("safety.selector.nofly_buffer_m", 20.0)
        self.declare_parameter("safety.selector.nofly_weight", 5.0)
        self.declare_parameter("safety.selector.snap_to_terrain", True)

        self.declare_parameter("safety.reachability.brake", 0.2)
        self.declare_parameter("safety.reachability.min_time_s", 2.0)
        self.declare_parameter("safety.reachability.max_time_s", 200.0)
        self.declare_parameter("safety.reachability.wind_margin_mps", 0.2)
        self.declare_parameter("safety.reachability.wind_uncertainty_mps", 0.0)
        self.declare_parameter("safety.reachability.gust_margin_mps", 0.0)
        self.declare_parameter("safety.reachability.min_altitude_m", 5.0)
        self.declare_parameter("safety.reachability.enforce_circle", True)

        self.declare_parameter("safety.risk.grid_file", "")
        self.declare_parameter("safety.risk.grid_weight", 1.0)
        self.declare_parameter("safety.risk.clip_min", 0.0)
        self.declare_parameter("safety.risk.clip_max", 1.0)
        self.declare_parameter("safety.risk.oob_value", 1.0)
        self.declare_parameter("safety.risk.layer_files", [])
        self.declare_parameter("safety.risk.layer_weights", [])
        self.declare_parameter("safety.risk.layer_names", [])

        # Headwind / aimpoint planning
        self.declare_parameter("headwind.enable", True)
        self.declare_parameter("headwind.wind_ratio_trigger", 0.8)
        self.declare_parameter("headwind.max_aim_offset_m", 200.0)
        self.declare_parameter("headwind.min_progress_mps", 0.0)
        self.declare_parameter("headwind.aimpoint_nofly_projection", True)

        # ------------------------------------------------------------------
        # Build components
        # ------------------------------------------------------------------
        params_yaml = str(self.get_parameter("dynamics_params_yaml").value)
        params = Params.from_yaml(params_yaml) if params_yaml else Params()

        integrator_method = parse_integrator_type(str(self.get_parameter("integrator.method").value))
        integrator_cfg = IntegratorConfig(method=integrator_method, dt_max=float(self.get_parameter("integrator.dt_max").value))
        dynamics = SixDOFDynamics(params=params, integrator=integrator_cfg)

        planner_cfg = PlannerConfig(
            gpm_num_nodes=int(self.get_parameter("gpm.num_nodes").value),
            gpm_scheme=str(self.get_parameter("gpm.scheme").value),
            tf_guess=float(self.get_parameter("gpm.tf_guess").value),
            max_constraint_violation_for_accept=float(self.get_parameter("gpm.max_constraint_violation_for_accept").value),
            use_library=bool(self.get_parameter("use_library").value),
            k_neighbors=int(self.get_parameter("library.k_neighbors").value),
            enable_gpm_fine_tuning=bool(self.get_parameter("library.enable_gpm_fine_tuning").value),
            fine_tuning_trigger_m=float(self.get_parameter("library.fine_tuning_trigger_m").value),
            library_terminal_pos_tol_m=float(self.get_parameter("library.terminal_pos_tol_m").value),
            library_cost_w_terminal_pos=float(self.get_parameter("library.cost_w_terminal_pos").value),
            library_cost_w_heading=float(self.get_parameter("library.cost_w_heading").value),
            library_cost_w_control=float(self.get_parameter("library.cost_w_control").value),
            library_cost_w_time=float(self.get_parameter("library.cost_w_time").value),
            library_cost_w_path_length=float(self.get_parameter("library.cost_w_path_length").value),
            library_coarse_k_neighbors=int(self.get_parameter("library.coarse_k_neighbors").value),
            library_fine_k_neighbors=int(self.get_parameter("library.fine_k_neighbors").value),
            library_require_coarse_match=bool(self.get_parameter("library.require_coarse_match").value),
            library_fallback_to_coarse=bool(self.get_parameter("library.fallback_to_coarse").value),
            library_max_feature_distance=float(self.get_parameter("library.max_feature_distance").value),
            library_max_altitude_error_m=float(self.get_parameter("library.max_altitude_error_m").value),
            library_max_distance_error_m=float(self.get_parameter("library.max_distance_error_m").value),
            library_max_bearing_error_deg=float(self.get_parameter("library.max_bearing_error_deg").value),
            library_max_wind_speed_error_mps=float(self.get_parameter("library.max_wind_speed_error_mps").value),
            library_max_rel_wind_error_deg=float(self.get_parameter("library.max_rel_wind_error_deg").value),
            library_min_scale_xy=float(self.get_parameter("library.min_scale_xy").value),
            library_max_scale_xy=float(self.get_parameter("library.max_scale_xy").value),
            library_min_scale_z=float(self.get_parameter("library.min_scale_z").value),
            library_max_scale_z=float(self.get_parameter("library.max_scale_z").value),
            library_skip_if_unreachable_wind=bool(self.get_parameter("library.skip_if_unreachable_wind").value),
            library_min_track_ground_speed_mps=float(self.get_parameter("library.min_track_ground_speed_mps").value),
            solver_method=str(self.get_parameter("solver.method").value),
            solver_maxiter=int(self.get_parameter("solver.maxiter").value),
            solver_ftol=float(self.get_parameter("solver.ftol").value),
            solver_max_solve_time_s=float(self.get_parameter("solver.max_solve_time_s").value),
            w_terminal_pos=float(self.get_parameter("weights.w_terminal_pos").value),
            w_terminal_vel=float(self.get_parameter("weights.w_terminal_vel").value),
            w_running_u=float(self.get_parameter("weights.w_running_u").value),
            w_running_du=float(self.get_parameter("weights.w_running_du").value),
            w_running_w=float(self.get_parameter("weights.w_running_w").value),
            w_running_wdot=float(self.get_parameter("weights.w_running_wdot").value),
            w_running_energy=float(self.get_parameter("weights.w_running_energy").value),
            fallback_dt=float(self.get_parameter("fallback.dt").value),
            fallback_brake=float(self.get_parameter("fallback.brake").value),
            fallback_wind_correction_gain=float(self.get_parameter("fallback.wind_correction_gain").value),
            fallback_min_ground_speed=float(self.get_parameter("fallback.min_ground_speed").value),
            tf_min=float(self.get_parameter("constraints.tf_min").value),
            tf_max=float(self.get_parameter("constraints.tf_max").value),
            delta_rate_max=float(self.get_parameter("constraints.delta_rate_max").value),
            Vh_min=float(self.get_parameter("constraints.Vh_min").value),
            Vh_max=float(self.get_parameter("constraints.Vh_max").value),
            roll_max_deg=float(self.get_parameter("constraints.roll_max_deg").value),
            yaw_rate_max_deg=float(self.get_parameter("constraints.yaw_rate_max_deg").value),
            enforce_terminal_upwind_heading=bool(self.get_parameter("constraints.terminal_upwind_heading").value),
            terminal_heading_tol_deg=float(self.get_parameter("constraints.terminal_heading_tol_deg").value),
            terrain_type=str(self.get_parameter("constraints.terrain_type").value),
            terrain_height0_m=float(self.get_parameter("constraints.terrain_height0_m").value),
            terrain_slope_n=float(self.get_parameter("constraints.terrain_slope_n").value),
            terrain_slope_e=float(self.get_parameter("constraints.terrain_slope_e").value),
            terrain_clearance_m=float(self.get_parameter("constraints.terrain_clearance_m").value),
            terrain_file=str(self.get_parameter("constraints.terrain_file").value),
            no_fly_circles=(),
            no_fly_polygons=(),
            no_fly_polygons_file=str(self.get_parameter("constraints.no_fly_polygons_file").value),
            headwind_enable=bool(self.get_parameter("headwind.enable").value),
            headwind_wind_ratio_trigger=float(self.get_parameter("headwind.wind_ratio_trigger").value),
            headwind_max_aim_offset_m=float(self.get_parameter("headwind.max_aim_offset_m").value),
            headwind_min_progress_mps=float(self.get_parameter("headwind.min_progress_mps").value),
            headwind_aimpoint_nofly_projection=bool(self.get_parameter("headwind.aimpoint_nofly_projection").value),
        )
        no_fly_raw = str(self.get_parameter("constraints.no_fly_circles").value).strip()
        if no_fly_raw:
            try:
                parsed = json.loads(no_fly_raw)
                circles: list[tuple[float, float, float, float]] = []
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, list) and len(item) >= 3:
                            cn = float(item[0])
                            ce = float(item[1])
                            r = float(item[2])
                            c = float(item[3]) if len(item) >= 4 else 0.0
                            circles.append((cn, ce, r, c))
                planner_cfg = PlannerConfig(**{**planner_cfg.__dict__, "no_fly_circles": tuple(circles)})
            except Exception as e:
                self.get_logger().warn(f"Failed to parse constraints.no_fly_circles='{no_fly_raw}': {e}")

        poly_raw = str(self.get_parameter("constraints.no_fly_polygons").value).strip()
        if poly_raw:
            try:
                parsed = json.loads(poly_raw)
                polygons: list[NoFlyPolygon] = []
                if isinstance(parsed, list):
                    for item in parsed:
                        clearance = 0.0
                        vertices = None
                        if isinstance(item, dict):
                            vertices = item.get("vertices")
                            clearance = float(item.get("clearance_m", 0.0))
                        else:
                            vertices = item
                        if isinstance(vertices, list) and len(vertices) >= 3:
                            verts = [(float(v[0]), float(v[1])) for v in vertices if isinstance(v, (list, tuple)) and len(v) >= 2]
                            if len(verts) >= 3:
                                polygons.append(NoFlyPolygon(vertices=np.asarray(verts, dtype=float), clearance_m=clearance))
                if polygons:
                    planner_cfg = PlannerConfig(**{**planner_cfg.__dict__, "no_fly_polygons": tuple(polygons)})
            except Exception as e:
                self.get_logger().warn(f"Failed to parse constraints.no_fly_polygons='{poly_raw}': {e}")

        landing_site_selector: LandingSiteSelector | None = None
        if bool(self.get_parameter("safety.enable").value):
            clip_min = float(self.get_parameter("safety.risk.clip_min").value)
            clip_max = float(self.get_parameter("safety.risk.clip_max").value)
            oob_value = float(self.get_parameter("safety.risk.oob_value").value)

            layers: list[RiskLayer] = []
            grid_file = str(self.get_parameter("safety.risk.grid_file").value).strip()
            if grid_file:
                try:
                    grid = RiskGrid.from_file(grid_file, oob_value=oob_value)
                    weight = float(self.get_parameter("safety.risk.grid_weight").value)
                    layers.append(RiskLayer(name=pathlib.Path(grid_file).stem or "risk", weight=weight, grid=grid))
                except Exception as e:
                    self.get_logger().warn(f"Failed to load risk grid '{grid_file}': {e}")

            layer_files = list(self.get_parameter("safety.risk.layer_files").value or [])
            layer_weights = list(self.get_parameter("safety.risk.layer_weights").value or [])
            layer_names = list(self.get_parameter("safety.risk.layer_names").value or [])
            for i, path in enumerate(layer_files):
                path = str(path).strip()
                if not path:
                    continue
                try:
                    weight = float(layer_weights[i]) if i < len(layer_weights) else 1.0
                    name = str(layer_names[i]) if i < len(layer_names) else pathlib.Path(path).stem
                    grid = RiskGrid.from_file(path, oob_value=oob_value)
                    layers.append(RiskLayer(name=name or f"layer_{i}", weight=weight, grid=grid))
                except Exception as e:
                    self.get_logger().warn(f"Failed to load risk layer '{path}': {e}")

            risk_map = RiskMapAggregator(layers=layers, clip_min=clip_min, clip_max=clip_max) if layers else None

            reach_cfg = ReachabilityConfig(
                brake=float(self.get_parameter("safety.reachability.brake").value),
                min_time_s=float(self.get_parameter("safety.reachability.min_time_s").value),
                max_time_s=float(self.get_parameter("safety.reachability.max_time_s").value),
                wind_margin_mps=float(self.get_parameter("safety.reachability.wind_margin_mps").value),
                wind_uncertainty_mps=float(self.get_parameter("safety.reachability.wind_uncertainty_mps").value),
                gust_margin_mps=float(self.get_parameter("safety.reachability.gust_margin_mps").value),
                min_altitude_m=float(self.get_parameter("safety.reachability.min_altitude_m").value),
                terrain_clearance_m=float(planner_cfg.terrain_clearance_m),
                enforce_circle=bool(self.get_parameter("safety.reachability.enforce_circle").value),
            )
            selector_cfg = LandingSiteSelectorConfig(
                enabled=True,
                grid_resolution_m=float(self.get_parameter("safety.selector.grid_resolution_m").value),
                search_radius_m=float(self.get_parameter("safety.selector.search_radius_m").value),
                max_candidates=int(self.get_parameter("safety.selector.max_candidates").value),
                random_seed=int(self.get_parameter("safety.selector.random_seed").value),
                w_risk=float(self.get_parameter("safety.selector.w_risk").value),
                w_distance=float(self.get_parameter("safety.selector.w_distance").value),
                w_reach_margin=float(self.get_parameter("safety.selector.w_reach_margin").value),
                w_energy=float(self.get_parameter("safety.selector.w_energy").value),
                nofly_buffer_m=float(self.get_parameter("safety.selector.nofly_buffer_m").value),
                nofly_weight=float(self.get_parameter("safety.selector.nofly_weight").value),
                snap_to_terrain=bool(self.get_parameter("safety.selector.snap_to_terrain").value),
                min_progress_mps=float(self.get_parameter("headwind.min_progress_mps").value),
                reachability=reach_cfg,
            )
            landing_site_selector = LandingSiteSelector(config=selector_cfg, risk_map=risk_map)
            self.get_logger().info("Safety landing site selector enabled")

        library: Optional[TrajectoryLibrary] = None
        library_coarse: Optional[TrajectoryLibrary] = None
        library_fine: Optional[TrajectoryLibrary] = None

        library_path = str(self.get_parameter("library_path").value).strip()
        coarse_path = str(self.get_parameter("library.coarse_path").value).strip()
        fine_path = str(self.get_parameter("library.fine_path").value).strip()
        if not fine_path:
            fine_path = library_path

        if planner_cfg.use_library and coarse_path:
            try:
                library_coarse = TrajectoryLibrary.load(coarse_path)
                self.get_logger().info(f"Loaded COARSE library: {len(library_coarse)} trajectories from {coarse_path}")
            except Exception as e:
                self.get_logger().error(f"Failed to load coarse library '{coarse_path}': {e}")
                library_coarse = None

        if planner_cfg.use_library and fine_path:
            try:
                library_fine = TrajectoryLibrary.load(fine_path)
                self.get_logger().info(f"Loaded FINE library: {len(library_fine)} trajectories from {fine_path}")
                library = library_fine
            except Exception as e:
                self.get_logger().error(f"Failed to load fine library '{fine_path}': {e}")
                library_fine = None
                library = None

        self.planner = PlannerCore(
            dynamics=dynamics,
            config=planner_cfg,
            library=library,
            library_coarse=library_coarse,
            library_fine=library_fine,
            landing_site_selector=landing_site_selector,
        )

        # Target update policy (gust-robust hysteresis/locks)
        self._target_policy = TargetUpdatePolicy(self._load_policy_config())
        self._current_phase = GuidancePhase.CRUISE

        # Noise config
        self.rng = np.random.default_rng(int(self.get_parameter("noise.seed").value))
        self.noise_cfg = NoiseConfig(
            position_sigma=np.asarray(self.get_parameter("noise.position_sigma").value, dtype=float),
            velocity_sigma=np.asarray(self.get_parameter("noise.velocity_sigma").value, dtype=float),
            attitude_sigma=np.asarray(self.get_parameter("noise.attitude_sigma").value, dtype=float),
            angular_rate_sigma=np.asarray(self.get_parameter("noise.angular_rate_sigma").value, dtype=float),
        )

        # ------------------------------------------------------------------
        # State
        # ------------------------------------------------------------------
        self._odom: Optional[Odometry] = None
        self._imu_w_B: Optional[np.ndarray] = None
        self._wind_ned_to: Optional[np.ndarray] = None
        self._wind_recv_time_s: Optional[float] = None
        self._wind_msg_time_s: Optional[float] = None
        self._wind_frame_id: str = ""
        self._wind_input_frame_used: str = ""
        self._wind_convention_used: str = ""
        self._wind_clipped: bool = False
        self._target_ned: Optional[np.ndarray] = np.asarray(self.get_parameter("target.position_ned").value, dtype=float).reshape(3)
        self._safe_target_ned: Optional[np.ndarray] = None
        self._aimpoint_ned: Optional[np.ndarray] = None
        self._polar = PolarTable()

        self._force_replan = True

        # ------------------------------------------------------------------
        # ROS I/O
        # ------------------------------------------------------------------
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.create_subscription(Odometry, "/parafoil/odom", self._on_odom, 10)
        self.create_subscription(PoseStamped, "/target", self._on_target, 10)
        self.create_subscription(Imu, "/parafoil/imu", self._on_imu, sensor_qos)
        self.create_subscription(String, "/guidance_phase", self._on_phase, 10)

        if bool(self.get_parameter("wind.use_topic").value):
            self.create_subscription(Vector3Stamped, str(self.get_parameter("wind.topic").value), self._on_wind, 10)

        self.pub_path = self.create_publisher(Path, "/planned_trajectory", 10)
        # Compatibility with v2 visualizers
        self.pub_path_v2 = self.create_publisher(Path, "/planned_path", 10)
        self.pub_preview = self.create_publisher(MarkerArray, "/trajectory_preview", 10)
        self.pub_status = self.create_publisher(String, "/planner_status", 10)

        self.create_service(Trigger, "/replan", self._on_replan)
        self.create_service(SetTarget, "/set_target", self._on_set_target)

        rate = float(self.get_parameter("planner_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 0.1), self._tick)

        self.get_logger().info("parafoil_planner_v3 started")

    def _on_replan(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:  # noqa: ARG002
        self._force_replan = True
        res.success = True
        res.message = "replan requested"
        return res

    def _on_set_target(self, req: SetTarget.Request, res: SetTarget.Response) -> SetTarget.Response:
        self._target_ned = np.array([req.north_m, req.east_m, req.down_m], dtype=float)
        self._force_replan = True
        res.success = True
        res.message = "target updated"
        return res

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_imu(self, msg: Imu) -> None:
        self._imu_w_B = np.array(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z], dtype=float
        )

    def _on_wind(self, msg: Vector3Stamped) -> None:
        raw = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        now_s = self.get_clock().now().nanoseconds * 1e-9
        stamp_s = _stamp_to_sec(msg.header.stamp)
        if stamp_s <= 1e-6:
            stamp_s = now_s

        input_frame_raw = str(self.get_parameter("wind.input_frame").value)
        convention_raw = str(self.get_parameter("wind.convention").value)
        try:
            input_frame = parse_wind_input_frame(input_frame_raw)
        except ValueError as e:
            self.get_logger().error(str(e))
            input_frame = WindInputFrame.NED
        try:
            convention = parse_wind_convention(convention_raw)
        except ValueError as e:
            self.get_logger().error(str(e))
            convention = WindConvention.TO

        input_frame_used = input_frame
        if input_frame == WindInputFrame.AUTO:
            guessed = frame_from_frame_id(msg.header.frame_id)
            input_frame_used = guessed if guessed is not None else WindInputFrame.NED

        try:
            wind_ned = to_ned_wind_to(raw, input_frame=input_frame_used, convention=convention)
        except Exception as e:
            self.get_logger().error(f"Failed to convert wind estimate: {e}")
            return

        max_speed_mps = float(self.get_parameter("wind.max_speed_mps").value)
        wind_ned, clipped = clip_wind_xy(wind_ned, max_speed_mps=max_speed_mps)

        self._wind_ned_to = wind_ned
        self._wind_recv_time_s = now_s
        self._wind_msg_time_s = stamp_s
        self._wind_frame_id = str(msg.header.frame_id)
        self._wind_input_frame_used = input_frame_used.value
        self._wind_convention_used = convention.value
        self._wind_clipped = bool(clipped)

    def _on_target(self, msg: PoseStamped) -> None:
        # /target is PoseStamped in world ENU. Convert to NED.
        p_enu = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self._target_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)
        self._force_replan = True

    def _on_phase(self, msg: String) -> None:
        phase = str(msg.data).strip().lower()
        mapping = {
            "cruise": GuidancePhase.CRUISE,
            "approach": GuidancePhase.APPROACH,
            "flare": GuidancePhase.FLARE,
            "landed": GuidancePhase.LANDED,
            "abort": GuidancePhase.ABORT,
        }
        if phase in mapping:
            self._current_phase = mapping[phase]
        else:
            self.get_logger().warn(f"Unknown guidance phase '{msg.data}'")

    def _wind_est(self) -> tuple[Wind, dict]:
        now_s = self.get_clock().now().nanoseconds * 1e-9
        diag = {
            "source": "default",
            "age_s": float("inf"),
            "frame_id": str(self._wind_frame_id),
            "input_frame": str(self._wind_input_frame_used),
            "convention": str(self._wind_convention_used),
            "clipped": bool(self._wind_clipped),
        }

        if not bool(self.get_parameter("wind.use_topic").value):
            default = np.asarray(self.get_parameter("wind.default_ned").value, dtype=float).reshape(3)
            diag["source"] = "param"
            diag["age_s"] = 0.0
            return Wind(v_I=default), diag

        if self._wind_ned_to is None or self._wind_recv_time_s is None:
            default = np.asarray(self.get_parameter("wind.default_ned").value, dtype=float).reshape(3)
            diag["source"] = "default(no_msg)"
            diag["age_s"] = float("inf")
            return Wind(v_I=default), diag

        timeout_s = float(self.get_parameter("wind.timeout_s").value)
        age_s = float(now_s - self._wind_recv_time_s)
        diag["age_s"] = float(age_s)
        if timeout_s > 0.0 and age_s > timeout_s:
            default = np.asarray(self.get_parameter("wind.default_ned").value, dtype=float).reshape(3)
            diag["source"] = "default(stale)"
            return Wind(v_I=default), diag

        diag["source"] = "topic"
        return Wind(v_I=self._wind_ned_to), diag

    def _load_policy_config(self) -> TargetUpdatePolicyConfig:
        approach_allow_update = self.get_parameter("target.update_policy.approach_allow_update").value
        if isinstance(approach_allow_update, str):
            approach_allow_update = approach_allow_update.strip().lower()
        return TargetUpdatePolicyConfig(
            enabled=bool(self.get_parameter("target.update_policy.enabled").value),
            enable_hysteresis=bool(self.get_parameter("target.update_policy.enable_hysteresis").value),
            score_hysteresis=float(self.get_parameter("target.update_policy.score_hysteresis").value),
            dist_hysteresis_m=float(self.get_parameter("target.update_policy.dist_hysteresis_m").value),
            cruise_allow_update=bool(self.get_parameter("target.update_policy.cruise_allow_update").value),
            approach_allow_update=approach_allow_update,
            flare_lock=bool(self.get_parameter("target.update_policy.flare_lock").value),
            emergency_margin_mps=float(self.get_parameter("target.update_policy.emergency_margin_mps").value),
            emergency_cooldown_s=float(self.get_parameter("target.update_policy.emergency_cooldown_s").value),
            approach_significant_factor=float(self.get_parameter("target.update_policy.approach_significant_factor").value),
            smooth_transition=bool(self.get_parameter("target.update_policy.smooth_transition").value),
            smooth_rate_mps=float(self.get_parameter("target.update_policy.smooth_rate_mps").value),
        )

    def _compute_current_target_margin(self, state: State, wind: Wind, target_xy: np.ndarray) -> float:
        terrain = self.planner._load_terrain()
        terrain_h = float(terrain.height_m(float(target_xy[0]), float(target_xy[1]))) if terrain is not None else 0.0
        h_agl = float(state.altitude - terrain_h - float(self.planner.config.terrain_clearance_m))
        min_alt = float(self.get_parameter("safety.reachability.min_altitude_m").value)
        if h_agl < min_alt:
            return float("-inf")
        brake = float(self.get_parameter("safety.reachability.brake").value)
        _, sink = self._polar.interpolate(brake)
        if sink <= 1e-6:
            return float("-inf")
        tgo = float(h_agl / sink)
        min_time = float(self.get_parameter("safety.reachability.min_time_s").value)
        max_time = float(self.get_parameter("safety.reachability.max_time_s").value)
        if tgo < min_time or tgo > max_time:
            return float("-inf")
        d = np.asarray(target_xy, dtype=float) - state.position_xy
        v_req = d / max(tgo, 1e-6)
        v_air, _ = self._polar.interpolate(brake)
        margin = float(v_air - np.linalg.norm(v_req - wind.v_I[:2]))
        return margin

    def _time_to_land(self, state: State) -> float:
        terrain = self.planner._load_terrain()
        terrain_h = float(terrain.height_m(float(state.p_I[0]), float(state.p_I[1]))) if terrain is not None else 0.0
        h_agl = float(state.altitude - terrain_h - float(self.planner.config.terrain_clearance_m))
        min_alt = float(self.get_parameter("safety.reachability.min_altitude_m").value)
        if h_agl < min_alt:
            return -1.0
        brake = float(self.get_parameter("safety.reachability.brake").value)
        _, sink = self._polar.interpolate(brake)
        if sink <= 1e-6:
            return -1.0
        return float(h_agl / sink)

    def _target_for_planner(self, state: State, wind: Wind) -> Target:
        mode = str(self.get_parameter("target.auto_mode").value).strip().lower()
        if mode in {"manual", "off", "", "safety"} or self._target_ned is None:
            return Target(p_I=self._target_ned)
        if mode in {"current", "here"}:
            p = state.p_I.copy()
            p[2] = 0.0
            return Target(p_I=p)
        if mode in {"reach_center", "downwind"}:
            tgo = self._time_to_land(state)
            if tgo <= 0.0:
                return Target(p_I=self._target_ned)
            center = state.position_xy + wind.v_I[:2] * float(tgo)
            p = np.array([float(center[0]), float(center[1]), 0.0], dtype=float)
            return Target(p_I=p)
        return Target(p_I=self._target_ned)

    def _state_est(self) -> Optional[State]:
        if self._odom is None:
            return None

        odom = self._odom
        t = _stamp_to_sec(odom.header.stamp)

        # Pose in ENU -> NED
        p_enu = np.array(
            [odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z],
            dtype=float,
        )
        p_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)

        # Twist linear in ENU -> NED
        v_enu = np.array(
            [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z],
            dtype=float,
        )
        v_ned = np.array([v_enu[1], v_enu[0], -v_enu[2]], dtype=float)

        q_ned = _quat_enu_to_ned(odom.pose.pose.orientation)
        w_B = self._imu_w_B if self._imu_w_B is not None else np.array(
            [odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z], dtype=float
        )

        # Add configurable Gaussian noise
        p_ned = p_ned + self.rng.normal(0.0, self.noise_cfg.position_sigma, 3)
        v_ned = v_ned + self.rng.normal(0.0, self.noise_cfg.velocity_sigma, 3)
        w_B = w_B + self.rng.normal(0.0, self.noise_cfg.angular_rate_sigma, 3)

        # Attitude noise on roll/pitch/yaw
        from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy

        roll, pitch, yaw = quat_to_rpy(q_ned)
        nr, np_, ny = self.noise_cfg.attitude_sigma.tolist()
        roll += float(self.rng.normal(0.0, nr))
        pitch += float(self.rng.normal(0.0, np_))
        yaw += float(self.rng.normal(0.0, ny))
        q_ned = quat_from_euler(roll, pitch, yaw)

        return State(p_I=p_ned, v_I=v_ned, q_IB=q_ned, w_B=w_B, t=t)

    def _publish_path(self, trajectory) -> None:
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "world"

        for wp in trajectory.waypoints:
            pose = PoseStamped()
            pose.header = msg.header
            x, y, z = _ned_to_enu_xyz(wp.state.p_I)
            pose.pose.position.x = x
            pose.pose.position.y = y
            pose.pose.position.z = z
            qw, qx, qy, qz = _quat_ned_to_enu(wp.state.q_IB)
            pose.pose.orientation.w = qw
            pose.pose.orientation.x = qx
            pose.pose.orientation.y = qy
            pose.pose.orientation.z = qz
            msg.poses.append(pose)

        self.pub_path.publish(msg)
        self.pub_path_v2.publish(msg)
        self._publish_preview_markers(trajectory)

    def _publish_preview_markers(self, trajectory) -> None:
        stamp = self.get_clock().now().to_msg()
        frame_id = "world"

        line = Marker()
        line.header.stamp = stamp
        line.header.frame_id = frame_id
        line.ns = "planned_trajectory"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = 0.2
        line.color.r = 0.0
        line.color.g = 0.7
        line.color.b = 1.0
        line.color.a = 1.0

        for wp in trajectory.waypoints:
            x, y, z = _ned_to_enu_xyz(wp.state.p_I)
            pt = Point()
            pt.x = x
            pt.y = y
            pt.z = z
            line.points.append(pt)

        target_marker = Marker()
        target_marker.header.stamp = stamp
        target_marker.header.frame_id = frame_id
        target_marker.ns = "planned_trajectory"
        target_marker.id = 1
        target_marker.type = Marker.SPHERE
        target_marker.action = Marker.ADD
        target_marker.scale.x = 2.0
        target_marker.scale.y = 2.0
        target_marker.scale.z = 2.0
        target_marker.color.r = 1.0
        target_marker.color.g = 0.2
        target_marker.color.b = 0.2
        target_marker.color.a = 1.0
        if self._target_ned is not None:
            x, y, z = _ned_to_enu_xyz(self._target_ned)
            target_marker.pose.position.x = x
            target_marker.pose.position.y = y
            target_marker.pose.position.z = z
            target_marker.pose.orientation.w = 1.0

        safe_marker = None
        if self._safe_target_ned is not None:
            safe_marker = Marker()
            safe_marker.header.stamp = stamp
            safe_marker.header.frame_id = frame_id
            safe_marker.ns = "planned_trajectory"
            safe_marker.id = 2
            safe_marker.type = Marker.SPHERE
            safe_marker.action = Marker.ADD
            safe_marker.scale.x = 1.5
            safe_marker.scale.y = 1.5
            safe_marker.scale.z = 1.5
            safe_marker.color.r = 0.2
            safe_marker.color.g = 1.0
            safe_marker.color.b = 0.2
            safe_marker.color.a = 1.0
            x, y, z = _ned_to_enu_xyz(self._safe_target_ned)
            safe_marker.pose.position.x = x
            safe_marker.pose.position.y = y
            safe_marker.pose.position.z = z
            safe_marker.pose.orientation.w = 1.0

        aim_marker = None
        if self._aimpoint_ned is not None:
            aim_marker = Marker()
            aim_marker.header.stamp = stamp
            aim_marker.header.frame_id = frame_id
            aim_marker.ns = "planned_trajectory"
            aim_marker.id = 3
            aim_marker.type = Marker.SPHERE
            aim_marker.action = Marker.ADD
            aim_marker.scale.x = 1.2
            aim_marker.scale.y = 1.2
            aim_marker.scale.z = 1.2
            aim_marker.color.r = 0.2
            aim_marker.color.g = 0.4
            aim_marker.color.b = 1.0
            aim_marker.color.a = 1.0
            x, y, z = _ned_to_enu_xyz(self._aimpoint_ned)
            aim_marker.pose.position.x = x
            aim_marker.pose.position.y = y
            aim_marker.pose.position.z = z
            aim_marker.pose.orientation.w = 1.0

        arr = MarkerArray()
        arr.markers.append(line)
        arr.markers.append(target_marker)
        if safe_marker is not None:
            arr.markers.append(safe_marker)
        if aim_marker is not None:
            arr.markers.append(aim_marker)
        self.pub_preview.publish(arr)

    def _tick(self) -> None:
        state = self._state_est()
        if state is None or self._target_ned is None:
            return

        wind, wind_diag = self._wind_est()
        wind_speed = float(np.linalg.norm(wind.v_I[:2]))
        v_air_max, _ = self._polar.interpolate(0.0)
        wind_ratio = float(wind_speed / max(float(v_air_max), 1e-6))
        desired_target = self._target_for_planner(state, wind)

        if self._force_replan:
            self._target_policy.reset()

        selection_override = None
        policy_reason: UpdateReason | None = None
        planning_target = desired_target

        if self.planner.landing_site_selector is not None:
            try:
                terrain = self.planner._load_terrain()
                no_fly, no_fly_polygons = self.planner._build_no_fly()
                new_selection = self.planner.landing_site_selector.select(
                    state=state,
                    desired_target=desired_target,
                    wind=wind,
                    terrain=terrain,
                    no_fly_circles=no_fly,
                    no_fly_polygons=no_fly_polygons,
                )
            except Exception as e:  # pragma: no cover - defensive for ROS runtime
                self.get_logger().warn(f"Safety selector failed: {e}")
                new_selection = None

            if new_selection is not None:
                now_s = self.get_clock().now().nanoseconds * 1e-9
                if str(new_selection.reason) == "unreachable_wind":
                    # Force safety target update when wind makes the desired target unreachable.
                    self._target_policy.force_update(
                        new_selection.target,
                        float(new_selection.score),
                        float(new_selection.reach_margin_mps),
                        new_selection,
                        now_s,
                        reason=UpdateReason.UNREACHABLE_WIND,
                    )
                    planning_target = new_selection.target
                    policy_reason = UpdateReason.UNREACHABLE_WIND
                    selection_override = new_selection
                else:
                    current_target = self._target_policy.current_target or new_selection.target
                    current_margin = self._compute_current_target_margin(state, wind, current_target.position_xy)
                    planning_target, policy_reason = self._target_policy.update(
                        new_selection,
                        self._current_phase,
                        now_s,
                        current_margin,
                    )
                    policy_state = self._target_policy.current_state
                    selection_override = policy_state.selection if policy_state is not None else new_selection

        traj, info = self.planner.plan(state, planning_target, wind, landing_site_selection=selection_override)
        selection = self.planner.last_site_selection
        aimpoint_target = self.planner.last_aimpoint_target
        if selection is not None and selection.reason != "disabled":
            self._safe_target_ned = selection.target.p_I.copy()
        else:
            self._safe_target_ned = None
        if aimpoint_target is not None:
            self._aimpoint_ned = aimpoint_target.p_I.copy()
        else:
            self._aimpoint_ned = None
        self._publish_path(traj)
        self._force_replan = False

        s = String()
        safety_msg = ""
        if selection is not None and selection.reason != "disabled":
            meta = selection.metadata or {}
            tgo = meta.get("touchdown_tgo_s")
            v_air = meta.get("touchdown_v_air_mps")
            sink = meta.get("touchdown_sink_mps")
            v_g_along = meta.get("touchdown_v_g_along_max_mps")
            aim_used = meta.get("aimpoint_used")
            aim_n = meta.get("aimpoint_n")
            aim_e = meta.get("aimpoint_e")
            safety_msg = (
                f" safety={selection.reason} reason={selection.reason} risk={selection.risk:.3g} "
                f"dist={selection.distance_to_desired_m:.1f}m margin={selection.reach_margin_mps:.2f}mps"
            )
            if tgo is not None and np.isfinite(tgo):
                safety_msg += f" tgo={float(tgo):.1f}s"
            if v_air is not None and np.isfinite(v_air):
                safety_msg += f" v_air={float(v_air):.2f}"
            if sink is not None and np.isfinite(sink):
                safety_msg += f" sink={float(sink):.2f}"
            if v_g_along is not None and np.isfinite(v_g_along):
                safety_msg += f" v_g_along_max={float(v_g_along):.2f}"
            if aim_used is not None:
                safety_msg += f" aim_used={int(bool(aim_used))}"
            if aim_n is not None and aim_e is not None:
                safety_msg += f" aim={float(aim_n):.1f},{float(aim_e):.1f}"
            if policy_reason is not None:
                safety_msg += f" policy={policy_reason.value}"
        s.data = (
            f"success={info.success} iters={info.iterations} "
            f"time={info.solve_time:.3f}s cost={info.cost:.3f} "
            f"max_violation={info.max_violation:.3g} terminal_err={info.terminal_error_m:.2f}m "
            f"msg={info.message}"
            f" wind={wind.v_I[0]:+.2f},{wind.v_I[1]:+.2f},{wind.v_I[2]:+.2f}"
            f" wind_spd={wind_speed:.2f} V_air_max={float(v_air_max):.2f} ratio={wind_ratio:.2f}"
            f" wind_src={wind_diag.get('source')}"
            f" wind_age={float(wind_diag.get('age_s', 0.0)):.2f}s"
            f"{safety_msg}"
        )
        self.pub_status.publish(s)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PlannerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
