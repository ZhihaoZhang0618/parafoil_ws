from __future__ import annotations

from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String

from parafoil_planner_v3.guidance.approach_guidance import ApproachGuidance
from parafoil_planner_v3.guidance.cruise_guidance import CruiseGuidance
from parafoil_planner_v3.guidance.flare_guidance import FlareGuidance
from parafoil_planner_v3.guidance.phase_manager import PhaseManager
from parafoil_planner_v3.guidance.wind_filter import WindFilter, WindFilterConfig
from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.types import Control, GuidancePhase, State, Target, Wind
from parafoil_planner_v3.utils.quaternion_utils import quat_to_rpy, wrap_pi
from parafoil_planner_v3.utils.wind_utils import (
    WindConvention,
    WindInputFrame,
    clip_wind_xy,
    frame_from_frame_id,
    parse_wind_convention,
    parse_wind_input_frame,
    to_ned_wind_to,
)


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


def _quat_enu_to_ned(q_enu) -> np.ndarray:
    return np.array([q_enu.w, q_enu.y, q_enu.x, -q_enu.z], dtype=float)


class GuidanceNode(Node):
    def __init__(self) -> None:
        super().__init__("parafoil_guidance_v3")

        # Parameters
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("L1_distance", 20.0)
        self.declare_parameter("brake_max", 0.6)
        self.declare_parameter("abort_brake", 0.3)
        self.declare_parameter("wind.use_topic", True)
        self.declare_parameter("wind.topic", "/wind_estimate")
        self.declare_parameter("wind.default_ned", [0.0, 2.0, 0.0])
        # /wind_estimate conversion -> internal standard: NED + wind-to
        self.declare_parameter("wind.input_frame", "ned")  # ned|enu|auto
        self.declare_parameter("wind.convention", "to")  # to|from
        self.declare_parameter("wind.timeout_s", 2.0)
        self.declare_parameter("wind.max_speed_mps", 0.0)  # 0 disables clipping
        self.declare_parameter("wind.filter.enable", True)
        self.declare_parameter("wind.filter.tau_s", 2.0)
        self.declare_parameter("wind.filter.max_delta_mps", 2.0)
        self.declare_parameter("wind.filter.gust_threshold_mps", 3.0)
        self.declare_parameter("wind.filter.gust_hold_s", 1.0)
        self.declare_parameter("target.position_ned", [0.0, 0.0, 0.0])

        # Path tracking / strong-wind L1 options
        self.declare_parameter("path_tracking.mode", "legacy")  # legacy|strong_wind_l1|auto
        self.declare_parameter("strong_wind_l1.enable", True)
        self.declare_parameter("strong_wind_l1.wind_ratio_trigger", 0.8)
        self.declare_parameter("strong_wind_l1.use_yaw_feedforward", True)
        self.declare_parameter("strong_wind_l1.min_air_speed_mps", 0.5)
        self.declare_parameter("strong_wind_l1.cte_gain", 1.0)
        self.declare_parameter("strong_wind_l1.cte_max_deg", 45.0)
        self.declare_parameter("strong_wind_l1.crosswind_margin_mps", 0.2)
        self.declare_parameter("strong_wind_l1.brake_adjust_enable", True)
        # When enabled, ignore Path yaw feedforward in strong wind and face (approximately) upwind instead.
        # This reduces downwind drift and implements "backward-drift" semantics even if the planned ground path
        # keeps moving downwind.
        self.declare_parameter("strong_wind_l1.force_upwind_yaw", False)

        # Phase manager / abort config
        self.declare_parameter("phase.approach_entry_distance", 180.0)
        self.declare_parameter("phase.altitude_margin", 1.1)
        self.declare_parameter("phase.glide_ratio_nominal", 3.0)
        self.declare_parameter("phase.approach_altitude_extra", 5.0)
        self.declare_parameter("phase.flare_altitude", 10.0)
        self.declare_parameter("phase.flare_distance", 30.0)
        self.declare_parameter("phase.flare_distance_altitude_factor", 2.0)
        self.declare_parameter("phase.abort_min_altitude_m", 2.0)
        self.declare_parameter("phase.abort_min_altitude_distance_m", 60.0)
        self.declare_parameter("phase.abort_max_glide_ratio", 10.0)
        self.declare_parameter("phase.abort_max_time_s", 300.0)
        self.declare_parameter("phase.abort_max_phase_time_s", 180.0)

        # Cruise guidance config
        self.declare_parameter("cruise.strategy", "auto")
        self.declare_parameter("cruise.final_leg_length", 180.0)
        self.declare_parameter("cruise.hold_threshold_alt_excess", 25.0)
        self.declare_parameter("cruise.racetrack_threshold_alt_excess", 60.0)
        self.declare_parameter("cruise.hold_radius", 40.0)
        self.declare_parameter("cruise.hold_direction", 1)
        self.declare_parameter("cruise.racetrack_length", 120.0)
        self.declare_parameter("cruise.racetrack_radius", 40.0)
        self.declare_parameter("cruise.s_turn_amplitude", 45.0)
        self.declare_parameter("cruise.s_turn_period_s", 30.0)
        self.declare_parameter("cruise.brake_cruise", 0.2)
        self.declare_parameter("cruise.strong_wind_ratio", 1.05)
        self.declare_parameter("cruise.strong_wind_entry_mode", "downwind")
        self.declare_parameter("cruise.strong_wind_force_direct", True)

        # Approach guidance config
        self.declare_parameter("approach.brake_min", 0.0)
        self.declare_parameter("approach.brake_max", 0.6)
        self.declare_parameter("approach.wind_correction_gain", 1.0)
        self.declare_parameter("approach.wind_max_drift_m", 80.0)
        self.declare_parameter("approach.min_ground_speed", 0.5)
        self.declare_parameter("approach.strong_wind_ratio", 1.05)
        self.declare_parameter("approach.strong_wind_strategy", "wind_line_projection")
        self.declare_parameter("approach.strong_wind_downwind_bias_m", 40.0)
        self.declare_parameter("approach.ground_speed_lookahead_m", 40.0)
        self.declare_parameter("approach.ground_speed_min_time_s", 5.0)
        self.declare_parameter("approach.ground_speed_max_time_s", 120.0)

        # Flare guidance config
        self.declare_parameter("flare.mode", "spec_full_brake")
        self.declare_parameter("flare.initial_brake", 0.2)
        self.declare_parameter("flare.max_brake", 1.0)
        self.declare_parameter("flare.ramp_time", 0.5)
        self.declare_parameter("flare.full_brake_duration_s", 3.0)
        self.declare_parameter("flare.touchdown_brake_altitude_m", 0.2)

        # State
        self._odom: Optional[Odometry] = None
        self._wind_ned_to: Optional[np.ndarray] = None
        self._wind_recv_time_s: Optional[float] = None
        self._wind_frame_id: str = ""
        self._wind_input_frame_used: str = ""
        self._wind_convention_used: str = ""
        self._wind_clipped: bool = False
        self._target_ned: np.ndarray = np.asarray(self.get_parameter("target.position_ned").value, dtype=float).reshape(3)
        self._path_ned: list[np.ndarray] = []
        self._path_yaw: list[float] = []
        self._path_s: list[float] = []
        self._last_tick_time: float | None = None
        self._last_diag_log_time: float | None = None
        self._last_track_mode: str = ""

        # Guidance components
        from parafoil_planner_v3.guidance.phase_manager import PhaseConfig

        phase_cfg = PhaseConfig(
            approach_entry_distance=float(self.get_parameter("phase.approach_entry_distance").value),
            altitude_margin=float(self.get_parameter("phase.altitude_margin").value),
            glide_ratio_nominal=float(self.get_parameter("phase.glide_ratio_nominal").value),
            approach_altitude_extra=float(self.get_parameter("phase.approach_altitude_extra").value),
            flare_altitude=float(self.get_parameter("phase.flare_altitude").value),
            flare_distance=float(self.get_parameter("phase.flare_distance").value),
            flare_distance_altitude_factor=float(self.get_parameter("phase.flare_distance_altitude_factor").value),
            abort_min_altitude_m=float(self.get_parameter("phase.abort_min_altitude_m").value),
            abort_min_altitude_distance_m=float(self.get_parameter("phase.abort_min_altitude_distance_m").value),
            abort_max_glide_ratio=float(self.get_parameter("phase.abort_max_glide_ratio").value),
            abort_max_time_s=float(self.get_parameter("phase.abort_max_time_s").value),
            abort_max_phase_time_s=float(self.get_parameter("phase.abort_max_phase_time_s").value),
        )
        self.phase_mgr = PhaseManager(phase_cfg)

        from parafoil_planner_v3.guidance.cruise_guidance import CruiseGuidanceConfig

        cruise_cfg = CruiseGuidanceConfig(
            final_leg_length=float(self.get_parameter("cruise.final_leg_length").value),
            hold_threshold_alt_excess=float(self.get_parameter("cruise.hold_threshold_alt_excess").value),
            racetrack_threshold_alt_excess=float(self.get_parameter("cruise.racetrack_threshold_alt_excess").value),
            hold_radius=float(self.get_parameter("cruise.hold_radius").value),
            hold_direction=int(self.get_parameter("cruise.hold_direction").value),
            racetrack_length=float(self.get_parameter("cruise.racetrack_length").value),
            racetrack_radius=float(self.get_parameter("cruise.racetrack_radius").value),
            s_turn_amplitude=float(self.get_parameter("cruise.s_turn_amplitude").value),
            s_turn_period_s=float(self.get_parameter("cruise.s_turn_period_s").value),
            strategy=str(self.get_parameter("cruise.strategy").value),
            brake_cruise=float(self.get_parameter("cruise.brake_cruise").value),
            strong_wind_ratio=float(self.get_parameter("cruise.strong_wind_ratio").value),
            strong_wind_entry_mode=str(self.get_parameter("cruise.strong_wind_entry_mode").value),
            strong_wind_force_direct=bool(self.get_parameter("cruise.strong_wind_force_direct").value),
        )
        self.cruise = CruiseGuidance(cruise_cfg)

        from parafoil_planner_v3.guidance.approach_guidance import ApproachGuidanceConfig

        approach_cfg = ApproachGuidanceConfig(
            brake_min=float(self.get_parameter("approach.brake_min").value),
            brake_max=float(self.get_parameter("approach.brake_max").value),
            wind_correction_gain=float(self.get_parameter("approach.wind_correction_gain").value),
            wind_max_drift_m=float(self.get_parameter("approach.wind_max_drift_m").value),
            min_ground_speed=float(self.get_parameter("approach.min_ground_speed").value),
            strong_wind_ratio=float(self.get_parameter("approach.strong_wind_ratio").value),
            strong_wind_strategy=str(self.get_parameter("approach.strong_wind_strategy").value),
            strong_wind_downwind_bias_m=float(self.get_parameter("approach.strong_wind_downwind_bias_m").value),
            ground_speed_lookahead_m=float(self.get_parameter("approach.ground_speed_lookahead_m").value),
            ground_speed_min_time_s=float(self.get_parameter("approach.ground_speed_min_time_s").value),
            ground_speed_max_time_s=float(self.get_parameter("approach.ground_speed_max_time_s").value),
        )
        self.approach = ApproachGuidance(approach_cfg)

        from parafoil_planner_v3.guidance.flare_guidance import FlareGuidanceConfig

        flare_cfg = FlareGuidanceConfig(
            mode=str(self.get_parameter("flare.mode").value),
            flare_initial_brake=float(self.get_parameter("flare.initial_brake").value),
            flare_max_brake=float(self.get_parameter("flare.max_brake").value),
            flare_ramp_time=float(self.get_parameter("flare.ramp_time").value),
            flare_full_brake_duration_s=float(self.get_parameter("flare.full_brake_duration_s").value),
            touchdown_brake_altitude_m=float(self.get_parameter("flare.touchdown_brake_altitude_m").value),
        )
        self.flare = FlareGuidance(flare_cfg)
        self.polar = PolarTable()
        self._wind_filter = WindFilter(
            WindFilterConfig(
                enable=bool(self.get_parameter("wind.filter.enable").value),
                tau_s=float(self.get_parameter("wind.filter.tau_s").value),
                max_delta_mps=float(self.get_parameter("wind.filter.max_delta_mps").value),
                gust_threshold_mps=float(self.get_parameter("wind.filter.gust_threshold_mps").value),
                gust_hold_s=float(self.get_parameter("wind.filter.gust_hold_s").value),
            )
        )

        # ROS I/O
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )
        self.create_subscription(Odometry, "/parafoil/odom", self._on_odom, 10)
        self.create_subscription(PoseStamped, "/target", self._on_target, 10)
        self.create_subscription(Path, "/planned_trajectory", self._on_path, 10)
        if bool(self.get_parameter("wind.use_topic").value):
            self.create_subscription(Vector3Stamped, str(self.get_parameter("wind.topic").value), self._on_wind, 10)

        self.pub_cmd = self.create_publisher(Vector3Stamped, "/control_command", 10)
        self.pub_cmd_sim = self.create_publisher(Vector3Stamped, "/rockpara_actuators_node/auto_commands", 10)
        self.pub_phase = self.create_publisher(String, "/guidance_phase", 10)
        self.pub_track_mode = self.create_publisher(String, "/path_tracking_mode", 10)

        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self.get_logger().info("parafoil_guidance_v3 started")

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_wind(self, msg: Vector3Stamped) -> None:
        raw = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)
        now_s = self.get_clock().now().nanoseconds * 1e-9

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
        self._wind_frame_id = str(msg.header.frame_id)
        self._wind_input_frame_used = input_frame_used.value
        self._wind_convention_used = convention.value
        self._wind_clipped = bool(clipped)

    def _on_target(self, msg: PoseStamped) -> None:
        p_enu = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self._target_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)

    def _on_path(self, msg: Path) -> None:
        # /planned_trajectory is in world ENU; convert to NED xy list.
        pts: list[np.ndarray] = []
        yaws: list[float] = []
        for pose in msg.poses:
            p_enu = pose.pose.position
            pts.append(np.array([p_enu.y, p_enu.x], dtype=float))
            q_enu = pose.pose.orientation
            q_ned = _quat_enu_to_ned(q_enu)
            _, _, yaw = quat_to_rpy(q_ned)
            yaws.append(float(yaw))
        self._path_ned = pts
        self._path_yaw = yaws
        self._path_s = self._polyline_cumulative_s(pts)

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

    def _state(self) -> Optional[State]:
        if self._odom is None:
            return None
        odom = self._odom
        t = _stamp_to_sec(odom.header.stamp)

        # ENU -> NED
        p_enu = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z], dtype=float)
        p_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)

        v_enu = np.array(
            [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z],
            dtype=float,
        )
        v_ned = np.array([v_enu[1], v_enu[0], -v_enu[2]], dtype=float)

        q_ned = _quat_enu_to_ned(odom.pose.pose.orientation)
        w_B = np.array([odom.twist.twist.angular.x, odom.twist.twist.angular.y, odom.twist.twist.angular.z], dtype=float)

        return State(p_I=p_ned, v_I=v_ned, q_IB=q_ned, w_B=w_B, t=t)

    def _publish_control(self, control: Control) -> None:
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "body"
        msg.vector.x = float(control.delta_L)
        msg.vector.y = float(control.delta_R)
        msg.vector.z = 0.0
        self.pub_cmd.publish(msg)
        self.pub_cmd_sim.publish(msg)

    def _closest_idx(self, p_xy: np.ndarray) -> int:
        if not self._path_ned:
            return 0
        d2 = [float(np.dot(w - p_xy, w - p_xy)) for w in self._path_ned]
        return int(np.argmin(d2))

    def _lookahead_point(self, p_xy: np.ndarray, L1: float) -> Optional[np.ndarray]:
        if len(self._path_ned) < 2:
            return None
        i0 = self._closest_idx(p_xy)
        dist = 0.0
        for i in range(i0, len(self._path_ned) - 1):
            p1 = self._path_ned[i]
            p2 = self._path_ned[i + 1]
            ds = float(np.linalg.norm(p2 - p1))
            if dist + ds >= L1:
                t = 0.0 if ds < 1e-6 else (L1 - dist) / ds
                return p1 + t * (p2 - p1)
            dist += ds
        return self._path_ned[-1].copy()

    def _remaining_distance(self, p_xy: np.ndarray) -> float:
        if len(self._path_ned) < 2:
            return 0.0
        s_proj, _cte, _seg_idx, _seg_t, _t_hat = self._project_to_path(p_xy)
        s_end = float(self._path_s[-1]) if self._path_s else 0.0
        return float(max(s_end - s_proj, 0.0))

    @staticmethod
    def _polyline_cumulative_s(path_xy: list[np.ndarray]) -> list[float]:
        if len(path_xy) < 2:
            return [0.0] * len(path_xy)
        s = [0.0]
        for i in range(1, len(path_xy)):
            s.append(s[-1] + float(np.linalg.norm(path_xy[i] - path_xy[i - 1])))
        return s

    def _project_to_path(self, p_xy: np.ndarray) -> tuple[float, float, int, float, np.ndarray]:
        """
        Project p_xy onto path polyline.
        Returns (s_proj, cross_track_error, seg_idx, seg_t, seg_t_hat).
        """
        if len(self._path_ned) < 2 or not self._path_s:
            return 0.0, 0.0, 0, 0.0, np.array([1.0, 0.0], dtype=float)
        p_xy = np.asarray(p_xy, dtype=float).reshape(2)
        best_d2 = float("inf")
        best_s = 0.0
        best_i = 0
        best_t = 0.0
        best_sign = 0.0
        best_t_hat = np.array([1.0, 0.0], dtype=float)
        for i in range(len(self._path_ned) - 1):
            a = self._path_ned[i]
            b = self._path_ned[i + 1]
            ab = b - a
            ab2 = float(np.dot(ab, ab))
            if ab2 < 1e-12:
                continue
            t = float(np.clip(np.dot(p_xy - a, ab) / ab2, 0.0, 1.0))
            proj = a + t * ab
            d = p_xy - proj
            d2 = float(np.dot(d, d))
            if d2 < best_d2:
                best_d2 = d2
                best_s = float(self._path_s[i] + t * np.sqrt(ab2))
                # NOTE: We use heading angles as bearing-from-N (atan2(E, N)), which is a clockwise-positive
                # convention. For the cross-track correction term to have the expected sign (positive cte -> steer
                # right), we flip the usual 2D cross-product sign here.
                cross = float(ab[0] * d[1] - ab[1] * d[0])
                best_sign = float(-np.sign(cross)) if abs(cross) > 1e-12 else 0.0
                best_i = i
                best_t = t
                n = float(np.linalg.norm(ab))
                best_t_hat = (ab / n) if n > 1e-9 else np.array([1.0, 0.0], dtype=float)
        return float(best_s), float(best_sign * np.sqrt(best_d2)), int(best_i), float(best_t), best_t_hat

    def _interp_on_path(self, s_query: float) -> tuple[np.ndarray, np.ndarray, int, float]:
        if len(self._path_ned) < 2 or not self._path_s:
            return np.array([0.0, 0.0], dtype=float), np.array([1.0, 0.0], dtype=float), 0, 0.0
        s_query = float(np.clip(s_query, self._path_s[0], self._path_s[-1]))
        i = int(np.searchsorted(np.asarray(self._path_s), s_query, side="right") - 1)
        i = int(np.clip(i, 0, len(self._path_ned) - 2))
        s0 = float(self._path_s[i])
        s1 = float(self._path_s[i + 1])
        a = self._path_ned[i]
        b = self._path_ned[i + 1]
        ds = float(max(s1 - s0, 1e-9))
        t = float(np.clip((s_query - s0) / ds, 0.0, 1.0))
        p = (1.0 - t) * a + t * b
        ab = b - a
        n = float(np.linalg.norm(ab))
        t_hat = (ab / n) if n > 1e-9 else np.array([1.0, 0.0], dtype=float)
        return p.astype(float), t_hat.astype(float), int(i), float(t)

    def _lookahead_pose(self, p_xy: np.ndarray, L1: float) -> tuple[np.ndarray, float | None, np.ndarray, float]:
        if len(self._path_ned) < 2 or not self._path_s:
            return p_xy.astype(float), None, np.array([1.0, 0.0], dtype=float), 0.0
        s_proj, cte, _seg_idx, _seg_t, _t_hat = self._project_to_path(p_xy)
        s_query = float(min(s_proj + float(L1), float(self._path_s[-1])))
        p_L1, t_hat, idx, t = self._interp_on_path(s_query)
        yaw_ff = None
        if self._path_yaw and len(self._path_yaw) == len(self._path_ned):
            y0 = float(self._path_yaw[idx])
            y1 = float(self._path_yaw[min(idx + 1, len(self._path_yaw) - 1)])
            yaw_ff = float(wrap_pi(y0 + float(wrap_pi(y1 - y0)) * t))
        return p_L1, yaw_ff, t_hat, float(cte)

    def _adjust_brake_for_crosswind(
        self,
        brake_sym: float,
        wind_xy: np.ndarray,
        t_hat: np.ndarray,
        brake_min: float,
        brake_max: float,
        margin: float,
    ) -> float:
        wind_xy = np.asarray(wind_xy, dtype=float).reshape(2)
        t_hat = np.asarray(t_hat, dtype=float).reshape(2)
        n = float(np.linalg.norm(t_hat))
        if n < 1e-9:
            return float(np.clip(brake_sym, brake_min, brake_max))
        t_hat = t_hat / n
        cross = float(np.linalg.norm(wind_xy - float(np.dot(wind_xy, t_hat)) * t_hat))
        candidates = [float(b) for b in self.polar.brake if brake_min - 1e-9 <= float(b) <= brake_max + 1e-9]
        candidates = sorted(candidates, reverse=True)
        for b in candidates:
            V_air, _ = self.polar.interpolate(float(b))
            if float(V_air) >= cross + float(margin):
                return float(np.clip(b, brake_min, brake_max))
        return float(np.clip(brake_min, brake_min, brake_max))

    def _tick(self) -> None:
        state = self._state()
        if state is None:
            return
        desired_target = Target(p_I=self._target_ned)
        raw_wind, _wind_diag = self._wind_est()
        dt = 0.0 if self._last_tick_time is None else float(max(state.t - self._last_tick_time, 0.0))
        self._last_tick_time = float(state.t)
        filtered = self._wind_filter.update(raw_wind.v_I, dt)
        wind = Wind(v_I=filtered)

        # In safety-first mode the planner may choose a different landing site than the desired /target.
        # Keep guidance phases (approach/flare/abort) consistent with the *planned* trajectory endpoint
        # when a path is available; otherwise fall back to the desired target.
        target_for_phase = desired_target
        if self._path_ned:
            last_xy = self._path_ned[-1]
            target_for_phase = Target(
                p_I=np.array([float(last_xy[0]), float(last_xy[1]), float(desired_target.p_I[2])], dtype=float)
            )

        trans = self.phase_mgr.update(state, target_for_phase, wind)
        if trans.triggered and trans.to_phase == GuidancePhase.FLARE:
            self.flare.on_enter(state)

        phase = self.phase_mgr.current_phase
        # If planner provided a path, follow it with a simple lookahead.
        L1 = float(self.get_parameter("L1_distance").value)
        track_xy = None
        yaw_ff = None
        t_hat = np.array([1.0, 0.0], dtype=float)
        cte = 0.0
        if self._path_ned:
            track_xy, yaw_ff, t_hat, cte = self._lookahead_pose(state.position_xy, L1)
        if track_xy is None:
            track_xy = desired_target.position_xy

        # Energy-based symmetric brake: required slope ~ H / remaining distance.
        S_rem = self._remaining_distance(state.position_xy) if self._path_ned else float(
            np.linalg.norm(desired_target.position_xy - state.position_xy)
        )
        H_rem = float(max(state.altitude - target_for_phase.altitude, 0.0))
        k_req = 0.0 if S_rem < 1e-3 else float(H_rem / S_rem)
        b_slope = float(self.polar.select_brake_for_required_slope(k_req))

        def _wind_ratio(brake_sym: float) -> float:
            wind_speed = float(np.linalg.norm(wind.v_I[:2]))
            v_air = np.asarray(state.v_I[:2], dtype=float) - np.asarray(wind.v_I[:2], dtype=float)
            Va_meas = float(np.linalg.norm(v_air))
            if Va_meas > 0.3:
                return float(wind_speed / max(Va_meas, 1e-6))
            V_air, _ = self.polar.interpolate(float(brake_sym))
            if float(V_air) <= 1e-6:
                return 0.0
            return float(wind_speed / float(V_air))

        def _is_strong_wind(brake_sym: float) -> bool:
            ratio = _wind_ratio(brake_sym)
            return bool(ratio >= float(self.get_parameter("strong_wind_l1.wind_ratio_trigger").value))

        def _use_strong_wind_l1(brake_sym: float) -> bool:
            if not self._path_ned:
                return False
            if not bool(self.get_parameter("strong_wind_l1.enable").value):
                return False
            mode = str(self.get_parameter("path_tracking.mode").value).strip().lower()
            if mode == "strong_wind_l1":
                return True
            if mode != "auto":
                return False
            return _is_strong_wind(brake_sym)

        def _strong_wind_cfg() -> "StrongWindL1Config":
            from parafoil_planner_v3.guidance.control_laws import StrongWindL1Config

            return StrongWindL1Config(
                min_air_speed=float(self.get_parameter("strong_wind_l1.min_air_speed_mps").value),
                cte_gain=float(self.get_parameter("strong_wind_l1.cte_gain").value),
                cte_max_rad=float(np.deg2rad(self.get_parameter("strong_wind_l1.cte_max_deg").value)),
                use_yaw_feedforward=bool(self.get_parameter("strong_wind_l1.use_yaw_feedforward").value),
            )

        track_mode = "legacy"
        strong_wind_debug: dict | None = None

        if phase == GuidancePhase.CRUISE:
            # Use cruise logic but override tracking point if path exists.
            b = float(np.clip(self.cruise.config.brake_cruise, 0.0, float(self.get_parameter("brake_max").value)))
            strong_wind = _is_strong_wind(b)
            if self._path_ned and (strong_wind or not self.cruise.should_override_path(wind)):
                if _use_strong_wind_l1(b):
                    from parafoil_planner_v3.guidance.control_laws import track_path_strong_wind_l1

                    yaw_ff_use = yaw_ff
                    if bool(self.get_parameter("strong_wind_l1.force_upwind_yaw").value) and strong_wind:
                        wxy = np.asarray(wind.v_I[:2], dtype=float)
                        if float(np.linalg.norm(wxy)) > 0.2:
                            yaw_ff_use = float(np.arctan2(-wxy[1], -wxy[0]))

                    if bool(self.get_parameter("strong_wind_l1.brake_adjust_enable").value):
                        b = self._adjust_brake_for_crosswind(
                            b,
                            wind.v_I[:2],
                            t_hat,
                            0.0,
                            float(self.get_parameter("brake_max").value),
                            float(self.get_parameter("strong_wind_l1.crosswind_margin_mps").value),
                        )
                    track_mode = "strong_wind_l1"
                    control, strong_wind_debug = track_path_strong_wind_l1(
                        state=state,
                        wind_xy=wind.v_I[:2],
                        target_xy=track_xy,
                        path_tangent_hat=t_hat,
                        yaw_ff=yaw_ff_use,
                        brake_sym=b,
                        cfg=self.cruise.config.lateral,
                        l1_dist=L1,
                        cross_track_error=cte,
                        sw_cfg=_strong_wind_cfg(),
                    )
                else:
                    from parafoil_planner_v3.guidance.control_laws import track_point_control

                    control = track_point_control(state, track_xy, brake_sym=b, cfg=self.cruise.config.lateral)
            else:
                control = self.cruise.compute_control(state, target_for_phase, wind, dt=0.0)
        elif phase == GuidancePhase.APPROACH:
            b = float(np.clip(b_slope, 0.0, float(self.get_parameter("brake_max").value)))
            strong_wind = _is_strong_wind(b)
            if self._path_ned and (
                strong_wind or not self.approach.should_override_path(state, target_for_phase, wind, brake=b)
            ):
                if _use_strong_wind_l1(b):
                    from parafoil_planner_v3.guidance.control_laws import track_path_strong_wind_l1

                    yaw_ff_use = yaw_ff
                    if bool(self.get_parameter("strong_wind_l1.force_upwind_yaw").value) and strong_wind:
                        wxy = np.asarray(wind.v_I[:2], dtype=float)
                        if float(np.linalg.norm(wxy)) > 0.2:
                            yaw_ff_use = float(np.arctan2(-wxy[1], -wxy[0]))

                    if bool(self.get_parameter("strong_wind_l1.brake_adjust_enable").value):
                        b = self._adjust_brake_for_crosswind(
                            b,
                            wind.v_I[:2],
                            t_hat,
                            float(self.get_parameter("approach.brake_min").value),
                            float(self.get_parameter("approach.brake_max").value),
                            float(self.get_parameter("strong_wind_l1.crosswind_margin_mps").value),
                        )
                    track_mode = "strong_wind_l1"
                    control, strong_wind_debug = track_path_strong_wind_l1(
                        state=state,
                        wind_xy=wind.v_I[:2],
                        target_xy=track_xy,
                        path_tangent_hat=t_hat,
                        yaw_ff=yaw_ff_use,
                        brake_sym=b,
                        cfg=self.approach.config.lateral,
                        l1_dist=L1,
                        cross_track_error=cte,
                        sw_cfg=_strong_wind_cfg(),
                    )
                else:
                    from parafoil_planner_v3.guidance.control_laws import track_point_control

                    control = track_point_control(state, track_xy, brake_sym=b, cfg=self.approach.config.lateral)
            else:
                control = self.approach.compute_control(state, target_for_phase, wind, dt=0.0)
        elif phase == GuidancePhase.FLARE:
            if self._path_ned:
                brake = float(self.flare.brake_command(state, target_for_phase))
                # In strong wind, keep using the air-relative path tracker during flare to avoid
                # downwind drift into no-fly corridors.
                if _use_strong_wind_l1(brake):
                    from parafoil_planner_v3.guidance.control_laws import track_path_strong_wind_l1

                    yaw_ff_use = yaw_ff
                    if bool(self.get_parameter("strong_wind_l1.force_upwind_yaw").value) and _is_strong_wind(brake):
                        wxy = np.asarray(wind.v_I[:2], dtype=float)
                        if float(np.linalg.norm(wxy)) > 0.2:
                            yaw_ff_use = float(np.arctan2(-wxy[1], -wxy[0]))

                    track_mode = "strong_wind_l1"
                    control, strong_wind_debug = track_path_strong_wind_l1(
                        state=state,
                        wind_xy=wind.v_I[:2],
                        target_xy=track_xy,
                        path_tangent_hat=t_hat,
                        yaw_ff=yaw_ff_use,
                        brake_sym=brake,
                        cfg=self.flare.config.lateral,
                        l1_dist=L1,
                        cross_track_error=cte,
                        sw_cfg=_strong_wind_cfg(),
                    )
                else:
                    # Flare brake schedule, but keep tracking last part of path.
                    from parafoil_planner_v3.guidance.control_laws import track_point_control

                    control = track_point_control(state, track_xy, brake_sym=brake, cfg=self.flare.config.lateral)
            else:
                control = self.flare.compute_control(state, target_for_phase, wind, dt=0.0)
        elif phase == GuidancePhase.ABORT:
            from parafoil_planner_v3.guidance.control_laws import track_point_control

            b = float(np.clip(float(self.get_parameter("abort_brake").value), 0.0, float(self.get_parameter("brake_max").value)))
            control = track_point_control(state, track_xy, brake_sym=b, cfg=self.approach.config.lateral)
        elif phase == GuidancePhase.LANDED:
            control = Control(0.0, 0.0)
        else:
            control = Control(0.0, 0.0)

        if track_mode and track_mode != self._last_track_mode:
            self.get_logger().info(f"path_tracking_mode={track_mode}")
            self._last_track_mode = track_mode

        if strong_wind_debug is not None:
            now_s = float(state.t)
            if self._last_diag_log_time is None or now_s - self._last_diag_log_time >= 1.0:
                self._last_diag_log_time = now_s
                wind_ratio = _wind_ratio(brake_sym=float(control.delta_L + control.delta_R) * 0.5)
                self.get_logger().info(
                    "strong_wind_l1"
                    f" chi_ref={strong_wind_debug.get('chi_ref', 0.0):.2f}"
                    f" chi_air={strong_wind_debug.get('chi_air', 0.0):.2f}"
                    f" chi_path={strong_wind_debug.get('chi_path', 0.0):.2f}"
                    f" chi_cte={strong_wind_debug.get('chi_cte', 0.0):.2f}"
                    f" Va={strong_wind_debug.get('Va', 0.0):.2f}"
                    f" wind_ratio={wind_ratio:.2f}"
                    f" cte={strong_wind_debug.get('cross_track_error', 0.0):.2f}"
                    f" yaw_rate_cmd={strong_wind_debug.get('yaw_rate_cmd', 0.0):.2f}"
                )

        self._publish_control(control)

        msg = String()
        msg.data = phase.value
        self.pub_phase.publish(msg)

        mode_msg = String()
        mode_msg.data = str(track_mode)
        self.pub_track_mode.publish(mode_msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GuidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
