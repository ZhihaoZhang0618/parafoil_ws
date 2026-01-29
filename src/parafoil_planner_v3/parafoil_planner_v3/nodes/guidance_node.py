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
        self.declare_parameter("wind.filter.enable", True)
        self.declare_parameter("wind.filter.tau_s", 2.0)
        self.declare_parameter("wind.filter.max_delta_mps", 2.0)
        self.declare_parameter("wind.filter.gust_threshold_mps", 3.0)
        self.declare_parameter("wind.filter.gust_hold_s", 1.0)
        self.declare_parameter("target.position_ned", [0.0, 0.0, 0.0])

        # Phase manager / abort config
        self.declare_parameter("phase.approach_entry_distance", 180.0)
        self.declare_parameter("phase.altitude_margin", 1.1)
        self.declare_parameter("phase.glide_ratio_nominal", 3.0)
        self.declare_parameter("phase.approach_altitude_extra", 5.0)
        self.declare_parameter("phase.flare_altitude", 10.0)
        self.declare_parameter("phase.flare_distance", 30.0)
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
        self._wind: Optional[np.ndarray] = None
        self._target_ned: np.ndarray = np.asarray(self.get_parameter("target.position_ned").value, dtype=float).reshape(3)
        self._path_ned: list[np.ndarray] = []
        self._last_tick_time: float | None = None

        # Guidance components
        from parafoil_planner_v3.guidance.phase_manager import PhaseConfig

        phase_cfg = PhaseConfig(
            approach_entry_distance=float(self.get_parameter("phase.approach_entry_distance").value),
            altitude_margin=float(self.get_parameter("phase.altitude_margin").value),
            glide_ratio_nominal=float(self.get_parameter("phase.glide_ratio_nominal").value),
            approach_altitude_extra=float(self.get_parameter("phase.approach_altitude_extra").value),
            flare_altitude=float(self.get_parameter("phase.flare_altitude").value),
            flare_distance=float(self.get_parameter("phase.flare_distance").value),
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

        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self.get_logger().info("parafoil_guidance_v3 started")

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_wind(self, msg: Vector3Stamped) -> None:
        self._wind = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)

    def _on_target(self, msg: PoseStamped) -> None:
        p_enu = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self._target_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)

    def _on_path(self, msg: Path) -> None:
        # /planned_trajectory is in world ENU; convert to NED xy list.
        pts: list[np.ndarray] = []
        for pose in msg.poses:
            p_enu = pose.pose.position
            pts.append(np.array([p_enu.y, p_enu.x], dtype=float))
        self._path_ned = pts

    def _wind_est(self) -> Wind:
        if self._wind is not None:
            return Wind(v_I=self._wind)
        default = np.asarray(self.get_parameter("wind.default_ned").value, dtype=float).reshape(3)
        return Wind(v_I=default)

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
        i0 = self._closest_idx(p_xy)
        S = float(np.linalg.norm(self._path_ned[i0] - p_xy))
        for i in range(i0, len(self._path_ned) - 1):
            S += float(np.linalg.norm(self._path_ned[i + 1] - self._path_ned[i]))
        return float(S)

    def _tick(self) -> None:
        state = self._state()
        if state is None:
            return
        target = Target(p_I=self._target_ned)
        raw_wind = self._wind_est()
        dt = 0.0 if self._last_tick_time is None else float(max(state.t - self._last_tick_time, 0.0))
        self._last_tick_time = float(state.t)
        filtered = self._wind_filter.update(raw_wind.v_I, dt)
        wind = Wind(v_I=filtered)

        trans = self.phase_mgr.update(state, target, wind)
        if trans.triggered and trans.to_phase == GuidancePhase.FLARE:
            self.flare.on_enter(state)

        phase = self.phase_mgr.current_phase
        # If planner provided a path, follow it with a simple lookahead.
        L1 = float(self.get_parameter("L1_distance").value)
        track_xy = self._lookahead_point(state.position_xy, L1) if self._path_ned else None
        if track_xy is None:
            track_xy = target.position_xy

        # Energy-based symmetric brake: required slope ~ H / remaining distance.
        S_rem = self._remaining_distance(state.position_xy) if self._path_ned else float(
            np.linalg.norm(target.position_xy - state.position_xy)
        )
        H_rem = float(max(state.altitude - target.altitude, 0.0))
        k_req = 0.0 if S_rem < 1e-3 else float(H_rem / S_rem)
        b_slope = float(self.polar.select_brake_for_required_slope(k_req))

        if phase == GuidancePhase.CRUISE:
            # Use cruise logic but override tracking point if path exists.
            if self._path_ned and not self.cruise.should_override_path(wind):
                b = float(np.clip(self.cruise.config.brake_cruise, 0.0, float(self.get_parameter("brake_max").value)))
                from parafoil_planner_v3.guidance.control_laws import track_point_control

                control = track_point_control(state, track_xy, brake_sym=b, cfg=self.cruise.config.lateral)
            else:
                control = self.cruise.compute_control(state, target, wind, dt=0.0)
        elif phase == GuidancePhase.APPROACH:
            b = float(np.clip(b_slope, 0.0, float(self.get_parameter("brake_max").value)))
            if self._path_ned and not self.approach.should_override_path(state, target, wind, brake=b):
                from parafoil_planner_v3.guidance.control_laws import track_point_control

                control = track_point_control(state, track_xy, brake_sym=b, cfg=self.approach.config.lateral)
            else:
                control = self.approach.compute_control(state, target, wind, dt=0.0)
        elif phase == GuidancePhase.FLARE:
            if self._path_ned:
                # Flare brake schedule, but keep tracking last part of path.
                from parafoil_planner_v3.guidance.control_laws import track_point_control

                brake = float(self.flare.brake_command(state, target))
                control = track_point_control(state, track_xy, brake_sym=brake, cfg=self.flare.config.lateral)
            else:
                control = self.flare.compute_control(state, target, wind, dt=0.0)
        elif phase == GuidancePhase.ABORT:
            from parafoil_planner_v3.guidance.control_laws import track_point_control

            b = float(np.clip(float(self.get_parameter("abort_brake").value), 0.0, float(self.get_parameter("brake_max").value)))
            control = track_point_control(state, track_xy, brake_sym=b, cfg=self.approach.config.lateral)
        elif phase == GuidancePhase.LANDED:
            control = Control(0.0, 0.0)
        else:
            control = Control(0.0, 0.0)

        self._publish_control(control)

        msg = String()
        msg.data = phase.value
        self.pub_phase.publish(msg)


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
