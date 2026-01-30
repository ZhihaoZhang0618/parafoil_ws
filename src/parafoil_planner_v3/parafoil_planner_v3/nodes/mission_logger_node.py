from __future__ import annotations

from typing import Optional
import json

import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Vector3Stamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import String

from parafoil_planner_v3.logging.mission_logger import MissionLogger


def _stamp_to_sec(stamp) -> float:
    return float(stamp.sec) + 1e-9 * float(stamp.nanosec)


class MissionLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__("parafoil_mission_logger")

        self.declare_parameter("output_dir", "/tmp/mission_logs")
        self.declare_parameter("run_id", "")
        self.declare_parameter("log_rate_hz", 5.0)
        self.declare_parameter("auto_stop_on_landing", True)
        self.declare_parameter("landed_altitude_m", 0.0)
        self.declare_parameter("target_topic", "/target")

        self._odom: Optional[Odometry] = None
        self._control: Optional[Vector3Stamped] = None
        self._phase: Optional[str] = None
        self._planner_status: dict | str | None = None
        self._track_mode: Optional[str] = None
        self._target_ned = np.array([0.0, 0.0, 0.0], dtype=float)

        out_dir = str(self.get_parameter("output_dir").value)
        run_id = str(self.get_parameter("run_id").value)
        self.logger = MissionLogger(output_dir=out_dir, run_id=run_id, mode="ros2", tags=["mission_logger"])
        self.logger.log_config(scenario={"target_ned": self._target_ned.tolist()})

        self.create_subscription(Odometry, "/parafoil/odom", self._on_odom, 10)
        self.create_subscription(Vector3Stamped, "/control_command", self._on_control, 10)
        self.create_subscription(Path, "/planned_trajectory", self._on_path, 10)
        self.create_subscription(String, "/guidance_phase", self._on_phase, 10)
        self.create_subscription(String, "/planner_status", self._on_planner_status, 10)
        self.create_subscription(String, "/path_tracking_mode", self._on_track_mode, 10)
        self.create_subscription(PoseStamped, str(self.get_parameter("target_topic").value), self._on_target, 10)

        rate = float(self.get_parameter("log_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1e-3), self._tick)
        self._last_phase: Optional[str] = None

        self.get_logger().info("mission_logger_node started")

    def _on_odom(self, msg: Odometry) -> None:
        self._odom = msg

    def _on_control(self, msg: Vector3Stamped) -> None:
        self._control = msg

    def _on_phase(self, msg: String) -> None:
        self._phase = str(msg.data)

    def _on_planner_status(self, msg: String) -> None:
        text = str(msg.data)
        payload: dict | str | None = None
        if text.strip().startswith("{"):
            try:
                parsed = json.loads(text)
                payload = parsed if isinstance(parsed, dict) else text
            except json.JSONDecodeError:
                payload = text
        else:
            payload = text
        self._planner_status = payload

    def _on_track_mode(self, msg: String) -> None:
        self._track_mode = str(msg.data)

    def _on_target(self, msg: PoseStamped) -> None:
        p_enu = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=float)
        self._target_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)
        self.logger.log_config(scenario={"target_ned": self._target_ned.tolist()})

    def _on_path(self, msg: Path) -> None:
        pts = []
        for pose in msg.poses:
            p_enu = pose.pose.position
            pts.append([float(p_enu.y), float(p_enu.x)])
        if self._odom is not None:
            state = self._state_dict(self._odom)
            self.logger.log_planner_step(_stamp_to_sec(self._odom.header.stamp), state, {"path_xy": pts}, None)

    def _state_dict(self, odom: Odometry) -> dict:
        t = _stamp_to_sec(odom.header.stamp)
        p_enu = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z], dtype=float)
        v_enu = np.array(
            [odom.twist.twist.linear.x, odom.twist.twist.linear.y, odom.twist.twist.linear.z],
            dtype=float,
        )
        p_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)
        v_ned = np.array([v_enu[1], v_enu[0], -v_enu[2]], dtype=float)
        return {
            "position": p_ned.tolist(),
            "velocity": v_ned.tolist(),
            "t": float(t),
        }

    def _tick(self) -> None:
        if self._odom is None:
            return
        odom = self._odom
        ts = _stamp_to_sec(odom.header.stamp)
        state = self._state_dict(odom)
        self.logger.log_state(ts, state)

        if self._control is not None:
            ctrl = {"delta_L": float(self._control.vector.x), "delta_R": float(self._control.vector.y)}
            self.logger.log_control(ts, ctrl)

        if self._phase and self._phase != self._last_phase:
            self.logger.log_event(ts, "phase_transition", {"to": self._phase})
            self._last_phase = self._phase

        if self._planner_status is not None:
            self.logger.log_planner_status(ts, self._planner_status)
        if self._track_mode is not None:
            self.logger.log_tracking_mode(ts, self._track_mode)

        if bool(self.get_parameter("auto_stop_on_landing").value):
            altitude = float(-state["position"][2])
            if altitude <= float(self.get_parameter("landed_altitude_m").value):
                self.logger.log_event(ts, "touchdown")
                self._save_and_shutdown()

    def _save_and_shutdown(self) -> None:
        path = self.logger.save()
        self.get_logger().info(f"Mission log saved: {path}")
        self.destroy_node()
        rclpy.shutdown()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MissionLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.logger.save()
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()
