from __future__ import annotations

import json
from pathlib import Path
import json
from typing import Iterable, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Vector3Stamped
from nav_msgs.msg import Odometry

from parafoil_planner_v3.environment import NoFlyCircle, NoFlyPolygon, load_no_fly_polygons
from parafoil_planner_v3.dynamics.aerodynamics import PolarTable
from parafoil_planner_v3.landing_site_selector import RiskGrid


def _ned_to_enu_xyz(p_ned: np.ndarray) -> tuple[float, float, float]:
    p = np.asarray(p_ned, dtype=float).reshape(3)
    return float(p[1]), float(p[0]), float(-p[2])


def _parse_no_fly_circles(payload: str) -> list[NoFlyCircle]:
    payload = payload.strip()
    if not payload:
        return []
    parsed = json.loads(payload)
    circles: list[NoFlyCircle] = []
    if isinstance(parsed, list):
        for item in parsed:
            if isinstance(item, list) and len(item) >= 3:
                cn = float(item[0])
                ce = float(item[1])
                r = float(item[2])
                c = float(item[3]) if len(item) >= 4 else 0.0
                circles.append(NoFlyCircle(center_n=cn, center_e=ce, radius_m=r, clearance_m=c))
    return circles


def _parse_no_fly_polygons(payload: str) -> list[NoFlyPolygon]:
    payload = payload.strip()
    if not payload:
        return []
    parsed = json.loads(payload)
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
    return polygons


class SafetyVizNode(Node):
    def __init__(self) -> None:
        super().__init__("parafoil_safety_viz")

        self.declare_parameter("frame_id", "world")
        self.declare_parameter("publish_rate_hz", 1.0)
        self.declare_parameter("no_fly_circles", "[]")
        self.declare_parameter("no_fly_polygons", "[]")
        self.declare_parameter("no_fly_polygons_file", "")

        self.declare_parameter("risk_grid_file", "")
        self.declare_parameter("risk_clip_min", 0.0)
        self.declare_parameter("risk_clip_max", 1.0)
        self.declare_parameter("safe_threshold", 0.2)
        self.declare_parameter("danger_threshold", 0.7)
        self.declare_parameter("risk_stride", 4)
        self.declare_parameter("risk_z_m", 0.2)

        self.declare_parameter("nofly_height_m", 1.0)
        self.declare_parameter("nofly_alpha", 0.4)

        self.declare_parameter("reachability.show", True)
        self.declare_parameter("reachability.brake", 0.2)
        self.declare_parameter("reachability.min_altitude_m", 5.0)
        self.declare_parameter("reachability.terrain_height0_m", 0.0)
        self.declare_parameter("reachability.clearance_m", 0.0)
        self.declare_parameter("reachability.wind_margin_mps", 0.2)
        self.declare_parameter("reachability.wind_uncertainty_mps", 0.0)
        self.declare_parameter("reachability.gust_margin_mps", 0.0)
        self.declare_parameter("reachability.sample_count", 48)
        self.declare_parameter("reachability.line_width", 0.4)
        self.declare_parameter("reachability.color", [0.2, 0.6, 1.0])
        self.declare_parameter("reachability.z_m", 1.0)
        self.declare_parameter("reachability.show_line", True)
        self.declare_parameter("reachability.show_fill", True)
        self.declare_parameter("reachability.fill_stride_m", 8.0)
        self.declare_parameter("reachability.fill_alpha", 0.25)
        self.declare_parameter("reachability.fill_color", [0.1, 0.8, 1.0])
        # Use string to allow launch substitution; parsed in _wind_estimate.
        self.declare_parameter("wind_default_ned", "[0.0, 2.0, 0.0]")

        self._frame_id = str(self.get_parameter("frame_id").value)

        self._circles = _parse_no_fly_circles(str(self.get_parameter("no_fly_circles").value))
        self._polygons = _parse_no_fly_polygons(str(self.get_parameter("no_fly_polygons").value))
        file_path = str(self.get_parameter("no_fly_polygons_file").value).strip()
        if file_path:
            try:
                self._polygons.extend(load_no_fly_polygons(file_path))
            except Exception as e:
                self.get_logger().warn(f"Failed to load no-fly polygons '{file_path}': {e}")

        self._risk_grid: Optional[RiskGrid] = None
        grid_path = str(self.get_parameter("risk_grid_file").value).strip()
        if grid_path:
            try:
                self._risk_grid = RiskGrid.from_file(grid_path)
            except Exception as e:
                self.get_logger().warn(f"Failed to load risk grid '{grid_path}': {e}")

        self._publisher = self.create_publisher(MarkerArray, "/safety_viz", 10)

        self._last_odom: Optional[Odometry] = None
        self._last_wind: Optional[np.ndarray] = None
        self._polar = PolarTable()

        self.create_subscription(Odometry, "/parafoil/odom", self._on_odom, 10)
        self.create_subscription(Vector3Stamped, "/wind_estimate", self._on_wind, 10)

        rate = float(self.get_parameter("publish_rate_hz").value)
        self._timer = self.create_timer(1.0 / max(rate, 0.1), self._publish)

        self.get_logger().info("safety_viz_node started")

    def _on_odom(self, msg: Odometry) -> None:
        self._last_odom = msg

    def _on_wind(self, msg: Vector3Stamped) -> None:
        self._last_wind = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=float)

    def _make_cylinder_marker(self, idx: int, center_n: float, center_e: float, radius: float, color: tuple[float, float, float]) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame_id
        m.ns = "safety_viz"
        m.id = idx
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        x, y, z = _ned_to_enu_xyz(np.array([center_n, center_e, 0.0], dtype=float))
        m.pose.position.x = x
        m.pose.position.y = y
        m.pose.position.z = z
        m.pose.orientation.w = 1.0
        m.scale.x = float(radius * 2.0)
        m.scale.y = float(radius * 2.0)
        m.scale.z = float(self.get_parameter("nofly_height_m").value)
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        m.color.a = float(self.get_parameter("nofly_alpha").value)
        return m

    def _make_polygon_marker(self, idx: int, polygon: NoFlyPolygon, color: tuple[float, float, float]) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame_id
        m.ns = "safety_viz"
        m.id = idx
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.3
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        m.color.a = float(self.get_parameter("nofly_alpha").value)

        verts = polygon.vertices
        for v in verts:
            x, y, z = _ned_to_enu_xyz(np.array([v[0], v[1], 0.0], dtype=float))
            pt = Point(x=x, y=y, z=z)
            m.points.append(pt)
        # Close loop
        x, y, z = _ned_to_enu_xyz(np.array([verts[0, 0], verts[0, 1], 0.0], dtype=float))
        m.points.append(Point(x=x, y=y, z=z))
        return m

    def _make_risk_marker(self, idx: int, points: list[Point], color: tuple[float, float, float], scale: float, alpha: float) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame_id
        m.ns = "safety_viz"
        m.id = idx
        m.type = Marker.CUBE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(scale)
        m.scale.y = float(scale)
        m.scale.z = float(scale * 0.1)
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        m.color.a = float(alpha)
        m.points = points
        return m

    def _make_reachability_marker(self, idx: int, points: list[Point], color: tuple[float, float, float], line_width: float) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame_id
        m.ns = "safety_reach"
        m.id = idx
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(line_width)
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        m.color.a = 0.9
        m.points = points
        return m

    def _make_fill_marker(self, idx: int, points: list[Point], color: tuple[float, float, float], scale: float, alpha: float) -> Marker:
        m = Marker()
        m.header.frame_id = self._frame_id
        m.ns = "safety_reach_fill"
        m.id = idx
        m.type = Marker.CUBE_LIST
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = float(scale)
        m.scale.y = float(scale)
        m.scale.z = float(scale * 0.2)
        m.color.r = float(color[0])
        m.color.g = float(color[1])
        m.color.b = float(color[2])
        m.color.a = float(alpha)
        m.points = points
        return m

    def _wind_estimate(self) -> np.ndarray:
        if self._last_wind is not None:
            return self._last_wind
        default_raw = self.get_parameter("wind_default_ned").value
        if isinstance(default_raw, str):
            try:
                parsed = json.loads(default_raw)
            except Exception:
                parsed = [0.0, 0.0, 0.0]
        else:
            parsed = default_raw
        default = np.asarray(parsed, dtype=float).reshape(3)
        return default

    def _reachability_circle(self) -> tuple[float, float, float, float] | None:
        if self._last_odom is None:
            return None
        odom = self._last_odom
        p_enu = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z], dtype=float)
        p_ned = np.array([p_enu[1], p_enu[0], -p_enu[2]], dtype=float)
        altitude = float(-p_ned[2])

        min_alt = float(self.get_parameter("reachability.min_altitude_m").value)
        if altitude < min_alt:
            return None

        terrain_h = float(self.get_parameter("reachability.terrain_height0_m").value)
        clearance = float(self.get_parameter("reachability.clearance_m").value)
        h_agl = altitude - terrain_h - clearance
        if h_agl <= 0.1:
            return None

        brake = float(self.get_parameter("reachability.brake").value)
        v_air, sink = self._polar.interpolate(brake)
        if sink <= 1e-6:
            return None
        tgo = float(h_agl / sink)

        wind = self._wind_estimate()
        margin = float(self.get_parameter("reachability.wind_margin_mps").value)
        wind_unc = float(self.get_parameter("reachability.wind_uncertainty_mps").value)
        gust_margin = float(self.get_parameter("reachability.gust_margin_mps").value)
        v_air_eff = max(v_air - margin - wind_unc - gust_margin, 0.0)
        center_n = float(p_ned[0] + wind[0] * tgo)
        center_e = float(p_ned[1] + wind[1] * tgo)
        radius = float(v_air_eff * tgo)
        if radius <= 0.5:
            return None
        z_line = float(self.get_parameter("reachability.z_m").value)
        return center_n, center_e, radius, z_line

    def _reachability_points(self) -> list[Point]:
        circle = self._reachability_circle()
        if circle is None:
            return []
        center_n, center_e, radius, z_line = circle
        count = int(self.get_parameter("reachability.sample_count").value)
        count = max(count, 16)
        points: list[Point] = []
        for k in range(count + 1):
            theta = 2.0 * np.pi * float(k) / float(count)
            n = center_n + radius * np.cos(theta)
            e = center_e + radius * np.sin(theta)
            x, y, _ = _ned_to_enu_xyz(np.array([n, e, 0.0], dtype=float))
            points.append(Point(x=x, y=y, z=z_line))
        return points

    def _reachability_fill_points(self) -> tuple[list[Point], float]:
        circle = self._reachability_circle()
        if circle is None:
            return [], 0.0
        center_n, center_e, radius, z_line = circle
        stride = float(self.get_parameter("reachability.fill_stride_m").value)
        if stride <= 0.1:
            stride = 5.0
        # If risk grid exists, use its resolution as a baseline
        if self._risk_grid is not None:
            stride = max(stride, float(self._risk_grid.resolution_m))

        n_min = center_n - radius
        n_max = center_n + radius
        e_min = center_e - radius
        e_max = center_e + radius
        n_vals = np.arange(n_min, n_max + stride * 0.5, stride, dtype=float)
        e_vals = np.arange(e_min, e_max + stride * 0.5, stride, dtype=float)
        points: list[Point] = []
        r2 = float(radius * radius)
        for n in n_vals:
            dn = n - center_n
            for e in e_vals:
                de = e - center_e
                if (dn * dn + de * de) <= r2:
                    x, y, _ = _ned_to_enu_xyz(np.array([n, e, 0.0], dtype=float))
                    points.append(Point(x=x, y=y, z=z_line))
        return points, stride

        count = int(self.get_parameter("reachability.sample_count").value)
        count = max(count, 16)
        points: list[Point] = []
        z_line = float(self.get_parameter("reachability.z_m").value)
        for k in range(count + 1):
            theta = 2.0 * np.pi * float(k) / float(count)
            n = center_n + radius * np.cos(theta)
            e = center_e + radius * np.sin(theta)
            x, y, _ = _ned_to_enu_xyz(np.array([n, e, 0.0], dtype=float))
            points.append(Point(x=x, y=y, z=z_line))
        return points

    def _publish(self) -> None:
        arr = MarkerArray()
        idx = 0

        # No-fly circles
        for circle in self._circles:
            radius = float(circle.radius_m + circle.clearance_m)
            arr.markers.append(self._make_cylinder_marker(idx, circle.center_n, circle.center_e, radius, (1.0, 0.0, 0.0)))
            idx += 1

        # No-fly polygons
        for poly in self._polygons:
            arr.markers.append(self._make_polygon_marker(idx, poly, (1.0, 0.0, 0.0)))
            idx += 1

        # Risk grid visualization
        if self._risk_grid is not None:
            clip_min = float(self.get_parameter("risk_clip_min").value)
            clip_max = float(self.get_parameter("risk_clip_max").value)
            safe_th = float(self.get_parameter("safe_threshold").value)
            danger_th = float(self.get_parameter("danger_threshold").value)
            stride = int(self.get_parameter("risk_stride").value)
            stride = max(stride, 1)
            z = float(self.get_parameter("risk_z_m").value)

            safe_pts: list[Point] = []
            danger_pts: list[Point] = []

            grid = self._risk_grid
            h = grid.risk_map
            res = float(grid.resolution_m)
            for i in range(0, h.shape[0], stride):
                for j in range(0, h.shape[1], stride):
                    risk = float(np.clip(h[i, j], clip_min, clip_max))
                    n = float(grid.origin_n + i * res)
                    e = float(grid.origin_e + j * res)
                    x, y, _ = _ned_to_enu_xyz(np.array([n, e, 0.0], dtype=float))
                    pt = Point(x=x, y=y, z=z)
                    if risk <= safe_th:
                        safe_pts.append(pt)
                    elif risk >= danger_th:
                        danger_pts.append(pt)

            display_scale = res * stride
            if safe_pts:
                arr.markers.append(self._make_risk_marker(idx, safe_pts, (0.0, 0.8, 0.2), display_scale, 0.35))
                idx += 1
            if danger_pts:
                arr.markers.append(self._make_risk_marker(idx, danger_pts, (1.0, 0.6, 0.0), display_scale, 0.45))
                idx += 1

        # Reachability visualization (wind-shifted circle)
        if bool(self.get_parameter("reachability.show").value):
            if bool(self.get_parameter("reachability.show_fill").value):
                fill_pts, stride = self._reachability_fill_points()
                if fill_pts:
                    fill_color = self.get_parameter("reachability.fill_color").value
                    if isinstance(fill_color, (list, tuple)) and len(fill_color) >= 3:
                        c_fill = (float(fill_color[0]), float(fill_color[1]), float(fill_color[2]))
                    else:
                        c_fill = (0.1, 0.8, 1.0)
                    alpha = float(self.get_parameter("reachability.fill_alpha").value)
                    arr.markers.append(self._make_fill_marker(idx, fill_pts, c_fill, stride, alpha))
                    idx += 1
            if bool(self.get_parameter("reachability.show_line").value):
                pts = self._reachability_points()
                if pts:
                    color = self.get_parameter("reachability.color").value
                    if isinstance(color, (list, tuple)) and len(color) >= 3:
                        c = (float(color[0]), float(color[1]), float(color[2]))
                    else:
                        c = (0.2, 0.6, 1.0)
                    line_width = float(self.get_parameter("reachability.line_width").value)
                    arr.markers.append(self._make_reachability_marker(idx, pts, c, line_width))
                    idx += 1

        # Publish
        for m in arr.markers:
            m.header.stamp = self.get_clock().now().to_msg()
        self._publisher.publish(arr)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyVizNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
