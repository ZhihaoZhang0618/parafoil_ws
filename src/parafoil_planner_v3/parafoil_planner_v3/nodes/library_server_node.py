from __future__ import annotations

from typing import Optional

import numpy as np

import rclpy
from rclpy.node import Node

from parafoil_planner_v3.trajectory_library.library_manager import TrajectoryLibrary

from parafoil_msgs.srv import QueryLibrary
from std_srvs.srv import Trigger


class LibraryServerNode(Node):
    def __init__(self) -> None:
        super().__init__("parafoil_trajectory_library_server")

        self.declare_parameter("library_path", "")
        self._library: Optional[TrajectoryLibrary] = None
        self._load()

        self.create_service(QueryLibrary, "/query_library", self._on_query)
        self.create_service(Trigger, "/reload_library", self._on_reload)

        self.get_logger().info("library_server_node started")

    def _load(self) -> None:
        path = str(self.get_parameter("library_path").value).strip()
        if not path:
            self._library = None
            return
        try:
            self._library = TrajectoryLibrary.load(path)
            self.get_logger().info(f"Loaded trajectory library: {len(self._library)} from {path}")
        except Exception as e:
            self._library = None
            self.get_logger().error(f"Failed to load library '{path}': {e}")

    def _on_query(self, req: QueryLibrary.Request, res: QueryLibrary.Response) -> QueryLibrary.Response:
        if self._library is None:
            res.success = False
            res.message = "library not loaded"
            return res

        k = int(req.k) if int(req.k) > 0 else 5
        features = np.array(
            [req.altitude_m, req.distance_m, req.bearing_rad, req.wind_speed_mps, req.wind_angle_rad], dtype=float
        )
        dist, idx = self._library.query_knn(features, k=k)

        idx_arr = np.atleast_1d(idx).astype(int)
        dist_arr = np.atleast_1d(dist).astype(float)
        types = [self._library[i].trajectory_type.value for i in idx_arr.tolist()]

        res.success = True
        res.message = f"ok k={len(idx_arr)}"
        res.indices = idx_arr.tolist()
        res.distances = dist_arr.tolist()
        res.trajectory_types = types
        return res

    def _on_reload(self, req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:  # noqa: ARG002
        self._load()
        res.success = self._library is not None
        res.message = "reloaded" if self._library is not None else "reload failed"
        return res


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LibraryServerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
