from __future__ import annotations

import numpy as np

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Vector3Stamped

from parafoil_dynamics.wind import WindConfig, WindModel


class FakeWindEstimator(Node):
    """
    Simple wind estimate publisher for simulation.

    Publishes /wind_estimate as a NED + wind-to vector [m/s], consistent with
    parafoil_dynamics WindModel and parafoil_planner_v3 internal convention.
    """

    def __init__(self) -> None:
        super().__init__("fake_wind_estimator")

        # Wind config (should match simulator)
        self.declare_parameter("wind.enable_steady", True)
        self.declare_parameter("wind.enable_gust", True)
        self.declare_parameter("wind.enable_colored", False)
        self.declare_parameter("wind.steady_wind_n", 0.0)
        self.declare_parameter("wind.steady_wind_e", 2.0)
        self.declare_parameter("wind.steady_wind_d", 0.0)
        self.declare_parameter("wind.gust_interval", 10.0)
        self.declare_parameter("wind.gust_duration", 2.0)
        self.declare_parameter("wind.gust_magnitude", 3.0)
        self.declare_parameter("wind.colored_tau", 2.0)
        self.declare_parameter("wind.colored_sigma", 1.0)
        self.declare_parameter("wind.seed", -1)

        # Estimation error model
        self.declare_parameter("estimate.steady_bias_n", 0.0)
        self.declare_parameter("estimate.steady_bias_e", 0.0)
        self.declare_parameter("estimate.steady_bias_d", 0.0)
        self.declare_parameter("estimate.gust_scale", 1.0)
        self.declare_parameter("estimate.gust_noise_sigma", 0.5)
        self.declare_parameter("estimate.seed", -1)

        self.declare_parameter("publish_rate", 10.0)
        self.declare_parameter("topic", "/wind_estimate")

        steady = np.array(
            [
                float(self.get_parameter("wind.steady_wind_n").value),
                float(self.get_parameter("wind.steady_wind_e").value),
                float(self.get_parameter("wind.steady_wind_d").value),
            ],
            dtype=float,
        )

        wind_seed = int(self.get_parameter("wind.seed").value)
        wind_seed = None if wind_seed < 0 else wind_seed
        config = WindConfig(
            enable_steady=bool(self.get_parameter("wind.enable_steady").value),
            enable_gust=bool(self.get_parameter("wind.enable_gust").value),
            enable_colored=bool(self.get_parameter("wind.enable_colored").value),
            steady_wind=steady,
            gust_interval=float(self.get_parameter("wind.gust_interval").value),
            gust_duration=float(self.get_parameter("wind.gust_duration").value),
            gust_magnitude=float(self.get_parameter("wind.gust_magnitude").value),
            colored_tau=float(self.get_parameter("wind.colored_tau").value),
            colored_sigma=float(self.get_parameter("wind.colored_sigma").value),
            seed=wind_seed,
        )

        self._wind_model = WindModel(config)
        self._wind_model.reset()

        est_seed = int(self.get_parameter("estimate.seed").value)
        self._rng = np.random.default_rng(None if est_seed < 0 else est_seed)

        self._t0 = self.get_clock().now()
        self._last_t = 0.0

        topic = str(self.get_parameter("topic").value)
        self._pub = self.create_publisher(Vector3Stamped, topic, 10)
        rate = float(self.get_parameter("publish_rate").value)
        self._timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)

        self.get_logger().info(f"fake_wind_estimator started (publishing {topic})")

    def _tick(self) -> None:
        now = self.get_clock().now()
        t = (now - self._t0).nanoseconds * 1e-9
        dt = max(t - self._last_t, 1e-3)
        self._last_t = t

        true_wind = self._wind_model.get_wind(t, dt)

        steady = self._wind_model.config.steady_wind if self._wind_model.config.enable_steady else np.zeros(3)
        gust_true = true_wind - steady

        steady_bias = np.array(
            [
                float(self.get_parameter("estimate.steady_bias_n").value),
                float(self.get_parameter("estimate.steady_bias_e").value),
                float(self.get_parameter("estimate.steady_bias_d").value),
            ],
            dtype=float,
        )
        gust_scale = float(self.get_parameter("estimate.gust_scale").value)
        gust_noise_sigma = float(self.get_parameter("estimate.gust_noise_sigma").value)
        gust_noise = self._rng.normal(0.0, gust_noise_sigma, 3)

        est = steady + steady_bias + gust_scale * gust_true + gust_noise

        msg = Vector3Stamped()
        msg.header.stamp = now.to_msg()
        msg.header.frame_id = "ned"
        msg.vector.x = float(est[0])
        msg.vector.y = float(est[1])
        msg.vector.z = float(est[2])
        self._pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FakeWindEstimator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

