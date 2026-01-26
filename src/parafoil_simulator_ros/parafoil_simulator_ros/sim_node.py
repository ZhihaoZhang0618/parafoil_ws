"""
Parafoil Simulator ROS2 Node.

This node runs the 6DoF parafoil simulation and publishes sensor data.

Subscriptions:
    /rockpara_actuators_node/auto_commands (geometry_msgs/Vector3Stamped):
        x = delta_left_cmd, y = delta_right_cmd, z = unused
        
Publications:
    /position (geometry_msgs/Vector3Stamped): Position in NED [m]
    /body_acc (geometry_msgs/Vector3Stamped): Body-frame specific force [m/s^2]
    /body_ang_vel (geometry_msgs/Vector3Stamped): Body-frame angular velocity [rad/s]
    
RViz2 Visualization:
    /parafoil/odom (nav_msgs/Odometry): Odometry for RViz2
    /parafoil/path (nav_msgs/Path): Flight trajectory
    /parafoil/pose (geometry_msgs/PoseStamped): Current pose
    /parafoil/marker (visualization_msgs/Marker): Parafoil model marker
    TF: world -> parafoil_body
"""

import sys
import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from geometry_msgs.msg import Vector3Stamped, PoseStamped, Point, Quaternion, TransformStamped
from nav_msgs.msg import Odometry, Path
from visualization_msgs.msg import Marker, MarkerArray
from std_msgs.msg import ColorRGBA
from tf2_ros import TransformBroadcaster

# Import dynamics library
sys.path.insert(0, '/home/aims/parafoil_ws/src/parafoil_dynamics')
from parafoil_dynamics.state import State, ControlCmd
from parafoil_dynamics.params import Params
from parafoil_dynamics.dynamics import dynamics, get_body_acceleration
from parafoil_dynamics.integrators import (
    integrate_with_substeps, IntegratorType, parse_integrator_type
)
from parafoil_dynamics.wind import WindModel, WindConfig
from parafoil_dynamics.sensors import SensorModel, SensorConfig
from parafoil_dynamics.math3d import quat_to_euler


class ParafoilSimulatorNode(Node):
    """
    ROS2 node for parafoil 6DoF simulation.
    """
    
    def __init__(self):
        super().__init__('parafoil_simulator')
        
        # Declare and load parameters
        self._declare_parameters()
        self._load_parameters()
        
        # Initialize simulation state
        self._init_simulation()
        
        # Setup ROS interfaces
        self._setup_publishers()
        self._setup_subscribers()
        self._setup_timer()
        
        self.get_logger().info(
            f"Parafoil simulator started at {self.initial_altitude}m altitude"
        )
        self.get_logger().info(
            f"Using {self.integrator_type.value} integrator, "
            f"ctl_dt={self.ctl_dt}s, dt_max={self.dt_max}s"
        )
    
    def _declare_parameters(self):
        """Declare all ROS2 parameters."""
        # Simulation timing
        self.declare_parameter('ctl_dt', 0.02)  # Control period [s]
        self.declare_parameter('dt_max', 0.005)  # Max substep [s]
        self.declare_parameter('integrator_type', 'rk4')
        
        # Initial conditions
        self.declare_parameter('initial_position', [0.0, 0.0, -500.0])
        self.declare_parameter('initial_velocity', [10.0, 0.0, 2.0])
        self.declare_parameter('initial_euler', [0.0, 0.0, 0.0])  # roll, pitch, yaw
        
        # Physical parameters
        self.declare_parameter('rho', 1.225)
        self.declare_parameter('g', 9.81)
        self.declare_parameter('m', 250.0)
        self.declare_parameter('I_B_diag', [100.0, 50.0, 120.0])
        
        # Geometry
        self.declare_parameter('S', 30.0)
        self.declare_parameter('b', 10.0)
        self.declare_parameter('c', 3.0)
        self.declare_parameter('S_pd', 0.5)
        self.declare_parameter('c_D_pd', 1.0)
        self.declare_parameter('r_canopy_B', [0.0, 0.0, -3.0])
        self.declare_parameter('r_pd_B', [0.0, 0.0, 2.0])
        self.declare_parameter('pendulum_arm', 1.5)
        
        # Aerodynamic coefficients (from reference parafoil model)
        self.declare_parameter('c_L0', 0.24)
        self.declare_parameter('c_La', 2.14)
        self.declare_parameter('c_Lds', 0.28)
        self.declare_parameter('c_D0', 0.12)
        self.declare_parameter('c_Da2', 0.33)
        self.declare_parameter('c_Dds', 0.43)
        # Stall model
        self.declare_parameter('alpha_stall', 0.25)
        self.declare_parameter('alpha_stall_brake', 0.10)
        self.declare_parameter('alpha_stall_width', 0.05)
        self.declare_parameter('c_D_stall', 0.50)
        self.declare_parameter('c_Yb', -0.23)  # Negative: sideslip produces restoring force
        self.declare_parameter('c_lp', -0.84)
        self.declare_parameter('c_lda', -0.005)
        self.declare_parameter('c_m0', 0.1)
        self.declare_parameter('c_ma', -0.72)
        self.declare_parameter('c_mq', -1.49)
        self.declare_parameter('c_nr', -0.27)
        self.declare_parameter('c_nda', -0.133)
        self.declare_parameter('c_nb', 0.15)   # Wind-yaw coupling (positive: right wind -> turn right)
        self.declare_parameter('c_n_weath', 0.02)  # Weathercock effect (positive: turns away from headwind)

        # Actuator
        self.declare_parameter('tau_act', 0.2)
        
        # Numerical
        self.declare_parameter('eps', 1e-6)
        self.declare_parameter('V_min', 1.0)
        
        # Wind
        self.declare_parameter('wind.enable_steady', False)
        self.declare_parameter('wind.enable_gust', False)
        self.declare_parameter('wind.enable_colored', False)
        self.declare_parameter('wind.steady_wind', [0.0, 0.0, 0.0])
        self.declare_parameter('wind.gust_interval', 10.0)
        self.declare_parameter('wind.gust_duration', 2.0)
        self.declare_parameter('wind.gust_magnitude', 3.0)
        self.declare_parameter('wind.colored_tau', 2.0)
        self.declare_parameter('wind.colored_sigma', 1.0)
        self.declare_parameter('wind.seed', -1)  # -1 means random
        
        # Sensor noise
        self.declare_parameter('sensor.position_noise_std', [0.0, 0.0, 0.0])
        self.declare_parameter('sensor.accel_noise_std', [0.0, 0.0, 0.0])
        self.declare_parameter('sensor.gyro_noise_std', [0.0, 0.0, 0.0])
        self.declare_parameter('sensor.seed', -1)
    
    def _load_parameters(self):
        """Load parameters from ROS2 parameter server."""
        # Timing
        self.ctl_dt = self.get_parameter('ctl_dt').value
        self.dt_max = self.get_parameter('dt_max').value
        integrator_str = self.get_parameter('integrator_type').value
        self.integrator_type = parse_integrator_type(integrator_str)
        
        # Initial conditions
        self.initial_position = np.array(
            self.get_parameter('initial_position').value
        )
        self.initial_velocity = np.array(
            self.get_parameter('initial_velocity').value
        )
        self.initial_euler = np.array(
            self.get_parameter('initial_euler').value
        )
        self.initial_altitude = -self.initial_position[2]
        
        # Build Params object
        self.params = Params(
            rho=self.get_parameter('rho').value,
            g=self.get_parameter('g').value,
            m=self.get_parameter('m').value,
            I_B=np.diag(self.get_parameter('I_B_diag').value),
            S=self.get_parameter('S').value,
            b=self.get_parameter('b').value,
            c=self.get_parameter('c').value,
            S_pd=self.get_parameter('S_pd').value,
            c_D_pd=self.get_parameter('c_D_pd').value,
            r_pd_B=np.array(self.get_parameter('r_pd_B').value),
            c_L0=self.get_parameter('c_L0').value,
            c_La=self.get_parameter('c_La').value,
            c_Lds=self.get_parameter('c_Lds').value,
            c_D0=self.get_parameter('c_D0').value,
            c_Da2=self.get_parameter('c_Da2').value,
            c_Dds=self.get_parameter('c_Dds').value,
            alpha_stall=self.get_parameter('alpha_stall').value,
            alpha_stall_brake=self.get_parameter('alpha_stall_brake').value,
            alpha_stall_width=self.get_parameter('alpha_stall_width').value,
            c_D_stall=self.get_parameter('c_D_stall').value,
            c_Yb=self.get_parameter('c_Yb').value,
            c_lp=self.get_parameter('c_lp').value,
            c_lda=self.get_parameter('c_lda').value,
            c_m0=self.get_parameter('c_m0').value,
            c_ma=self.get_parameter('c_ma').value,
            c_mq=self.get_parameter('c_mq').value,
            c_nr=self.get_parameter('c_nr').value,
            c_nda=self.get_parameter('c_nda').value,
            c_nb=self.get_parameter('c_nb').value,
            c_n_weath=self.get_parameter('c_n_weath').value,
            tau_act=self.get_parameter('tau_act').value,
            eps=self.get_parameter('eps').value,
            V_min=self.get_parameter('V_min').value,
            r_canopy_B=np.array(self.get_parameter('r_canopy_B').value),
            pendulum_arm=self.get_parameter('pendulum_arm').value,
        )
        
        # Wind configuration
        wind_seed = self.get_parameter('wind.seed').value
        self.wind_config = WindConfig(
            enable_steady=self.get_parameter('wind.enable_steady').value,
            enable_gust=self.get_parameter('wind.enable_gust').value,
            enable_colored=self.get_parameter('wind.enable_colored').value,
            steady_wind=np.array(
                self.get_parameter('wind.steady_wind').value
            ),
            gust_interval=self.get_parameter('wind.gust_interval').value,
            gust_duration=self.get_parameter('wind.gust_duration').value,
            gust_magnitude=self.get_parameter('wind.gust_magnitude').value,
            colored_tau=self.get_parameter('wind.colored_tau').value,
            colored_sigma=self.get_parameter('wind.colored_sigma').value,
            seed=None if wind_seed < 0 else wind_seed,
        )
        
        # Sensor configuration
        sensor_seed = self.get_parameter('sensor.seed').value
        self.sensor_config = SensorConfig(
            position_noise_std=np.array(
                self.get_parameter('sensor.position_noise_std').value
            ),
            accel_noise_std=np.array(
                self.get_parameter('sensor.accel_noise_std').value
            ),
            gyro_noise_std=np.array(
                self.get_parameter('sensor.gyro_noise_std').value
            ),
            seed=None if sensor_seed < 0 else sensor_seed,
        )
    
    def _init_simulation(self):
        """Initialize simulation state and models."""
        from parafoil_dynamics.math3d import quat_from_euler
        
        # Create initial state
        q_IB = quat_from_euler(
            self.initial_euler[0],
            self.initial_euler[1],
            self.initial_euler[2]
        )
        
        self.state = State(
            p_I=self.initial_position.copy(),
            v_I=self.initial_velocity.copy(),
            q_IB=q_IB,
            w_B=np.zeros(3),
            delta=np.zeros(2),
            t=0.0
        )
        
        # Current control command
        self.cmd = ControlCmd(delta_cmd=np.zeros(2))
        
        # Wind model
        self.wind_model = WindModel(self.wind_config)
        self.wind_model.reset()
        
        # Sensor model
        self.sensor_model = SensorModel(self.sensor_config)
        
        # Simulation running flag
        self.is_running = True
        
        # Statistics
        self.step_count = 0
        
        # Path history for visualization
        self.path_msg = Path()
        self.path_msg.header.frame_id = 'world'
    
    def _setup_publishers(self):
        """Setup ROS2 publishers."""
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        self.pub_position = self.create_publisher(
            Vector3Stamped, '/position', qos
        )
        self.pub_body_acc = self.create_publisher(
            Vector3Stamped, '/body_acc', qos
        )
        self.pub_body_ang_vel = self.create_publisher(
            Vector3Stamped, '/body_ang_vel', qos
        )
        
        # RViz2 visualization publishers
        qos_reliable = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        
        self.pub_odom = self.create_publisher(
            Odometry, '/parafoil/odom', qos_reliable
        )
        self.pub_path = self.create_publisher(
            Path, '/parafoil/path', qos_reliable
        )
        self.pub_pose = self.create_publisher(
            PoseStamped, '/parafoil/pose', qos_reliable
        )
        self.pub_marker = self.create_publisher(
            Marker, '/parafoil/marker', qos_reliable
        )
        
        # TF broadcaster
        self.tf_broadcaster = TransformBroadcaster(self)
    
    def _setup_subscribers(self):
        """Setup ROS2 subscribers."""
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT
        )
        
        self.sub_cmd = self.create_subscription(
            Vector3Stamped,
            '/rockpara_actuators_node/auto_commands',
            self._cmd_callback,
            qos
        )
    
    def _setup_timer(self):
        """Setup simulation timer."""
        self.timer = self.create_timer(self.ctl_dt, self._timer_callback)
    
    def _cmd_callback(self, msg: Vector3Stamped):
        """
        Handle incoming control commands.
        
        Args:
            msg: Vector3Stamped with x=delta_left, y=delta_right
        """
        delta_l = np.clip(msg.vector.x, 0.0, 1.0)
        delta_r = np.clip(msg.vector.y, 0.0, 1.0)
        self.cmd = ControlCmd(delta_cmd=np.array([delta_l, delta_r]))
    
    def _timer_callback(self):
        """Main simulation loop callback."""
        if not self.is_running:
            return
        
        # Check for landing
        if self.state.is_on_ground:
            self._handle_landing()
            return
        
        # Get wind at current time
        def wind_fn(t):
            return self.wind_model.get_wind(t, self.dt_max)
        
        # Integrate dynamics
        try:
            self.state = integrate_with_substeps(
                dynamics,
                self.state,
                self.cmd,
                self.params,
                self.ctl_dt,
                self.dt_max,
                self.integrator_type,
                wind_fn
            )
        except RuntimeError as e:
            self.get_logger().error(f"Integration error: {e}")
            self.is_running = False
            self.timer.cancel()
            return
        
        # Compute body-frame acceleration for sensors
        wind_I = self.wind_model.get_wind(self.state.t, self.dt_max)
        body_acc = get_body_acceleration(
            self.state, self.cmd, self.params, wind_I
        )
        
        # Get sensor measurements
        measurement = self.sensor_model.get_measurement(
            self.state, self.params, body_acc
        )
        
        # Publish measurements
        self._publish_measurements(measurement)
        
        # Publish RViz2 visualization
        self._publish_rviz_visualization()
        
        self.step_count += 1
        
        # Log periodically
        if self.step_count % 100 == 0:
            self._log_status()
    
    def _publish_measurements(self, measurement):
        """Publish sensor measurements to ROS topics."""
        stamp = self.get_clock().now().to_msg()
        
        # Position
        pos_msg = Vector3Stamped()
        pos_msg.header.stamp = stamp
        pos_msg.header.frame_id = 'ned'
        pos_msg.vector.x = measurement.position[0]
        pos_msg.vector.y = measurement.position[1]
        pos_msg.vector.z = measurement.position[2]
        self.pub_position.publish(pos_msg)
        
        # Body acceleration (specific force)
        acc_msg = Vector3Stamped()
        acc_msg.header.stamp = stamp
        acc_msg.header.frame_id = 'body'
        acc_msg.vector.x = measurement.body_acc[0]
        acc_msg.vector.y = measurement.body_acc[1]
        acc_msg.vector.z = measurement.body_acc[2]
        self.pub_body_acc.publish(acc_msg)
        
        # Body angular velocity
        gyro_msg = Vector3Stamped()
        gyro_msg.header.stamp = stamp
        gyro_msg.header.frame_id = 'body'
        gyro_msg.vector.x = measurement.body_ang_vel[0]
        gyro_msg.vector.y = measurement.body_ang_vel[1]
        gyro_msg.vector.z = measurement.body_ang_vel[2]
        self.pub_body_ang_vel.publish(gyro_msg)
    
    def _publish_rviz_visualization(self):
        """Publish visualization messages for RViz2."""
        stamp = self.get_clock().now().to_msg()
        
        # Convert NED to ENU for RViz2 (RViz uses ENU convention)
        # NED: x=North, y=East, z=Down
        # ENU: x=East, y=North, z=Up
        pos_enu = np.array([
            self.state.p_I[1],   # East
            self.state.p_I[0],   # North  
            -self.state.p_I[2]   # Up (negative of Down)
        ])
        
        # Convert quaternion from NED to ENU
        # q_NED = [w, x, y, z] -> q_ENU needs rotation
        q_ned = self.state.q_IB
        # For NED to ENU: rotate 90 deg around Z, then 180 deg around X
        # Simplified: swap x,y and negate z for visualization
        q_enu = Quaternion()
        q_enu.w = q_ned[0]
        q_enu.x = q_ned[2]   # swap
        q_enu.y = q_ned[1]   # swap
        q_enu.z = -q_ned[3]  # negate
        
        # 1. Publish PoseStamped
        pose_msg = PoseStamped()
        pose_msg.header.stamp = stamp
        pose_msg.header.frame_id = 'world'
        pose_msg.pose.position.x = pos_enu[0]
        pose_msg.pose.position.y = pos_enu[1]
        pose_msg.pose.position.z = pos_enu[2]
        pose_msg.pose.orientation = q_enu
        self.pub_pose.publish(pose_msg)
        
        # 2. Publish Odometry
        odom_msg = Odometry()
        odom_msg.header.stamp = stamp
        odom_msg.header.frame_id = 'world'
        odom_msg.child_frame_id = 'parafoil_body'
        odom_msg.pose.pose = pose_msg.pose
        
        # Velocity in ENU
        odom_msg.twist.twist.linear.x = self.state.v_I[1]   # East
        odom_msg.twist.twist.linear.y = self.state.v_I[0]   # North
        odom_msg.twist.twist.linear.z = -self.state.v_I[2]  # Up
        odom_msg.twist.twist.angular.x = self.state.w_B[0]
        odom_msg.twist.twist.angular.y = self.state.w_B[1]
        odom_msg.twist.twist.angular.z = self.state.w_B[2]
        self.pub_odom.publish(odom_msg)
        
        # 3. Add to path and publish (every 10 steps to reduce message size)
        if self.step_count % 10 == 0:
            self.path_msg.header.stamp = stamp
            self.path_msg.poses.append(pose_msg)
            
            # Limit path length to avoid memory issues
            max_path_length = 1000
            if len(self.path_msg.poses) > max_path_length:
                self.path_msg.poses = self.path_msg.poses[-max_path_length:]
            
            self.pub_path.publish(self.path_msg)
        
        # 4. Publish TF transform
        tf_msg = TransformStamped()
        tf_msg.header.stamp = stamp
        tf_msg.header.frame_id = 'world'
        tf_msg.child_frame_id = 'parafoil_body'
        tf_msg.transform.translation.x = pos_enu[0]
        tf_msg.transform.translation.y = pos_enu[1]
        tf_msg.transform.translation.z = pos_enu[2]
        tf_msg.transform.rotation = q_enu
        self.tf_broadcaster.sendTransform(tf_msg)
        
        # 5. Publish parafoil marker (arrow showing direction)
        marker_msg = Marker()
        marker_msg.header.stamp = stamp
        marker_msg.header.frame_id = 'world'
        marker_msg.ns = 'parafoil'
        marker_msg.id = 0
        marker_msg.type = Marker.ARROW
        marker_msg.action = Marker.ADD
        marker_msg.pose = pose_msg.pose
        marker_msg.scale.x = 5.0   # Arrow length
        marker_msg.scale.y = 2.0   # Arrow width
        marker_msg.scale.z = 1.0   # Arrow height
        marker_msg.color.r = 0.2
        marker_msg.color.g = 0.6
        marker_msg.color.b = 1.0
        marker_msg.color.a = 1.0
        self.pub_marker.publish(marker_msg)
        
        # 6. Publish canopy marker (box representing the parafoil canopy)
        canopy_marker = Marker()
        canopy_marker.header.stamp = stamp
        canopy_marker.header.frame_id = 'parafoil_body'
        canopy_marker.ns = 'parafoil'
        canopy_marker.id = 1
        canopy_marker.type = Marker.CUBE
        canopy_marker.action = Marker.ADD
        canopy_marker.pose.position.x = 0.0
        canopy_marker.pose.position.y = 0.0
        canopy_marker.pose.position.z = 2.0  # Above body frame
        canopy_marker.pose.orientation.w = 1.0
        canopy_marker.scale.x = self.params.c  # Chord
        canopy_marker.scale.y = self.params.b  # Span
        canopy_marker.scale.z = 0.3            # Thickness
        canopy_marker.color.r = 1.0
        canopy_marker.color.g = 0.5
        canopy_marker.color.b = 0.0
        canopy_marker.color.a = 0.8
        self.pub_marker.publish(canopy_marker)
    
    def _log_status(self):
        """Log current simulation status."""
        roll, pitch, yaw = quat_to_euler(self.state.q_IB)
        
        self.get_logger().info(
            f"t={self.state.t:.1f}s | "
            f"alt={self.state.altitude:.1f}m | "
            f"pos=({self.state.p_I[0]:.1f}, {self.state.p_I[1]:.1f}) | "
            f"v={np.linalg.norm(self.state.v_I):.1f}m/s | "
            f"yaw={np.degrees(yaw):.1f}deg | "
            f"brake=({self.cmd.delta_cmd[0]:.2f}, {self.cmd.delta_cmd[1]:.2f})"
        )
    
    def _handle_landing(self):
        """Handle landing event."""
        # Calculate landing statistics
        flight_time = self.state.t
        landing_pos = self.state.p_I.copy()
        distance = np.sqrt(landing_pos[0]**2 + landing_pos[1]**2)
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("LANDING DETECTED")
        self.get_logger().info(f"Flight time: {flight_time:.1f} s")
        self.get_logger().info(
            f"Landing position: N={landing_pos[0]:.1f}m, "
            f"E={landing_pos[1]:.1f}m"
        )
        self.get_logger().info(f"Distance from origin: {distance:.1f} m")
        self.get_logger().info(f"Total simulation steps: {self.step_count}")
        self.get_logger().info("=" * 50)
        
        # Stop simulation
        self.is_running = False
        self.timer.cancel()
        
        # Shutdown node
        self.get_logger().info("Simulation complete. Shutting down.")
        rclpy.shutdown()


def main(args=None):
    """Main entry point."""
    rclpy.init(args=args)
    
    try:
        node = ParafoilSimulatorNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except rclpy.executors.ExternalShutdownException:
        pass
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
