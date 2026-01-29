"""
E2E verification launch for parafoil_planner_v3.

This launch file supports:
- record_bag: Enable rosbag recording for post-analysis
- rviz: Launch RViz for visualization
- Configurable wind, target, and simulation parameters

Usage examples:
  # Basic verification run
  ros2 launch parafoil_planner_v3 e2e_verification.launch.py

  # With rosbag recording
  ros2 launch parafoil_planner_v3 e2e_verification.launch.py record_bag:=true bag_output:=/tmp/test_run

  # With RViz visualization
  ros2 launch parafoil_planner_v3 e2e_verification.launch.py rviz:=true

  # Custom scenario
  ros2 launch parafoil_planner_v3 e2e_verification.launch.py initial_altitude:=100 wind_steady_n:=3.0
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction, ExecuteProcess
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression, TextSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    run_sim = LaunchConfiguration("run_sim")
    planner_params = LaunchConfiguration("planner_params")
    gpm_params = LaunchConfiguration("gpm_params")
    dynamics_params = LaunchConfiguration("dynamics_params")
    guidance_params = LaunchConfiguration("guidance_params")
    use_library = LaunchConfiguration("use_library")
    library_path = LaunchConfiguration("library_path")
    start_library_server = LaunchConfiguration("start_library_server")

    # Recording and visualization
    record_bag = LaunchConfiguration("record_bag")
    bag_output = LaunchConfiguration("bag_output")
    record_mission_log = LaunchConfiguration("record_mission_log")
    mission_log_dir = LaunchConfiguration("mission_log_dir")
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    sim_params = LaunchConfiguration("sim_params")
    sim_params_override = LaunchConfiguration("sim_params_override")
    initial_altitude = LaunchConfiguration("initial_altitude")
    integrator_type = LaunchConfiguration("integrator_type")
    wind_enable_steady = LaunchConfiguration("wind_enable_steady")
    wind_enable_gust = LaunchConfiguration("wind_enable_gust")
    wind_enable_colored = LaunchConfiguration("wind_enable_colored")
    wind_steady_n = LaunchConfiguration("wind_steady_n")
    wind_steady_e = LaunchConfiguration("wind_steady_e")
    wind_steady_d = LaunchConfiguration("wind_steady_d")
    wind_gust_interval = LaunchConfiguration("wind_gust_interval")
    wind_gust_duration = LaunchConfiguration("wind_gust_duration")
    wind_gust_magnitude = LaunchConfiguration("wind_gust_magnitude")
    wind_colored_tau = LaunchConfiguration("wind_colored_tau")
    wind_colored_sigma = LaunchConfiguration("wind_colored_sigma")
    wind_seed = LaunchConfiguration("wind_seed")

    auto_target = LaunchConfiguration("auto_target")
    target_delay = LaunchConfiguration("target_delay_s")
    target_pub_rate = LaunchConfiguration("target_pub_rate")
    target_pub_times = LaunchConfiguration("target_pub_times")
    target_wait_subs = LaunchConfiguration("target_wait_subs")
    target_keep_alive = LaunchConfiguration("target_keep_alive_s")
    target_x = LaunchConfiguration("target_enu_x")
    target_y = LaunchConfiguration("target_enu_y")
    target_z = LaunchConfiguration("target_enu_z")

    default_planner = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "planner_params.yaml"])
    default_gpm = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "gpm_params.yaml"])
    default_dyn = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "dynamics_params.yaml"])
    default_guidance = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "guidance_params.yaml"])
    default_sim = PathJoinSubstitution([FindPackageShare("parafoil_simulator_ros"), "config", "params.yaml"])
    sim_launch = PathJoinSubstitution([FindPackageShare("parafoil_simulator_ros"), "launch", "sim.launch.py"])
    default_rviz = PathJoinSubstitution([FindPackageShare("parafoil_simulator_ros"), "config", "parafoil_integrated.rviz"])

    target_msg = PythonExpression(
        [
            "'{header: {frame_id: \"world\"}, pose: {position: {x: ' + str(",
            target_x,
            ") + ', y: ' + str(",
            target_y,
            ") + ', z: ' + str(",
            target_z,
            ") + '}}}'",
        ]
    )

    actions = [
        DeclareLaunchArgument("run_sim", default_value="true"),
        DeclareLaunchArgument("planner_params", default_value=default_planner),
        DeclareLaunchArgument("gpm_params", default_value=default_gpm),
        DeclareLaunchArgument("dynamics_params", default_value=default_dyn),
        DeclareLaunchArgument("guidance_params", default_value=default_guidance),
        DeclareLaunchArgument("use_library", default_value=TextSubstitution(text="false")),
        DeclareLaunchArgument("library_path", default_value=TextSubstitution(text="/tmp/parafoil_library.pkl")),
        DeclareLaunchArgument("start_library_server", default_value=TextSubstitution(text="true")),
        DeclareLaunchArgument("sim_params", default_value=default_sim),
        DeclareLaunchArgument("sim_params_override", default_value=""),
        DeclareLaunchArgument("initial_altitude", default_value="50.0"),
        DeclareLaunchArgument("integrator_type", default_value="rk4"),
        DeclareLaunchArgument("wind_enable_steady", default_value="true"),
        DeclareLaunchArgument("wind_enable_gust", default_value="false"),
        DeclareLaunchArgument("wind_enable_colored", default_value="false"),
        DeclareLaunchArgument("wind_steady_n", default_value="0.0"),
        DeclareLaunchArgument("wind_steady_e", default_value="2.0"),
        DeclareLaunchArgument("wind_steady_d", default_value="0.0"),
        DeclareLaunchArgument("wind_gust_interval", default_value="10.0"),
        DeclareLaunchArgument("wind_gust_duration", default_value="2.0"),
        DeclareLaunchArgument("wind_gust_magnitude", default_value="3.0"),
        DeclareLaunchArgument("wind_colored_tau", default_value="2.0"),
        DeclareLaunchArgument("wind_colored_sigma", default_value="1.0"),
        DeclareLaunchArgument("wind_seed", default_value="-1"),
        DeclareLaunchArgument("auto_target", default_value="true"),
        DeclareLaunchArgument("target_delay_s", default_value="3.0"),
        DeclareLaunchArgument("target_pub_rate", default_value="1.0"),
        DeclareLaunchArgument("target_pub_times", default_value="5"),
        DeclareLaunchArgument("target_wait_subs", default_value="1"),
        DeclareLaunchArgument("target_keep_alive_s", default_value="1.0"),
        DeclareLaunchArgument("target_enu_x", default_value="150.0"),
        DeclareLaunchArgument("target_enu_y", default_value="50.0"),
        DeclareLaunchArgument("target_enu_z", default_value="0.0"),
        # Recording and visualization
        DeclareLaunchArgument("record_bag", default_value="false", description="Enable rosbag recording"),
        DeclareLaunchArgument("bag_output", default_value="/tmp/parafoil_bag", description="Rosbag output directory"),
        DeclareLaunchArgument("record_mission_log", default_value="false", description="Enable mission_logger_node"),
        DeclareLaunchArgument("mission_log_dir", default_value="/tmp/mission_logs", description="Mission log output dir"),
        DeclareLaunchArgument("rviz", default_value="false", description="Launch RViz visualization"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz, description="RViz config file"),
    ]

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(sim_launch),
            condition=IfCondition(run_sim),
            launch_arguments={
                "params_file": sim_params,
                "params_override": sim_params_override,
                "initial_altitude": initial_altitude,
                "integrator_type": integrator_type,
                "wind_enable_steady": wind_enable_steady,
                "wind_enable_gust": wind_enable_gust,
                "wind_enable_colored": wind_enable_colored,
                "wind_steady_n": wind_steady_n,
                "wind_steady_e": wind_steady_e,
                "wind_steady_d": wind_steady_d,
                "wind_gust_interval": wind_gust_interval,
                "wind_gust_duration": wind_gust_duration,
                "wind_gust_magnitude": wind_gust_magnitude,
                "wind_colored_tau": wind_colored_tau,
                "wind_colored_sigma": wind_colored_sigma,
                "wind_seed": wind_seed,
            }.items(),
        )
    )

    actions.append(
        Node(
            package="parafoil_planner_v3",
            executable="planner_node",
            name="parafoil_planner_v3",
            output="screen",
            parameters=[
                planner_params,
                gpm_params,
                dynamics_params,
                {
                    "use_library": ParameterValue(use_library, value_type=bool),
                    "library_path": library_path,
                },
            ],
        )
    )
    actions.append(
        Node(
            package="parafoil_planner_v3",
            executable="library_server_node",
            name="parafoil_trajectory_library_server",
            output="screen",
            parameters=[{"library_path": library_path}],
            condition=IfCondition(start_library_server),
        )
    )
    actions.append(
        Node(
            package="parafoil_planner_v3",
            executable="guidance_node",
            name="parafoil_guidance_v3",
            output="screen",
            parameters=[guidance_params],
        )
    )

    actions.append(
        Node(
            package="parafoil_planner_v3",
            executable="mission_logger_node",
            name="parafoil_mission_logger",
            output="screen",
            parameters=[{"output_dir": mission_log_dir}],
            condition=IfCondition(record_mission_log),
        )
    )

    actions.append(
        TimerAction(
            period=target_delay,
            actions=[
                ExecuteProcess(
                    cmd=[
                        "ros2",
                        "topic",
                        "pub",
                        "-r",
                        target_pub_rate,
                        "-t",
                        target_pub_times,
                        "-w",
                        target_wait_subs,
                        "--keep-alive",
                        target_keep_alive,
                        "/target",
                        "geometry_msgs/msg/PoseStamped",
                        target_msg,
                    ],
                    output="screen",
                    condition=IfCondition(auto_target),
                )
            ],
        )
    )

    # Rosbag recording
    actions.append(
        ExecuteProcess(
            cmd=[
                "ros2", "bag", "record",
                "-o", bag_output,
                "/parafoil/odom",
                "/planned_trajectory",
                "/control_command",
                "/guidance_phase",
                "/planner_status",
                "/target",
                "/wind_estimate",
                "/parafoil/state",
            ],
            output="screen",
            condition=IfCondition(record_bag),
        )
    )

    # RViz visualization
    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(rviz),
        )
    )

    return LaunchDescription(actions)
