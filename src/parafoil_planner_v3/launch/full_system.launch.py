from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression, TextSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    run_sim = LaunchConfiguration("run_sim")
    planner_params = LaunchConfiguration("planner_params")
    gpm_params = LaunchConfiguration("gpm_params")
    dynamics_params = LaunchConfiguration("dynamics_params")
    guidance_params = LaunchConfiguration("guidance_params")
    use_library = LaunchConfiguration("use_library")
    library_path = LaunchConfiguration("library_path")
    start_library_server = LaunchConfiguration("start_library_server")
    start_wind_estimator = LaunchConfiguration("start_wind_estimator")
    wind_topic = LaunchConfiguration("wind_topic")

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

    default_planner = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "planner_params.yaml"])
    default_gpm = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "gpm_params.yaml"])
    default_dyn = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "dynamics_params.yaml"])
    default_guidance = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "guidance_params.yaml"])

    default_sim = PathJoinSubstitution([FindPackageShare("parafoil_simulator_ros"), "config", "params.yaml"])
    sim_launch = PathJoinSubstitution([FindPackageShare("parafoil_simulator_ros"), "launch", "sim.launch.py"])

    actions = [
        DeclareLaunchArgument("run_sim", default_value="true"),
        DeclareLaunchArgument("planner_params", default_value=default_planner),
        DeclareLaunchArgument("gpm_params", default_value=default_gpm),
        DeclareLaunchArgument("dynamics_params", default_value=default_dyn),
        DeclareLaunchArgument("guidance_params", default_value=default_guidance),
        DeclareLaunchArgument("use_library", default_value=TextSubstitution(text="false")),
        DeclareLaunchArgument("library_path", default_value=TextSubstitution(text="/tmp/parafoil_library.pkl")),
        DeclareLaunchArgument("start_library_server", default_value=TextSubstitution(text="true")),
        DeclareLaunchArgument("start_wind_estimator", default_value=TextSubstitution(text="true")),
        DeclareLaunchArgument("wind_topic", default_value=TextSubstitution(text="/wind_estimate")),
        DeclareLaunchArgument("sim_params", default_value=default_sim),
        DeclareLaunchArgument("sim_params_override", default_value=""),
        DeclareLaunchArgument("initial_altitude", default_value="50.0"),
        DeclareLaunchArgument("integrator_type", default_value="rk4"),
        DeclareLaunchArgument("wind_enable_steady", default_value="true"),
        DeclareLaunchArgument("wind_enable_gust", default_value="true"),
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
            executable="wind_estimator_node",
            name="fake_wind_estimator",
            output="screen",
            parameters=[
                {
                    "publish_rate": 10.0,
                    "topic": wind_topic,
                    "wind.enable_steady": ParameterValue(wind_enable_steady, value_type=bool),
                    "wind.enable_gust": ParameterValue(wind_enable_gust, value_type=bool),
                    "wind.enable_colored": ParameterValue(wind_enable_colored, value_type=bool),
                    "wind.steady_wind_n": ParameterValue(wind_steady_n, value_type=float),
                    "wind.steady_wind_e": ParameterValue(wind_steady_e, value_type=float),
                    "wind.steady_wind_d": ParameterValue(wind_steady_d, value_type=float),
                    "wind.gust_interval": ParameterValue(wind_gust_interval, value_type=float),
                    "wind.gust_duration": ParameterValue(wind_gust_duration, value_type=float),
                    "wind.gust_magnitude": ParameterValue(wind_gust_magnitude, value_type=float),
                    "wind.colored_tau": ParameterValue(wind_colored_tau, value_type=float),
                    "wind.colored_sigma": ParameterValue(wind_colored_sigma, value_type=float),
                    "wind.seed": ParameterValue(wind_seed, value_type=int),
                    "estimate.gust_noise_sigma": 0.0,
                }
            ],
            condition=IfCondition(PythonExpression([start_wind_estimator, " and ", run_sim])),
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
                    "wind.topic": wind_topic,
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
            parameters=[
                guidance_params,
                {
                    "wind.topic": wind_topic,
                },
            ],
        )
    )

    return LaunchDescription(actions)
