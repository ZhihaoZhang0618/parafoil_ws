from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    e2e_launch = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "launch", "e2e_verification.launch.py"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("run_sim", default_value="true"),
            DeclareLaunchArgument("planner_params", default_value=PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "planner_params.yaml"])),
            DeclareLaunchArgument("gpm_params", default_value=PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "gpm_params.yaml"])),
            DeclareLaunchArgument("dynamics_params", default_value=PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "dynamics_params.yaml"])),
            DeclareLaunchArgument("guidance_params", default_value=PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "guidance_params.yaml"])),
            DeclareLaunchArgument("use_library", default_value=TextSubstitution(text="true")),
            DeclareLaunchArgument("library_path", default_value=TextSubstitution(text="/tmp/parafoil_library.pkl")),
            DeclareLaunchArgument("start_library_server", default_value=TextSubstitution(text="true")),
            DeclareLaunchArgument("sim_params", default_value=PathJoinSubstitution([FindPackageShare("parafoil_simulator_ros"), "config", "params.yaml"])),
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
            DeclareLaunchArgument("target_enu_x", default_value="150.0"),
            DeclareLaunchArgument("target_enu_y", default_value="50.0"),
            DeclareLaunchArgument("target_enu_z", default_value="0.0"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(e2e_launch),
                launch_arguments={
                    "run_sim": LaunchConfiguration("run_sim"),
                    "planner_params": LaunchConfiguration("planner_params"),
                    "gpm_params": LaunchConfiguration("gpm_params"),
                    "dynamics_params": LaunchConfiguration("dynamics_params"),
                    "guidance_params": LaunchConfiguration("guidance_params"),
                    "use_library": LaunchConfiguration("use_library"),
                    "library_path": LaunchConfiguration("library_path"),
                    "start_library_server": LaunchConfiguration("start_library_server"),
                    "sim_params": LaunchConfiguration("sim_params"),
                    "initial_altitude": LaunchConfiguration("initial_altitude"),
                    "integrator_type": LaunchConfiguration("integrator_type"),
                    "wind_enable_steady": LaunchConfiguration("wind_enable_steady"),
                    "wind_enable_gust": LaunchConfiguration("wind_enable_gust"),
                    "wind_enable_colored": LaunchConfiguration("wind_enable_colored"),
                    "wind_steady_n": LaunchConfiguration("wind_steady_n"),
                    "wind_steady_e": LaunchConfiguration("wind_steady_e"),
                    "wind_steady_d": LaunchConfiguration("wind_steady_d"),
                    "wind_gust_interval": LaunchConfiguration("wind_gust_interval"),
                    "wind_gust_duration": LaunchConfiguration("wind_gust_duration"),
                    "wind_gust_magnitude": LaunchConfiguration("wind_gust_magnitude"),
                    "wind_colored_tau": LaunchConfiguration("wind_colored_tau"),
                    "wind_colored_sigma": LaunchConfiguration("wind_colored_sigma"),
                    "wind_seed": LaunchConfiguration("wind_seed"),
                    "auto_target": LaunchConfiguration("auto_target"),
                    "target_delay_s": LaunchConfiguration("target_delay_s"),
                    "target_enu_x": LaunchConfiguration("target_enu_x"),
                    "target_enu_y": LaunchConfiguration("target_enu_y"),
                    "target_enu_z": LaunchConfiguration("target_enu_z"),
                }.items(),
            ),
        ]
    )
