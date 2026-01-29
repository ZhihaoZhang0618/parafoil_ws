from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    planner_share = FindPackageShare("parafoil_planner_v3")
    sim_share = FindPackageShare("parafoil_simulator_ros")
    e2e_launch = PathJoinSubstitution([planner_share, "launch", "e2e_verification.launch.py"])
    guidance_params = PathJoinSubstitution([planner_share, "config", "guidance_params_strongwind.yaml"])
    planner_params = PathJoinSubstitution([planner_share, "config", "planner_params_strongwind.yaml"])
    sim_params = PathJoinSubstitution([sim_share, "config", "params.yaml"])
    sim_params_override = PathJoinSubstitution([sim_share, "config", "params_strongwind.yaml"])
    rviz_config_default = PathJoinSubstitution([sim_share, "config", "parafoil_integrated.rviz"])
    rviz = LaunchConfiguration("rviz")
    rviz_config = LaunchConfiguration("rviz_config")

    return LaunchDescription(
        [
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("rviz_config", default_value=rviz_config_default),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(e2e_launch),
                launch_arguments={
                    "sim_params": sim_params,
                    "sim_params_override": sim_params_override,
                    "planner_params": planner_params,
                    "guidance_params": guidance_params,
                    "use_library": "true",
                    "library_path": "/tmp/parafoil_library.pkl",
                    "initial_altitude": "70.0",
                    "wind_enable_steady": "true",
                    "wind_enable_gust": "false",
                    "wind_enable_colored": "false",
                    "wind_steady_n": "0.0",
                    "wind_steady_e": "6.0",
                    "target_enu_x": "120.0",
                    "target_enu_y": "0.0",
                    "target_enu_z": "0.0",
                    "rviz": rviz,
                    "rviz_config": rviz_config,
                }.items(),
            )
        ]
    )
