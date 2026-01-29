from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    planner_share = FindPackageShare("parafoil_planner_v3")
    sim_share = FindPackageShare("parafoil_simulator_ros")
    e2e_launch = PathJoinSubstitution([planner_share, "launch", "e2e_verification.launch.py"])
    guidance_params = PathJoinSubstitution([planner_share, "config", "guidance_params_gusty.yaml"])
    sim_params = PathJoinSubstitution([sim_share, "config", "params.yaml"])
    sim_params_override = PathJoinSubstitution([sim_share, "config", "params_gusty.yaml"])
    rviz_config = PathJoinSubstitution([sim_share, "config", "parafoil_integrated.rviz"])

    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(e2e_launch),
                launch_arguments={
                    "sim_params": sim_params,
                    "sim_params_override": sim_params_override,
                    "guidance_params": guidance_params,
                    "initial_altitude": "80.0",
                    "wind_enable_steady": "true",
                    "wind_enable_gust": "true",
                    "wind_enable_colored": "true",
                    "wind_steady_n": "0.0",
                    "wind_steady_e": "2.0",
                    "wind_gust_interval": "8.0",
                    "wind_gust_duration": "2.0",
                    "wind_gust_magnitude": "4.0",
                    "wind_colored_tau": "2.0",
                    "wind_colored_sigma": "2.0",
                    "target_enu_x": "140.0",
                    "target_enu_y": "30.0",
                    "target_enu_z": "0.0",
                    "rviz": "true",
                    "rviz_config": rviz_config,
                }.items(),
            )
        ]
    )
