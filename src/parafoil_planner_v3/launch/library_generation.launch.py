from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    output = LaunchConfiguration("output")
    config = LaunchConfiguration("config")

    # Use /tmp by default to avoid writing into install/share in non-dev installs.
    default_output = "/tmp/parafoil_library.pkl"
    default_config = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "library_params.yaml"])
    script = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "scripts", "generate_library.py"])

    return LaunchDescription(
        [
            DeclareLaunchArgument("output", default_value=default_output),
            DeclareLaunchArgument("config", default_value=default_config),
            ExecuteProcess(cmd=["python3", script, "--config", config, "--output", output], output="screen"),
        ]
    )
