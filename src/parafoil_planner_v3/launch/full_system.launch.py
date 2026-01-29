from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution
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

    default_planner = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "planner_params.yaml"])
    default_gpm = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "gpm_params.yaml"])
    default_dyn = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "dynamics_params.yaml"])
    default_guidance = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "guidance_params.yaml"])

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
    ]

    actions.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(sim_launch), condition=IfCondition(run_sim)))

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

    return LaunchDescription(actions)
