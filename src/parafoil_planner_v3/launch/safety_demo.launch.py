from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, TextSubstitution, PythonExpression
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    run_sim = LaunchConfiguration("run_sim")
    planner_params = LaunchConfiguration("planner_params")
    gpm_params = LaunchConfiguration("gpm_params")
    dynamics_params = LaunchConfiguration("dynamics_params")
    guidance_params = LaunchConfiguration("guidance_params")

    risk_grid_file = LaunchConfiguration("risk_grid_file")
    no_fly_circles = LaunchConfiguration("no_fly_circles")
    no_fly_polygons = LaunchConfiguration("no_fly_polygons")
    no_fly_polygons_file = LaunchConfiguration("no_fly_polygons_file")

    wind_steady_e = LaunchConfiguration("wind_steady_e")
    wind_steady_n = LaunchConfiguration("wind_steady_n")
    wind_steady_d = LaunchConfiguration("wind_steady_d")

    rviz_main = LaunchConfiguration("rviz_main")
    rviz_config = LaunchConfiguration("rviz_config")

    default_planner = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "planner_params_safety_demo.yaml"])
    default_gpm = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "gpm_params.yaml"])
    default_dyn = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "dynamics_params.yaml"])
    default_guidance = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "guidance_params.yaml"])
    default_risk = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "config", "demo_risk_grid.npz"])

    default_rviz = PathJoinSubstitution([FindPackageShare("parafoil_simulator_ros"), "config", "parafoil_integrated.rviz"])
    e2e_launch = PathJoinSubstitution([FindPackageShare("parafoil_planner_v3"), "launch", "e2e_verification.launch.py"])

    actions = [
        DeclareLaunchArgument("run_sim", default_value="true"),
        DeclareLaunchArgument("planner_params", default_value=default_planner),
        DeclareLaunchArgument("gpm_params", default_value=default_gpm),
        DeclareLaunchArgument("dynamics_params", default_value=default_dyn),
        DeclareLaunchArgument("guidance_params", default_value=default_guidance),
        DeclareLaunchArgument("risk_grid_file", default_value=default_risk),
        DeclareLaunchArgument("no_fly_circles", default_value=TextSubstitution(text="[[60.0,0.0,25.0,10.0],[-40.0,80.0,20.0,5.0]]")),
        DeclareLaunchArgument("no_fly_polygons", default_value=TextSubstitution(text="[[[-30.0,-60.0],[-30.0,-20.0],[10.0,-20.0],[10.0,-60.0]]]")),
        DeclareLaunchArgument("no_fly_polygons_file", default_value=""),
        DeclareLaunchArgument("wind_steady_n", default_value="0.0"),
        DeclareLaunchArgument("wind_steady_e", default_value="4.0"),
        DeclareLaunchArgument("wind_steady_d", default_value="0.0"),
        DeclareLaunchArgument("rviz_main", default_value="true"),
        DeclareLaunchArgument("rviz_config", default_value=default_rviz),
    ]

    actions.append(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(e2e_launch),
            launch_arguments={
                "run_sim": run_sim,
                "planner_params": planner_params,
                "gpm_params": gpm_params,
                "dynamics_params": dynamics_params,
                "guidance_params": guidance_params,
                "use_library": "false",
                "start_library_server": "false",
                "record_bag": "false",
                "rviz": "false",
                "wind_enable_steady": "true",
                "wind_steady_n": wind_steady_n,
                "wind_steady_e": wind_steady_e,
                "wind_steady_d": wind_steady_d,
            }.items(),
        )
    )

    wind_default_ned_str = PythonExpression(
        [
            "'[' + str(",
            wind_steady_n,
            ") + ',' + str(",
            wind_steady_e,
            ") + ',' + str(",
            wind_steady_d,
            ") + ']'",
        ]
    )

    actions.append(
        Node(
            package="parafoil_planner_v3",
            executable="safety_viz_node",
            name="parafoil_safety_viz",
            output="screen",
            parameters=[
                {
                    "frame_id": "world",
                    "publish_rate_hz": 1.0,
                    "risk_grid_file": ParameterValue(risk_grid_file, value_type=str),
                    "risk_clip_min": 0.0,
                    "risk_clip_max": 1.0,
                    "safe_threshold": 0.2,
                    "danger_threshold": 0.7,
                    "risk_stride": 4,
                    "risk_z_m": 0.2,
                    "wind_default_ned": ParameterValue(wind_default_ned_str, value_type=str),
                    "reachability.show": True,
                    "reachability.show_line": False,
                    "reachability.show_fill": True,
                    "reachability.brake": 0.2,
                    "reachability.min_altitude_m": 5.0,
                    "reachability.terrain_height0_m": 0.0,
                    "reachability.clearance_m": 0.0,
                    "reachability.wind_margin_mps": 0.2,
                    "reachability.wind_uncertainty_mps": 0.5,
                    "reachability.gust_margin_mps": 0.5,
                    "reachability.sample_count": 48,
                    "reachability.line_width": 1.2,
                    "reachability.color": [0.0, 1.0, 1.0],
                    "reachability.z_m": 1.0,
                    "reachability.fill_stride_m": 10.0,
                    "reachability.fill_alpha": 0.35,
                    "reachability.fill_color": [0.0, 0.9, 1.0],
                    "no_fly_circles": ParameterValue(no_fly_circles, value_type=str),
                    "no_fly_polygons": ParameterValue(no_fly_polygons, value_type=str),
                    "no_fly_polygons_file": ParameterValue(no_fly_polygons_file, value_type=str),
                }
            ],
        )
    )

    actions.append(
        Node(
            package="rviz2",
            executable="rviz2",
            name="rviz2",
            output="screen",
            arguments=["-d", rviz_config],
            condition=IfCondition(rviz_main),
        )
    )

    return LaunchDescription(actions)
