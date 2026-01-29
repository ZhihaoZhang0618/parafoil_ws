"""
Launch file for parafoil simulator.

Usage:
    ros2 launch parafoil_simulator_ros sim.launch.py
    
    # With custom parameters:
    ros2 launch parafoil_simulator_ros sim.launch.py \
        initial_altitude:=1000.0 \
        integrator_type:=euler
"""

import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from ament_index_python.packages import get_package_share_directory


def _build_sim_node(context):
    steady = [
        float(LaunchConfiguration('wind_steady_n').perform(context)),
        float(LaunchConfiguration('wind_steady_e').perform(context)),
        float(LaunchConfiguration('wind_steady_d').perform(context)),
    ]
    params_file = LaunchConfiguration('params_file').perform(context)
    params_override = LaunchConfiguration('params_override').perform(context)
    params_list = [params_file]
    if params_override:
        params_list.append(params_override)

    simulator_node = Node(
        package='parafoil_simulator_ros',
        executable='sim_node',
        name='parafoil_simulator',
        output='screen',
        parameters=[
            *params_list,
            {
                'integrator_type': LaunchConfiguration('integrator_type'),
                'initial_altitude': ParameterValue(
                    LaunchConfiguration('initial_altitude'),
                    value_type=float,
                ),
                'wind.enable_steady': ParameterValue(LaunchConfiguration('wind_enable_steady'), value_type=bool),
                'wind.enable_gust': ParameterValue(LaunchConfiguration('wind_enable_gust'), value_type=bool),
                'wind.enable_colored': ParameterValue(LaunchConfiguration('wind_enable_colored'), value_type=bool),
                'wind.steady_wind': steady,
                'wind.gust_interval': ParameterValue(LaunchConfiguration('wind_gust_interval'), value_type=float),
                'wind.gust_duration': ParameterValue(LaunchConfiguration('wind_gust_duration'), value_type=float),
                'wind.gust_magnitude': ParameterValue(LaunchConfiguration('wind_gust_magnitude'), value_type=float),
                'wind.colored_tau': ParameterValue(LaunchConfiguration('wind_colored_tau'), value_type=float),
                'wind.colored_sigma': ParameterValue(LaunchConfiguration('wind_colored_sigma'), value_type=float),
                'wind.seed': ParameterValue(LaunchConfiguration('wind_seed'), value_type=int),
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
        ],
    )
    return [simulator_node]


def generate_launch_description():
    # Get package directory
    pkg_dir = get_package_share_directory('parafoil_simulator_ros')
    
    # Default parameters file
    default_params_file = os.path.join(pkg_dir, 'config', 'params.yaml')
    
    # Declare launch arguments
    params_file_arg = DeclareLaunchArgument(
        'params_file',
        default_value=default_params_file,
        description='Path to parameters YAML file'
    )
    params_override_arg = DeclareLaunchArgument(
        'params_override',
        default_value='',
        description='Optional override parameters YAML file'
    )
    
    initial_altitude_arg = DeclareLaunchArgument(
        'initial_altitude',
        default_value='50.0',
        description='Initial altitude in meters'
    )
    
    integrator_arg = DeclareLaunchArgument(
        'integrator_type',
        default_value='rk4',
        description='Integration method: euler, semi_implicit, rk4'
    )
    
    wind_enable_steady_arg = DeclareLaunchArgument('wind_enable_steady', default_value='true')
    wind_enable_gust_arg = DeclareLaunchArgument('wind_enable_gust', default_value='true')
    wind_enable_colored_arg = DeclareLaunchArgument('wind_enable_colored', default_value='false')
    wind_steady_n_arg = DeclareLaunchArgument('wind_steady_n', default_value='0.0')
    wind_steady_e_arg = DeclareLaunchArgument('wind_steady_e', default_value='2.0')
    wind_steady_d_arg = DeclareLaunchArgument('wind_steady_d', default_value='0.0')
    wind_gust_interval_arg = DeclareLaunchArgument('wind_gust_interval', default_value='10.0')
    wind_gust_duration_arg = DeclareLaunchArgument('wind_gust_duration', default_value='2.0')
    wind_gust_magnitude_arg = DeclareLaunchArgument('wind_gust_magnitude', default_value='3.0')
    wind_colored_tau_arg = DeclareLaunchArgument('wind_colored_tau', default_value='2.0')
    wind_colored_sigma_arg = DeclareLaunchArgument('wind_colored_sigma', default_value='1.0')
    wind_seed_arg = DeclareLaunchArgument('wind_seed', default_value='-1')
    
    return LaunchDescription([
        params_file_arg,
        params_override_arg,
        initial_altitude_arg,
        integrator_arg,
        wind_enable_steady_arg,
        wind_enable_gust_arg,
        wind_enable_colored_arg,
        wind_steady_n_arg,
        wind_steady_e_arg,
        wind_steady_d_arg,
        wind_gust_interval_arg,
        wind_gust_duration_arg,
        wind_gust_magnitude_arg,
        wind_colored_tau_arg,
        wind_colored_sigma_arg,
        wind_seed_arg,
        OpaqueFunction(function=_build_sim_node),
    ])
