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
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


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
    
    initial_altitude_arg = DeclareLaunchArgument(
        'initial_altitude',
        default_value='500.0',
        description='Initial altitude in meters'
    )
    
    integrator_arg = DeclareLaunchArgument(
        'integrator_type',
        default_value='rk4',
        description='Integration method: euler, semi_implicit, rk4'
    )
    
    # Simulator node
    simulator_node = Node(
        package='parafoil_simulator_ros',
        executable='sim_node',
        name='parafoil_simulator',
        output='screen',
        parameters=[
            LaunchConfiguration('params_file'),
            {
                'integrator_type': LaunchConfiguration('integrator_type'),
            }
        ],
        remappings=[
            # Add any topic remappings here if needed
        ],
    )
    
    return LaunchDescription([
        params_file_arg,
        initial_altitude_arg,
        integrator_arg,
        simulator_node,
    ])
