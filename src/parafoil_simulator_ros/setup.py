from setuptools import setup
import os
from glob import glob

package_name = 'parafoil_simulator_ros'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include config files (yaml and rviz)
        (os.path.join('share', package_name, 'config'), 
            glob('config/*.yaml') + glob('config/*.rviz')),
        # Include launch files
        (os.path.join('share', package_name, 'launch'), 
            glob('launch/*.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Parafoil Simulator Team',
    maintainer_email='parafoil@example.com',
    description='ROS2 node for parafoil 6DoF simulation',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'sim_node = parafoil_simulator_ros.sim_node:main',
            'keyboard_teleop = parafoil_simulator_ros.keyboard_teleop:main',
        ],
    },
)
