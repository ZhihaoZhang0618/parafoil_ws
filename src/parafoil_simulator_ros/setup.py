from setuptools import setup
import os
import shutil
from glob import glob

package_name = 'parafoil_simulator_ros'

try:
    from distutils.command.install_data import install_data  # type: ignore
except Exception:  # pragma: no cover
    install_data = None  # type: ignore


cmdclass = {}


if install_data is not None:
    class symlink_data(install_data):  # noqa: N801
        """Like colcon's symlink_data, but handle new files with --force.

        colcon invokes `setup.py ... symlink_data --force` when using
        `colcon build --symlink-install`. Some colcon versions try to remove
        the *destination directory* instead of the destination file, which
        breaks when adding new data_files. This implementation removes the
        target only if it exists.
        """

        def copy_file(self, src, dst, **kwargs):  # noqa: D102
            if kwargs.get('link'):
                return super().copy_file(src, dst, **kwargs)

            if os.path.isdir(dst):
                target = os.path.join(dst, os.path.basename(src))
            else:
                target = dst
            if os.path.lexists(target):
                if os.path.isdir(target) and not os.path.islink(target):
                    shutil.rmtree(target)
                else:
                    os.remove(target)

            kwargs['link'] = 'sym'
            src = os.path.abspath(src)
            return super().copy_file(src, dst, **kwargs)

    cmdclass['symlink_data'] = symlink_data

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
    cmdclass=cmdclass,
    entry_points={
        'console_scripts': [
            'sim_node = parafoil_simulator_ros.sim_node:main',
            'keyboard_teleop = parafoil_simulator_ros.keyboard_teleop:main',
        ],
    },
)
