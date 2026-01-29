from setuptools import setup, find_packages
import os
import shutil
from glob import glob

package_name = 'parafoil_planner_v3'

try:
    from distutils.command.install_data import install_data  # type: ignore
except Exception:  # pragma: no cover
    install_data = None  # type: ignore


cmdclass = {}


if install_data is not None:
    class symlink_data(install_data):  # noqa: N801
        """Like colcon's symlink_data, but handle new/changed files with --force."""

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
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.npz')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'scripts'), glob('scripts/*.py')),
        (os.path.join('share', package_name, 'prompts'), glob('prompts/*.md')),
        (os.path.join('share', package_name, 'library'), glob('library/*')),
    ],
    install_requires=['setuptools', 'numpy', 'scipy', 'pyyaml'],
    zip_safe=True,
    maintainer='AIMS Lab',
    maintainer_email='aims@example.com',
    description='Parafoil planner v3',
    license='MIT',
    tests_require=['pytest'],
    cmdclass=cmdclass,
    entry_points={
        'console_scripts': [
            'planner_node = parafoil_planner_v3.nodes.planner_node:main',
            'guidance_node = parafoil_planner_v3.nodes.guidance_node:main',
            'library_server_node = parafoil_planner_v3.nodes.library_server_node:main',
            'mission_logger_node = parafoil_planner_v3.nodes.mission_logger_node:main',
            'safety_viz_node = parafoil_planner_v3.nodes.safety_viz_node:main',
        ],
    },
)
