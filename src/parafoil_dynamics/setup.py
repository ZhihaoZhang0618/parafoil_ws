from setuptools import setup, find_packages

package_name = 'parafoil_dynamics'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['tests']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'pyyaml',
    ],
    extras_require={
        'test': ['pytest'],
    },
    zip_safe=True,
    author='Parafoil Simulator Team',
    author_email='parafoil@example.com',
    description='Pure Python 6DoF parafoil dynamics library',
    license='MIT',
    python_requires='>=3.8',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [],
    },
)
