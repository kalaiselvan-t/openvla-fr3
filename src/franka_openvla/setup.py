from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'franka_openvla'

# Get all model directories and their files
model_data_files = []
models_dir = 'models'
if os.path.exists(models_dir):
    for model_name in os.listdir(models_dir):
        model_path = os.path.join(models_dir, model_name)
        if os.path.isdir(model_path):
            model_files = glob(os.path.join(model_path, '*'))
            if model_files:
                model_data_files.append((
                    f'share/{package_name}/models/{model_name}',
                    model_files
                ))

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/worlds', glob('worlds/*.world') + glob('worlds/*.sdf')),
    ] + model_data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='user',
    maintainer_email='im.kalaiselvan.t@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'openvla_node = franka_openvla.openvla_node:main',
            'keyboard_servo_teleop = franka_openvla.keyboard_servo_teleop:main',
        ],
    },
)
