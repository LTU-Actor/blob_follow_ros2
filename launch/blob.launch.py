from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory

import os

config = os.path.join(
        get_package_share_directory('blob_follow_ros2'),
        'config',
        'blob_params.yaml'
        )

def masker():
    return Node(
        package='blob_follow_ros2',
        executable='blob_follow',
        name='blob_follow',
        output='screen',
        parameters=[
            config
            ],
    )
    
    
def generate_launch_description():
    return LaunchDescription([
        masker()
    ])