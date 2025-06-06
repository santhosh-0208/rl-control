from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pendulum_control',
            executable='pendulum_sim',
            name='pendulum_sim',
            output='screen'
        ),
        Node(
            package='pendulum_control',
            executable='pid_controller',
            name='pid_controller',
            output='screen'
        ),
        Node(
            package='pendulum_control',
            executable='ac_controller',
            name='ac_controller',
            output='screen'
        ),
        Node(
            package='pendulum_control',
            executable='nn_controller',
            name='nn_controller',
            output='screen'
        )
    ])
