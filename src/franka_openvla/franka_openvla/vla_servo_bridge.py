#!/usr/bin/env python3
"""
VLA to MoveIt Servo Bridge Node

Converts OpenVLA delta actions to MoveIt Servo twist commands.
This node bridges the gap between VLA model predictions and robot control.
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger
from vla_interfaces.msg import VLAAction
import time


class VLAServoRidge(Node):
    def __init__(self):
        super().__init__('vla_servo_bridge')

        # Declare parameters
        self.declare_parameter('vla_action_topic', '/vla/delta_actions')
        self.declare_parameter('servo_command_topic', '/servo_node/delta_twist_cmds')
        self.declare_parameter('command_frame', 'fr3_hand_tcp')
        self.declare_parameter('linear_scale', 1.0)  # Scale factor for linear velocities
        self.declare_parameter('angular_scale', 1.0)  # Scale factor for angular velocities
        self.declare_parameter('max_linear_vel', 0.3)  # Max linear velocity (m/s equivalent)
        self.declare_parameter('max_angular_vel', 0.5)  # Max angular velocity (rad/s equivalent)
        self.declare_parameter('command_timeout', 0.5)  # Timeout for VLA commands
        self.declare_parameter('auto_start_servo', True)  # Automatically start servo on init

        # Get parameters
        self.vla_action_topic = self.get_parameter('vla_action_topic').value
        self.servo_command_topic = self.get_parameter('servo_command_topic').value
        self.command_frame = self.get_parameter('command_frame').value
        self.linear_scale = self.get_parameter('linear_scale').value
        self.angular_scale = self.get_parameter('angular_scale').value
        self.max_linear_vel = self.get_parameter('max_linear_vel').value
        self.max_angular_vel = self.get_parameter('max_angular_vel').value
        self.command_timeout = self.get_parameter('command_timeout').value
        self.auto_start_servo = self.get_parameter('auto_start_servo').value

        self.get_logger().info('VLA-Servo Bridge Initializing...')
        self.get_logger().info(f'  VLA topic: {self.vla_action_topic}')
        self.get_logger().info(f'  Servo topic: {self.servo_command_topic}')
        self.get_logger().info(f'  Command frame: {self.command_frame}')
        self.get_logger().info(f'  Linear scale: {self.linear_scale}')
        self.get_logger().info(f'  Angular scale: {self.angular_scale}')

        # Service client for starting servo
        self.start_servo_client = self.create_client(Trigger, '/servo_node/start_servo')

        # Publisher for servo twist commands
        self.twist_pub = self.create_publisher(
            TwistStamped,
            self.servo_command_topic,
            10
        )

        # Subscriber for VLA actions
        self.vla_sub = self.create_subscription(
            VLAAction,
            self.vla_action_topic,
            self.vla_action_callback,
            10
        )

        # Tracking
        self.last_action_time = None
        self.action_count = 0
        self.servo_started = False

        # Keepalive timer - publish zero commands if no VLA action received
        self.keepalive_timer = self.create_timer(0.1, self.keepalive_callback)

        # Start servo if requested
        if self.auto_start_servo:
            self.get_logger().info('Waiting for servo services...')
            if self.start_servo_client.wait_for_service(timeout_sec=10.0):
                self.get_logger().info('Starting servo...')
                self.start_servo()
            else:
                self.get_logger().warn('Servo service not available. Will try to start on first action.')

        self.get_logger().info('=================================')
        self.get_logger().info('VLA-Servo Bridge Ready!')
        self.get_logger().info('Waiting for VLA actions...')
        self.get_logger().info('=================================')

    def start_servo(self):
        """Call the start_servo service"""
        if self.servo_started:
            return

        request = Trigger.Request()
        future = self.start_servo_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('✓ Servo started successfully!')
                self.servo_started = True
            else:
                self.get_logger().warn(f'Servo start returned: {future.result().message}')
        else:
            self.get_logger().error('Failed to call start_servo service')

    def vla_action_callback(self, msg):
        """
        Convert VLA action to servo twist command.

        VLAAction contains:
        - delta_pos: (dx, dy, dz) in meters - DELTA changes
        - delta_rot: (d_roll, d_pitch, d_yaw) in radians - DELTA changes
        - gripper: 0.0=open, 1.0=closed

        For MoveIt Servo (unitless mode):
        - We treat deltas as velocity commands (scaled appropriately)
        - Servo will integrate these to produce smooth motion
        """
        try:
            # Start servo on first action if not started
            if not self.servo_started and self.auto_start_servo:
                self.start_servo()

            # Create twist message
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = self.command_frame

            # Map delta position to linear velocity
            # Scale and clamp to safe limits
            twist_msg.twist.linear.x = self.clamp(
                msg.delta_pos.x * self.linear_scale,
                -self.max_linear_vel,
                self.max_linear_vel
            )
            twist_msg.twist.linear.y = self.clamp(
                msg.delta_pos.y * self.linear_scale,
                -self.max_linear_vel,
                self.max_linear_vel
            )
            twist_msg.twist.linear.z = self.clamp(
                msg.delta_pos.z * self.linear_scale,
                -self.max_linear_vel,
                self.max_linear_vel
            )

            # Map delta rotation to angular velocity
            # Scale and clamp to safe limits
            twist_msg.twist.angular.x = self.clamp(
                msg.delta_rot.x * self.angular_scale,
                -self.max_angular_vel,
                self.max_angular_vel
            )
            twist_msg.twist.angular.y = self.clamp(
                msg.delta_rot.y * self.angular_scale,
                -self.max_angular_vel,
                self.max_angular_vel
            )
            twist_msg.twist.angular.z = self.clamp(
                msg.delta_rot.z * self.angular_scale,
                -self.max_angular_vel,
                self.max_angular_vel
            )

            # Publish twist command
            self.twist_pub.publish(twist_msg)

            # Update tracking
            self.last_action_time = time.time()
            self.action_count += 1

            # Log periodically
            if self.action_count % 10 == 0:
                self.get_logger().info(
                    f'VLA Action #{self.action_count}:\n'
                    f'  Linear: [{twist_msg.twist.linear.x:.3f}, '
                    f'{twist_msg.twist.linear.y:.3f}, '
                    f'{twist_msg.twist.linear.z:.3f}]\n'
                    f'  Angular: [{twist_msg.twist.angular.x:.3f}, '
                    f'{twist_msg.twist.angular.y:.3f}, '
                    f'{twist_msg.twist.angular.z:.3f}]\n'
                    f'  Gripper: {msg.gripper:.2f}',
                    throttle_duration_sec=1.0
                )

            # TODO: Handle gripper command
            # Could publish to a separate gripper controller topic

        except Exception as e:
            self.get_logger().error(f'Error in VLA action callback: {e}')

    def keepalive_callback(self):
        """
        Send zero commands if no VLA action received recently.
        This keeps servo alive and stops motion if VLA stops publishing.
        """
        if self.last_action_time is None:
            return

        time_since_last = time.time() - self.last_action_time

        # If timeout exceeded, send stop command
        if time_since_last > self.command_timeout:
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = self.command_frame
            # All velocities are zero by default
            self.twist_pub.publish(twist_msg)

    def clamp(self, value, min_val, max_val):
        """Clamp value between min and max"""
        return max(min_val, min(max_val, value))


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = VLAServoRidge()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            node.get_logger().error(f'Node error: {e}')
    finally:
        if node:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
