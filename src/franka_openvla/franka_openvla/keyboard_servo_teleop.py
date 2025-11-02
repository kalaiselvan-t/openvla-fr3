#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import TwistStamped
from std_srvs.srv import Trigger
import sys
import tty
import termios
import select
import time

class KeyboardServoTeleop(Node):
    def __init__(self):
        super().__init__('keyboard_servo_teleop')

        # Publisher for twist commands
        self.publisher = self.create_publisher(
            TwistStamped,
            '/servo_node/delta_twist_cmds',
            10
        )

        # Service clients to start/stop servo
        self.start_servo_client = self.create_client(Trigger, '/servo_node/start_servo')
        self.stop_servo_client = self.create_client(Trigger, '/servo_node/stop_servo')

        # Speed settings (unitless [-1, 1] for servo with command_in_type: "unitless")
        self.linear_speed = 0.3
        self.angular_speed = 0.3

        # Frame control: 'base' or 'ee'
        self.command_frame = 'base'  # Start with base frame
        self.base_frame = 'fr3_link0'
        self.ee_frame = 'fr3_hand_tcp'

        # Timer for continuous zero-command publishing (keep servo alive)
        self.last_command_time = time.time()
        self.keepalive_timer = self.create_timer(0.1, self.keepalive_callback)  # 10 Hz

        # Track if we have active motion
        self.active_motion = False

        self.get_logger().info('Keyboard Servo Teleop Initializing...')
        self.get_logger().info('Waiting for servo services...')

        # Wait for servo services
        if not self.start_servo_client.wait_for_service(timeout_sec=10.0):
            self.get_logger().error('Servo services not available! Make sure servo_node is running.')
            raise RuntimeError('Servo services not available')

        self.get_logger().info('✓ Servo services found. Starting servo...')
        self.start_servo()

        self.get_logger().info('=================================')
        self.get_logger().info('Keyboard Servo Teleop Ready!')
        self.get_logger().info('=================================')
        self.get_logger().info('Controls:')
        self.get_logger().info('  w/s: forward/backward (X)')
        self.get_logger().info('  a/d: left/right (Y)')
        self.get_logger().info('  q/e: up/down (Z)')
        self.get_logger().info('  j/l: yaw left/right')
        self.get_logger().info('  i/k: pitch up/down')
        self.get_logger().info('  u/o: roll left/right')
        self.get_logger().info('  +/-: increase/decrease speed')
        self.get_logger().info('  f: toggle frame (base/ee)')
        self.get_logger().info('  SPACE: stop motion')
        self.get_logger().info('  ESC/Ctrl+C: quit')
        self.get_logger().info('=================================')
        self.get_logger().info(f'Command frame: {self.command_frame} ({self.get_current_frame()})')
        self.get_logger().info(f'Speed: linear={self.linear_speed:.2f}, angular={self.angular_speed:.2f}')
        self.get_logger().info('=================================')

        self.settings = termios.tcgetattr(sys.stdin)

    def start_servo(self):
        """Call the start_servo service"""
        request = Trigger.Request()
        future = self.start_servo_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('✓ Servo started successfully!')
            else:
                self.get_logger().warn(f'Servo start returned: {future.result().message}')
        else:
            self.get_logger().error('Failed to call start_servo service')

    def stop_servo(self):
        """Call the stop_servo service"""
        self.get_logger().info('Stopping servo...')
        request = Trigger.Request()
        future = self.stop_servo_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=5.0)

        if future.result() is not None:
            if future.result().success:
                self.get_logger().info('✓ Servo stopped successfully!')
            else:
                self.get_logger().warn(f'Servo stop returned: {future.result().message}')

    def get_current_frame(self):
        """Get the current command frame"""
        return self.base_frame if self.command_frame == 'base' else self.ee_frame

    def toggle_frame(self):
        """Toggle between base and end-effector frames"""
        self.command_frame = 'ee' if self.command_frame == 'base' else 'base'
        self.get_logger().info(f'Command frame: {self.command_frame} ({self.get_current_frame()})')

    def keepalive_callback(self):
        """Send zero commands periodically if no recent commands (keeps servo alive)"""
        # Only send keepalive if no command in last 0.2 seconds and no active motion
        if not self.active_motion and (time.time() - self.last_command_time) > 0.2:
            self.publish_twist()  # Send zeros

    def get_key(self, timeout=0.1):
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
        else:
            key = ''
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
        return key

    def publish_twist(self, linear_x=0.0, linear_y=0.0, linear_z=0.0,
                     angular_x=0.0, angular_y=0.0, angular_z=0.0):
        """Publish twist command with proper frame and timestamp"""
        msg = TwistStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.get_current_frame()  # Use fr3_link0 or fr3_hand_tcp

        msg.twist.linear.x = linear_x
        msg.twist.linear.y = linear_y
        msg.twist.linear.z = linear_z
        msg.twist.angular.x = angular_x
        msg.twist.angular.y = angular_y
        msg.twist.angular.z = angular_z

        self.publisher.publish(msg)
        self.last_command_time = time.time()

        # Track if we have non-zero motion
        self.active_motion = (linear_x != 0.0 or linear_y != 0.0 or linear_z != 0.0 or
                             angular_x != 0.0 or angular_y != 0.0 or angular_z != 0.0)

    def run(self):
        try:
            while rclpy.ok():
                key = self.get_key(timeout=0.1)

                if not key:
                    continue

                if key == '\x1b':  # ESC
                    break
                elif key == '\x03':  # Ctrl+C
                    break
                elif key == ' ':  # SPACE - stop
                    self.publish_twist()
                    self.get_logger().info('STOP')
                elif key == 'f' or key == 'F':  # Toggle frame
                    self.toggle_frame()

                # Linear motion
                elif key == 'w':
                    self.publish_twist(linear_x=self.linear_speed)
                    self.get_logger().info(f'Forward: {self.linear_speed}')
                elif key == 's':
                    self.publish_twist(linear_x=-self.linear_speed)
                    self.get_logger().info(f'Backward: {-self.linear_speed}')
                elif key == 'a':
                    self.publish_twist(linear_y=self.linear_speed)
                    self.get_logger().info(f'Left: {self.linear_speed}')
                elif key == 'd':
                    self.publish_twist(linear_y=-self.linear_speed)
                    self.get_logger().info(f'Right: {-self.linear_speed}')
                elif key == 'q':
                    self.publish_twist(linear_z=self.linear_speed)
                    self.get_logger().info(f'Up: {self.linear_speed}')
                elif key == 'e':
                    self.publish_twist(linear_z=-self.linear_speed)
                    self.get_logger().info(f'Down: {-self.linear_speed}')

                # Angular motion
                elif key == 'j':
                    self.publish_twist(angular_z=self.angular_speed)
                    self.get_logger().info(f'Rotate Left: {self.angular_speed}')
                elif key == 'l':
                    self.publish_twist(angular_z=-self.angular_speed)
                    self.get_logger().info(f'Rotate Right: {-self.angular_speed}')
                elif key == 'i':
                    self.publish_twist(angular_y=self.angular_speed)
                    self.get_logger().info(f'Pitch Up: {self.angular_speed}')
                elif key == 'k':
                    self.publish_twist(angular_y=-self.angular_speed)
                    self.get_logger().info(f'Pitch Down: {-self.angular_speed}')
                elif key == 'u':
                    self.publish_twist(angular_x=self.angular_speed)
                    self.get_logger().info(f'Roll Left: {self.angular_speed}')
                elif key == 'o':
                    self.publish_twist(angular_x=-self.angular_speed)
                    self.get_logger().info(f'Roll Right: {-self.angular_speed}')

                # Speed control
                elif key == '+' or key == '=':
                    self.linear_speed = min(1.0, self.linear_speed + 0.05)
                    self.angular_speed = min(1.0, self.angular_speed + 0.05)
                    self.get_logger().info(f'Speed increased: {self.linear_speed:.2f}')
                elif key == '-' or key == '_':
                    self.linear_speed = max(0.05, self.linear_speed - 0.05)
                    self.angular_speed = max(0.05, self.angular_speed - 0.05)
                    self.get_logger().info(f'Speed decreased: {self.linear_speed:.2f}')

        except Exception as e:
            self.get_logger().error(f'Error: {e}')
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.settings)
            self.publish_twist()  # Send stop command
            self.stop_servo()  # Stop servo gracefully
            self.get_logger().info('Shutting down...')


def main(args=None):
    rclpy.init(args=args)
    node = KeyboardServoTeleop()

    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
