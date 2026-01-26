#!/usr/bin/env python3
"""
Keyboard Teleop for Parafoil Simulator.

Control the parafoil using keyboard (direct brake control):
    W / ↑  : Increase both brakes (symmetric brake)
    S / ↓  : Decrease both brakes
    A / ←  : Pull LEFT brake (turn left)
    D / →  : Pull RIGHT brake (turn right)
    Q      : Release LEFT brake
    E      : Release RIGHT brake
    Space  : Reset brakes to zero
    R      : Full brake (both to 100%)
    ESC    : Quit

Usage:
    ros2 run parafoil_simulator_ros keyboard_teleop
"""

import sys
import termios
import tty
import select
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Vector3Stamped


# Key codes
KEY_UP = '\x1b[A'
KEY_DOWN = '\x1b[B'
KEY_RIGHT = '\x1b[C'
KEY_LEFT = '\x1b[D'


def get_key(timeout=0.1):
    """Get a key press without blocking."""
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setraw(sys.stdin.fileno())
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if rlist:
            key = sys.stdin.read(1)
            # Check for escape sequence (arrow keys)
            if key == '\x1b':
                key += sys.stdin.read(2)
            return key
        return None
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


class KeyboardTeleop(Node):
    """Keyboard teleop node for parafoil control."""
    
    def __init__(self):
        super().__init__('keyboard_teleop')
        
        # Control state
        self.delta_l = 0.0  # Left brake [0, 1]
        self.delta_r = 0.0  # Right brake [0, 1]
        
        # Control increment
        self.increment = 0.05
        
        # Publisher
        self.pub = self.create_publisher(
            Vector3Stamped,
            '/rockpara_actuators_node/auto_commands',
            10
        )
        
        # Publish timer (20 Hz)
        self.timer = self.create_timer(0.05, self.publish_cmd)
        
        self.print_help()
    
    def print_help(self):
        """Print control instructions."""
        print("\n" + "=" * 50)
        print("  Parafoil Keyboard Teleop - Direct Brake Control")
        print("=" * 50)
        print("""
Controls (direct brake line control):
  W / ↑  : Pull BOTH brakes (symmetric brake, slow down)
  S / ↓  : Release BOTH brakes (speed up)

  A / ←  : Pull LEFT brake  (turn left)
  D / →  : Pull RIGHT brake (turn right)

  Q      : Release LEFT brake
  E      : Release RIGHT brake

  Space  : Release all (brakes to 0%)
  R      : Full brake (both to 100%)
  ESC    : Quit

Note: Left brake -> turn left, Right brake -> turn right
Current state will be displayed below.
Press any control key to start...
""")
    
    def update_display(self):
        """Update the display with current brake values."""
        bar_width = 20
        left_bar = int(self.delta_l * bar_width)
        right_bar = int(self.delta_r * bar_width)
        
        left_display = '█' * left_bar + '░' * (bar_width - left_bar)
        right_display = '█' * right_bar + '░' * (bar_width - right_bar)
        
        delta_s = (self.delta_l + self.delta_r) / 2
        delta_a = self.delta_r - self.delta_l
        
        # Clear line and print status
        print(f"\r  L [{left_display}] {self.delta_l:.2f}  |  R [{right_display}] {self.delta_r:.2f}  |  Sym={delta_s:.2f}  Asym={delta_a:+.2f}  ", end='', flush=True)
    
    def process_key(self, key):
        """Process a key press."""
        if key is None:
            return True
        
        # Quit
        if key == '\x1b' and len(key) == 1:  # ESC only (not arrow key)
            return False
        if key == '\x03':  # Ctrl+C
            return False
        
        # Symmetric brake (both)
        if key in ['w', 'W', KEY_UP]:
            self.delta_l = min(1.0, self.delta_l + self.increment)
            self.delta_r = min(1.0, self.delta_r + self.increment)
        elif key in ['s', 'S', KEY_DOWN]:
            self.delta_l = max(0.0, self.delta_l - self.increment)
            self.delta_r = max(0.0, self.delta_r - self.increment)

        # Pull LEFT brake (turn left)
        elif key in ['a', 'A', KEY_LEFT]:
            self.delta_l = min(1.0, self.delta_l + self.increment)

        # Pull RIGHT brake (turn right)
        elif key in ['d', 'D', KEY_RIGHT]:
            self.delta_r = min(1.0, self.delta_r + self.increment)

        # Release individual brakes
        elif key in ['q', 'Q']:
            self.delta_l = max(0.0, self.delta_l - self.increment)
        elif key in ['e', 'E']:
            self.delta_r = max(0.0, self.delta_r - self.increment)
        
        # Reset
        elif key == ' ':
            self.delta_l = 0.0
            self.delta_r = 0.0
        
        # Full brake
        elif key in ['r', 'R']:
            self.delta_l = 1.0
            self.delta_r = 1.0
        
        self.update_display()
        return True
    
    def publish_cmd(self):
        """Publish current control command."""
        msg = Vector3Stamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.vector.x = self.delta_l
        msg.vector.y = self.delta_r
        msg.vector.z = 0.0
        self.pub.publish(msg)


def main():
    rclpy.init()
    
    node = KeyboardTeleop()
    
    try:
        while rclpy.ok():
            # Process ROS callbacks
            rclpy.spin_once(node, timeout_sec=0.01)
            
            # Get and process key
            key = get_key(timeout=0.05)
            if not node.process_key(key):
                break
    except KeyboardInterrupt:
        pass
    finally:
        print("\n\nShutting down keyboard teleop...")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
