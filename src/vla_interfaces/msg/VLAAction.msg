# VLA Action message - represents delta end-effector pose + gripper command
# All pose values are DELTAS (relative changes), not absolute positions

std_msgs/Header header

# Delta end-effector pose (relative changes)
geometry_msgs/Vector3 delta_pos     # Delta position (dx, dy, dz) in meters
geometry_msgs/Vector3 delta_rot     # Delta rotation (d_roll, d_pitch, d_yaw) in radians

# Gripper command (absolute value)
float32 gripper                      # Gripper state: 0.0 = open, 1.0 = closed
