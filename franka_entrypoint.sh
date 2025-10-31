#!/bin/bash
set -e

# Source ROS2 environment
source /opt/ros/$ROS_DISTRO/setup.bash

# Source workspace overlay if it exists
if [ -f /ros2_ws/install/setup.bash ]; then
    source /ros2_ws/install/setup.bash
fi

exec "$@"