# Dockerfile combining Franka ROS2 + OpenVLA

FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    ROS_DISTRO=humble

# Build arguments (matching original Dockerfile)
ARG USER_UID=1001
ARG USER_GID=1001
ARG USERNAME=user

# ============================================================
# STEP 1: Install ROS2 Humble base + essential tools
# ============================================================
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    git \
    wget \
    nano \
    vim \
    sudo \
    bash-completion \
    gdb \
    openssh-client \
    # OpenCV system dependencies
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libgl1-mesa-glx \
    python3-pip \
    && curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
        http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" \
        > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y \
    ros-humble-ros-base \
    ros-humble-rclpy \
    ros-humble-std-msgs \
    ros-humble-sensor-msgs \
    ros-humble-geometry-msgs \
    ros-humble-cv-bridge \
    python3-colcon-common-extensions \
    python3-rosdep \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rosdep init

# ============================================================
# STEP 2: Install all Franka ROS2 dependencies
# ============================================================
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-humble-ros-gz \
    ros-humble-sdformat-urdf \
    ros-humble-joint-state-publisher-gui \
    ros-humble-ros2controlcli \
    ros-humble-controller-interface \
    ros-humble-hardware-interface-testing \
    ros-humble-ament-cmake-clang-format \
    ros-humble-ament-cmake-clang-tidy \
    ros-humble-controller-manager \
    ros-humble-ros2-control-test-assets \
    libignition-gazebo6-dev \
    libignition-plugin-dev \
    ros-humble-hardware-interface \
    ros-humble-control-msgs \
    ros-humble-backward-ros \
    ros-humble-generate-parameter-library \
    ros-humble-realtime-tools \
    ros-humble-joint-state-publisher \
    ros-humble-joint-state-broadcaster \
    ros-humble-moveit-ros-move-group \
    ros-humble-moveit-kinematics \
    ros-humble-moveit-planners-ompl \
    ros-humble-moveit-ros-visualization \
    ros-humble-joint-trajectory-controller \
    ros-humble-moveit-simple-controller-manager \
    ros-humble-rviz2 \
    ros-humble-xacro \
    ros-humble-ros2-control \
    ros-humble-ros2-controllers \
    ros-humble-py-binding-tools \
    ros-humble-moveit-servo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# STEP 3: Install OpenVLA dependencies (PyTorch + transformers)
# ============================================================
RUN pip install accelerate bitsandbytes vcstool && \
    pip install -r https://raw.githubusercontent.com/openvla/openvla/main/requirements-min.txt

# ============================================================
# STEP 4: Setup user (matching original Dockerfile)
# ============================================================
RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers \
    && echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> /home/$USERNAME/.bashrc \
    && echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> /home/$USERNAME/.bashrc

USER $USERNAME

# ============================================================
# STEP 5: Setup workspace and install Franka dependencies
# ============================================================
WORKDIR /ros2_ws

# Copy source code first
COPY --chown=$USERNAME:$USERNAME ./src ./src

# Install dependencies
RUN sudo apt-get update \
    && rosdep update \
    && rosdep install --from-paths src --ignore-src --rosdistro $ROS_DISTRO -y \
    && sudo apt-get clean \
    && sudo rm -rf /var/lib/apt/lists/* \
    && rm -rf /home/$USERNAME/.ros

# ============================================================
# STEP 6: Setup entrypoint
# ============================================================
COPY ./franka_entrypoint.sh /franka_entrypoint.sh
RUN sudo chmod +x /franka_entrypoint.sh

# Set the default shell to bash
SHELL [ "/bin/bash", "-c" ]
ENTRYPOINT [ "/franka_entrypoint.sh" ]
CMD [ "/bin/bash" ]
WORKDIR /ros2_ws
