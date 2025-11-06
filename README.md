# OpenVLA-FR3: Vision-Language-Action Robot Control

A complete ROS2 integration enabling natural language control of the Franka FR3 robotic arm using OpenVLA (7B vision-language-action model) with real-time reactive control via MoveIt Servo.

[![ROS2 Humble](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![OpenVLA](https://img.shields.io/badge/Model-OpenVLA--7B-orange)](https://openvla.github.io/)
[![MoveIt2](https://img.shields.io/badge/Motion-MoveIt2%20Servo-green)](https://moveit.ros.org/)
[![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey)](https://ubuntu.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

---

## Overview

This project bridges large vision-language models with real-time robot control, enabling the Franka FR3 to understand natural language instructions and execute manipulation tasks autonomously. The system uses a quantized OpenVLA model for inference on consumer GPUs and MoveIt Servo for reactive Cartesian control.

**Key Innovation:** Delta action formulation with VLA-Servo bridge pattern enables reactive, collision-aware motion while maintaining the benefits of learned visuomotor policies.

---

## System Architecture

```
Natural Language Instruction → OpenVLA-7B (4-bit quantized)
                                    ↓
                            Camera Image (RGB)
                                    ↓
                        VLAAction (delta pose + gripper)
                                    ↓
                            VLA-Servo Bridge
                                    ↓
                    TwistStamped (velocity commands)
                                    ↓
                            MoveIt Servo
                                    ↓
                        Joint Trajectory Controller
                                    ↓
                        Gazebo Simulation / Real Hardware
```

---

## Features

### Core Capabilities
- **Vision-Language-Action Inference**: OpenVLA-7B fine-tuned on LIBERO spatial tasks
- **4-bit Quantization**: Runs on consumer GPUs (6GB VRAM) using BitsAndBytes
- **Real-Time Control**: MoveIt Servo for reactive Cartesian velocity control at 30Hz
- **Delta Action Formulation**: Relative pose changes enable reactive, closed-loop control
- **Collision Awareness**: MoveIt planning scene monitoring with automatic collision avoidance
- **Natural Language Interface**: Task specification via plain English instructions

### Control Modes
- **Autonomous VLA Control**: Vision-language-conditioned action prediction
- **Keyboard Teleoperation**: Manual control for data collection and debugging
- **Dual Frame Control**: Base frame or end-effector frame reference
- **Multiple Controllers**: Position, velocity, impedance, and trajectory control

### Development Tools
- **Docker Environment**: Containerized setup with CUDA 12.1, ROS2 Humble, PyTorch
- **LIBERO-Compatible**: Camera positioning and task environment match LIBERO benchmark
- **RViz Visualization**: Interactive motion planning interface
- **Gazebo Simulation**: Full physics simulation with RGB-D sensors

---

## Architecture Components

### Custom ROS2 Packages

#### 1. `franka_openvla`
Main integration package containing:
- **openvla_node.py**: VLA inference with 4-bit quantization
- **vla_servo_bridge.py**: Translates VLA actions to Servo twist commands
- **keyboard_servo_teleop.py**: Manual teleoperation for data collection
- **fr3.launch.py**: Orchestrates entire system (Gazebo, MoveIt, Servo, VLA)
- World files and object models (bins, cubes, tables)
- Configuration files for Servo and controllers

#### 2. `vla_interfaces`
Custom message definitions:
```
VLAAction.msg:
  std_msgs/Header header
  geometry_msgs/Vector3 delta_pos    # [dx, dy, dz] in meters
  geometry_msgs/Vector3 delta_rot    # [droll, dpitch, dyaw] in radians
  float32 gripper                     # 0.0 = open, 1.0 = closed
```

### System Integration

#### OpenVLA Node
- **Model**: `openvla/openvla-7b-finetuned-libero-spatial`
- **Input**: RGB images from `/rgbd_camera/image` (640x480)
- **Output**: Delta actions published to `/vla/delta_actions`
- **Parameters**:
  - `instruction`: Task description (e.g., "pick the cube and place in red bin")
  - `unnorm_key`: Action unnormalization (default: `libero_spatial`)
  - `model_name`: HuggingFace model path

#### VLA-Servo Bridge
- **Converts**: VLAAction → TwistStamped
- **Safety**: Velocity clamping (max linear: 0.3 m/s, max angular: 0.5 rad/s)
- **Timeout**: Stops motion if no VLA action received within 0.5s
- **Keepalive**: Maintains Servo connection with periodic zero commands

#### MoveIt Servo Configuration
- **Control rate**: 30 Hz
- **Planning group**: `fr3_arm` (7-DOF)
- **End-effector**: `fr3_hand_tcp`
- **Collision checking**: 10 Hz with singularity avoidance
- **Joint limits**: Enforced with safety margins

---

## Installation

### Prerequisites
- Ubuntu 22.04
- NVIDIA GPU with CUDA support (6GB+ VRAM recommended)
- Docker and Docker Compose with NVIDIA runtime

### Setup

1. **Clone repository**:
```bash
git clone <repository-url>
cd openvla-fr3
```

2. **Build Docker container**:
```bash
docker-compose build
```

3. **Launch container**:
```bash
docker-compose up -d
```

4. **Build ROS2 workspace** (first time only):
```bash
docker exec -it vla_unified bash
source /opt/ros/humble/setup.bash
cd /ros2_ws
colcon build --symlink-install
source install/setup.bash
```

---

## Usage

### Launch Full System

Start the complete VLA control stack (Gazebo, MoveIt, Servo, OpenVLA):

```bash
docker exec vla_unified bash -c "
  source /opt/ros/humble/setup.bash &&
  source /ros2_ws/install/setup.bash &&
  ros2 launch franka_openvla fr3.launch.py
"
```

This launches:
- Gazebo simulation with FR3 and manipulation environment
- ROS2 controllers (joint state, arm, gripper)
- MoveIt move_group with OMPL planning
- MoveIt Servo for real-time control
- RViz for visualization
- OpenVLA inference node
- VLA-Servo bridge
- RGB-D camera with LIBERO-style positioning

### Keyboard Teleoperation

Manual control for testing and data collection:

```bash
docker exec -it vla_unified bash -c "
  source /opt/ros/humble/setup.bash &&
  source /ros2_ws/install/setup.bash &&
  ros2 run franka_openvla keyboard_servo_teleop
"
```

**Controls**:
- **Linear**: W/S (X), A/D (Y), Q/E (Z)
- **Angular**: J/L (yaw), I/K (pitch), U/O (roll)
- **Frame Toggle**: F (base frame ↔ end-effector frame)
- **Speed**: +/- (increase/decrease velocity scaling)
- **Stop**: SPACE
- **Quit**: ESC

### Change VLA Task Instruction

Modify the instruction parameter in `fr3.launch.py`:

```python
Node(
    package='franka_openvla',
    executable='openvla_node',
    parameters=[{
        'instruction': 'pick up the cube and place it in the red bin',
        'unnorm_key': 'libero_spatial'
    }]
)
```

Or pass as launch argument:
```bash
ros2 launch franka_openvla fr3.launch.py instruction:="move the object to the blue bin"
```

---

## Simulation Environment

The Gazebo world includes:
- **Work Table**: Main manipulation surface (0.6m from robot base)
- **Colored Bins**: Green, black, red, blue bins for sorting tasks
- **Target Cube**: Manipulable object on table
- **RGB-D Camera**: LIBERO-style agentview
  - Position: 1.2m forward, 1.0m height
  - Orientation: Looking at robot workspace
  - Outputs: RGB image, depth, point cloud, camera info

---

## Configuration

### Key Configuration Files

1. **`config/servo_params.yaml`**: MoveIt Servo settings
   - Velocity limits (linear: 0.4 m/s, rotational: 0.8 rad/s)
   - Control rate (30 Hz)
   - Safety factors (0.3x for conservative control)
   - Collision checking parameters

2. **`config/moveit_controllers.yaml`**: Controller interfaces
   - arm_controller (FollowJointTrajectory)
   - gripper_controller (GripperCommand)

3. **`config/franka_gazebo_controllers.yaml`**: Gazebo ros2_control
   - Joint state broadcaster
   - Position, velocity, and impedance controllers
   - Controller gains and limits

### Adjusting VLA-Servo Bridge Parameters

Edit `franka_openvla/vla_servo_bridge.py`:

```python
self.linear_scale = 0.5    # Velocity scaling (0.1-1.0)
self.angular_scale = 0.3   # Angular velocity scaling
self.max_linear = 0.3      # Max linear velocity (m/s)
self.max_angular = 0.5     # Max angular velocity (rad/s)
self.timeout = 0.5         # Command timeout (seconds)
```

---

## Development

### Project Structure

```
openvla-fr3/
├── src/
│   ├── franka_openvla/           # Main integration package
│   │   ├── franka_openvla/
│   │   │   ├── openvla_node.py
│   │   │   ├── vla_servo_bridge.py
│   │   │   └── keyboard_servo_teleop.py
│   │   ├── launch/
│   │   │   └── fr3.launch.py
│   │   ├── config/
│   │   │   ├── servo_params.yaml
│   │   │   └── moveit_controllers.yaml
│   │   ├── worlds/
│   │   │   └── franka_world.sdf
│   │   └── models/                # Object models
│   ├── vla_interfaces/            # Custom messages
│   ├── franka_gazebo/             # Simulation support
│   ├── franka_fr3_moveit_config/  # MoveIt configuration
│   ├── franka_hardware/           # Real robot interface
│   ├── franka_gripper/            # Gripper control
│   └── franka_msgs/               # Franka messages
├── Dockerfile
├── docker-compose.yml
└── README.md
```

### Building from Source

```bash
cd /ros2_ws
colcon build --symlink-install --packages-select vla_interfaces franka_openvla
source install/setup.bash
```

### Running Individual Nodes

```bash
# OpenVLA inference only
ros2 run franka_openvla openvla_node --ros-args -p instruction:="pick up the cube"

# VLA-Servo bridge only
ros2 run franka_openvla vla_servo_bridge

# Keyboard teleoperation only
ros2 run franka_openvla keyboard_servo_teleop
```

---

## Technical Details

### OpenVLA Model

- **Architecture**: Vision encoder + language encoder + action decoder
- **Parameters**: 7 billion (4-bit quantized to ~3.5GB memory)
- **Training**: Fine-tuned on LIBERO spatial manipulation tasks
- **Inference Speed**: ~5-10 Hz on consumer GPUs (RTX 3060+)
- **Action Space**: 7-DOF delta actions (3 position + 3 orientation + 1 gripper)

### Control Pipeline

1. **Perception**: RGB camera image captured at 30 Hz
2. **Inference**: OpenVLA predicts delta action from image + instruction
3. **Translation**: VLA-Servo bridge converts to twist commands
4. **Planning**: MoveIt Servo computes collision-free joint velocities
5. **Execution**: Joint trajectory controller sends commands to robot
6. **Feedback**: Loop closes with new camera observation

### Safety Features

- **Velocity Clamping**: Multi-level limits (VLA bridge + Servo + controller)
- **Collision Checking**: Real-time monitoring with planning scene
- **Singularity Avoidance**: Automatic damping near singularities
- **Command Timeout**: Stops robot if VLA fails or crashes
- **Joint Limits**: Enforced with configurable safety margins
- **Emergency Stop**: Keyboard SPACE key or ROS service call
---

