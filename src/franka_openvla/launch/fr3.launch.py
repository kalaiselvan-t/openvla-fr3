#!/usr/bin/env python3
import os
import xacro, yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, ExecuteProcess, RegisterEventHandler, Shutdown, TimerAction, SetEnvironmentVariable
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node, SetParameter
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare

def load_yaml(package_name, file_path):
    package_path = get_package_share_directory(package_name)
    absolute_file_path = os.path.join(package_path, file_path)

    try:
        with open(absolute_file_path, 'r') as file:
            return yaml.safe_load(file)
    except EnvironmentError:  # parent of IOError, OSError *and* WindowsError where available
        return None


def generate_launch_description():
    #=================================Arguments=====================================
    # Launch arguments
    declared_arguments = []
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="true",
            description="Use simulation time",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "world",
            default_value=PathJoinSubstitution([
                FindPackageShare("franka_openvla"),
                "worlds", "franka_world.sdf"
            ]),
            description="Gazebo world file to load",
        )
    )

    # Initialize Arguments
    use_sim_time = LaunchConfiguration("use_sim_time")
    world = LaunchConfiguration("world")

    #====================================Robot and Semantic Description=====================================
    # Get URDF via xacro
    franka_xacro_file = os.path.join(
        get_package_share_directory('franka_description'),
        'robots', 'fr3', 'fr3.urdf.xacro'
    )

    robot_description_config = Command([
        FindExecutable(name='xacro'), ' ', franka_xacro_file,
        ' hand:=true',
        ' robot_ip:=dont-care',
        ' gazebo:=true',  # Enable gz_ros2_control plugin for Ignition
        ' ros2_control:=true',
        ' use_fake_hardware:=false'
    ])

    robot_description = {"robot_description": ParameterValue(robot_description_config, value_type=str)}

    # Get SRDF via xacro for semantic description
    robot_description_semantic_content = Command([
        FindExecutable(name="xacro"), " ",
        PathJoinSubstitution([
            FindPackageShare("franka_description"),
            "robots", "fr3", "fr3.srdf.xacro"
        ]),
        " hand:=true"
    ])

    robot_description_semantic = {
        "robot_description_semantic": ParameterValue(robot_description_semantic_content, value_type=str)
    }

    # Robot State Publisher
    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
        arguments=['--ros-args', '--log-level', 'WARN'],
    )

    #===========================================Gazebo launch and spawn=====================================
    # Set Gazebo resource path - CRITICAL: use parent directory like official launch
    franka_models_path = os.path.dirname(get_package_share_directory('franka_description'))
    franka_openvla_models_path = os.path.join(get_package_share_directory('franka_openvla'), 'models')
    os.environ['GZ_SIM_RESOURCE_PATH'] = f"{franka_models_path}:{franka_openvla_models_path}"

    # Gazebo Environment
    pkg_ros_gz_sim = get_package_share_directory("ros_gz_sim")
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_ros_gz_sim, "launch", "gz_sim.launch.py")
        ),
        launch_arguments={
            "gz_args": [world, " -r"],
            "on_exit_shutdown": "true"
        }.items(),
    )

    # Spawn Robot in Gazebo
    spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        arguments=[
            "-topic", "robot_description",
            "-name", "fr3",
            "-z", "0.0",
            "--ros-args", "--log-level", "WARN"
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    # Bridge between Gazebo and ROS2
    gz_ros2_bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=[
            "/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock",
            "/tf@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/tf_static@tf2_msgs/msg/TFMessage[gz.msgs.Pose_V",
            "/model/target_cube/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
            # RGB-D Camera topics
            "/rgbd_camera/image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/rgbd_camera/depth_image@sensor_msgs/msg/Image[gz.msgs.Image",
            "/rgbd_camera/points@sensor_msgs/msg/PointCloud2[gz.msgs.PointCloudPacked",
            "/rgbd_camera/camera_info@sensor_msgs/msg/CameraInfo[gz.msgs.CameraInfo",
            '--ros-args', '--log-level', 'WARN'
        ],
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )

    #=================================Controllers=====================================
    # Load controllers after spawn completes - event-driven like official launch
    load_joint_state_broadcaster = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active', 'joint_state_broadcaster'],
        output='screen'
    )

    # load the standard joint trajectory controller
    load_trajectory_controller = ExecuteProcess(
    cmd=[
        'ros2', 'control', 'load_controller',
        '--set-state', 'active',
        'arm_controller'  # Use the name defined in your YAML for the trajectory controller
    ],
    output='screen')

    # load the standard gripper controller
    load_gripper_controller = ExecuteProcess(
    cmd=[
        'ros2', 'control', 'load_controller',
        '--set-state', 'active',
        'gripper_controller'
    ],
    output='screen')

    kinematics_yaml = load_yaml(
        'franka_fr3_moveit_config', 'config/kinematics.yaml'
    )

    # Provide fallback if kinematics config can't be loaded
    if kinematics_yaml is None:
        print("Warning: Could not load kinematics config, using default")
        kinematics_yaml = {}

    #=================================Moveit Planning=======================================
    # Planning Functionality
    ompl_planning_pipeline_config = {
        'move_group': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        },
        'use_sim_time': True,
    }

    ompl_planning_pipeline_config_mtc = {
        'ompl': {
            'planning_plugin': 'ompl_interface/OMPLPlanner',
            'request_adapters': 'default_planner_request_adapters/AddTimeOptimalParameterization '
                                'default_planner_request_adapters/ResolveConstraintFrames '
                                'default_planner_request_adapters/FixWorkspaceBounds '
                                'default_planner_request_adapters/FixStartStateBounds '
                                'default_planner_request_adapters/FixStartStateCollision '
                                'default_planner_request_adapters/FixStartStatePathConstraints',
            'start_state_max_bounds_error': 0.1,
        },
        'use_sim_time': True,
    }

    ompl_planning_yaml = load_yaml(
        'franka_fr3_moveit_config', 'config/ompl_planning.yaml'
    )

    if ompl_planning_yaml is not None:
        ompl_planning_pipeline_config['move_group'].update(ompl_planning_yaml)
        ompl_planning_pipeline_config_mtc['ompl'].update(ompl_planning_yaml)
    else:
        print("Warning: Could not load OMPL planning config, using defaults")

    # Trajectory Execution Functionality
    moveit_simple_controllers_yaml = load_yaml(
    'franka_openvla', 'config/moveit_controllers.yaml'
    )

    if moveit_simple_controllers_yaml is None:
        print("Warning: Could not load MoveIt controllers config, using default")
        moveit_simple_controllers_yaml = {}

    moveit_controllers = {
        'moveit_simple_controller_manager': moveit_simple_controllers_yaml,
        'moveit_controller_manager': 'moveit_simple_controller_manager'
                                     '/MoveItSimpleControllerManager',
        'use_sim_time': True,
    }

    trajectory_execution = {
        'moveit_manage_controllers': True,
        'trajectory_execution.allowed_execution_duration_scaling': 1.2,
        'trajectory_execution.allowed_goal_duration_margin': 0.5,
        'trajectory_execution.allowed_start_tolerance': 0.01,
        'use_sim_time': True,
    }

    planning_scene_monitor_parameters = {
        'publish_planning_scene': True,
        'publish_geometry_updates': True,
        'publish_state_updates': True,
        'publish_transforms_updates': True,
        'planning_scene_monitor.robot_description_timeout': 10.0,
        'planning_scene_monitor.joint_state_timeout': 0.0,  # Disable timestamp validation for simulation
        'planning_scene_monitor.attached_collision_object_timeout': 5.0,
        'planning_scene_monitor.wait_for_initial_state_timeout': 10.0,
        'use_sim_time': True,
    }

    # MTC Execution Capability Configuration
    move_group_capabilities = {
        "capabilities": "move_group/ExecuteTaskSolutionCapability"
    }

    # Start the actual move_group node/action server
    run_move_group_node = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        namespace='',
        output='screen',
        parameters=[
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            move_group_capabilities,
            {"use_sim_time": True},
        ],
        arguments=['--ros-args', '--log-level', 'WARN'],
    )

    # Event handler: Launch MoveIt after arm_controller loads
    moveit_launch = RegisterEventHandler(
        OnProcessExit(
            target_action=load_trajectory_controller,  # Wait for arm_controller to load
            on_exit=[run_move_group_node],
        )
    )

    #=================================MoveIt Servo=====================================
    # Load Servo configuration following official MoveIt Servo pattern
    servo_yaml = load_yaml('franka_openvla', 'config/servo_params.yaml')

    # Wrap in "moveit_servo" namespace as required by MoveIt Servo
    servo_params = {"moveit_servo": servo_yaml}

    print(f"[Servo Config] Loaded config with move_group_name: {servo_yaml.get('move_group_name', 'NOT_FOUND')}")

    # MoveIt Servo node for real-time Cartesian control
    # Following official servo_example.launch.py pattern
    servo_node = Node(
        package='moveit_servo',
        executable='servo_node_main',
        name='servo_node',
        output='screen',
        parameters=[
            servo_params,  # Must be first! Contains move_group_name
            robot_description,
            robot_description_semantic,
            kinematics_yaml,
        ],
        arguments=['--ros-args', '--log-level', 'INFO'],
    )

    #=================================Rviz=====================================
    # RViz
    rviz_base = os.path.join(get_package_share_directory(
        'franka_fr3_moveit_config'), 'rviz')
    rviz_full_config = os.path.join(rviz_base, 'moveit.rviz')

    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='log',
        arguments=['-d', rviz_full_config, '--ros-args', '--log-level', 'WARN'],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            kinematics_yaml,
            {"use_sim_time": use_sim_time},
        ],
    )

    # Event handler: Launch pick&place after MoveIt services are available
    wait_for_moveit_service = ExecuteProcess(
        cmd=['bash', '-c', 'until ros2 service list | grep -q "plan_kinematic_path"; do echo "Waiting for MoveIt planning service..."; sleep 1; done; echo "MoveIt service available!"'],
        output='screen'
    )

    # LIBERO agentview camera: across table from robot, looking back at it
    static_tf_camera_node = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_world_tf',
        arguments=[
            '1.2', '0.0', '1.0',            # Position: across table, centered, 1.0m height (lower)
            '3.14159', '0.7', '0',          # Orientation: yaw=180° (backward), pitch=40° (down more)
            'world',
            'workspace_camera/camera_link/rgbd_camera_sensor'
        ],
        parameters=[{"use_sim_time": True}],
    )

    #=================================Nodes to launch=====================================

    # Event handler: Launch Servo after MoveIt services are available
    servo_launch = RegisterEventHandler(
        OnProcessExit(
            target_action=wait_for_moveit_service,
            on_exit=[servo_node],
        )
    )

    nodes_to_launch = [
        SetParameter(name='use_sim_time', value=True),
        robot_state_publisher_node,
        gazebo,
        spawn_entity,
        gz_ros2_bridge,
        load_joint_state_broadcaster,
        load_trajectory_controller,
        load_gripper_controller,
        rviz_node,
        moveit_launch,
        wait_for_moveit_service,
        servo_launch,  # Launch servo after MoveIt is ready
        static_tf_camera_node
    ]

    return LaunchDescription(declared_arguments + nodes_to_launch)