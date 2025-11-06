import sys
import time
import rclpy
import torch
import numpy as np
from PIL import Image as PILImage
from functools import wraps
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from geometry_msgs.msg import Vector3
from vla_interfaces.msg import VLAAction
from bitsandbytes.nn import Params4bit, Int8Params
from transformers.modeling_utils import PreTrainedModel
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

class VLA(Node):
    """ROS2 node for OpenVLA inference"""
    def __init__(self):
        super().__init__('openvla_node')

        # Declare parameters
        self.declare_parameter('model_name', 'openvla/openvla-7b-finetuned-libero-spatial')
        self.declare_parameter('unnorm_key', 'libero_spatial')
        self.declare_parameter('camera_topic', '/rgbd_camera/image')
        self.declare_parameter('action_topic', '/vla/delta_actions')
        self.declare_parameter('instruction', 'pick the cube on the table')

        # Get parameters
        self.model_name = self.get_parameter('model_name').value
        self.unnorm_key = self.get_parameter('unnorm_key').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.action_topic = self.get_parameter('action_topic').value
        self.instruction = self.get_parameter('instruction').value

        # Timing tracking for inference speed measurement
        self.last_inference_time = None
        self.inference_times = []
        self.inference_count = 0

        self.get_logger().info(f'OpenVLA Node starting...')
        self.get_logger().info(f'   Model: {self.model_name}')
        self.get_logger().info(f'   Camera topic: {self.camera_topic}')
        self.get_logger().info(f'   Instruction: {self.instruction}')

        # CV Bridge for ROS <-> OpenCV conversion
        self.bridge = CvBridge()

        # Load model
        self.get_logger().info('Loading the model...')
        self.model = None
        self.processor = None
        self.load_model()

        # Set use_sim_time parameter
        self.set_parameters([rclpy.parameter.Parameter('use_sim_time', rclpy.Parameter.Type.BOOL, True)])

        # Create subscription with matching QoS
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,  # Gazebo uses BEST_EFFORT
            depth=10
        )

        # Image subscriber
        self.subscription = self.create_subscription(
            Image,
            self.camera_topic,
            self.image_callback,
            qos_profile
        )

        # Action publisher - now using custom VLAAction message
        self.action_pub = self.create_publisher(
            VLAAction,
            self.action_topic,
            10
        )

        self.get_logger().info('OpenVLA Node ready!')
        self.get_logger().info('Publishing delta actions (not absolute poses)!')

    def load_model(self):
        """Load 4-bit quantized OpenVLA model with all necessary patches"""
        self.get_logger().info(f'   Patching .to() method...')
        original_to = PreTrainedModel.to

        # Patch .to() method to bypass quantization error
        @wraps(original_to)
        def patched_to(self, *args, **kwargs):
            if getattr(self, "quantization_method", None) is not None:
                return self
            return original_to(self, *args, **kwargs)
        
        PreTrainedModel.to = patched_to

        # Configure 4-bit quantization
        self.get_logger().info(f'   Configuring 4-bit quantization...')
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        # Load processor
        self.get_logger().info(f'   Loading processor')
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        # Load model
        self.get_logger().info(f'    Loading 4-bit model')
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            quantization_config=quant_config,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        # Convert vision backbone to bfloat16
        self.get_logger().info(f'   Converting vision backbone to bfloat16')
        if hasattr(self.model, 'vision_backbone'):
            self.model.vision_backbone = self.model.vision_backbone.to(torch.bfloat16)
        if hasattr(self.model, 'projector'):
            self.model.projector = self.model.projector.to(torch.bfloat16)
        
        # Move buffers to GPU
        self.get_logger().info(f'   Moving buffers to GPU')
        for name, buffer in self.model.named_buffers():
            if buffer.device.type == 'cpu':
                buffer.data = buffer.data.to('cuda')
        
        # Convert all non-quantized modules to bfloat16
        self.get_logger().info(f'   Converting non-quantized modules...')
        self.convert_to_bfloat16(self.model)

        gpu_mem = torch.cuda.memory_allocated(0) / 1e9
        self.get_logger().info(f'   Model loaded. GPU memory: {gpu_mem:.2f} GB')
    
    def convert_to_bfloat16(self, module):
        """Recursively convert non-quantized modules to bfloat16"""
        for name, child in module.named_children():
            # Skip quantized layers
            if hasattr(child, 'weight') and isinstance(child.weight, (Params4bit, Int8Params)):
                continue

            # Recursively convert children
            self.convert_to_bfloat16(child)

            # Convert parameters
            for param_name, param in child.named_parameters(recurse=False):
                if param.dtype in [torch.float32, torch.float16]:
                    param.data = param.data.to(torch.bfloat16)

            # Convert buffers
            for buffer_name, buffer in child.named_buffers(recurse=False):
                if buffer.dtype in [torch.float32, torch.float16]:
                    buffer.data = buffer.data.to(torch.bfloat16)



    def image_callback(self, msg):
        """Process incoming camera image and run VLA inference."""
        try:
            # Start timing
            callback_start = time.time()

            # Convert ROS Image to PIL Image
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            pil_image = PILImage.fromarray(cv_image)

            # Run inference
            inference_start = time.time()
            action = self.predict_action(pil_image, self.instruction)
            inference_end = time.time()

            # Publish action
            self.publish_action(action)

            # Calculate timing statistics
            callback_end = time.time()
            self.update_timing_stats(callback_start, callback_end, inference_end - inference_start)

        except Exception as e:
            self.get_logger().error(f'Error in image callback: {e}')
    
    def predict_action(self, image, instruction):
        """Run VLA Inference on image with instruction"""
        # Format prompt
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Process inputs
        inputs = self.processor(prompt, image)

        # Convert to bfloat16 and move to GPU
        inputs = {k: v.to("cuda", dtype=torch.bfloat16) if isinstance(v, torch.Tensor) and v.dtype in
                [torch.float32, torch.float16]
                else v.to("cuda") if isinstance(v, torch.Tensor)
                else v
                for k, v in inputs.items()}

        # Run inference
        with torch.no_grad():
            action = self.model.predict_action(
                **inputs,
                unnorm_key=self.unnorm_key,
                do_sample=False
            )

        # Convert to numpy
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()

        self.get_logger().info(f"Predicted action: {action}")

        return action

    def publish_action(self, action):
        """Publish predicted delta action using VLAAction message

        IMPORTANT: OpenVLA outputs DELTA actions (relative changes), not absolute poses!
        Action format: [delta_x, delta_y, delta_z, delta_roll, delta_pitch, delta_yaw, gripper]
        """
        msg = VLAAction()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "fr3_link0"

        # Delta position (relative changes in meters)
        msg.delta_pos = Vector3()
        msg.delta_pos.x = float(action[0])
        msg.delta_pos.y = float(action[1])
        msg.delta_pos.z = float(action[2])

        # Delta rotation (relative changes in radians)
        msg.delta_rot = Vector3()
        msg.delta_rot.x = float(action[3])  # delta_roll
        msg.delta_rot.y = float(action[4])  # delta_pitch
        msg.delta_rot.z = float(action[5])  # delta_yaw

        # Gripper (absolute: 0=open, 1=closed)
        msg.gripper = float(action[6])

        self.action_pub.publish(msg)

    def update_timing_stats(self, callback_start, callback_end, inference_time):
        """Update and log timing statistics for inference speed measurement"""
        self.inference_count += 1

        # Calculate frequency
        if self.last_inference_time is not None:
            time_since_last = callback_start - self.last_inference_time
            current_freq = 1.0 / time_since_last if time_since_last > 0 else 0
            self.inference_times.append(time_since_last)

            # Keep only last 50 measurements for rolling average
            if len(self.inference_times) > 50:
                self.inference_times.pop(0)

            # Log every 10 inferences
            if self.inference_count % 10 == 0:
                avg_period = np.mean(self.inference_times)
                avg_freq = 1.0 / avg_period if avg_period > 0 else 0
                total_callback_time = callback_end - callback_start

                self.get_logger().info(
                    f'Inference Stats (last 50 samples):\n'
                    f'  Average Frequency: {avg_freq:.2f} Hz\n'
                    f'  Current Frequency: {current_freq:.2f} Hz\n'
                    f'  Inference Time: {inference_time*1000:.1f} ms\n'
                    f'  Total Callback Time: {total_callback_time*1000:.1f} ms\n'
                    f'  Count: {self.inference_count}',
                    throttle_duration_sec=5.0
                )

        self.last_inference_time = callback_start


    def destroy_node(self):
        """Override destroy_node to ensure proper cleanup."""
        self.cleanup_gpu()
        super().destroy_node()
    
    def cleanup_gpu(self):
        if hasattr(self, '_gpu_cleaned') and self._gpu_cleaned:
            return  # Already cleaned

        # Clear model references
        if hasattr(self, 'model') and self.model is not None:
            try:
                if hasattr(self.model, 'cpu'):
                    self.model.cpu()  # Move to CPU first
                del self.model
                self.model = None
            except:
                pass

        if hasattr(self, 'processor') and self.processor is not None:
            try:
                del self.processor
                self.processor = None
            except:
                pass

        # Force garbage collection
        try:
            import gc
            gc.collect()
        except:
            pass

        # Clear CUDA cache
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass

        # Mark as cleaned
        self._gpu_cleaned = True

        try:
            gpu_mem = torch.cuda.memory_allocated(0) / 1e9
            print(f'GPU memory cleared. Remaining: {gpu_mem:.2f} GB')
        except:
            pass

def main(args=None):
    rclpy.init(args=args)
    node = None
    
    try:
        node = VLA()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        if node:
            try:
                print(f"Node error: {e}")
            except:
                print(f'Node error: {e}')
    finally:
        if node:
            try:
                node.destroy_node()
            except Exception as e:
                print(f'Cleanup warning: {e}')
        
        try:
            rclpy.shutdown()
        except:
            pass 


