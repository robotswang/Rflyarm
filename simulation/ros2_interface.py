"""ROS 2 command and state interface for the Isaac Lab Rflyarm simulation."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

# This project uses rclpy directly and does not create OmniGraph bridge nodes.
# The launcher has already selected Isaac Sim's matching ROS libraries; expose
# its Python 3.12 message/rclpy packages without enabling renderer-heavy UI
# dependencies from the full bridge extension.
isaac_sim_root = Path(sys.executable).resolve().parents[3]
ros_core_extension = isaac_sim_root / "exts" / "isaacsim.ros2.core"
ros_python_path = ros_core_extension / "humble" / "rclpy"
if not ros_python_path.is_dir():
    raise FileNotFoundError(f"Isaac Sim ROS 2 Humble backend not found: {ros_python_path}")
if str(ros_python_path) not in sys.path:
    sys.path.insert(0, str(ros_python_path))

import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from tf2_msgs.msg import TFMessage


BODY_FRAME = "body"
DEPTH_CAMERA_FRAME = "depth_camera_optical_frame"


class Ros2Interface:
    """Expose the stable Rflyarm ROS 2 API without OmniGraph or Pegasus."""

    def __init__(
        self,
        robot,
        flight_controller,
        arm_controller,
        depth_camera=None,
        publish_hz: float = 60.0,
        camera_publish_hz: float = 15.0,
    ):
        self.robot = robot
        self.flight = flight_controller
        self.arm = arm_controller
        self.depth_camera = depth_camera
        self.publish_period = 1.0 / max(float(publish_hz), 1.0)
        self.publish_accumulator = 0.0
        self.camera_publish_period = 1.0 / max(float(camera_publish_hz), 1.0)
        self.camera_publish_accumulator = 0.0
        self.last_camera_frame = -1
        self.last_sim_time_ns = -1

        if not rclpy.ok():
            rclpy.init(args=None)
        self.node = rclpy.create_node("rflyarm_simulation")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.drone_pose_publisher = self.node.create_publisher(PoseStamped, "/drone/pose", qos)
        self.joint_state_publisher = self.node.create_publisher(JointState, "/joint_states", qos)
        self.ee_pose_publisher = self.node.create_publisher(PoseStamped, "/arm/ee_pose", qos)
        clock_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.clock_publisher = self.node.create_publisher(Clock, "/clock", clock_qos)
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=2,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.depth_image_publisher = self.node.create_publisher(
            Image, "/depth_camera/depth/image_raw", camera_qos
        )
        self.color_image_publisher = self.node.create_publisher(
            Image, "/depth_camera/color/image_raw", camera_qos
        )
        self.color_camera_info_publisher = self.node.create_publisher(
            CameraInfo, "/depth_camera/color/camera_info", camera_qos
        )
        self.depth_camera_info_publisher = self.node.create_publisher(
            CameraInfo, "/depth_camera/depth/camera_info", camera_qos
        )
        tf_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        static_tf_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.tf_publisher = self.node.create_publisher(TFMessage, "/tf", tf_qos)
        self.tf_static_publisher = self.node.create_publisher(
            TFMessage, "/tf_static", static_tf_qos
        )
        self.node.create_subscription(PoseStamped, "/drone/cmd_pose", self._flight_command_callback, qos)
        self.node.create_subscription(JointState, "/joint_command", self._joint_command_callback, qos)
        self.node.create_subscription(PoseStamped, "/arm/cmd_pose", self._arm_pose_callback, qos)

        # Fail fast if the project-local URDF/Lula model cannot be used.
        self.arm.kinematics.load()
        self._publish_camera_static_tf()
        self.node.get_logger().info(
            "Rflyarm ROS 2 ready: controls, state, /depth_camera/color/image_raw, "
            "/depth_camera/depth/image_raw, camera_info, /tf"
        )

    @staticmethod
    def _yaw_from_quaternion(message) -> float:
        x = float(message.x)
        y = float(message.y)
        z = float(message.z)
        w = float(message.w)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1.0e-8:
            raise ValueError("Quaternion norm is zero")
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _flight_command_callback(self, message: PoseStamped) -> None:
        try:
            if message.header.frame_id not in ("", "map"):
                raise ValueError("/drone/cmd_pose frame_id must be 'map'")
            yaw = self._yaw_from_quaternion(message.pose.orientation)
            self.flight.set_target(
                (message.pose.position.x, message.pose.position.y, message.pose.position.z), yaw
            )
        except Exception as exc:
            self.node.get_logger().error(f"Rejected flight command: {exc}")

    def _joint_command_callback(self, message: JointState) -> None:
        try:
            self.arm.set_named_targets(message.name, message.position)
        except Exception as exc:
            self.node.get_logger().error(f"Rejected joint command: {exc}")

    def _arm_pose_callback(self, message: PoseStamped) -> None:
        try:
            result = self.arm.set_cartesian_target(
                frame_id=message.header.frame_id,
                position=(message.pose.position.x, message.pose.position.y, message.pose.position.z),
                quaternion_xyzw=(
                    message.pose.orientation.x,
                    message.pose.orientation.y,
                    message.pose.orientation.z,
                    message.pose.orientation.w,
                ),
            )
            self.node.get_logger().info(
                "IK accepted: position residual %.6f m, orientation residual %.3f deg"
                % (result.position_error_m, math.degrees(result.orientation_error_rad))
            )
        except Exception as exc:
            self.node.get_logger().error(f"Rejected arm pose command: {exc}")

    def process_commands(self) -> None:
        """Process pending ROS commands without advancing any ROS-side clock."""

        rclpy.spin_once(self.node, timeout_sec=0.0)

    def publish_states(self, sim_time_ns: int, dt: float) -> None:
        """Publish state using the authoritative post-step simulation time."""

        sim_time_ns = int(sim_time_ns)
        if sim_time_ns < self.last_sim_time_ns:
            raise ValueError(
                f"Simulation time moved backwards: {sim_time_ns} < {self.last_sim_time_ns}"
            )
        self.last_sim_time_ns = sim_time_ns
        self.publish_accumulator += max(float(dt), 0.0)
        self.camera_publish_accumulator += max(float(dt), 0.0)
        if (
            self.depth_camera is not None
            and self.camera_publish_accumulator >= self.camera_publish_period
        ):
            self.camera_publish_accumulator %= self.camera_publish_period
            self._publish_camera_images(sim_time_ns)
        if self.publish_accumulator < self.publish_period:
            return
        self.publish_accumulator %= self.publish_period
        self._publish_clock(sim_time_ns)
        self._publish_drone_pose(sim_time_ns)
        self._publish_body_tf(sim_time_ns)
        self._publish_joint_states(sim_time_ns)
        self._publish_ee_pose(sim_time_ns)

    @staticmethod
    def _stamp(sim_time_ns: int) -> Time:
        stamp = Time()
        stamp.sec = int(sim_time_ns // 1_000_000_000)
        stamp.nanosec = int(sim_time_ns % 1_000_000_000)
        return stamp

    def _publish_clock(self, sim_time_ns: int) -> None:
        message = Clock()
        message.clock = self._stamp(sim_time_ns)
        self.clock_publisher.publish(message)

    def _publish_drone_pose(self, sim_time_ns: int) -> None:
        position, quaternion, _linear_velocity, _angular_velocity = self.flight.state()
        message = PoseStamped()
        message.header.stamp = self._stamp(sim_time_ns)
        message.header.frame_id = "map"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            float(value) for value in position[0].tolist()
        )
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = (float(value) for value in quaternion[0].tolist())
        self.drone_pose_publisher.publish(message)

    def _publish_body_tf(self, sim_time_ns: int) -> None:
        position, quaternion, _linear_velocity, _angular_velocity = self.flight.state()
        transform = TransformStamped()
        transform.header.stamp = self._stamp(sim_time_ns)
        transform.header.frame_id = "map"
        transform.child_frame_id = BODY_FRAME
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = (
            float(value) for value in position[0].tolist()
        )
        (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ) = (float(value) for value in quaternion[0].tolist())
        message = TFMessage()
        message.transforms = [transform]
        self.tf_publisher.publish(message)

    def _publish_camera_static_tf(self) -> None:
        if self.depth_camera is None:
            return
        offset = self.depth_camera.cfg.offset
        transform = TransformStamped()
        transform.header.stamp = self._stamp(0)
        transform.header.frame_id = BODY_FRAME
        transform.child_frame_id = DEPTH_CAMERA_FRAME
        (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ) = (float(value) for value in offset.pos)
        (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ) = (float(value) for value in offset.rot)
        message = TFMessage()
        message.transforms = [transform]
        self.tf_static_publisher.publish(message)

    def _publish_camera_images(self, sim_time_ns: int) -> None:
        camera_data = self.depth_camera.data
        frame = int(self.depth_camera.frame.torch[0].item())
        if frame == self.last_camera_frame:
            return
        self.last_camera_frame = frame

        rgb_tensor = camera_data.output["rgb"].torch[0]
        rgb = np.ascontiguousarray(rgb_tensor.detach().cpu().numpy(), dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise RuntimeError(f"Unexpected RGB image shape: {rgb.shape}")

        depth_tensor = camera_data.output["distance_to_image_plane"].torch[0]
        if depth_tensor.ndim == 3 and depth_tensor.shape[-1] == 1:
            depth_tensor = depth_tensor[..., 0]
        depth = np.ascontiguousarray(depth_tensor.detach().cpu().numpy(), dtype=np.float32)
        height, width = depth.shape
        stamp = self._stamp(sim_time_ns)

        color_image = Image()
        color_image.header.stamp = stamp
        color_image.header.frame_id = DEPTH_CAMERA_FRAME
        color_image.height = height
        color_image.width = width
        color_image.encoding = "rgb8"
        color_image.is_bigendian = False
        color_image.step = width * 3
        color_image.data = rgb.tobytes()
        self.color_image_publisher.publish(color_image)

        depth_image = Image()
        depth_image.header.stamp = stamp
        depth_image.header.frame_id = DEPTH_CAMERA_FRAME
        depth_image.height = height
        depth_image.width = width
        depth_image.encoding = "32FC1"
        depth_image.is_bigendian = False
        depth_image.step = width * 4
        depth_image.data = depth.tobytes()
        self.depth_image_publisher.publish(depth_image)

        intrinsic = camera_data.intrinsic_matrices.torch[0].detach().cpu().numpy()
        info = CameraInfo()
        info.header.stamp = stamp
        info.header.frame_id = DEPTH_CAMERA_FRAME
        info.height = height
        info.width = width
        info.distortion_model = "plumb_bob"
        info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
        info.k = intrinsic.reshape(-1).astype(float).tolist()
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.p = [
            float(intrinsic[0, 0]),
            0.0,
            float(intrinsic[0, 2]),
            0.0,
            0.0,
            float(intrinsic[1, 1]),
            float(intrinsic[1, 2]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        self.color_camera_info_publisher.publish(info)
        self.depth_camera_info_publisher.publish(info)

    def _publish_joint_states(self, sim_time_ns: int) -> None:
        names, positions, velocities, efforts = self.arm.joint_state()
        message = JointState()
        message.header.stamp = self._stamp(sim_time_ns)
        message.name = names
        message.position = positions
        message.velocity = velocities
        message.effort = efforts
        self.joint_state_publisher.publish(message)

    def _publish_ee_pose(self, sim_time_ns: int) -> None:
        try:
            position, quaternion = self.arm.end_effector_pose()
        except Exception as exc:
            self.node.get_logger().error(f"FK failed: {exc}")
            return
        message = PoseStamped()
        message.header.stamp = self._stamp(sim_time_ns)
        message.header.frame_id = "base_link"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            float(value) for value in position
        )
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = (float(value) for value in quaternion)
        self.ee_pose_publisher.publish(message)

    def shutdown(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if rclpy.ok():
            rclpy.shutdown()
