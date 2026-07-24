"""ROS 2 command and state interface for the Isaac Lab Rflyarm simulation."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import torch

import isaaclab.utils.math as math_utils

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
from geometry_msgs.msg import PoseStamped, TransformStamped, TwistStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import Bool, Float64, String
from tf2_msgs.msg import TFMessage


WORLD_FRAME = "world"
MAP_FRAME = "map"
BODY_FRAME = "body"
BASE_FRAME = "base_link"
TOOL_FRAME = "tool_center"
DEPTH_CAMERA_FRAME = "depth_camera_optical_frame"
TOOL_CENTER_OFFSET_LINK6 = (0.0004, 0.0070, 0.1552)


class Ros2Interface:
    """Expose the stable Rflyarm ROS 2 API without OmniGraph or Pegasus."""

    def __init__(
        self,
        robot,
        flight_controller,
        arm_controller,
        ceiling_bulb=None,
        depth_camera=None,
        publish_hz: float = 60.0,
        camera_publish_hz: float = 15.0,
    ):
        self.robot = robot
        self.flight = flight_controller
        self.arm = arm_controller
        self.ceiling_bulb = ceiling_bulb
        self.depth_camera = depth_camera
        self.publish_period = 1.0 / max(float(publish_hz), 1.0)
        self.publish_accumulator = 0.0
        self.camera_publish_period = 1.0 / max(float(camera_publish_hz), 1.0)
        self.camera_publish_accumulator = 0.0
        self.last_camera_frame = -1
        self.last_sim_time_ns = -1

        body_ids, body_names = robot.find_bodies(
            [BODY_FRAME, BASE_FRAME, "Link6"], preserve_order=True
        )
        if body_names != [BODY_FRAME, BASE_FRAME, "Link6"]:
            raise RuntimeError(
                "TF bodies do not match body/base/tool chain: "
                f"{body_names}"
            )
        self.body_id, self.base_id, self.link6_id = (
            int(body_id) for body_id in body_ids
        )
        self.tool_offset_link6 = torch.tensor(
            TOOL_CENTER_OFFSET_LINK6,
            device=robot.device,
            dtype=torch.float32,
        ).reshape(1, 3).repeat(robot.num_instances, 1)

        body_pose = self._body_pose_world(self.body_id)
        base_pose = self._body_pose_world(self.base_id)
        (
            self.body_to_base_position,
            self.body_to_base_quaternion,
        ) = self._relative_pose(body_pose, base_pose)
        self.body_to_base_position = self.body_to_base_position.detach().clone()
        self.body_to_base_quaternion = self.body_to_base_quaternion.detach().clone()

        if self.depth_camera is not None:
            offset = self.depth_camera.cfg.offset
            self.body_to_camera_position = torch.tensor(
                offset.pos, device=robot.device, dtype=torch.float32
            ).reshape(1, 3).repeat(robot.num_instances, 1)
            self.body_to_camera_quaternion = torch.tensor(
                offset.rot, device=robot.device, dtype=torch.float32
            ).reshape(1, 4).repeat(robot.num_instances, 1)
        else:
            self.body_to_camera_position = None
            self.body_to_camera_quaternion = None

        self.last_map_to_body = None
        self.last_base_to_tool = None
        self._tf_truth_metrics = {
            "body_position_error_m": math.inf,
            "body_orientation_error_deg": math.inf,
            "base_static_position_drift_m": math.inf,
            "base_static_orientation_drift_deg": math.inf,
            "tool_position_error_m": math.inf,
            "tool_orientation_error_deg": math.inf,
            "camera_position_error_m": math.inf,
            "camera_orientation_error_deg": math.inf,
        }

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
        self.drone_velocity_publisher = self.node.create_publisher(
            TwistStamped, "/drone/velocity", qos
        )
        self.drone_locked_publisher = self.node.create_publisher(
            Bool, "/drone/locked", qos
        )
        self.joint_state_publisher = self.node.create_publisher(JointState, "/joint_states", qos)
        self.ee_pose_publisher = self.node.create_publisher(PoseStamped, "/arm/ee_pose", qos)
        self.ceiling_bulb_state_publisher = None
        self.ceiling_bulb_distance_publisher = None
        self.ceiling_bulb_pose_publisher = None
        self.ceiling_bulb_initial_pose_publisher = None
        if self.ceiling_bulb is not None:
            self.ceiling_bulb_state_publisher = self.node.create_publisher(
                String, "/ceiling_bulb/state", qos
            )
            self.ceiling_bulb_distance_publisher = self.node.create_publisher(
                Float64, "/ceiling_bulb/distance", qos
            )
            self.ceiling_bulb_pose_publisher = self.node.create_publisher(
                PoseStamped, "/ceiling_bulb/pose", qos
            )
            self.ceiling_bulb_initial_pose_publisher = self.node.create_publisher(
                PoseStamped, "/ceiling_bulb/initial_pose", qos
            )
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
        self.node.create_subscription(
            Bool, "/drone/cmd_lock", self._flight_lock_callback, qos
        )
        self.node.create_subscription(JointState, "/joint_command", self._joint_command_callback, qos)
        self.node.create_subscription(PoseStamped, "/arm/cmd_pose", self._arm_pose_callback, qos)
        if self.ceiling_bulb is not None:
            self.node.create_subscription(
                String, "/ceiling_bulb/cmd", self._ceiling_bulb_command_callback, qos
            )

        # Fail fast if the project-local URDF/Lula model cannot be used.
        self.arm.kinematics.load()
        self._publish_static_tf()
        self.node.get_logger().info(
            "Rflyarm ROS 2 ready: controls, state, /depth_camera/color/image_raw, "
            "/depth_camera/depth/image_raw, camera_info, world/map/body/"
            "base_link/tool_center TF"
        )

    @staticmethod
    def _proxy_to_torch(array) -> torch.Tensor:
        value = getattr(array, "torch", array)
        return value if torch.is_tensor(value) else torch.as_tensor(value)

    def _body_pose_world(self, body_id: int) -> torch.Tensor:
        poses = self._proxy_to_torch(self.robot.data.body_link_pose_w)
        return poses[:, int(body_id)]

    @staticmethod
    def _relative_pose(
        parent_pose: torch.Tensor,
        child_pose: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return math_utils.subtract_frame_transforms(
            parent_pose[:, :3],
            parent_pose[:, 3:],
            child_pose[:, :3],
            child_pose[:, 3:],
        )

    def _tool_pose_world_truth(self) -> tuple[torch.Tensor, torch.Tensor]:
        link6_pose = self._body_pose_world(self.link6_id)
        return math_utils.combine_frame_transforms(
            link6_pose[:, :3],
            link6_pose[:, 3:],
            self.tool_offset_link6,
            None,
        )

    def _tool_pose_base_truth(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_pose = self._body_pose_world(self.base_id)
        tool_position, tool_quaternion = self._tool_pose_world_truth()
        tool_pose = torch.cat((tool_position, tool_quaternion), dim=1)
        return self._relative_pose(base_pose, tool_pose)

    @staticmethod
    def _pose_errors(
        actual_position: torch.Tensor,
        actual_quaternion: torch.Tensor,
        expected_position: torch.Tensor,
        expected_quaternion: torch.Tensor,
    ) -> tuple[float, float]:
        position_error = float(
            torch.linalg.vector_norm(
                actual_position - expected_position, dim=1
            ).max().item()
        )
        actual_quaternion = actual_quaternion / torch.linalg.vector_norm(
            actual_quaternion, dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        expected_quaternion = expected_quaternion / torch.linalg.vector_norm(
            expected_quaternion, dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        dot = torch.sum(
            actual_quaternion * expected_quaternion, dim=1
        ).abs().clamp(0.0, 1.0)
        orientation_error_deg = math.degrees(
            float((2.0 * torch.acos(dot)).max().item())
        )
        return position_error, orientation_error_deg

    def _transform_message(
        self,
        parent_frame: str,
        child_frame: str,
        position,
        quaternion,
        sim_time_ns: int,
    ) -> TransformStamped:
        position_values = torch.as_tensor(position).reshape(-1).tolist()
        quaternion_values = torch.as_tensor(quaternion).reshape(-1).tolist()
        transform = TransformStamped()
        transform.header.stamp = self._stamp(sim_time_ns)
        transform.header.frame_id = str(parent_frame)
        transform.child_frame_id = str(child_frame)
        (
            transform.transform.translation.x,
            transform.transform.translation.y,
            transform.transform.translation.z,
        ) = (float(value) for value in position_values)
        (
            transform.transform.rotation.x,
            transform.transform.rotation.y,
            transform.transform.rotation.z,
            transform.transform.rotation.w,
        ) = (float(value) for value in quaternion_values)
        return transform

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

    def _flight_lock_callback(self, message: Bool) -> None:
        try:
            if bool(message.data):
                self.flight.lock_platform()
                detail = "locked by a PhysX FixedJoint at the current physical world pose"
            else:
                self.flight.unlock_platform()
                detail = "unlocked while flight control remains active"
            self.node.get_logger().info(f"Drone platform {detail}")
        except Exception as exc:
            self.node.get_logger().error(f"Rejected platform lock command: {exc}")

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

    def _ceiling_bulb_command_callback(self, message: String) -> None:
        try:
            command = str(message.data).strip().lower()
            if command in ("complete_removal", "release", "release_bulb"):
                self.ceiling_bulb.complete_removal()
                detail = "released after the replacement sequence"
            elif command in ("engage", "engage_loose", "attach"):
                self.ceiling_bulb.engage_loose()
                detail = "re-engaged from the existing gripper grasp"
            elif command in ("resume_grasp", "release_hold"):
                self.ceiling_bulb.resume_grasp()
                detail = "released the temporary hold after regrasp"
            elif command in ("hold_tightening_stroke", "finish_tightening"):
                allow_lock = command == "finish_tightening"
                actual_angle_degrees = self.ceiling_bulb.hold_tightening_stroke(
                    allow_lock=allow_lock
                )
                detail = (
                    "sampled actual bulb/socket angle "
                    f"{actual_angle_degrees:.3f} degrees"
                )
            else:
                raise ValueError(
                    "/ceiling_bulb/cmd accepts complete_removal, engage_loose, "
                    "resume_grasp, hold_tightening_stroke, or finish_tightening; "
                    "the bulb is already held by the gripper and is never loaded "
                    "by this interface"
                )
            self.node.get_logger().info(f"Ceiling bulb {detail}")
        except Exception as exc:
            self.node.get_logger().error(f"Rejected ceiling bulb command: {exc}")

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
        self._publish_drone_locked()
        self._publish_dynamic_tf(sim_time_ns)
        self._publish_joint_states(sim_time_ns)
        self._publish_ee_pose(sim_time_ns)
        self._publish_ceiling_bulb_state()

    def _publish_drone_locked(self) -> None:
        message = Bool()
        message.data = self.flight.platform_locked
        self.drone_locked_publisher.publish(message)

    def _publish_ceiling_bulb_state(self) -> None:
        if self.ceiling_bulb_state_publisher is None:
            return
        message = String()
        message.data = self.ceiling_bulb.progress.state.value
        self.ceiling_bulb_state_publisher.publish(message)
        distance_message = Float64()
        distance_message.data = self.ceiling_bulb.separation_m
        self.ceiling_bulb_distance_publisher.publish(distance_message)
        current_pose = self.ceiling_bulb.current_root_pose_xyzw
        self.ceiling_bulb_pose_publisher.publish(
            self._pose_message(current_pose, self.last_sim_time_ns)
        )
        initial_pose = self.ceiling_bulb.initial_root_pose_xyzw
        if initial_pose is not None:
            self.ceiling_bulb_initial_pose_publisher.publish(
                self._pose_message(initial_pose, self.last_sim_time_ns)
            )

    def _pose_message(self, pose_xyzw: tuple[float, ...], sim_time_ns: int) -> PoseStamped:
        message = PoseStamped()
        message.header.stamp = self._stamp(sim_time_ns)
        message.header.frame_id = "map"
        (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = pose_xyzw
        return message

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
        position, quaternion, linear_velocity, angular_velocity = self.flight.state()
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

        velocity_message = TwistStamped()
        velocity_message.header.stamp = message.header.stamp
        velocity_message.header.frame_id = "map"
        (
            velocity_message.twist.linear.x,
            velocity_message.twist.linear.y,
            velocity_message.twist.linear.z,
        ) = (float(value) for value in linear_velocity[0].tolist())
        (
            velocity_message.twist.angular.x,
            velocity_message.twist.angular.y,
            velocity_message.twist.angular.z,
        ) = (float(value) for value in angular_velocity[0].tolist())
        self.drone_velocity_publisher.publish(velocity_message)

    def _publish_dynamic_tf(self, sim_time_ns: int) -> None:
        position, quaternion, _linear_velocity, _angular_velocity = self.flight.state()
        tool_position, tool_quaternion = self._tool_pose_base_truth()
        self.last_map_to_body = (
            position.detach().clone(),
            quaternion.detach().clone(),
        )
        self.last_base_to_tool = (
            tool_position.detach().clone(),
            tool_quaternion.detach().clone(),
        )
        message = TFMessage()
        message.transforms = [
            self._transform_message(
                MAP_FRAME,
                BODY_FRAME,
                position[0],
                quaternion[0],
                sim_time_ns,
            ),
            self._transform_message(
                BASE_FRAME,
                TOOL_FRAME,
                tool_position[0],
                tool_quaternion[0],
                sim_time_ns,
            ),
        ]
        self.tf_publisher.publish(message)
        self._update_dynamic_tf_truth_metrics()

    def _publish_static_tf(self) -> None:
        identity_position = torch.zeros((3,), dtype=torch.float32)
        identity_quaternion = torch.tensor(
            (0.0, 0.0, 0.0, 1.0), dtype=torch.float32
        )
        transforms = [
            self._transform_message(
                WORLD_FRAME,
                MAP_FRAME,
                identity_position,
                identity_quaternion,
                0,
            ),
            self._transform_message(
                BODY_FRAME,
                BASE_FRAME,
                self.body_to_base_position[0],
                self.body_to_base_quaternion[0],
                0,
            ),
        ]
        if self.depth_camera is not None:
            transforms.append(
                self._transform_message(
                    BODY_FRAME,
                    DEPTH_CAMERA_FRAME,
                    self.body_to_camera_position[0],
                    self.body_to_camera_quaternion[0],
                    0,
                )
            )
        message = TFMessage()
        message.transforms = transforms
        self.tf_static_publisher.publish(message)

    def _update_dynamic_tf_truth_metrics(self) -> None:
        if self.last_map_to_body is None or self.last_base_to_tool is None:
            return
        map_body_position, map_body_quaternion = self.last_map_to_body
        base_tool_position, base_tool_quaternion = self.last_base_to_tool

        body_pose_truth = self._body_pose_world(self.body_id)
        (
            self._tf_truth_metrics["body_position_error_m"],
            self._tf_truth_metrics["body_orientation_error_deg"],
        ) = self._pose_errors(
            map_body_position,
            map_body_quaternion,
            body_pose_truth[:, :3],
            body_pose_truth[:, 3:],
        )

        base_pose_truth = self._body_pose_world(self.base_id)
        current_body_to_base_position, current_body_to_base_quaternion = (
            self._relative_pose(body_pose_truth, base_pose_truth)
        )
        (
            self._tf_truth_metrics["base_static_position_drift_m"],
            self._tf_truth_metrics["base_static_orientation_drift_deg"],
        ) = self._pose_errors(
            self.body_to_base_position,
            self.body_to_base_quaternion,
            current_body_to_base_position,
            current_body_to_base_quaternion,
        )

        map_base_position, map_base_quaternion = (
            math_utils.combine_frame_transforms(
                map_body_position,
                map_body_quaternion,
                self.body_to_base_position,
                self.body_to_base_quaternion,
            )
        )
        map_tool_position, map_tool_quaternion = (
            math_utils.combine_frame_transforms(
                map_base_position,
                map_base_quaternion,
                base_tool_position,
                base_tool_quaternion,
            )
        )
        tool_position_truth, tool_quaternion_truth = self._tool_pose_world_truth()
        (
            self._tf_truth_metrics["tool_position_error_m"],
            self._tf_truth_metrics["tool_orientation_error_deg"],
        ) = self._pose_errors(
            map_tool_position,
            map_tool_quaternion,
            tool_position_truth,
            tool_quaternion_truth,
        )

    def _update_camera_tf_truth_metrics(self) -> None:
        if self.depth_camera is None:
            return
        body_position, body_quaternion, _linear, _angular = self.flight.state()
        camera_position_tf, camera_quaternion_tf = (
            math_utils.combine_frame_transforms(
                body_position,
                body_quaternion,
                self.body_to_camera_position,
                self.body_to_camera_quaternion,
            )
        )
        camera_position_truth = self._proxy_to_torch(
            self.depth_camera.data.pos_w
        )
        camera_quaternion_truth = self._proxy_to_torch(
            self.depth_camera.data.quat_w_ros
        )
        (
            self._tf_truth_metrics["camera_position_error_m"],
            self._tf_truth_metrics["camera_orientation_error_deg"],
        ) = self._pose_errors(
            camera_position_tf,
            camera_quaternion_tf,
            camera_position_truth,
            camera_quaternion_truth,
        )

    def tf_truth_metrics(self) -> dict[str, float]:
        """Return the latest TF-vs-PhysX/camera ground-truth errors."""

        return dict(self._tf_truth_metrics)

    def _publish_camera_images(self, sim_time_ns: int) -> None:
        camera_data = self.depth_camera.data
        frame = int(self.depth_camera.frame.torch[0].item())
        if frame == self.last_camera_frame:
            return
        self.last_camera_frame = frame
        self._update_camera_tf_truth_metrics()

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
            position, quaternion = self._tool_pose_base_truth()
        except Exception as exc:
            self.node.get_logger().error(f"Tool-center truth failed: {exc}")
            return
        message = PoseStamped()
        message.header.stamp = self._stamp(sim_time_ns)
        message.header.frame_id = BASE_FRAME
        message.pose.position.x, message.pose.position.y, message.pose.position.z = (
            float(value) for value in position[0].tolist()
        )
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = (float(value) for value in quaternion[0].tolist())
        self.ee_pose_publisher.publish(message)

    def shutdown(self) -> None:
        if self.node is not None:
            self.node.destroy_node()
            self.node = None
        if rclpy.ok():
            rclpy.shutdown()
