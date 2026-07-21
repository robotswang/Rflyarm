"""ROS 2 command and state interface for the Isaac Lab Rflyarm simulation."""

from __future__ import annotations

import math
from pathlib import Path
import sys

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
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


class Ros2Interface:
    """Expose the stable Rflyarm ROS 2 API without OmniGraph or Pegasus."""

    def __init__(self, robot, flight_controller, arm_controller, publish_hz: float = 60.0):
        self.robot = robot
        self.flight = flight_controller
        self.arm = arm_controller
        self.publish_period = 1.0 / max(float(publish_hz), 1.0)
        self.publish_accumulator = 0.0
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
        self.node.create_subscription(PoseStamped, "/drone/cmd_pose", self._flight_command_callback, qos)
        self.node.create_subscription(JointState, "/joint_command", self._joint_command_callback, qos)
        self.node.create_subscription(PoseStamped, "/arm/cmd_pose", self._arm_pose_callback, qos)

        # Fail fast if the project-local URDF/Lula model cannot be used.
        self.arm.kinematics.load()
        self.node.get_logger().info(
            "Rflyarm ROS 2 ready: /drone/cmd_pose, /joint_command, /arm/cmd_pose, /clock"
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
        if self.publish_accumulator < self.publish_period:
            return
        self.publish_accumulator %= self.publish_period
        self._publish_clock(sim_time_ns)
        self._publish_drone_pose(sim_time_ns)
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
