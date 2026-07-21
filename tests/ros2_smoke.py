#!/usr/bin/env python3
"""End-to-end ROS 2 smoke test for a running Rflyarm simulation."""

from __future__ import annotations

import argparse
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState


ARM_NAMES = [f"joint_{index}" for index in range(1, 7)]
ARM_TARGET = [0.10, -0.35, 0.30, 0.05, 0.20, -0.10]
FLIGHT_TARGET = (0.20, -0.10, 1.70)
EE_TARGET_POSITION = (0.01443163, -0.43654181, 0.37850003)
EE_TARGET_QUATERNION = (0.68457114, 0.01167725, 0.01096253, 0.72877007)


class RflyarmSmokeTest(Node):
    """Publish both command types and validate all public state streams."""

    def __init__(self) -> None:
        super().__init__("rflyarm_ros2_smoke_test")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.flight_publisher = self.create_publisher(PoseStamped, "/drone/cmd_pose", qos)
        self.arm_publisher = self.create_publisher(JointState, "/joint_command", qos)
        self.ee_publisher = self.create_publisher(PoseStamped, "/arm/cmd_pose", qos)
        self.create_subscription(PoseStamped, "/drone/pose", self._drone_callback, qos)
        self.create_subscription(JointState, "/joint_states", self._joint_callback, qos)
        self.create_subscription(PoseStamped, "/arm/ee_pose", self._ee_callback, qos)
        self.create_subscription(Clock, "/clock", self._clock_callback, qos)
        self.latest_drone = None
        self.latest_drone_stamp_ns = None
        self.latest_joints = None
        self.latest_joint_stamp_ns = None
        self.latest_ee = None
        self.latest_ee_stamp_ns = None
        self.latest_clock_ns = None
        self.first_clock_ns = None
        self.commands_started = False
        self.joint_phase_passed = False
        self.joint_phase_error = math.inf

    def _drone_callback(self, message: PoseStamped) -> None:
        self.latest_drone_stamp_ns = self._stamp_ns(message.header.stamp)
        self.latest_drone = (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        )

    def _joint_callback(self, message: JointState) -> None:
        positions = dict(zip(message.name, message.position))
        if all(name in positions for name in ARM_NAMES):
            self.latest_joint_stamp_ns = self._stamp_ns(message.header.stamp)
            self.latest_joints = [positions[name] for name in ARM_NAMES]

    def _ee_callback(self, message: PoseStamped) -> None:
        self.latest_ee_stamp_ns = self._stamp_ns(message.header.stamp)
        self.latest_ee = (
            (
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
            ),
            (
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
                message.pose.orientation.w,
            ),
        )

    @staticmethod
    def _stamp_ns(stamp) -> int:
        return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)

    def _clock_callback(self, message: Clock) -> None:
        self.latest_clock_ns = self._stamp_ns(message.clock)
        if self.first_clock_ns is None:
            self.first_clock_ns = self.latest_clock_ns

    @property
    def discovered(self) -> bool:
        return (
            self.flight_publisher.get_subscription_count() > 0
            and self.arm_publisher.get_subscription_count() > 0
            and self.ee_publisher.get_subscription_count() > 0
        )

    @property
    def received_all_states(self) -> bool:
        return (
            self.latest_drone is not None
            and self.latest_joints is not None
            and self.latest_ee is not None
            and self.latest_clock_ns is not None
        )

    @property
    def simulation_time_valid(self) -> bool:
        stamps = (
            self.latest_drone_stamp_ns,
            self.latest_joint_stamp_ns,
            self.latest_ee_stamp_ns,
            self.latest_clock_ns,
        )
        return (
            all(stamp is not None for stamp in stamps)
            and self.latest_clock_ns > self.first_clock_ns
            and max(stamps) - min(stamps) <= 50_000_000
        )

    def publish_commands(self) -> None:
        flight = PoseStamped()
        flight.header.stamp.sec = self.latest_clock_ns // 1_000_000_000
        flight.header.stamp.nanosec = self.latest_clock_ns % 1_000_000_000
        flight.header.frame_id = "map"
        flight.pose.position.x, flight.pose.position.y, flight.pose.position.z = FLIGHT_TARGET
        flight.pose.orientation.w = 1.0
        self.flight_publisher.publish(flight)

        if not self.joint_phase_passed:
            arm = JointState()
            arm.header.stamp = flight.header.stamp
            arm.name = ARM_NAMES
            arm.position = ARM_TARGET
            self.arm_publisher.publish(arm)
        else:
            ee = PoseStamped()
            ee.header.stamp = flight.header.stamp
            ee.header.frame_id = "base_link"
            ee.pose.position.x, ee.pose.position.y, ee.pose.position.z = EE_TARGET_POSITION
            (
                ee.pose.orientation.x,
                ee.pose.orientation.y,
                ee.pose.orientation.z,
                ee.pose.orientation.w,
            ) = EE_TARGET_QUATERNION
            self.ee_publisher.publish(ee)
        self.commands_started = True

    def flight_error(self) -> float:
        return math.sqrt(sum((value - target) ** 2 for value, target in zip(self.latest_drone, FLIGHT_TARGET)))

    def arm_error(self) -> float:
        return max(abs(value - target) for value, target in zip(self.latest_joints, ARM_TARGET))

    def ee_errors(self) -> tuple[float, float]:
        position, quaternion = self.latest_ee
        position_error = math.sqrt(
            sum((value - target) ** 2 for value, target in zip(position, EE_TARGET_POSITION))
        )
        dot = abs(sum(value * target for value, target in zip(quaternion, EE_TARGET_QUATERNION)))
        quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
        target_norm = math.sqrt(sum(value * value for value in EE_TARGET_QUATERNION))
        dot = max(-1.0, min(1.0, dot / (quaternion_norm * target_norm)))
        orientation_error = 2.0 * math.acos(dot)
        return position_error, orientation_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=45.0)
    args = parser.parse_args()

    rclpy.init()
    node = RflyarmSmokeTest()
    deadline = time.monotonic() + args.timeout
    next_publish = 0.0
    ee_position_error = math.inf
    ee_orientation_error = math.inf
    try:
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.05)
            now = time.monotonic()
            if node.discovered and node.received_all_states and now >= next_publish:
                node.publish_commands()
                next_publish = now + 0.10
            if (
                node.commands_started
                and not node.joint_phase_passed
                and node.flight_error() < 0.20
                and node.arm_error() < 0.03
            ):
                node.joint_phase_error = node.arm_error()
                node.joint_phase_passed = True
                node.get_logger().info("Joint command phase passed; testing /arm/cmd_pose")
            if node.joint_phase_passed:
                ee_position_error, ee_orientation_error = node.ee_errors()
            else:
                ee_position_error, ee_orientation_error = math.inf, math.inf
            if (
                ee_position_error < 0.01
                and ee_orientation_error < math.radians(2.0)
                and node.simulation_time_valid
            ):
                print(
                    "[VERIFY][ROS2] PASS "
                    f"flight_error={node.flight_error():.4f} m "
                    f"joint_error={node.joint_phase_error:.6f} rad "
                    f"ee_position_error={ee_position_error:.6f} m "
                    f"ee_orientation_error={math.degrees(ee_orientation_error):.4f} deg "
                    f"sim_time={node.latest_clock_ns * 1.0e-9:.3f} s"
                )
                return 0
        missing = []
        if not node.discovered:
            missing.append("command subscribers")
        if node.latest_drone is None:
            missing.append("/drone/pose")
        if node.latest_joints is None:
            missing.append("/joint_states")
        if node.latest_ee is None:
            missing.append("/arm/ee_pose")
        if node.latest_clock_ns is None:
            missing.append("/clock")
        detail = ", ".join(missing) or (
            f"flight_error={node.flight_error():.4f} m, joint_error={node.joint_phase_error:.6f} rad, "
            f"ee_position_error={ee_position_error:.6f} m, "
            f"ee_orientation_error={math.degrees(ee_orientation_error):.4f} deg"
        )
        print(f"[VERIFY][ROS2] FAIL timeout: {detail}")
        return 1
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
