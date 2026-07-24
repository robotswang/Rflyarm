#!/usr/bin/env python3
"""Reinstall the bulb already held by the running replacement demo."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parent
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
ROS_PYTHON = Path("/usr/bin/python3")
BOOTSTRAP_MARKER = "RFLYARM_INSTALL_BULB_ROS_BOOTSTRAPPED"
STARTUP_TIMEOUT_S = 300.0
COMMAND_TIMEOUT_S = 300.0
POSITION_TOLERANCE_M = 0.015
ORIENTATION_TOLERANCE_RAD = math.radians(5.0)
PLATFORM_PRECONTACT_TARGET = (0.053, -0.2, 7.93)
PLATFORM_TARGET = (0.053, -0.2, 7.95)
CEILING_CONTACT_MIN_Z_M = 7.940
HORIZONTAL_POSITION_TOLERANCE_M = 0.01
PLATFORM_SPEED_LIMIT_MPS = 0.01
PRECONTACT_HORIZONTAL_TOLERANCE_M = 0.01
PRECONTACT_Z_TOLERANCE_M = 0.01
MAX_INSTALL_DISTANCE_M = 0.05
MAX_INSTALL_BULB_SPEED_MPS = 0.005
FINAL_BULB_POSITION_TOLERANCE_M = 0.002
FINAL_BULB_ORIENTATION_TOLERANCE_RAD = math.radians(0.2)
INSTALLATION_TWIST_CYCLES = 4


def _bootstrap_ros() -> None:
    """Re-exec this file with the system ROS 2 Humble environment loaded."""

    if os.environ.get(BOOTSTRAP_MARKER) == "1":
        return
    if not ROS_SETUP.is_file():
        raise FileNotFoundError(f"ROS 2 Humble setup not found: {ROS_SETUP}")
    if not ROS_PYTHON.is_file():
        raise FileNotFoundError(f"System Python not found: {ROS_PYTHON}")

    environment = os.environ.copy()
    for name in (
        "PYTHONPATH",
        "ROS_DISTRO",
        "AMENT_PREFIX_PATH",
        "COLCON_PREFIX_PATH",
        "RMW_IMPLEMENTATION",
        "LD_LIBRARY_PATH",
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
    ):
        environment.pop(name, None)
    environment[BOOTSTRAP_MARKER] = "1"

    command = [
        "/usr/bin/bash",
        "-c",
        'set -e; source "$1"; shift; exec "$@"',
        "install_bulb.py",
        str(ROS_SETUP),
        str(ROS_PYTHON),
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    os.execve(command[0], command, environment)


_bootstrap_ros()

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, String

from simulation.ceiling_bulb import loose_endpoint_quaternion_xyzw


@dataclass(frozen=True)
class ArmPose:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


LOAD_POSE = ArmPose(
    position=(0.0, 0.2, 0.38),
    quaternion_xyzw=(0.1830, -0.1830, 0.6830, 0.6830),
)
CLEARANCE_POSE = ArmPose(
    position=(0.0, 0.2, 0.28),
    quaternion_xyzw=LOAD_POSE.quaternion_xyzw,
)
INSERT_POSE = ArmPose(
    position=(0.0, 0.2, 0.38),
    quaternion_xyzw=(-0.1830, 0.1830, 0.6830, 0.6830),
)
RETREAT_POSE = ArmPose(
    position=(0.0, 0.165, 0.31938),
    quaternion_xyzw=(-0.1830, 0.1830, 0.6830, 0.6830),
)


class BulbInstaller(Node):
    """ROS 2 client that installs the bulb already held by the gripper."""

    def __init__(self) -> None:
        super().__init__("rflyarm_install_bulb")
        self.sim_time_s: float | None = None
        self.arm_pose: PoseStamped | None = None
        self.drone_pose: PoseStamped | None = None
        self.drone_pose_sequence = 0
        self.drone_velocity: TwistStamped | None = None
        self.drone_velocity_sequence = 0
        self.bulb_distance_m: float | None = None
        self.bulb_distance_sequence = 0
        self.bulb_pose: PoseStamped | None = None
        self.bulb_pose_sequence = 0
        self.bulb_speed_mps: float | None = None
        self._previous_bulb_pose_sample: tuple[
            float, tuple[float, float, float]
        ] | None = None
        self.bulb_initial_pose: PoseStamped | None = None
        self.bulb_state: str | None = None
        self.bulb_state_sequence = 0
        self.arm_sequence = 0
        self.platform_locked: bool | None = None
        self.platform_lock_sequence = 0

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        clock_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.drone_publisher = self.create_publisher(
            PoseStamped, "/drone/cmd_pose", qos
        )
        self.joint_publisher = self.create_publisher(
            JointState, "/joint_command", qos
        )
        self.arm_publisher = self.create_publisher(
            PoseStamped, "/arm/cmd_pose", qos
        )
        self.bulb_publisher = self.create_publisher(
            String, "/ceiling_bulb/cmd", qos
        )
        self.platform_lock_publisher = self.create_publisher(
            Bool, "/drone/cmd_lock", qos
        )
        self.clock_subscription = self.create_subscription(
            Clock, "/clock", self._clock_callback, clock_qos
        )
        self.arm_subscription = self.create_subscription(
            PoseStamped, "/arm/ee_pose", self._arm_pose_callback, qos
        )
        self.drone_subscription = self.create_subscription(
            PoseStamped, "/drone/pose", self._drone_pose_callback, qos
        )
        self.drone_velocity_subscription = self.create_subscription(
            TwistStamped, "/drone/velocity", self._drone_velocity_callback, qos
        )
        self.bulb_state_subscription = self.create_subscription(
            String, "/ceiling_bulb/state", self._bulb_state_callback, qos
        )
        self.bulb_distance_subscription = self.create_subscription(
            Float64,
            "/ceiling_bulb/distance",
            self._bulb_distance_callback,
            qos,
        )
        self.bulb_pose_subscription = self.create_subscription(
            PoseStamped, "/ceiling_bulb/pose", self._bulb_pose_callback, qos
        )
        self.bulb_initial_pose_subscription = self.create_subscription(
            PoseStamped,
            "/ceiling_bulb/initial_pose",
            self._bulb_initial_pose_callback,
            qos,
        )
        self.platform_locked_subscription = self.create_subscription(
            Bool, "/drone/locked", self._platform_locked_callback, qos
        )

    def _clock_callback(self, message: Clock) -> None:
        self.sim_time_s = (
            float(message.clock.sec) + float(message.clock.nanosec) * 1.0e-9
        )

    def _arm_pose_callback(self, message: PoseStamped) -> None:
        self.arm_pose = message
        self.arm_sequence += 1

    def _drone_pose_callback(self, message: PoseStamped) -> None:
        self.drone_pose = message
        self.drone_pose_sequence += 1

    def _drone_velocity_callback(self, message: TwistStamped) -> None:
        self.drone_velocity = message
        self.drone_velocity_sequence += 1

    def _bulb_distance_callback(self, message: Float64) -> None:
        self.bulb_distance_m = float(message.data)
        self.bulb_distance_sequence += 1

    def _bulb_pose_callback(self, message: PoseStamped) -> None:
        self.bulb_pose = message
        sample_time_s = (
            float(message.header.stamp.sec)
            + float(message.header.stamp.nanosec) * 1.0e-9
        )
        position = (
            float(message.pose.position.x),
            float(message.pose.position.y),
            float(message.pose.position.z),
        )
        if self._previous_bulb_pose_sample is not None:
            previous_time_s, previous_position = self._previous_bulb_pose_sample
            elapsed_s = sample_time_s - previous_time_s
            if elapsed_s > 1.0e-9:
                self.bulb_speed_mps = math.sqrt(
                    sum(
                        (current - previous) ** 2
                        for current, previous in zip(position, previous_position)
                    )
                ) / elapsed_s
        self._previous_bulb_pose_sample = (sample_time_s, position)
        self.bulb_pose_sequence += 1

    def _bulb_initial_pose_callback(self, message: PoseStamped) -> None:
        self.bulb_initial_pose = message

    def _bulb_state_callback(self, message: String) -> None:
        self.bulb_state = str(message.data).strip().upper()
        self.bulb_state_sequence += 1

    def _platform_locked_callback(self, message: Bool) -> None:
        self.platform_locked = bool(message.data)
        self.platform_lock_sequence += 1

    def _spin_until(self, predicate, label: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok():
            if predicate():
                return
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise TimeoutError(f"Timed out waiting for {label}")
            rclpy.spin_once(self, timeout_sec=min(0.1, remaining))
        raise RuntimeError(f"ROS 2 stopped while waiting for {label}")

    def wait_until_ready(self) -> None:
        publishers = (
            self.drone_publisher,
            self.joint_publisher,
            self.arm_publisher,
            self.bulb_publisher,
            self.platform_lock_publisher,
        )

        def ready() -> bool:
            return (
                self.sim_time_s is not None
                and self.arm_pose is not None
                and self.drone_pose is not None
                and self.drone_velocity is not None
                and self.bulb_state is not None
                and self.bulb_distance_m is not None
                and self.bulb_pose is not None
                and self.bulb_initial_pose is not None
                and self.platform_locked is not None
                and all(publisher.get_subscription_count() > 0 for publisher in publishers)
            )

        print("[Rflyarm] waiting for the running replacement simulation")
        self._spin_until(ready, "running replacement simulation", STARTUP_TIMEOUT_S)

    def wait_for_sim_duration(self, duration_s: float) -> None:
        if self.sim_time_s is None:
            self._spin_until(
                lambda: self.sim_time_s is not None,
                "simulation clock",
                STARTUP_TIMEOUT_S,
            )
        target = float(self.sim_time_s) + duration_s
        self._spin_until(
            lambda: self.sim_time_s is not None and self.sim_time_s >= target,
            f"simulation duration {duration_s:.2f} s",
            COMMAND_TIMEOUT_S,
        )

    @staticmethod
    def _pose_errors(message: PoseStamped, target: ArmPose) -> tuple[float, float]:
        position = (
            float(message.pose.position.x),
            float(message.pose.position.y),
            float(message.pose.position.z),
        )
        position_error = math.sqrt(
            sum((current - desired) ** 2 for current, desired in zip(position, target.position))
        )
        quaternion = (
            float(message.pose.orientation.x),
            float(message.pose.orientation.y),
            float(message.pose.orientation.z),
            float(message.pose.orientation.w),
        )
        target_quaternion = target.quaternion_xyzw
        norm = math.sqrt(sum(value * value for value in quaternion))
        target_norm = math.sqrt(sum(value * value for value in target_quaternion))
        if norm < 1.0e-8 or target_norm < 1.0e-8:
            return position_error, math.inf
        dot = abs(
            sum(current * desired for current, desired in zip(quaternion, target_quaternion))
            / (norm * target_norm)
        )
        orientation_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
        return position_error, orientation_error

    def wait_for_arm_pose(self, target: ArmPose, label: str, after_sequence: int = 0) -> None:
        last_errors = (math.inf, math.inf)

        def reached() -> bool:
            nonlocal last_errors
            if self.arm_pose is None or self.arm_sequence <= after_sequence:
                return False
            last_errors = self._pose_errors(self.arm_pose, target)
            return (
                last_errors[0] <= POSITION_TOLERANCE_M
                and last_errors[1] <= ORIENTATION_TOLERANCE_RAD
            )

        print(f"[Rflyarm] waiting for arm pose: {label}")
        try:
            self._spin_until(reached, f"arm pose '{label}'", COMMAND_TIMEOUT_S)
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; last errors were {last_errors[0]:.4f} m and "
                f"{math.degrees(last_errors[1]):.2f} deg"
            ) from exc

    def _current_drone_position(self) -> tuple[float, float, float]:
        if self.drone_pose is None:
            raise RuntimeError("drone pose is not available")
        position = self.drone_pose.pose.position
        return float(position.x), float(position.y), float(position.z)

    def _drone_speed(self) -> float:
        if self.drone_velocity is None:
            return math.inf
        velocity = self.drone_velocity.twist.linear
        return math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )

    def wait_for_precontact_alignment(self) -> None:
        last_values = (math.inf, math.inf, math.inf, math.inf)

        def aligned() -> bool:
            nonlocal last_values
            x, y, z = self._current_drone_position()
            speed = self._drone_speed()
            x_error = abs(x - PLATFORM_PRECONTACT_TARGET[0])
            y_error = abs(y - PLATFORM_PRECONTACT_TARGET[1])
            z_error = abs(z - PLATFORM_PRECONTACT_TARGET[2])
            last_values = (x_error, y_error, z_error, speed)
            return (
                x_error < PRECONTACT_HORIZONTAL_TOLERANCE_M
                and y_error < PRECONTACT_HORIZONTAL_TOLERANCE_M
                and z_error < PRECONTACT_Z_TOLERANCE_M
                and speed <= PLATFORM_SPEED_LIMIT_MPS
            )

        print(
            "[Rflyarm] waiting at z=%.3f before ceiling contact: "
            "|dx| < %.3f m, |dy| < %.3f m, |dz| < %.3f m, "
            "speed <= %.3f m/s"
            % (
                PLATFORM_PRECONTACT_TARGET[2],
                PRECONTACT_HORIZONTAL_TOLERANCE_M,
                PRECONTACT_HORIZONTAL_TOLERANCE_M,
                PRECONTACT_Z_TOLERANCE_M,
                PLATFORM_SPEED_LIMIT_MPS,
            )
        )
        try:
            self._spin_until(
                aligned,
                "pre-contact platform alignment",
                COMMAND_TIMEOUT_S,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; |dx|={last_values[0]:.4f} m, "
                f"|dy|={last_values[1]:.4f} m, "
                f"|dz|={last_values[2]:.4f} m, "
                f"speed={last_values[3]:.4f} m/s"
            ) from exc

    def wait_for_platform_arrival(
        self,
        after_pose_sequence: int,
        after_velocity_sequence: int,
    ) -> None:
        last_values = (math.inf, math.inf, -math.inf, math.inf)

        def arrived() -> bool:
            nonlocal last_values
            if (
                self.drone_pose_sequence <= after_pose_sequence
                or self.drone_velocity_sequence <= after_velocity_sequence
            ):
                return False
            x, y, z = self._current_drone_position()
            speed = self._drone_speed()
            x_error = abs(x - PLATFORM_TARGET[0])
            y_error = abs(y - PLATFORM_TARGET[1])
            last_values = (x_error, y_error, z, speed)
            return (
                x_error < HORIZONTAL_POSITION_TOLERANCE_M
                and y_error < HORIZONTAL_POSITION_TOLERANCE_M
                and z >= CEILING_CONTACT_MIN_Z_M
                and speed <= PLATFORM_SPEED_LIMIT_MPS
            )

        print(
            "[Rflyarm] waiting for ceiling contact: "
            "|dx| < %.3f m, |dy| < %.3f m, z >= %.3f m, speed <= %.3f m/s"
            % (
                HORIZONTAL_POSITION_TOLERANCE_M,
                HORIZONTAL_POSITION_TOLERANCE_M,
                CEILING_CONTACT_MIN_Z_M,
                PLATFORM_SPEED_LIMIT_MPS,
            )
        )
        try:
            self._spin_until(arrived, "ceiling contact arrival", COMMAND_TIMEOUT_S)
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; |dx|={last_values[0]:.4f} m, "
                f"|dy|={last_values[1]:.4f} m, z={last_values[2]:.4f} m, "
                f"speed={last_values[3]:.4f} m/s"
            ) from exc

    def wait_for_install_alignment(
        self,
        after_distance_sequence: int = 0,
        after_pose_sequence: int = 0,
    ) -> None:
        last_distance = math.inf
        last_speed = math.inf

        def aligned() -> bool:
            nonlocal last_distance, last_speed
            last_distance = (
                math.inf if self.bulb_distance_m is None else self.bulb_distance_m
            )
            last_speed = (
                math.inf if self.bulb_speed_mps is None else self.bulb_speed_mps
            )
            return (
                self.bulb_distance_sequence > after_distance_sequence
                and self.bulb_pose_sequence > after_pose_sequence
                and last_distance <= MAX_INSTALL_DISTANCE_M
                and last_speed <= MAX_INSTALL_BULB_SPEED_MPS
            )

        try:
            self._spin_until(
                aligned,
                "slow close-range bulb alignment",
                COMMAND_TIMEOUT_S,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; last bulb/socket distance={last_distance:.4f} m, "
                f"bulb speed={last_speed:.4f} m/s"
            ) from exc

    @staticmethod
    def _message_pose_errors(
        current: PoseStamped,
        target: PoseStamped,
    ) -> tuple[float, float]:
        position_error = math.sqrt(
            (float(current.pose.position.x) - float(target.pose.position.x)) ** 2
            + (float(current.pose.position.y) - float(target.pose.position.y)) ** 2
            + (float(current.pose.position.z) - float(target.pose.position.z)) ** 2
        )
        current_quaternion = (
            float(current.pose.orientation.x),
            float(current.pose.orientation.y),
            float(current.pose.orientation.z),
            float(current.pose.orientation.w),
        )
        target_quaternion = (
            float(target.pose.orientation.x),
            float(target.pose.orientation.y),
            float(target.pose.orientation.z),
            float(target.pose.orientation.w),
        )
        current_norm = math.sqrt(sum(value * value for value in current_quaternion))
        target_norm = math.sqrt(sum(value * value for value in target_quaternion))
        if current_norm < 1.0e-8 or target_norm < 1.0e-8:
            return position_error, math.inf
        dot = abs(
            sum(
                current_value * target_value
                for current_value, target_value in zip(
                    current_quaternion,
                    target_quaternion,
                )
            )
            / (current_norm * target_norm)
        )
        orientation_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
        return position_error, orientation_error

    def wait_for_exact_pre_rotation_pose(self) -> None:
        last_errors = (math.inf, math.inf)

        def matched() -> bool:
            nonlocal last_errors
            if self.bulb_pose is None or self.bulb_initial_pose is None:
                return False
            target = PoseStamped()
            target.pose.position = self.bulb_initial_pose.pose.position
            initial_orientation = self.bulb_initial_pose.pose.orientation
            (
                target.pose.orientation.x,
                target.pose.orientation.y,
                target.pose.orientation.z,
                target.pose.orientation.w,
            ) = loose_endpoint_quaternion_xyzw(
                (
                    float(initial_orientation.x),
                    float(initial_orientation.y),
                    float(initial_orientation.z),
                    float(initial_orientation.w),
                )
            )
            last_errors = self._message_pose_errors(
                self.bulb_pose,
                target,
            )
            return (
                last_errors[0] <= FINAL_BULB_POSITION_TOLERANCE_M
                and last_errors[1] <= FINAL_BULB_ORIENTATION_TOLERANCE_RAD
            )

        try:
            self._spin_until(
                matched,
                "bulb aligned to its loose world pose before rotation",
                COMMAND_TIMEOUT_S,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; last errors={last_errors[0]:.4f} m and "
                f"{math.degrees(last_errors[1]):.3f} deg"
            ) from exc

    def wait_for_exact_installed_pose(self) -> None:
        last_errors = (math.inf, math.inf)

        def matched() -> bool:
            nonlocal last_errors
            if self.bulb_pose is None or self.bulb_initial_pose is None:
                return False
            last_errors = self._message_pose_errors(
                self.bulb_pose,
                self.bulb_initial_pose,
            )
            return (
                last_errors[0] <= FINAL_BULB_POSITION_TOLERANCE_M
                and last_errors[1] <= FINAL_BULB_ORIENTATION_TOLERANCE_RAD
            )

        try:
            self._spin_until(
                matched,
                "bulb restored to its original installed world pose",
                COMMAND_TIMEOUT_S,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; last errors={last_errors[0]:.4f} m and "
                f"{math.degrees(last_errors[1]):.3f} deg"
            ) from exc

    def wait_for_state(
        self,
        states: set[str],
        label: str,
        after_sequence: int = 0,
    ) -> None:
        self._spin_until(
            lambda: (
                self.bulb_state_sequence > after_sequence
                and self.bulb_state in states
            ),
            f"bulb state {label}",
            COMMAND_TIMEOUT_S,
        )

    def publish_drone_target(self, x: float, y: float, z: float) -> None:
        message = PoseStamped()
        message.header.frame_id = "map"
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.position.z = z
        message.pose.orientation.w = 1.0
        self.drone_publisher.publish(message)

    def publish_gripper(self, position: float) -> None:
        message = JointState()
        message.name = ["gripper"]
        message.position = [position]
        self.joint_publisher.publish(message)

    def publish_arm_pose(self, target: ArmPose) -> None:
        message = PoseStamped()
        message.header.frame_id = "base_link"
        (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ) = target.position
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = target.quaternion_xyzw
        previous_sequence = self.arm_sequence
        self.arm_publisher.publish(message)
        self.wait_for_arm_pose(target, "installation stroke", previous_sequence)

    def publish_bulb_command(self, command: str) -> int:
        message = String()
        message.data = command
        previous_sequence = self.bulb_state_sequence
        self.bulb_publisher.publish(message)
        return previous_sequence

    def publish_platform_lock(self, enabled: bool) -> int:
        message = Bool()
        message.data = bool(enabled)
        previous_sequence = self.platform_lock_sequence
        self.platform_lock_publisher.publish(message)
        return previous_sequence

    def wait_for_platform_lock(self, expected: bool, after_sequence: int) -> None:
        self._spin_until(
            lambda: (
                self.platform_lock_sequence > after_sequence
                and self.platform_locked is expected
            ),
            f"platform locked={expected}",
            COMMAND_TIMEOUT_S,
        )

    def run_installation(self) -> None:
        self.wait_until_ready()
        if self.bulb_state != "UNLOCKED":
            raise RuntimeError(
                f"expected the replacement demo to leave the bulb UNLOCKED, got {self.bulb_state}"
            )
        if self.platform_locked:
            unlock_sequence = self.publish_platform_lock(False)
            self.wait_for_platform_lock(False, unlock_sequence)

        # Lower the already gripper-held bulb before the platform approaches
        # the ceiling, so the flight body—not the bulb—makes ceiling contact.
        self.publish_arm_pose(CLEARANCE_POSE)

        self.publish_drone_target(*PLATFORM_PRECONTACT_TARGET)
        print(
            "[Rflyarm] sent direct pre-contact platform command %s"
            % (PLATFORM_PRECONTACT_TARGET,)
        )
        self.wait_for_precontact_alignment()
        pose_sequence = self.drone_pose_sequence
        velocity_sequence = self.drone_velocity_sequence
        self.publish_drone_target(*PLATFORM_TARGET)
        print(
            "[Rflyarm] sent single final-contact platform command %s"
            % (PLATFORM_TARGET,)
        )
        self.wait_for_platform_arrival(pose_sequence, velocity_sequence)

        lock_sequence = self.publish_platform_lock(True)
        self.wait_for_platform_lock(True, lock_sequence)
        print(
            "[Rflyarm] platform hard-locked at actual flight-body position %s; "
            "flight control and rotors remain active"
            % (self._current_drone_position(),)
        )

        # With the flight body fixed against the ceiling, raise the held bulb
        # back to its installation pose before checking alignment conditions.
        self.publish_arm_pose(LOAD_POSE)

        distance_sequence = self.bulb_distance_sequence
        pose_sequence = self.bulb_pose_sequence
        self.wait_for_install_alignment(distance_sequence, pose_sequence)

        state_sequence = self.publish_bulb_command("engage_loose")
        self.wait_for_state(
            {"SCREWING"},
            "SCREWING after engage_loose",
            state_sequence,
        )
        self.wait_for_exact_pre_rotation_pose()

        for cycle in range(1, INSTALLATION_TWIST_CYCLES + 1):
            print(f"[Rflyarm] installation stroke {cycle}/{INSTALLATION_TWIST_CYCLES}")
            self.publish_gripper(0.25)
            self.wait_for_sim_duration(0.5)
            state_sequence = self.publish_bulb_command("resume_grasp")
            self.wait_for_state(
                {"SCREWING"},
                "SCREWING after resume_grasp",
                state_sequence,
            )
            self.publish_arm_pose(INSERT_POSE)
            final_cycle = cycle == INSTALLATION_TWIST_CYCLES
            command = "finish_tightening" if final_cycle else "hold_tightening_stroke"
            state_sequence = self.publish_bulb_command(command)
            self.wait_for_state(
                {"SCREWING", "LOCKED"},
                "SCREWING or LOCKED after actual-angle check",
                state_sequence,
            )

            if not final_cycle:
                if self.bulb_state != "SCREWING":
                    raise RuntimeError(
                        "bulb locked before all four installation strokes completed"
                    )
                self.publish_gripper(1.0)
                self.publish_arm_pose(LOAD_POSE)

        if self.bulb_state != "LOCKED":
            raise RuntimeError(
                f"installation ended before LOCKED; current state is {self.bulb_state}"
            )

        self.wait_for_exact_installed_pose()
        self.publish_gripper(1.0)
        self.publish_arm_pose(RETREAT_POSE)
        unlock_sequence = self.publish_platform_lock(False)
        self.wait_for_platform_lock(False, unlock_sequence)
        self.publish_drone_target(0.0, 0.0, 0.0)
        self.wait_for_sim_duration(1.0)
        print("[Rflyarm] bulb installation complete; existing simulation remains running")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    return parser.parse_args()


def main() -> int:
    parse_args()
    try:
        rclpy.init(args=None)
        node = BulbInstaller()
        try:
            node.run_installation()
        finally:
            node.destroy_node()
        return 0
    except KeyboardInterrupt:
        print("[Rflyarm] bulb installation interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[Rflyarm] bulb installation failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
