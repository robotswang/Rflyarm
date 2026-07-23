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
POSITION_TOLERANCE_M = 0.005
ORIENTATION_TOLERANCE_RAD = math.radians(2.0)
CEILING_TARGET = (0.053, -0.2, 7.9)
TRANSIT_TARGET = (CEILING_TARGET[0], CEILING_TARGET[1], 7.2)
TRANSIT_POSITION_TOLERANCE_M = 0.10
TRANSIT_SPEED_LIMIT_MPS = 0.20
APPROACH_STEP_M = 0.05
APPROACH_POSITION_TOLERANCE_M = 0.025
APPROACH_SPEED_LIMIT_MPS = 0.08
FINAL_SPEED_LIMIT_MPS = 0.03
MAX_INSTALL_DISTANCE_M = 0.05
FINAL_SETTLE_DURATION_S = 1.0
FINAL_BULB_POSITION_TOLERANCE_M = 0.002
FINAL_BULB_ORIENTATION_TOLERANCE_RAD = math.radians(0.2)


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
from std_msgs.msg import Float64, String


@dataclass(frozen=True)
class ArmPose:
    position: tuple[float, float, float]
    quaternion_xyzw: tuple[float, float, float, float]


LOAD_POSE = ArmPose(
    position=(0.0, 0.2, 0.38),
    quaternion_xyzw=(0.1830, -0.1830, 0.6830, 0.6830),
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

    def __init__(self, rotation_count: int) -> None:
        super().__init__("rflyarm_install_bulb")
        self.rotation_count = rotation_count
        self.sim_time_s: float | None = None
        self.arm_pose: PoseStamped | None = None
        self.drone_pose: PoseStamped | None = None
        self.drone_velocity: TwistStamped | None = None
        self.bulb_distance_m: float | None = None
        self.bulb_pose: PoseStamped | None = None
        self.bulb_initial_pose: PoseStamped | None = None
        self.bulb_state: str | None = None
        self.bulb_state_sequence = 0
        self.arm_sequence = 0

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

    def _clock_callback(self, message: Clock) -> None:
        self.sim_time_s = (
            float(message.clock.sec) + float(message.clock.nanosec) * 1.0e-9
        )

    def _arm_pose_callback(self, message: PoseStamped) -> None:
        self.arm_pose = message
        self.arm_sequence += 1

    def _drone_pose_callback(self, message: PoseStamped) -> None:
        self.drone_pose = message

    def _drone_velocity_callback(self, message: TwistStamped) -> None:
        self.drone_velocity = message

    def _bulb_distance_callback(self, message: Float64) -> None:
        self.bulb_distance_m = float(message.data)

    def _bulb_pose_callback(self, message: PoseStamped) -> None:
        self.bulb_pose = message

    def _bulb_initial_pose_callback(self, message: PoseStamped) -> None:
        self.bulb_initial_pose = message

    def _bulb_state_callback(self, message: String) -> None:
        self.bulb_state = str(message.data).strip().upper()
        self.bulb_state_sequence += 1

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

    def _drone_position_error(self, target: tuple[float, float, float]) -> float:
        if self.drone_pose is None:
            return math.inf
        current = self.drone_pose.pose.position
        return math.sqrt(
            (float(current.x) - target[0]) ** 2
            + (float(current.y) - target[1]) ** 2
            + (float(current.z) - target[2]) ** 2
        )

    def _drone_speed(self) -> float:
        if self.drone_velocity is None:
            return math.inf
        velocity = self.drone_velocity.twist.linear
        return math.sqrt(
            float(velocity.x) ** 2
            + float(velocity.y) ** 2
            + float(velocity.z) ** 2
        )

    def wait_for_drone_settled(
        self,
        target: tuple[float, float, float],
        position_tolerance_m: float,
        speed_limit_mps: float,
        label: str,
    ) -> None:
        last_position_error = math.inf
        last_speed = math.inf

        def reached() -> bool:
            nonlocal last_position_error, last_speed
            last_position_error = self._drone_position_error(target)
            last_speed = self._drone_speed()
            return (
                last_position_error <= position_tolerance_m
                and last_speed <= speed_limit_mps
            )

        print(
            f"[Rflyarm] waiting for {label}: position <= "
            f"{position_tolerance_m:.3f} m, speed <= {speed_limit_mps:.3f} m/s"
        )
        try:
            self._spin_until(reached, label, COMMAND_TIMEOUT_S)
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; last position error={last_position_error:.4f} m, "
                f"speed={last_speed:.4f} m/s"
            ) from exc

    def wait_for_install_alignment(self) -> None:
        last_distance = math.inf
        last_speed = math.inf

        def aligned() -> bool:
            nonlocal last_distance, last_speed
            last_distance = (
                math.inf if self.bulb_distance_m is None else self.bulb_distance_m
            )
            last_speed = self._drone_speed()
            return (
                last_distance <= MAX_INSTALL_DISTANCE_M
                and last_speed <= FINAL_SPEED_LIMIT_MPS
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
                f"speed={last_speed:.4f} m/s"
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
                "bulb restored to its original world pose",
                COMMAND_TIMEOUT_S,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; last errors={last_errors[0]:.4f} m and "
                f"{math.degrees(last_errors[1]):.3f} deg"
            ) from exc

    def approach_ceiling_slowly(self) -> None:
        self.publish_drone_target(*TRANSIT_TARGET)
        self.wait_for_drone_settled(
            TRANSIT_TARGET,
            TRANSIT_POSITION_TOLERANCE_M,
            TRANSIT_SPEED_LIMIT_MPS,
            "safe transit point below the socket",
        )

        waypoint_z = TRANSIT_TARGET[2] + APPROACH_STEP_M
        while waypoint_z < CEILING_TARGET[2] - 1.0e-9:
            target = (CEILING_TARGET[0], CEILING_TARGET[1], waypoint_z)
            self.publish_drone_target(*target)
            self.wait_for_drone_settled(
                target,
                APPROACH_POSITION_TOLERANCE_M,
                APPROACH_SPEED_LIMIT_MPS,
                f"slow approach waypoint z={waypoint_z:.2f} m",
            )
            waypoint_z += APPROACH_STEP_M

        self.publish_drone_target(*CEILING_TARGET)
        self.wait_for_drone_settled(
            CEILING_TARGET,
            APPROACH_POSITION_TOLERANCE_M,
            FINAL_SPEED_LIMIT_MPS,
            "final socket approach",
        )
        self.wait_for_install_alignment()
        self.wait_for_sim_duration(FINAL_SETTLE_DURATION_S)
        self.wait_for_drone_settled(
            CEILING_TARGET,
            APPROACH_POSITION_TOLERANCE_M,
            FINAL_SPEED_LIMIT_MPS,
            "stable final socket approach",
        )
        self.wait_for_install_alignment()

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

    def run_installation(self) -> None:
        self.wait_until_ready()
        if self.bulb_state != "UNLOCKED":
            raise RuntimeError(
                f"expected the replacement demo to leave the bulb UNLOCKED, got {self.bulb_state}"
            )

        # The bulb is already fixed in the gripper. Do not send an arm motion
        # before this check and never issue a load/prepare command.
        self.wait_for_arm_pose(LOAD_POSE, "safe gripper-held pose")

        self.approach_ceiling_slowly()
        self.wait_for_arm_pose(LOAD_POSE, "safe pose at the ceiling")

        state_sequence = self.publish_bulb_command("engage_loose")
        self.wait_for_state(
            {"SCREWING"},
            "SCREWING after engage_loose",
            state_sequence,
        )

        step_degrees = 180.0 / self.rotation_count
        for cycle in range(1, self.rotation_count + 1):
            print(f"[Rflyarm] installation stroke {cycle}/{self.rotation_count}")
            self.publish_gripper(0.25)
            self.wait_for_sim_duration(0.5)
            if cycle > 1:
                state_sequence = self.publish_bulb_command("resume_grasp")
                self.wait_for_state(
                    {"SCREWING"},
                    "SCREWING after resume_grasp",
                    state_sequence,
                )
            self.publish_arm_pose(INSERT_POSE)
            state_sequence = self.publish_bulb_command(
                f"tighten_{step_degrees:.12g}"
            )
            self.wait_for_state(
                {"SCREWING", "LOCKED"},
                "SCREWING or LOCKED after tightening",
                state_sequence,
            )
            if self.bulb_state == "LOCKED":
                break

            self.publish_gripper(1.0)
            self.publish_arm_pose(LOAD_POSE)

        if self.bulb_state != "LOCKED":
            raise RuntimeError(
                f"installation ended before LOCKED; current state is {self.bulb_state}"
            )

        self.wait_for_exact_installed_pose()

        self.publish_gripper(1.0)
        self.publish_arm_pose(RETREAT_POSE)
        self.publish_drone_target(0.0, 0.0, 0.0)
        self.wait_for_sim_duration(1.0)
        print("[Rflyarm] bulb installation complete; existing simulation remains running")


def positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("rotation-count must be a positive integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rotation_count",
        nargs="?",
        type=positive_integer,
        default=3,
        help="maximum tightening strokes (default: 3)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rclpy.init(args=None)
        node = BulbInstaller(args.rotation_count)
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
