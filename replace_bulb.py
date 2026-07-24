#!/usr/bin/env python3
"""Run the automated Rflyarm ceiling-bulb replacement demonstration."""

from __future__ import annotations

import argparse
from contextlib import suppress
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PROJECT_DIR = Path(__file__).resolve().parent
ROS_SETUP = Path("/opt/ros/humble/setup.bash")
ROS_PYTHON = Path("/usr/bin/python3")
BOOTSTRAP_MARKER = "RFLYARM_REPLACE_BULB_ROS_BOOTSTRAPPED"
STARTUP_TIMEOUT_S = 300.0
COMMAND_TIMEOUT_S = 300.0


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
        "replace_bulb.py",
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
from std_msgs.msg import Bool, String


INITIAL_ARM_JOINTS = (
    -3.074416,
    -2.131500,
    2.052645,
    -0.185639,
    -0.406412,
    1.532976,
)
INSERT_POSITION = (0.0, 0.2, 0.38)
INSERT_QUATERNION_XYZW = (-0.1830, 0.1830, 0.6830, 0.6830)
TWIST_QUATERNION_XYZW = (0.1830, -0.1830, 0.6830, 0.6830)
PLATFORM_PRECONTACT_TARGET = (0.053, -0.2, 7.93)
PLATFORM_TARGET = (0.053, -0.2, 7.95)
CEILING_CONTACT_MIN_Z_M = 7.940
HORIZONTAL_POSITION_TOLERANCE_M = 0.01
PLATFORM_SPEED_LIMIT_MPS = 0.01
PRECONTACT_HORIZONTAL_TOLERANCE_M = 0.01
PRECONTACT_Z_TOLERANCE_M = 0.01
INITIAL_ARM_JOINT_TOLERANCE_RAD = 0.03
ARM_POSITION_TOLERANCE_M = 0.015
ARM_ORIENTATION_TOLERANCE_DEG = 5.0
ARM_JOINT_SPEED_LIMIT_RAD_S = 0.05
GRIPPER_SPEED_LIMIT_RAD_S = 0.05
REMOVAL_TWIST_CYCLES = 4


class BulbReplacementDemo(Node):
    """Publish the original bulb-replacement sequence using simulation time."""

    def __init__(self, simulator: subprocess.Popen) -> None:
        super().__init__("rflyarm_replace_bulb")
        self.simulator = simulator
        self.sim_time_s: float | None = None
        self.bulb_state: str | None = None
        self.bulb_state_sequence = 0
        self.drone_pose: PoseStamped | None = None
        self.drone_pose_sequence = 0
        self.drone_velocity: TwistStamped | None = None
        self.drone_velocity_sequence = 0
        self.joint_positions: dict[str, float] = {}
        self.joint_velocities: dict[str, float] = {}
        self.ee_pose: PoseStamped | None = None
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
        self.bulb_state_subscription = self.create_subscription(
            String, "/ceiling_bulb/state", self._bulb_state_callback, qos
        )
        self.drone_pose_subscription = self.create_subscription(
            PoseStamped, "/drone/pose", self._drone_pose_callback, qos
        )
        self.drone_velocity_subscription = self.create_subscription(
            TwistStamped, "/drone/velocity", self._drone_velocity_callback, qos
        )
        self.joint_state_subscription = self.create_subscription(
            JointState, "/joint_states", self._joint_state_callback, qos
        )
        self.ee_pose_subscription = self.create_subscription(
            PoseStamped, "/arm/ee_pose", self._ee_pose_callback, qos
        )
        self.platform_locked_subscription = self.create_subscription(
            Bool, "/drone/locked", self._platform_locked_callback, qos
        )

    def _clock_callback(self, message: Clock) -> None:
        self.sim_time_s = (
            float(message.clock.sec) + float(message.clock.nanosec) * 1.0e-9
        )

    def _bulb_state_callback(self, message: String) -> None:
        self.bulb_state = str(message.data).strip().upper()
        self.bulb_state_sequence += 1

    def _drone_pose_callback(self, message: PoseStamped) -> None:
        self.drone_pose = message
        self.drone_pose_sequence += 1

    def _drone_velocity_callback(self, message: TwistStamped) -> None:
        self.drone_velocity = message
        self.drone_velocity_sequence += 1

    def _joint_state_callback(self, message: JointState) -> None:
        self.joint_positions = {
            str(name): float(position)
            for name, position in zip(message.name, message.position)
        }
        self.joint_velocities = {
            str(name): float(velocity)
            for name, velocity in zip(message.name, message.velocity)
        }

    def _ee_pose_callback(self, message: PoseStamped) -> None:
        self.ee_pose = message

    def _platform_locked_callback(self, message: Bool) -> None:
        self.platform_locked = bool(message.data)
        self.platform_lock_sequence += 1

    def _check_simulator(self) -> None:
        return_code = self.simulator.poll()
        if return_code is not None:
            raise RuntimeError(
                f"Isaac Sim exited before the command sequence completed: {return_code}"
            )

    def _spin_until(self, predicate, label: str, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while rclpy.ok():
            self._check_simulator()
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
                and self.bulb_state is not None
                and self.drone_pose is not None
                and self.drone_velocity is not None
                and bool(self.joint_positions)
                and bool(self.joint_velocities)
                and self.ee_pose is not None
                and self.platform_locked is not None
                and all(
                    publisher.get_subscription_count() > 0
                    for publisher in publishers
                )
            )

        print("[Rflyarm] waiting for Isaac Sim ROS 2 interfaces")
        self._spin_until(ready, "Isaac Sim ROS 2 interfaces", STARTUP_TIMEOUT_S)

    def wait_for_sim_time(self, target_s: float) -> None:
        self._spin_until(
            lambda: self.sim_time_s is not None and self.sim_time_s >= target_s,
            f"simulation time {target_s:.1f} s",
            COMMAND_TIMEOUT_S,
        )

    def wait_for_sim_duration(self, duration_s: float) -> None:
        if self.sim_time_s is None:
            self._spin_until(
                lambda: self.sim_time_s is not None,
                "simulation clock",
                STARTUP_TIMEOUT_S,
            )
        target_s = float(self.sim_time_s) + float(duration_s)
        self.wait_for_sim_time(target_s)

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

    def wait_for_initial_arm_joints(self) -> None:
        names = [f"joint_{index}" for index in range(1, 7)]
        last_error = math.inf

        def reached() -> bool:
            nonlocal last_error
            if any(name not in self.joint_positions for name in names):
                return False
            last_error = max(
                abs(self.joint_positions[name] - desired)
                for name, desired in zip(names, INITIAL_ARM_JOINTS)
            )
            return last_error <= INITIAL_ARM_JOINT_TOLERANCE_RAD

        try:
            self._spin_until(
                reached,
                "initial arm joint configuration",
                COMMAND_TIMEOUT_S,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; max joint error={last_error:.4f} rad"
            ) from exc

    @staticmethod
    def _quaternion_error_deg(
        actual: tuple[float, float, float, float],
        target: tuple[float, float, float, float],
    ) -> float:
        actual_norm = math.sqrt(sum(value * value for value in actual))
        target_norm = math.sqrt(sum(value * value for value in target))
        if actual_norm < 1.0e-8 or target_norm < 1.0e-8:
            return math.inf
        dot = abs(
            sum(a * b for a, b in zip(actual, target))
            / (actual_norm * target_norm)
        )
        return math.degrees(2.0 * math.acos(min(1.0, max(0.0, dot))))

    def _max_arm_joint_speed(self) -> float:
        names = [f"joint_{index}" for index in range(1, 7)]
        if any(name not in self.joint_velocities for name in names):
            return math.inf
        return max(abs(self.joint_velocities[name]) for name in names)

    def wait_for_arm_pose(
        self,
        position: tuple[float, float, float],
        quaternion_xyzw: tuple[float, float, float, float],
        position_tolerance_m: float = ARM_POSITION_TOLERANCE_M,
    ) -> None:
        last_values = (math.inf, math.inf, math.inf)

        def reached() -> bool:
            nonlocal last_values
            if self.ee_pose is None:
                return False
            actual_position = self.ee_pose.pose.position
            position_error = math.sqrt(
                (float(actual_position.x) - position[0]) ** 2
                + (float(actual_position.y) - position[1]) ** 2
                + (float(actual_position.z) - position[2]) ** 2
            )
            actual_orientation = self.ee_pose.pose.orientation
            orientation_error = self._quaternion_error_deg(
                (
                    float(actual_orientation.x),
                    float(actual_orientation.y),
                    float(actual_orientation.z),
                    float(actual_orientation.w),
                ),
                quaternion_xyzw,
            )
            joint_speed = self._max_arm_joint_speed()
            last_values = (position_error, orientation_error, joint_speed)
            return (
                position_error <= position_tolerance_m
                and orientation_error <= ARM_ORIENTATION_TOLERANCE_DEG
                and joint_speed <= ARM_JOINT_SPEED_LIMIT_RAD_S
            )

        try:
            self._spin_until(reached, "arm pose", COMMAND_TIMEOUT_S)
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; position_error={last_values[0]:.4f} m, "
                f"orientation_error={last_values[1]:.2f} deg, "
                f"max_joint_speed={last_values[2]:.4f} rad/s"
            ) from exc

    def wait_for_arm_rotation(
        self,
        quaternion_xyzw: tuple[float, float, float, float],
    ) -> None:
        last_orientation_error = math.inf

        def reached() -> bool:
            nonlocal last_orientation_error
            if self.ee_pose is None:
                return False
            orientation = self.ee_pose.pose.orientation
            last_orientation_error = self._quaternion_error_deg(
                (
                    float(orientation.x),
                    float(orientation.y),
                    float(orientation.z),
                    float(orientation.w),
                ),
                quaternion_xyzw,
            )
            return last_orientation_error <= ARM_ORIENTATION_TOLERANCE_DEG

        try:
            self._spin_until(
                reached,
                "final arm rotation",
                COMMAND_TIMEOUT_S,
            )
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; orientation_error={last_orientation_error:.2f} deg"
            ) from exc

    def wait_for_gripper(self, target: float) -> None:
        last_values = (math.inf, math.inf)

        def reached() -> bool:
            nonlocal last_values
            if (
                "gripper" not in self.joint_positions
                or "gripper" not in self.joint_velocities
            ):
                return False
            position = self.joint_positions["gripper"]
            speed = abs(self.joint_velocities["gripper"])
            last_values = (position, speed)
            if target < 0.5:
                position_reached = position <= target + 0.15
            else:
                position_reached = position >= target - 0.05
            return position_reached and speed <= GRIPPER_SPEED_LIMIT_RAD_S

        try:
            self._spin_until(reached, "gripper motion", COMMAND_TIMEOUT_S)
        except TimeoutError as exc:
            raise TimeoutError(
                f"{exc}; gripper={last_values[0]:.3f}, "
                f"speed={last_values[1]:.4f} rad/s"
            ) from exc

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

    def publish_drone_target(self, x: float, y: float, z: float) -> None:
        message = PoseStamped()
        message.header.frame_id = "map"
        message.pose.position.x = x
        message.pose.position.y = y
        message.pose.position.z = z
        message.pose.orientation.w = 1.0
        self.drone_publisher.publish(message)

    def publish_arm_joints(self, positions: tuple[float, ...]) -> None:
        message = JointState()
        message.name = [f"joint_{index}" for index in range(1, 7)]
        message.position = list(positions)
        self.joint_publisher.publish(message)

    def publish_gripper(self, position: float) -> None:
        message = JointState()
        message.name = ["gripper"]
        message.position = [position]
        self.joint_publisher.publish(message)

    def publish_bulb_command(self, command: str) -> int:
        message = String()
        message.data = command
        previous_sequence = self.bulb_state_sequence
        self.bulb_publisher.publish(message)
        return previous_sequence

    def wait_for_bulb_state(self, state: str, after_sequence: int) -> None:
        self._spin_until(
            lambda: (
                self.bulb_state_sequence > after_sequence
                and self.bulb_state == state
            ),
            f"ceiling bulb state {state}",
            COMMAND_TIMEOUT_S,
        )

    def publish_arm_pose(
        self,
        position: tuple[float, float, float],
        quaternion_xyzw: tuple[float, float, float, float],
    ) -> None:
        message = PoseStamped()
        message.header.frame_id = "base_link"
        (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        ) = position
        (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        ) = quaternion_xyzw
        self.arm_publisher.publish(message)

    def run_sequence(self) -> None:
        self.wait_until_ready()
        if self.bulb_state != "LOCKED":
            raise RuntimeError(
                f"expected ceiling bulb state LOCKED at startup, got {self.bulb_state}"
            )
        if self.platform_locked:
            sequence = self.publish_platform_lock(False)
            self.wait_for_platform_lock(False, sequence)

        self.publish_arm_joints(INITIAL_ARM_JOINTS)
        self.publish_drone_target(*PLATFORM_PRECONTACT_TARGET)
        print(
            "[Rflyarm] sent direct pre-contact platform command %s"
            % (PLATFORM_PRECONTACT_TARGET,)
        )
        self.wait_for_initial_arm_joints()
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

        self.wait_for_sim_duration(1.0)
        pregrasp_position = (0.0, 0.165, 0.31938)
        self.publish_arm_pose(pregrasp_position, INSERT_QUATERNION_XYZW)
        self.wait_for_arm_pose(pregrasp_position, INSERT_QUATERNION_XYZW)

        self.publish_arm_pose(INSERT_POSITION, INSERT_QUATERNION_XYZW)
        self.wait_for_arm_pose(INSERT_POSITION, INSERT_QUATERNION_XYZW)

        # Release and reset after every wrist stroke except the last one, which
        # keeps hold of the removed bulb.
        for cycle in range(REMOVAL_TWIST_CYCLES):
            self.publish_gripper(0.25)
            self.wait_for_gripper(0.25)
            self.publish_arm_pose(INSERT_POSITION, TWIST_QUATERNION_XYZW)

            if cycle < REMOVAL_TWIST_CYCLES - 1:
                self.wait_for_arm_pose(
                    INSERT_POSITION,
                    TWIST_QUATERNION_XYZW,
                )
                self.publish_gripper(1.0)
                self.wait_for_gripper(1.0)
                self.publish_arm_pose(INSERT_POSITION, INSERT_QUATERNION_XYZW)
                self.wait_for_arm_pose(
                    INSERT_POSITION, INSERT_QUATERNION_XYZW
                )
            else:
                # The final stroke must physically reach its requested rotation,
                # but departure does not wait for position or speed settling.
                self.wait_for_arm_rotation(TWIST_QUATERNION_XYZW)

        # Once the final rotation has reached its endpoint, do not apply any
        # additional arm-position, arm-speed, or bulb-state gate.
        self.publish_bulb_command("complete_removal")

        unlock_sequence = self.publish_platform_lock(False)
        self.wait_for_platform_lock(False, unlock_sequence)
        self.publish_drone_target(0.0, 0.0, 0.0)
        print(
            "[Rflyarm] bulb-replacement command sequence complete; "
            "close Isaac Sim or press Ctrl-C to exit"
        )

    def wait_for_simulator_exit(self) -> int:
        while rclpy.ok() and self.simulator.poll() is None:
            rclpy.spin_once(self, timeout_sec=0.1)
        return_code = self.simulator.poll()
        return 0 if return_code is None else int(return_code)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--headless",
        action="store_true",
        help="run Isaac Sim without opening the visualizer",
    )
    return parser.parse_args()


def start_simulator(headless: bool) -> subprocess.Popen:
    launcher = PROJECT_DIR / "run_simulation.py"
    if not launcher.is_file() or not os.access(launcher, os.X_OK):
        raise FileNotFoundError(f"Simulation launcher not found: {launcher}")
    arguments = [str(launcher), "--device", "cpu", "--render-hz", "30"]
    if headless:
        arguments.append("--headless")
    return subprocess.Popen(
        arguments,
        cwd=PROJECT_DIR,
        start_new_session=True,
    )


def stop_simulator(simulator: subprocess.Popen) -> None:
    if simulator.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(simulator.pid, signal.SIGTERM)
    try:
        simulator.wait(timeout=10.0)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(simulator.pid, signal.SIGKILL)
        simulator.wait()


def main() -> int:
    args = parse_args()
    simulator = start_simulator(args.headless)
    node: BulbReplacementDemo | None = None
    try:
        rclpy.init(args=None)
        node = BulbReplacementDemo(simulator)
        node.run_sequence()
        return node.wait_for_simulator_exit()
    except KeyboardInterrupt:
        print("[Rflyarm] bulb-replacement demo interrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"[Rflyarm] bulb-replacement demo failed: {exc}", file=sys.stderr)
        return 1
    finally:
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        stop_simulator(simulator)


if __name__ == "__main__":
    raise SystemExit(main())
