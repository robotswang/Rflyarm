#!/usr/bin/env python3
"""Run the automated Rflyarm ceiling-bulb replacement demonstration."""

from __future__ import annotations

import argparse
from contextlib import suppress
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
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState
from std_msgs.msg import String


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


class BulbReplacementDemo(Node):
    """Publish the original bulb-replacement sequence using simulation time."""

    def __init__(self, simulator: subprocess.Popen) -> None:
        super().__init__("rflyarm_replace_bulb")
        self.simulator = simulator
        self.sim_time_s: float | None = None
        self.bulb_state: str | None = None
        self.bulb_state_sequence = 0

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
        self.bulb_state_subscription = self.create_subscription(
            String, "/ceiling_bulb/state", self._bulb_state_callback, qos
        )

    def _clock_callback(self, message: Clock) -> None:
        self.sim_time_s = (
            float(message.clock.sec) + float(message.clock.nanosec) * 1.0e-9
        )

    def _bulb_state_callback(self, message: String) -> None:
        self.bulb_state = str(message.data).strip().upper()
        self.bulb_state_sequence += 1

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
        )

        def ready() -> bool:
            return self.sim_time_s is not None and all(
                publisher.get_subscription_count() > 0 for publisher in publishers
            )

        print("[Rflyarm] waiting for Isaac Sim ROS 2 interfaces")
        self._spin_until(ready, "Isaac Sim ROS 2 interfaces", STARTUP_TIMEOUT_S)

    def wait_for_sim_time(self, target_s: float) -> None:
        self._spin_until(
            lambda: self.sim_time_s is not None and self.sim_time_s >= target_s,
            f"simulation time {target_s:.1f} s",
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

        self.wait_for_sim_time(1.0)
        self.publish_drone_target(0.053, -0.2, 7.9)
        self.publish_arm_joints(INITIAL_ARM_JOINTS)

        self.wait_for_sim_time(10.0)
        self.publish_drone_target(0.053, -0.2, 100.0) # 抵住天花板，实际高度应为 7.94

        self.wait_for_sim_time(11.0)
        self.publish_arm_pose(
            (0.0, 0.165, 0.31938), INSERT_QUATERNION_XYZW
        )

        self.wait_for_sim_time(13.0)
        self.publish_arm_pose(INSERT_POSITION, INSERT_QUATERNION_XYZW)

        # Five wrist strokes reproduce the original shell sequence. The first
        # four release and reset the gripper; the fifth keeps hold of the bulb.
        for cycle in range(5):
            cycle_start = 15.0 + 5.0 * cycle
            self.wait_for_sim_time(cycle_start)
            self.publish_gripper(0.25)

            self.wait_for_sim_time(cycle_start + 0.5)
            self.publish_arm_pose(INSERT_POSITION, TWIST_QUATERNION_XYZW)

            if cycle < 4:
                self.wait_for_sim_time(cycle_start + 2.5)
                self.publish_gripper(1.0)

                self.wait_for_sim_time(cycle_start + 3.0)
                self.publish_arm_pose(INSERT_POSITION, INSERT_QUATERNION_XYZW)

        self.wait_for_sim_time(37.5)
        state_sequence = self.publish_bulb_command("complete_removal")
        self.wait_for_bulb_state("UNLOCKED", state_sequence)
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
