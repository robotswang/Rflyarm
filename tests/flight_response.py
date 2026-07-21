#!/usr/bin/env python3
"""Measure one large position-and-yaw step response over ROS 2."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


@dataclass(frozen=True)
class Target:
    name: str
    group: int
    position: tuple[float, float, float]
    yaw_deg: float
    measured_step: bool


TARGETS = (
    Target("point1", 1, (6.0, 6.0, 5.0), 90.0, True),
    Target("point2", 1, (0.0, 0.0, 1.5), 0.0, True),
)


@dataclass
class ResponseResult:
    name: str
    group: int
    measured_step: bool
    start: tuple[float, float, float]
    target: tuple[float, float, float]
    target_yaw_deg: float
    rise_time_s: float
    position_settling_time_s: float
    yaw_settling_time_s: float
    settling_time_s: float
    wall_settling_time_s: float
    overshoot_m: float
    max_cross_track_m: float
    max_tilt_deg: float
    final_error_m: float
    final_yaw_error_deg: float


def wrap_angle(angle_rad: float) -> float:
    """Wrap an angle to [-pi, pi)."""

    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


class FlightResponseMonitor(Node):
    """Publish step commands and evaluate the resulting pose stream."""

    def __init__(
        self,
        tolerance_m: float,
        yaw_tolerance_deg: float,
        hold_s: float,
        timeout_s: float,
    ) -> None:
        super().__init__("rflyarm_flight_response")
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self.publisher = self.create_publisher(PoseStamped, "/drone/cmd_pose", qos)
        self.create_subscription(PoseStamped, "/drone/pose", self._pose_callback, qos)
        self.tolerance_m = float(tolerance_m)
        self.yaw_tolerance_rad = math.radians(float(yaw_tolerance_deg))
        self.hold_s = float(hold_s)
        self.timeout_s = float(timeout_s)
        self.latest_pose = None
        self.latest_quaternion = None
        self.latest_sim_time_s = None
        self.target_index = -1
        self.results: list[ResponseResult] = []
        self.failed = False
        self._reset_phase_state()

    def _reset_phase_state(self) -> None:
        self.start_position = None
        self.sample_count = 0
        self.first_90_percent_time_s = None
        self.position_tolerance_start_s = None
        self.yaw_tolerance_start_s = None
        self.combined_tolerance_start_s = None
        self.max_progress = 0.0
        self.max_cross_track = 0.0
        self.max_tilt_deg = 0.0
        self.last_error = math.inf
        self.last_yaw_error = math.inf
        self.publish_count = 0
        self.phase_start_sim_time_s = None
        self.phase_start_wall = None
        self.combined_tolerance_start_wall = None

    @property
    def discovered(self) -> bool:
        return self.publisher.get_subscription_count() > 0

    @property
    def complete(self) -> bool:
        return len(self.results) == len(TARGETS)

    def start_next_target(self) -> None:
        self.target_index += 1
        self._reset_phase_state()
        self.start_position = self.latest_pose
        self.phase_start_sim_time_s = self.latest_sim_time_s
        self.phase_start_wall = time.monotonic()
        self.publish_command()
        target = TARGETS[self.target_index]
        phase = "measured step" if target.measured_step else "staging"
        self.get_logger().info(
            f"Starting {target.name} ({phase}): start={self.start_position}, "
            f"target={target.position}, yaw={target.yaw_deg:.1f}deg"
        )

    def publish_command(self) -> None:
        target = TARGETS[self.target_index]
        yaw_rad = math.radians(target.yaw_deg)
        message = PoseStamped()
        if self.latest_sim_time_s is not None:
            stamp_ns = round(self.latest_sim_time_s * 1_000_000_000)
            message.header.stamp.sec = stamp_ns // 1_000_000_000
            message.header.stamp.nanosec = stamp_ns % 1_000_000_000
        message.header.frame_id = "map"
        message.pose.position.x, message.pose.position.y, message.pose.position.z = target.position
        message.pose.orientation.z = math.sin(0.5 * yaw_rad)
        message.pose.orientation.w = math.cos(0.5 * yaw_rad)
        self.publisher.publish(message)
        self.publish_count += 1

    @staticmethod
    def _normalized_quaternion(quaternion) -> tuple[float, float, float, float]:
        x, y, z, w = quaternion
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm < 1.0e-8:
            raise ValueError("Pose quaternion norm is zero")
        return x / norm, y / norm, z / norm, w / norm

    @classmethod
    def _tilt_deg(cls, quaternion) -> float:
        x, y, _z, _w = cls._normalized_quaternion(quaternion)
        body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
        return math.degrees(math.acos(max(-1.0, min(1.0, body_z_world_z))))

    @classmethod
    def _yaw_rad(cls, quaternion) -> float:
        x, y, z, w = cls._normalized_quaternion(quaternion)
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _pose_callback(self, message: PoseStamped) -> None:
        sim_time_s = message.header.stamp.sec + message.header.stamp.nanosec * 1.0e-9
        if self.latest_sim_time_s is not None and sim_time_s < self.latest_sim_time_s:
            self.get_logger().error(
                f"/drone/pose simulation time moved backwards: "
                f"{sim_time_s:.9f} < {self.latest_sim_time_s:.9f}"
            )
            self.failed = True
            return
        self.latest_sim_time_s = sim_time_s
        self.latest_pose = (
            message.pose.position.x,
            message.pose.position.y,
            message.pose.position.z,
        )
        self.latest_quaternion = (
            message.pose.orientation.x,
            message.pose.orientation.y,
            message.pose.orientation.z,
            message.pose.orientation.w,
        )
        if self.target_index < 0 or self.complete or self.failed:
            return

        self.sample_count += 1
        if self.publish_count < 3 and self.sample_count % 5 == 0:
            self.publish_command()

        target = TARGETS[self.target_index]
        start = self.start_position
        displacement = tuple(target.position[index] - start[index] for index in range(3))
        distance = math.sqrt(sum(value * value for value in displacement))
        offset = tuple(self.latest_pose[index] - start[index] for index in range(3))
        error_vector = tuple(
            self.latest_pose[index] - target.position[index] for index in range(3)
        )
        error = math.sqrt(sum(value * value for value in error_vector))
        yaw_error = abs(wrap_angle(self._yaw_rad(self.latest_quaternion) - math.radians(target.yaw_deg)))
        self.last_error = error
        self.last_yaw_error = yaw_error

        if distance > 1.0e-8:
            progress = sum(offset[index] * displacement[index] for index in range(3)) / distance**2
            projected = tuple(progress * displacement[index] for index in range(3))
            cross_track = math.sqrt(
                sum((offset[index] - projected[index]) ** 2 for index in range(3))
            )
            self.max_progress = max(self.max_progress, progress)
            self.max_cross_track = max(self.max_cross_track, cross_track)
            if self.first_90_percent_time_s is None and progress >= 0.9:
                self.first_90_percent_time_s = sim_time_s

        self.max_tilt_deg = max(self.max_tilt_deg, self._tilt_deg(self.latest_quaternion))

        if error <= self.tolerance_m:
            if self.position_tolerance_start_s is None:
                self.position_tolerance_start_s = sim_time_s
        else:
            self.position_tolerance_start_s = None

        if yaw_error <= self.yaw_tolerance_rad:
            if self.yaw_tolerance_start_s is None:
                self.yaw_tolerance_start_s = sim_time_s
        else:
            self.yaw_tolerance_start_s = None

        if error <= self.tolerance_m and yaw_error <= self.yaw_tolerance_rad:
            if self.combined_tolerance_start_s is None:
                self.combined_tolerance_start_s = sim_time_s
                self.combined_tolerance_start_wall = time.monotonic()
        else:
            self.combined_tolerance_start_s = None
            self.combined_tolerance_start_wall = None

        combined_hold_s = (
            0.0
            if self.combined_tolerance_start_s is None
            else sim_time_s - self.combined_tolerance_start_s
        )
        if combined_hold_s >= self.hold_s:
            if (
                self.phase_start_sim_time_s is None
                or self.phase_start_wall is None
                or self.position_tolerance_start_s is None
                or self.yaw_tolerance_start_s is None
                or self.combined_tolerance_start_s is None
                or self.combined_tolerance_start_wall is None
            ):
                raise RuntimeError("Incomplete response timing state")
            settling_time_s = self.combined_tolerance_start_s - self.phase_start_sim_time_s
            rise_time_s = (
                self.first_90_percent_time_s
                if self.first_90_percent_time_s is not None
                else sim_time_s
            ) - self.phase_start_sim_time_s
            result = ResponseResult(
                name=target.name,
                group=target.group,
                measured_step=target.measured_step,
                start=start,
                target=target.position,
                target_yaw_deg=target.yaw_deg,
                rise_time_s=rise_time_s,
                position_settling_time_s=(
                    self.position_tolerance_start_s - self.phase_start_sim_time_s
                ),
                yaw_settling_time_s=(
                    self.yaw_tolerance_start_s - self.phase_start_sim_time_s
                ),
                settling_time_s=settling_time_s,
                wall_settling_time_s=self.combined_tolerance_start_wall - self.phase_start_wall,
                overshoot_m=max(0.0, self.max_progress - 1.0) * distance,
                max_cross_track_m=self.max_cross_track,
                max_tilt_deg=self.max_tilt_deg,
                final_error_m=error,
                final_yaw_error_deg=math.degrees(yaw_error),
            )
            self.results.append(result)
            label = "STEP" if result.measured_step else "STAGE"
            print(
                f"[RESPONSE][{label}][{result.name}] rise={result.rise_time_s:.3f}s "
                f"position_settle={result.position_settling_time_s:.3f}s "
                f"yaw_settle={result.yaw_settling_time_s:.3f}s "
                f"combined_settle={result.settling_time_s:.3f}s "
                f"wall_settle={result.wall_settling_time_s:.3f}s "
                f"rtf={result.settling_time_s / max(result.wall_settling_time_s, 1.0e-9):.3f} "
                f"overshoot={result.overshoot_m:.4f}m "
                f"cross_track={result.max_cross_track_m:.4f}m "
                f"max_tilt={result.max_tilt_deg:.2f}deg "
                f"final_error={result.final_error_m:.4f}m "
                f"final_yaw_error={result.final_yaw_error_deg:.2f}deg"
            )
            if not self.complete:
                self.start_next_target()
        elif sim_time_s - self.phase_start_sim_time_s >= self.timeout_s:
            print(
                f"[RESPONSE][{target.name}] FAIL timeout "
                f"position_error={error:.4f}m yaw_error={math.degrees(yaw_error):.2f}deg"
            )
            self.failed = True


def validate_targets() -> None:
    """Ensure each requested A-to-B test moves at least 3 m on every axis."""

    for group in (1,):
        first, second = (target for target in TARGETS if target.group == group)
        axis_delta = tuple(
            abs(second.position[index] - first.position[index]) for index in range(3)
        )
        if any(delta < 3.0 for delta in axis_delta):
            raise ValueError(f"Group {group} axis deltas must all be >= 3 m, got {axis_delta}")
        if abs(wrap_angle(math.radians(second.yaw_deg - first.yaw_deg))) < 1.0e-6:
            raise ValueError(f"Group {group} must change yaw")
        print(
            f"[RESPONSE][GROUP {group}] A={first.position}, yaw={first.yaw_deg:.1f}deg; "
            f"B={second.position}, yaw={second.yaw_deg:.1f}deg; delta={axis_delta}m"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tolerance", type=float, default=0.15, help="Position tolerance in metres.")
    parser.add_argument(
        "--yaw-tolerance", type=float, default=5.0, help="Yaw tolerance in degrees."
    )
    parser.add_argument("--hold", type=float, default=1.0, help="Continuous in-tolerance time.")
    parser.add_argument(
        "--phase-timeout", type=float, default=45.0, help="Per-target simulation-time timeout."
    )
    parser.add_argument(
        "--max-settling-time",
        type=float,
        default=5.0,
        help="Maximum allowed settling time for each commanded point.",
    )
    parser.add_argument(
        "--max-wall-settling-time",
        type=float,
        default=5.0,
        help="Maximum allowed wall-clock settling time for each commanded point.",
    )
    parser.add_argument("--wall-timeout", type=float, default=600.0, help="Overall wall timeout.")
    args = parser.parse_args()

    validate_targets()
    rclpy.init()
    node = FlightResponseMonitor(
        args.tolerance, args.yaw_tolerance, args.hold, args.phase_timeout
    )
    deadline = time.monotonic() + args.wall_timeout
    try:
        while rclpy.ok() and time.monotonic() < deadline and not node.complete and not node.failed:
            rclpy.spin_once(node, timeout_sec=0.05)
            if node.target_index < 0 and node.discovered and node.latest_pose is not None:
                node.start_next_target()
        if node.complete:
            measured = [result for result in node.results if result.measured_step]
            total_settling_time = sum(result.settling_time_s for result in measured)
            max_settling_time = max(result.settling_time_s for result in measured)
            max_wall_settling_time = max(result.wall_settling_time_s for result in measured)
            max_overshoot = max(result.overshoot_m for result in measured)
            max_tilt = max(result.max_tilt_deg for result in measured)
            max_yaw_error = max(result.final_yaw_error_deg for result in measured)
            within_time_limit = all(
                result.settling_time_s <= args.max_settling_time
                and result.wall_settling_time_s <= args.max_wall_settling_time
                for result in measured
            )
            print(
                f"[RESPONSE][SUMMARY] {'PASS' if within_time_limit else 'FAIL'} "
                f"measured_steps={len(measured)} "
                f"total_settle={total_settling_time:.3f}s "
                f"max_settle={max_settling_time:.3f}s "
                f"limit={args.max_settling_time:.3f}s "
                f"max_wall_settle={max_wall_settling_time:.3f}s "
                f"wall_limit={args.max_wall_settling_time:.3f}s "
                f"max_overshoot={max_overshoot:.4f}m max_tilt={max_tilt:.2f}deg "
                f"max_final_yaw_error={max_yaw_error:.2f}deg"
            )
            return 0 if within_time_limit else 1
        print("[RESPONSE][SUMMARY] FAIL")
        return 1
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
