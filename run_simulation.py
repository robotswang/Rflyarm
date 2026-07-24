#!/usr/bin/env python3
"""Run Rflyarm with Isaac Sim 6.0.1 and Isaac Lab 3.0.0 Beta 2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = PROJECT_DIR.parent
ISAACLAB_DIR = Path(
    os.environ.get("ISAACLAB_PATH", str(WORKSPACE_DIR / "IsaacLab"))
).expanduser().resolve()
ISAACSIM_DIR = Path(
    os.environ.get("ISAACSIM_PATH", str(WORKSPACE_DIR / "isaacsim"))
).expanduser().resolve()
ISAACLAB_LAUNCHER = ISAACLAB_DIR / "isaaclab.sh"
ISAACSIM_ROS_SETUP = ISAACSIM_DIR / "setup_ros_env.sh"
BOOTSTRAP_MARKER = "RFLYARM_ISAACLAB_BOOTSTRAPPED"


def _visualizer_selected(arguments: list[str]) -> bool:
    return any(
        argument in ("--viz", "--visualizer", "--headless")
        or argument.startswith("--viz=")
        or argument.startswith("--visualizer=")
        for argument in arguments
    )


def _bootstrap_isaaclab() -> None:
    """Re-exec this file through Isaac Lab with the matching ROS libraries."""

    if os.environ.get(BOOTSTRAP_MARKER) == "1":
        return
    if not ISAACLAB_LAUNCHER.is_file() or not os.access(ISAACLAB_LAUNCHER, os.X_OK):
        raise FileNotFoundError(f"Isaac Lab launcher not found: {ISAACLAB_LAUNCHER}")
    if not ISAACSIM_ROS_SETUP.is_file():
        raise FileNotFoundError(
            f"Isaac Sim ROS environment not found: {ISAACSIM_ROS_SETUP}"
        )

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

    simulation_arguments = list(sys.argv[1:])
    if not _visualizer_selected(simulation_arguments):
        simulation_arguments[:0] = ["--viz", "kit"]

    # setup_ros_env.sh must be sourced because it exports the ROS library
    # paths used by Isaac Sim's Python 3.12 process. The shell is only an
    # internal bootstrap step; run_simulation.py remains the public entrypoint.
    command = [
        "/usr/bin/bash",
        "-c",
        'set -e; source "$1"; shift; exec "$@"',
        "run_simulation.py",
        str(ISAACSIM_ROS_SETUP),
        str(ISAACLAB_LAUNCHER),
        "-p",
        str(Path(__file__).resolve()),
        *simulation_arguments,
    ]
    os.execve(command[0], command, environment)


_bootstrap_isaaclab()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max-steps", type=int, default=0, help="Stop after N physics steps; 0 runs until closed.")
parser.add_argument("--target-altitude", type=float, default=1.5, help="Default flight target altitude in metres.")
parser.add_argument("--verify-flight", action="store_true", help="Run a deterministic flight-platform acceptance test.")
parser.add_argument("--verify-arm", action="store_true", help="Run the arm acceptance test while holding hover.")
parser.add_argument(
    "--verify-tf",
    action="store_true",
    help="Verify the compact TF tree against PhysX tool/base/body and camera truth.",
)
parser.add_argument(
    "--verify-platform-lock",
    action="store_true",
    help="Verify the PhysX FixedJoint platform lock under rotor and arm loads.",
)
parser.add_argument("--verify-rotors", action="store_true", help="Verify visual rotor synchronization.")
parser.add_argument(
    "--verify-ceiling-bulb",
    action="store_true",
    help="Verify the ceiling-only passive bulb joint and software release.",
)
parser.add_argument("--no-ros", action="store_true", help="Disable ROS 2 even in the interactive run.")
parser.add_argument("--physics-hz", type=float, default=250.0, help="Physics update frequency in Hz.")
parser.add_argument("--render-hz", type=float, default=60.0, help="Maximum visual render frequency in Hz.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.physics_hz <= 0.0:
    parser.error("--physics-hz must be positive")
if args_cli.render_hz <= 0.0:
    parser.error("--render-hz must be positive")
if args_cli.verify_tf and args_cli.no_ros:
    parser.error("--verify-tf requires ROS 2; remove --no-ros")

# The scene always contains an RTX depth camera. Enable camera rendering before
# AppLauncher creates Kit, including when callers omit --enable_cameras.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math
import torch
from pxr import Gf, UsdGeom, Vt

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from simulation.aerial_manipulator import AerialManipulatorDynamics
from simulation.arm_controller import ArmController
from simulation.ceiling_bulb import (
    CeilingBulbMechanism,
    CeilingBulbState,
    define_ceiling_bulb_joint,
)
from simulation.flight_controller import FlightController, define_platform_lock_joint
from simulation.rotor_visualizer import RotorVisualizer
from simulation.scene import RflyarmSceneCfg


PHYSICS_DT = 1.0 / args_cli.physics_hz
RENDER_INTERVAL = max(1, round(args_cli.physics_hz / args_cli.render_hz))
FLIGHT_TARGET_XY = (0.0, 0.0)
ARM_TEST_TARGET = (0.20, -0.50, 0.45, 0.10, 0.35, -0.20)
CEILING_BULB_GRASP_RING_RADIUS_M = 0.01525
# Midpoint of the exposed stem between the head top (-0.07375 m) and
# socket bottom (-0.045 m), after the ceiling asset's 0.5 scale.
CEILING_BULB_GRASP_RING_LOCAL_Z_M = -0.059375
CEILING_BULB_GRASP_RING_AXIAL_WIDTH_M = 0.004
CEILING_BULB_GRASP_RING_SEGMENTS = 64


def _add_grasp_point_marker(stage):
    """Add a non-physical red sphere at the Lula tool_center grasp point."""
    path = "/World/grasp_point_marker"
    sphere = UsdGeom.Sphere.Define(stage, path)
    sphere.GetRadiusAttr().Set(0.012)
    sphere.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.05, 0.02)])
    sphere.GetDisplayOpacityAttr().Set([1.0])
    xform = UsdGeom.Xformable(sphere.GetPrim())
    translate_op = xform.AddTranslateOp()
    return translate_op


def _update_grasp_point_marker(robot, translate_op) -> None:
    """Explicitly follow Link6 world pose (Fabric does not propagate USD children)."""
    link6_id, names = robot.find_bodies(["Link6"], preserve_order=True)
    if names != ["Link6"]:
        return
    pos = robot.data.body_pos_w[0, link6_id[0]].detach().cpu().tolist()
    quat = robot.data.body_quat_w[0, link6_id[0]].detach().cpu().tolist()
    x, y, z, w = quat
    # Keep the visualization marker aligned with the IK/FK tool_center frame.
    ox, oy, oz = 0.0004, 0.0070, 0.1552
    # Rotate local grasp-point offset by Link6 quaternion.
    tx = (1 - 2*(y*y + z*z))*ox + 2*(x*y - z*w)*oy + 2*(x*z + y*w)*oz
    ty = 2*(x*y + z*w)*ox + (1 - 2*(x*x + z*z))*oy + 2*(y*z - x*w)*oz
    tz = 2*(x*z - y*w)*ox + 2*(y*z + x*w)*oy + (1 - 2*(x*x + y*y))*oz
    translate_op.Set(Gf.Vec3d(pos[0] + tx, pos[1] + ty, pos[2] + tz))


def _add_ceiling_angle_markers(stage):
    markers = {}
    colors = {"locked": (0.1, 1.0, 0.1), "release": (1.0, 0.1, 0.1), "current": (1.0, 0.8, 0.05)}
    for name, color in colors.items():
        sphere = UsdGeom.Sphere.Define(stage, f"/World/ceiling_{name}_point")
        sphere.GetRadiusAttr().Set(0.008)
        sphere.GetDisplayColorAttr().Set([Gf.Vec3f(*color)])
        markers[name] = {
            "translate": UsdGeom.Xformable(sphere.GetPrim()).AddTranslateOp(),
            "imageable": UsdGeom.Imageable(sphere.GetPrim()),
        }
    return markers


def _add_ceiling_bulb_grasp_ring(stage):
    """Add a non-physical cyan band on the bulb stem's cylindrical surface."""

    root = UsdGeom.Xform.Define(stage, "/World/ceiling_bulb_grasp_ring")
    root_xform = UsdGeom.Xformable(root)
    translate_op = root_xform.AddTranslateOp()
    orient_op = root_xform.AddOrientOp()

    mesh = UsdGeom.Mesh.Define(stage, "/World/ceiling_bulb_grasp_ring/band")
    half_width = 0.5 * CEILING_BULB_GRASP_RING_AXIAL_WIDTH_M
    local_points = []
    for local_z in (
        CEILING_BULB_GRASP_RING_LOCAL_Z_M - half_width,
        CEILING_BULB_GRASP_RING_LOCAL_Z_M + half_width,
    ):
        for index in range(CEILING_BULB_GRASP_RING_SEGMENTS):
            angle = 2.0 * math.pi * index / CEILING_BULB_GRASP_RING_SEGMENTS
            local_points.append(
                Gf.Vec3f(
                    CEILING_BULB_GRASP_RING_RADIUS_M * math.cos(angle),
                    CEILING_BULB_GRASP_RING_RADIUS_M * math.sin(angle),
                    local_z,
                )
            )
    face_indices = []
    for index in range(CEILING_BULB_GRASP_RING_SEGMENTS):
        next_index = (index + 1) % CEILING_BULB_GRASP_RING_SEGMENTS
        face_indices.extend(
            (
                index,
                next_index,
                CEILING_BULB_GRASP_RING_SEGMENTS + next_index,
                CEILING_BULB_GRASP_RING_SEGMENTS + index,
            )
        )
    mesh.CreatePointsAttr(Vt.Vec3fArray(local_points))
    mesh.CreateFaceVertexCountsAttr(
        Vt.IntArray([4] * CEILING_BULB_GRASP_RING_SEGMENTS)
    )
    mesh.CreateFaceVertexIndicesAttr(Vt.IntArray(face_indices))
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr(Vt.Vec3fArray([Gf.Vec3f(0.0, 1.0, 1.0)]))
    mesh.CreateDisplayOpacityAttr(Vt.FloatArray([1.0]))
    return translate_op, orient_op


def _update_ceiling_bulb_grasp_ring(scene, ring) -> None:
    """Follow the physical ceiling bulb explicitly despite Fabric rendering."""

    translate_op, orient_op = ring
    bulb = scene["ceiling_bulb"]
    position = bulb.data.root_pos_w[0].detach().cpu().tolist()
    quaternion = bulb.data.root_quat_w[0].detach().cpu().tolist()
    x, y, z, w = quaternion
    translate_op.Set(Gf.Vec3d(*position))
    orient_op.Set(Gf.Quatf(w, Gf.Vec3f(x, y, z)))


def _update_depth_camera_pose(
    robot,
    camera,
    body_id: int,
    offset_position: torch.Tensor,
    offset_quaternion: torch.Tensor,
) -> None:
    """Explicitly follow the flight body because Fabric skips USD children."""

    body_pose = robot.data.body_link_pose_w.torch[:, int(body_id)]
    camera_position, camera_quaternion = math_utils.combine_frame_transforms(
        body_pose[:, :3],
        body_pose[:, 3:],
        offset_position,
        offset_quaternion,
    )
    camera.set_world_poses(
        positions=camera_position,
        orientations=camera_quaternion,
        convention="ros",
    )


def _update_ceiling_angle_markers(scene, markers, angle_rad, show_current=True):
    socket = scene["ceiling_socket"]
    pos = socket.data.root_pos_w[0].detach().cpu().tolist()
    quat = socket.data.root_quat_w[0].detach().cpu().tolist()
    x, y, z, w = quat
    def world_point(local):
        ox, oy, oz = local
        tx = (1-2*(y*y+z*z))*ox + 2*(x*y-z*w)*oy + 2*(x*z+y*w)*oz
        ty = 2*(x*y+z*w)*ox + (1-2*(x*x+z*z))*oy + 2*(y*z-x*w)*oz
        tz = 2*(x*z-y*w)*ox + 2*(y*z+x*w)*oy + (1-2*(x*x+y*y))*oz
        return Gf.Vec3d(pos[0]+tx, pos[1]+ty, pos[2]+tz)
    radius, height = 0.045, 0.025
    markers["locked"]["translate"].Set(world_point((radius, 0.0, height)))
    markers["release"]["translate"].Set(world_point((-radius, 0.0, height)))
    current = markers["current"]
    current["translate"].Set(
        world_point((radius * math.cos(angle_rad), radius * math.sin(angle_rad), height)))
    if show_current:
        current["imageable"].MakeVisible()
    else:
        current["imageable"].MakeInvisible()


def _flight_metrics(flight: FlightController, target: torch.Tensor) -> dict[str, float]:
    positions, quaternions, velocities, _angular_velocities = flight.state()
    position = positions[0]
    velocity = velocities[0]
    quaternion = quaternions[0]
    # For xyzw quaternions, cos(tilt) is the world-Z projection of body Z.
    x, y, _z, w = quaternion
    body_z_world_z = 1.0 - 2.0 * (x * x + y * y)
    tilt = torch.acos(torch.clamp(body_z_world_z, -1.0, 1.0))
    return {
        "position_error": float(torch.linalg.vector_norm(position - target).item()),
        "horizontal_error": float(torch.linalg.vector_norm(position[:2] - target[:2]).item()),
        "altitude_error": float(torch.abs(position[2] - target[2]).item()),
        "speed": float(torch.linalg.vector_norm(velocity).item()),
        "tilt_deg": math.degrees(float(tilt.item())),
        "x": float(position[0].item()),
        "y": float(position[1].item()),
        "z": float(position[2].item()),
    }


def run_simulator(
    sim: SimulationContext,
    scene: InteractiveScene,
    ceiling_bulb: CeilingBulbMechanism,
    platform_lock_joint,
) -> None:
    robot = scene["rflyarm"]
    grasp_point_marker = _add_grasp_point_marker(sim.stage)
    ceiling_angle_markers = _add_ceiling_angle_markers(sim.stage)
    ceiling_bulb_grasp_ring = _add_ceiling_bulb_grasp_ring(sim.stage)
    _update_ceiling_bulb_grasp_ring(scene, ceiling_bulb_grasp_ring)
    sim_dt = sim.get_physics_dt()

    flight = FlightController(
        robot=robot,
        dt=sim_dt,
        target=(FLIGHT_TARGET_XY[0], FLIGHT_TARGET_XY[1], args_cli.target_altitude, 0.0),
        lock_joint=platform_lock_joint,
    )
    dynamics = AerialManipulatorDynamics(robot=robot, dt=sim_dt)
    arm = ArmController(robot=robot)
    rotor_visualizer = RotorVisualizer(robot=robot)
    depth_camera = scene["depth_camera"]
    camera_offset_position = torch.tensor(
        depth_camera.cfg.offset.pos,
        device=robot.device,
        dtype=torch.float32,
    ).reshape(1, 3)
    camera_offset_quaternion = torch.tensor(
        depth_camera.cfg.offset.rot,
        device=robot.device,
        dtype=torch.float32,
    ).reshape(1, 4)
    _update_depth_camera_pose(
        robot,
        depth_camera,
        flight.body_id,
        camera_offset_position,
        camera_offset_quaternion,
    )

    print(f"[Rflyarm] bodies ({robot.num_bodies}): {robot.body_names}")
    print(f"[Rflyarm] joints ({robot.num_joints}): {robot.joint_names}")
    print(f"[Rflyarm] total mass: {flight.mass_kg:.6f} kg")
    print(
        f"[Rflyarm] timing: physics_dt={sim_dt:.9f}s "
        f"physics_hz={1.0 / sim_dt:.3f} render_interval={RENDER_INTERVAL}"
    )
    print(f"[Rflyarm] initial flight-body position: {flight.state()[0][0].tolist()}")

    ros = None
    if not args_cli.no_ros and not (
        args_cli.verify_flight
        or args_cli.verify_arm
        or args_cli.verify_platform_lock
        or args_cli.verify_rotors
        or args_cli.verify_ceiling_bulb
    ):
        from simulation.ros2_interface import Ros2Interface

        ros = Ros2Interface(
            robot=robot,
            flight_controller=flight,
            arm_controller=arm,
            ceiling_bulb=ceiling_bulb,
            depth_camera=scene["depth_camera"],
        )

    target = flight.target_position
    if args_cli.verify_flight and args_cli.max_steps == 0:
        args_cli.max_steps = 2500
    if args_cli.verify_arm and args_cli.max_steps == 0:
        args_cli.max_steps = 3500
    if args_cli.verify_rotors and args_cli.max_steps == 0:
        args_cli.max_steps = 500
    if args_cli.verify_ceiling_bulb and args_cli.max_steps == 0:
        args_cli.max_steps = 1700
    if args_cli.verify_tf and args_cli.max_steps == 0:
        args_cli.max_steps = 500
    if args_cli.verify_platform_lock and args_cli.max_steps == 0:
        args_cli.max_steps = 1200

    arm_test_start = 1500
    platform_lock_request_step = 600
    platform_lock_arm_step = 750
    step = 0
    tf_metrics = None
    lock_max_position_drift = 0.0
    lock_max_orientation_drift_deg = 0.0
    lock_max_linear_speed = 0.0
    lock_max_angular_speed = 0.0
    lock_min_total_rotor_thrust = math.inf
    try:
        while simulation_app.is_running():
            if args_cli.verify_arm and step == arm_test_start:
                arm.set_joint_target(ARM_TEST_TARGET)
                print(f"[VERIFY][ARM] target applied at step {step}: {list(ARM_TEST_TARGET)}")
            if args_cli.verify_platform_lock and step == platform_lock_request_step:
                flight.lock_platform()
                print(
                    f"[VERIFY][PLATFORM_LOCK] lock requested at step {step}"
                )
            if args_cli.verify_platform_lock and step == platform_lock_arm_step:
                arm.set_joint_target(ARM_TEST_TARGET)
                print(
                    "[VERIFY][PLATFORM_LOCK] arm disturbance target applied "
                    f"at step {step}: {list(ARM_TEST_TARGET)}"
                )

            if ros is not None:
                ros.process_commands()

            arm.update(sim_dt)
            wrench = flight.compute()
            dynamics.apply_wrench(wrench)

            scene.write_data_to_sim()
            rotor_visualizer.update(dynamics.last_applied, sim_dt)
            sim.step()
            scene.update(sim_dt)
            _update_depth_camera_pose(
                robot,
                depth_camera,
                flight.body_id,
                camera_offset_position,
                camera_offset_quaternion,
            )
            _update_grasp_point_marker(robot, grasp_point_marker)
            ceiling_bulb.update(sim_dt)
            _update_ceiling_bulb_grasp_ring(scene, ceiling_bulb_grasp_ring)
            _update_ceiling_angle_markers(
                scene,
                ceiling_angle_markers,
                ceiling_bulb._wrapped_angle(),
                show_current=ceiling_bulb.progress.state is not CeilingBulbState.UNLOCKED,
            )
            step += 1
            sim_time_ns = round(step * sim_dt * 1_000_000_000)

            if args_cli.verify_platform_lock and flight.platform_locked:
                locked_pose = flight.locked_body_pose
                if locked_pose is None:
                    raise RuntimeError("platform reports locked without a captured pose")
                position, quaternion, linear_velocity, angular_velocity = flight.state()
                position_drift = float(
                    torch.linalg.vector_norm(
                        position - locked_pose[:, :3], dim=1
                    ).max().item()
                )
                normalized_quaternion = quaternion / torch.linalg.vector_norm(
                    quaternion, dim=1, keepdim=True
                ).clamp_min(1.0e-8)
                normalized_locked_quaternion = (
                    locked_pose[:, 3:]
                    / torch.linalg.vector_norm(
                        locked_pose[:, 3:], dim=1, keepdim=True
                    ).clamp_min(1.0e-8)
                )
                quaternion_dot = torch.sum(
                    normalized_quaternion * normalized_locked_quaternion, dim=1
                ).abs().clamp(0.0, 1.0)
                orientation_drift_deg = math.degrees(
                    float((2.0 * torch.acos(quaternion_dot)).max().item())
                )
                linear_speed = float(
                    torch.linalg.vector_norm(linear_velocity, dim=1).max().item()
                )
                angular_speed = float(
                    torch.linalg.vector_norm(angular_velocity, dim=1).max().item()
                )
                total_rotor_thrust = float(
                    dynamics.last_applied.sum(dim=1).min().item()
                )
                lock_max_position_drift = max(
                    lock_max_position_drift, position_drift
                )
                lock_max_orientation_drift_deg = max(
                    lock_max_orientation_drift_deg, orientation_drift_deg
                )
                lock_max_linear_speed = max(
                    lock_max_linear_speed, linear_speed
                )
                lock_max_angular_speed = max(
                    lock_max_angular_speed, angular_speed
                )
                lock_min_total_rotor_thrust = min(
                    lock_min_total_rotor_thrust, total_rotor_thrust
                )

            if ros is not None:
                ros.publish_states(sim_time_ns=sim_time_ns, dt=sim_dt)

            if step % 500 == 0 and (
                args_cli.verify_flight
                or args_cli.verify_arm
                or args_cli.verify_tf
                or args_cli.verify_platform_lock
                or args_cli.verify_rotors
                or args_cli.verify_ceiling_bulb
            ):
                metrics = _flight_metrics(flight, target)
                print(
                    "[Rflyarm] step=%d sim_time=%.3fs position=(%.3f, %.3f, %.3f) "
                    "error=%.3f speed=%.3f tilt=%.2fdeg"
                    % (
                        step,
                        sim_time_ns * 1.0e-9,
                        metrics["x"],
                        metrics["y"],
                        metrics["z"],
                        metrics["position_error"],
                        metrics["speed"],
                        metrics["tilt_deg"],
                    )
                )

            if args_cli.max_steps > 0 and step >= args_cli.max_steps:
                break
    finally:
        if ros is not None:
            if args_cli.verify_tf:
                tf_metrics = ros.tf_truth_metrics()
            ros.shutdown()

    if args_cli.verify_flight:
        metrics = _flight_metrics(flight, target)
        passed = (
            metrics["position_error"] < 0.25
            and metrics["speed"] < 0.35
            and metrics["tilt_deg"] < 12.0
        )
        print(f"[VERIFY][FLIGHT] {'PASS' if passed else 'FAIL'} {metrics}")
        if not passed:
            raise RuntimeError("Flight verification failed")

    if args_cli.verify_arm:
        errors = arm.position_errors(ARM_TEST_TARGET)
        metrics = _flight_metrics(flight, target)
        max_error = float(torch.max(torch.abs(errors)).item())
        passed = max_error < 0.03 and metrics["position_error"] < 0.35 and metrics["speed"] < 0.45
        print(
            f"[VERIFY][ARM] {'PASS' if passed else 'FAIL'} max_joint_error={max_error:.6f} "
            f"errors={errors.tolist()} flight={metrics}"
        )
        if not passed:
            raise RuntimeError("Arm verification failed")

    if args_cli.verify_tf:
        if tf_metrics is None:
            raise RuntimeError("TF verification did not initialize ROS 2")
        passed = (
            all(math.isfinite(value) for value in tf_metrics.values())
            and tf_metrics["body_position_error_m"] < 1.0e-5
            and tf_metrics["body_orientation_error_deg"] < 0.01
            and tf_metrics["base_static_position_drift_m"] < 1.0e-5
            and tf_metrics["base_static_orientation_drift_deg"] < 0.01
            and tf_metrics["tool_position_error_m"] < 1.0e-5
            and tf_metrics["tool_orientation_error_deg"] < 0.01
            and tf_metrics["camera_position_error_m"] < 1.0e-4
            and tf_metrics["camera_orientation_error_deg"] < 0.02
        )
        print(f"[VERIFY][TF] {'PASS' if passed else 'FAIL'} {tf_metrics}")
        if not passed:
            raise RuntimeError("TF ground-truth verification failed")

    if args_cli.verify_platform_lock:
        cumulative_rotation = rotor_visualizer.cumulative_rotation[0]
        min_rotor_rotation = float(
            torch.min(torch.abs(cumulative_rotation)).item()
        )
        joint_enabled_while_locked = bool(
            platform_lock_joint.GetJointEnabledAttr().Get()
        )
        locked_before_release = flight.platform_locked
        flight.unlock_platform()
        joint_disabled_after_release = not bool(
            platform_lock_joint.GetJointEnabledAttr().Get()
        )
        unlocked_after_release = not flight.platform_locked
        passed = (
            locked_before_release
            and joint_enabled_while_locked
            and joint_disabled_after_release
            and unlocked_after_release
            and lock_max_position_drift < 2.0e-4
            and lock_max_orientation_drift_deg < 0.02
            and lock_max_linear_speed < 0.01
            and lock_max_angular_speed < 0.01
            and lock_min_total_rotor_thrust > 100.0
            and min_rotor_rotation > 1.0
        )
        print(
            f"[VERIFY][PLATFORM_LOCK] {'PASS' if passed else 'FAIL'} "
            f"max_position_drift={lock_max_position_drift:.9f} m "
            f"max_orientation_drift={lock_max_orientation_drift_deg:.6f} deg "
            f"max_linear_speed={lock_max_linear_speed:.9f} m/s "
            f"max_angular_speed={lock_max_angular_speed:.9f} rad/s "
            f"min_total_rotor_thrust={lock_min_total_rotor_thrust:.3f} N "
            f"min_rotor_rotation={min_rotor_rotation:.3f} rad "
            f"joint_enabled_while_locked={joint_enabled_while_locked} "
            f"joint_disabled_after_release={joint_disabled_after_release}"
        )
        if not passed:
            raise RuntimeError("Platform FixedJoint verification failed")

    if args_cli.verify_rotors:
        cumulative = rotor_visualizer.cumulative_rotation[0]
        position_error = rotor_visualizer.position_error()[0]
        expected_directions = torch.tensor(
            (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0), device=robot.device
        )
        direction_ok = torch.all(torch.sign(cumulative) == expected_directions)
        max_position_error = float(torch.max(torch.abs(position_error)).item())
        min_rotation = float(torch.min(torch.abs(cumulative)).item())
        passed = bool(direction_ok.item()) and min_rotation > 1.0 and max_position_error < 0.01
        print(
            f"[VERIFY][ROTORS] {'PASS' if passed else 'FAIL'} "
            f"cumulative_rotation={cumulative.tolist()} "
            f"min_abs_rotation={min_rotation:.3f} rad "
            f"max_position_error={max_position_error:.6f} rad"
        )
        if not passed:
            raise RuntimeError("Rotor visual verification failed")

    if args_cli.verify_ceiling_bulb:
        joint_enabled = bool(ceiling_bulb.joint.GetJointEnabledAttr().Get())
        relative_angle_deg = math.degrees(ceiling_bulb._wrapped_angle())
        current_marker_visibility = ceiling_angle_markers["current"]["imageable"].ComputeVisibility()
        current_marker_hidden = current_marker_visibility == UsdGeom.Tokens.invisible
        passed = (
            ceiling_bulb.progress.state is CeilingBulbState.UNLOCKED
            and not joint_enabled
            and current_marker_hidden
        )
        print(
            f"[VERIFY][CEILING_BULB] {'PASS' if passed else 'FAIL'} "
            f"state={ceiling_bulb.progress.state.value} "
            f"angle={ceiling_bulb.progress.logical_angle_deg:.3f}deg "
            f"loosened={ceiling_bulb.progress.loosened_angle_deg:.3f}deg "
            f"remaining={ceiling_bulb.progress.remaining_angle_deg:.3f}deg "
            f"relative_angle={relative_angle_deg:.3f}deg "
            f"joint_enabled={joint_enabled} "
            f"current_marker_visibility={current_marker_visibility}"
        )
        if not passed:
            raise RuntimeError("Ceiling bulb verification failed")


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=RENDER_INTERVAL,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(4.5, 4.5, 3.2), target=(0.0, 0.0, 1.0))

    scene = InteractiveScene(RflyarmSceneCfg(num_envs=1, env_spacing=0.0))
    platform_lock_joint = define_platform_lock_joint(sim.stage)
    ceiling_bulb_joint = define_ceiling_bulb_joint(
        sim.stage,
        verify_velocity_deg_s=-60.0 if args_cli.verify_ceiling_bulb else None,
    )
    sim.reset()
    scene.update(0.0)
    ceiling_bulb = CeilingBulbMechanism(
        joint=ceiling_bulb_joint,
        socket=scene["ceiling_socket"],
        bulb=scene["ceiling_bulb"],
    )
    ceiling_bulb.reset()
    print("[Rflyarm] Isaac Lab scene setup complete")
    run_simulator(sim, scene, ceiling_bulb, platform_lock_joint)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Rflyarm] fatal error: {exc}", file=sys.stderr)
        raise
    finally:
        simulation_app.close()
