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

# The scene always contains an RTX depth camera. Enable camera rendering before
# AppLauncher creates Kit, including when callers omit --enable_cameras.
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import math
import torch
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from simulation.aerial_manipulator import AerialManipulatorDynamics
from simulation.arm_controller import ArmController
from simulation.ceiling_bulb import (
    CeilingBulbMechanism,
    CeilingBulbState,
    define_ceiling_bulb_joint,
)
from simulation.flight_controller import FlightController
from simulation.rotor_visualizer import RotorVisualizer
from simulation.scene import RflyarmSceneCfg


PHYSICS_DT = 1.0 / args_cli.physics_hz
RENDER_INTERVAL = max(1, round(args_cli.physics_hz / args_cli.render_hz))
FLIGHT_TARGET_XY = (0.0, 0.0)
ARM_TEST_TARGET = (0.20, -0.50, 0.45, 0.10, 0.35, -0.20)


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
) -> None:
    robot = scene["rflyarm"]
    grasp_point_marker = _add_grasp_point_marker(sim.stage)
    ceiling_angle_markers = _add_ceiling_angle_markers(sim.stage)
    sim_dt = sim.get_physics_dt()

    flight = FlightController(
        robot=robot,
        dt=sim_dt,
        target=(FLIGHT_TARGET_XY[0], FLIGHT_TARGET_XY[1], args_cli.target_altitude, 0.0),
    )
    dynamics = AerialManipulatorDynamics(robot=robot, dt=sim_dt)
    arm = ArmController(robot=robot)
    rotor_visualizer = RotorVisualizer(robot=robot)

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

    arm_test_start = 1500
    step = 0
    try:
        while simulation_app.is_running():
            if args_cli.verify_arm and step == arm_test_start:
                arm.set_joint_target(ARM_TEST_TARGET)
                print(f"[VERIFY][ARM] target applied at step {step}: {list(ARM_TEST_TARGET)}")

            if ros is not None:
                ros.process_commands()

            arm.update(sim_dt)
            wrench = flight.compute()
            dynamics.apply_wrench(wrench)

            scene.write_data_to_sim()
            rotor_visualizer.update(dynamics.last_applied, sim_dt)
            sim.step()
            scene.update(sim_dt)
            _update_grasp_point_marker(robot, grasp_point_marker)
            ceiling_bulb.update(sim_dt)
            _update_ceiling_angle_markers(
                scene,
                ceiling_angle_markers,
                ceiling_bulb._wrapped_angle(),
                show_current=ceiling_bulb.progress.state is not CeilingBulbState.UNLOCKED,
            )
            step += 1
            sim_time_ns = round(step * sim_dt * 1_000_000_000)

            if ros is not None:
                ros.publish_states(sim_time_ns=sim_time_ns, dt=sim_dt)

            if step % 500 == 0:
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
    run_simulator(sim, scene, ceiling_bulb)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Rflyarm] fatal error: {exc}", file=sys.stderr)
        raise
    finally:
        simulation_app.close()
