#!/usr/bin/env python3
"""Run Rflyarm with Isaac Sim 6.0.1 and Isaac Lab 3.0.0 Beta 2."""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--max-steps", type=int, default=0, help="Stop after N physics steps; 0 runs until closed.")
parser.add_argument("--target-altitude", type=float, default=1.5, help="Default flight target altitude in metres.")
parser.add_argument("--verify-flight", action="store_true", help="Run a deterministic flight-platform acceptance test.")
parser.add_argument("--verify-arm", action="store_true", help="Run the arm acceptance test while holding hover.")
parser.add_argument("--verify-rotors", action="store_true", help="Verify visual rotor synchronization.")
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
import sys

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene
from isaaclab.sim import SimulationContext

from simulation.aerial_manipulator import AerialManipulatorDynamics
from simulation.arm_controller import ArmController
from simulation.flight_controller import FlightController
from simulation.rotor_visualizer import RotorVisualizer
from simulation.scene import RflyarmSceneCfg


PHYSICS_DT = 1.0 / args_cli.physics_hz
RENDER_INTERVAL = max(1, round(args_cli.physics_hz / args_cli.render_hz))
FLIGHT_TARGET_XY = (0.0, 0.0)
ARM_TEST_TARGET = (0.20, -0.50, 0.45, 0.10, 0.35, -0.20)


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


def run_simulator(sim: SimulationContext, scene: InteractiveScene) -> None:
    robot = scene["rflyarm"]
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
    if not args_cli.no_ros and not (args_cli.verify_flight or args_cli.verify_arm or args_cli.verify_rotors):
        from simulation.ros2_interface import Ros2Interface

        ros = Ros2Interface(
            robot=robot,
            flight_controller=flight,
            arm_controller=arm,
            depth_camera=scene["depth_camera"],
        )

    target = flight.target_position
    if args_cli.verify_flight and args_cli.max_steps == 0:
        args_cli.max_steps = 2500
    if args_cli.verify_arm and args_cli.max_steps == 0:
        args_cli.max_steps = 3500
    if args_cli.verify_rotors and args_cli.max_steps == 0:
        args_cli.max_steps = 500

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


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(
        dt=PHYSICS_DT,
        render_interval=RENDER_INTERVAL,
        device=args_cli.device,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=(4.5, 4.5, 3.2), target=(0.0, 0.0, 1.0))

    scene = InteractiveScene(RflyarmSceneCfg(num_envs=1, env_spacing=0.0))
    sim.reset()
    scene.update(0.0)
    print("[Rflyarm] Isaac Lab scene setup complete")
    run_simulator(sim, scene)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[Rflyarm] fatal error: {exc}", file=sys.stderr)
        raise
    finally:
        simulation_app.close()
