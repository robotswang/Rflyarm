"""Purely visual rotor-joint synchronization driven by applied thrust."""

from __future__ import annotations

import math

import torch

from simulation.aerial_manipulator import MAX_THRUST_N, ROTOR_DIRECTIONS


ROTOR_JOINT_NAMES = [f"rotor_{index}" for index in range(1, 7)]
VISUAL_IDLE_RAD_S = 10.0
VISUAL_MAX_RAD_S = 80.0
ACTIVE_THRUST_N = 1.0


class RotorVisualizer:
    """Animate rotor joints without adding forces, torques, or angular momentum."""

    def __init__(self, robot):
        self.robot = robot
        self.device = robot.device
        joint_ids, joint_names = robot.find_joints(ROTOR_JOINT_NAMES, preserve_order=True)
        if joint_names != ROTOR_JOINT_NAMES:
            raise RuntimeError(f"Rotor joints do not match the expected order: {joint_names}")
        self.joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.int32)
        self.directions = torch.tensor(ROTOR_DIRECTIONS, device=self.device, dtype=torch.float32).reshape(1, 6)

        positions = self._to_torch(robot.data.joint_pos)
        self.angles = positions[:, self.joint_ids].clone()
        self.zero_velocity = torch.zeros_like(self.angles)
        self.angular_velocity = torch.zeros_like(self.angles)
        self.cumulative_rotation = torch.zeros_like(self.angles)

    def _to_torch(self, array) -> torch.Tensor:
        value = getattr(array, "torch", array)
        return value.to(self.device) if torch.is_tensor(value) else torch.as_tensor(value, device=self.device)

    def update(self, applied_thrust: torch.Tensor, dt: float) -> None:
        """Advance visual phase and write rotor positions directly to PhysX.

        Direct position writes avoid motor-drive reaction torques. Joint velocities
        remain zero because the actual flight wrench is already supplied by the
        Isaac Lab Contrib thruster model.
        """
        thrust = torch.clamp(applied_thrust, 0.0, MAX_THRUST_N)
        speed_ratio = torch.sqrt(thrust / MAX_THRUST_N)
        speed = VISUAL_IDLE_RAD_S + (VISUAL_MAX_RAD_S - VISUAL_IDLE_RAD_S) * speed_ratio
        speed = torch.where(thrust > ACTIVE_THRUST_N, speed, torch.zeros_like(speed))
        self.angular_velocity = speed * self.directions

        delta = self.angular_velocity * float(dt)
        self.cumulative_rotation += delta
        self.angles = torch.remainder(self.angles + delta + math.pi, 2.0 * math.pi) - math.pi

        self.robot.write_joint_position_to_sim_index(position=self.angles, joint_ids=self.joint_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=self.zero_velocity, joint_ids=self.joint_ids)

    def position_error(self) -> torch.Tensor:
        """Return wrapped measured-minus-commanded visual joint error."""
        positions = self._to_torch(self.robot.data.joint_pos)[:, self.joint_ids]
        return torch.remainder(positions - self.angles + math.pi, 2.0 * math.pi) - math.pi
