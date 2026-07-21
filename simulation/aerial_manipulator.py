"""Six-rotor dynamics adapter for a mixed flight-and-arm articulation."""

from __future__ import annotations

import math

import torch

from isaaclab_contrib.actuators import Thruster, ThrusterCfg
from isaaclab_contrib.utils.types import MultiRotorActions


ROTOR_NAMES = [f"rotor{i}" for i in range(6)]
ROTOR_DIRECTIONS = (-1.0, 1.0, -1.0, 1.0, -1.0, 1.0)
MAX_ROTOR_RAD_S = 1100.0
ROTOR_CONSTANT_RAD_S = 0.00125
THRUST_CONSTANT_RPS = ROTOR_CONSTANT_RAD_S * (2.0 * math.pi) ** 2
MAX_THRUST_N = ROTOR_CONSTANT_RAD_S * MAX_ROTOR_RAD_S**2

# Maps individual rotor thrust [N] to [Fx, Fy, Fz, Tx, Ty, Tz].  Positions are
# measured from the USD body frame and yaw moment is k_m/k_f = 0.02 m.
ALLOCATION_MATRIX = (
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
    (1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
    (0.0, 0.4399, 0.4399, 0.0, -0.4399, -0.4399),
    (-0.508, -0.254, 0.254, 0.508, 0.254, -0.254),
    (-0.02, 0.02, -0.02, 0.02, -0.02, 0.02),
)


class AerialManipulatorDynamics:
    """Apply an Isaac Lab Contrib thruster model to a standard articulation.

    Isaac Lab's ``Multirotor`` asset intentionally rejects regular arm joints.
    Rflyarm therefore remains a normal ``Articulation`` and composes the official
    ``Thruster`` actuator with its joint actuators here.
    """

    def __init__(self, robot, dt: float):
        self.robot = robot
        self.device = robot.device
        body_ids, body_names = robot.find_bodies(["body"], preserve_order=True)
        if body_names != ["body"]:
            raise RuntimeError(f"Expected one flight body named 'body', got {body_names}")
        self.body_ids = torch.tensor(body_ids, device=self.device, dtype=torch.int32)

        self.allocation = torch.tensor(ALLOCATION_MATRIX, device=self.device, dtype=torch.float32)
        self.inverse_allocation = torch.linalg.pinv(self.allocation)
        mass = self._to_torch(robot.data.body_mass).sum(dim=1, keepdim=True)
        hover_thrust = mass * 9.81 / 6.0
        hover_rps = torch.sqrt(hover_thrust / THRUST_CONSTANT_RPS).repeat(1, 6)

        cfg = ThrusterCfg(
            dt=float(dt),
            thrust_range=(0.0, MAX_THRUST_N),
            thrust_const_range=(THRUST_CONSTANT_RPS, THRUST_CONSTANT_RPS),
            tau_inc_range=(0.02, 0.02),
            tau_dec_range=(0.04, 0.04),
            torque_to_thrust_ratio=0.02,
            max_thrust_rate=100000.0,
            thruster_names_expr=ROTOR_NAMES,
        )
        self.thruster = Thruster(
            cfg=cfg,
            thruster_names=ROTOR_NAMES,
            thruster_ids=slice(None),
            num_envs=robot.num_instances,
            device=self.device,
            init_thruster_rps=hover_rps,
        )
        self.last_target = hover_thrust.repeat(1, 6)
        self.last_applied = self.last_target.clone()

    def _to_torch(self, array) -> torch.Tensor:
        value = getattr(array, "torch", array)
        return value.to(self.device) if torch.is_tensor(value) else torch.as_tensor(value, device=self.device)

    def apply_wrench(self, wrench_b: torch.Tensor) -> None:
        """Allocate and apply a body-frame wrench for the next physics step."""
        desired = wrench_b @ self.inverse_allocation.T
        self.last_target = torch.clamp(desired, 0.0, MAX_THRUST_N)
        action = MultiRotorActions(thrusts=self.last_target, thruster_indices=slice(None))
        self.last_applied = self.thruster.compute(action).thrusts
        applied_wrench = self.last_applied @ self.allocation.T

        self.robot.instantaneous_wrench_composer.add_forces_and_torques_index(
            forces=applied_wrench[:, None, :3],
            torques=applied_wrench[:, None, 3:],
            body_ids=self.body_ids,
            is_global=False,
        )
