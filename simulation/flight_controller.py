"""Geometric flight controller using Isaac Lab articulation body state."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab_contrib.controllers.lee_controller_utils import compute_desired_orientation


class FlightController:
    """Track a world-frame ``(x, y, z, yaw)`` target with a body wrench.

    The collected USD's PhysX articulation root link is ``Link3``.  Flight state
    must therefore be read explicitly from the rigid body named ``body`` instead
    of from ``robot.data.root_*``.
    """

    def __init__(self, robot, dt: float, target=(0.0, 0.0, 1.5, 0.0)):
        self.robot = robot
        self.device = robot.device
        self.dt = float(dt)
        if self.dt <= 0.0:
            raise ValueError(f"Flight controller dt must be positive, got {self.dt}")
        body_ids, body_names = robot.find_bodies(["body"], preserve_order=True)
        if body_names != ["body"]:
            raise RuntimeError(f"Expected one flight body named 'body', got {body_names}")
        self.body_id = body_ids[0]

        self.command = torch.tensor(target, device=self.device, dtype=torch.float32).repeat(robot.num_instances, 1)
        self.mass = self._to_torch(robot.data.body_mass).sum(dim=1)
        self.gravity = 9.81

        # Cascaded position-to-velocity and velocity-to-acceleration gains.  A
        # bounded velocity reference makes step commands brake early instead of
        # crossing the target at high speed.
        self.k_pos = torch.tensor((0.8, 0.8, 1.8), device=self.device)
        self.k_vel = torch.tensor((3.0, 3.0, 5.0), device=self.device)
        self.k_int = torch.tensor((0.05, 0.05, 0.08), device=self.device)
        self.k_rot = torch.tensor((300.0, 300.0, 300.0), device=self.device)
        self.k_angvel = torch.tensor((125.0, 125.0, 125.0), device=self.device)
        self.position_integral = torch.zeros((robot.num_instances, 3), device=self.device)
        self.integral_limit = 1.0
        self.integral_band = 0.5
        self.max_horizontal_velocity = 2.5
        self.max_vertical_velocity = 1.5
        self.max_horizontal_acceleration = 2.2
        self.max_upward_acceleration = 4.5
        self.max_downward_acceleration = 3.5
        # The six rotors have much less yaw authority than roll/pitch authority
        # because yaw comes only from reaction torque (k_m/k_f = 0.02 m).  An
        # unbounded 90-degree yaw step saturates alternating rotors and turns
        # the requested yaw torque into excess collective thrust.  Keep the
        # command inside the approximately 21 N*m hover allocation envelope.
        self.max_yaw_torque = 20.0
        self.rotation_matrix_buffer = torch.zeros((robot.num_instances, 3, 3), device=self.device)

    def _to_torch(self, array) -> torch.Tensor:
        value = getattr(array, "torch", array)
        return value.to(self.device) if torch.is_tensor(value) else torch.as_tensor(value, device=self.device)

    @property
    def mass_kg(self) -> float:
        return float(self.mass[0].item())

    @property
    def target_position(self) -> torch.Tensor:
        return self.command[0, :3]

    def set_target(self, position, yaw: float = 0.0) -> None:
        self.command[:, :3] = torch.as_tensor(position, device=self.device, dtype=torch.float32)
        self.command[:, 3] = float(yaw)
        # A previous target's integral bias must not drive the next step command.
        self.position_integral.zero_()

    def state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pose = self._to_torch(self.robot.data.body_link_pose_w)[:, self.body_id]
        velocity = self._to_torch(self.robot.data.body_com_vel_w)[:, self.body_id]
        position = pose[:, :3]
        quaternion = pose[:, 3:]
        linear_velocity_w = velocity[:, :3]
        angular_velocity_w = velocity[:, 3:]
        angular_velocity_b = math_utils.quat_apply_inverse(quaternion, angular_velocity_w)
        return position, quaternion, linear_velocity_w, angular_velocity_b

    def compute(self) -> torch.Tensor:
        position, quaternion, linear_velocity_w, angular_velocity_b = self.state()
        position_error = self.command[:, :3] - position

        in_band = torch.linalg.vector_norm(position_error, dim=1) < self.integral_band
        self.position_integral[in_band] += position_error[in_band] * self.dt
        self.position_integral[~in_band] = 0.0
        self.position_integral.clamp_(-self.integral_limit, self.integral_limit)

        desired_velocity_w = self.k_pos * position_error
        horizontal_velocity = desired_velocity_w[:, :2]
        horizontal_speed = torch.linalg.vector_norm(horizontal_velocity, dim=1, keepdim=True)
        horizontal_velocity_scale = torch.clamp(
            self.max_horizontal_velocity / torch.clamp(horizontal_speed, min=1.0e-6),
            max=1.0,
        )
        desired_velocity_w[:, :2] = horizontal_velocity * horizontal_velocity_scale
        desired_velocity_w[:, 2].clamp_(-self.max_vertical_velocity, self.max_vertical_velocity)

        desired_acceleration_w = (
            self.k_vel * (desired_velocity_w - linear_velocity_w)
            + self.k_int * self.position_integral
        )

        horizontal = desired_acceleration_w[:, :2]
        horizontal_norm = torch.linalg.vector_norm(horizontal, dim=1, keepdim=True)
        horizontal_scale = torch.clamp(
            self.max_horizontal_acceleration / torch.clamp(horizontal_norm, min=1.0e-6),
            max=1.0,
        )
        desired_acceleration_w[:, :2] = horizontal * horizontal_scale
        desired_acceleration_w[:, 2].clamp_(
            -self.max_downward_acceleration,
            self.max_upward_acceleration,
        )

        desired_acceleration_w[:, 2] += self.gravity
        desired_force_w = self.mass[:, None] * desired_acceleration_w

        rotation = math_utils.matrix_from_quat(quaternion)
        body_z_w = rotation[:, :, 2]
        total_thrust = torch.sum(desired_force_w * body_z_w, dim=1).clamp(min=0.0)
        desired_quaternion = compute_desired_orientation(
            desired_force_w, self.command[:, 3], self.rotation_matrix_buffer
        )

        error_quaternion = math_utils.quat_mul(math_utils.quat_inv(quaternion), desired_quaternion)
        error_rotation = math_utils.matrix_from_quat(error_quaternion)
        skew = error_rotation.transpose(-1, -2) - error_rotation
        rotation_error = 0.5 * torch.stack(
            (-skew[:, 1, 2], skew[:, 0, 2], -skew[:, 0, 1]), dim=1
        )
        torque = -self.k_rot * rotation_error - self.k_angvel * angular_velocity_b
        torque[:, 2].clamp_(-self.max_yaw_torque, self.max_yaw_torque)

        wrench = torch.zeros((self.robot.num_instances, 6), device=self.device)
        wrench[:, 2] = total_thrust
        wrench[:, 3:] = torque
        return wrench
