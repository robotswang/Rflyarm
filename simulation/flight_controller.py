"""Geometric flight controller using Isaac Lab articulation body state."""

from __future__ import annotations

import torch

import isaaclab.utils.math as math_utils
from isaaclab_contrib.controllers.lee_controller_utils import compute_desired_orientation


PLATFORM_LOCK_BODY_PATH = "/World/layout/rflyarm/body"
PLATFORM_LOCK_JOINT_PATH = "/World/platform_lock_joint"


def define_platform_lock_joint(stage):
    """Create the disabled external FixedJoint used to lock the flight body."""

    from pxr import Gf, Sdf, UsdGeom, UsdPhysics

    body_prim = stage.GetPrimAtPath(PLATFORM_LOCK_BODY_PATH)
    if not body_prim.IsValid():
        raise RuntimeError(
            f"Platform-lock rigid body not found: {PLATFORM_LOCK_BODY_PATH}"
        )
    if not body_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"Platform-lock target is not a rigid body: {PLATFORM_LOCK_BODY_PATH}"
        )

    # Initialize the disabled world-side frame at the authored body pose so
    # PhysX does not report a misleading disjoint-joint warning at startup.
    body_world = UsdGeom.XformCache().GetLocalToWorldTransform(body_prim)
    body_transform = Gf.Transform(body_world)
    initial_position = body_transform.GetTranslation()
    initial_quaternion = body_transform.GetRotation().GetQuat()
    initial_imaginary = initial_quaternion.GetImaginary()

    joint = UsdPhysics.FixedJoint.Define(stage, PLATFORM_LOCK_JOINT_PATH)
    joint.CreateBody1Rel().SetTargets([Sdf.Path(PLATFORM_LOCK_BODY_PATH)])
    joint.CreateLocalPos0Attr(
        Gf.Vec3f(
            float(initial_position[0]),
            float(initial_position[1]),
            float(initial_position[2]),
        )
    )
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr(
        Gf.Quatf(
            float(initial_quaternion.GetReal()),
            Gf.Vec3f(
                float(initial_imaginary[0]),
                float(initial_imaginary[1]),
                float(initial_imaginary[2]),
            ),
        )
    )
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateJointEnabledAttr(False)
    return joint


class FlightController:
    """Track a world-frame ``(x, y, z, yaw)`` target with a body wrench.

    The collected USD's PhysX articulation root link is ``Link3``.  Flight state
    must therefore be read explicitly from the rigid body named ``body`` instead
    of from ``robot.data.root_*``.
    """

    def __init__(
        self,
        robot,
        dt: float,
        target=(0.0, 0.0, 1.5, 0.0),
        lock_joint=None,
    ):
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
        # Integral compensation removes the residual position bias caused by
        # tilt/thrust allocation and actuator discretization.
        self.k_int = torch.tensor((0.30, 0.30, 0.50), device=self.device)
        self.k_rot = torch.tensor((300.0, 300.0, 300.0), device=self.device)
        self.k_angvel = torch.tensor((125.0, 125.0, 125.0), device=self.device)
        self.position_integral = torch.zeros((robot.num_instances, 3), device=self.device)
        self.integral_limit = 2.0
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
        self.lock_joint = lock_joint
        self._platform_locked = False
        self._locked_body_pose = None

    def _to_torch(self, array) -> torch.Tensor:
        value = getattr(array, "torch", array)
        return value.to(self.device) if torch.is_tensor(value) else torch.as_tensor(value, device=self.device)

    @property
    def mass_kg(self) -> float:
        return float(self.mass[0].item())

    @property
    def target_position(self) -> torch.Tensor:
        return self.command[0, :3]

    @property
    def platform_locked(self) -> bool:
        return self._platform_locked

    @property
    def locked_body_pose(self) -> torch.Tensor | None:
        return self._locked_body_pose

    def set_target(self, position, yaw: float = 0.0) -> None:
        if self._platform_locked:
            raise RuntimeError("drone target is locked; unlock the platform first")
        new_position = torch.as_tensor(
            position,
            device=self.device,
            dtype=torch.float32,
        )
        changed_axes = torch.abs(self.command[:, :3] - new_position) > 1.0e-6
        integral_decay = torch.where(
            changed_axes,
            torch.full_like(self.position_integral, 0.25),
            torch.ones_like(self.position_integral),
        )
        self.position_integral.mul_(integral_decay)
        self.command[:, :3] = new_position
        self.command[:, 3] = float(yaw)
        # Preserve integral compensation on axes whose command did not change.
        # This is important during the final vertical approach: repeated z-only
        # ROS commands must not discard the x/y force needed to resist contact
        # friction at the ceiling.

    def lock_platform(self) -> None:
        """Enable a FixedJoint whose anchor frame matches the physical body pose."""

        if self._platform_locked:
            return
        if self.lock_joint is None:
            raise RuntimeError("platform FixedJoint was not configured")
        self._locked_body_pose = self._to_torch(
            self.robot.data.body_link_pose_w
        )[:, self.body_id].clone()
        # Keep the rotors and flight controller active without letting the
        # locked x/y residual wind up against the platform constraint. Retain
        # the commanded ceiling preload only in z.
        self.command[:, :2] = self._locked_body_pose[:, :2]
        self.position_integral.zero_()

        # With body0 omitted, joint frame 0 is expressed directly in world.
        # Set it to the captured PhysX body pose while frame 1 remains the body
        # origin, so enabling the joint introduces no positional snap.
        from pxr import Gf

        local_position = self._locked_body_pose[0, :3].detach().cpu().tolist()
        local_quaternion = self._locked_body_pose[0, 3:].detach().cpu().tolist()
        self.lock_joint.GetLocalPos0Attr().Set(Gf.Vec3f(*local_position))
        self.lock_joint.GetLocalRot0Attr().Set(
            Gf.Quatf(
                float(local_quaternion[3]),
                Gf.Vec3f(*local_quaternion[:3]),
            )
        )
        self.lock_joint.GetJointEnabledAttr().Set(False)
        # Remove only the residual vehicle root velocity at engagement. From
        # this point onward PhysX resolves all rotor, arm, and contact forces
        # through the joint instead of a per-frame pose teleport.
        self.robot.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(
                (self.robot.num_instances, 6),
                device=self.device,
                dtype=self._locked_body_pose.dtype,
            )
        )
        self.lock_joint.GetJointEnabledAttr().Set(True)
        self._platform_locked = True

    def unlock_platform(self) -> None:
        """Release the articulation root without stopping the flight controller."""

        if not self._platform_locked:
            return
        if self.lock_joint is None:
            raise RuntimeError("platform FixedJoint was not configured")
        current_pose = self._to_torch(
            self.robot.data.body_link_pose_w
        )[:, self.body_id].clone()
        self.lock_joint.GetJointEnabledAttr().Set(False)
        self._platform_locked = False
        self._locked_body_pose = None
        self.command[:, :3] = current_pose[:, :3]
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

        if not self._platform_locked:
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
