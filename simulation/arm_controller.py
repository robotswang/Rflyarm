"""Isaac Lab joint-position controller for the Rflyarm arm and gripper."""

from __future__ import annotations

import torch
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from simulation.arm_kinematics import ArmKinematics


ARM_JOINT_NAMES = [f"joint_{index}" for index in range(1, 7)]
GRIPPER_MASTER_NAME = "gripper_r1"
# Isaac Lab exposes the first three coordinates with the opposite sign from Lula/URDF.
KINEMATICS_JOINT_SIGNS = (-1.0, -1.0, -1.0, 1.0, 1.0, 1.0)


class ArmController:
    """Own the arm position targets and apply slew limiting every physics step."""

    def __init__(
        self,
        robot,
        max_speed: float = 1.5,
        cartesian_speed: float = 0.15,
        cartesian_angular_speed: float = np.deg2rad(10.0),
    ):
        self.robot = robot
        self.device = robot.device
        ids, names = robot.find_joints(ARM_JOINT_NAMES, preserve_order=True)
        if names != ARM_JOINT_NAMES:
            raise RuntimeError(f"Arm joints do not match the expected order: {names}")
        self.joint_ids = torch.tensor(ids, device=self.device, dtype=torch.int32)
        self.kinematics_joint_signs = torch.tensor(
            KINEMATICS_JOINT_SIGNS, device=self.device, dtype=torch.float32
        )
        measured = self._positions()
        self.commanded = measured.clone()
        self.target = measured.clone()
        self.max_speed = float(max_speed)

        gripper_ids, gripper_names = robot.find_joints([GRIPPER_MASTER_NAME], preserve_order=True)
        if gripper_names != [GRIPPER_MASTER_NAME]:
            raise RuntimeError(f"Gripper master joint not found: {gripper_names}")
        self.gripper_id = torch.tensor(gripper_ids, device=self.device, dtype=torch.int32)
        self.gripper_target = torch.full_like(self._all_positions()[:, self.gripper_id], 0.5)
        self.kinematics = ArmKinematics()
        self.last_ik_solution = None
        self.cartesian_speed = float(cartesian_speed)
        self.cartesian_angular_speed = float(cartesian_angular_speed)
        if self.cartesian_speed <= 0.0:
            raise ValueError("cartesian_speed must be positive")
        if self.cartesian_angular_speed <= 0.0:
            raise ValueError("cartesian_angular_speed must be positive")
        self._cartesian_start = None
        self._cartesian_goal = None
        self._cartesian_elapsed = 0.0
        self._cartesian_duration = 0.0
        self._last_cartesian_position = None

    def _proxy_to_torch(self, array) -> torch.Tensor:
        value = getattr(array, "torch", array)
        return value.to(self.device) if torch.is_tensor(value) else torch.as_tensor(value, device=self.device)

    def _all_positions(self) -> torch.Tensor:
        return self._proxy_to_torch(self.robot.data.joint_pos)

    def _positions(self) -> torch.Tensor:
        return self._all_positions()[:, self.joint_ids]

    def set_joint_target(self, target) -> None:
        value = torch.as_tensor(target, device=self.device, dtype=torch.float32)
        if value.numel() != 6:
            raise ValueError("Arm target must contain joint_1 through joint_6")
        self.target[:] = value.reshape(1, 6)

    def set_named_targets(self, names, positions) -> None:
        if len(names) != len(positions):
            raise ValueError("Joint command name and position arrays must have equal length")
        for name, position in zip(names, positions):
            if name in ARM_JOINT_NAMES:
                self.target[:, ARM_JOINT_NAMES.index(name)] = float(position)
            elif name == "gripper":
                command = max(0.0, min(1.0, float(position)))
                self.gripper_target[:] = -1.0 + 1.5 * command
            else:
                raise ValueError(f"Unknown joint name: {name}")

    def set_cartesian_target(self, frame_id, position, quaternion_xyzw):
        """Start a smooth Cartesian trajectory to a validated full-pose target."""
        position, quaternion_xyzw = self.kinematics.validate_pose(
            frame_id, position, quaternion_xyzw)
        measured = (
            self._positions()[0] * self.kinematics_joint_signs
        ).detach().cpu().numpy()
        fallback_seeds = []
        if self.last_ik_solution is not None:
            fallback_seeds.append(self.last_ik_solution)
        fallback_seeds.append(torch.zeros(6).numpy())
        result = self.kinematics.solve(
            frame_id=frame_id,
            position=position,
            quaternion_xyzw=quaternion_xyzw,
            warm_start=measured,
            fallback_seeds=fallback_seeds,
        )
        # The endpoint solve only validates reachability.  Do not use that
        # endpoint configuration as the seed for the first interpolated pose:
        # it can belong to a different IK branch from the live configuration.
        # Start the trajectory from the measured configuration instead.
        self.last_ik_solution = measured.copy()
        start_position, start_quaternion = self.end_effector_pose()
        distance = float(np.linalg.norm(position - start_position))
        # Consecutive commands with the same Cartesian position define a pure
        # attitude transition. Keep translation exactly constant instead of
        # re-planning from the current tracking residual.
        if self._last_cartesian_position is not None and np.array_equal(
            position, self._last_cartesian_position):
            start_position = position.copy()
        self._cartesian_start = (start_position.copy(), start_quaternion.copy())
        self._cartesian_goal = (position.copy(), quaternion_xyzw.copy())
        self._cartesian_elapsed = 0.0
        angular_distance = float(
            (
                Rotation.from_quat(start_quaternion).inv()
                * Rotation.from_quat(quaternion_xyzw)
            ).magnitude()
        )
        # Bound translation and rotation independently.  The previous fixed
        # two-second duration ignored rotation angle, so a 60-degree wrist
        # stroke advanced faster than the physical joint drives could track
        # synchronously and converted their lag into tool-center translation.
        self._cartesian_duration = max(
            distance / self.cartesian_speed,
            angular_distance / self.cartesian_angular_speed,
            0.1,
        )
        self._last_cartesian_position = position.copy()
        return result

    def update(self, dt: float) -> None:
        if self._cartesian_goal is not None:
            self._cartesian_elapsed = min(
                self._cartesian_elapsed + float(dt), self._cartesian_duration)
            alpha = self._cartesian_elapsed / self._cartesian_duration
            start_pos, start_quat = self._cartesian_start
            goal_pos, goal_quat = self._cartesian_goal
            position = (1.0 - alpha) * start_pos + alpha * goal_pos
            rotations = Rotation.from_quat(np.stack((start_quat, goal_quat)))
            slerp = Slerp([0.0, 1.0], rotations)
            quaternion = slerp([alpha]).as_quat()[0]
            measured = (self._positions()[0] * self.kinematics_joint_signs).detach().cpu().numpy()
            try:
                # Prefer the measured configuration so every intermediate IK
                # solve stays on the branch physically being tracked.  The
                # previous solution remains a fallback for solver robustness.
                warm_start = measured
                fallback_seeds = (
                    self.last_ik_solution,
                ) if self.last_ik_solution is not None else ()
                result = self.kinematics.solve(
                    "base_link", position, quaternion, warm_start,
                    fallback_seeds=fallback_seeds)
                self.last_ik_solution = result.joint_positions.copy()
                self.target[:] = torch.as_tensor(
                    result.joint_positions, device=self.device, dtype=torch.float32
                ).reshape(1, 6) * self.kinematics_joint_signs
            except RuntimeError:
                pass
            if alpha >= 1.0:
                self._cartesian_goal = None
        # Keep a joint-rate safety bound even during Cartesian trajectories.
        # Normally IK changes are below this limit; it prevents a rare branch
        # switch or solver glitch from becoming a one-frame physical jump.
        max_delta = self.max_speed * float(dt)
        delta = torch.clamp(self.target - self.commanded, -max_delta, max_delta)
        self.commanded += delta
        self.robot.set_joint_position_target_index(target=self.commanded, joint_ids=self.joint_ids)
        self.robot.set_joint_position_target_index(target=self.gripper_target, joint_ids=self.gripper_id)

    def position_errors(self, target) -> torch.Tensor:
        desired = torch.as_tensor(target, device=self.device, dtype=torch.float32).reshape(1, 6)
        return (self._positions() - desired)[0]

    def joint_state(self):
        """Return public joint names plus measured position, velocity and effort."""
        positions = self._all_positions()[0]
        velocities = self._proxy_to_torch(self.robot.data.joint_vel)[0]
        try:
            efforts = self._proxy_to_torch(self.robot.data.applied_torque)[0]
        except Exception:
            efforts = torch.zeros_like(positions)
        ids = self.joint_ids.to(dtype=torch.long)
        gripper_id = self.gripper_id.to(dtype=torch.long)
        gripper_position = torch.clamp((positions[gripper_id] + 1.0) / 1.5, 0.0, 1.0)
        public_names = ARM_JOINT_NAMES + ["gripper"]
        public_positions = torch.cat((positions[ids], gripper_position))
        public_velocities = torch.cat((velocities[ids], velocities[gripper_id]))
        public_efforts = torch.cat((efforts[ids], efforts[gripper_id]))
        return (
            public_names,
            public_positions.detach().cpu().tolist(),
            public_velocities.detach().cpu().tolist(),
            public_efforts.detach().cpu().tolist(),
        )

    def end_effector_pose(self):
        measured = (
            self._positions()[0] * self.kinematics_joint_signs
        ).detach().cpu().numpy()
        return self.kinematics.forward(measured)
