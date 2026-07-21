#!/usr/bin/env python
"""Lula FK/IK helper for the Rflyarm arm.

This module deliberately contains no ROS node or PhysX write. ``ArmController``
owns all command arbitration and is the only component allowed to write joint
targets. Keeping Lula as a pure numerical helper prevents the Cartesian and
joint interfaces from fighting each other.

All poses handled here are expressed in the arm's ``base_link`` frame.  The
solver base therefore remains at Lula's default identity pose; platform motion
in the world frame is intentionally irrelevant.
"""

from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


ARM_JOINT_NAMES = ("joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6")
BASE_FRAME = "base_link"
EE_FRAME = "tool_center"

_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URDF_PATH = str(_REPO_ROOT / "assets/kinematics/arm.urdf")
DEFAULT_ROBOT_DESCRIPTION_PATH = str(
    _REPO_ROOT / "assets/kinematics/robot_description.yaml")


@dataclass(frozen=True)
class IKResult:
    """A validated Lula solution and its FK round-trip residuals."""

    joint_positions: np.ndarray
    position_error_m: float
    orientation_error_rad: float


class ArmKinematics:
    """Validated FK/IK wrapper around the Lula solver bundled with Isaac Sim 6.0.1."""

    def __init__(
        self,
        robot_description_path: str = DEFAULT_ROBOT_DESCRIPTION_PATH,
        urdf_path: str = DEFAULT_URDF_PATH,
        base_frame: str = BASE_FRAME,
        ee_frame: str = EE_FRAME,
        solver_position_tolerance_m: float = 0.002,
        solver_orientation_tolerance_rad: float = np.deg2rad(1.0),
        acceptance_position_error_m: float = 0.005,
        acceptance_orientation_error_rad: float = np.deg2rad(2.0),
        safe_joint_limit_rad: float = 3.10,
    ):
        self.robot_description_path = os.path.abspath(robot_description_path)
        self.urdf_path = os.path.abspath(urdf_path)
        self.base_frame = str(base_frame)
        self.ee_frame = str(ee_frame)
        self.solver_position_tolerance_m = float(solver_position_tolerance_m)
        self.solver_orientation_tolerance_rad = float(solver_orientation_tolerance_rad)
        self.acceptance_position_error_m = float(acceptance_position_error_m)
        self.acceptance_orientation_error_rad = float(acceptance_orientation_error_rad)
        self.safe_joint_limit_rad = float(safe_joint_limit_rad)

        self._lula = None
        self._robot_description = None
        self._kinematics = None
        self._lower_limits = None
        self._upper_limits = None

    @property
    def loaded(self) -> bool:
        return self._kinematics is not None

    @staticmethod
    def _as_finite_vector(values, size: int, label: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.shape != (size,):
            raise ValueError("%s must contain exactly %d values" % (label, size))
        if not np.all(np.isfinite(vector)):
            raise ValueError("%s contains NaN or infinity" % label)
        return vector

    @staticmethod
    def _wrap_to_pi(joints: np.ndarray) -> np.ndarray:
        return (np.asarray(joints, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi

    def validate_pose(self, frame_id, position, quaternion_xyzw):
        """Validate and normalize a ROS Cartesian target.

        A frame other than ``base_link`` is rejected instead of being silently
        interpreted in the wrong coordinate system.
        """

        if str(frame_id) != self.base_frame:
            raise ValueError(
                "frame_id must be '%s' (received '%s')" % (self.base_frame, frame_id))
        position = self._as_finite_vector(position, 3, "position")
        quaternion = self._as_finite_vector(quaternion_xyzw, 4, "quaternion")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1.0e-8:
            raise ValueError("quaternion norm is zero")
        return position, quaternion / norm

    def load(self):
        """Load and verify the Isaac Sim 6.0.1 Lula model on first use."""

        if self._kinematics is not None:
            return
        for path, label in (
            (self.robot_description_path, "robot description"),
            (self.urdf_path, "URDF"),
        ):
            if not os.path.isfile(path):
                raise FileNotFoundError("%s file not found: %s" % (label, path))

        # Use the Lula numerical library bundled with Isaac Sim 6 directly.  This
        # keeps FK/IK independent of the deprecated ``isaacsim.core.api`` layer.
        isaac_sim_root = Path(sys.executable).resolve().parents[3]
        lula_extension = isaac_sim_root / "extsDeprecated" / "isaacsim.robot_motion.lula"
        prebundle_path = lula_extension / "pip_prebundle"
        if not prebundle_path.exists():
            raise FileNotFoundError(f"Isaac Sim 6 Lula component not found: {prebundle_path}")
        prebundle = str(prebundle_path)
        if prebundle not in sys.path:
            sys.path.insert(0, prebundle)
        import lula

        robot_description = lula.load_robot(
            self.robot_description_path, self.urdf_path
        )
        kinematics = robot_description.kinematics()
        joint_names = tuple(
            robot_description.c_space_coord_name(index)
            for index in range(robot_description.num_c_space_coords())
        )
        if joint_names != ARM_JOINT_NAMES:
            raise RuntimeError(
                "Lula cspace mismatch: expected %s, received %s" %
                (ARM_JOINT_NAMES, joint_names))
        frames = tuple(kinematics.frame_names())
        if self.base_frame not in frames or self.ee_frame not in frames:
            raise RuntimeError(
                "Lula frames must include '%s' and '%s': %s" %
                (self.base_frame, self.ee_frame, frames))

        limits = [
            kinematics.c_space_coord_limits(index)
            for index in range(kinematics.num_c_space_coords())
        ]
        lower = np.asarray([limit.lower for limit in limits], dtype=np.float64)
        upper = np.asarray([limit.upper for limit in limits], dtype=np.float64)
        # The verified joint controller intentionally keeps every arm joint a
        # small margin inside +/-pi.  IK uses the identical executable limits.
        self._lower_limits = np.maximum(lower, -self.safe_joint_limit_rad)
        self._upper_limits = np.minimum(upper, self.safe_joint_limit_rad)
        self._lula = lula
        self._robot_description = robot_description
        self._kinematics = kinematics

    def forward(self, joint_positions):
        """Return ``(position, quaternion_xyzw)`` for ``tool_center``."""

        return self.forward_frame(self.ee_frame, joint_positions)

    def forward_frame(self, frame_name, joint_positions):
        """Return a named Lula frame pose relative to ``base_link``."""

        self.load()
        if str(frame_name) not in self._kinematics.frame_names():
            raise ValueError("unknown Lula frame: %s" % frame_name)
        joints = self._as_finite_vector(joint_positions, 6, "joint_positions")
        pose = self._kinematics.pose(np.expand_dims(joints, 1), str(frame_name))
        position = pose.translation
        rotation = pose.rotation.matrix()
        position = self._as_finite_vector(position, 3, "FK position")
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise RuntimeError("Lula FK returned an invalid rotation matrix")
        quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
        return position, quaternion_xyzw

    def _candidate_residual(self, joints, target_position, target_quaternion_xyzw):
        fk_position, fk_quaternion = self.forward(joints)
        position_error = float(np.linalg.norm(fk_position - target_position))
        relative_rotation = (
            Rotation.from_quat(target_quaternion_xyzw).inv()
            * Rotation.from_quat(fk_quaternion)
        )
        orientation_error = float(relative_rotation.magnitude())
        return position_error, orientation_error

    def solve(self, frame_id, position, quaternion_xyzw, warm_start, fallback_seeds=()):
        """Solve a full-pose target and return the closest validated solution.

        The live measured configuration is supplied as ``warm_start``.  Optional
        prior/home configurations are tried as fallbacks.  Lula success alone is
        not enough: every candidate is checked for finite values, executable
        joint limits, and FK round-trip error before it can become a target.
        """

        target_position, target_quaternion = self.validate_pose(
            frame_id, position, quaternion_xyzw)
        self.load()

        reference = self._wrap_to_pi(
            self._as_finite_vector(warm_start, 6, "warm_start"))
        seeds = [reference]
        for seed in fallback_seeds:
            try:
                seed = self._wrap_to_pi(
                    self._as_finite_vector(seed, 6, "fallback seed"))
            except ValueError:
                continue
            if not any(np.linalg.norm(seed - item) < 1.0e-9 for item in seeds):
                seeds.append(seed)

        valid_results = []
        failure_notes = []
        for seed in seeds:
            try:
                target_rotation = Rotation.from_quat(target_quaternion).as_matrix()
                target_pose = self._lula.Pose3(
                    self._lula.Rotation3(target_rotation), target_position
                )
                config = self._lula.CyclicCoordDescentIkConfig()
                config.position_tolerance = self.solver_position_tolerance_m
                config.orientation_tolerance = 2.0 * np.sin(
                    0.5 * self.solver_orientation_tolerance_rad
                )
                config.cspace_seeds = [seed]
                solver_result = self._lula.compute_ik_ccd(
                    self._kinematics, target_pose, self.ee_frame, config
                )
                candidate = solver_result.cspace_position
                success = solver_result.success
            except Exception as exc:
                failure_notes.append("solver exception: %s" % exc)
                continue
            if not success:
                failure_notes.append("Lula did not converge from one seed")
                continue
            try:
                candidate = self._wrap_to_pi(
                    self._as_finite_vector(candidate, 6, "IK solution"))
            except ValueError as exc:
                failure_notes.append(str(exc))
                continue
            if (np.any(candidate < self._lower_limits - 1.0e-8)
                    or np.any(candidate > self._upper_limits + 1.0e-8)):
                failure_notes.append("solution violates executable joint limits")
                continue
            try:
                position_error, orientation_error = self._candidate_residual(
                    candidate, target_position, target_quaternion)
            except Exception as exc:
                failure_notes.append("FK validation failed: %s" % exc)
                continue
            if position_error > self.acceptance_position_error_m:
                failure_notes.append(
                    "position residual %.6f m exceeds %.6f m" %
                    (position_error, self.acceptance_position_error_m))
                continue
            if orientation_error > self.acceptance_orientation_error_rad:
                failure_notes.append(
                    "orientation residual %.6f rad exceeds %.6f rad" %
                    (orientation_error, self.acceptance_orientation_error_rad))
                continue
            distance_from_live = float(np.linalg.norm(
                self._wrap_to_pi(candidate - reference)))
            valid_results.append((
                distance_from_live,
                IKResult(candidate, position_error, orientation_error),
            ))

        if not valid_results:
            detail = "; ".join(failure_notes[-3:]) or "no valid solver result"
            raise RuntimeError("IK target rejected: " + detail)
        valid_results.sort(key=lambda item: item[0])
        return valid_results[0][1]
