"""Ceiling-only passive bulb joint and software release state machine."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


CEILING_BULB_ROOT_PATH = "/World/layout/target_bulb"
CEILING_SOCKET_PATH = f"{CEILING_BULB_ROOT_PATH}/Socket"
CEILING_BULB_PATH = f"{CEILING_BULB_ROOT_PATH}/Bulb"
CEILING_JOINT_PATH = f"{CEILING_BULB_ROOT_PATH}/ceiling_screw_joint"

JOINT_LIMIT_DEG = 180.0
RELEASE_STROKE_DEG = 180.0
UNLOCK_TOLERANCE_DEG = 5.0
LOCKED_TOLERANCE_DEG = 1.0
INSTALL_ALIGNMENT_DURATION_S = 2.0


class CeilingBulbState(str, Enum):
    """Discrete states of the ceiling bulb's virtual screw engagement."""

    LOCKED = "LOCKED"
    UNSCREWING = "UNSCREWING"
    UNLOCKED = "UNLOCKED"
    SCREWING = "SCREWING"
    ALIGNING = "ALIGNING"


def _wrap_to_pi(angle_rad: float) -> float:
    return (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi


def relative_twist_z(socket_xyzw, bulb_xyzw) -> float:
    """Return the wrapped bulb rotation about the socket's local Z axis."""

    sx, sy, sz, sw = (float(value) for value in socket_xyzw)
    bx, by, bz, bw = (float(value) for value in bulb_xyzw)
    # q_relative = conjugate(q_socket) * q_bulb, both in xyzw order.
    cx, cy, cz, cw = -sx, -sy, -sz, sw
    relative_z = cw * bz + cx * by - cy * bx + cz * bw
    relative_w = cw * bw - cx * bx - cy * by - cz * bz
    return _wrap_to_pi(2.0 * math.atan2(relative_z, relative_w))


@dataclass
class CeilingBulbProgress:
    """Track a 0-180 degree logical screw state (0 loose, 180 locked)."""

    release_stroke_deg: float = RELEASE_STROKE_DEG
    unlock_tolerance_deg: float = UNLOCK_TOLERANCE_DEG
    locked_tolerance_deg: float = LOCKED_TOLERANCE_DEG

    def __post_init__(self) -> None:
        if not 0.0 < self.release_stroke_deg < 360.0:
            raise ValueError("release_stroke_deg must be between 0 and 360 degrees")
        if not 0.0 <= self.unlock_tolerance_deg < self.release_stroke_deg:
            raise ValueError("unlock_tolerance_deg must be smaller than release_stroke_deg")
        if not 0.0 <= self.locked_tolerance_deg < self.release_stroke_deg:
            raise ValueError("locked_tolerance_deg must be smaller than release_stroke_deg")
        self._release_stroke_rad = math.radians(self.release_stroke_deg)
        self._unlock_tolerance_rad = math.radians(self.unlock_tolerance_deg)
        self._locked_tolerance_rad = math.radians(self.locked_tolerance_deg)
        self.reset(0.0)

    def reset(self, wrapped_angle_rad: float) -> None:
        self.previous_wrapped_angle_rad = _wrap_to_pi(wrapped_angle_rad)
        self.loosened_angle_rad = 0.0
        self.rejected_tightening = False
        self.state = CeilingBulbState.LOCKED

    def engage_loose(self, wrapped_angle_rad: float) -> None:
        """Re-engage an already grasped bulb at the loose screw endpoint."""

        if self.state is not CeilingBulbState.UNLOCKED:
            raise RuntimeError("ceiling bulb can only be re-engaged from UNLOCKED")
        self.previous_wrapped_angle_rad = _wrap_to_pi(wrapped_angle_rad)
        self.loosened_angle_rad = self._release_stroke_rad
        self.rejected_tightening = False
        self.state = CeilingBulbState.SCREWING

    def complete_removal(self, wrapped_angle_rad: float) -> None:
        """Finalize the completed removal sequence at the loose endpoint.

        The real contact simulation can under-report wrist rotation when the
        bulb is already constrained by the gripper.  The replacement command
        sequence is therefore the authority that removal has completed.
        """

        self.previous_wrapped_angle_rad = _wrap_to_pi(wrapped_angle_rad)
        self.loosened_angle_rad = self._release_stroke_rad
        self.rejected_tightening = False
        self.state = CeilingBulbState.UNLOCKED

    def tighten_step(self, degrees: float) -> None:
        """Advance deterministic installation progress after one wrist stroke."""

        if self.state is not CeilingBulbState.SCREWING:
            raise RuntimeError("ceiling bulb must be SCREWING before tightening")
        degrees = float(degrees)
        if not math.isfinite(degrees) or degrees <= 0.0:
            raise ValueError("tightening degrees must be positive and finite")
        self.loosened_angle_rad = max(
            0.0,
            self.loosened_angle_rad - math.radians(degrees),
        )
        if self.loosened_angle_rad <= self._locked_tolerance_rad:
            self.loosened_angle_rad = 0.0
            self.state = CeilingBulbState.LOCKED

    @property
    def remaining_angle_deg(self) -> float:
        return math.degrees(max(0.0, self._release_stroke_rad - self.loosened_angle_rad))

    @property
    def loosened_angle_deg(self) -> float:
        return math.degrees(self.loosened_angle_rad)

    @property
    def logical_angle_deg(self) -> float:
        return self.release_stroke_deg - self.loosened_angle_deg

    def update(self, wrapped_angle_rad: float) -> CeilingBulbState:
        if self.state is CeilingBulbState.UNLOCKED:
            return self.state

        wrapped_angle_rad = _wrap_to_pi(wrapped_angle_rad)
        delta = _wrap_to_pi(wrapped_angle_rad - self.previous_wrapped_angle_rad)
        self.previous_wrapped_angle_rad = wrapped_angle_rad
        # Installation progress is committed explicitly by tighten_step().
        # Do not count the same wrist stroke again from the physical joint
        # observation, otherwise one motion advances the logical screw twice.
        if self.state is CeilingBulbState.SCREWING:
            return self.state
        # The software coordinate starts at 0 degrees locked. Negative physical
        # rotation is interpreted as loosening and accumulates toward 180 deg.
        candidate = self.loosened_angle_rad - delta
        self.rejected_tightening = candidate < 0.0
        self.loosened_angle_rad = min(
            self._release_stroke_rad,
            max(0.0, candidate),
        )
        remaining = self._release_stroke_rad - self.loosened_angle_rad
        if self.state is CeilingBulbState.SCREWING:
            if self.loosened_angle_rad <= self._locked_tolerance_rad:
                self.state = CeilingBulbState.LOCKED
            else:
                self.state = CeilingBulbState.SCREWING
        elif remaining <= self._unlock_tolerance_rad:
            self.state = CeilingBulbState.UNLOCKED
        elif self.loosened_angle_rad <= self._locked_tolerance_rad:
            self.state = CeilingBulbState.LOCKED
        else:
            self.state = CeilingBulbState.UNSCREWING
        return self.state


def define_ceiling_bulb_joint(stage, verify_velocity_deg_s: float | None = None):
    """Author the passive joint only on the ceiling bulb stage instance."""

    from pxr import Gf, Sdf, UsdPhysics

    for prim_path in (CEILING_SOCKET_PATH, CEILING_BULB_PATH):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            raise RuntimeError(f"Ceiling bulb rigid body not found: {prim_path}")
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise RuntimeError(f"Ceiling bulb prim is not a rigid body: {prim_path}")

    joint = UsdPhysics.RevoluteJoint.Define(stage, CEILING_JOINT_PATH)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(CEILING_SOCKET_PATH)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(CEILING_BULB_PATH)])
    joint.CreateAxisAttr(UsdPhysics.Tokens.z)
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
    # Use the equivalent physical interval [-180, 0] because the assembled
    # socket/bulb frames start at physical zero. Software maps this to the
    # requested logical range 180 (locked) down to 0 (loose).
    joint.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLowerLimitAttr().Set(-180.0)
    joint.CreateUpperLimitAttr().Set(0.0)
    drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
    drive.CreateTypeAttr(UsdPhysics.Tokens.force)
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(20.0)
    drive.CreateMaxForceAttr(1000.0)
    drive.CreateTargetVelocityAttr(0.0)
    joint.CreateCollisionEnabledAttr(False)
    joint.CreateJointEnabledAttr(True)

    if verify_velocity_deg_s is not None:
        drive = UsdPhysics.DriveAPI.Apply(joint.GetPrim(), "angular")
        drive.CreateTypeAttr(UsdPhysics.Tokens.force)
        drive.CreateStiffnessAttr(0.0)
        drive.CreateDampingAttr(10.0)
        drive.CreateMaxForceAttr(100.0)
        drive.CreateTargetVelocityAttr(float(verify_velocity_deg_s))

    return joint


class CeilingBulbMechanism:
    """Observe removal and re-engage the same gripper-held bulb."""

    def __init__(self, joint, socket, bulb):
        self.joint = joint
        self.socket = socket
        self.bulb = bulb
        self.progress = CeilingBulbProgress()
        self._reported_state = None
        self._initial_root_pose_w = None
        self._alignment_start_pose_w = None
        self._alignment_elapsed_s = 0.0

    @staticmethod
    def _quaternion_xyzw(rigid_object):
        value = rigid_object.data.root_quat_w
        value = getattr(value, "torch", value)
        value = value[0]
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        return tuple(float(component) for component in value)

    def _wrapped_angle(self) -> float:
        return relative_twist_z(
            self._quaternion_xyzw(self.socket),
            self._quaternion_xyzw(self.bulb),
        )

    @staticmethod
    def _position_xyz(rigid_object) -> tuple[float, float, float]:
        value = rigid_object.data.root_pos_w
        value = getattr(value, "torch", value)
        value = value[0]
        if hasattr(value, "detach"):
            value = value.detach().cpu().tolist()
        return tuple(float(component) for component in value)

    @property
    def separation_m(self) -> float:
        socket_position = self._position_xyz(self.socket)
        bulb_position = self._position_xyz(self.bulb)
        return math.sqrt(
            sum(
                (socket_coordinate - bulb_coordinate) ** 2
                for socket_coordinate, bulb_coordinate in zip(
                    socket_position, bulb_position
                )
            )
        )

    def _set_bulb_kinematic(self, enabled: bool) -> None:
        from pxr import UsdPhysics

        stage = self.joint.GetPrim().GetStage()
        bulb_prim = stage.GetPrimAtPath(CEILING_BULB_PATH)
        if not bulb_prim.IsValid():
            raise RuntimeError(f"ceiling bulb rigid body not found: {CEILING_BULB_PATH}")
        rigid_body = UsdPhysics.RigidBodyAPI(bulb_prim)
        rigid_body.CreateKinematicEnabledAttr().Set(bool(enabled))

    def _root_pose_tensor(self):
        value = self.bulb.data.root_pose_w
        return getattr(value, "torch", value)

    @staticmethod
    def _pose_components(pose_tensor) -> tuple[float, ...]:
        values = pose_tensor[0]
        if hasattr(values, "detach"):
            values = values.detach().cpu().tolist()
        return tuple(float(value) for value in values)

    @property
    def initial_root_pose_xyzw(self) -> tuple[float, ...] | None:
        if self._initial_root_pose_w is None:
            return None
        return self._pose_components(self._initial_root_pose_w)

    @property
    def current_root_pose_xyzw(self) -> tuple[float, ...]:
        return self._pose_components(self._root_pose_tensor())

    def reset(self) -> None:
        self._set_bulb_kinematic(False)
        self.joint.CreateJointEnabledAttr(True).Set(True)
        self.progress.reset(self._wrapped_angle())
        self._initial_root_pose_w = self._root_pose_tensor().clone()
        self._alignment_start_pose_w = None
        self._alignment_elapsed_s = 0.0
        self._reported_state = None
        self._report_state()

    def engage_loose(self) -> None:
        """Start installation without recreating a snapping physical joint."""

        wrapped_angle_rad = self._wrapped_angle()
        self.joint.GetJointEnabledAttr().Set(False)
        self._set_bulb_kinematic(False)
        self.progress.engage_loose(wrapped_angle_rad)
        self._report_state()

    def resume_grasp(self) -> None:
        """Release the temporary hold after the gripper has closed again."""

        if self.progress.state is not CeilingBulbState.SCREWING:
            raise RuntimeError("ceiling bulb must be SCREWING before resuming grasp")
        self._set_bulb_kinematic(False)

    def complete_removal(self) -> None:
        """Release the original bulb after the replacement sequence finishes."""

        self._set_bulb_kinematic(False)
        self.progress.complete_removal(self._wrapped_angle())
        self.joint.GetJointEnabledAttr().Set(False)
        self._report_state()

    def tighten_step(self, degrees: float) -> None:
        """Record the screw progress produced by a completed wrist stroke."""

        self.progress.tighten_step(degrees)
        self.progress.previous_wrapped_angle_rad = self._wrapped_angle()
        # Hold the bulb at the completed stroke while the gripper opens and
        # returns to its reset pose.  At LOCKED this becomes the final ceiling
        # attachment, avoiding a runtime joint recreation and its snap impulse.
        self._set_bulb_kinematic(True)
        if self.progress.state is CeilingBulbState.LOCKED:
            self._alignment_start_pose_w = self._root_pose_tensor().clone()
            self._alignment_elapsed_s = 0.0
            self.progress.state = CeilingBulbState.ALIGNING
        self._report_state()

    def _report_state(self) -> None:
        if self.progress.state is self._reported_state:
            return
        self._reported_state = self.progress.state
        print(
            "[Rflyarm][CEILING_BULB] state=%s angle=%.1fdeg loosened=%.1fdeg remaining=%.1fdeg"
            % (
                self.progress.state.value,
                self.progress.logical_angle_deg,
                self.progress.loosened_angle_deg,
                self.progress.remaining_angle_deg,
            )
        )

    def _advance_alignment(self, dt: float) -> None:
        if self._initial_root_pose_w is None or self._alignment_start_pose_w is None:
            raise RuntimeError("bulb alignment poses are not initialized")
        import torch

        self._alignment_elapsed_s += max(float(dt), 0.0)
        alpha = min(1.0, self._alignment_elapsed_s / INSTALL_ALIGNMENT_DURATION_S)
        start = self._alignment_start_pose_w
        target = self._initial_root_pose_w
        start_quaternion = start[:, 3:7]
        target_quaternion = target[:, 3:7]
        if bool(torch.sum(start_quaternion * target_quaternion).item() < 0.0):
            target_quaternion = -target_quaternion
        pose = start * (1.0 - alpha) + target.clone() * alpha
        pose[:, 3:7] = (1.0 - alpha) * start_quaternion + alpha * target_quaternion
        pose[:, 3:7] /= torch.linalg.vector_norm(
            pose[:, 3:7], dim=1, keepdim=True
        ).clamp_min(1.0e-8)
        self.bulb.write_root_pose_to_sim_index(root_pose=pose)
        if alpha >= 1.0:
            self.progress.state = CeilingBulbState.LOCKED
            self._report_state()

    def update(self, dt: float = 0.0) -> None:
        if self.progress.state is CeilingBulbState.ALIGNING:
            self._advance_alignment(dt)
            return
        previous_state = self.progress.state
        state = self.progress.update(self._wrapped_angle())
        if state is CeilingBulbState.UNLOCKED and previous_state is not state:
            self.joint.GetJointEnabledAttr().Set(False)
        self._report_state()
