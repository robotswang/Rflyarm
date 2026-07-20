#!/usr/bin/env python
"""Single-owner ROS 2 joint and Cartesian controller for Rflyarm.

The flight controller remains backends[0].  This backend is the *only* writer of
arm position targets and exposes the same public topic names/QoS as the verified
``~/rfly_arm`` setup:

* subscribe ``/joint_command`` (sensor_msgs/JointState)
* subscribe ``/arm/cmd_pose`` (geometry_msgs/PoseStamped, base_link)
* publish ``/joint_states`` (sensor_msgs/JointState, position/velocity/effort)
* publish ``/arm/ee_pose`` (geometry_msgs/PoseStamped, base_link -> tool_center)

The current CAD gripper has one revolute master DOF (``gripper_r1``).  Its public
ROS name is the mechanism-level ``gripper`` so the interface exposes exactly one
gripper field rather than implying two independently controllable fingers.
"""

import carb
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import JointState

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig
from simulation.arm_kinematics import (
    ARM_JOINT_NAMES,
    ArmKinematics,
    DEFAULT_ROBOT_DESCRIPTION_PATH,
    DEFAULT_URDF_PATH,
)


class _EmptyConfig(BackendConfig):
    pass


_ARM_JOINTS = list(ARM_JOINT_NAMES)
_GRIPPER_PUBLIC = "gripper"
_PUBLIC_JOINTS = _ARM_JOINTS + [_GRIPPER_PUBLIC]
_GRIPPER_MASTER = "gripper_r1"

# Public gripper convention matches the current revolute master: 0 rad=open,
# 0.5 rad=closed.
_GRIPPER_CLOSED_RAD = 0.5


class ArmController(Backend):
    """Drive one arm articulation from joint or Cartesian ROS 2 commands.

    Subscribes: /joint_command (sensor_msgs/JointState)
      msg.name     -- list of joint names to command
      msg.position -- desired positions [rad], matched positionally to name

    Subscribes: /arm/cmd_pose (geometry_msgs/PoseStamped)
      pose         -- desired tool_center pose relative to base_link
    """

    def __init__(self,
                 articulation_path: str = "/World/rflyarm",
                 joint_command_topic: str = "/joint_command",
                 joint_states_topic: str = "/joint_states",
                 target_pose_topic: str = "/arm/cmd_pose",
                 ee_pose_topic: str = "/arm/ee_pose",
                 robot_description_path: str = DEFAULT_ROBOT_DESCRIPTION_PATH,
                 urdf_path: str = DEFAULT_URDF_PATH,
                 base_frame: str = "base_link",
                 ee_frame: str = "tool_center",
                 usd_base_link_path: str = "/World/rflyarm/arm_geo/Geometry/base_link",
                 usd_link_root_path: str = "/World/rflyarm/arm_geo/Geometry",
                 alignment_debug: bool = False,
                 publish_hz: float = 60.0,
                 arm_max_speed: float = 1.5,
                 gripper_max_speed: float = 1.0,
                 node_name: str = "rflyarm_arm_controller"):
        super().__init__(_EmptyConfig())

        self._articulation_path = articulation_path
        self._targets = {}
        self._commanded = {}
        self._pending_commands = []
        self._last_ik_solution = None
        self._arm_control_mode = "joint"
        self._cb_counter = 0
        self._art = None
        self._dof_index = None
        self._dc = None
        self._state_all = None
        self._publish_period = 1.0 / max(float(publish_hz), 1.0)
        self._publish_accum = 0.0
        self._max_speed = {name: float(arm_max_speed) for name in _ARM_JOINTS}
        self._max_speed[_GRIPPER_MASTER] = float(gripper_max_speed)
        self._kinematics = ArmKinematics(
            robot_description_path=robot_description_path,
            urdf_path=urdf_path,
            base_frame=base_frame,
            ee_frame=ee_frame,
        )
        self._kinematics_error_logged = False
        self._usd_base_link_path = str(usd_base_link_path)
        self._usd_link_root_path = str(usd_link_root_path)
        self._alignment_debug = bool(alignment_debug)
        self._alignment_accum = 0.0
        self._last_alignment_q = None

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(node_name)
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        self._sub = self.node.create_subscription(
            JointState, joint_command_topic, self._cmd_cb, qos)
        carb.log_warn("[ArmController] subscribing joint commands on: " + joint_command_topic)

        self._js_pub = self.node.create_publisher(
            JointState, joint_states_topic, qos)
        carb.log_warn("[ArmController] publishing joint states on: " + joint_states_topic)
        self._pose_sub = self.node.create_subscription(
            PoseStamped, target_pose_topic, self._pose_cmd_cb, qos)
        carb.log_warn(
            "[ArmController] subscribing Cartesian targets on: " + target_pose_topic)
        self._ee_pub = self.node.create_publisher(
            PoseStamped, ee_pose_topic, qos)
        carb.log_warn(
            "[ArmController] publishing base-relative EE pose on: " + ee_pose_topic)

    def _cmd_cb(self, msg: JointState):
        self._cb_counter += 1
        if not msg.name:
            carb.log_warn("[ArmController] rejected command with empty name array")
            return
        if len(msg.name) != len(msg.position):
            carb.log_warn(
                "[ArmController] rejected command: name/position length mismatch (%d/%d)" %
                (len(msg.name), len(msg.position)))
            return

        accepted = {}
        for i in range(len(msg.name)):
            name = msg.name[i]
            value = float(msg.position[i])
            if not np.isfinite(value):
                carb.log_warn("[ArmController] rejected non-finite target for '%s'" % name)
                return
            if name in _ARM_JOINTS:
                # The USD hard limits are approximately +/-pi.  Keep a small safety margin.
                accepted[name] = float(np.clip(value, -3.10, 3.10))
            elif name == _GRIPPER_PUBLIC:
                accepted[_GRIPPER_MASTER] = float(np.clip(value, 0.0, _GRIPPER_CLOSED_RAD))
            else:
                carb.log_warn(
                    "[ArmController] ignoring unknown joint name '%s' "
                    "(expected one of %s)" % (name, _PUBLIC_JOINTS))

        if accepted:
            self._pending_commands.append(("joint", accepted))

    def _pose_cmd_cb(self, msg: PoseStamped):
        self._cb_counter += 1
        p = msg.pose.position
        o = msg.pose.orientation
        # Copy primitive values out of the ROS message.  Solving is deferred until
        # the articulation is available so the measured joints can seed Lula.
        command = {
            "frame_id": str(msg.header.frame_id),
            "position": np.array([p.x, p.y, p.z], dtype=np.float64),
            "quaternion_xyzw": np.array([o.x, o.y, o.z, o.w], dtype=np.float64),
        }
        self._pending_commands.append(("pose", command))

    def _measured_arm_positions(self):
        if self._art is None or self._dof_index is None:
            return None
        try:
            return np.array([
                float(self._dc.get_dof_position(self._dof_index[name]))
                for name in _ARM_JOINTS
            ], dtype=np.float64)
        except Exception:
            return None

    def _apply_pending_commands(self):
        if not self._pending_commands:
            return
        commands = self._pending_commands
        self._pending_commands = []
        for command_type, command in commands:
            if command_type == "joint":
                self._targets.update(command)
                if any(name in command for name in _ARM_JOINTS):
                    self._arm_control_mode = "joint"
                continue

            measured = self._measured_arm_positions()
            if measured is None:
                carb.log_warn(
                    "[ArmController] Cartesian target deferred: arm state unavailable")
                self._pending_commands.append((command_type, command))
                continue
            fallback_seeds = []
            if self._last_ik_solution is not None:
                fallback_seeds.append(self._last_ik_solution)
            fallback_seeds.append(np.zeros(6, dtype=np.float64))
            try:
                result = self._kinematics.solve(
                    frame_id=command["frame_id"],
                    position=command["position"],
                    quaternion_xyzw=command["quaternion_xyzw"],
                    warm_start=measured,
                    fallback_seeds=fallback_seeds,
                )
            except Exception as exc:
                carb.log_warn("[ArmController] " + str(exc))
                continue

            self._last_ik_solution = result.joint_positions.copy()
            self._targets.update(dict(zip(_ARM_JOINTS, result.joint_positions)))
            self._arm_control_mode = "cartesian"
            carb.log_warn(
                "[ArmController] IK accepted: q=%s, position residual=%.6f m, "
                "orientation residual=%.3f deg" % (
                    np.array2string(result.joint_positions, precision=4),
                    result.position_error_m,
                    np.rad2deg(result.orientation_error_rad),
                ))

    def _acquire(self):
        if self._art is not None:
            return
        from omni.isaac.dynamic_control import _dynamic_control
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._state_all = _dynamic_control.STATE_ALL
        self._art = self._dc.get_articulation(self._articulation_path)
        if self._art == 0 or self._art is None:
            self._art = None
            return
        n = self._dc.get_articulation_dof_count(self._art)
        self._dof_index = {}
        for i in range(n):
            dof = self._dc.get_articulation_dof(self._art, i)
            name = self._dc.get_dof_name(dof)
            self._dof_index[name] = dof
        carb.log_warn("[ArmController] articulation acquired: %d DOFs (%s)" %
                      (n, ", ".join(self._dof_index.keys())))

        missing = [name for name in _ARM_JOINTS + [_GRIPPER_MASTER]
                   if name not in self._dof_index]
        if missing:
            carb.log_error("[ArmController] required DOFs missing: " + str(missing))

        # Initialize the slew-limited command from the measured state.  This prevents an
        # acquisition-time jump and makes each new target continuous from the live pose.
        for name in _ARM_JOINTS + [_GRIPPER_MASTER]:
            dof = self._dof_index.get(name)
            if dof is None:
                continue
            q = float(self._dc.get_dof_position(dof))
            self._commanded[name] = q
            self._targets.setdefault(name, q)

    def update(self, dt: float):
        for _ in range(32):
            n_before = self._cb_counter
            rclpy.spin_once(self.node, timeout_sec=0)
            if self._cb_counter == n_before:
                break

        if self._art is None:
            self._acquire()
            if self._art is None:
                return

        # Process joint and Cartesian callbacks in arrival order.  Both update
        # the same target dictionary; only this backend writes those targets.
        self._apply_pending_commands()

        # One continuous writer: slew the commanded setpoint toward the latest target,
        # then apply it every physics step.  This replaces discontinuous target jumps.
        for name, target in self._targets.items():
            dof = self._dof_index.get(name)
            if dof is None:
                continue
            commanded = self._commanded.get(name, float(self._dc.get_dof_position(dof)))
            max_step = self._max_speed.get(name, 1.0) * max(float(dt), 0.0)
            commanded += float(np.clip(target - commanded, -max_step, max_step))
            self._commanded[name] = commanded
            self._dc.set_dof_position_target(dof, commanded)

        self._publish_accum += max(float(dt), 0.0)
        if self._publish_accum >= self._publish_period:
            self._publish_accum %= self._publish_period
            self._publish_joint_states()
            self._publish_ee_pose()

        if self._alignment_debug:
            self._alignment_accum += max(float(dt), 0.0)
            if self._alignment_accum >= 1.0:
                self._alignment_accum %= 1.0
                self._maybe_log_kinematic_alignment()

    def _publish_joint_states(self):
        if self._art is None or self._dof_index is None:
            return
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()

        def read_state(internal_name):
            dof = self._dof_index.get(internal_name)
            if dof is None:
                return None
            try:
                state = self._dc.get_dof_state(dof, self._state_all)
                return float(state.pos), float(state.vel), float(state.effort)
            except Exception:
                return float(self._dc.get_dof_position(dof)), 0.0, 0.0

        for name in _ARM_JOINTS:
            state = read_state(name)
            if state is None:
                continue
            q, qd, effort = state
            msg.name.append(name)
            msg.position.append(q)
            msg.velocity.append(qd)
            msg.effort.append(effort)

        gripper_state = read_state(_GRIPPER_MASTER)
        if gripper_state is not None:
            q, qd, effort = gripper_state
            msg.name.append(_GRIPPER_PUBLIC)
            msg.position.append(q)
            msg.velocity.append(qd)
            msg.effort.append(effort)

        if msg.name:
            self._js_pub.publish(msg)

    def _publish_ee_pose(self):
        measured = self._measured_arm_positions()
        if measured is None:
            return
        try:
            position, quaternion = self._kinematics.forward(measured)
            self._kinematics_error_logged = False
        except Exception as exc:
            if not self._kinematics_error_logged:
                carb.log_error("[ArmController] FK unavailable: " + str(exc))
                self._kinematics_error_logged = True
            return

        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = self._kinematics.base_frame
        msg.pose.position.x = float(position[0])
        msg.pose.position.y = float(position[1])
        msg.pose.position.z = float(position[2])
        msg.pose.orientation.x = float(quaternion[0])
        msg.pose.orientation.y = float(quaternion[1])
        msg.pose.orientation.z = float(quaternion[2])
        msg.pose.orientation.w = float(quaternion[3])
        self._ee_pub.publish(msg)

    @staticmethod
    def _matrix_from_pose(position, quaternion_wxyz):
        quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=np.float64)
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = Rotation.from_quat(
            quaternion_wxyz[[1, 2, 3, 0]]).as_matrix()
        matrix[:3, 3] = np.asarray(position, dtype=np.float64)
        return matrix

    def _usd_pose_in_base(self, prim_path):
        from isaacsim.core.utils.xforms import get_world_pose

        base_position, base_quaternion = get_world_pose(
            self._usd_base_link_path, fabric=True)
        link_position, link_quaternion = get_world_pose(str(prim_path), fabric=True)
        base_world = self._matrix_from_pose(base_position, base_quaternion)
        link_world = self._matrix_from_pose(link_position, link_quaternion)
        link_base = np.linalg.inv(base_world) @ link_world
        position = link_base[:3, 3]
        quaternion_xyzw = Rotation.from_matrix(link_base[:3, :3]).as_quat()
        return position, quaternion_xyzw

    @staticmethod
    def _pose_residual(reference_position, reference_quaternion,
                       measured_position, measured_quaternion):
        position_error = float(np.linalg.norm(
            np.asarray(reference_position) - np.asarray(measured_position)))
        rotation_error = float((
            Rotation.from_quat(reference_quaternion).inv()
            * Rotation.from_quat(measured_quaternion)
        ).magnitude())
        return position_error, rotation_error

    def _usd_tool_center_in_base(self):
        left_position, _ = self._usd_pose_in_base(
            self._usd_link_root_path + "/Gripper_l3")
        right_position, _ = self._usd_pose_in_base(
            self._usd_link_root_path + "/Gripper_r3")
        _, link6_quaternion = self._usd_pose_in_base(
            self._usd_link_root_path + "/Link6")
        return 0.5 * (left_position + right_position), link6_quaternion

    def _maybe_log_kinematic_alignment(self):
        measured = self._measured_arm_positions()
        if measured is None:
            return
        try:
            velocities = np.array([
                float(self._dc.get_dof_state(
                    self._dof_index[name], self._state_all).vel)
                for name in _ARM_JOINTS
            ], dtype=np.float64)
        except Exception:
            velocities = np.zeros(6, dtype=np.float64)
        if np.max(np.abs(velocities)) > 0.01:
            return
        if (self._last_alignment_q is not None
                and np.max(np.abs(measured - self._last_alignment_q)) < 0.02):
            return

        try:
            carb.log_warn(
                "[ArmAlignment] q=" + np.array2string(measured, precision=6))
            for index, frame_name in enumerate(_ARM_JOINTS):
                # USD/URDF link-frame names remain the CAD names Link1..Link6;
                # only articulation joint/DOF names use joint_1..joint_6.
                link_name = "Link%d" % (index + 1)
                lula_position, lula_quaternion = self._kinematics.forward_frame(
                    link_name, measured)
                usd_position, usd_quaternion = self._usd_pose_in_base(
                    self._usd_link_root_path + "/" + link_name)
                position_error, rotation_error = self._pose_residual(
                    lula_position, lula_quaternion, usd_position, usd_quaternion)
                carb.log_warn(
                    "[ArmAlignment] %s pos_err=%.9f m rot_err=%.6f deg "
                    "lula_p=%s usd_p=%s" % (
                        link_name,
                        position_error,
                        np.rad2deg(rotation_error),
                        np.array2string(lula_position, precision=6),
                        np.array2string(usd_position, precision=6),
                    ))

            lula_position, lula_quaternion = self._kinematics.forward(measured)
            usd_position, usd_quaternion = self._usd_tool_center_in_base()
            position_error, rotation_error = self._pose_residual(
                lula_position, lula_quaternion, usd_position, usd_quaternion)
            carb.log_warn(
                "[ArmAlignment] tool_center pos_err=%.9f m rot_err=%.6f deg "
                "lula_p=%s usd_mid_p=%s" % (
                    position_error,
                    np.rad2deg(rotation_error),
                    np.array2string(lula_position, precision=6),
                    np.array2string(usd_position, precision=6),
                ))
            self._last_alignment_q = measured.copy()
        except Exception as exc:
            carb.log_error("[ArmAlignment] diagnostic failed: " + str(exc))

    def input_reference(self):
        # Not driving rotors; multirotor picks up backends[0]'s output for that.
        return None

    def update_state(self, state):
        pass

    def update_sensor(self, sensor_type, data):
        pass

    def update_graphical_sensor(self, sensor_type, data):
        pass

    def start(self):
        pass

    def stop(self):
        pass

    def reset(self):
        self._art = None
        self._targets.clear()
        self._commanded.clear()
        self._pending_commands.clear()
        self._last_ik_solution = None
        self._arm_control_mode = "joint"
        self._publish_accum = 0.0
        self._alignment_accum = 0.0
        self._last_alignment_q = None
