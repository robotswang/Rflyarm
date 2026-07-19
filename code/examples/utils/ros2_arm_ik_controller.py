#!/usr/bin/env python
"""
| File: ros2_arm_ik_controller.py
| Description: ROS2 backend that turns Cartesian end-effector targets into arm joint commands
| via Lula IK. Subscribes /arm/target_pose (PoseStamped) and, on each new message, solves IK
| ONCE (seeded from HOME, not from live joints -- this is the task1 blood-lesson) to produce
| a stable 6-DOF joint configuration. The result is held as _last_solution and driven onto the
| arm articulation via dynamic_control every physics tick. Gripper joints are left untouched
| (they are pinned in robot_description.yaml and handled by ROS2ArmController for the gripper
| open/close control).
|
| The IK target is interpreted in the arm's base_link frame -- i.e. relative to the arm root,
| which itself is welded to the hexrotor body. This matches the stage1 task1 convention and
| makes the target invariant to the platform's flight motion.
|
| End-effector frame = "tool_center": the midpoint between the two finger tips (a fixed frame
| off Link6 added to the URDF for grasp targeting).
|
| Attach as a backend on the Hexrotor after ROS2ArmController. This backend does NOT drive
| rotors (input_reference returns None).
"""

import os

import carb
import numpy as np

import rclpy
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig


class _EmptyConfig(BackendConfig):
    pass


# Resolve URDF + YAML relative to the repo root (this file lives at code/examples/utils/).
_UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_UTILS_DIR, "../../.."))
_URDF = os.path.join(_REPO_ROOT, "assets/robot_arm_urdf/robot_arm_lula.urdf")
_YAML = os.path.join(_REPO_ROOT, "assets/robot_arm_urdf/robot_description.yaml")

_EE_FRAME = "tool_center"
_HOME_SEED = np.zeros(6)                    # IK warm-start; do NOT feed live joints
_ARM_JOINT_NAMES = ["Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"]


class ROS2ArmIKController(Backend):
    """Solve IK once per received Cartesian target, then drive Joint1..6 to that solution.

    Subscribes: <namespace>/target_pose (geometry_msgs/PoseStamped)
      pose is interpreted in the arm base_link frame.
    """

    def __init__(self,
                 articulation_path: str = "/World/rflyarm",
                 namespace: str = "arm",
                 target_pose_topic: str = "target_pose",
                 node_name: str = "rflyarm_arm_ik_controller"):
        super().__init__(_EmptyConfig())

        self._articulation_path = articulation_path
        self._last_solution = None
        self._art = None
        self._dof_index = None
        self._dc = None
        self._solver = None

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(node_name)
        topic = namespace + "/" + target_pose_topic
        self._sub = self.node.create_subscription(
            PoseStamped, topic, self._on_target, 10)
        carb.log_warn("[ROS2ArmIKController] subscribing target pose on: /" + topic)

        # Publish the actual end-effector (tool_center) pose in the arm base_link frame every tick.
        # sensor_data QoS (BestEffort, keep-last=1) matches the high-rate state-stream convention
        # used by /arm/joint_states and /drone/pose.
        ee_topic = namespace + "/ee_pose"
        self._ee_pub = self.node.create_publisher(
            PoseStamped, ee_topic, qos_profile_sensor_data)
        carb.log_warn("[ROS2ArmIKController] publishing ee pose on: /" + ee_topic)

    def _lazy_load_solver(self):
        if self._solver is not None:
            return
        try:
            from isaacsim.robot_motion.motion_generation.lula.kinematics import LulaKinematicsSolver
        except Exception:
            # Older Isaac layout
            from omni.isaac.motion_generation.lula.kinematics import LulaKinematicsSolver
        self._solver = LulaKinematicsSolver(
            robot_description_path=_YAML,
            urdf_path=_URDF,
        )
        try:
            frames = self._solver.get_all_frame_names()
            carb.log_warn("[ROS2ArmIKController] Lula ready. frames: " + str(frames))
        except Exception:
            pass

    def _on_target(self, msg: PoseStamped):
        self._lazy_load_solver()
        p = msg.pose.position
        o = msg.pose.orientation
        pos = np.array([p.x, p.y, p.z])
        # ROS quat is (x,y,z,w); Isaac/Lula wants (w,x,y,z).
        quat = np.array([o.w, o.x, o.y, o.z])

        try:
            q, ok = self._solver.compute_inverse_kinematics(
                frame_name=_EE_FRAME,
                target_position=pos,
                target_orientation=quat,
                warm_start=_HOME_SEED,
            )
        except Exception as e:
            carb.log_warn("[ROS2ArmIKController] IK exception: " + str(e))
            return

        if ok:
            self._last_solution = np.asarray(q, dtype=float)
            # Round-trip check: run FK on the IK solution to see how close it lands to the
            # requested target. Non-zero residual here means Lula's IK numeric tolerance is
            # the floor; PhysX drive stiffness can't beat it.
            try:
                fk_pos, _ = self._solver.compute_forward_kinematics(_EE_FRAME, self._last_solution)
                residual = np.asarray(fk_pos) - pos
                carb.log_warn(
                    "[ROS2ArmIKController] IK ok -> q=%s  FK(q)=%s  residual=%s (|.|=%.4f m)" % (
                        np.array2string(self._last_solution, precision=3),
                        np.array2string(np.asarray(fk_pos), precision=4),
                        np.array2string(residual, precision=4),
                        float(np.linalg.norm(residual)),
                    ))
            except Exception:
                carb.log_warn("[ROS2ArmIKController] IK ok -> " + np.array2string(self._last_solution, precision=3))
        else:
            carb.log_warn("[ROS2ArmIKController] IK did not converge for target " +
                          np.array2string(pos, precision=3))

    def _acquire_articulation(self):
        if self._art is not None:
            return
        from omni.isaac.dynamic_control import _dynamic_control
        self._dc = _dynamic_control.acquire_dynamic_control_interface()
        self._art = self._dc.get_articulation(self._articulation_path)
        if self._art == 0 or self._art is None:
            self._art = None
            return
        n = self._dc.get_articulation_dof_count(self._art)
        self._dof_index = {}
        for i in range(n):
            dof = self._dc.get_articulation_dof(self._art, i)
            self._dof_index[self._dc.get_dof_name(dof)] = dof

    def update(self, dt: float):
        rclpy.spin_once(self.node, timeout_sec=0)

        if self._art is None:
            self._acquire_articulation()
            if self._art is None:
                # Articulation not ready yet -- can't drive nor read FK.
                return

        # Load Lula lazily so consumers can see /arm/ee_pose from startup (via FK) even before
        # the first Cartesian target arrives. Loader is a no-op after first call.
        self._lazy_load_solver()

        # Drive Joint1..6 to the latest IK solution (if we have one). Gripper joints are handled
        # by ROS2ArmController; leaving _last_solution None means "no Cartesian target yet, hold
        # whatever ROS2ArmController set".
        if self._last_solution is not None:
            for i, name in enumerate(_ARM_JOINT_NAMES):
                dof = self._dof_index.get(name)
                if dof is None:
                    continue
                self._dc.set_dof_position_target(dof, float(self._last_solution[i]))

        # Publish end-effector pose every tick as long as solver + articulation are ready, so
        # consumers see the ee pose from startup, not only after the first IK target.
        self._publish_ee_pose()

    def _publish_ee_pose(self):
        if self._solver is None or self._art is None:
            return
        if not hasattr(self._solver, "compute_forward_kinematics"):
            return
        try:
            q = np.array([self._dc.get_dof_position(self._dof_index[n])
                          for n in _ARM_JOINT_NAMES])
            pos, ori = self._solver.compute_forward_kinematics(_EE_FRAME, q)
        except Exception:
            return

        # Lula FK returns (pos: 3-vec, R: 3x3 rotation matrix). Convert to a ROS quat (x,y,z,w)
        # via scipy so downstream consumers get a proper geometry_msgs/Quaternion.
        ori = np.asarray(ori)
        if ori.shape == (3, 3):
            from scipy.spatial.transform import Rotation
            qx, qy, qz, qw = Rotation.from_matrix(ori).as_quat()
        elif ori.shape == (4,):
            # Some Isaac releases return (w, x, y, z) directly -- accept both shapes.
            qw, qx, qy, qz = float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])
        else:
            return

        msg = PoseStamped()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.pose.position.x = float(pos[0])
        msg.pose.position.y = float(pos[1])
        msg.pose.position.z = float(pos[2])
        msg.pose.orientation.x = float(qx)
        msg.pose.orientation.y = float(qy)
        msg.pose.orientation.z = float(qz)
        msg.pose.orientation.w = float(qw)
        self._ee_pub.publish(msg)

    # ------ abstract stubs (this backend does not touch rotors / state) ------
    def input_reference(self):
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
        self._last_solution = None
