#!/usr/bin/env python
"""
| File: ros2_pose_controller_rflyarm.py
| Description: ROS2 pose-tracking controller for the Rflyarm hexrotor platform. Built on
| NonlinearControllerArm so it inherits the heavier platform's mass (4.3 kg) and softened gains.
| Subscribes a target SE(3) pose on /drone/cmd_pose (PoseStamped, ENU) and holds it as a hover
| setpoint.
|
| Runs inside the Isaac process using raw rclpy (like ROS2Backend). External publishers are
| separate processes -> cross-process DDS discovery works, no relay needed.
|
| Backend ordering: rotors are driven by backends[0].input_reference(), so this MUST be first;
| put the publish-only ROS2Backend after it.
"""

import carb
import numpy as np
from scipy.spatial.transform import Rotation

import rclpy
from geometry_msgs.msg import PoseStamped

from nonlinear_controller_arm import NonlinearControllerArm


class ROS2PoseControllerRflyarm(NonlinearControllerArm):
    """Geometric controller for the Rflyarm whose position/yaw setpoint comes from a
    ROS2 PoseStamped topic (hover-and-hold at the commanded pose).

    Subscribes: <namespace>/cmd_pose (geometry_msgs/PoseStamped, ENU / map frame)
    """

    def __init__(self, namespace: str = "drone", cmd_pose_topic: str = "cmd_pose",
                 node_name: str = "rflyarm_pose_controller",
                 takeoff_altitude: float = 1.5, **kwargs):

        kwargs.pop("trajectory_file", None)
        kwargs.pop("hover_setpoint", None)
        super().__init__(trajectory_file=None, hover_setpoint=None, **kwargs)

        self._takeoff_altitude = takeoff_altitude
        self._p_setpoint = None
        self._setpoint_initialized = False

        self._yaw_target = 0.0
        self._yaw_ref = 0.0
        self._yaw_rate = 0.6

        # Integrator sizing: the arm has a lateral COM offset that produces a persistent
        # ~40 Nm gravity moment on the platform, forcing the flight controller into a steady
        # ~8 deg tilt that adds a ~1.5 m/s^2 sideways thrust component. Without a wide-enough
        # int_band the integrator freezes for far targets and the drone parks 1+ m off. The
        # band must cover the largest routine step (~10 m); the clip is sized to the
        # correction that Ki eventually needs to accumulate.
        self._int_limit = 30.0
        self._int_band = 10.0

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(node_name)
        topic = namespace + "/" + cmd_pose_topic
        self._cmd_sub = self.node.create_subscription(
            PoseStamped, topic, self._cmd_pose_callback, 10)
        carb.log_warn("[ROS2PoseControllerRflyarm] subscribing target pose on: /" + topic)

    def _cmd_pose_callback(self, msg: PoseStamped):
        self._p_setpoint = np.array([msg.pose.position.x,
                                     msg.pose.position.y,
                                     msg.pose.position.z])
        q = [msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w]
        self._yaw_target = float(Rotation.from_quat(q).as_euler("ZYX")[0])
        self._setpoint_initialized = True
        carb.log_warn("[ROS2PoseControllerRflyarm] new setpoint p=%s yaw_target=%.3f" %
                      (np.array2string(self._p_setpoint, precision=3), self._yaw_target))

    def update_state(self, state):
        super().update_state(state)
        if not self._setpoint_initialized:
            self._p_setpoint = np.array([state.position[0], state.position[1], self._takeoff_altitude])
            yaw0 = float(Rotation.from_quat(state.attitude).as_euler("ZYX")[0])
            self._yaw_target = yaw0
            self._yaw_ref = yaw0
            self._setpoint_initialized = True

    def update(self, dt: float):
        rclpy.spin_once(self.node, timeout_sec=0)

        err = (self._yaw_target - self._yaw_ref + np.pi) % (2 * np.pi) - np.pi
        max_step = self._yaw_rate * dt
        self._yaw_ref += float(np.clip(err, -max_step, max_step))

        int_before = np.array(self.int)
        far_from_target = (self._p_setpoint is not None and
                           np.linalg.norm(self.p - self._p_setpoint) > self._int_band)

        super().update(dt)

        if far_from_target:
            self.int = int_before
        self.int = np.clip(self.int, -self._int_limit, self._int_limit)

    def pd(self, t, s, reverse=False):
        if self._p_setpoint is None:
            return np.zeros(3)
        return self._p_setpoint

    def d_pd(self, t, s, reverse=False):
        return np.zeros(3)

    def dd_pd(self, t, s, reverse=False):
        return np.zeros(3)

    def ddd_pd(self, t, s, reverse=False):
        return np.zeros(3)

    def yaw_d(self, t, s):
        return self._yaw_ref

    def d_yaw_d(self, t, s):
        return 0.0
