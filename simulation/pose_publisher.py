#!/usr/bin/env python
"""Publish the Rflyarm flight pose as a ROS 2 ``PoseStamped`` message."""

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.qos import qos_profile_sensor_data

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig


class _EmptyConfig(BackendConfig):
    pass


class PosePublisher(Backend):
    """Publish the vehicle position and attitude in the configured frame."""

    def __init__(
        self,
        topic: str = "/drone/pose",
        frame_id: str = "map",
        publish_hz: float = 60.0,
        node_name: str = "rflyarm_pose_publisher",
    ):
        super().__init__(_EmptyConfig())
        self._topic = str(topic)
        self._frame_id = str(frame_id)
        self._publish_period = 1.0 / max(float(publish_hz), 1.0)
        self._publish_accum = 0.0
        self._latest_state = None

        try:
            rclpy.init()
        except Exception:
            pass
        self.node = rclpy.create_node(node_name)
        self._publisher = self.node.create_publisher(
            PoseStamped, self._topic, qos_profile_sensor_data)

    def update_state(self, state):
        # Copy primitive values because Pegasus may reuse its State arrays.
        self._latest_state = {
            "position": tuple(float(value) for value in state.position),
            "attitude": tuple(float(value) for value in state.attitude),
        }

    def update(self, dt: float):
        if self._latest_state is None or self.node is None:
            return
        self._publish_accum += max(float(dt), 0.0)
        if self._publish_accum < self._publish_period:
            return
        self._publish_accum %= self._publish_period

        state = self._latest_state
        message = PoseStamped()
        message.header.stamp = self.node.get_clock().now().to_msg()
        message.header.frame_id = self._frame_id

        position = state["position"]
        attitude = state["attitude"]
        message.pose.position.x = position[0]
        message.pose.position.y = position[1]
        message.pose.position.z = position[2]
        message.pose.orientation.x = attitude[0]
        message.pose.orientation.y = attitude[1]
        message.pose.orientation.z = attitude[2]
        message.pose.orientation.w = attitude[3]

        self._publisher.publish(message)

    def input_reference(self):
        return None

    def update_sensor(self, sensor_type, data):
        pass

    def update_graphical_sensor(self, sensor_type, data):
        pass

    def start(self):
        pass

    def stop(self):
        if self.node is not None:
            self.node.destroy_node()
            self.node = None

    def reset(self):
        self._latest_state = None
        self._publish_accum = 0.0
