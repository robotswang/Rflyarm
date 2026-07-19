#!/usr/bin/env python
"""
| File: ros2_arm_controller.py
| Description: ROS2 backend for the arm articulation embedded in the Rflyarm platform. Subscribes
| sensor_msgs/JointState on /arm/joint_command and forwards each named joint's position target to
| the arm articulation via Isaac's dynamic_control interface.
|
| The arm is a *separate* articulation from the hexrotor (bridged by a fixed joint on the USD side),
| so driving it does not disturb the flight controller running on backends[0]. This backend does
| NOT contribute to rotor commands: input_reference() returns None so multirotor.py falls through
| to backends[0].
|
| Attach as backends[1] (or later) in the launch script.
"""

import carb

import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState

from pegasus.simulator.logic.backends.backend import Backend, BackendConfig


class _EmptyConfig(BackendConfig):
    pass


# Joint names that appear in the flat USD; all others are mimic-driven.
# Also the set we publish on /arm/joint_states -- rotor DOFs and mimic gripper
# followers are excluded (rotors are driven by the flight controller and carry
# no arm-side semantics; the 5 mimic joints just echo Gripper_r1).
_COMMANDABLE_JOINTS = [
    "Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6",
    "Gripper_r1",  # left side + gripper_r2/r3/l2/l3 follow via mimic
]


class ROS2ArmController(Backend):
    """Drives the arm articulation from a ROS2 JointState topic.

    Subscribes: <namespace>/joint_command (sensor_msgs/JointState)
      msg.name     -- list of joint names to command
      msg.position -- desired positions [rad], matched positionally to name
    """

    def __init__(self,
                 articulation_path: str = "/World/rflyarm",
                 namespace: str = "arm",
                 joint_command_topic: str = "joint_command",
                 node_name: str = "rflyarm_arm_controller"):
        super().__init__(_EmptyConfig())

        self._articulation_path = articulation_path
        self._targets = {}
        self._cb_counter = 0
        self._art = None
        self._dof_index = None
        self._dc = None

        try:
            rclpy.init()
        except Exception:
            pass

        self.node = rclpy.create_node(node_name)
        topic = namespace + "/" + joint_command_topic
        self._sub = self.node.create_subscription(
            JointState, topic, self._cmd_cb, 10)
        carb.log_warn("[ROS2ArmController] subscribing joint commands on: /" + topic)

        # Publish current joint positions of the arm-side commandable joints (6 arm + Gripper_r1)
        # so external nodes can see the arm state. Rotor DOFs and gripper mimic followers are
        # excluded -- they carry no arm-side semantics. sensor_data QoS matches /joint_states
        # convention (BestEffort, keep-last=1) for a high-rate state stream.
        js_topic = namespace + "/joint_states"
        self._js_pub = self.node.create_publisher(
            JointState, js_topic, qos_profile_sensor_data)
        carb.log_warn("[ROS2ArmController] publishing joint states on: /" + js_topic)

    def _cmd_cb(self, msg: JointState):
        # Increment on every callback so the drain loop in update() can detect "message
        # arrived this spin" regardless of dedup / unknown-name filtering.
        self._cb_counter += 1
        if not msg.name:
            return
        n = min(len(msg.name), len(msg.position))
        for i in range(n):
            name = msg.name[i]
            if name not in _COMMANDABLE_JOINTS:
                carb.log_warn(
                    "[ROS2ArmController] ignoring unknown joint name '%s' "
                    "(expected one of %s)" % (name, _COMMANDABLE_JOINTS))
                continue
            self._targets[name] = float(msg.position[i])

    def _acquire(self):
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
            name = self._dc.get_dof_name(dof)
            self._dof_index[name] = dof
        carb.log_warn("[ROS2ArmController] articulation acquired: %d DOFs (%s)" %
                      (n, ", ".join(self._dof_index.keys())))

        # PhysX drive PD tuning intentionally left to USD defaults. USD has stiffness+
        # armature+damping tuned as a matched set for each joint; overriding stiffness
        # alone via dc.set_dof_properties broke Joint1/Joint2 in earlier attempts (target
        # accepted, but PhysX PD couldn't drive the joint through its armature=5 virtual
        # inertia). Root fix for the ~15deg Joint1/2 servo error against gravity load
        # requires either reducing armature in USD or adding a gravity-comp feedforward
        # torque here -- both larger changes. Keep USD defaults; live with the residual.

    def update(self, dt: float):
        # Drain the whole subscription queue this tick -- spin_once pops one message at a
        # time, so bursty publishers (e.g. `ros2 topic pub -r 10 --times 5`) would otherwise
        # spread across several physics steps. `_cb_counter` bumps on every callback fire, so
        # we can detect "spin_once did nothing" without touching rclpy internals. Capped at a
        # generous 32 iterations so a runaway publisher can't stall the physics step.
        for _ in range(32):
            n_before = self._cb_counter
            rclpy.spin_once(self.node, timeout_sec=0)
            if self._cb_counter == n_before:
                break

        if self._art is None:
            self._acquire()
            if self._art is None:
                return

        if not self._targets:
            # Even with no command yet, still publish current joint states so consumers can watch.
            self._publish_joint_states()
            return

        for name, pos in self._targets.items():
            dof = self._dof_index.get(name)
            if dof is None:
                continue
            self._dc.set_dof_position_target(dof, pos)

        # One-shot diagnostic: after the first command each session, log target-vs-actual
        # for Joint1..6 so we can see if PhysX actually accepted our targets.
        if not getattr(self, "_diag_logged", False):
            try:
                lines = []
                for jn in ("Joint1", "Joint2", "Joint3", "Joint4", "Joint5", "Joint6"):
                    d = self._dof_index.get(jn)
                    if d is None:
                        continue
                    tgt = self._dc.get_dof_position_target(d)
                    act = self._dc.get_dof_position(d)
                    lines.append("%s: target=%.3f actual=%.3f" % (jn, float(tgt), float(act)))
                carb.log_warn("[ROS2ArmController] targets accepted: " + " | ".join(lines))
                self._diag_logged = True
            except Exception as e:
                carb.log_warn("[ROS2ArmController] target diag failed: " + str(e))

        self._publish_joint_states()

    def _publish_joint_states(self):
        if self._art is None or self._dof_index is None:
            return
        msg = JointState()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        for name in _COMMANDABLE_JOINTS:
            dof = self._dof_index.get(name)
            if dof is None:
                continue
            try:
                q = float(self._dc.get_dof_position(dof))
            except Exception:
                continue
            msg.name.append(name)
            msg.position.append(q)
        self._js_pub.publish(msg)

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
