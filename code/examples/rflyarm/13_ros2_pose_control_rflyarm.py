#!/usr/bin/env python
"""
| File: 13_ros2_pose_control_rflyarm.py
| Description: Task 2 for the Rflyarm hexrotor platform -- ROS2 spatial-pose publish & subscribe.
| Send a target pose on /drone/cmd_pose, the platform flies there and hovers. It also publishes
| its current pose on /drone/pose so an external client can watch it.
|
| Backends on the Hexrotor:
|   1. ROS2PoseControllerRflyarm (backends[0], drives the 6 rotors): subscribes /drone/cmd_pose
|      (PoseStamped, ENU) and flies the platform to that pose. Mass + gains are set by
|      tools/set_body_mass.py; the values wired below match the current USD body mass.
|   2. ROS2Backend (publish-only): publishes state, notably /drone/pose. sub_control=False.
|
| Publish side:  ros2 topic echo /drone/pose
| Command side:  python3 drone_console.py
|
| Run: isaac_run <abs path>/PegasusSimulator/examples/13_ros2_pose_control_rflyarm.py
"""

import carb
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.backends.ros2_backend import ROS2Backend
from pegasus.simulator.logic.vehicles.hexrotor import Hexrotor, HexrotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/../utils')
from ros2_pose_controller_rflyarm import ROS2PoseControllerRflyarm
from ros2_arm_controller import ROS2ArmController
from ros2_arm_ik_controller import ROS2ArmIKController

from scipy.spatial.transform import Rotation
from pathlib import Path

INIT_POS = [0.0, 0.0, 0.5]
TAKEOFF_ALT = 1.5
NAMESPACE = "drone"       # cmd topic   -> /drone/cmd_pose
STATE_ID = ""             # state topic -> /drone/pose  (single-drone build)


class PegasusApp:

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()

        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

        config = HexrotorConfig()
        # ORDER MATTERS: rotors are driven by backends[0].input_reference().
        config.backends = [
            ROS2PoseControllerRflyarm(
                namespace=NAMESPACE,
                cmd_pose_topic="cmd_pose",
                takeoff_altitude=TAKEOFF_ALT,
                results_file=self.curr_dir + "/results/rflyarm_pose_statistics.npz",
                Kp=[150.0, 150.0, 150.0],
                Kd=[225.0, 225.0, 225.0],
                Ki=[30.0, 30.0, 30.0],
                Kr=[300.0, 300.0, 300.0],
                Kw=[125.0, 125.0, 125.0],
            ),
            # Arm-joint controller: subscribes /arm/joint_command and drives the arm articulation.
            # Not on backends[0], so rotor commands come from the pose controller above.
            ROS2ArmController(
                articulation_path="/World/rflyarm",
                namespace="arm",
                joint_command_topic="joint_command",
            ),
            # Cartesian arm IK: subscribes /arm/target_pose, solves Lula IK once per target
            # (HOME-seeded), drives Joint1..6. Target is in the arm base_link frame.
            ROS2ArmIKController(
                articulation_path="/World/rflyarm",
                namespace="arm",
                target_pose_topic="target_pose",
            ),
            ROS2Backend(vehicle_id=STATE_ID, num_rotors=6, config={
                "namespace": NAMESPACE,
                "pose_topic": "pose",
                "pub_state": True,
                "pub_sensors": False,
                "pub_graphical_sensors": False,
                "pub_tf": False,
                "sub_control": False,
            }),
        ]

        Hexrotor(
            "/World/rflyarm",
            ROBOTS['Rflyarm'],
            0,
            INIT_POS,
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )

        self.world.reset()
        self.stop_sim = False

    def run(self):
        self.timeline.play()
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=True)
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
