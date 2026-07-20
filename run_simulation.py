#!/usr/bin/env python
"""Run the GUI simulation with flight, arm, IK, and ROS 2 interfaces.

Backend order is significant: ``FlightController`` must remain first because
the vehicle reads rotor commands from ``backends[0]``.
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
from pegasus.simulator.logic.vehicles.hexrotor import Hexrotor, HexrotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from simulation.arm_controller import ArmController
from simulation.flight_controller import FlightController
from simulation.pose_publisher import PosePublisher

from scipy.spatial.transform import Rotation
from pathlib import Path

INIT_POS = [0.0, 0.0, 0.5]
TAKEOFF_ALT = 1.5
NAMESPACE = "drone"       # cmd topic   -> /drone/cmd_pose
REPO_ROOT = Path(__file__).resolve().parent
ARM_URDF = REPO_ROOT / "assets/kinematics/arm.urdf"
ARM_DESCRIPTION = REPO_ROOT / "assets/kinematics/robot_description.yaml"


class PegasusApp:

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()

        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        config = HexrotorConfig()
        # ORDER MATTERS: rotors are driven by backends[0].input_reference().
        config.backends = [
            FlightController(
                namespace=NAMESPACE,
                cmd_pose_topic="cmd_pose",
                takeoff_altitude=TAKEOFF_ALT,
                Kp=[150.0, 150.0, 150.0],
                Kd=[225.0, 225.0, 225.0],
                Ki=[30.0, 30.0, 30.0],
                Kr=[300.0, 300.0, 300.0],
                Kw=[125.0, 125.0, 125.0],
            ),
            # Single-owner arm controller.  Flight remains exclusively on backends[0].
            ArmController(
                articulation_path="/World/rflyarm",
                joint_command_topic="/joint_command",
                joint_states_topic="/joint_states",
                target_pose_topic="/arm/cmd_pose",
                ee_pose_topic="/arm/ee_pose",
                robot_description_path=str(ARM_DESCRIPTION),
                urdf_path=str(ARM_URDF),
                base_frame="base_link",
                ee_frame="tool_center",
                usd_base_link_path="/World/rflyarm/arm_geo/Geometry/base_link",
                usd_link_root_path="/World/rflyarm/arm_geo/Geometry",
                alignment_debug=True,
                publish_hz=60.0,
                arm_max_speed=1.5,
            ),
            PosePublisher(
                topic="/drone/pose",
                frame_id="map",
                publish_hz=60.0,
            ),
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
