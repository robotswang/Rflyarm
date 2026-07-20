#!/usr/bin/env python
"""Run the warehouse simulation with flight, arm, IK, and ROS 2 interfaces.

Backend order is significant: ``FlightController`` must remain first because
the vehicle reads rotor commands from ``backends[0]``.
"""

import argparse
from pathlib import Path

import carb
from isaacsim import SimulationApp

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument("--headless", action="store_true")
_parser.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="Stop after this many simulation steps; 0 runs until the app closes.",
)
_args, _unknown = _parser.parse_known_args()

simulation_app = SimulationApp({"headless": _args.headless})

# -----------------------------------
import omni.timeline
from isaacsim.core.api.world import World

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.ros2.bridge")

from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

from simulation.arm_controller import ArmController
from simulation.flight_controller import FlightController
from simulation.hexrotor import Hexrotor, HexrotorConfig
from simulation.pose_publisher import PosePublisher

from pxr import Sdf, Usd, UsdGeom

TAKEOFF_ALT = 1.5
NAMESPACE = "drone"       # cmd topic   -> /drone/cmd_pose
REPO_ROOT = Path(__file__).resolve().parent
WAREHOUSE_USD = REPO_ROOT / "assets/Collected_warehouse/warehouse.usd"
ARM_URDF = REPO_ROOT / "assets/kinematics/arm.urdf"
ARM_DESCRIPTION = REPO_ROOT / "assets/kinematics/robot_description.yaml"
WAREHOUSE_PRIM = "/World/layout"
EMBEDDED_RFLYARM_PRIM = WAREHOUSE_PRIM + "/rflyarm"
RFLYARM_PRIM = "/World/rflyarm"


class PegasusApp:

    def __init__(self):
        self.timeline = omni.timeline.get_timeline_interface()
        self.pg = PegasusInterface()

        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world
        vehicle_usd, init_pos, init_orientation = self._load_warehouse()

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
                articulation_path=RFLYARM_PRIM,
                joint_command_topic="/joint_command",
                joint_states_topic="/joint_states",
                target_pose_topic="/arm/cmd_pose",
                ee_pose_topic="/arm/ee_pose",
                robot_description_path=str(ARM_DESCRIPTION),
                urdf_path=str(ARM_URDF),
                base_frame="base_link",
                ee_frame="tool_center",
                usd_base_link_path=RFLYARM_PRIM + "/arm_geo/Geometry/base_link",
                usd_link_root_path=RFLYARM_PRIM + "/arm_geo/Geometry",
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
            RFLYARM_PRIM,
            vehicle_usd,
            0,
            init_pos,
            init_orientation,
            config=config,
        )

        self.world.reset()
        self.stop_sim = False

    def _load_warehouse(self):
        """Reference the warehouse and prepare its collected Rflyarm asset."""
        if not WAREHOUSE_USD.is_file():
            raise FileNotFoundError("Warehouse USD not found: " + str(WAREHOUSE_USD))

        warehouse_layer = Sdf.Layer.FindOrOpen(str(WAREHOUSE_USD))
        if warehouse_layer is None:
            raise RuntimeError("Failed to inspect warehouse USD: " + str(WAREHOUSE_USD))
        vehicle_spec = warehouse_layer.GetPrimAtPath("/Root/rflyarm")
        if vehicle_spec is None:
            raise RuntimeError("Rflyarm has no specification in the warehouse root layer")

        references = list(vehicle_spec.referenceList.prependedItems)
        references += list(vehicle_spec.referenceList.explicitItems)
        references += list(vehicle_spec.referenceList.appendedItems)
        vehicle_references = [ref for ref in references if ref.assetPath]
        if len(vehicle_references) != 1:
            raise RuntimeError(
                "Expected one Rflyarm asset reference, found: "
                + str(len(vehicle_references)))
        vehicle_usd = Sdf.ComputeAssetPathRelativeToLayer(
            warehouse_layer, vehicle_references[0].assetPath)
        if not Path(vehicle_usd).is_file():
            raise FileNotFoundError(
                "Collected Rflyarm USD not found: " + vehicle_usd)

        self.pg.load_asset(str(WAREHOUSE_USD), WAREHOUSE_PRIM)
        vehicle = self.world.stage.GetPrimAtPath(EMBEDDED_RFLYARM_PRIM)
        if not vehicle.IsValid():
            raise RuntimeError(
                "Embedded Rflyarm prim not found at: " + EMBEDDED_RFLYARM_PRIM)

        transform = UsdGeom.Xformable(vehicle).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default())
        translation = transform.ExtractTranslation()
        orientation = transform.ExtractRotationQuat()
        imaginary = orientation.GetImaginary()
        init_pos = [translation[0], translation[1], translation[2]]
        init_orientation = [
            imaginary[0], imaginary[1], imaginary[2], orientation.GetReal()]

        # The collected scene already contains the vehicle. Keep the source USD
        # untouched, deactivate that referenced copy in the runtime stage, and
        # let Pegasus spawn an identical controlled instance from the collected
        # asset at /World/rflyarm.
        vehicle.SetActive(False)

        carb.log_warn(
            "[Rflyarm] loaded collected warehouse and prepared vehicle asset: "
            + vehicle_usd)
        return vehicle_usd, init_pos, init_orientation

    def run(self):
        self.timeline.play()
        steps = 0
        while simulation_app.is_running() and not self.stop_sim:
            self.world.step(render=not _args.headless)
            steps += 1
            if _args.max_steps > 0 and steps >= _args.max_steps:
                break
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()


def main():
    pg_app = PegasusApp()
    pg_app.run()


if __name__ == "__main__":
    main()
