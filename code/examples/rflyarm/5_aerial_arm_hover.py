#!/usr/bin/env python
"""
| File: 5_aerial_arm_hover.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Launches the Aerial Arm hexrotor platform (robot-arm geometry on a single flycube-style
| rigid body, with 6 rotors) and holds a fixed hover setpoint using a pure-Python nonlinear controller.
| This is the minimal "can this platform fly?" test.
"""

# Imports to start Isaac Sim from this script
import carb
from isaacsim import SimulationApp

# Start Isaac Sim's simulation environment
simulation_app = SimulationApp({"headless": False})

# -----------------------------------
# The actual script should start here
# -----------------------------------
import omni.timeline
from omni.isaac.core.world import World

# Import the Pegasus API for simulating drones
from pegasus.simulator.params import ROBOTS, SIMULATION_ENVIRONMENTS
from pegasus.simulator.logic.vehicles.hexrotor import Hexrotor, HexrotorConfig
from pegasus.simulator.logic.interface.pegasus_interface import PegasusInterface

# Import the custom python control backend (has hover_setpoint mode, mass=4.3)
import sys, os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)) + '/../utils')
from nonlinear_controller_arm import NonlinearControllerArm

# Auxiliary scipy and numpy modules
from scipy.spatial.transform import Rotation

# Use pathlib for parsing paths
from pathlib import Path


class PegasusApp:
    """
    A Template class that serves as an example on how to build a simple Isaac Sim standalone App.
    """

    def __init__(self):
        """
        Method that initializes the PegasusApp and is used to setup the simulation environment.
        """

        # Acquire the timeline that will be used to start/stop the simulation
        self.timeline = omni.timeline.get_timeline_interface()

        # Start the Pegasus Interface
        self.pg = PegasusInterface()

        # Acquire the World
        self.pg._world = World(**self.pg._world_settings)
        self.world = self.pg.world

        # Launch one of the worlds provided by NVIDIA
        self.pg.load_environment(SIMULATION_ENVIRONMENTS["Curved Gridroom"])

        # Get the current directory used to save results
        self.curr_dir = str(Path(os.path.dirname(os.path.realpath(__file__))).resolve())

        # Create the aerial arm hexrotor with a fixed-hover nonlinear controller.
        # Softened gains for the heavier platform; hover at 1.5 m above the spawn point.
        config = HexrotorConfig()
        config.backends = [NonlinearControllerArm(
            trajectory_file=None,
            results_file=self.curr_dir + "/results/rflyarm_statistics.npz",
            hover_setpoint=[0.0, 0.0, 1.5],
            Kp=[4.0, 4.0, 4.0],
            Kd=[3.0, 3.0, 3.0],
            Ki=[0.2, 0.2, 0.2],
            Kr=[0.5, 0.5, 0.5],
            Kw=[0.2, 0.2, 0.2]
        )]

        Hexrotor(
            "/World/rflyarm",
            ROBOTS['Rflyarm'],
            0,
            [0.0, 0.0, 0.5],
            Rotation.from_euler("XYZ", [0.0, 0.0, 0.0], degrees=True).as_quat(),
            config=config,
        )

        # Reset the simulation environment so that all articulations (aka robots) are initialized
        self.world.reset()

    def run(self):
        """
        Method that implements the application main loop, where the physics steps are executed.
        """

        # Start the simulation
        self.timeline.play()

        # The "infinite" loop
        while simulation_app.is_running():

            # Update the UI of the app and perform the physics step
            self.world.step(render=True)

        # Cleanup and stop
        carb.log_warn("PegasusApp Simulation App is closing.")
        self.timeline.stop()
        simulation_app.close()

def main():

    # Instantiate the template app
    pg_app = PegasusApp()

    # Run the application loop
    pg_app.run()

if __name__ == "__main__":
    main()
