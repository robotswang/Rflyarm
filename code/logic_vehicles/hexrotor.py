"""
| File: hexrotor.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2024, Marcelo Jacinto. All rights reserved.
| Description: Copy of Multirotor generalized to any number of rotors (used for the 6-rotor Aerial Arm).
| The ONLY functional change from Multirotor is that the force-application loop iterates over
| self._thrusters._num_rotors instead of a hardcoded 4. The vehicle uses the standard flycube-style
| naming (body at /body, rotors at /rotorN, joints jointN, articulation root on the top-level vehicle),
| so the base Vehicle class works unchanged.
"""

import numpy as np

from omni.isaac.dynamic_control import _dynamic_control

# The vehicle interface (standard Vehicle; the Aerial Arm USD follows flycube naming so no override needed)
from pegasus.simulator.logic.vehicles.vehicle import Vehicle

# Mavlink interface
from pegasus.simulator.logic.backends.px4_mavlink_backend import PX4MavlinkBackend, PX4MavlinkBackendConfig

# Sensors and dynamics setup
from pegasus.simulator.logic.dynamics import LinearDrag
from pegasus.simulator.logic.thrusters import QuadraticThrustCurve
from pegasus.simulator.logic.sensors import Barometer, IMU, Magnetometer, GPS

# Location of the Aerial Arm asset
from pegasus.simulator.params import ROBOTS


class HexrotorConfig:
    """
    A data class that is used for configuring a Hexrotor (6 rotors).
    """

    def __init__(self):
        # Stage prefix of the vehicle when spawning in the world
        self.stage_prefix = "hexrotor"

        # The USD file that describes the visual aspect of the vehicle
        self.usd_file = ROBOTS["Rflyarm"]

        # Thrust curve for the hexrotor. Total system ~107.7 kg -> needs ~1057 N hover.
        # rotor_constant 1.25e-3 gives each rotor up to 1.25e-3 * 1100^2 = 1512 N at max rad/s
        # (six rotors ~9075 N), plenty of margin. Scale set by tools/set_body_mass.py.
        self.thrust_curve = QuadraticThrustCurve(config={
            "num_rotors": 6,
            "rotor_constant": [0.00125, 0.00125, 0.00125, 0.00125, 0.00125, 0.00125],
            "rolling_moment_coefficient": [2.5e-05, 2.5e-05, 2.5e-05, 2.5e-05, 2.5e-05, 2.5e-05],
            "rot_dir": [-1, 1, -1, 1, -1, 1],
            "min_rotor_velocity": [0, 0, 0, 0, 0, 0],
            "max_rotor_velocity": [1100, 1100, 1100, 1100, 1100, 1100],
        })
        self.drag = LinearDrag([0.50, 0.30, 0.0])

        # The default sensors
        self.sensors = [Barometer(), IMU(), Magnetometer(), GPS()]
        self.graphical_sensors = []
        self.graphs = []

        # By default use mavlink; usually overridden by a custom Python controller in the launch script.
        self.backends = [PX4MavlinkBackend(config=PX4MavlinkBackendConfig())]


class Hexrotor(Vehicle):
    """Hexrotor class - a generic 6-rotor multirotor (used for the Aerial Arm platform)."""

    def __init__(
        self,
        stage_prefix: str = "hexrotor",
        usd_file: str = "",
        vehicle_id: int = 0,
        init_pos=[0.0, 0.0, 0.5],
        init_orientation=[0.0, 0.0, 0.0, 1.0],
        config=HexrotorConfig(),
    ):
        """Initializes the hexrotor object

        Args:
            stage_prefix (str): The name the vehicle will present in the simulator when spawned. Defaults to "hexrotor".
            usd_file (str): The USD file that describes the looks and shape of the vehicle. Defaults to "".
            vehicle_id (int): The id to be used for the vehicle. Defaults to 0.
            init_pos (list): The initial position in the inertial frame (ENU). Defaults to [0.0, 0.0, 0.5].
            init_orientation (list): The initial orientation quaternion [qx, qy, qz, qw]. Defaults to [0.0, 0.0, 0.0, 1.0].
            config (HexrotorConfig, optional): Defaults to HexrotorConfig().
        """

        # 1. Initiate the Vehicle object itself
        super().__init__(stage_prefix, usd_file, init_pos, init_orientation, config.sensors, config.graphical_sensors, config.graphs, config.backends)

        # 2. Setup the dynamics of the system
        self._thrusters = config.thrust_curve
        self._drag = config.drag

    def start(self):
        pass

    def stop(self):
        pass

    def update(self, dt: float):
        """
        Computes and applies the forces to the vehicle based on the motor speed. Called every physics step.

        Args:
            dt (float): The time elapsed between the previous and current function calls (s).
        """

        # Get the articulation root of the vehicle
        articulation = self.get_dc_interface().get_articulation(self._stage_prefix)

        # Get the desired angular velocities for each rotor from the first backend expressed in rad/s
        if len(self._backends) != 0:
            desired_rotor_velocities = self._backends[0].input_reference()
        else:
            desired_rotor_velocities = [0.0 for i in range(self._thrusters._num_rotors)]

        # Input the desired rotor velocities in the thruster model
        self._thrusters.set_input_reference(desired_rotor_velocities)

        # Get the desired forces to apply to the vehicle
        forces_z, _, rolling_moment = self._thrusters.update(self._state, dt)

        # Apply force to each rotor (generalized to num_rotors instead of hardcoded 4)
        for i in range(self._thrusters._num_rotors):

            # Apply the force in Z on the rotor frame
            self.apply_force([0.0, 0.0, forces_z[i]], body_part="/rotor" + str(i))

            # Generate the rotating propeller visual effect
            self.handle_propeller_visual(i, forces_z[i], articulation)

        # Apply the torque to the body frame of the vehicle that corresponds to the rolling moment
        self.apply_torque([0.0, 0.0, rolling_moment], "/body")

        # Compute the total linear drag force to apply to the vehicle's body frame
        drag = self._drag.update(self._state, dt)
        self.apply_force(drag, body_part="/body")

        # Call the update methods in all backends
        for backend in self._backends:
            backend.update(dt)

    def handle_propeller_visual(self, rotor_number, force: float, articulation):
        """
        Sets the joint velocity of each rotor (for animation) based on the force being applied.

        Args:
            rotor_number (int): The rotor index to animate
            force (float): The force being applied on that rotor
            articulation: The articulation group the rotor joints belong to
        """

        # Rotate the joint to yield the visual of a rotor spinning (animation only)
        joint = self.get_dc_interface().find_articulation_dof(articulation, "joint" + str(rotor_number))

        if 0.0 < force < 0.1:
            self.get_dc_interface().set_dof_velocity(joint, 5 * self._thrusters.rot_dir[rotor_number])
        elif 0.1 <= force:
            self.get_dc_interface().set_dof_velocity(joint, 100 * self._thrusters.rot_dir[rotor_number])
        else:
            self.get_dc_interface().set_dof_velocity(joint, 0)

    def force_and_torques_to_velocities(self, force: float, torque: np.ndarray):
        """
        Get the target angular velocities for each rotor, given the total desired thrust [N] and
        torque [Nm] in the body frame. Generalized to num_rotors via the thrust curve.

        Args:
            force (np.ndarray): The force to apply in the body frame [N]
            torque (np.ndarray): The torque to apply in the body frame [Nm]

        Returns:
            list: angular velocities [rad/s] for each rotor
        """

        # Get the body frame of the vehicle
        rb = self.get_dc_interface().get_rigid_body(self._stage_prefix + "/body")

        # Get the rotors of the vehicle
        rotors = [self.get_dc_interface().get_rigid_body(self._stage_prefix + "/rotor" + str(i)) for i in range(self._thrusters._num_rotors)]

        # Relative position of the rotors with respect to the body frame
        relative_poses = self.get_dc_interface().get_relative_body_poses(rb, rotors)

        # Allocation matrix (4 constraints: total thrust + 3 torques) x num_rotors
        aloc_matrix = np.zeros((4, self._thrusters._num_rotors))
        aloc_matrix[0, :] = np.array(self._thrusters._rotor_constant)
        aloc_matrix[1, :] = np.array([relative_poses[i].p[1] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])
        aloc_matrix[2, :] = np.array([-relative_poses[i].p[0] * self._thrusters._rotor_constant[i] for i in range(self._thrusters._num_rotors)])
        aloc_matrix[3, :] = np.array([self._thrusters._rolling_moment_coefficient[i] * self._thrusters._rot_dir[i] for i in range(self._thrusters._num_rotors)])

        # Pseudo-inverse to get squared angular velocities from thrust and torques
        aloc_inv = np.linalg.pinv(aloc_matrix)
        squared_ang_vel = aloc_inv @ np.array([force, torque[0], torque[1], torque[2]])

        # No negative squared angular velocities
        squared_ang_vel[squared_ang_vel < 0] = 0.0

        # Saturate while preserving relations
        max_thrust_vel_squared = np.power(self._thrusters.max_rotor_velocity[0], 2)
        max_val = np.max(squared_ang_vel)
        if max_val >= max_thrust_vel_squared:
            normalize = np.maximum(max_val / max_thrust_vel_squared, 1.0)
            squared_ang_vel = squared_ang_vel / normalize

        # Angular velocities [rad/s]
        ang_vel = np.sqrt(squared_ang_vel)
        return ang_vel
