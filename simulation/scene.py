"""Isaac Lab scene configuration for the local SimpleRoom and Rflyarm."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass


REPO_ROOT = Path(__file__).resolve().parents[1]
# World0.usd is the collected Kit stage snapshot and has no defaultPrim.
# Spawn the referenced composition layer, which is a reusable USD asset.
WORLD_USD = REPO_ROOT / "assets" / "Collected_World" / "SubUSDs" / "World.usd"
RFLYARM_PRIM_PATH = "/World/layout/rflyarm"

if not WORLD_USD.is_file():
    raise FileNotFoundError(f"World USD not found: {WORLD_USD}")


@configclass
class RflyarmSceneCfg(InteractiveSceneCfg):
    """Load the local world once and bind its embedded aerial manipulator."""

    world = AssetBaseCfg(
        prim_path="/World/layout",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORLD_USD)),
    )

    rflyarm = ArticulationCfg(
        prim_path=RFLYARM_PRIM_PATH,
        spawn=None,
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint_[1-6]"],
                stiffness=None,
                damping=None,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*"],
                stiffness=None,
                damping=None,
            ),
            "rotor_visuals": ImplicitActuatorCfg(
                joint_names_expr=["rotor_[1-6]"],
                stiffness=0.0,
                damping=0.0,
            ),
        },
    )
