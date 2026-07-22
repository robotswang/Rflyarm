"""Isaac Lab scene configuration for the local SimpleRoom and Rflyarm."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass


REPO_ROOT = Path(__file__).resolve().parents[1]
# World0.usd is the collected Kit stage snapshot and has no defaultPrim.
# Spawn the referenced composition layer, which is a reusable USD asset.
WORLD_USD = REPO_ROOT / "assets" / "Collected_World" / "SubUSDs" / "World.usd"
INCLINED_PANEL_USD = REPO_ROOT / "assets" / "Collected_World" / "SubUSDs" / "inclined_panel.usda"
RFLYARM_PRIM_PATH = "/World/layout/rflyarm"

for asset_path in (WORLD_USD, INCLINED_PANEL_USD):
    if not asset_path.is_file():
        raise FileNotFoundError(f"USD asset not found: {asset_path}")


@configclass
class RflyarmSceneCfg(InteractiveSceneCfg):
    """Load the local world once and bind its embedded aerial manipulator."""

    world = AssetBaseCfg(
        prim_path="/World/layout",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORLD_USD)),
    )

    inclined_panel = AssetBaseCfg(
        prim_path="/World/layout/inclined_panel",
        spawn=sim_utils.UsdFileCfg(usd_path=str(INCLINED_PANEL_USD)),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -5.43, 7.17),
            rot=(0.3826834324, 0.0, 0.0, 0.9238795325),
        ),
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

    depth_camera = CameraCfg(
        prim_path=f"{RFLYARM_PRIM_PATH}/body/depth_camera",
        update_period=1.0 / 15.0,
        height=240,
        width=320,
        # Keep a color AOV active. A depth-only RTX render product switches
        # Kit into its global "No Rendering" mode and blacks out the viewport.
        data_types=["rgb", "distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=10.5,
            focus_distance=0.4,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.0),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.14, 0.0, 0.10),
            rot=(0.0, 0.0, 0.0, 1.0),
            convention="ros",
        ),
    )
