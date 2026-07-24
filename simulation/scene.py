"""Isaac Lab scene configuration for the local SimpleRoom and Rflyarm."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils.configclass import configclass
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg


REPO_ROOT = Path(__file__).resolve().parents[1]
# World0.usd is the collected Kit stage snapshot and has no defaultPrim.
# Spawn the referenced composition layer, which is a reusable USD asset.
WORLD_USD = REPO_ROOT / "assets" / "Collected_World" / "SubUSDs" / "World.usd"
INCLINED_PANEL_USD = REPO_ROOT / "assets" / "Collected_World" / "SubUSDs" / "inclined_panel.usda"
TARGET_BULB_USD = REPO_ROOT / "assets" / "Collected_World" / "SubUSDs" / "target_bulb.usd"
RFLYARM_PRIM_PATH = "/World/layout/rflyarm"
INCLINED_PANEL_ROT = (0.1736481777, 0.0, 0.0, 0.9848077530)
INCLINED_BULB_ROT = (0.1227878040, -0.6963642403, -0.1227878040, 0.6963642403)

for asset_path in (WORLD_USD, INCLINED_PANEL_USD, TARGET_BULB_USD):
    if not asset_path.is_file():
        raise FileNotFoundError(f"USD asset not found: {asset_path}")


@configclass
class RflyarmSceneCfg(InteractiveSceneCfg):
    """Load the local world once and bind its embedded aerial manipulator."""

    world = AssetBaseCfg(
        prim_path="/World/layout",
        spawn=sim_utils.UsdFileCfg(usd_path=str(WORLD_USD)),
    )

    # Bind tensor views only to the ceiling instance. The shared target_bulb.usd
    # and the separately spawned inclined_target_bulb remain unmodified.
    ceiling_socket = RigidObjectCfg(
        prim_path="/World/layout/target_bulb/Socket",
        spawn=None,
    )

    ceiling_bulb = RigidObjectCfg(
        prim_path="/World/layout/target_bulb/Bulb",
        spawn=None,
    )

    inclined_panel = AssetBaseCfg(
        prim_path="/World/layout/inclined_panel",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(INCLINED_PANEL_USD),
            physics_material_path="physics_material",
            physics_material=PhysxRigidBodyMaterialCfg(
                static_friction=2.0,
                dynamic_friction=1.5,
                restitution=0.0,
                friction_combine_mode="max",
                restitution_combine_mode="min",
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, -5.088840, 7.739291),
            rot=INCLINED_PANEL_ROT,
        ),
    )

    inclined_target_bulb = AssetBaseCfg(
        prim_path="/World/layout/inclined_target_bulb",
        spawn=sim_utils.UsdFileCfg(
            usd_path=str(TARGET_BULB_USD),
            scale=(0.5, 0.5, 0.5),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            # The bulb keeps the same local orientation as the ceiling bulb;
            # its closest bulb surface is 1 cm from the inclined panel.
            # Shift along the panel normal so the nearest bulb surface is
            # exactly 1 cm from the panel (including the 0.5 asset scale).
            pos=(0.03, -5.041508, 7.609246),
            rot=INCLINED_BULB_ROT,
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
        # Fabric does not propagate a moving rigid body's USD transform to
        # visual children reliably. Spawn the sensor independently and drive
        # its world pose from the physical flight-body truth every sim step.
        prim_path="/World/depth_camera",
        update_period=1.0 / 15.0,
        # Keep the reported camera pose synchronized with the explicitly
        # driven world transform instead of returning the initialization pose.
        update_latest_camera_pose=True,
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
