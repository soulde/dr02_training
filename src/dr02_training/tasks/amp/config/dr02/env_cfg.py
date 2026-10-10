"""DR02 Pro flat-ground AMP locomotion environment."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import UniformVelocityCommandCfg
from isaaclab.envs.mdp.rewards import track_ang_vel_z_exp
from isaaclab.managers import (
    EventTermCfg as EventTerm,
)
from isaaclab.managers import (
    ObservationGroupCfg as ObsGroup,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    SceneEntityCfg,
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
)
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg

from isaaclab_tasks.core.velocity.mdp.rewards import track_lin_vel_xy_yaw_frame_exp

from dr02_training.assets.dr02 import DR02_PRO_CFG
from dr02_training.assets.dr02.physics_cfg import DR02PhysicsCfg
from dr02_training.assets.dr02.profile import JOINT_ORDER

from ... import mdp

AMP_KEY_BODY_NAMES = ["left_ankle_y_link", "left_toe_link", "right_ankle_y_link", "right_toe_link"]
AMP_OBSERVED_BODY_NAMES = ["left_ankle_y_link", "left_ankle_x_link", "right_ankle_y_link", "right_ankle_x_link"]
AMP_VIRTUAL_BODY_OFFSETS = {
    "left_toe_link": ("left_ankle_x_link", (0.11, 0.0, -0.015)),
    "right_toe_link": ("right_ankle_x_link", (0.11, 0.0, -0.015)),
}


@configclass
class AMPSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot = DR02_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class AMPCommandsCfg:
    base_velocity = UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(8.0, 12.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-2.0, 4.0), lin_vel_y=(-2.0, 2.0), ang_vel_z=(-1.0, 1.0), heading=(-3.14, 3.14)
        ),
    )


@configclass
class AMPActionCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot", joint_names=list(JOINT_ORDER), preserve_order=True, use_default_offset=True, scale=0.5
    )


@configclass
class AMPObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
            noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
            noise=UniformNoiseCfg(n_min=-0.5, n_max=0.5),
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)

    @configclass
    class AMPCfg(ObsGroup):
        root_height = None
        root_orientation = None
        root_linear_velocity = ObsTerm(func=mdp.amp_root_linear_velocity)
        root_angular_velocity = ObsTerm(func=mdp.amp_root_angular_velocity)
        joint_position = ObsTerm(
            func=mdp.amp_joint_position,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
        )
        joint_velocity = ObsTerm(
            func=mdp.amp_joint_velocity,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
        )
        link_positions = ObsTerm(
            func=mdp.amp_link_positions,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=AMP_OBSERVED_BODY_NAMES, preserve_order=True),
                "key_body_names": AMP_KEY_BODY_NAMES,
                "virtual_body_offsets": AMP_VIRTUAL_BODY_OFFSETS,
            },
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AMPCfg = AMPCfg()


@configclass
class AMPEventsCfg:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True),
            "position_range": (-0.08, 0.08),
            "velocity_range": (-0.5, 0.5),
        },
    )
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.1, 0.1),
                "roll": (-0.2, 0.2),
                "pitch": (-0.2, 0.2),
                "yaw": (-0.2, 0.2),
            },
        },
    )
    reset_reference = EventTerm(
        func=mdp.reset_amp_reference_state,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
    )


@configclass
class AMPRewardsCfg:
    track_lin_vel_xy = RewTerm(
        func=track_lin_vel_xy_yaw_frame_exp,
        weight=4.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    track_ang_vel_z = RewTerm(
        func=track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "base_velocity", "std": 0.25},
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-200.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_acc = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
    )
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-6,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
    )


@configclass
class AMPTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    root_height = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.48})
    trunk_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "body"]), "threshold": 10.0},
    )


@configclass
class DR02AMPFlatEnvCfg(ManagerBasedRLEnvCfg):
    scene: AMPSceneCfg = AMPSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: AMPObservationsCfg = AMPObservationsCfg()
    actions: AMPActionCfg = AMPActionCfg()
    commands: AMPCommandsCfg = AMPCommandsCfg()
    events: AMPEventsCfg = AMPEventsCfg()
    rewards: AMPRewardsCfg = AMPRewardsCfg()
    terminations: AMPTerminationsCfg = AMPTerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 40.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = DR02PhysicsCfg()
