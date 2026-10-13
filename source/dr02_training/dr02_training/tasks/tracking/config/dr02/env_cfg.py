"""DR02 Pro flat-ground motion tracking environment."""

import os
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp import JointPositionActionCfg
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

from dr02_asset import DR02_PRO_CFG
from dr02_asset.physics_cfg import DR02PhysicsCfg
from dr02_asset.profile import JOINT_ORDER, TRACKING_ANCHOR_BODY, TRACKING_BODY_NAMES
from dr02_training.visualization import configure_sim_visualizer

from ... import mdp


def _default_motion_file() -> str:
    """Read the optional local reference motion path from the environment."""
    return os.environ.get("DR02_TRACKING_MOTION_FILE", "")


@configclass
class TrackingSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
    )
    robot: ArticulationCfg = DR02_PRO_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class TrackingCommandsCfg:
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        motion_file=_default_motion_file() or MISSING,
        joint_names=list(JOINT_ORDER),
        source_body_names=list(mdp.AMP_MOTION_BODY_NAMES),
        source_joint_names=list(JOINT_ORDER),
        anchor_body_name=TRACKING_ANCHOR_BODY,
        body_names=list(TRACKING_BODY_NAMES),
        resampling_time_range=(1.0e9, 1.0e9),
        pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    )


@configclass
class TrackingActionsCfg:
    joint_pos = JointPositionActionCfg(
        asset_name="robot", joint_names=list(JOINT_ORDER), preserve_order=True, use_default_offset=True, scale=0.5
    )


@configclass
class TrackingObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(
            func=mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
            noise=UniformNoiseCfg(n_min=-0.25, n_max=0.25),
        )
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
        )
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
        motion_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TrackingRewardsCfg:
    anchor_pos = RewTerm(
        func=mdp.motion_anchor_position_error, weight=0.5, params={"command_name": "motion", "std": 0.3}
    )
    anchor_ori = RewTerm(
        func=mdp.motion_anchor_orientation_error, weight=0.5, params={"command_name": "motion", "std": 0.4}
    )
    body_pos = RewTerm(func=mdp.motion_body_position_error, weight=1.0, params={"command_name": "motion", "std": 0.3})
    body_ori = RewTerm(
        func=mdp.motion_body_orientation_error, weight=1.0, params={"command_name": "motion", "std": 0.4}
    )
    body_lin_vel = RewTerm(
        func=mdp.motion_body_linear_velocity_error, weight=1.0, params={"command_name": "motion", "std": 1.0}
    )
    body_ang_vel = RewTerm(
        func=mdp.motion_body_angular_velocity_error, weight=1.0, params={"command_name": "motion", "std": 3.14}
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True)},
    )
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[
                    r"^(?!left_ankle_x_link$)(?!right_ankle_x_link$)(?!left_wrist_x_link$)(?!right_wrist_x_link$).+$"
                ],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TrackingTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(func=mdp.bad_anchor_pos_z, params={"command_name": "motion", "threshold": 0.25})
    anchor_ori = DoneTerm(func=mdp.bad_anchor_ori, params={"command_name": "motion", "threshold": 0.8})
    end_effector_pos = DoneTerm(
        func=mdp.bad_body_pos_z,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "body_names": ["left_ankle_x_link", "right_ankle_x_link", "left_wrist_x_link", "right_wrist_x_link"],
        },
    )


@configclass
class DR02TrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene: TrackingSceneCfg = TrackingSceneCfg(num_envs=2048, env_spacing=2.5)
    observations: TrackingObservationsCfg = TrackingObservationsCfg()
    actions: TrackingActionsCfg = TrackingActionsCfg()
    commands: TrackingCommandsCfg = TrackingCommandsCfg()
    rewards: TrackingRewardsCfg = TrackingRewardsCfg()
    terminations: TrackingTerminationsCfg = TrackingTerminationsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics = DR02PhysicsCfg()
        configure_sim_visualizer(self.sim)
