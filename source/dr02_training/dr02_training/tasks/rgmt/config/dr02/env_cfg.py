"""DR02 Pro RGMT motion-tracking environment configurations."""

import json
import os
from dataclasses import MISSING
from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from dr02_asset.profile import (
    AMP_MOTION_BODY_NAMES,
    JOINT_ORDER,
    TRACKING_ANCHOR_BODY,
    TRACKING_BODY_NAMES,
)
from dr02_training.tasks.tracking.config.dr02.env_cfg import (
    DR02TrackingEnvCfg,
    TrackingRewardsCfg,
    TrackingTerminationsCfg,
)

from ... import mdp

RGMT_DEPLOYMENT_CONTRACT_PATH = Path(__file__).resolve().parents[2] / "deploy" / "rgmt" / "flat.json"
RGMT_DEPLOYMENT_CONTRACT = json.loads(RGMT_DEPLOYMENT_CONTRACT_PATH.read_text(encoding="utf-8"))
if tuple(RGMT_DEPLOYMENT_CONTRACT["control"]["action_joint_order"]) != JOINT_ORDER:
    raise ValueError(
        f"RGMT deployment joint order does not match the DR02 asset profile: {RGMT_DEPLOYMENT_CONTRACT_PATH}"
    )


def _default_motion_path() -> str:
    """Read the optional local motion corpus path from the environment."""
    return os.environ.get("DR02_RGMT_MOTION_PATH", "")


@configclass
class RGMTCommandsCfg:
    motion = mdp.RGMTMotionCommandCfg(
        asset_name="robot",
        motion_path=_default_motion_path() or MISSING,
        joint_names=list(JOINT_ORDER),
        source_body_names=list(AMP_MOTION_BODY_NAMES),
        source_joint_names=list(JOINT_ORDER),
        anchor_body_name=TRACKING_ANCHOR_BODY,
        body_names=list(TRACKING_BODY_NAMES),
        resampling_time_range=(1.0e9, 1.0e9),
        pose_range={"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
    )


@configclass
class RGMTActionsCfg:
    residual_joint_pos = mdp.RGMTJointPositionActionCfg(
        asset_name="robot",
        joint_names=list(JOINT_ORDER),
        preserve_order=True,
        scale=float(RGMT_DEPLOYMENT_CONTRACT["control"]["action_scale_rad"]),
        reference_command_name="motion",
        recovery_grace_period_s=3.0,
    )


@configclass
class RGMTObservationsCfg:
    @configclass
    class ProprioCurrentCfg(ObsGroup):
        observation = ObsTerm(func=mdp.rgmt_proprioceptive_frame)

    @configclass
    class ProprioHistoryCfg(ObsGroup):
        observation = ObsTerm(func=mdp.rgmt_proprioceptive_history)

    @configclass
    class CommandWindowCfg(ObsGroup):
        observation = ObsTerm(func=mdp.rgmt_command_window, params={"command_name": "motion"})

    @configclass
    class CriticCurrentCfg(ObsGroup):
        observation = ObsTerm(func=mdp.rgmt_clean_proprioceptive_frame)

    @configclass
    class CriticHistoryCfg(ObsGroup):
        observation = ObsTerm(func=mdp.rgmt_clean_proprioceptive_history)

    @configclass
    class CriticPrivilegedCfg(ObsGroup):
        observation = ObsTerm(func=mdp.rgmt_privileged_body_state, params={"command_name": "motion"})

    proprio_current: ProprioCurrentCfg = ProprioCurrentCfg()
    proprio_history: ProprioHistoryCfg = ProprioHistoryCfg()
    command_window: CommandWindowCfg = CommandWindowCfg()
    critic_current: CriticCurrentCfg = CriticCurrentCfg()
    critic_history: CriticHistoryCfg = CriticHistoryCfg()
    critic_privileged: CriticPrivilegedCfg = CriticPrivilegedCfg()

    def __post_init__(self):
        for group in (
            self.proprio_current,
            self.proprio_history,
            self.command_window,
            self.critic_current,
            self.critic_history,
            self.critic_privileged,
        ):
            group.enable_corruption = False


@configclass
class RGMTTerminationsCfg(TrackingTerminationsCfg):
    anchor_pos = DoneTerm(
        func=mdp.rgmt_bad_anchor_pos_z,
        params={"command_name": "motion", "threshold": 0.25, "grace_period_s": 3.0},
    )
    anchor_ori = DoneTerm(
        func=mdp.rgmt_bad_anchor_ori,
        params={"command_name": "motion", "threshold": 0.8, "grace_period_s": 3.0},
    )
    end_effector_pos = DoneTerm(
        func=mdp.rgmt_bad_body_pos_z,
        params={
            "command_name": "motion",
            "threshold": 0.25,
            "grace_period_s": 3.0,
            "body_names": ["left_ankle_x_link", "right_ankle_x_link", "left_wrist_x_link", "right_wrist_x_link"],
        },
    )


@configclass
class DR02RGMTFlatEnvCfg(DR02TrackingEnvCfg):
    observations: RGMTObservationsCfg = RGMTObservationsCfg()
    actions: RGMTActionsCfg = RGMTActionsCfg()
    commands: RGMTCommandsCfg = RGMTCommandsCfg()
    rewards: TrackingRewardsCfg = TrackingRewardsCfg()
    terminations: RGMTTerminationsCfg = RGMTTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 2048


@configclass
class DR02RGMTFlatPlayEnvCfg(DR02RGMTFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
