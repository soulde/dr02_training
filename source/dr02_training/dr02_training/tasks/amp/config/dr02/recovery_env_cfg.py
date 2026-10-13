"""Mixed motion-prior AMP and fall-recovery environment for DR02 Pro."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from dr02_asset.profile import JOINT_ORDER

from ... import mdp
from .env_cfg import AMPEventsCfg, AMPTerminationsCfg, DR02AMPFlatEnvCfg


@configclass
class DR02AMPRecoveryEventsCfg(AMPEventsCfg):
    reset_reference = EventTerm(
        func=mdp.reset_amp_mixed_state,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=list(JOINT_ORDER), preserve_order=True),
            "recovery_probability": 0.15,
            "recovery_root_height_range": (0.18, 0.65),
            "recovery_joint_velocity_range": (-5.0, 5.0),
        },
    )
    recovery_assist_force = EventTerm(
        func=mdp.apply_recovery_assist_force,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base_link"]),
            "max_force": 450.0,
            # 10,000 runner iterations * 24 steps per env per iteration.
            "anneal_steps": 240_000,
        },
    )


@configclass
class DR02AMPRecoveryTerminationsCfg(AMPTerminationsCfg):
    root_height = DoneTerm(
        func=mdp.recovery_root_height_below_minimum,
        params={"minimum_height": 0.48, "window_s": 3.0},
    )
    trunk_contact = DoneTerm(
        func=mdp.recovery_illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["base_link", "body"]),
            "threshold": 10.0,
            "window_s": 3.0,
        },
    )
    recovery_orientation = DoneTerm(
        func=mdp.recovery_bad_orientation,
        params={"limit_angle": 0.7, "window_s": 3.0},
    )
    recovery_deadline = DoneTerm(
        func=mdp.recovery_failed_after_window,
        params={"window_s": 3.0, "minimum_height": 0.48, "max_tilt_angle": 0.7},
    )


@configclass
class DR02AMPRecoveryEnvCfg(DR02AMPFlatEnvCfg):
    """AMP task with 15% random recovery episodes and one shared policy."""

    events: DR02AMPRecoveryEventsCfg = DR02AMPRecoveryEventsCfg()
    terminations: DR02AMPRecoveryTerminationsCfg = DR02AMPRecoveryTerminationsCfg()
