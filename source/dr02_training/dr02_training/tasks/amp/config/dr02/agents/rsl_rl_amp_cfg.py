"""RSL-RL AMP settings for the DR02 Pro flat task."""

import os

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from dr02_asset.profile import AMP_MOTION_BODY_NAMES, JOINT_ORDER

from ..env_cfg import AMP_KEY_BODY_NAMES


@configclass
class RslRlAMPAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "rsl_rl.algorithms:AMP"
    motion_dir: str = os.environ.get("DR02_AMP_MOTION_DIR", "")
    motion_dataset_format: str = "beyondmimic"
    body_names: list[str] = list(AMP_MOTION_BODY_NAMES)
    key_body_names: list[str] = list(AMP_KEY_BODY_NAMES)
    joint_names: list[str] = list(JOINT_ORDER)
    reference_state_initialization: bool = True
    amp_replay_buffer_size: int = 1_000_000
    discriminator_hidden_dims: list[int] = [512, 256]
    discriminator_activation: str = "lrelu"
    discriminator_loss_scale: float = 1.0
    discriminator_logit_regularization_scale: float = 0.0
    discriminator_gradient_penalty_scale: float = 10.0
    discriminator_weight_decay_scale: float = 0.0
    task_reward_lerp: float = 0.65
    amp_reward_coef: float = 0.2


@configclass
class DR02AMPRunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30_000
    save_interval = 500
    clip_actions = 100.0
    experiment_name = "dr02_amp_flat"
    empirical_normalization = False
    obs_groups = {"actor": ["policy"], "critic": ["critic"], "discriminator": ["amp"]}
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        noise_std_type="log",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlAMPAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=5.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.02,
        max_grad_norm=1.0,
    )


@configclass
class DR02AMPRoughRunnerCfg(DR02AMPRunnerCfg):
    """RSL-RL AMP settings for rough terrain, with a separate experiment name."""

    experiment_name = "dr02_amp_rough"


@configclass
class DR02AMPRecoveryRunnerCfg(DR02AMPRunnerCfg):
    """The shared AMP policy configuration for mixed recovery training."""

    experiment_name = "dr02_amp_recovery"
