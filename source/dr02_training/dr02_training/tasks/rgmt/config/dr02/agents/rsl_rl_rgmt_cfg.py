"""PPO configuration for DR02 RGMT's Transformer actor and asymmetric critic."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlMLPModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class RslRlRGMTModelCfg(RslRlMLPModelCfg):
    """Model settings consumed by the custom RGMT actor or critic."""

    class_name: str = ""
    current_groups: list[str] = []
    history_group: str = ""
    command_group: str = "command_window"
    embedding_dim: int = 128
    num_heads: int = 4
    history_layers: int = 1
    feedforward_dim: int = 256
    dropout: float = 0.0


@configclass
class DR02RGMTOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner defaults for DR02 RGMT motion tracking."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "dr02_rgmt"
    empirical_normalization = False
    obs_groups = {
        "actor": ["proprio_current", "proprio_history", "command_window"],
        "critic": ["critic_current", "critic_history", "command_window", "critic_privileged"],
    }
    actor = RslRlRGMTModelCfg(
        class_name=(
            "dr02_training.tasks.rgmt.mdp.rgmt_policy:RGMTActor"
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(
            init_std=0.8,
            std_type="scalar",
        ),
        stochastic=True,
        init_noise_std=0.8,
        current_groups=["proprio_current"],
        history_group="proprio_history",
    )
    critic = RslRlRGMTModelCfg(
        class_name=(
            "dr02_training.tasks.rgmt.mdp.rgmt_policy:RGMTCritic"
        ),
        hidden_dims=[256, 128],
        activation="elu",
        obs_normalization=False,
        stochastic=False,
        init_noise_std=0.0,
        current_groups=["critic_current", "critic_privileged"],
        history_group="critic_history",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
