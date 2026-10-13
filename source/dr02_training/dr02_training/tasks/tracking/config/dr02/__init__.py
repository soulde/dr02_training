"""DR02 motion tracking environment registration."""

import gymnasium as gym

gym.register(
    id="Dr02Training-Tracking-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DR02TrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_tracking_cfg:DR02TrackingRunnerCfg",
        "default_agent": "rsl_rl",
    },
)
