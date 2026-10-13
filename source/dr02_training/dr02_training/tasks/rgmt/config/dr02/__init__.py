"""Gym registration for DR02 RGMT motion-tracking environments."""

import gymnasium as gym

gym.register(
    id="Dr02Training-RGMT-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DR02RGMTFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_rgmt_cfg:DR02RGMTOnPolicyRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Dr02Training-RGMT-Flat-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DR02RGMTFlatPlayEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_rgmt_cfg:DR02RGMTOnPolicyRunnerCfg",
        "default_agent": "rsl_rl",
    },
)
