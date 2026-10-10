"""DR02 AMP environment registration."""

import gymnasium as gym

gym.register(
    id="Dr02Training-AMP-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:DR02AMPFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_cfg:DR02AMPRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Dr02Training-AMP-Rough-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:DR02AMPRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_cfg:DR02AMPRoughRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Dr02Training-AMP-Recovery-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.recovery_env_cfg:DR02AMPRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_cfg:DR02AMPRecoveryRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Dr02Training-AMP-Flat-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:DR02AMPCameraFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_cfg:DR02AMPRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Dr02Training-AMP-Rough-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:DR02AMPCameraRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_cfg:DR02AMPRoughRunnerCfg",
        "default_agent": "rsl_rl",
    },
)

gym.register(
    id="Dr02Training-AMP-Recovery-Camera-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.camera_env_cfg:DR02AMPCameraRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_amp_cfg:DR02AMPRecoveryRunnerCfg",
        "default_agent": "rsl_rl",
    },
)
