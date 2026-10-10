"""Mixed AMP/recovery sampling and recovery-specific MDP terms."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import mdp as env_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_from_euler_xyz

from .events import reset_amp_reference_state

_ROBOT_CFG = SceneEntityCfg("robot")


def sample_recovery_mask(num_envs: int, probability: float, device: str | torch.device) -> torch.Tensor:
    """Sample which parallel environments start a recovery episode."""
    if not 0.0 <= probability <= 1.0:
        raise ValueError(f"Recovery probability must be in [0, 1], got {probability}")
    return torch.rand(num_envs, device=device) < probability


def recovery_assist_scale(global_step: int, anneal_steps: int) -> float:
    """Linearly anneal the recovery assist from one to zero over global env steps."""
    if anneal_steps <= 0:
        raise ValueError(f"Assist anneal steps must be positive, got {anneal_steps}")
    return max(0.0, 1.0 - max(0, global_step) / anneal_steps)


def reset_amp_mixed_state(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    recovery_probability: float = 0.15,
    recovery_root_height_range: tuple[float, float] = (0.18, 0.65),
    recovery_joint_velocity_range: tuple[float, float] = (-5.0, 5.0),
):
    """Reset sampled environments from AMP motion or a randomized fallen pose."""
    env = env.unwrapped
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    if env_ids.numel() == 0:
        return

    if getattr(env, "_amp_rsi_sampler", None) is None:
        if not getattr(env, "_amp_rsi_initial_reset_deferred", False):
            env._amp_rsi_initial_reset_deferred = True
            return
        raise RuntimeError("AMP mixed reset ran before the runner registered agent.sample_rsi")

    if not hasattr(env, "_amp_recovery_mask"):
        env._amp_recovery_mask = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=env.device)
    recovery_mask = sample_recovery_mask(env_ids.numel(), recovery_probability, env.device)
    recovery_ids = env_ids[recovery_mask]
    motion_ids = env_ids[~recovery_mask]
    env._amp_recovery_mask[env_ids] = recovery_mask

    if motion_ids.numel() > 0:
        reset_amp_reference_state(env, motion_ids, asset_cfg)
    if recovery_ids.numel() > 0:
        _reset_random_recovery_pose(
            env, recovery_ids, asset_cfg, recovery_root_height_range, recovery_joint_velocity_range
        )


def _reset_random_recovery_pose(
    env,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    root_height_range: tuple[float, float],
    joint_velocity_range: tuple[float, float],
):
    """Place recovery environments in random low, tilted, multi-contact states."""
    robot: Articulation = env.scene[asset_cfg.name]
    origins = env.scene.env_origins[env_ids]
    device = robot.device
    dtype = robot.data.root_pos_w.torch.dtype
    count = env_ids.numel()

    root_position = origins.clone()
    root_position[:, :2] += torch.empty((count, 2), device=device, dtype=dtype).uniform_(-0.35, 0.35)
    root_position[:, 2] += torch.empty(count, device=device, dtype=dtype).uniform_(*root_height_range)
    roll_pitch = torch.empty((count, 2), device=device, dtype=dtype).uniform_(-torch.pi, torch.pi)
    yaw = torch.empty(count, device=device, dtype=dtype).uniform_(-torch.pi, torch.pi)
    root_quat = quat_from_euler_xyz(roll_pitch[:, 0], roll_pitch[:, 1], yaw)
    root_pose = torch.cat((root_position, root_quat), dim=-1)
    root_velocity = torch.empty((count, 6), device=device, dtype=dtype)
    root_velocity[:, :3].uniform_(-0.5, 0.5)
    root_velocity[:, 3:].uniform_(-1.0, 1.0)

    joint_limits = robot.data.soft_joint_pos_limits[env_ids]
    joint_positions = joint_limits[..., 0] + torch.rand_like(joint_limits[..., 0]) * (
        joint_limits[..., 1] - joint_limits[..., 0]
    )
    joint_velocities = torch.empty_like(joint_positions).uniform_(*joint_velocity_range)

    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
    robot.write_joint_position_to_sim_index(position=joint_positions, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_velocities, env_ids=env_ids)


def recovery_window_mask(env, window_s: float) -> torch.Tensor:
    """Return recovery environments still inside their grace window."""
    recovery_mask = getattr(env, "_amp_recovery_mask", None)
    if recovery_mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    elapsed = env.episode_length_buf * env.step_dt
    return recovery_mask & (elapsed < window_s)


def recovery_root_height_below_minimum(
    env, minimum_height: float, asset_cfg: SceneEntityCfg = _ROBOT_CFG, window_s: float = 3.0
) -> torch.Tensor:
    """Apply the usual low-root termination outside the recovery grace window."""
    failed = env_mdp.root_height_below_minimum(env, minimum_height, asset_cfg)
    return failed & ~recovery_window_mask(env, window_s)


def recovery_illegal_contact(
    env, threshold: float, sensor_cfg: SceneEntityCfg, window_s: float = 3.0
) -> torch.Tensor:
    """Apply the usual trunk-contact termination outside the recovery grace window."""
    failed = env_mdp.illegal_contact(env, threshold, sensor_cfg)
    return failed & ~recovery_window_mask(env, window_s)


def recovery_bad_orientation(
    env, limit_angle: float, asset_cfg: SceneEntityCfg = _ROBOT_CFG, window_s: float = 3.0
) -> torch.Tensor:
    """Apply the orientation limit to recovery episodes after their grace window."""
    recovery_mask = getattr(env, "_amp_recovery_mask", None)
    if recovery_mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    failed = env_mdp.bad_orientation(env, limit_angle, asset_cfg)
    elapsed = env.episode_length_buf * env.step_dt
    return failed & recovery_mask & (elapsed >= window_s)


def recovery_failed_after_window(
    env,
    window_s: float = 3.0,
    minimum_height: float = 0.48,
    max_tilt_angle: float = 0.7,
    asset_cfg: SceneEntityCfg = _ROBOT_CFG,
) -> torch.Tensor:
    """Terminate a recovery episode at the deadline unless the robot is upright."""
    recovery_mask = getattr(env, "_amp_recovery_mask", None)
    if recovery_mask is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    robot = env.scene[asset_cfg.name]
    elapsed = env.episode_length_buf * env.step_dt
    height = robot.data.root_pos_w.torch[:, 2]
    projected_gravity_z = robot.data.projected_gravity_b.torch[:, 2]
    tilt = torch.acos((-projected_gravity_z).clamp(-1.0, 1.0))
    standing = (height >= minimum_height) & (tilt <= max_tilt_angle)
    return recovery_mask & (elapsed >= window_s) & ~standing


def apply_recovery_assist_force(
    env,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    max_force: float = 200.0,
    anneal_steps: int = 240_000,
):
    """Apply a randomized world-up force to recovery environments during the curriculum."""
    robot: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=robot.device)
    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=robot.device)
    if env_ids.numel() == 0:
        return

    recovery_mask = getattr(env, "_amp_recovery_mask", None)
    if recovery_mask is None:
        recovery_mask = torch.zeros(env.scene.num_envs, dtype=torch.bool, device=robot.device)
    scale = recovery_assist_scale(env.common_step_counter, anneal_steps)
    force = torch.zeros((env_ids.numel(), len(asset_cfg.body_ids), 3), device=robot.device)
    sampled_force = torch.rand(env_ids.numel(), device=robot.device) * max_force * scale
    force[:, :, 2] = sampled_force[:, None] * recovery_mask[env_ids, None]
    torque = torch.zeros_like(force)
    robot.permanent_wrench_composer.set_forces_and_torques_index(
        forces=force,
        torques=torque,
        body_ids=asset_cfg.body_ids,
        env_ids=env_ids,
        is_global=True,
    )
