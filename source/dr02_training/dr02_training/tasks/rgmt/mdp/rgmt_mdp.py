"""RGMT observations, actions, and recovery terms for DR02 Pro."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.joint_actions import JointAction
from isaaclab.managers import ActionTermCfg
from isaaclab.utils import configclass

from dr02_asset.profile import JOINT_ORDER

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


PROPRIOCEPTIVE_NOISE_RANGES = (0.05, 0.2, 0.01, 0.5)
"""Uniform half-ranges for gravity, angular velocity, joint position, and joint velocity noise."""


def sample_recovery_mask(
    num_envs: int,
    recovery_fraction: float,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample recovery starts independently at the configured per-reset rate."""
    if num_envs < 0:
        raise ValueError(f"num_envs must be non-negative, got {num_envs}")
    if not 0.0 <= recovery_fraction <= 1.0:
        raise ValueError(f"recovery_fraction must be in [0, 1], got {recovery_fraction}")
    return torch.rand(num_envs, device=device, generator=generator) < recovery_fraction


def assistance_magnitude(step: int, maximum: float, anneal_steps: int) -> float:
    """Linearly anneal recovery support from ``maximum`` to zero."""
    if maximum < 0.0:
        raise ValueError(f"maximum assistance must be non-negative, got {maximum}")
    if anneal_steps <= 0:
        raise ValueError(f"anneal_steps must be positive, got {anneal_steps}")
    fraction = min(max(step, 0) / anneal_steps, 1.0)
    return maximum * (1.0 - fraction)


def sample_recovery_assistance(
    recovery_mask: torch.Tensor,
    maximum: float,
    step: int,
    anneal_steps: int,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample an independently randomized, annealed upward force per recovery environment."""
    if recovery_mask.ndim != 1 or recovery_mask.dtype != torch.bool:
        raise ValueError("recovery_mask must be a rank-1 bool tensor")
    maximum_now = assistance_magnitude(step, maximum, anneal_steps)
    samples = torch.rand(recovery_mask.shape, device=recovery_mask.device, generator=generator) * maximum_now
    return torch.where(recovery_mask, samples, torch.zeros_like(samples))


def apply_recovery_grace_period(
    terminated: torch.Tensor,
    recovery_mask: torch.Tensor,
    elapsed_seconds: torch.Tensor,
    grace_period_s: float,
) -> torch.Tensor:
    """Suppress recovery termination signals strictly before their grace expires."""
    if terminated.shape != recovery_mask.shape or terminated.shape != elapsed_seconds.shape:
        raise ValueError("termination, recovery, and elapsed-time tensors must have matching shapes")
    if recovery_mask.dtype != torch.bool:
        raise ValueError("recovery_mask must be a bool tensor")
    if grace_period_s < 0.0:
        raise ValueError(f"grace_period_s must be non-negative, got {grace_period_s}")
    grace_active = recovery_mask & (elapsed_seconds < grace_period_s)
    return terminated & ~grace_active


def recovery_assistance_mask(
    recovery_mask: torch.Tensor, elapsed_seconds: torch.Tensor, assistance_duration_s: float
) -> torch.Tensor:
    """Select recovering environments that still receive support assistance."""
    if recovery_mask.shape != elapsed_seconds.shape or recovery_mask.dtype != torch.bool:
        raise ValueError("recovery_mask and elapsed_seconds must have matching shapes and bool recovery_mask")
    if assistance_duration_s < 0.0:
        raise ValueError(f"assistance_duration_s must be non-negative, got {assistance_duration_s}")
    return recovery_mask & (elapsed_seconds < assistance_duration_s)


def update_recovery_assistance(
    asset,
    command,
    elapsed_seconds: torch.Tensor,
    previous_env_ids: torch.Tensor,
    grace_period_s: float,
) -> torch.Tensor:
    """Refresh RGMT's transient upward support wrench and return active env IDs."""
    active_mask = recovery_assistance_mask(command.recovery_mask, elapsed_seconds, grace_period_s)
    active_ids = torch.where(active_mask)[0]
    clear_ids = torch.unique(torch.cat((previous_env_ids, active_ids)))
    composer = asset.permanent_wrench_composer
    if clear_ids.numel():
        composer.reset(env_ids=clear_ids)
    if active_ids.numel():
        force = torch.zeros(active_ids.numel(), 1, 3, device=elapsed_seconds.device)
        force[..., 2] = command.recovery_assistance[active_ids].unsqueeze(-1)
        composer.add_forces_and_torques_index(
            forces=force,
            torques=torch.zeros_like(force),
            body_ids=[command.robot_anchor_body_index],
            env_ids=active_ids,
            is_global=True,
        )
    return active_ids


def build_proprioceptive_frame(
    projected_gravity: torch.Tensor,
    base_angular_velocity: torch.Tensor,
    relative_joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    previous_action: torch.Tensor,
) -> torch.Tensor:
    """Concatenate RGMT proprioception in the paper's feature order."""
    features = (
        projected_gravity,
        base_angular_velocity,
        relative_joint_position,
        joint_velocity,
        previous_action,
    )
    if any(value.ndim != 2 for value in features):
        raise ValueError("proprioceptive features must all have shape [batch, feature]")
    if any(value.shape[0] != projected_gravity.shape[0] for value in features):
        raise ValueError("proprioceptive features must have the same batch dimension")
    if projected_gravity.shape[1] != 3 or base_angular_velocity.shape[1] != 3:
        raise ValueError("projected gravity and base angular velocity must each have width 3")
    joint_width = relative_joint_position.shape[1]
    if joint_velocity.shape[1] != joint_width or previous_action.shape[1] != joint_width:
        raise ValueError("joint position, joint velocity, and previous action widths must match")
    return torch.cat(features, dim=-1)


def update_proprioceptive_history(
    history: torch.Tensor | None,
    current_frame: torch.Tensor,
    reset_mask: torch.Tensor | None = None,
    history_length: int = 10,
) -> torch.Tensor:
    """Append a current frame, repeat it on reset, or initialize a repeated history."""
    if current_frame.ndim != 2:
        raise ValueError("current_frame must have shape [batch, feature]")
    if history_length <= 0:
        raise ValueError(f"history_length must be positive, got {history_length}")
    batch_size = current_frame.shape[0]
    if reset_mask is None:
        reset_mask = torch.zeros(batch_size, dtype=torch.bool, device=current_frame.device)
    if reset_mask.shape != (batch_size,) or reset_mask.dtype != torch.bool:
        raise ValueError("reset_mask must be a bool tensor with shape [batch]")

    repeated = current_frame.unsqueeze(1).expand(-1, history_length, -1)
    if history is None:
        return repeated.clone()
    if history.shape != (batch_size, history_length, current_frame.shape[1]):
        raise ValueError(
            f"history must have shape {(batch_size, history_length, current_frame.shape[1])}, "
            f"got {tuple(history.shape)}"
        )
    shifted = torch.cat((history[:, 1:], current_frame.unsqueeze(1)), dim=1)
    return torch.where(reset_mask[:, None, None], repeated, shifted)


def align_joint_values(
    values: torch.Tensor, source_joint_names: Sequence[str], target_joint_names: Sequence[str]
) -> torch.Tensor:
    """Reorder a rank-2 joint tensor from source names into target name order."""
    if values.ndim != 2 or values.shape[1] != len(source_joint_names):
        raise ValueError("joint values width must match source_joint_names")
    if len(set(source_joint_names)) != len(source_joint_names):
        raise ValueError("source joint names must be unique")
    if len(set(target_joint_names)) != len(target_joint_names):
        raise ValueError("target joint names must be unique")
    missing = [name for name in target_joint_names if name not in source_joint_names]
    if missing:
        raise ValueError(f"missing joint names: {missing}")
    indexes = torch.tensor([source_joint_names.index(name) for name in target_joint_names], device=values.device)
    return values.index_select(1, indexes)


def residual_joint_targets(
    reference_joint_position: torch.Tensor, residual_action: torch.Tensor, action_scale: float | torch.Tensor
) -> torch.Tensor:
    """Map residual actions to joint targets around the corresponding reference pose."""
    if reference_joint_position.ndim != 2 or residual_action.ndim != 2:
        raise ValueError("reference positions and residual actions must be rank-2 tensors")
    if reference_joint_position.shape != residual_action.shape:
        raise ValueError("reference positions and residual actions must have the same shape and joint order")
    return reference_joint_position + action_scale * residual_action


def _tensor(value: torch.Tensor) -> torch.Tensor:
    """Unwrap an Isaac Lab proxy array when the environment uses proxy-backed state."""
    return value.torch if hasattr(value, "torch") else value


def rgmt_proprioceptive_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return a shared, noisy DR02 proprioceptive frame for this environment step."""
    step = env.common_step_counter
    cached_frame = getattr(env, "_rgmt_proprio_frame", None)
    same_step = getattr(env, "_rgmt_proprio_step", None) == step
    episode_lengths = env.episode_length_buf
    if same_step and cached_frame is not None:
        previous_episode_lengths = env._rgmt_proprio_episode_lengths
        reset_mask = (episode_lengths == 0) & (previous_episode_lengths != 0)
        if not torch.any(reset_mask):
            return cached_frame

    robot = env.scene["robot"]
    data = robot.data
    joint_ids = getattr(env, "_rgmt_joint_ids", None)
    if joint_ids is None:
        joint_ids, _ = robot.find_joints(list(JOINT_ORDER), preserve_order=True)
        joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=env.device)
        env._rgmt_joint_ids = joint_ids
    gravity = _tensor(data.projected_gravity_b)
    angular_velocity = _tensor(data.root_ang_vel_b)
    relative_joint_position = _tensor(data.joint_pos)[:, joint_ids] - _tensor(data.default_joint_pos)[:, joint_ids]
    joint_velocity = _tensor(data.joint_vel)[:, joint_ids]
    previous_action = env.action_manager.action
    ranges = PROPRIOCEPTIVE_NOISE_RANGES
    noisy_features = []
    for feature, half_range in zip(
        (gravity, angular_velocity, relative_joint_position, joint_velocity), ranges, strict=True
    ):
        noisy_features.append(feature + (torch.rand_like(feature) * 2.0 - 1.0) * half_range)
    frame = build_proprioceptive_frame(*noisy_features, previous_action)
    if same_step and cached_frame is not None:
        cached_frame[reset_mask] = frame[reset_mask]
        frame = cached_frame
    env._rgmt_proprio_step = step
    env._rgmt_proprio_frame = frame
    env._rgmt_proprio_episode_lengths = episode_lengths.clone()
    return frame


def rgmt_proprioceptive_history(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the 10-step noisy proprioceptive history, initialized at episode reset."""
    frame = rgmt_proprioceptive_frame(env)
    step = env.common_step_counter
    history = getattr(env, "_rgmt_proprio_history", None)
    last_step = getattr(env, "_rgmt_history_step", None)
    if history is None:
        history = update_proprioceptive_history(None, frame)
    else:
        episode_lengths = env.episode_length_buf
        previous_episode_lengths = env._rgmt_history_episode_lengths
        reset_mask = (episode_lengths == 0) & (previous_episode_lengths != 0)
        if last_step != step:
            reset_mask = episode_lengths == 0
        if last_step != step or torch.any(reset_mask):
            history = update_proprioceptive_history(history, frame, reset_mask)
    env._rgmt_proprio_history = history
    env._rgmt_history_step = step
    env._rgmt_history_episode_lengths = env.episode_length_buf.clone()
    return history


def rgmt_clean_proprioceptive_frame(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return an uncorrupted proprioceptive frame for the critic."""
    robot = env.scene["robot"]
    data = robot.data
    joint_ids = getattr(env, "_rgmt_joint_ids", None)
    if joint_ids is None:
        joint_ids, _ = robot.find_joints(list(JOINT_ORDER), preserve_order=True)
        joint_ids = torch.as_tensor(joint_ids, dtype=torch.long, device=env.device)
        env._rgmt_joint_ids = joint_ids
    return build_proprioceptive_frame(
        _tensor(data.projected_gravity_b),
        _tensor(data.root_ang_vel_b),
        _tensor(data.joint_pos)[:, joint_ids] - _tensor(data.default_joint_pos)[:, joint_ids],
        _tensor(data.joint_vel)[:, joint_ids],
        env.action_manager.action,
    )


def rgmt_clean_proprioceptive_history(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Maintain a clean critic history independently from the noisy actor history."""
    frame = rgmt_clean_proprioceptive_frame(env)
    step = env.common_step_counter
    history = getattr(env, "_rgmt_clean_history", None)
    last_step = getattr(env, "_rgmt_clean_history_step", None)
    if history is None:
        history = update_proprioceptive_history(None, frame)
    else:
        episode_lengths = env.episode_length_buf
        previous_episode_lengths = env._rgmt_clean_history_episode_lengths
        reset_mask = (episode_lengths == 0) & (previous_episode_lengths != 0)
        if last_step != step:
            reset_mask = episode_lengths == 0
        if last_step != step or torch.any(reset_mask):
            history = update_proprioceptive_history(history, frame, reset_mask)
    env._rgmt_clean_history = history
    env._rgmt_clean_history_step = step
    env._rgmt_clean_history_episode_lengths = env.episode_length_buf.clone()
    return history


def rgmt_privileged_body_state(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Expose clean tracked-body pose and root velocities to the asymmetric critic.

    Body positions are expressed relative to the robot tracking anchor. This keeps
    the feature independent of each vectorized environment's world origin.
    """
    command = env.command_manager.get_term(command_name)
    position = command.robot_body_pos_w - command.robot_anchor_pos_w[:, None, :]
    orientation = command.robot_body_quat_w
    root_velocity = torch.cat(
        (
            _tensor(env.scene["robot"].data.root_lin_vel_b),
            _tensor(env.scene["robot"].data.root_ang_vel_b),
        ),
        dim=-1,
    )
    return torch.cat((position.flatten(1), orientation.flatten(1), root_velocity), dim=-1)


def rgmt_command_window(env: ManagerBasedRLEnv, command_name: str = "motion") -> torch.Tensor:
    """Return the motion command's clip-bounded 21-frame window."""
    command = env.command_manager.get_term(command_name)
    return command.get_command_window()


class RGMTJointPositionAction(JointAction):
    """Apply scaled residual actions around each environment's current reference pose."""

    cfg: RGMTJointPositionActionCfg

    def __init__(self, cfg: RGMTJointPositionActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._previous_assisted_env_ids = torch.empty(0, dtype=torch.long, device=self.device)

    def apply_actions(self) -> None:
        reference_command = self._env.command_manager.get_term(self.cfg.reference_command_name)
        reference = reference_command.joint_pos[:, self._joint_ids]
        targets = residual_joint_targets(reference, self.processed_actions, action_scale=1.0)
        self._asset.set_joint_position_target_index(target=targets, joint_ids=self._joint_ids)

        grace_period_s = self.cfg.recovery_grace_period_s
        elapsed = self._env.episode_length_buf * self._env.step_dt
        self._previous_assisted_env_ids = update_recovery_assistance(
            self._asset,
            reference_command,
            elapsed,
            self._previous_assisted_env_ids,
            grace_period_s,
        )


@configclass
class RGMTJointPositionActionCfg(ActionTermCfg):
    """Configuration for reference-relative residual joint position control."""

    class_type: type = RGMTJointPositionAction
    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    scale: float | dict[str, float] = 1.0
    offset: float | dict[str, float] = 0.0
    preserve_order: bool = True
    reference_command_name: str = "motion"
    recovery_grace_period_s: float = 3.0


def _recovery_aware_termination(
    env: ManagerBasedRLEnv, failed: torch.Tensor, grace_period_s: float, command_name: str = "motion"
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    elapsed = env.episode_length_buf * env.step_dt
    return apply_recovery_grace_period(failed, command.recovery_mask, elapsed, grace_period_s)


def rgmt_bad_anchor_pos_z(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, grace_period_s: float = 3.0
) -> torch.Tensor:
    """Tracking anchor-height failure, deferred during recovery grace."""
    from .tracking_mdp import bad_anchor_pos_z

    return _recovery_aware_termination(
        env, bad_anchor_pos_z(env, command_name, threshold), grace_period_s, command_name
    )


def rgmt_bad_anchor_ori(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, grace_period_s: float = 3.0
) -> torch.Tensor:
    """Tracking anchor-orientation failure, deferred during recovery grace."""
    from .tracking_mdp import bad_anchor_ori

    return _recovery_aware_termination(
        env, bad_anchor_ori(env, command_name, threshold), grace_period_s, command_name
    )


def rgmt_bad_body_pos_z(
    env: ManagerBasedRLEnv,
    command_name: str,
    threshold: float,
    body_names: list[str] | None = None,
    grace_period_s: float = 3.0,
) -> torch.Tensor:
    """Tracked end-effector height failure, deferred during recovery grace."""
    from .tracking_mdp import bad_body_pos_z

    return _recovery_aware_termination(
        env, bad_body_pos_z(env, command_name, threshold, body_names), grace_period_s, command_name
    )
