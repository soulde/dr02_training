"""Motion-tracking MDP terms for DR02 Pro."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
    yaw_quat,
)

from dr02_asset.profile import AMP_MOTION_BODY_NAMES, JOINT_ORDER, TRACKING_ANCHOR_BODY

from .tracking_motion import (
    align_motion_arrays,
    convert_body_quat_wxyz_to_xyzw,
    expand_reference_quat,
    expand_reference_transform,
    mean_body_squared_error,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MotionLoader:
    """Load a BeyondMimic NPZ and align it to the configured Chocolate names."""

    def __init__(
        self,
        motion_file: str,
        body_names: Sequence[str],
        joint_names: Sequence[str],
        device: str,
        source_body_names: Sequence[str] | None = None,
        source_joint_names: Sequence[str] | None = None,
    ) -> None:
        data = np.load(motion_file, allow_pickle=True)
        body_metadata = [str(name) for name in data["body_names"]] if "body_names" in data else source_body_names
        joint_metadata = [str(name) for name in data["joint_names"]] if "joint_names" in data else source_joint_names
        aligned = align_motion_arrays(
            body_pos=data["body_pos_w"],
            body_quat=data["body_quat_w"],
            body_lin_vel=data["body_lin_vel_w"],
            body_ang_vel=data["body_ang_vel_w"],
            body_names=body_metadata,
            target_body_names=body_names,
            joint_pos=data["joint_pos"],
            joint_vel=data["joint_vel"],
            joint_names=joint_metadata,
            target_joint_names=joint_names,
        )
        self.fps = float(data["fps"])
        self.joint_pos = torch.as_tensor(aligned.joint_pos, dtype=torch.float32, device=device)
        self.joint_vel = torch.as_tensor(aligned.joint_vel, dtype=torch.float32, device=device)
        self.body_pos_w = torch.as_tensor(aligned.body_pos, dtype=torch.float32, device=device)
        self.body_quat_w = torch.as_tensor(
            convert_body_quat_wxyz_to_xyzw(aligned.body_quat), dtype=torch.float32, device=device
        )
        self.body_lin_vel_w = torch.as_tensor(aligned.body_lin_vel, dtype=torch.float32, device=device)
        self.body_ang_vel_w = torch.as_tensor(aligned.body_ang_vel, dtype=torch.float32, device=device)
        self.time_step_total = self.joint_pos.shape[0]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        body_ids, _ = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        self.body_indexes = torch.as_tensor(body_ids, dtype=torch.long, device=self.device)
        joint_ids, _ = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        self.joint_indexes = torch.as_tensor(joint_ids, dtype=torch.long, device=self.device)
        self.motion = MotionLoader(
            cfg.motion_file,
            cfg.body_names,
            cfg.joint_names,
            self.device,
            source_body_names=cfg.source_body_names,
            source_joint_names=cfg.source_joint_names,
        )
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[..., 3] = 1.0
        self.metrics.update(
            {
                name: torch.zeros(self.num_envs, device=self.device)
                for name in (
                    "error_anchor_pos",
                    "error_anchor_rot",
                    "error_body_pos",
                    "error_body_rot",
                    "error_body_lin_vel",
                    "error_body_ang_vel",
                    "error_joint_pos",
                    "error_joint_vel",
                )
            }
        )
        self.bin_count = max(1, int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1)
        self.bin_failed_count = torch.zeros(self.bin_count, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        self.time_steps[env_ids] = torch.randint(self.motion.time_step_total, (len(env_ids),), device=self.device)
        root_pos = self.anchor_pos_w[env_ids].clone()
        root_ori = self.anchor_quat_w[env_ids].clone()
        ranges = torch.tensor(
            [self.cfg.pose_range.get(k, (0.0, 0.0)) for k in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=self.device,
        )
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos += noise[:, :3]
        root_ori[:] = quat_mul(quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), root_ori)
        root_pose = torch.cat((root_pos, root_ori), dim=-1)
        root_velocity = torch.cat(
            (
                self.body_lin_vel_w[env_ids, self.motion_anchor_body_index],
                self.body_ang_vel_w[env_ids, self.motion_anchor_body_index],
            ),
            dim=-1,
        )
        self.robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
        self.robot.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)
        joint_pos = self.joint_pos[env_ids]
        joint_vel = self.joint_vel[env_ids]
        self.robot.write_joint_position_to_sim_index(position=joint_pos, joint_ids=self.joint_indexes, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, joint_ids=self.joint_indexes, env_ids=env_ids)

    def _update_command(self) -> None:
        self.time_steps += 1
        env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
        self._resample_command(env_ids)
        anchor_pos = self.anchor_pos_w
        anchor_quat = self.anchor_quat_w
        robot_anchor_pos = self.robot_anchor_pos_w
        robot_anchor_quat = self.robot_anchor_quat_w
        delta_ori = yaw_quat(quat_mul(robot_anchor_quat, quat_inv(anchor_quat)))
        delta_pos = robot_anchor_pos.clone()
        delta_pos[:, 2] = anchor_pos[:, 2]
        self.body_quat_relative_w = quat_mul(expand_reference_quat(delta_ori, self.body_quat_w), self.body_quat_w)
        self.body_pos_relative_w = delta_pos[:, None] + quat_apply(
            expand_reference_quat(delta_ori, self.body_quat_w), self.body_pos_w - anchor_pos[:, None]
        )

    def _update_metrics(self) -> None:
        self.metrics["error_anchor_pos"] = torch.linalg.vector_norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1)
        self.metrics["error_anchor_rot"] = quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w)
        self.metrics["error_body_pos"] = torch.linalg.vector_norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(-1)
        self.metrics["error_body_rot"] = quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(
            -1
        )
        self.metrics["error_body_lin_vel"] = torch.linalg.vector_norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(-1)
        self.metrics["error_body_ang_vel"] = torch.linalg.vector_norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(-1)
        self.metrics["error_joint_pos"] = torch.linalg.vector_norm(
            self.joint_pos - self.robot.data.joint_pos[:, self.joint_indexes], dim=-1
        )
        self.metrics["error_joint_vel"] = torch.linalg.vector_norm(
            self.joint_vel - self.robot.data.joint_vel[:, self.joint_indexes], dim=-1
        )


@configclass
class MotionCommandCfg(CommandTermCfg):
    class_type: type = MotionCommand
    asset_name: str = MISSING
    motion_file: str = MISSING
    joint_names: list[str] = list(JOINT_ORDER)
    source_body_names: list[str] = list(AMP_MOTION_BODY_NAMES)
    source_joint_names: list[str] = list(JOINT_ORDER)
    anchor_body_name: str = TRACKING_ANCHOR_BODY
    body_names: list[str] = MISSING
    pose_range: dict[str, tuple[float, float]] = {}
    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")


def generated_commands(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    return env.command_manager.get_command(command_name)


def joint_pos_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=list(JOINT_ORDER))
    asset = env.scene[asset_cfg.name]
    return asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]


def joint_vel_rel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg | None = None) -> torch.Tensor:
    if asset_cfg is None:
        asset_cfg = SceneEntityCfg("robot", joint_names=list(JOINT_ORDER))
    return env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]


def last_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    return env.action_manager.action


def motion_anchor_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return pos


def motion_anchor_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.anchor_pos_w, command.anchor_quat_w
    )
    return matrix_from_quat(ori)[..., :2].reshape(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    reference_pos, reference_quat, body_pos_w = expand_reference_transform(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.robot_body_pos_w
    )
    body_quat_w = command.robot_body_quat_w.reshape(-1, 4)
    pos, _ = subtract_frame_transforms(reference_pos, reference_quat, body_pos_w, body_quat_w)
    return pos.reshape(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    reference_pos, reference_quat, body_pos_w = expand_reference_transform(
        command.robot_anchor_pos_w, command.robot_anchor_quat_w, command.robot_body_pos_w
    )
    body_quat_w = command.robot_body_quat_w.reshape(-1, 4)
    _, ori = subtract_frame_transforms(reference_pos, reference_quat, body_pos_w, body_quat_w)
    return matrix_from_quat(ori)[..., :2].reshape(env.num_envs, -1)


def _body_error(command: MotionCommand, body_names: list[str] | None = None) -> torch.Tensor:
    indexes = [i for i, name in enumerate(command.cfg.body_names) if body_names is None or name in body_names]
    return command.body_pos_relative_w[:, indexes] - command.robot_body_pos_w[:, indexes]


def motion_anchor_position_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return torch.exp(-torch.sum((command.anchor_pos_w - command.robot_anchor_pos_w) ** 2, dim=-1) / std**2)


def motion_anchor_orientation_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return torch.exp(-(quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2) / std**2)


def motion_body_position_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    return torch.exp(-mean_body_squared_error(_body_error(env.command_manager.get_term(command_name))) / std**2)


def motion_body_orientation_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.body_quat_relative_w, command.robot_body_quat_w)
    return torch.exp(-torch.square(error).mean(-1) / std**2)


def motion_body_linear_velocity_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    error = command.body_lin_vel_w - command.robot_body_lin_vel_w
    return torch.exp(-mean_body_squared_error(error) / std**2)


def motion_body_angular_velocity_error(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    error = command.body_ang_vel_w - command.robot_body_ang_vel_w
    return torch.exp(-mean_body_squared_error(error) / std**2)


def bad_anchor_pos_z(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, 2] - command.robot_anchor_pos_w[:, 2]) > threshold


def bad_anchor_ori(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) > threshold


def bad_body_pos_z(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    error = _body_error(env.command_manager.get_term(command_name), body_names)
    return torch.any(torch.abs(error[..., 2]) > threshold, dim=-1)
