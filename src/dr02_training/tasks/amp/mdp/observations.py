"""AMP discriminator features for the DR02 source motion contract."""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply


def amp_root_linear_velocity(env) -> torch.Tensor:
    return env.scene["robot"].data.root_link_lin_vel_w.torch


def critic_terrain_heights(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return scanner terrain heights relative to the robot root for the critic."""
    sensor = env.scene.sensors[sensor_cfg.name]
    heights = sensor.data.ray_hits_w.torch[..., 2] - env.scene["robot"].data.root_pos_w.torch[:, 2:3]
    return heights.reshape(env.num_envs, -1)


def amp_root_angular_velocity(env) -> torch.Tensor:
    return env.scene["robot"].data.root_link_ang_vel_w.torch


def amp_joint_position(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_pos.torch[:, asset_cfg.joint_ids]


def amp_joint_velocity(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return env.scene[asset_cfg.name].data.joint_vel.torch[:, asset_cfg.joint_ids]


def amp_link_positions(
    env,
    asset_cfg: SceneEntityCfg,
    key_body_names: list[str],
    virtual_body_offsets: dict[str, tuple[str, tuple[float, float, float]]] | None = None,
) -> torch.Tensor:
    """Return root-relative key points, including toe markers absent from the URDF."""
    robot = env.scene[asset_cfg.name]
    body_names = list(robot.body_names)
    body_pos_w = robot.data.body_pos_w.torch
    body_quat_w = robot.data.body_quat_w.torch
    virtual_body_offsets = virtual_body_offsets or {}
    points = []
    for name in key_body_names:
        if name in body_names:
            points.append(body_pos_w[:, body_names.index(name)])
            continue
        if name not in virtual_body_offsets:
            raise ValueError(f"AMP key point {name!r} has no robot body or virtual offset")
        parent_name, local_offset = virtual_body_offsets[name]
        if parent_name not in body_names:
            raise ValueError(f"AMP virtual point {name!r} parent {parent_name!r} is absent")
        parent_index = body_names.index(parent_name)
        offset = torch.as_tensor(local_offset, device=body_pos_w.device, dtype=body_pos_w.dtype)
        offset_w = quat_apply(body_quat_w[:, parent_index], offset.expand(env.num_envs, -1))
        points.append(body_pos_w[:, parent_index] + offset_w)
    return (torch.stack(points, dim=1) - robot.data.root_pos_w.torch[:, None, :]).flatten(start_dim=1)
