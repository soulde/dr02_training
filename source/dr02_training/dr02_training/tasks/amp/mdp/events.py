# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def reset_amp_reference_state(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    terrain_clearance: float = 0.005,
    xy_range: tuple[float, float] | None = None,
    terrain_height_sensor: str | None = None,
):
    """Reset from the RSL-RL AMP sampler's XYZW reference-state contract."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    if len(env_ids) == 0:
        return

    env = env.unwrapped
    sampler = getattr(env, "_amp_rsi_sampler", None)
    if sampler is None:
        if not getattr(env, "_amp_rsi_initial_reset_deferred", False):
            env._amp_rsi_initial_reset_deferred = True
            return
        raise RuntimeError("AMP RSI reset ran before the runner registered agent.sample_rsi")
    env._amp_rsi_initial_reset_deferred = True
    layout = getattr(env, "_amp_rsi_joint_layout", None)
    if layout is None:
        raise RuntimeError("AMP RSI reset requires a registered joint layout")

    env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=env.device)
    reference = sampler(env_ids)
    robot: Articulation = env.scene[asset_cfg.name]
    env_origins = env.scene.env_origins[env_ids]

    # Mocap XY is not a spawn coordinate; place the sampled pose at each
    # environment origin and retain the reference height above the terrain.
    root_position = reference["root_pos"].to(device=robot.device, dtype=torch.float32).clone()
    root_position[:, 0:2] = env_origins[:, 0:2]
    if xy_range is not None:
        root_position[:, 0:2] += torch.empty_like(root_position[:, 0:2]).uniform_(*xy_range)
    if terrain_height_sensor is None:
        root_position[:, 2] += env_origins[:, 2]
    else:
        root_position[:, 2] += _amp_reset_ground_height(env, root_position[:, 0:2], env_ids, terrain_height_sensor)
    # RSL-RL's RSI contract already returns Isaac Lab's XYZW root quaternion.
    root_quat_xyzw = reference["root_quat"].to(device=robot.device, dtype=torch.float32)
    root_velocity = torch.cat(
        (
            reference["root_lin_vel"].to(device=robot.device, dtype=torch.float32),
            reference["root_ang_vel"].to(device=robot.device, dtype=torch.float32),
        ),
        dim=-1,
    )
    if "root_height_offset" in reference:
        root_position[:, 2] += reference["root_height_offset"].to(device=robot.device, dtype=torch.float32)
        root_velocity[:, 2] += reference["root_height_velocity_offset"].to(device=robot.device, dtype=torch.float32)
    else:
        root_position[:, 2] += terrain_clearance
    root_pose = torch.cat((root_position, root_quat_xyzw), dim=-1)
    robot.write_root_pose_to_sim_index(root_pose=root_pose, env_ids=env_ids)
    robot.write_root_link_velocity_to_sim_index(root_velocity=root_velocity, env_ids=env_ids)

    # The accompanying layout owns the dataset's joint order; map by name to
    # the articulation order before writing the sampled state.
    reference_joint_names = list(layout["joint_names"])
    if reference_joint_names != list(robot.joint_names):
        if set(reference_joint_names) != set(robot.joint_names):
            raise ValueError("RSL-RL RSI joint layout does not match the robot articulation")
        reorder = [reference_joint_names.index(name) for name in robot.joint_names]
        joint_pos = reference["joint_pos"].to(device=robot.device, dtype=torch.float32)[:, reorder]
        joint_vel = reference["joint_vel"].to(device=robot.device, dtype=torch.float32)[:, reorder]
    else:
        joint_pos = reference["joint_pos"].to(device=robot.device, dtype=torch.float32)
        joint_vel = reference["joint_vel"].to(device=robot.device, dtype=torch.float32)
    robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
    robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)


def _amp_reset_ground_height(
    env: ManagerBasedEnv, xy: torch.Tensor, env_ids: torch.Tensor, sensor_name: str
) -> torch.Tensor:
    """Query new spawn XY against the scanner mesh, not its stale ray hits."""
    from isaaclab.utils.warp import raycast_mesh

    sensor = env.scene.sensors[sensor_name]
    mesh = sensor.meshes[(sensor.cfg.mesh_prim_paths[0], str(sensor.device))]
    ray_starts = torch.zeros((len(env_ids), 3), device=xy.device, dtype=xy.dtype)
    ray_starts[:, :2] = xy
    # Start well above the standard 8 m rough terrain cells, including pits.
    ray_starts[:, 2] = env.scene.env_origins[env_ids, 2] + 20.0
    ray_directions = torch.zeros_like(ray_starts)
    ray_directions[:, 2] = -1.0
    ground_z = raycast_mesh(ray_starts, ray_directions, mesh)[0][:, 2]
    if not torch.isfinite(ground_z).all():
        raise RuntimeError("AMP RSI terrain height query missed the ground at a sampled spawn position")
    return ground_z
