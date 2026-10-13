"""Multi-clip motion data and command term for RGMT."""

from __future__ import annotations

import glob
from collections.abc import Sequence
from dataclasses import MISSING, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

from dr02_asset.profile import (
    AMP_MOTION_BODY_NAMES,
    JOINT_ORDER,
    TRACKING_ANCHOR_BODY,
    TRACKING_BODY_NAMES,
)

from .rgmt_mdp import sample_recovery_assistance, sample_recovery_mask
from dr02_training.tasks.tracking.mdp.tracking_motion import (
    align_motion_arrays,
    convert_body_quat_wxyz_to_xyzw,
    expand_reference_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


RGMT_COMMAND_WINDOW_LENGTH = 21


@dataclass(frozen=True)
class _LoadedClip:
    """One name-aligned motion clip held as device tensors."""

    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos_w: torch.Tensor
    body_quat_w: torch.Tensor
    body_lin_vel_w: torch.Tensor
    body_ang_vel_w: torch.Tensor
    command_features: torch.Tensor
    fps: float

    @property
    def frame_count(self) -> int:
        return self.joint_pos.shape[0]


class RGMTMotionDataset:
    """Load and validate one NPZ motion or a directory/glob of compatible clips."""

    REQUIRED_KEYS = (
        "fps",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_pos",
        "joint_vel",
    )

    def __init__(
        self,
        motion_path: str | Path | Sequence[str | Path],
        body_names: Sequence[str],
        joint_names: Sequence[str],
        device: str | torch.device,
        anchor_body_name: str,
        source_body_names: Sequence[str] = AMP_MOTION_BODY_NAMES,
        source_joint_names: Sequence[str] = JOINT_ORDER,
    ) -> None:
        paths = self._resolve_paths(motion_path)
        self.device = torch.device(device)
        self.body_names = list(body_names)
        self.joint_names = list(joint_names)
        self.source_body_names = list(source_body_names)
        self.source_joint_names = list(source_joint_names)
        if anchor_body_name not in self.body_names:
            raise ValueError(f"anchor body '{anchor_body_name}' is missing from body_names")
        anchor_index = self.body_names.index(anchor_body_name)
        clips = [self._load_clip(path, anchor_index) for path in paths]
        self.clip_count = len(clips)
        self.clip_lengths = torch.tensor(
            [clip.frame_count for clip in clips], dtype=torch.long, device=self.device
        )
        self.clip_fps = torch.tensor([clip.fps for clip in clips], dtype=torch.float32, device=self.device)
        self.clip_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long, device=self.device), self.clip_lengths.cumsum(0)[:-1])
        )
        self.total_frames = int(self.clip_lengths.sum().item())
        self.motion = SimpleNamespace(
            joint_pos=torch.cat([clip.joint_pos for clip in clips]),
            joint_vel=torch.cat([clip.joint_vel for clip in clips]),
            body_pos_w=torch.cat([clip.body_pos_w for clip in clips]),
            body_quat_w=torch.cat([clip.body_quat_w for clip in clips]),
            body_lin_vel_w=torch.cat([clip.body_lin_vel_w for clip in clips]),
            body_ang_vel_w=torch.cat([clip.body_ang_vel_w for clip in clips]),
            command_features=torch.cat([clip.command_features for clip in clips]),
            time_step_total=self.total_frames,
        )
        del clips
        print(f"[INFO] Loaded {self.clip_count} RGMT motion clips ({self.total_frames} frames total).")

    @staticmethod
    def _resolve_paths(motion_path: str | Path | Sequence[str | Path]) -> list[Path]:
        if isinstance(motion_path, (str, Path)):
            value = str(motion_path)
            path = Path(value)
            if path.is_dir():
                paths = sorted(path.glob("*.npz"))
            elif glob.has_magic(value):
                paths = [Path(item) for item in sorted(glob.glob(value))]
            else:
                paths = [path]
        else:
            paths = sorted(Path(item) for item in motion_path)
        if not paths:
            raise ValueError(f"no NPZ motion clips found at {motion_path}")
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"motion files do not exist: {missing}")
        if any(path.suffix.lower() != ".npz" for path in paths):
            raise ValueError("RGMT motion clips must use the .npz format")
        return paths

    def _load_clip(self, path: Path, anchor_index: int) -> _LoadedClip:
        with np.load(path, allow_pickle=True) as data:
            missing_keys = [key for key in self.REQUIRED_KEYS if key not in data]
            if missing_keys:
                raise ValueError(f"{path} is missing required arrays: {missing_keys}")
            fps_values = np.asarray(data["fps"]).reshape(-1)
            if fps_values.size != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0:
                raise ValueError(f"{path} must contain a positive fps value")

            body_arrays = {key: np.asarray(data[key], dtype=np.float32) for key in self.REQUIRED_KEYS[1:5]}
            joint_arrays = {key: np.asarray(data[key], dtype=np.float32) for key in self.REQUIRED_KEYS[5:]}
            body_metadata = self._read_names(data, "body_names")
            joint_metadata = self._read_names(data, "joint_names")

        if body_metadata is None:
            body_metadata = self.source_body_names
        if joint_metadata is None:
            joint_metadata = self.source_joint_names

        frame_counts = {array.shape[0] for array in (*body_arrays.values(), *joint_arrays.values()) if array.ndim > 0}
        if any(array.ndim == 0 for array in (*body_arrays.values(), *joint_arrays.values())) or len(frame_counts) != 1:
            raise ValueError(f"{path} motion arrays must have a consistent frame count")
        frame_count = frame_counts.pop()
        if frame_count <= 0:
            raise ValueError(f"{path} must contain at least one frame")
        if not all(np.isfinite(array).all() for array in (*body_arrays.values(), *joint_arrays.values())):
            raise ValueError(f"{path} contains non-finite motion values")

        body_pos = body_arrays["body_pos_w"]
        body_quat = body_arrays["body_quat_w"]
        body_lin_vel = body_arrays["body_lin_vel_w"]
        body_ang_vel = body_arrays["body_ang_vel_w"]
        joint_pos = joint_arrays["joint_pos"]
        joint_vel = joint_arrays["joint_vel"]
        if body_pos.ndim != 3 or body_pos.shape[-1] != 3:
            raise ValueError(f"{path} body_pos_w must have shape [frames, bodies, 3]")
        if body_quat.shape != (*body_pos.shape[:2], 4):
            raise ValueError(f"{path} body_quat_w must have shape [frames, bodies, 4]")
        if body_lin_vel.shape != body_pos.shape or body_ang_vel.shape != body_pos.shape:
            raise ValueError(f"{path} body velocity arrays must match body_pos_w shape")
        if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
            raise ValueError(f"{path} joint_pos and joint_vel must have the same [frames, joints] shape")
        if len(body_metadata) != body_pos.shape[1]:
            raise ValueError(f"{path} body_names length does not match the motion body dimension")
        if len(joint_metadata) != joint_pos.shape[1]:
            raise ValueError(f"{path} joint_names length does not match the motion joint dimension")
        if len(set(body_metadata)) != len(body_metadata) or len(set(joint_metadata)) != len(joint_metadata):
            raise ValueError(f"{path} body_names and joint_names must not contain duplicates")

        aligned = align_motion_arrays(
            body_pos=body_pos,
            body_names=body_metadata,
            target_body_names=self.body_names,
            body_quat=body_quat,
            body_lin_vel=body_lin_vel,
            body_ang_vel=body_ang_vel,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            joint_names=joint_metadata,
            target_joint_names=self.joint_names,
        )
        body_pos_t = torch.as_tensor(aligned.body_pos, dtype=torch.float32, device=self.device)
        body_quat_wxyz = torch.as_tensor(aligned.body_quat, dtype=torch.float32, device=self.device)
        body_quat_xyzw = torch.as_tensor(
            convert_body_quat_wxyz_to_xyzw(aligned.body_quat), dtype=torch.float32, device=self.device
        )
        body_lin_vel_t = torch.as_tensor(aligned.body_lin_vel, dtype=torch.float32, device=self.device)
        body_ang_vel_t = torch.as_tensor(aligned.body_ang_vel, dtype=torch.float32, device=self.device)
        joint_pos_t = torch.as_tensor(aligned.joint_pos, dtype=torch.float32, device=self.device)
        joint_vel_t = torch.as_tensor(aligned.joint_vel, dtype=torch.float32, device=self.device)
        anchor_quat = body_quat_wxyz[:, anchor_index]
        world_gravity = torch.zeros_like(body_lin_vel_t[:, anchor_index])
        world_gravity[:, 2] = -1.0
        command_features = torch.cat(
            (
                _quat_rotate_inverse_wxyz(body_lin_vel_t[:, anchor_index], anchor_quat),
                _quat_rotate_inverse_wxyz(body_ang_vel_t[:, anchor_index], anchor_quat),
                _quat_rotate_inverse_wxyz(world_gravity, anchor_quat),
                joint_pos_t,
            ),
            dim=-1,
        )
        return _LoadedClip(
            joint_pos=joint_pos_t,
            joint_vel=joint_vel_t,
            body_pos_w=body_pos_t,
            body_quat_w=body_quat_xyzw,
            body_lin_vel_w=body_lin_vel_t,
            body_ang_vel_w=body_ang_vel_t,
            command_features=command_features,
            fps=float(fps_values[0]),
        )

    @staticmethod
    def _read_names(data: np.lib.npyio.NpzFile, key: str) -> list[str] | None:
        if key not in data:
            return None
        return [str(name) for name in np.asarray(data[key]).reshape(-1)]

    def get_window(self, clip_ids: torch.Tensor, frame_ids: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """Return command features for offsets with each index clamped inside its clip."""
        if clip_ids.ndim != 1 or frame_ids.shape != clip_ids.shape:
            raise ValueError("clip_ids and frame_ids must be rank-1 tensors with identical shapes")
        if offsets.ndim == 1:
            offsets = offsets.unsqueeze(0).expand(clip_ids.shape[0], -1)
        if offsets.ndim != 2 or offsets.shape[0] != clip_ids.shape[0]:
            raise ValueError("offsets must have shape [window] or [batch, window]")
        if torch.any(clip_ids < 0) or torch.any(clip_ids >= self.clip_count):
            raise ValueError("clip_ids contain an out-of-range clip index")
        output = torch.empty(
            clip_ids.shape[0],
            offsets.shape[1],
            self.motion.command_features.shape[-1],
            dtype=self.motion.command_features.dtype,
            device=self.device,
        )
        for clip_id in range(self.clip_count):
            batch_mask = clip_ids == clip_id
            if not torch.any(batch_mask):
                continue
            local_frames = frame_ids[batch_mask, None] + offsets[batch_mask]
            local_frames = local_frames.clamp(0, self.clip_lengths[clip_id] - 1)
            global_frames = self.clip_offsets[clip_id] + local_frames
            output[batch_mask] = self.motion.command_features[global_frames]
        return output


def _quat_rotate_inverse_wxyz(vector: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    """Rotate vectors from world coordinates into a wxyz quaternion's local frame."""
    xyz = -quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(xyz, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.cross(xyz, twice_cross, dim=-1)


def advance_motion_positions(
    frame_positions: torch.Tensor,
    clip_ids: torch.Tensor,
    clip_lengths: torch.Tensor,
    clip_fps: torch.Tensor,
    step_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance clip positions by real elapsed time and report clips that ended."""
    if frame_positions.ndim != 1 or clip_ids.shape != frame_positions.shape:
        raise ValueError("frame_positions and clip_ids must be rank-1 tensors with identical shapes")
    if step_dt <= 0:
        raise ValueError(f"step_dt must be positive, got {step_dt}")
    next_positions = frame_positions + clip_fps[clip_ids] * step_dt
    finished = next_positions >= clip_lengths[clip_ids]
    return next_positions, finished


class RGMTMotionCommand(CommandTerm):
    """Sample reference poses and clip-bounded command windows for each environment."""

    cfg: RGMTMotionCommandCfg

    def __init__(self, cfg: RGMTMotionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if not 0.0 <= cfg.recovery_fraction <= 1.0:
            raise ValueError("recovery_fraction must be in [0, 1]")
        if cfg.recovery_assistance_max_newtons < 0.0:
            raise ValueError("recovery_assistance_max_newtons must be non-negative")
        if cfg.recovery_assistance_anneal_steps <= 0:
            raise ValueError("recovery_assistance_anneal_steps must be positive")
        if cfg.recovery_grace_period_s < 0.0:
            raise ValueError("recovery_grace_period_s must be non-negative")
        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(cfg.anchor_body_name)
        self.motion_anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)
        body_ids, _ = self.robot.find_bodies(cfg.body_names, preserve_order=True)
        self.body_indexes = torch.as_tensor(body_ids, dtype=torch.long, device=self.device)
        self.joint_indexes, _ = self.robot.find_joints(cfg.joint_names, preserve_order=True)
        self.joint_indexes = torch.as_tensor(self.joint_indexes, dtype=torch.long, device=self.device)
        self.dataset = RGMTMotionDataset(
            cfg.motion_path,
            cfg.body_names,
            cfg.joint_names,
            self.device,
            cfg.anchor_body_name,
            cfg.source_body_names,
            cfg.source_joint_names,
        )
        self.motion = self.dataset.motion
        if cfg.clip_sampling not in {"uniform", "proportional"}:
            raise ValueError("clip_sampling must be 'uniform' or 'proportional'")
        if cfg.clip_sampling == "uniform":
            self.clip_probabilities = torch.full(
                (self.dataset.clip_count,), 1 / self.dataset.clip_count, device=self.device
            )
        else:
            self.clip_probabilities = self.dataset.clip_lengths.float() / self.dataset.total_frames
        self.clip_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.frame_ids = torch.zeros_like(self.clip_ids)
        self.frame_positions = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.time_steps = torch.zeros_like(self.clip_ids)
        self.window_offsets = torch.arange(
            -(RGMT_COMMAND_WINDOW_LENGTH // 2), RGMT_COMMAND_WINDOW_LENGTH // 2 + 1, device=self.device
        )
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[..., 3] = 1.0
        self.recovery_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.recovery_assistance = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._episode_reset_in_progress = False
        metric_names = (
            "error_anchor_pos",
            "error_anchor_rot",
            "error_body_pos",
            "error_body_rot",
            "error_body_lin_vel",
            "error_body_ang_vel",
            "error_joint_pos",
            "error_joint_vel",
            "recovery_active_fraction",
            "recovery_assistance_newtons",
        )
        self.metrics.update({name: torch.zeros(self.num_envs, device=self.device) for name in metric_names})

    @property
    def command(self) -> torch.Tensor:
        """Current reference joint position and velocity."""
        return torch.cat((self.joint_pos, self.joint_vel), dim=-1)

    @property
    def joint_pos(self) -> torch.Tensor:
        """Current reference joint position."""
        return self.motion.joint_pos[self.time_steps]

    @property
    def joint_vel(self) -> torch.Tensor:
        """Current reference joint velocity."""
        return self.motion.joint_vel[self.time_steps]

    @property
    def body_pos_w(self) -> torch.Tensor:
        """Current reference body positions in world coordinates."""
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        """Current reference body orientations in simulator quaternion order."""
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        """Current reference body linear velocities in world coordinates."""
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        """Current reference body angular velocities in world coordinates."""
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        """Current reference anchor position."""
        return self.body_pos_w[:, self.motion_anchor_body_index]

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        """Current reference anchor orientation."""
        return self.body_quat_w[:, self.motion_anchor_body_index]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        """Robot body positions selected in the configured tracking order."""
        return _as_tensor(self.robot.data.body_pos_w)[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        """Robot body orientations selected in the configured tracking order."""
        return _as_tensor(self.robot.data.body_quat_w)[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        """Robot body linear velocities selected in the configured tracking order."""
        return _as_tensor(self.robot.data.body_lin_vel_w)[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        """Robot body angular velocities selected in the configured tracking order."""
        return _as_tensor(self.robot.data.body_ang_vel_w)[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        """Robot anchor position."""
        return _as_tensor(self.robot.data.body_pos_w)[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        """Robot anchor orientation."""
        return _as_tensor(self.robot.data.body_quat_w)[:, self.robot_anchor_body_index]

    def get_command_window(self) -> torch.Tensor:
        """Return the 21-frame reference command centered at the current motion frame."""
        return self.dataset.get_window(self.clip_ids, self.frame_ids, self.window_offsets)

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        """Reset commands and sample recovery starts only for the new episodes."""
        self._episode_reset_in_progress = True
        try:
            return super().reset(env_ids)
        finally:
            self._episode_reset_in_progress = False

    def _resample_command(self, env_ids: Sequence[int] | slice, *, preserve_clip: bool = False) -> None:
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if ids.numel() == 0:
            return
        sampled_clips = (
            self.clip_ids[ids]
            if preserve_clip
            else torch.multinomial(self.clip_probabilities, ids.numel(), replacement=True)
        )
        sampled_frames = torch.floor(
            torch.rand(ids.numel(), device=self.device) * self.dataset.clip_lengths[sampled_clips].to(torch.float32)
        ).long()
        if self._episode_reset_in_progress:
            self.recovery_mask[ids] = sample_recovery_mask(ids.numel(), self.cfg.recovery_fraction, self.device)
            self.recovery_assistance[ids] = sample_recovery_assistance(
                self.recovery_mask[ids],
                self.cfg.recovery_assistance_max_newtons,
                self._env.common_step_counter,
                self.cfg.recovery_assistance_anneal_steps,
            )
        self.clip_ids[ids] = sampled_clips
        self.frame_ids[ids] = sampled_frames
        self.frame_positions[ids] = sampled_frames.to(torch.float32)
        self.time_steps[ids] = self.dataset.clip_offsets[sampled_clips] + sampled_frames

        root_pos = self.anchor_pos_w[ids].clone()
        root_ori = self.anchor_quat_w[ids].clone()
        ranges = torch.tensor(
            [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
            device=self.device,
        )
        noise = sample_uniform(ranges[:, 0], ranges[:, 1], (ids.numel(), 6), device=self.device)
        recovery_slots = self.recovery_mask[ids]
        if torch.any(recovery_slots):
            recovery_ranges = torch.tensor(
                [self.cfg.recovery_pose_range.get(key, (0.0, 0.0)) for key in ("x", "y", "z", "roll", "pitch", "yaw")],
                device=self.device,
            )
            recovery_noise = sample_uniform(
                recovery_ranges[:, 0], recovery_ranges[:, 1], (ids.numel(), 6), device=self.device
            )
            noise = torch.where(recovery_slots[:, None], recovery_noise, noise)
        root_pos += noise[:, :3]
        root_ori = quat_mul(quat_from_euler_xyz(noise[:, 3], noise[:, 4], noise[:, 5]), root_ori)
        zero_velocity = torch.zeros_like(root_pos)
        self.robot.write_root_pose_to_sim_index(torch.cat((root_pos, root_ori), dim=-1), env_ids=ids)
        self.robot.write_root_link_velocity_to_sim_index(
            torch.cat((zero_velocity, zero_velocity), dim=-1), env_ids=ids
        )
        self.robot.write_joint_position_to_sim_index(
            position=self.joint_pos[ids], joint_ids=self.joint_indexes, env_ids=ids
        )
        self.robot.write_joint_velocity_to_sim_index(
            velocity=self.joint_vel[ids], joint_ids=self.joint_indexes, env_ids=ids
        )

    def _update_command(self) -> None:
        self.frame_positions, finished = advance_motion_positions(
            self.frame_positions,
            self.clip_ids,
            self.dataset.clip_lengths,
            self.dataset.clip_fps,
            self._env.step_dt,
        )
        self.frame_ids = torch.floor(self.frame_positions).long()
        finished_ids = torch.where(finished)[0]
        if finished_ids.numel() > 0:
            # A clip stays assigned to its environment for the episode. Restart
            # a new segment from that same clip when its current segment ends.
            self._resample_command(finished_ids, preserve_clip=True)
        self.time_steps = self.dataset.clip_offsets[self.clip_ids] + self.frame_ids

        delta_ori = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        delta_pos = self.robot_anchor_pos_w.clone()
        delta_pos[:, 2] = self.anchor_pos_w[:, 2]
        self.body_quat_relative_w = quat_mul(expand_reference_quat(delta_ori, self.body_quat_w), self.body_quat_w)
        self.body_pos_relative_w = delta_pos[:, None] + quat_apply(
            expand_reference_quat(delta_ori, self.body_quat_w), self.body_pos_w - self.anchor_pos_w[:, None]
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
            self.joint_pos - _as_tensor(self.robot.data.joint_pos)[:, self.joint_indexes], dim=-1
        )
        self.metrics["error_joint_vel"] = torch.linalg.vector_norm(
            self.joint_vel - _as_tensor(self.robot.data.joint_vel)[:, self.joint_indexes], dim=-1
        )
        self.metrics["recovery_active_fraction"] = self.recovery_mask.float()
        self.metrics["recovery_assistance_newtons"] = self.recovery_mask.float() * self.recovery_assistance


def _as_tensor(value: torch.Tensor) -> torch.Tensor:
    """Unwrap an Isaac Lab proxy array where required."""
    return value.torch if hasattr(value, "torch") else value


@configclass
class RGMTMotionCommandCfg(CommandTermCfg):
    """Configuration for a multi-clip RGMT reference command."""

    class_type: type = RGMTMotionCommand
    asset_name: str = "robot"
    motion_path: str | list[str] = MISSING
    joint_names: list[str] = list(JOINT_ORDER)
    source_body_names: list[str] = list(AMP_MOTION_BODY_NAMES)
    source_joint_names: list[str] = list(JOINT_ORDER)
    anchor_body_name: str = TRACKING_ANCHOR_BODY
    body_names: list[str] = list(TRACKING_BODY_NAMES)
    pose_range: dict[str, tuple[float, float]] = {}
    clip_sampling: str = "uniform"
    recovery_fraction: float = 0.15
    recovery_pose_range: dict[str, tuple[float, float]] = {
        "z": (-0.15, 0.25),
        "roll": (-1.2, 1.2),
        "pitch": (-1.2, 1.2),
        "yaw": (-3.14, 3.14),
    }
    recovery_assistance_max_newtons: float = 200.0
    recovery_assistance_anneal_steps: int = 100_000_000
    recovery_grace_period_s: float = 3.0
