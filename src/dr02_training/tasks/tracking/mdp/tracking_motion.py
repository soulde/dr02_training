"""Name-based alignment for BeyondMimic-compatible motion NPZ files."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class AlignedMotionArrays:
    body_pos: np.ndarray
    body_quat: np.ndarray | None = None
    body_lin_vel: np.ndarray | None = None
    body_ang_vel: np.ndarray | None = None
    joint_pos: np.ndarray | None = None
    joint_vel: np.ndarray | None = None


def expand_reference_transform(reference_pos, reference_quat, target_pos):
    """Repeat one reference frame for every target body and flatten the batch."""
    num_bodies = target_pos.shape[1]
    expanded_pos = reference_pos.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 3)
    expanded_quat = reference_quat.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    flattened_target_pos = target_pos.reshape(-1, 3)
    return expanded_pos, expanded_quat, flattened_target_pos


def expand_reference_quat(reference_quat, target_quat):
    """Repeat one reference quaternion for every target body."""
    return reference_quat.unsqueeze(1).expand_as(target_quat)


def mean_body_squared_error(error: torch.Tensor) -> torch.Tensor:
    """Sum squared XYZ error per body, then average over tracked bodies."""
    return torch.square(error).sum(dim=-1).mean(dim=-1)


def convert_body_quat_wxyz_to_xyzw(body_quat: np.ndarray) -> np.ndarray:
    """Convert legacy Isaac Lab 2.x/exported quaternions to Isaac Lab 3.x order."""
    if body_quat.shape[-1] != 4:
        raise ValueError(f"expected quaternion shape (..., 4), got {body_quat.shape}")
    return body_quat[..., [1, 2, 3, 0]]


def _indices(
    source_names: Sequence[str] | None,
    target_names: Sequence[str],
    label: str,
    source_count: int,
) -> list[int]:
    if source_names is None:
        if source_count < len(target_names):
            raise ValueError(f"motion has fewer {label} entries than requested")
        return list(range(len(target_names)))

    source_names = [str(name) for name in source_names]
    missing = [name for name in target_names if name not in source_names]
    if missing:
        raise ValueError(f"missing {label} names: {missing}")
    return [source_names.index(name) for name in target_names]


def align_motion_arrays(
    *,
    body_pos: np.ndarray,
    body_names: Sequence[str] | None,
    target_body_names: Sequence[str],
    joint_pos: np.ndarray | None,
    joint_vel: np.ndarray | None,
    joint_names: Sequence[str] | None,
    target_joint_names: Sequence[str] | None,
    body_quat: np.ndarray | None = None,
    body_lin_vel: np.ndarray | None = None,
    body_ang_vel: np.ndarray | None = None,
) -> AlignedMotionArrays:
    """Select and reorder motion arrays into the target robot's name order."""
    body_indices = _indices(body_names, target_body_names, "body", body_pos.shape[1])
    joint_indices = None
    if joint_pos is not None and target_joint_names is not None:
        joint_indices = _indices(joint_names, target_joint_names, "joint", joint_pos.shape[1])

    def select(array: np.ndarray | None, indices: list[int] | None) -> np.ndarray | None:
        return None if array is None else array[:, indices]

    return AlignedMotionArrays(
        body_pos=body_pos[:, body_indices],
        body_quat=select(body_quat, body_indices),
        body_lin_vel=select(body_lin_vel, body_indices),
        body_ang_vel=select(body_ang_vel, body_indices),
        joint_pos=select(joint_pos, joint_indices),
        joint_vel=select(joint_vel, joint_indices),
    )
