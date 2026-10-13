"""Compatibility for legacy DR02 AMP clips that omit virtual toe markers."""

from __future__ import annotations

from collections.abc import Sequence


_VIRTUAL_BODY_OFFSETS = {
    "left_toe_link": ("left_ankle_x_link", (0.11, 0.0, -0.015)),
    "right_toe_link": ("right_ankle_x_link", (0.11, 0.0, -0.015)),
}


def _rotate_xyzw(quaternion, vector):
    """Rotate a local vector by an XYZW quaternion using torch operations."""
    import torch

    q_xyz = quaternion[..., :3]
    q_w = quaternion[..., 3:4]
    vector = vector.expand_as(q_xyz)
    twice_cross = 2.0 * torch.linalg.cross(q_xyz, vector, dim=-1)
    return vector + q_w * twice_cross + torch.linalg.cross(q_xyz, twice_cross, dim=-1)


def add_virtual_toes(motion: dict, body_names: Sequence[str]) -> dict:
    """Append legacy clip toe markers using the same offsets as AMP observations.

    DR02's legacy locomotion clips store 30 physical bodies, while the configured
    AMP body contract also includes two virtual toes. The clips have no embedded
    body-name metadata, so this is deliberately limited to the exact 30-to-32
    DR02 contract and fails clearly for any other partial contract.
    """
    expected_virtual_names = tuple(_VIRTUAL_BODY_OFFSETS)
    body_names = tuple(body_names)
    body_pos = motion["body_pos_w"]
    stored_bodies = body_pos.shape[1]
    if stored_bodies == len(body_names):
        return motion
    if (
        len(body_names) != 32
        or stored_bodies != 30
        or body_names[-2:] != expected_virtual_names
        or any(name not in body_names for name in _VIRTUAL_BODY_OFFSETS)
    ):
        raise ValueError(
            "DR02 AMP motion body contract mismatch: expected 30 stored physical bodies "
            f"or 32 bodies including virtual toes, got {stored_bodies} stored and {len(body_names)} configured"
        )

    import torch

    for field in ("body_quat_w", "body_quat_xyzw", "body_lin_vel_w", "body_ang_vel_w"):
        if motion[field].shape[1] != stored_bodies:
            raise ValueError(f"DR02 AMP motion field {field!r} has an inconsistent body dimension")

    for toe_name, (parent_name, local_offset) in _VIRTUAL_BODY_OFFSETS.items():
        parent_index = body_names.index(parent_name)
        quaternion_xyzw = motion["body_quat_xyzw"][:, parent_index]
        offset = torch.tensor(local_offset, dtype=body_pos.dtype, device=body_pos.device)
        toe_pos = body_pos[:, parent_index] + _rotate_xyzw(quaternion_xyzw, offset)
        motion["body_pos_w"] = torch.cat((motion["body_pos_w"], toe_pos[:, None, :]), dim=1)
        # The AMP reference features use toe positions; the other toe fields are
        # not consumed but are kept dimensionally consistent for the body contract.
        for field in ("body_quat_w", "body_quat_xyzw", "body_lin_vel_w", "body_ang_vel_w"):
            parent_values = motion[field][:, parent_index : parent_index + 1]
            motion[field] = torch.cat((motion[field], parent_values), dim=1)
    return motion


def install_legacy_motion_compatibility(task: str) -> None:
    """Patch RSL-RL's BeyondMimic loader for DR02 AMP only."""
    if task != "Dr02Training-AMP-Flat-v0":
        return

    from dr02_asset.profile import AMP_MOTION_BODY_NAMES
    from rsl_rl.datasets.motion_dataset import MotionDataset

    if getattr(MotionDataset, "_dr02_toe_compat_installed", False):
        return

    original_load = MotionDataset._load_motion_file

    def load_with_virtual_toes(self, motion_file, *, body_names, joint_names):
        motion = original_load(self, motion_file, body_names=body_names, joint_names=joint_names)
        if motion is not None and body_names is not None and tuple(body_names) == tuple(AMP_MOTION_BODY_NAMES):
            motion = add_virtual_toes(motion, body_names)
        return motion

    MotionDataset._load_motion_file = load_with_virtual_toes
    MotionDataset._dr02_toe_compat_installed = True
