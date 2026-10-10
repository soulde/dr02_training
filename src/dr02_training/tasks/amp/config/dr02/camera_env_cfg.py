# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Camera-equipped DR02 AMP environments with three D435 RGB-D streams."""

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

from .env_cfg import DR02AMPFlatEnvCfg
from .recovery_env_cfg import DR02AMPRecoveryEnvCfg
from .rough_env_cfg import DR02AMPRoughEnvCfg

_D435_INTRINSICS_640X480 = [382.6, 0.0, 319.5, 0.0, 382.6, 239.5, 0.0, 0.0, 1.0]
_D435_LINKS = {
    "d435_head": "d435_head_link",
    "d435_front": "d435_front_link",
    "d435_rear": "d435_rear_link",
}


def make_d435_camera_cfg(camera_name: str) -> CameraCfg:
    """Attach a D435 stream to its retained fixed camera link in the URDF."""
    if camera_name not in _D435_LINKS:
        raise ValueError(f"Unknown DR02 D435 camera: {camera_name}")
    link_name = _D435_LINKS[camera_name]
    spawn = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
        intrinsic_matrix=_D435_INTRINSICS_640X480,
        width=640,
        height=480,
        clipping_range=(0.1, 10.0),
    )
    return CameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{link_name}",
        update_period=1.0 / 30.0,
        # The fixed URDF joint already places and orients each camera link.
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0), convention="world"),
        data_types=["rgb", "distance_to_image_plane"],
        spawn=spawn,
        width=640,
        height=480,
    )


def _add_d435_cameras(scene) -> None:
    for camera_name in _D435_LINKS:
        setattr(scene, camera_name, make_d435_camera_cfg(camera_name))


@configclass
class DR02AMPCameraFlatEnvCfg(DR02AMPFlatEnvCfg):
    """Flat AMP environment with head, front, and rear D435 cameras."""

    def __post_init__(self):
        super().__post_init__()
        _add_d435_cameras(self.scene)


@configclass
class DR02AMPCameraRoughEnvCfg(DR02AMPRoughEnvCfg):
    """Rough-terrain AMP environment with head, front, and rear D435 cameras."""

    def __post_init__(self):
        super().__post_init__()
        _add_d435_cameras(self.scene)


@configclass
class DR02AMPCameraRecoveryEnvCfg(DR02AMPRecoveryEnvCfg):
    """Recovery AMP environment with head, front, and rear D435 cameras."""

    def __post_init__(self):
        super().__post_init__()
        _add_d435_cameras(self.scene)
