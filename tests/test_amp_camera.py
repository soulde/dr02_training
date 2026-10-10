# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Tests for the AMP RGB-D camera task variants."""

import gymnasium as gym
import xml.etree.ElementTree as ET
from pathlib import Path

from dr02_training.tasks.amp.config.dr02.camera_env_cfg import (
    DR02AMPCameraFlatEnvCfg,
    DR02AMPCameraRecoveryEnvCfg,
    DR02AMPCameraRoughEnvCfg,
)
from dr02_training.tasks.amp.config.dr02.env_cfg import DR02AMPFlatEnvCfg
from dr02_training.tasks.amp.config.dr02.recovery_env_cfg import DR02AMPRecoveryEnvCfg
from dr02_training.tasks.amp.config.dr02.rough_env_cfg import DR02AMPRoughEnvCfg


CAMERA_LINKS = ("d435_head_link", "d435_front_link", "d435_rear_link")
CAMERA_JOINTS = {
    "d435_head_link": ((0.11871, 0.0175, 0.5096), (0.0, 0.61086, 0.0)),
    "d435_front_link": ((0.12559, 0.0175, 0.078407), (0.0, 1.0472, 0.0)),
    "d435_rear_link": ((-0.11241, -0.0175, -0.037264), (0.0, 1.0472, 3.1416)),
}
CAMERA_CONFIGS = (
    ("Dr02Training-AMP-Flat-Camera-v0", DR02AMPCameraFlatEnvCfg, DR02AMPFlatEnvCfg),
    ("Dr02Training-AMP-Rough-Camera-v0", DR02AMPCameraRoughEnvCfg, DR02AMPRoughEnvCfg),
    ("Dr02Training-AMP-Recovery-Camera-v0", DR02AMPCameraRecoveryEnvCfg, DR02AMPRecoveryEnvCfg),
)


def _term_names(group) -> set[str]:
    return {name for name, value in vars(group).items() if value is not None and not name.startswith("_")}


def test_amp_camera_tasks_are_registered():
    for task_id, cfg_type, _ in CAMERA_CONFIGS:
        spec = gym.spec(task_id)
        assert spec.kwargs["env_cfg_entry_point"].endswith(f":{cfg_type.__name__}")
        assert spec.kwargs["default_agent"] == "rsl_rl"


def test_amp_camera_cfg_exposes_three_d435_streams():
    urdf_path = Path(__file__).parents[1] / "src/dr02_training/assets/dr02/description/urdf/dr02_pro.urdf"
    urdf = ET.parse(urdf_path).getroot()
    joint_origins = {
        joint.find("child").attrib["link"]: joint.find("origin").attrib
        for joint in urdf.findall("joint")
        if joint.find("child") is not None and joint.find("child").attrib.get("link") in CAMERA_JOINTS
    }
    for link, (expected_xyz, expected_rpy) in CAMERA_JOINTS.items():
        assert tuple(map(float, joint_origins[link]["xyz"].split())) == expected_xyz
        assert tuple(map(float, joint_origins[link]["rpy"].split())) == expected_rpy

    for _, cfg_type, _ in CAMERA_CONFIGS:
        scene = cfg_type().scene
        for name in ("d435_head", "d435_front", "d435_rear"):
            camera = getattr(scene, name)
            link = name + "_link"
            assert camera.prim_path == f"{{ENV_REGEX_NS}}/Robot/{link}"
            # Camera link poses come from the fixed joints in the URDF. A zero
            # sensor offset avoids applying the source body-relative pose twice.
            assert camera.offset.pos == (0.0, 0.0, 0.0)
            assert camera.offset.rot == (0.0, 0.0, 0.0, 1.0)
            assert camera.data_types == ["rgb", "distance_to_image_plane"]
            assert (camera.width, camera.height) == (640, 480)
            assert camera.update_period == 1.0 / 30.0
            assert camera.spawn.clipping_range == (0.1, 10.0)


def test_amp_camera_cfg_preserves_policy_observations():
    for _, camera_cfg_type, base_cfg_type in CAMERA_CONFIGS:
        camera_cfg, base_cfg = camera_cfg_type(), base_cfg_type()
        assert _term_names(camera_cfg.observations.policy) == _term_names(base_cfg.observations.policy)
        assert _term_names(camera_cfg.observations.critic) == _term_names(base_cfg.observations.critic)
        assert _term_names(camera_cfg.observations.amp) == _term_names(base_cfg.observations.amp)
