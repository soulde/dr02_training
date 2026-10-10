# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for project-owned Viser training/play configuration."""

from types import SimpleNamespace

from isaaclab_visualizers.viser import ViserVisualizer

from dr02_training.visualization import configure_sim_visualizer, make_viser_cfg


def test_train_viser_starts_rendering_paused_without_pausing_simulation():
    cfg = make_viser_cfg(paused=True)
    visualizer = cfg.class_type(cfg)
    assert isinstance(visualizer, ViserVisualizer)
    assert visualizer._paused_rendering is True
    assert visualizer._paused_simulation is False


def test_play_viser_starts_rendering_running():
    cfg = make_viser_cfg(paused=False)
    visualizer = cfg.class_type(cfg)
    assert cfg.class_type is ViserVisualizer
    assert visualizer._paused_rendering is False
    assert visualizer._paused_simulation is False


def test_viser_config_caps_visible_envs():
    assert make_viser_cfg(paused=True, max_visible_envs=512).max_visible_envs == 512


def test_sim_visualizer_uses_requested_mode(monkeypatch):
    from dr02_training.cli import VISUALIZER_MODE_ENV

    sim_cfg = SimpleNamespace(visualizer_cfgs=[])
    monkeypatch.setenv(VISUALIZER_MODE_ENV, "train")
    configure_sim_visualizer(sim_cfg)
    assert len(sim_cfg.visualizer_cfgs) == 1
    assert sim_cfg.visualizer_cfgs[0].max_visible_envs == 512
    assert sim_cfg.visualizer_cfgs[0].class_type is not ViserVisualizer

    monkeypatch.setenv(VISUALIZER_MODE_ENV, "play")
    configure_sim_visualizer(sim_cfg)
    assert sim_cfg.visualizer_cfgs[0].class_type is ViserVisualizer
