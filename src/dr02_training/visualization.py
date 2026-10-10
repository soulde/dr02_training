# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Project-owned Viser setup for training and playback."""

import os

from isaaclab_visualizers.viser import ViserVisualizer, ViserVisualizerCfg

VISUALIZER_MODE_ENV = "DR02_TRAINING_VISUALIZER_MODE"


class PausedRenderingViserVisualizer(ViserVisualizer):
    """Start the viewer without rendering while leaving simulation active."""

    def __init__(self, cfg: ViserVisualizerCfg):
        super().__init__(cfg)
        self._paused_rendering = True


def make_viser_cfg(*, paused: bool, max_visible_envs: int = 512) -> ViserVisualizerCfg:
    """Build the Viser config used by the package train/play commands."""
    return ViserVisualizerCfg(
        class_type=PausedRenderingViserVisualizer if paused else ViserVisualizer,
        max_visible_envs=max_visible_envs,
        open_browser=False,
    )


def configure_sim_visualizer(sim_cfg) -> None:
    """Install the mode-specific Viser config when called by package commands."""
    mode = os.environ.get(VISUALIZER_MODE_ENV)
    if mode is None:
        return
    if mode not in {"train", "play"}:
        raise ValueError(f"Invalid {VISUALIZER_MODE_ENV}={mode!r}; expected 'train' or 'play'.")
    sim_cfg.visualizer_cfgs = [make_viser_cfg(paused=mode == "train")]
