# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the package-owned train and play command arguments."""

import pytest

from dr02_training import cli


def test_train_request_defaults_and_hyphenated_overrides():
    default_request = cli.build_train_request(["--task", "Dr02Training-AMP-Recovery-v0"])
    assert default_request.num_envs == 4096
    assert default_request.max_iterations == 10000
    assert default_request.backend_args == ("--visualizer", "viser")

    request = cli.build_train_request(
        [
            "--task",
            "Dr02Training-AMP-Recovery-v0",
            "--num-envs",
            "12",
            "--max-iterations",
            "3",
            "--seed",
            "5",
            "--rsl-rl-arg",
            "value",
            "commands.base_velocity.ranges.lin_vel_x=[0.0,1.0]",
        ]
    )
    assert request.backend == "rsl_rl"
    assert request.task == "Dr02Training-AMP-Recovery-v0"
    assert request.num_envs == 12
    assert request.max_iterations == 3
    assert request.seed == 5
    assert request.backend_args == ("--visualizer", "viser", "--rsl-rl-arg", "value")
    assert request.hydra_args == ("commands.base_velocity.ranges.lin_vel_x=[0.0,1.0]",)


def test_play_defaults_to_latest_and_forwards_explicit_checkpoint():
    request = cli.build_play_request(["--task", "Dr02Training-AMP-Flat-Camera-v0", "--checkpoint", "/tmp/policy.pt"])
    assert request.backend == "rsl_rl"
    assert request.task == "Dr02Training-AMP-Flat-Camera-v0"
    assert request.checkpoint == "/tmp/policy.pt"
    assert request.num_envs == 512
    assert request.backend_args == ("--visualizer", "viser")

    default_request = cli.build_play_request(["--task", "Dr02Training-AMP-Flat-v0"])
    assert default_request.checkpoint == "latest"


def test_missing_task_checkpoint_returns_nonzero(monkeypatch, capsys):
    def fail_play(_request):
        raise ValueError("No compatible checkpoint found for this task")

    monkeypatch.setattr(cli.entrypoints, "play", fail_play)
    assert cli.play_main(["--task", "Dr02Training-AMP-Flat-v0"]) == 1
    assert "No compatible checkpoint found for this task" in capsys.readouterr().err


@pytest.mark.parametrize("main,mode", [(cli.train_main, "train"), (cli.play_main, "play")])
def test_entrypoint_sets_and_restores_visualizer_mode(monkeypatch, main, mode):
    monkeypatch.setenv(cli.VISUALIZER_MODE_ENV, "existing")
    observed = []

    def dispatch(_request):
        observed.append(cli.os.environ[cli.VISUALIZER_MODE_ENV])
        return 0

    monkeypatch.setattr(cli.entrypoints, "train" if mode == "train" else "play", dispatch)
    main(["--task", "Dr02Training-AMP-Flat-v0"])
    assert observed == [mode]
    assert cli.os.environ[cli.VISUALIZER_MODE_ENV] == "existing"
