# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable package-owned RSL-RL AMP train and play commands."""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Callable, Sequence

from isaaclab_rl import entrypoints
from isaaclab_rl.entrypoints import PlaybackRequest, TrainingRequest

from .visualization import VISUALIZER_MODE_ENV


def _parse_request_args(
    argv: Sequence[str], *, training: bool
) -> tuple[argparse.Namespace, tuple[str, ...], tuple[str, ...]]:
    parser = argparse.ArgumentParser(description="Train or play a DR02 RSL-RL policy.")
    parser.add_argument("--task", required=True, help="Registered Gym task ID.")
    parser.add_argument("--num-envs", type=int, default=4096 if training else 512)
    parser.add_argument("--checkpoint", default=None if training else "latest")
    parser.add_argument("--agent", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--video", action="store_true")
    if training:
        parser.add_argument("--max-iterations", type=int, default=10000)
        parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--visualizer", choices=("viser",), default="viser")
    args, passthrough = parser.parse_known_args(list(argv))

    backend_args = ["--visualizer", args.visualizer]
    hydra_args = []
    index = 0
    while index < len(passthrough):
        token = passthrough[index]
        if token.startswith("-"):
            backend_args.append(token)
            if "=" not in token and index + 1 < len(passthrough) and not passthrough[index + 1].startswith("-"):
                index += 1
                backend_args.append(passthrough[index])
        else:
            hydra_args.append(token)
        index += 1
    return args, tuple(backend_args), tuple(hydra_args)


def build_train_request(argv: Sequence[str]) -> TrainingRequest:
    """Parse package options into the public Isaac Lab training request."""
    args, backend_args, hydra_args = _parse_request_args(argv, training=True)
    return TrainingRequest(
        backend="rsl_rl",
        task=args.task,
        agent=args.agent,
        num_envs=args.num_envs,
        seed=args.seed,
        max_iterations=args.max_iterations,
        device=args.device,
        video=args.video,
        distributed=args.distributed,
        backend_args=backend_args,
        hydra_args=hydra_args,
    )


def build_play_request(argv: Sequence[str]) -> PlaybackRequest:
    """Parse package options into the public Isaac Lab playback request."""
    args, backend_args, hydra_args = _parse_request_args(argv, training=False)
    return PlaybackRequest(
        backend="rsl_rl",
        task=args.task,
        checkpoint=args.checkpoint,
        agent=args.agent,
        num_envs=args.num_envs,
        seed=args.seed,
        device=args.device,
        video=args.video,
        backend_args=backend_args,
        hydra_args=hydra_args,
    )


def _dispatch(mode: str, dispatch_fn: Callable, request) -> int:
    previous = os.environ.get(VISUALIZER_MODE_ENV)
    os.environ[VISUALIZER_MODE_ENV] = mode
    try:
        return dispatch_fn(request)
    except (FileNotFoundError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    finally:
        if previous is None:
            os.environ.pop(VISUALIZER_MODE_ENV, None)
        else:
            os.environ[VISUALIZER_MODE_ENV] = previous


def train_main(argv: Sequence[str] | None = None) -> int:
    """Run RSL-RL training from package-owned command line options."""
    request = build_train_request(sys.argv[1:] if argv is None else argv)
    return _dispatch("train", entrypoints.train, request)


def play_main(argv: Sequence[str] | None = None) -> int:
    """Run task-scoped latest or explicitly selected RSL-RL checkpoint playback."""
    request = build_play_request(sys.argv[1:] if argv is None else argv)
    return _dispatch("play", entrypoints.play, request)
