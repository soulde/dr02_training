# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Stable package-owned RSL-RL AMP train and play commands."""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from collections.abc import Callable, Sequence

from isaaclab_rl import entrypoints
from isaaclab_rl.entrypoints import PlaybackRequest, TrainingRequest

_RUNNER_OPTION_ALIASES = {"--run-name": "--run_name"}


def _physics_preset(request) -> str | None:
    """Return the selected physics preset, if one is explicit."""
    for override in request.hydra_args:
        key, separator, value = override.lstrip("+").partition("=")
        if separator and key == "physics":
            return value
    return None


def _requires_early_kit(request) -> bool:
    """Whether task config resolution must happen after Kit has started."""
    return _physics_preset(request) == "isaacsim_physx"


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
            option, separator, value = token.partition("=")
            option = _RUNNER_OPTION_ALIASES.get(option, option)
            backend_args.append(f"{option}{separator}{value}")
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
        checkpoint=args.checkpoint,
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
    # The package-owned CLI bypasses Isaac Lab's ``isaaclab cli train`` loader,
    # so load this distribution's task package before resolving the Gym task ID.
    from . import tasks as _tasks  # noqa: F401

    app_launcher = None
    if _requires_early_kit(request):
        # Resolving this task's config imports USD Python bindings. Kit must be
        # initialized first so its USD/PhysX schema extensions can register wrappers.
        # The URDF importer also needs MJC schemas registered before Kit creates
        # its first USD stage; use only the converter's plugin path, not the
        # standalone importer's PhysX fallback schema path.
        mjc_plugin_path = os.environ.get("DR02_MJC_PXR_PLUGIN_PATH")
        if not mjc_plugin_path:
            spec = importlib.util.find_spec("mujoco_usd_converter")
            if spec is not None and spec.submodule_search_locations:
                mjc_plugin_path = os.path.join(next(iter(spec.submodule_search_locations)), "plugins")
        if not mjc_plugin_path or not os.path.isdir(mjc_plugin_path):
            raise RuntimeError("The mujoco-usd-converter MJC schema plugin directory is unavailable.")
        os.environ["PXR_PLUGINPATH_NAME"] = mjc_plugin_path

        from isaaclab.app import AppLauncher

        launcher_args = {"headless": True}
        if request.device is not None:
            launcher_args["device"] = request.device
        if request.video:
            launcher_args["enable_cameras"] = True
        app_launcher = AppLauncher(launcher_args)
    elif _physics_preset(request) in {None, "newton_mjwarp"}:
        # Newton's standalone USD importer needs its fallback PhysX schemas.
        # Set this before visualization/config imports load pxr; never bake it
        # into the container environment where Kit would cache the wrong roots.
        schema_path = os.environ.get("DR02_STANDALONE_PXR_SCHEMA_PATH")
        if schema_path:
            os.environ.setdefault("PXR_PLUGINPATH_NAME", schema_path)

    if mode in {"train", "play"}:
        from .amp_motion_compat import install_legacy_motion_compatibility

        install_legacy_motion_compatibility(request.task)

    # This module imports Isaac Lab's Viser implementation, which imports pxr.
    # Keep it lazy so explicit Kit/PhysX runs can initialize Kit before USD.
    from .visualization import VISUALIZER_MODE_ENV

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
        if app_launcher is not None:
            app_launcher.app.close()


def train_main(argv: Sequence[str] | None = None) -> int:
    """Run RSL-RL training from package-owned command line options."""
    request = build_train_request(sys.argv[1:] if argv is None else argv)
    return _dispatch("train", entrypoints.train, request)


def play_main(argv: Sequence[str] | None = None) -> int:
    """Run task-scoped latest or explicitly selected RSL-RL checkpoint playback."""
    request = build_play_request(sys.argv[1:] if argv is None else argv)
    return _dispatch("play", entrypoints.play, request)
