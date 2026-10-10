# DR02 AMP Runtime and Docker Design

## Goal

Make the public DR02 AMP task package runnable through stable project-owned train/play commands and a reproducible detached Docker workflow, while retaining the package's installable task/asset boundary.

## Current state

- The repository registers AMP flat, rough, and recovery tasks and includes the DR02 source assets and profile.
- `pyproject.toml` uses editable paths into a sibling `IsaacLab-EA` checkout, so a clean public clone cannot resolve its current `uv sync` configuration by itself.
- AMP requires the `soulde/rsl_rl` fork. The local known-good checkout is at `8eef047f1d9b5e07d54d743387da2f0fb42791fa` and exposes the AMP implementation expected by these configs.
- There are no package-owned train/play entry points, project train/play launchers, or Docker files in this repository.
- The maintained robot_lab Docker workflow uses `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`; the project source uses Isaac Lab 3.0 EA APIs. Compatibility must be verified before finalizing this image choice.
- The DR02 URDF, MJCF, and profile already contain fixed links for head, front, and rear D435 cameras. robot_lab defines three 640×480 RGB/depth tiled camera streams for its DR02 velocity tasks; this package has no camera-enabled AMP task variants.

## Design

### Package train/play interface

Add project-prefixed console commands in `pyproject.toml`, implemented inside `dr02_training`, rather than depending on a separate integration checkout. The commands will dispatch to the active Isaac Lab/RSL-RL APIs and translate public hyphenated options to the runner's supported arguments.

`train.sh` will be a thin project-root-aware wrapper. Its defaults are 4096 training environments, at most 512 rendered environments, Viser rendering initially paused, 10000 iterations, and TensorBoard. It will accept `--num-envs` and `--max-iterations` and forward remaining supported options. `play.sh` will default to Viser with 512 environments/rendered environments and the latest checkpoint for the selected task; `--checkpoint` will allow an explicit path or selector. Missing task-specific checkpoints will produce a clear nonzero error.

The package train entry point will own Viser initialization and paused rendering behavior, leaving simulation and optimization running while rendering updates are paused. Play will initialize rendering immediately. TensorBoard will be a declared runtime dependency.

`train_smoke.sh` will run the skill's two acceptance presets sequentially with unique run names: 1 environment × 1 iteration, then 512 environments × 100 iterations. It will stop on the first failure and retain logs/checkpoints under the selected output root. The first preset will use valid rollout/minibatch dimensions for a real optimizer update.

### Camera-equipped AMP variants

Add camera-enabled variants for AMP flat, rough, and recovery tasks. Reuse one DR02 D435 configuration helper for three independently configurable tiled cameras (head, front, rear), with the source configuration's intrinsic matrix, RGB and distance-to-image-plane outputs, 640×480 resolution, 30 Hz update period, and 0.1–10.0 m clipping range. Adapt prim paths to this package's URDF importer settings and test that each camera attaches to the corresponding retained DR02 camera link. Keep the existing camera-free AMP tasks unchanged.

Camera rendering is an environment sensor capability; these variants will preserve the existing AMP policy observations and do not imply camera-based policy input. Preserve the source attribution and Apache-2.0 SPDX notice for adapted camera configuration code, and include the Apache-2.0 license text alongside the existing model license.

### Detached Docker workflow

Add `.dockerignore` plus `docker/Dockerfile`, `docker/docker-entrypoint.sh`, `docker/build.sh`, `docker/run.sh`, and `docker/exec.sh`. `.dockerignore` will exclude local environments, datasets, outputs, caches, and credentials. `run.sh` will use `docker run --detach --interactive --tty --init --gpus all` to create a named, persistent container; it will not use Docker Compose. `exec.sh` will only enter or run commands in that existing container and will fail clearly if it is missing or stopped.

The image will use an explicit Isaac Lab 3.0 image and its actual Python executable. The Dockerfile will pin compatible runtime tools/dependencies and remove any conflicting preinstalled `rsl-rl-lib`. At startup, the entrypoint will editable-install the host-mounted `dr02_training` package and mounted AMP RSL-RL checkout without dependency resolution, then `exec` the requested command.

The host-side `run.sh` will fetch the public RSL-RL repository at pinned commit `8eef047f1d9b5e07d54d743387da2f0fb42791fa` into a temporary host checkout, mount it read-only, and keep it available while the detached container exists. Credentials will not be placed in image layers or command arguments. The project source, RSL-RL source, runtime motion data (read-only), output/checkpoint directory, and simulator caches will be separate mounts. The launcher will validate source/data/output paths and the EULA setting before creating the container. It will never replace or silently start an existing named container.

### Public-clone installation and documentation

Make installation instructions work from a clean public clone without assuming `../IsaacLab-EA`. Preserve the current in-place development workflow for the local EA checkout, while documenting the container path as the reproducible AMP route. Runtime motion data and checkpoints remain external bind mounts and are not baked into the image. Document GPU/NVIDIA Container Toolkit prerequisites, NVIDIA EULA acceptance, headless Viser access, all task commands, container start/exec/stop lifecycle, and common failures.

### Scope boundaries

- No Docker Compose configuration.
- No training dataset, checkpoint, token, or host virtual environment is copied into the image or committed.
- No full training run is part of this change.
- Do not automatically stop, remove, or overwrite any existing container or training process.
- Do not pull/build the large simulator image or run the 512-environment smoke preset until GPU, memory, and active Pueue state have been checked.

## Acceptance criteria

1. A clean package install exposes the project train/play console commands and the `isaaclab.tasks` entry point without importing task registration through an ad-hoc script.
2. Unit tests cover argument translation, default values, checkpoint selection/error cases, and smoke preset sequencing without launching Isaac Sim.
3. The train and play entry points select Viser, apply the visible-environment cap, and implement their required paused/running rendering states; the training loop does not wait for a viewer connection.
4. Camera-enabled AMP flat, rough, and recovery IDs resolve to configs with the three D435 streams attached to the matching links. Camera-free variants keep their original scene and observation dimensions; camera-enabled variants do not silently add images to policy observations.
5. `train.sh`, `train_smoke.sh`, and `play.sh` are executable, parse named options, forward supported runner arguments, and return child exit codes.
6. Docker helper scripts pass shell syntax checks. `run.sh` starts a detached named container with GPU access, mounts task/RSL/data/output/cache paths, and refuses invalid paths, EULA state, and container-name conflicts. `exec.sh` never creates a replacement container.
7. The entrypoint installs mounted packages editable without dependency resolution and forwards signals/exit status with `exec`.
8. Documentation describes environment setup, data expectations, train/play/smoke commands, camera task IDs and sensor outputs, resource expectations, Docker prerequisites/EULA, persistent volumes, and troubleshooting.
9. Existing package, asset, task-registration, and recovery unit tests continue to pass. Runtime smoke checks are run only when the target image and resource checks permit them.

## Risks and verification

- Isaac Lab's visualizer API and RSL-RL script flags vary by release. Inspect the selected image and active source APIs before implementation; do not invent a `viser_start_paused` runner flag.
- The project currently points at Isaac Lab 3.0 EA while the adjacent robot_lab container uses 3.0.0-rc1. Verify package/task compatibility with the candidate image and pin the matching tag in the implementation.
- Docker builds and 512-environment simulations are resource-intensive. Static checks are mandatory; image build and GPU smoke runs require a clean resource check and must not disrupt existing workloads.
