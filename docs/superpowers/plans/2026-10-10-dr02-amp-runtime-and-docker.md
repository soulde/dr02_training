# DR02 AMP Runtime and Docker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add camera-enabled AMP tasks, package-owned train/play commands, and a detached Docker workflow for public use.

**Architecture:** Keep task configuration and command behavior in the installable `dr02_training` package. Thin host scripts and Docker helpers expose that package while motion data, source checkouts, caches, and outputs remain host mounts.

**Tech Stack:** Python 3.12, Isaac Lab 3.0 EA APIs, Gymnasium, RSL-RL AMP fork, Bash, Docker, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-10-10-dr02-amp-runtime-and-docker-design.md`

## Global Constraints

- Train defaults: 4096 environments, 512 rendered environments, initially paused Viser rendering, 10000 iterations, TensorBoard.
- Play defaults: Viser, 512 environments, and the selected task's latest checkpoint; explicit checkpoints override it.
- Smoke presets run sequentially: 1×1 then 512×100, with distinct run names and stop-on-failure.
- Camera variants add three 640×480 D435 RGB/depth tiled cameras at 30 Hz, clipping 0.1–10.0 m, without changing AMP observations.
- RSL-RL source is `https://github.com/soulde/rsl_rl.git` at `8eef047f1d9b5e07d54d743387da2f0fb42791fa`.
- Docker uses `nvcr.io/nvidia/isaac-lab:3.0.0-rc1`, its verified Python executable, and detached `docker run`; no Compose.
- Motion data is mounted read-only; source, fork, caches, and outputs remain host mounts. Never overwrite an existing named container.
- Check `nvidia-smi`, `free -h`, `pueue status`, and active containers before simulator/build resource use. Do not pull the simulator image or run full training.

## Review Focus

- Camera paths must resolve to retained imported links: `test_amp_camera_tasks_are_registered` and `test_amp_camera_cfg_exposes_three_d435_streams`.
- Camera variants must preserve policy observations: `test_amp_camera_cfg_preserves_policy_observations`.
- Viser render pause must leave simulation active: `test_train_viser_starts_rendering_paused_without_pausing_simulation` plus the 1×1 optimizer smoke.
- Checkpoint selection must be task-scoped and fail clearly when absent: `test_play_defaults_to_latest_and_forwards_explicit_checkpoint` and `test_missing_task_checkpoint_returns_nonzero`.
- Docker must reject EULA/name/path errors and retain mounted fork lifetime: `test_run_rejects_existing_name_and_missing_eula` and `test_run_mounts_pinned_fork_and_persistent_paths`.

---

### Task 1: Camera-equipped AMP tasks

**Files:** Create `src/dr02_training/tasks/amp/config/dr02/camera_env_cfg.py`, `licenses/Apache-2.0.txt`, and `tests/test_amp_camera.py`; modify the AMP config `__init__.py`, `tests/test_registration.py`, and `pyproject.toml`.

**Interfaces:** Add `DR02AMPCameraFlatEnvCfg`, `DR02AMPCameraRoughEnvCfg`, and `DR02AMPCameraRecoveryEnvCfg`, registered as `Dr02Training-AMP-{Flat,Rough,Recovery}-Camera-v0`.

- [ ] Write tests for task registration, all three camera link paths/source poses, RGB and depth outputs, resolution, period, clipping range, and unchanged AMP observation terms.
- [ ] Run `source ~/env_isaaclab_ea/bin/activate && pytest tests/test_amp_camera.py tests/test_registration.py -q`; confirm the new cases fail.
- [ ] Implement reusable D435 configs using the actual retained `d435_head_link`, `d435_front_link`, and `d435_rear_link` prim paths. Preserve source attribution/SPDX and declare the Apache license file in package metadata.
- [ ] Rerun the focused tests; commit as `feat: add camera-equipped DR02 AMP variants`.

### Task 2: Package train/play API and Viser behavior

**Files:** Create `src/dr02_training/cli.py`, `src/dr02_training/visualization.py`, `tests/test_cli.py`, and `tests/test_visualization.py`; modify `pyproject.toml` plus AMP flat/rough/recovery, tracking, and velocity base environment configs.

**Interfaces:** `build_train_request(argv: Sequence[str]) -> TrainingRequest`, `build_play_request(argv: Sequence[str]) -> PlaybackRequest`, `train_main(argv: Sequence[str] | None = None) -> int`, `play_main(argv: Sequence[str] | None = None) -> int`, and `make_viser_cfg(*, paused: bool, max_visible_envs: int = 512) -> ViserVisualizerCfg`.

- [ ] Test train/play defaults and hyphenated argument translation, preserved runner arguments, task-scoped latest selection and missing-checkpoint error, plus visualizer render/simulation states.
- [ ] Run `source ~/env_isaaclab_ea/bin/activate && pytest tests/test_cli.py tests/test_visualization.py -q`; confirm failures.
- [ ] Implement console scripts `dr02-training-train` and `dr02-training-play` via public `isaaclab_rl.entrypoints` request APIs. Use a package-owned Viser subclass to pause rendering in training while simulation remains active; play renders immediately. Set `max_visible_envs=512` and declare Viser/TensorBoard runtime dependencies.
- [ ] Rerun focused tests; commit as `feat: add package-owned DR02 train and play commands`.

### Task 3: Host train/play/smoke launchers

**Files:** Create executable `scripts/train.sh`, `scripts/play.sh`, `scripts/train_smoke.sh`, and `tests/test_launchers.py`.

**Interfaces:** Launchers call only the package console commands from Task 2 and pass through supported overrides.

- [ ] Test defaults/overrides, task and checkpoint forwarding, smoke order `(1,1)` then `(512,100)`, unique run names, and immediate failure propagation.
- [ ] Run `source ~/env_isaaclab_ea/bin/activate && pytest tests/test_launchers.py -q`; confirm failure.
- [ ] Implement thin wrappers with no environment/session/Docker/training logic; preserve child exit status.
- [ ] Run focused tests and `bash -n scripts/train.sh scripts/play.sh scripts/train_smoke.sh`; commit as `feat: add DR02 AMP train and play launchers`.

### Task 4: Detached Docker lifecycle

**Files:** Create `.dockerignore`, `docker/Dockerfile`, `docker/docker-entrypoint.sh`, `docker/build.sh`, `docker/run.sh`, `docker/exec.sh`, and `tests/test_docker_scripts.py`.

**Interfaces:** `build.sh` builds the image; `run.sh` creates a detached named container; `exec.sh` runs a command only in that existing running container; entrypoint installs mounted packages editable without dependency resolution and execs the requested command.

- [ ] Use a fake Docker CLI to test detached GPU invocation, separate source/fork/data/output/cache mounts, read-only motion and fork mounts, published Viser port 8080, EULA/path/name rejection, exec behavior, and entrypoint installation.
- [ ] Run `source ~/env_isaaclab_ea/bin/activate && pytest tests/test_docker_scripts.py -q`; confirm failure.
- [ ] Implement the pinned image/runtime and host-side pinned public fork checkout. Keep the fork checkout alive for the detached container lifetime and clean it up if startup fails; require `ACCEPT_EULA=Y`; never replace a container.
- [ ] Run focused tests and `bash -n docker/*.sh`; do not build or pull the simulator image. Commit as `feat: add detached Isaac Lab Docker workflow`.

### Task 5: Public installation and operations docs

**Files:** Modify `README.md`.

- [ ] Cross-check documented IDs/options against registration and launchers.
- [ ] Document clean-clone installation without assuming sibling `../IsaacLab-EA`, motion data expectations, train/play/smoke and camera task usage, detached Docker lifecycle/EULA/mounts, Viser access, resource needs, and troubleshooting.
- [ ] Run `source ~/env_isaaclab_ea/bin/activate && pytest tests/test_registration.py -q && uv build`; inspect wheel/sdist contents for package configs/license and absence of data/checkpoints. Commit as `docs: document DR02 AMP runtime and Docker workflow`.

### Task 6: Full verification

- [ ] Run `source ~/env_isaaclab_ea/bin/activate && pytest -q && pre-commit run --all-files`.
- [ ] Inspect `nvidia-smi`, `free -h`, `pueue status`, and `docker ps`; record current resource use without disrupting processes.
- [ ] Run registration/config checks. Run 1×1 and then 512×100 smoke only if resources are available and the simulator image is already present; verify optimizer progress, Viser access, and host checkpoint/log writes.
- [ ] Run `uv build`, `git diff --check`, and `git status --short --branch`; verify distributions contain intended code/licenses and no data, cache, credentials, or local environment.

## Plan self-review

- Spec coverage: Tasks 1–5 cover camera variants/license, package API/visualization, launchers, detached Docker, and public documentation; Task 6 covers the resource-gated checks.
- Step scan: each implementation unit starts with observable tests, confirms the initial failure, then implements and verifies before committing.
- Interface consistency: Task 3 consumes Task 2's named console commands; Docker installs and invokes those same package commands.
- Review focus maps camera links, observation shape, pause semantics, checkpoint scope, and Docker lifecycle to named tests; runtime optimizer progress remains a gated smoke check.
- Proportion: six independently reviewable deliverables map directly to the approved design; details are limited to fixed names, values, and checks needed by implementers.
