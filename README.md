# DR02 Training

An Isaac Lab project for the DeepRobotics DR02 Pro, organized as sibling `dr02_asset` and `dr02_training` packages
under `source/`. The asset package contains the adapted URDF, MJCF, meshes, shared robot profile, and Isaac Lab asset
configuration. AMP and tracking motion data stay outside the repository and are supplied at runtime.

## Tasks

| Task ID | Description |
| --- | --- |
| `Dr02Training-AMP-Flat-v0` | AMP motion prior on flat terrain |
| `Dr02Training-AMP-Rough-v0` | AMP motion prior on rough terrain |
| `Dr02Training-AMP-Recovery-v0` | AMP with 15% randomized recovery resets, annealed upward assistance, and a 3-second recovery window |
| `Dr02Training-AMP-Flat-Camera-v0` | Flat AMP with head/front/rear D435 RGB-D sensors |
| `Dr02Training-AMP-Rough-Camera-v0` | Rough AMP with head/front/rear D435 RGB-D sensors |
| `Dr02Training-AMP-Recovery-Camera-v0` | Recovery AMP with head/front/rear D435 RGB-D sensors |
| `Dr02Training-Tracking-Flat-v0` | Reference motion tracking |
| `Dr02Training-RGMT-Flat-v0` | RGMT temporal motion tracking with a 10-frame history and 21-frame command window |
| `Dr02Training-RGMT-Flat-Play-v0` | Single-environment playback for an RGMT checkpoint |
| `Dr02Training-Velocity-DR02` | Velocity-command locomotion |

Each camera variant attaches three 640×480, 30 Hz RGB and distance-to-image-plane streams to the fixed D435 links in
the DR02 URDF. The camera variants retain the original AMP policy, critic, and discriminator observation terms; they
do not add camera images to the policy input.

The recovery task uses one AMP policy and the same runner architecture as flat AMP. Each reset is a reference motion
reset with 85% probability or a randomized recovery reset with 15% probability. The recovery curriculum applies up to
450 N of world-up assistance and anneals it over the default 10,000 training iterations. Low-root, trunk-contact, and
orientation termination are deferred for three seconds so the policy can attempt to recover.

## Motion data

For AMP, set `DR02_AMP_MOTION_DIR` to a directory of BeyondMimic-compatible DR02 `.npz` motion files. For tracking,
set `DR02_TRACKING_MOTION_FILE` to one compatible `.npz` file, or override `commands.motion.motion_file` with a Hydra
argument. The repository does not include motion datasets.

RGMT accepts one `.npz` file, a directory, a quoted glob, or an explicit list through `commands.motion.motion_path`.
`DR02_RGMT_MOTION_PATH` can provide the path by default. Each file contains `fps`, `body_pos_w`, `body_quat_w`,
`body_lin_vel_w`, `body_ang_vel_w`, `joint_pos`, and `joint_vel`. Include `body_names` and `joint_names` arrays to align
the clip by name. Legacy files without metadata are interpreted in the configured `AMP_MOTION_BODY_NAMES` and
`JOINT_ORDER` source order. Body quaternions use BeyondMimic's wxyz order. RGMT loads clips in sorted path order and
clamps the 21-frame window to each clip's boundaries; a clip remains assigned to an environment for that episode.

The RGMT policy uses DR02's 29 joints, 93 proprioceptive features per frame, 38 command features per frame, and a
128-dimensional attention embedding. Recovery defaults select 15% of reset environments, sample up to 200 N of
upward assistance, anneal the maximum over 100 million environment steps, and allow a 3-second termination grace
period. Attention depth, noise, PPO, and recovery pose values are project adaptations where the paper does not specify
them for DR02. The task-specific action, observation, and effective actuator-gain contract is stored at
`source/dr02_training/dr02_training/tasks/rgmt/deploy/rgmt/flat.json`; RGMT reads its action scale from that contract.

## Install from a clean clone

Use an installed Isaac Lab 3.0 Early Access environment with Python 3.12. The editable source entries in
`[tool.uv.sources]` are for this repository's local development checkout; a clean clone should install into the
already prepared Isaac Lab environment without resolving or reinstalling its simulator dependencies:

```bash
source /path/to/env_isaaclab_ea/bin/activate
python -m pip install --no-deps --editable ./source/dr02_asset
python -m pip install --no-deps --editable ./source/dr02_training
```

The Isaac Lab environment must already provide Isaac Lab, Isaac Sim/Newton, Gymnasium, the Isaac Lab visualizers, and
the RSL-RL runner dependencies. AMP also requires the public `soulde/rsl_rl` fork at the revision used by this project:

```bash
git clone https://github.com/soulde/rsl_rl.git
git -C rsl_rl checkout --detach 8eef047f1d9b5e07d54d743387da2f0fb42791fa
python -m pip install --no-deps --editable ./rsl_rl
```

RGMT also requires the reusable Transformer modules from RSL-RL commit
`4fc611ebdd71ec6971314ed6f5c1f4b1a7319601` (`feat/rgmt-policy-reproduction`). That commit is currently available in
the local RSL-RL checkout and has no published `origin` branch. Make it available in your RSL-RL repository, then use
a separate worktree so the general AMP checkout can stay on its own revision:

```bash
git -C /path/to/rsl_rl worktree add --detach /path/to/rsl_rl-rgmt 4fc611ebdd71ec6971314ed6f5c1f4b1a7319601
python -m pip install --no-deps --editable /path/to/rsl_rl-rgmt
```

Install the project editable into the same Isaac Lab environment after installing the RGMT RSL-RL worktree.

The training package registers tasks through the `isaaclab.tasks` entry point and installs `dr02-training-train` and
`dr02-training-play` commands. The asset and training distributions are separate, with the training package depending
on the asset package. The root `pyproject.toml` is a uv workspace for local development; its Isaac Lab source paths
reference a neighboring `../IsaacLab-EA` directory.

## Train, smoke test, and play

Run the launchers from the activated Isaac Lab environment. Training defaults to 4096 environments, 10,000 iterations,
TensorBoard logging, Viser with at most 512 visible environments, and rendering paused while simulation and optimization
continue. Open `http://localhost:8080` and click the Viser `Pause Rendering` button once to resume rendering. Use
`--num-envs` and `--max-iterations` to override the first two values.

```bash
DR02_AMP_MOTION_DIR=/data/dr02/motions scripts/train.sh --task Dr02Training-AMP-Flat-v0
DR02_AMP_MOTION_DIR=/data/dr02/motions scripts/train.sh \
  --task Dr02Training-AMP-Recovery-v0 --num-envs 2048 --max-iterations 5000

# The two fixed acceptance presets run sequentially and stop at the first failure.
DR02_AMP_MOTION_DIR=/data/dr02/motions scripts/train_smoke.sh --task Dr02Training-AMP-Recovery-v0

# Play selects the latest checkpoint from the selected task; specify a file or supported selector to override it.
scripts/play.sh --task Dr02Training-AMP-Recovery-v0
scripts/play.sh --task Dr02Training-AMP-Recovery-v0 --checkpoint /path/to/model_10000.pt

# Train RGMT from a user-provided multi-clip corpus and play with one environment.
DR02_RGMT_MOTION_PATH='/data/dr02/rgmt/*.npz' scripts/train.sh \
  --task Dr02Training-RGMT-Flat-v0 --num-envs 2048
DR02_RGMT_MOTION_PATH='/data/dr02/rgmt/*.npz' scripts/play.sh \
  --task Dr02Training-RGMT-Flat-Play-v0 --num-envs 1 --checkpoint /path/to/model_10000.pt
```

`train_smoke.sh` runs 1 environment × 1 iteration and then 512 environments × 100 iterations, with distinct run names.
Use `scripts/train.sh` for arbitrary sizes. Logs and checkpoints are written under the current working directory's
`logs/rsl_rl/<experiment>` tree. `play.sh` requires a checkpoint for the chosen task and exits with an error if none is
available; it does not select another task's policy.

Camera task IDs are available for training and playback using the same launchers. They enable camera sensors for
rendering or sensor-side experiments, while the current AMP policy remains proprioceptive.

## Detached Docker workflow

Docker is the reproducible AMP route when a compatible host Isaac Lab environment is not already installed. The host
needs Docker Engine, NVIDIA Container Toolkit, a supported NVIDIA driver/GPU, and enough disk space for the simulator
image. `docker/build.sh` builds `soulde/dr02-training:isaac-lab-3.0.0-rc1` from the local Isaac Lab 3.0.0-rc1 image;
the scripts never pull that large base image automatically.

After accepting NVIDIA's Isaac Lab license, build the project image and start a detached named container:

```bash
docker/build.sh

ACCEPT_EULA=Y \
DR02_MOTION_DIR=/data/dr02/motions \
DR02_OUTPUT_DIR="$PWD/experiments/dr02-amp" \
DR02_CONTAINER_NAME=dr02-amp-training \
docker/run.sh
```

`run.sh` requires the base image to be present locally and the motion directory to exist. It clones the pinned public
RSL-RL fork on the host, mounts the project and fork read-only, mounts motion data read-only, and keeps output and
simulator caches on the host. Training logs/checkpoints go to `DR02_OUTPUT_DIR`. Viser port 8080 is published; set
`DR02_VISER_PORT` to change the host port. The named container runs in the background and is not replaced if it
already exists.

Run training or playback inside the running container:

```bash
docker/exec.sh dr02-training-train --task Dr02Training-AMP-Recovery-v0
docker/exec.sh dr02-training-play --task Dr02Training-AMP-Recovery-v0 --checkpoint latest
```

Stop and remove the container explicitly when finished. The pinned fork checkout is retained at
`~/.cache/dr02_training/rsl_rl/<container-name>` for as long as the container can use its bind mount; after removing
the container, remove that matching checkout if it is no longer needed:

```bash
docker stop dr02-amp-training
docker rm dr02-amp-training
rm -rf "$HOME/.cache/dr02_training/rsl_rl/dr02-amp-training"
```

The host output directory is independent and is not removed with the container. The image excludes motion data,
checkpoints, credentials, and host environments. The container copies the read-only fork checkout to a writable
temporary directory for its setuptools editable install, installs both mounted source packages editable without
dependency resolution, then runs as the host user's numeric UID/GID so new output files remain manageable.

## Troubleshooting

- If the task cannot find motion files, check `DR02_AMP_MOTION_DIR` (host) or the `DR02_MOTION_DIR` passed to
  `docker/run.sh`; the container receives the read-only path `/workspace/motions`.
- If AMP imports fail, confirm the installed RSL-RL checkout is exactly
  `8eef047f1d9b5e07d54d743387da2f0fb42791fa`.
- If playback reports no checkpoint, verify the selected task's `logs/rsl_rl/<experiment>` directory or pass an
  explicit checkpoint path.
- If Viser is unreachable, check that the container is running and the configured host port (`8080` by default) is
  published and not blocked by a firewall.
- If Docker reports a name conflict, inspect that container and remove it explicitly if it is no longer needed.

## Development checks

```bash
source ~/env_isaaclab_ea/bin/activate
pytest -q
pre-commit run --all-files
```
