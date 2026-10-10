# dr02_training

An installable downstream Isaac Lab task package generated with a standard uv `src` layout.

The registered tasks are:

- `Dr02Training-Velocity-DR02`
- `Dr02Training-AMP-Flat-v0`
- `Dr02Training-AMP-Rough-v0`
- `Dr02Training-AMP-Recovery-v0`
- `Dr02Training-Tracking-Flat-v0`

DR02 Pro URDF, MJCF, meshes, the shared robot profile, and Isaac Lab asset configuration are in the installable
package. AMP and tracking motions remain user-provided runtime data. Set `DR02_AMP_MOTION_DIR` to a directory of
BeyondMimic-compatible DR02 NPZ files for AMP. Set `DR02_TRACKING_MOTION_FILE` to one NPZ file for tracking, or
override `commands.motion.motion_file` on the Isaac Lab command line. The bundled DR02 robot profile records the
29-joint order and the 32-body reference dataset order, including the two reconstructed toe sites.

The AMP recovery task samples each reset as standard AMP reference initialization (85%) or a randomized recovery
pose (15%). Recovery poses receive a world-up force sampled from 0–450 N, linearly annealed over the default 10,000
training iterations, and low-root/contact/orientation termination is deferred for three seconds. The recovery task
uses the same AMP runner and policy architecture as the flat task.

The AMP task uses the AMP extension in the maintained RSL-RL fork. Install that RSL-RL checkout into the target
Isaac Lab environment after the editable task package; the base Isaac Lab RSL-RL extra alone does not provide AMP.

## Installation

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then create the project environment. The default
environment uses the Newton backend and does not install Isaac Sim:

```bash
uv sync
```

This project was generated from an Isaac Lab source checkout. Its `pyproject.toml` uses editable relative paths to that
checkout, so regenerate the project or update `[tool.uv.sources]` if either directory moves.

Optional backends are available through extras. Pass the extra to each `uv run` command that needs it:

```bash
# Standalone OV PhysX
uv run --extra ovphysx isaaclab random_agent --task <TASK_NAME> physics=ovphysx

# Isaac Sim with PhysX and RTX rendering
uv run --extra isaacsim isaaclab random_agent --task <TASK_NAME> physics=isaacsim_physx
```

The `ov` extra installs both the `ovphysx` and `ovrtx` runtimes. You can also select `ovrtx` independently when using
Newton physics with the OVRTX renderer. Commit `pyproject.toml` and `uv.lock` so collaborators use the same environment.

## Run the generated tasks

Replace the placeholders below with a task listed above and a selected RL library.

```bash
# List this project's environments and their available presets
uv run python scripts/list_envs.py --show_presets

# Exercise an environment without a trained policy
uv run isaaclab zero_agent --task <TASK_NAME> --num_envs 16
uv run isaaclab random_agent --task <TASK_NAME> --num_envs 16

# Run DR02 tracking with an existing local reference clip
DR02_TRACKING_MOTION_FILE=/path/to/dr02_motion.npz uv run isaaclab zero_agent \
  --task Dr02Training-Tracking-Flat-v0 --num_envs 4

# Train DR02 AMP with an existing local motion dataset and the AMP-enabled RSL-RL fork
DR02_AMP_MOTION_DIR=/path/to/dr02_motions uv run isaaclab train \
  --rl_library rsl_rl --task Dr02Training-AMP-Flat-v0

# Train DR02 AMP on rough terrain with the same motion dataset
DR02_AMP_MOTION_DIR=/path/to/dr02_motions uv run isaaclab train \
  --rl_library rsl_rl --task Dr02Training-AMP-Rough-v0

# Train DR02 AMP with standard and fall-recovery episodes
DR02_AMP_MOTION_DIR=/path/to/dr02_motions uv run isaaclab train \
  --rl_library rsl_rl --task Dr02Training-AMP-Recovery-v0

# Train and play
uv run isaaclab train --rl_library <RL_LIBRARY> --task <TASK_NAME>
uv run isaaclab play --rl_library <RL_LIBRARY> --task <TASK_NAME> --checkpoint latest

# Distributed training
uv run isaaclab train_multigpu --rl_library <RL_LIBRARY> --task <TASK_NAME> --num_gpus 2

# Benchmark startup, runtime, training, or play
uv run isaaclab benchmark runtime --task <TASK_NAME> --num_envs 16 --num_steps 1000
uv run isaaclab benchmark training --rl_library <RL_LIBRARY> --task <TASK_NAME> --max_iterations 10
```

Use `physics=<PRESET>` to select one of the presets shown by `list_envs.py`.

## Project structure

Task families live in `src/dr02_training/tasks`. Within a manager-based family, task-wide MDP terms live in `mdp`, while
robot-specific scenes, registrations, and agent configurations live in `config/dr02`. Add another robot by
creating a sibling of `config/dr02`; add another task by creating a sibling task-family directory.

## Development

Run the generated registration test and code-quality checks through the project environment:

```bash
uv run pytest
uv run pre-commit run --all-files
```

The test helpers under `source/isaaclab_tasks/test` in the Isaac Lab repository are not part of the installed
`isaaclab_tasks` package. Keep test fixtures in this project and use public Isaac Lab APIs. If you copy
`env_test_utils.py`, it becomes vendored code whose upstream changes you must track.

To configure VS Code or Cursor, run the `setup_python_env` task or invoke its command directly:

```bash
uv run isaaclab --editor
```

The setup command selects the active interpreter and generates a git-ignored `pyrightconfig.json`. The generated
configuration inherits the project's checked-in Pyright settings and adds the Isaac Sim extensions, project `src` root,
and any Isaac Lab packages discovered in the active Python environment. This supports both Pylance in VS Code and
basedpyright in Cursor.

In VS Code, use Pylance and select the interpreter that ran the setup command. In Cursor, install the
[basedpyright extension](https://marketplace.visualstudio.com/items?itemName=detachhead.basedpyright) instead of
Pylance, select the same interpreter, and reload the window. Both language servers read `pyrightconfig.json`.

When using the `isaacsim` extra, include it while generating the editor configuration so the command can discover the
Isaac Sim installation:

```bash
uv run --extra isaacsim isaaclab --editor
```

For an Isaac Sim binaries installation that is not available in the project environment, provide its path explicitly:

```bash
# Linux
uv run isaaclab --editor --isaac_path <isaac-sim-path>

# Windows
uv run isaaclab --editor --isaac_path <isaac-sim-path>
```

## Troubleshooting

If Pylance or basedpyright cannot resolve modules, confirm that the selected interpreter matches the one used to run the
setup command, then reload the editor window. To add a missing extension or reduce indexing memory, edit the `extraPaths`
array in the root `pyrightconfig.json`; remove simulator extension directories that the project does not use.
