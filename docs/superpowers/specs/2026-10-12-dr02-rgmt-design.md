# DR02 RGMT Task Design

## Goal

Add an RGMT-style reference-motion tracking task for DR02 Pro, using the RGMT task in `chocolate_training` as the behavioral reference while deriving all observation and action dimensions from DR02's 29-joint profile. Keep the existing DR02 tracking and AMP task IDs and behavior intact.

## Scope

- Register flat training and single-environment playback tasks as `Dr02Training-RGMT-Flat-v0` and `Dr02Training-RGMT-Flat-Play-v0`.
- Implement an RGMT-specific motion command that loads one NPZ, a directory, a glob, or a list of motion clips, aligns arrays by embedded names (with the existing configured-order fallback for legacy files), samples clips and frames, and builds a 21-frame command window clamped to each clip's boundaries.
- Provide 10-frame proprioceptive histories, RGMT observation terms, reference-joint-position residual actions, and a Transformer actor with asymmetric critic through the existing RSL-RL PPO model interface.
- Add recovery reset sampling for 15% of environments, upward assistance sampled up to 200 N and annealed over 100 million environment steps, and a 3-second termination grace period.
- Reuse DR02's existing tracking rewards, robot profile, articulation, and physics configuration where applicable.
- Document motion inputs, task IDs, configuration overrides, and adaptation values.

## Non-goals

- Modifying the general `rsl_rl` package or PPO algorithm.
- Changing existing tracking, AMP, or velocity tasks.
- Providing or transforming motion datasets, or launching a training run.
- Claiming an exact paper reproduction. RGMT is adapted to DR02's actuator model, dimensions, and assets.

## Architecture

All RGMT code remains under `dr02_training.tasks.rgmt`. The task-specific model implements the same constructor and forward/distribution methods expected by the installed RSL-RL PPO runner. It uses local PyTorch attention blocks so no general RSL-RL fork changes are required. Actor inputs are current noisy proprioception, explicit proprioceptive history, and a reference command sequence. The history encoder uses causal self-attention and max pooling to form a query for cross-attention over command tokens. Actor output is a Gaussian residual around the reference joint position. The critic uses clean proprioception, the same history and command context, and privileged body state; actor and critic are independent model instances.

For 29 actuated joints, the RGMT feature contracts are 93 values per proprioceptive frame (`projected_gravity[3]`, `base_angular_velocity[3]`, relative joint positions `[29]`, joint velocities `[29]`, previous actions `[29]`) and 38 values per command frame (`base_linear_velocity[3]`, base angular velocity `[3]`, projected gravity `[3]`, reference joint positions `[29]`). The temporal lengths are 10 history frames and 21 command frames, with a 128-wide attention embedding. Command-window indices clamp to the active clip's first and last frame; reset initializes history by repeating the reset observation.

The RGMT motion command aligns reference bodies and joints to `TRACKING_BODY_NAMES` and `JOINT_ORDER`, retains per-clip offsets and lengths, samples clips uniformly by default, and exposes the current reference state required by the existing tracking rewards. Invalid or inconsistent motion arrays fail during command initialization with a path-specific error. A selected clip is fixed for an episode, and frame progression never indexes another clip.

Recovery behavior is task-local. At reset, the configured recovery fraction selects randomized recovery initial states; other environments begin from sampled reference states. Selected recovery environments receive a sampled world-up assistance force on the configured anchor body, with its maximum linearly annealed from 200 N to zero over 100 million environment steps. Tracking instability terminations are deferred for 3 seconds for recovery environments; timeout remains active. These are Chocolate port defaults and project adaptations, not values established by the RGMT publication except where noted in its reproduction notes.

## Files and integration

- `tasks/rgmt/mdp/rgmt_motion.py`: clip discovery, validation, alignment, reference command and temporal windows.
- `tasks/rgmt/mdp/rgmt_mdp.py`: proprioception/history buffers, residual joint action, recovery events, and grace-period terminations.
- `tasks/rgmt/mdp/rgmt_policy.py`: task-local Transformer actor/critic implementing RSL-RL's model API.
- `tasks/rgmt/config/dr02/env_cfg.py` and `agents/rsl_rl_rgmt_cfg.py`: environment and PPO configuration.
- `tasks/rgmt/config/dr02/__init__.py` and task package initializers: Gym registration and import discovery.
- Existing registration tests and README: task entry point contracts and user-facing task/data documentation.

## Failure handling

Motion loading checks required keys, compatible array ranks and frame counts, positive finite FPS, finite numeric values, and name coverage. It rejects empty clip collections and inconsistent joint/body contracts before simulation training proceeds. Clip boundary behavior is deterministic clamping. Missing runtime motion paths use a clear configuration error, consistent with the current tracking task's required local data.

## Validation and acceptance

- Registration resolves both RGMT IDs to the intended environment and runner configs.
- Configuration inspection confirms 29-action output and DR02 joint/body ordering.
- Synthetic tensors with batch size greater than one produce finite actor actions and critic values with the expected shapes.
- Motion-window indices remain inside each selected clip at both boundaries, and episode history resets to repeated initial observations.
- The residual action target equals reference joint position plus scaled policy residual.
- Recovery selection, assistance annealing, and grace-period termination are represented in config and state updates.
- Existing tasks and their IDs remain unchanged.

No simulator training or motion-data download is part of implementation validation. A training smoke run requires the user's chosen motion corpus and explicit runtime authorization under the machine resource rules.

## Adaptation values

The temporal lengths (10, 21) and embedding width (128) follow the Chocolate RGMT reproduction. Attention head count (4), feed-forward width (256), one history Transformer layer, GELU activation, and zero dropout follow that implementation's choices. PPO starts from DR02 tracking settings. Recovery fraction (15%), assistance ceiling (200 N), grace window (3 s), pose ranges, and annealing duration (100 million environment steps) follow Chocolate's documented defaults; the publication does not specify all low-level embodiment-specific values. These values remain explicit configuration fields.

## Risks

- DR02 and Chocolate motion exports may use different name sets and quaternion conventions; alignment must use the existing DR02 tracking conventions and known source metadata fallback.
- The installed RSL-RL runner may make assumptions about flattened observations or stochastic distribution outputs; the custom model must honor its current `MLPModel` API exactly.
- Recovery force application depends on the Isaac Lab 3.0 Early Access articulation API and must be represented without changing the current AMP recovery implementation.
- Multi-clip datasets can consume substantial GPU memory if all arrays are copied to device. The loader should keep only the tensors needed for training and move them once, with clip counts and frame totals logged.
