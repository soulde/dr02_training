#!/usr/bin/env bash
set -euo pipefail

isaaclab_python="${ISAACLAB_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
project_dir="${DR02_PROJECT_MOUNT:-/workspace/dr02_training}"
rsl_rl_dir="${DR02_RSL_RL_MOUNT:-/workspace/rsl_rl}"
if [[ ! -x "${isaaclab_python}" ]]; then
    echo "Isaac Lab Python executable is missing: ${isaaclab_python}" >&2
    exit 1
fi
if [[ ! -d "${project_dir}" || ! -d "${rsl_rl_dir}" ]]; then
    echo "Expected mounted package paths at ${project_dir} and ${rsl_rl_dir}" >&2
    exit 1
fi

rsl_rl_install_dir="$(mktemp -d /tmp/dr02-rsl-rl.XXXXXX)"
cp -a -- "${rsl_rl_dir}/." "${rsl_rl_install_dir}/"
uv pip install --python "${isaaclab_python}" --no-deps --editable "${rsl_rl_install_dir}"
uv pip install --python "${isaaclab_python}" --no-deps --editable "${project_dir}/source/dr02_asset"
uv pip install --python "${isaaclab_python}" --no-deps --editable "${project_dir}/source/dr02_training"

if [[ "$(id -u)" == "0" && "${DR02_RUN_UID:-}" =~ ^[0-9]+$ && "${DR02_RUN_GID:-}" =~ ^[0-9]+$ ]]; then
    exec gosu "${DR02_RUN_UID}:${DR02_RUN_GID}" "$@"
fi

exec "$@"
