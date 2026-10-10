#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
container_name="${DR02_CONTAINER_NAME:-dr02-amp-training}"
image="${DR02_DOCKER_IMAGE:-soulde/dr02-training:isaac-lab-3.0.0-rc1}"
rsl_url="https://github.com/soulde/rsl_rl.git"
rsl_commit="8eef047f1d9b5e07d54d743387da2f0fb42791fa"
motion_dir="${DR02_MOTION_DIR:-}"
output_dir="${DR02_OUTPUT_DIR:-}"
cache_dir="${DR02_CACHE_DIR:-${HOME}/.cache/ov}"
fork_cache="${DR02_RSL_RL_CACHE_DIR:-${HOME}/.cache/dr02_training/rsl_rl}"

fail() {
    echo "docker/run.sh: $*" >&2
    exit 1
}

[[ "${ACCEPT_EULA:-}" == "Y" ]] || fail "set ACCEPT_EULA=Y after accepting the NVIDIA Isaac Lab license."
[[ -n "${motion_dir}" && -d "${motion_dir}" ]] || fail "DR02_MOTION_DIR must point to an existing motion-data directory."
[[ -n "${output_dir}" ]] || fail "DR02_OUTPUT_DIR must name a persistent host output directory."
[[ "${container_name}" =~ ^[A-Za-z0-9][A-Za-z0-9_.-]*$ ]] || fail "invalid Docker container name '${container_name}'."
mkdir -p "${output_dir}" "${cache_dir}" "${fork_cache}"
motion_dir="$(realpath "${motion_dir}")"
output_dir="$(realpath "${output_dir}")"
cache_dir="$(realpath "${cache_dir}")"
fork_cache="$(realpath "${fork_cache}")"
for host_path in "${project_root}" "${motion_dir}" "${output_dir}" "${cache_dir}" "${fork_cache}"; do
    [[ "${host_path}" != *,* ]] || fail "Docker bind-mount paths cannot contain commas: '${host_path}'."
done

if docker container ls --all --format '{{.Names}}' | grep -Fxq "${container_name}"; then
    fail "container '${container_name}' already exists; inspect or remove it explicitly before retrying."
fi
docker image inspect "${image}" >/dev/null 2>&1 || fail "image '${image}' is not present locally; build it explicitly with docker/build.sh."

fork_dir="${fork_cache}/${container_name}"
fork_created=0
cleanup_on_failure() {
    status=$?
    if (( status != 0 && fork_created == 1 )); then
        rm -rf -- "${fork_dir}"
    fi
    exit "$status"
}
trap cleanup_on_failure EXIT

if [[ -d "${fork_dir}" ]]; then
    existing_commit="$(git -C "${fork_dir}" rev-parse HEAD 2>/dev/null || true)"
    [[ "${existing_commit}" == "${rsl_commit}" ]] || fail "fork checkout exists at '${fork_dir}' with a different revision."
else
    fork_created=1
    git clone --no-checkout "${rsl_url}" "${fork_dir}" || fail "failed to clone the public RSL-RL fork."
    git -C "${fork_dir}" fetch --depth 1 origin "${rsl_commit}" || fail "failed to fetch pinned RSL-RL revision."
    git -C "${fork_dir}" checkout --detach "${rsl_commit}" || fail "failed to check out pinned RSL-RL revision."
fi

docker run --detach --interactive --tty --init --gpus all \
    --name "${container_name}" \
    --publish "${DR02_VISER_PORT:-8080}:8080" \
    --workdir /workspace/output \
    --mount "type=bind,source=${project_root},target=/workspace/dr02_training,readonly" \
    --mount "type=bind,source=${fork_dir},target=/workspace/rsl_rl,readonly" \
    --mount "type=bind,source=${motion_dir},target=/workspace/motions,readonly" \
    --mount "type=bind,source=${output_dir},target=/workspace/output" \
    --mount "type=bind,source=${cache_dir},target=/workspace/cache" \
    --env ACCEPT_EULA=Y \
    --env DR02_AMP_MOTION_DIR=/workspace/motions \
    --env "DR02_RUN_UID=$(id -u)" \
    --env "DR02_RUN_GID=$(id -g)" \
    --env ISAACLAB_PYTHON=/workspace/isaaclab/_isaac_sim/python.sh \
    --env XDG_CACHE_HOME=/workspace/cache \
    "${image}" bash -lc 'while :; do sleep 3600; done'
