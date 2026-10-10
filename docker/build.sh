#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
image="${DR02_DOCKER_IMAGE:-soulde/dr02-training:isaac-lab-3.0.0-rc1}"
base_image="${DR02_ISAACLAB_BASE_IMAGE:-nvcr.io/nvidia/isaac-lab:3.0.0-rc1}"

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
    echo "docker/build.sh: base image '${base_image}' is not present locally; obtain it explicitly before building." >&2
    exit 1
fi

exec docker build \
    --build-arg "ISAACLAB_BASE_IMAGE=${base_image}" \
    --tag "${image}" \
    --file "${script_dir}/Dockerfile" \
    "$@" \
    "$project_root"
