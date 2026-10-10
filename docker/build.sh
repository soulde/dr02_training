#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
image="${DR02_DOCKER_IMAGE:-soulde/dr02-training:isaac-lab-3.0.0-rc1}"

exec docker build --tag "${image}" --file "${script_dir}/Dockerfile" "$project_root" "$@"
