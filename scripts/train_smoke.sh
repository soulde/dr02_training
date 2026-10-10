#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/.." && pwd)"
cd "${project_root}"

for arg in "$@"; do
    case "${arg}" in
        --num-envs|--num-envs=*|--max-iterations|--max-iterations=*|--run-name|--run-name=*)
            echo "train_smoke.sh fixes --num-envs, --max-iterations, and --run-name" >&2
            exit 2
            ;;
    esac
done

run_id="$(date -u +%Y%m%dT%H%M%S%N)-$$"
dr02-training-train --num-envs 1 --max-iterations 1 --run-name "smoke-minimal-${run_id}" "$@"
dr02-training-train --num-envs 512 --max-iterations 100 --run-name "smoke-representative-${run_id}" "$@"
