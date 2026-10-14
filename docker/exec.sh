#!/usr/bin/env bash
set -euo pipefail

container_name="${DR02_CONTAINER_NAME:-dr02-amp-training}"
running="$(docker inspect --format '{{.State.Running}}' "${container_name}" 2>/dev/null || true)"
if [[ "${running}" != "true" ]]; then
    echo "docker/exec.sh: container '${container_name}' is missing or stopped; start it with docker/run.sh." >&2
    exit 1
fi

if (( $# == 0 )); then
    set -- /bin/bash -l
fi
case "${1:-}" in
    dr02-training-train|dr02-training-play)
        command="$1"
        shift
        has_physics_override=0
        for arg in "$@"; do
            case "${arg}" in
                physics=*|+physics=*|++physics=*)
                    has_physics_override=1
                    break
                    ;;
            esac
        done
        if (( has_physics_override == 0 )); then
            set -- "$@" physics=isaacsim_physx
        fi
        isaaclab_python="${ISAACLAB_PYTHON:-/workspace/isaaclab/_isaac_sim/python.sh}"
        launcher="/workspace/isaaclab/_isaac_sim/kit/python/bin/${command}"
        set -- "${isaaclab_python}" "${launcher}" "$@"
        ;;
esac
docker_args=(exec --user "$(id -u):$(id -g)" --workdir /workspace/output)
if [[ -t 0 && -t 1 ]]; then
    docker_args+=(--interactive --tty)
fi
exec docker "${docker_args[@]}" "${container_name}" "$@"
