# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Mocked host-side tests for the detached Docker workflow."""

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).parents[1]
RSL_COMMIT = "8eef047f1d9b5e07d54d743387da2f0fb42791fa"


def _fake_tools(tmp_path: Path) -> tuple[dict[str, str], Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    calls = tmp_path / "docker-calls.jsonl"
    docker = bin_dir / "docker"
    docker.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "args = sys.argv[1:]\n"
        "with open(os.environ['DOCKER_CALLS'], 'a') as f: f.write(json.dumps(args) + '\\n')\n"
        "if args[:3] == ['container', 'ls', '--all']:\n"
        "    print(os.environ.get('FAKE_CONTAINERS', ''))\n"
        "elif args[:2] == ['image', 'inspect']:\n"
        "    sys.exit(0)\n"
        "elif args and args[0] == 'inspect':\n"
        "    if os.environ.get('FAKE_INSPECT_FAIL') == '1': sys.exit(1)\n"
        "    print(os.environ.get('FAKE_RUNNING', 'true'))\n"
        "elif args and args[0] == 'run':\n"
        "    open(os.environ['DOCKER_RUN_ARGS'], 'w').write(json.dumps(args))\n"
        "    print('fake-container-id')\n"
        "    sys.exit(int(os.environ.get('FAKE_RUN_EXIT', '0')))\n"
        "elif args and args[0] == 'exec':\n"
        "    sys.exit(int(os.environ.get('FAKE_EXEC_EXIT', '0')))\n"
        "elif args and args[0] == 'build':\n"
        "    sys.exit(0)\n",
        encoding="utf-8",
    )
    docker.chmod(0o755)
    git = bin_dir / "git"
    git.write_text(
        "#!/usr/bin/env python3\n"
        "import os, pathlib, sys\n"
        "args = sys.argv[1:]\n"
        "if args[0] == 'clone':\n"
        "    pathlib.Path(args[-1]).mkdir(parents=True)\n"
        "elif args[:2] == ['-C', args[1]] and args[2] == 'fetch': pass\n"
        "elif args[:2] == ['-C', args[1]] and args[2] == 'checkout':\n"
        "    pathlib.Path(args[1], '.fake-commit').write_text(args[-1])\n"
        "elif args[:2] == ['-C', args[1]] and args[2:] == ['rev-parse', 'HEAD']:\n"
        "    print(pathlib.Path(args[1], '.fake-commit').read_text())\n"
        "else: sys.exit(2)\n",
        encoding="utf-8",
    )
    git.chmod(0o755)
    env = os.environ | {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "DOCKER_CALLS": str(calls),
        "DOCKER_RUN_ARGS": str(tmp_path / "docker-run-args.json"),
    }
    return env, calls


def _run(script: str, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(REPO / "docker" / script), *args], cwd=REPO, env=env, check=False, text=True, capture_output=True
    )


def _run_args(env: dict[str, str]) -> list[str]:
    return json.loads(Path(env["DOCKER_RUN_ARGS"]).read_text(encoding="utf-8"))


def _valid_env(tmp_path: Path) -> dict[str, str]:
    motion, output, cache, fork_cache = (tmp_path / name for name in ("motions", "output", "sim-cache", "fork-cache"))
    motion.mkdir()
    output.mkdir()
    return {
        "ACCEPT_EULA": "Y",
        "DR02_MOTION_DIR": str(motion),
        "DR02_OUTPUT_DIR": str(output),
        "DR02_CACHE_DIR": str(cache),
        "DR02_RSL_RL_CACHE_DIR": str(fork_cache),
        "DR02_CONTAINER_NAME": "amp-camera-test",
    }


def test_run_starts_detached_gpu_container_and_mounts_data_read_only(tmp_path):
    env, _ = _fake_tools(tmp_path)
    env.update(_valid_env(tmp_path))
    result = _run("run.sh", [], env)
    assert result.returncode == 0, result.stderr
    args = _run_args(env)
    assert args[:3] == ["run", "--detach", "--interactive"]
    assert "--gpus" in args and args[args.index("--gpus") + 1] == "all"
    assert "--publish" in args and args[args.index("--publish") + 1] == "8080:8080"
    assert any(value.startswith("DR02_RUN_UID=") for value in args)
    assert any(value.startswith("DR02_RUN_GID=") for value in args)
    mounts = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--mount"]
    expected_mounts = (
        f"source={REPO},target=/workspace/dr02_training,readonly",
        f"source={env['DR02_MOTION_DIR']},target=/workspace/motions,readonly",
        f"source={env['DR02_OUTPUT_DIR']},target=/workspace/output",
        f"source={env['DR02_CACHE_DIR']},target=/workspace/cache",
    )
    for mount in expected_mounts:
        assert any(mount in option for option in mounts)
    fork_mount = next(option for option in mounts if "target=/workspace/rsl_rl" in option)
    assert fork_mount.endswith(",readonly")
    assert RSL_COMMIT in next((tmp_path / "fork-cache").rglob(".fake-commit")).read_text()


def test_run_rejects_existing_name_and_missing_eula(tmp_path):
    env, calls = _fake_tools(tmp_path)
    env.update(_valid_env(tmp_path))
    del env["ACCEPT_EULA"]
    missing_eula = _run("run.sh", [], env)
    assert missing_eula.returncode != 0
    assert not calls.exists()

    env["ACCEPT_EULA"] = "Y"
    env["FAKE_CONTAINERS"] = "amp-camera-test"
    conflict = _run("run.sh", [], env)
    assert conflict.returncode != 0
    assert "already exists" in conflict.stderr
    assert not Path(env["DOCKER_RUN_ARGS"]).exists()


def test_run_rejects_invalid_paths_and_container_names(tmp_path):
    env, _ = _fake_tools(tmp_path)
    env.update(_valid_env(tmp_path))
    env["DR02_CONTAINER_NAME"] = "bad/name"
    invalid_name = _run("run.sh", [], env)
    assert invalid_name.returncode != 0
    assert "Docker container name" in invalid_name.stderr
    assert not Path(env["DOCKER_RUN_ARGS"]).exists()

    env["DR02_CONTAINER_NAME"] = "amp-camera-test"
    env["DR02_MOTION_DIR"] = str(tmp_path / "missing-motions")
    invalid_path = _run("run.sh", [], env)
    assert invalid_path.returncode != 0
    assert "DR02_MOTION_DIR" in invalid_path.stderr
    assert not Path(env["DOCKER_RUN_ARGS"]).exists()


def test_run_cleans_new_fork_checkout_if_container_start_fails(tmp_path):
    env, _ = _fake_tools(tmp_path)
    env.update(_valid_env(tmp_path))
    env["FAKE_RUN_EXIT"] = "31"
    result = _run("run.sh", [], env)
    assert result.returncode == 31
    assert not list((tmp_path / "fork-cache").glob("amp-camera-test"))


def test_exec_requires_existing_running_container(tmp_path):
    env, calls = _fake_tools(tmp_path)
    env["DR02_CONTAINER_NAME"] = "amp-camera-test"
    env["FAKE_INSPECT_FAIL"] = "1"
    missing = _run("exec.sh", ["dr02-training-train", "--task", "foo"], env)
    assert missing.returncode != 0
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 1

    env["FAKE_INSPECT_FAIL"] = "0"
    env["FAKE_RUNNING"] = "false"
    stopped = _run("exec.sh", ["bash"], env)
    assert stopped.returncode != 0
    assert len(calls.read_text(encoding="utf-8").splitlines()) == 2

    env["FAKE_RUNNING"] = "true"
    env["FAKE_EXEC_EXIT"] = "29"
    running = _run("exec.sh", ["dr02-training-train", "--task", "foo"], env)
    assert running.returncode == 29
    docker_calls = [json.loads(line) for line in calls.read_text(encoding="utf-8").splitlines()]
    assert len(docker_calls) == 4
    assert "--user" in docker_calls[-1]


def test_entrypoint_installs_mounted_packages_without_dependency_resolution(tmp_path):
    env, calls = _fake_tools(tmp_path)
    uv = Path(env["PATH"].split(":")[0]) / "uv"
    uv.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['UV_CALLS'], 'a') as f: f.write(json.dumps(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    uv.chmod(0o755)
    gosu = Path(env["PATH"].split(":")[0]) / "gosu"
    gosu.write_text(
        "#!/usr/bin/env bash\n"
        "echo \"$1\" > \"$GOSU_ID_FILE\"\n"
        "shift\n"
        "exec \"$@\"\n",
        encoding="utf-8",
    )
    gosu.chmod(0o755)
    fake_id = Path(env["PATH"].split(":")[0]) / "id"
    fake_id.write_text("#!/usr/bin/env bash\necho 0\n", encoding="utf-8")
    fake_id.chmod(0o755)
    isaaclab_python = tmp_path / "python.sh"
    isaaclab_python.write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    isaaclab_python.chmod(0o755)
    project_dir, fork_dir = tmp_path / "project", tmp_path / "rsl_rl"
    project_dir.mkdir()
    fork_dir.mkdir()
    env["ISAACLAB_PYTHON"] = str(isaaclab_python)
    env["DR02_PROJECT_MOUNT"] = str(project_dir)
    env["DR02_RSL_RL_MOUNT"] = str(fork_dir)
    env["DR02_RUN_UID"] = "1234"
    env["DR02_RUN_GID"] = "5678"
    env["GOSU_ID_FILE"] = str(tmp_path / "gosu-id")
    env["UV_CALLS"] = str(tmp_path / "uv-calls.jsonl")
    result = _run("docker-entrypoint.sh", ["bash", "-c", "exit 29"], env)
    assert result.returncode == 29
    installs = [json.loads(line) for line in Path(env["UV_CALLS"]).read_text().splitlines()]
    assert len(installs) == 2
    assert all("--no-deps" in call and "--editable" in call for call in installs)
    assert installs[0][-1] == str(fork_dir)
    assert installs[1][-1] == str(project_dir)
    assert Path(env["GOSU_ID_FILE"]).read_text().strip() == "1234:5678"
