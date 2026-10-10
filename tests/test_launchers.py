# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Shell-level tests for the thin project launchers."""

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).parents[1]


def _stub_command(tmp_path: Path, *, exit_code: int = 0) -> tuple[dict[str, str], Path]:
    log = tmp_path / "calls.jsonl"
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir()
    command = stub_dir / "dr02-training-train"
    command.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "with open(os.environ['LAUNCHER_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps(sys.argv[1:]) + '\\n')\n"
        f"sys.exit({exit_code})\n",
        encoding="utf-8",
    )
    command.chmod(0o755)
    play = stub_dir / "dr02-training-play"
    play.write_text(command.read_text(encoding="utf-8"), encoding="utf-8")
    play.chmod(0o755)
    env = os.environ | {"PATH": f"{stub_dir}:{os.environ['PATH']}", "LAUNCHER_LOG": str(log)}
    return env, log


def _run(script: str, args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run([str(REPO / "scripts" / script), *args], cwd=REPO, env=env, check=False, text=True)


def _calls(log: Path) -> list[list[str]]:
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_train_launcher_passes_defaults_and_overrides(tmp_path):
    env, log = _stub_command(tmp_path)
    result = _run("train.sh", ["--task", "Dr02Training-AMP-Flat-v0", "--num-envs", "32"], env)
    assert result.returncode == 0
    args = _calls(log)[0]
    assert args[:4] == ["--num-envs", "4096", "--max-iterations", "10000"]
    assert args[-2:] == ["--num-envs", "32"]


def test_play_launcher_passes_task_and_checkpoint(tmp_path):
    env, log = _stub_command(tmp_path)
    result = _run(
        "play.sh",
        ["--task", "Dr02Training-AMP-Flat-v0", "--checkpoint", "/tmp/policy.pt"],
        env,
    )
    assert result.returncode == 0
    args = _calls(log)[0]
    assert args[:4] == ["--num-envs", "512", "--checkpoint", "latest"]
    assert args[-4:] == ["--task", "Dr02Training-AMP-Flat-v0", "--checkpoint", "/tmp/policy.pt"]


def test_train_smoke_runs_presets_in_order_with_unique_names(tmp_path):
    env, log = _stub_command(tmp_path)
    result = _run("train_smoke.sh", ["--task", "Dr02Training-AMP-Recovery-v0"], env)
    assert result.returncode == 0
    first, second = _calls(log)
    assert first[:4] == ["--num-envs", "1", "--max-iterations", "1"]
    assert second[:4] == ["--num-envs", "512", "--max-iterations", "100"]
    first_name = first[first.index("--run-name") + 1]
    second_name = second[second.index("--run-name") + 1]
    assert first_name.startswith("smoke-minimal-")
    assert second_name.startswith("smoke-representative-")
    assert first_name != second_name
    assert first[-2:] == second[-2:] == ["--task", "Dr02Training-AMP-Recovery-v0"]


def test_launchers_forward_child_exit_code_and_smoke_stops_on_failure(tmp_path):
    env, log = _stub_command(tmp_path, exit_code=37)
    assert _run("train.sh", ["--task", "Dr02Training-AMP-Flat-v0"], env).returncode == 37
    assert _run("play.sh", ["--task", "Dr02Training-AMP-Flat-v0"], env).returncode == 37
    log.unlink()
    assert _run("train_smoke.sh", ["--task", "Dr02Training-AMP-Flat-v0"], env).returncode == 37
    assert len(_calls(log)) == 1
