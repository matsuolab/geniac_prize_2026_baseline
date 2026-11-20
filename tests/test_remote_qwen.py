"""remote_qwen/serve.sh: how it starts vLLM, checked with stand-ins for vLLM and unshare."""

import os
import stat
import subprocess
from pathlib import Path

SERVE = Path(__file__).resolve().parents[1] / "remote_qwen" / "serve.sh"

# Prints what serve.sh gave the server: its arguments, the umask and the offline switch.
FAKE_PYTHON = """#!/bin/sh
echo "argv: $*"
echo "umask: $(umask)"
echo "offline: ${HF_HUB_OFFLINE:-unset}"
"""
# `unshare -r -n sh -c <script> [sh <command>...]`: the probe succeeds, a launch runs the command and says so.
FAKE_UNSHARE = """#!/bin/sh
shift 2
[ "$3" = "ip link set lo up" ] && exit 0
shift 4
echo "in namespace"
exec "$@"
"""


def _executable(path: Path, text: str) -> Path:
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def _serve(tmp_path: Path, model: str) -> str:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    _executable(bin_dir / "unshare", FAKE_UNSHARE)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "SERVE_PY": str(_executable(tmp_path / "python", FAKE_PYTHON)),
        "QWEN_SOCKET": str(tmp_path / "sock" / "qwen.sock"),
        "QWEN_MODEL_PATH": model,
    }
    done = subprocess.run(["bash", str(SERVE)], env=env, capture_output=True, text=True, check=True)
    assert stat.S_IMODE((tmp_path / "sock").stat().st_mode) == 0o700
    return done.stdout + done.stderr


def test_local_weights_start_in_a_network_namespace_with_an_owner_only_umask(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    out = _serve(tmp_path, str(model))
    assert "in namespace" in out
    assert f"--uds {tmp_path / 'sock' / 'qwen.sock'} --model {model}" in out
    assert "umask: 0077" in out
    assert "offline: 1" in out
    assert "WARNING" not in out


def test_hub_weights_keep_the_network_and_warn(tmp_path):
    out = _serve(tmp_path, "Qwen/Qwen3.8-27B")
    assert "in namespace" not in out
    assert "WARNING: no network namespace" in out
    assert "umask: 0077" in out
    assert "offline: unset" in out
