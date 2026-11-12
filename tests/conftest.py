"""Shared fixtures: the real cube_lifting trials and a builder for small synthetic CaP-X trial folders."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REAL = FIXTURES / "cube_lifting_real"

TEMPLATE = (
    "The following code blocks were just executed:\n```python\n{executed_code}\n```\n"
    "The current console stdout is:\n```\n{console_stdout}\n```\n"
    "The current console stderr is:\n```\n{console_stderr}\n```\n"
    "Please respond with EXACTLY ONE of the following:\n- REGENERATE ...\n- FINISH ..."
)
VDM_HEADER = "Included below is the observed differences between the current state and the previous state:"
TAIL = "\nBased on the code output, decide.\n"
SYSTEM = "You are a helpful assistant."
PREFIX = "You are controlling a robot. Goal: lift the cube.\n"


def config_text(template: str = TEMPLATE) -> str:
    body = "\n".join(("      " + line) if line else "" for line in template.split("\n"))
    return (
        "env:\n  cfg:\n    prompt: |\n      You are controlling a robot.\n\n    multi_turn_prompt: |\n"
        + body
        + "\n\nuse_img_differencing: true\ntrials: 10\nnum_workers: 2\nsave_multiturn_prompts: true\n"
    )


def _decision_user(executed: str, stdout: str, stderr: str, vdm: str) -> str:
    return (
        PREFIX
        + TEMPLATE.format(executed_code=executed, console_stdout=stdout, console_stderr=stderr)
        + "\n\n"
        + (VDM_HEADER + "\n" + vdm + TAIL)
    )


def make_trial(
    root: Path,
    seed: int,
    turns: list[dict[str, Any]],
    *,
    task_completed: bool = True,
    sandbox_rc: int = 0,
    final_stdout: str = "done",
) -> Path:
    """Write a synthetic trial folder.

    ``turns``: first item ``{"code": str}`` (the initial program), then one item per decision call:
    ``{"decision": "regenerate", "code": str, "stdout": str, "stderr": str}`` (stdout/stderr of the code that ran
    just before this call) or ``{"decision": "finish", "stdout": str, "stderr": str}``.
    """
    reward = 1.0 if task_completed else 0.0
    d = root / f"trial_{seed:02d}_sandboxrc_{sandbox_rc}_reward_{reward:.3f}_taskcompleted_{int(task_completed)}"
    d.mkdir(parents=True)
    calls: list[dict[str, Any]] = [
        {
            "initial_prompt": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": PREFIX}]},
            ],
            "code_blocks": [turns[0]["code"]],
            "decision": "initial",
        }
    ]
    blocks = [turns[0]["code"]]
    stderr_hist: list[str] = []
    last_stdout, last_stderr = "", ""
    for k, t in enumerate(turns[1:], start=1):
        vdm = f"vdm text after block {k}"
        (d / f"differencing_feedback_{k - 1:02d}.txt").write_text(vdm)
        if t["stderr"]:
            stderr_hist.append(t["stderr"])
        last_stdout, last_stderr = t["stdout"], t["stderr"]
        if t["decision"] == "regenerate":
            user = _decision_user("\n".join(blocks), t["stdout"], t["stderr"], vdm)
            calls.append(
                {
                    "multi_turn_prompt": [
                        {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                        {"role": "user", "content": [{"type": "text", "text": user}]},
                    ],
                    "block_idx": [k],
                    "code_blocks": [t["code"]],
                    "decision": "regenerate",
                    "reasoning": "",
                }
            )
            blocks.append(t["code"])
        else:
            calls.append({"decision": "finish", "reasoning": "FINISH"})
    (d / "all_responses.json").write_text(json.dumps(calls))
    joined = "\n\n".join(stderr_hist) if stderr_hist else last_stderr
    (d / "summary.txt").write_text(
        f"Trial summary\n  Stdout: {last_stdout or final_stdout}\n  Stderr: {joined}\n  Reward: 1.0\n"
    )
    (d / "code.py").write_text("\n".join(blocks) + "\n")
    return d


@pytest.fixture
def real_fixture(tmp_path: Path) -> Path:
    """A writable copy of the real cube_lifting fixture (sources.yaml at its root)."""
    dst = tmp_path / "real"
    shutil.copytree(REAL, dst)
    return dst


@pytest.fixture
def synthetic_run(tmp_path: Path) -> Path:
    """Two successful synthetic trials of task 'cube_lifting' (seeds 301, 302) plus a sources.yaml."""
    root = tmp_path / "run"
    trials = root / "outputs" / "model" / "cube_lifting"
    (root).mkdir(parents=True)
    (root / "cube_lifting.yaml").write_text(config_text())
    make_trial(
        trials,
        301,
        [
            {"code": "print('a')"},
            {
                "decision": "regenerate",
                "code": "print('b')",
                "stdout": "a",
                "stderr": "Traceback (most recent call last):\n  boom",
            },
            {"decision": "finish", "stdout": "b", "stderr": ""},
        ],
    )
    make_trial(
        trials,
        302,
        [
            {"code": "x = 1"},
            {"decision": "regenerate", "code": "x = 2", "stdout": "", "stderr": ""},
            {"decision": "regenerate", "code": "x = 3", "stdout": "", "stderr": "warn"},
            {"decision": "finish", "stdout": "ok", "stderr": "late"},
        ],
    )
    (root / "sources.yaml").write_text(
        "sources:\n  - run_tag: syn\n    task: cube_lifting\n    trials_dir: outputs/model/cube_lifting\n"
        "    config: cube_lifting.yaml\n"
    )
    return root


# (new seed, copied from seed, outcome) for the pair fixture: failures are copies of real successful trials renamed
# to taskcompleted_0, so the exact-rebuild check still holds for them.
PAIR_TRIALS = ((392, 302, "success"), (352, 306, "failure"), (360, 307, "failure"), (395, 302, "failure"))


def build_pair_fixture(root: Path) -> Path:
    """Real cube_lifting trials plus synthetic extra trials, listed under a Robosuite task (cube_lifting) and a
    LIBERO-PRO task name (libero_fixture, 50 layouts). Returns the sources file."""
    shutil.copytree(REAL, root)
    tdir = root / "trials" / "Qwen_Qwen3.8-27B" / "cube_lifting"
    for new, old, outcome in PAIR_TRIALS:
        src = next(tdir.glob(f"trial_{old}_*"))
        done, reward = ("1", "1.000") if outcome == "success" else ("0", "0.000")
        shutil.copytree(src, tdir / f"trial_{new}_sandboxrc_0_reward_{reward}_taskcompleted_{done}")
    s = root / "sources.yaml"
    s.write_text(
        "sources:\n"
        "  - {run_tag: fixture, task: cube_lifting, trials_dir: trials/Qwen_Qwen3.8-27B/cube_lifting, "
        "config: cube_lifting.yaml}\n"
        "  - {run_tag: fixture, task: libero_fixture, trials_dir: trials/Qwen_Qwen3.8-27B/cube_lifting, "
        "config: cube_lifting.yaml, layouts: 50}\n"
    )
    return s
