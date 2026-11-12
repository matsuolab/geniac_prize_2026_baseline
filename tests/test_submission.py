"""submission/: the gateway task configs carry the baseline's APIs and prompts, and install.sh is wired to its files.

The image build itself (install.sh on the image's CaP-X) cannot run in CI; docs/tutorial/10_upload.md has it.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
SUB = REPO / "submission"
BASE = REPO / "configs" / "capx" / "geniac_prize"
# gateway config -> the baseline task config it is made from
SOURCES = {
    "franka_libero.yaml": "libero/libero_goal_drawer.yaml",
    "franka_robosuite.yaml": "robosuite/cube_lifting.yaml",
    "franka_nut_assembly.yaml": "robosuite/nut_assembly.yaml",
    "franka_spill_wipe.yaml": "robosuite/spill_wipe.yaml",
    "franka_two_arm_lift.yaml": "robosuite/two_arm_lift.yaml",
    "franka_two_arm_handover.yaml": "robosuite/two_arm_handover.yaml",
}
# fields of capx.submission.interfaces.TaskRunConfig (extra="forbid": any other key stops the runner)
TASK_RUN_FIELDS = {
    "apis",
    "prompt",
    "multi_turn_prompt",
    "api_servers",
    "local_models",
    "record_video",
    "step_chunk",
    "r1pro",
}


def test_every_gateway_config_has_a_source() -> None:
    assert sorted(p.name for p in (SUB / "gateway").glob("*.yaml")) == sorted(SOURCES)


@pytest.mark.parametrize("name", sorted(SOURCES))
def test_gateway_config_is_the_baseline_config_with_the_gateway_goal(name: str) -> None:
    got = yaml.safe_load((SUB / "gateway" / name).read_text())
    src = yaml.safe_load((BASE / SOURCES[name]).read_text())
    cfg = src["env"]["cfg"]
    assert set(got) <= TASK_RUN_FIELDS
    assert got["apis"] == cfg["apis"]
    assert got["prompt"] == re.sub(r"^Goal: .*$", "Goal: {task_instruction}", cfg["prompt"], count=1, flags=re.M)
    assert got["prompt"].count("{task_instruction}") == 1
    assert got["multi_turn_prompt"] == cfg["multi_turn_prompt"]
    assert got["api_servers"] == src["api_servers"]


def test_libero_and_cube_prompts_differ_only_in_the_goal() -> None:
    """One fallback config serves every LIBERO-PRO task, and one the cube tasks: their prompts must agree."""
    for pattern, ref in (("libero/*.yaml", "franka_libero.yaml"), ("robosuite/cube_*.yaml", "franka_robosuite.yaml")):
        want = yaml.safe_load((SUB / "gateway" / ref).read_text())
        for p in sorted(BASE.glob(pattern)):
            cfg = yaml.safe_load(p.read_text())["env"]["cfg"]
            prompt = re.sub(r"^Goal: .*$", "Goal: {task_instruction}", cfg["prompt"], count=1, flags=re.M)
            assert (cfg["apis"], prompt) == (want["apis"], want["prompt"]), p.name


def test_task_map_names_shipped_configs() -> None:
    entries = yaml.safe_load((SUB / "task_map.yaml").read_text())
    assert entries and all(set(e) == {"task_id", "config"} for e in entries)
    assert all((SUB / "gateway" / e["config"]).is_file() for e in entries)


def test_trial_state_patch_targets_the_gateway_env_only() -> None:
    text = (SUB / "trial-state.patch").read_text()
    assert re.findall(r"^\+\+\+ b/(\S+)$", text, re.M) == ["capx/submission/code_env.py"]


def test_install_script_parses_and_names_its_patches() -> None:
    script = SUB / "install.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert '"$here/skill-library.patch" "$here/trial-state.patch"' in text
    assert (REPO / "patches" / "geniac_prize" / "skill-library.patch").is_file()
