"""patches/geniac_prize/: the patched geniac_prize_2026_code writes VDM text files that `convert rows` accepts, and
carries the same LIBERO skills as the ASPIRE tree the baseline data came from.

The patches cannot be applied in CI (no geniac_prize_2026_code clone), so the tests run the patch's own writer lines
on a copy of the real cube_lifting trials and compare the skill patch's added lines with the ASPIRE patch's.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

from geniac_baseline.collect.preflight import _URL_ENV, LIBERO_ADDED_SKILLS
from geniac_baseline.data import discover_trials, load_sources, write_rows
from geniac_baseline.data.tag import read_jsonl

PATCHES = Path(__file__).parent.parent / "patches"
PATCH = PATCHES / "geniac_prize" / "geniac-prize.patch"
SKILL_PATCH = PATCHES / "geniac_prize" / "skill-library.patch"
ASPIRE_PATCH = PATCHES / "aspire" / "aspire-f4c8939-local-changes.patch"
SKILL_LIB = "integrations/franka/libero_reduced_skill_library.py"


def _file_hunks(path: str, patch: Path = PATCH) -> str:
    """Added and context lines of one file in the patch."""
    text = patch.read_text()
    start = text.index(f"+++ b/{path}\n")
    end = text.find("\ndiff --git ", start)
    return text[start : end if end >= 0 else len(text)]


def _writer_snippet() -> str:
    """The block the patch adds to _save_trial_artifacts that writes differencing_feedback_XX.txt."""
    lines = _file_hunks("capx/utils/launch_utils.py").splitlines()
    added = [ln[1:] for ln in lines if ln.startswith("+") and not ln.startswith("+++")]
    i = next(k for k, ln in enumerate(added) if "if differencing_feedback_history:" in ln)
    block = [added[i]]
    for ln in added[i + 1 :]:
        if ln.strip() and not ln.startswith(" " * 8):
            break
        block.append(ln)
    return textwrap.dedent("\n".join(block))


def test_patched_writer_output_converts_like_the_baseline_data(real_fixture: Path, tmp_path: Path) -> None:
    root = tmp_path / "fixture"
    shutil.copytree(real_fixture, root)
    snippet = _writer_snippet()
    n = 0
    for trial_dir in sorted((root / "trials").rglob("trial_*")):
        files = sorted(trial_dir.glob("differencing_feedback_*.txt"))
        history = [f.read_text() for f in files]
        for f in files:
            f.unlink()
        exec(snippet, {"trial_dir": trial_dir, "differencing_feedback_history": history})
        assert sorted(p.name for p in trial_dir.glob("differencing_feedback_*.txt")) == [f.name for f in files]
        n += len(files)
    assert n > 0
    trials, _ = discover_trials(load_sources(root / "sources.yaml"), include=("success",))
    write_rows(trials, tmp_path / "rows")
    got = sorted(read_jsonl(tmp_path / "rows" / "rows.jsonl"), key=lambda r: r["id"])
    want = sorted(read_jsonl(real_fixture / "expected_rows.jsonl"), key=lambda r: r["id"])
    assert got == want


def test_patch_url_env_names_match_the_preflight_check() -> None:
    for rel, env, port in _URL_ENV.values():
        if rel == "vision/graspnet.py":  # already reads GRASPNET_SERVICE_URL; not in the patch
            continue
        hunk = _file_hunks(f"capx/integrations/{rel}")
        assert re.search(rf'^\+\w+ = os\.environ\.get\("{env}", "http://127\.0\.0\.1:{port}"\)$', hunk, re.M), rel


def test_patch_targets_capx_files_only() -> None:
    paths = re.findall(r"^\+\+\+ b/(\S+)$", PATCH.read_text(), re.M)
    assert paths and all(p.startswith("capx/") for p in paths)
    assert not any("skill_library" in p for p in paths)


def _added(path: str, patch: Path) -> list[str]:
    return [ln[1:] for ln in _file_hunks(path, patch).splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def test_skill_patch_targets_the_libero_skill_library_only() -> None:
    assert re.findall(r"^\+\+\+ b/(\S+)$", SKILL_PATCH.read_text(), re.M) == [f"capx/{SKILL_LIB}"]


def test_skill_patch_adds_what_the_aspire_patch_added() -> None:
    """The base files differ only in ASPIRE's license header, so the same added lines give the baseline's skills."""
    added = _added(f"capx/{SKILL_LIB}", SKILL_PATCH)
    assert added == _added(f"aspire/sim/cap/{SKILL_LIB}", ASPIRE_PATCH)
    registered = re.findall(r'^ {8}fns\["(\w+)"\] = self\.\1$', "\n".join(added), re.M)
    assert tuple(registered) == LIBERO_ADDED_SKILLS


def test_setup_script_applies_both_patches() -> None:
    text = (PATCH.parent / "setup.sh").read_text()
    assert '"$here/geniac-prize.patch" "$here/skill-library.patch"' in text


def test_setup_script_parses() -> None:
    subprocess.run(["bash", "-n", str(PATCH.parent / "setup.sh")], check=True)
