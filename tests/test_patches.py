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

from geniac_baseline.collect.preflight import LIBERO_ADDED_SKILLS
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


def test_patch_leaves_the_server_addresses_alone() -> None:
    """The clients' server addresses are geniac_prize_2026_code's own socket names; no hunk moves them."""
    paths = re.findall(r"^\+\+\+ b/(\S+)$", PATCH.read_text(), re.M)
    assert not any(p.startswith(("capx/integrations/vision/", "capx/integrations/motion/")) for p in paths)
    assert "SERVICE_URL" not in PATCH.read_text() and "127.0.0.1" not in PATCH.read_text()


def test_patch_targets_capx_files_only() -> None:
    paths = re.findall(r"^\+\+\+ b/(\S+)$", PATCH.read_text(), re.M)
    assert paths and all(p.startswith("capx/") for p in paths)
    assert not any("skill_library" in p for p in paths)


def _added(path: str, patch: Path) -> list[str]:
    return [ln[1:] for ln in _file_hunks(path, patch).splitlines() if ln.startswith("+") and not ln.startswith("+++")]


def test_skill_patch_targets_the_libero_skill_library_only() -> None:
    assert re.findall(r"^\+\+\+ b/(\S+)$", SKILL_PATCH.read_text(), re.M) == [f"capx/{SKILL_LIB}"]


SOCKET_IMPORT = "from capx.utils import unix_http_client"


def _as_aspire(added: list[str]) -> list[str]:
    """The skill patch's added lines without its one change: the model call goes over the Unix socket."""
    lines = [ln for ln in added if ln != SOCKET_IMPORT and "which plain requests cannot send to" not in ln]
    return [ln.replace("unix_http_client.post(", "requests.post(") for ln in lines]


def test_skill_patch_adds_what_the_aspire_patch_added() -> None:
    """The base files differ only in ASPIRE's license header, so the same added lines give the baseline's skills.

    The one difference is the transport of point_at_object's model call (test below)."""
    added = _added(f"capx/{SKILL_LIB}", SKILL_PATCH)
    assert _as_aspire(added) == _added(f"aspire/sim/cap/{SKILL_LIB}", ASPIRE_PATCH)
    registered = re.findall(r'^ {8}fns\["(\w+)"\] = self\.\1$', "\n".join(added), re.M)
    assert tuple(registered) == LIBERO_ADDED_SKILLS


def test_setup_script_applies_both_patches() -> None:
    text = (PATCH.parent / "setup.sh").read_text()
    assert '"$here/geniac-prize.patch" "$here/skill-library.patch"' in text


def test_setup_script_parses() -> None:
    subprocess.run(["bash", "-n", str(PATCH.parent / "setup.sh")], check=True)


def test_setup_applies_every_patch_in_the_directory() -> None:
    setup = (PATCHES / "geniac_prize" / "setup.sh").read_text()
    names = sorted(p.name for p in (PATCHES / "geniac_prize").glob("*.patch"))
    assert names == ["geniac-prize.patch", "skill-library.patch"]
    assert all(f'"$here/{n}"' in setup for n in names)


def test_skill_patch_sends_its_model_call_over_the_socket() -> None:
    """No added line posts with plain requests: CAPX_VLM_URL names a Unix socket, which requests cannot reach."""
    added = _added(f"capx/{SKILL_LIB}", SKILL_PATCH)
    aspire = _added(f"aspire/sim/cap/{SKILL_LIB}", ASPIRE_PATCH)
    assert SOCKET_IMPORT in added
    assert not any("requests.post(" in ln.replace("unix_http_client.post(", "") for ln in added)
    assert sum("unix_http_client.post(" in ln for ln in added) == sum("requests.post(" in ln for ln in aspire) > 0
