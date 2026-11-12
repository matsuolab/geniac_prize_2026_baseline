"""Rows + the exact rebuild check, on the real cube_lifting trials and on synthetic ones."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from geniac_baseline.data import RebuildCheckError, discover_trials, load_sources, validate_rows, write_rows
from geniac_baseline.data.prompts import split_saved, yaml_template
from geniac_baseline.data.rows import check_rebuild_rule
from geniac_baseline.data.tag import read_jsonl

from .conftest import TEMPLATE


def _trials(root: Path, include=("success",)):
    trials, _ = discover_trials(load_sources(root / "sources.yaml"), include=include)
    return trials


def test_real_rows_match_the_original_conversion(real_fixture: Path, tmp_path: Path) -> None:
    """expected_rows.jsonl was produced by the script that built the baseline data (see fixtures/README.md)."""
    man = write_rows(_trials(real_fixture), tmp_path / "rows")
    got = sorted(read_jsonl(tmp_path / "rows" / "rows.jsonl"), key=lambda r: r["id"])
    want = sorted(read_jsonl(real_fixture / "expected_rows.jsonl"), key=lambda r: r["id"])
    assert got == want
    assert man["rule_check"] == {"saved_rebuild_exact": 4}
    assert man["counts"] == {"decision|row|rebuilt": 3, "decision|row|saved": 4, "first_turn|row|None": 3}
    assert validate_rows(got) == 10


def test_real_rebuilt_finish_rows_have_the_saved_prompt_shape(real_fixture: Path, tmp_path: Path) -> None:
    write_rows(_trials(real_fixture), tmp_path / "rows")
    rows = read_jsonl(tmp_path / "rows" / "rows.jsonl")
    saved = [r for r in rows if r["metadata"]["feedback"] == "saved"]
    rebuilt = [r for r in rows if r["metadata"]["feedback"] == "rebuilt"]
    assert {r["messages"][2]["content"][0]["text"] for r in rebuilt} == {"FINISH"}
    prefix = {split_saved(r["messages"][1]["content"][0]["text"])["prefix"] for r in saved + rebuilt}
    assert len(prefix) == 1  # every decision prompt of the task starts the same way


def test_template_is_read_verbatim(real_fixture: Path) -> None:
    tpl = yaml_template(real_fixture / "cube_lifting.yaml")
    assert tpl.startswith("The following code blocks were just executed:\n```python\n{executed_code}\n```")
    assert tpl.endswith("- The word 'FINISH' if the task appears to be complete")


def test_tampered_saved_prompt_refuses_and_writes_nothing(real_fixture: Path, tmp_path: Path) -> None:
    trial = next((real_fixture / "trials").rglob("trial_307_*"))
    p = trial / "all_responses.json"
    calls = json.loads(p.read_text())
    text = calls[1]["multi_turn_prompt"][-1]["content"][0]["text"]
    calls[1]["multi_turn_prompt"][-1]["content"][0]["text"] = text.replace(
        "If there were no errors", "If there were no  errors", 1
    )
    p.write_text(json.dumps(calls))
    out = tmp_path / "rows"
    with pytest.raises(RebuildCheckError, match="refusing to write rows"):
        write_rows(_trials(real_fixture), out)
    assert not out.exists()


def test_unparseable_saved_prompt_refuses(real_fixture: Path, tmp_path: Path) -> None:
    trial = next((real_fixture / "trials").rglob("trial_302_*"))
    p = trial / "all_responses.json"
    calls = json.loads(p.read_text())
    text = calls[1]["multi_turn_prompt"][-1]["content"][0]["text"]
    calls[1]["multi_turn_prompt"][-1]["content"][0]["text"] = text.replace(
        "The current console stdout is:", "stdout:", 1
    )
    p.write_text(json.dumps(calls))
    with pytest.raises(RebuildCheckError, match="markers"):
        write_rows(_trials(real_fixture), tmp_path / "rows")


def test_changed_template_refuses(real_fixture: Path, tmp_path: Path) -> None:
    cfg = real_fixture / "cube_lifting.yaml"
    cfg.write_text(
        cfg.read_text().replace(
            "If there were errors, consider regenerating the code.", "If there were errors, regenerate."
        )
    )
    with pytest.raises(RebuildCheckError, match="do not rebuild exactly"):
        write_rows(_trials(real_fixture), tmp_path / "rows")


def test_missing_vdm_file_refuses_with_the_reason(real_fixture: Path, tmp_path: Path) -> None:
    for f in (real_fixture / "trials").rglob("differencing_feedback_*.txt"):
        f.unlink()
    with pytest.raises(RebuildCheckError) as e:
        write_rows(_trials(real_fixture), tmp_path / "rows")
    assert "differencing_feedback" in str(e.value)
    assert "unpatched geniac_prize_2026_code" in str(e.value)
    assert not (tmp_path / "rows").exists()


def test_two_prefixes_for_one_task_refuse(synthetic_run: Path, tmp_path: Path) -> None:
    trial = next(synthetic_run.rglob("trial_302_*"))
    p = trial / "all_responses.json"
    calls = json.loads(p.read_text())
    msg = calls[1]["multi_turn_prompt"][-1]["content"][0]
    msg["text"] = "Another preamble.\n" + msg["text"]
    p.write_text(json.dumps(calls))
    with pytest.raises(RebuildCheckError, match="exactly one prompt prefix"):
        check_rebuild_rule(_trials(synthetic_run))


def test_no_saved_prompt_at_all_refuses(synthetic_run: Path) -> None:
    for p in synthetic_run.rglob("all_responses.json"):
        calls = json.loads(p.read_text())
        for c in calls:
            c.pop("multi_turn_prompt", None)
        p.write_text(json.dumps(calls))
    with pytest.raises(RebuildCheckError, match="no saved decision prompt"):
        check_rebuild_rule(_trials(synthetic_run))


def test_synthetic_rows(synthetic_run: Path, tmp_path: Path) -> None:
    write_rows(_trials(synthetic_run), tmp_path / "rows")
    rows = {r["id"]: r for r in read_jsonl(tmp_path / "rows" / "rows.jsonl")}
    assert sorted(rows) == [
        "cube_lifting__syn__trial301-t0",
        "cube_lifting__syn__trial301-t1",
        "cube_lifting__syn__trial301-t2",
        "cube_lifting__syn__trial302-t0",
        "cube_lifting__syn__trial302-t1",
        "cube_lifting__syn__trial302-t2",
        "cube_lifting__syn__trial302-t3",
    ]
    fin = rows["cube_lifting__syn__trial302-t3"]["messages"][1]["content"][0]["text"]
    # the FINISH prompt carries the last block's stdout and only the stderr the last block added
    expected_mid = TEMPLATE.format(executed_code="x = 1\nx = 2\nx = 3", console_stdout="ok", console_stderr="late")
    assert expected_mid in fin
    assert fin.endswith("vdm text after block 3\nBased on the code output, decide.\n")
    assert (
        rows["cube_lifting__syn__trial301-t1"]["messages"][2]["content"][0]["text"]
        == "REGENERATE\n```python\nprint('b')\n```"
    )


def test_refuses_to_overwrite(synthetic_run: Path, tmp_path: Path) -> None:
    (tmp_path / "rows").mkdir()
    with pytest.raises(FileExistsError):
        write_rows(_trials(synthetic_run), tmp_path / "rows")


def test_libero_rows_carry_the_layout_count(synthetic_run: Path, tmp_path: Path) -> None:
    src = synthetic_run / "sources.yaml"
    src.write_text(src.read_text().replace("task: cube_lifting", "task: libero_goal_drawer") + "    layouts: 50\n")
    write_rows(_trials(synthetic_run), tmp_path / "rows")
    rows = read_jsonl(tmp_path / "rows" / "rows.jsonl")
    assert {r["metadata"]["n_layouts"] for r in rows} == {50}
    validate_rows(rows)
