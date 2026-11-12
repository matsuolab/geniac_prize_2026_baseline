"""Trial folder names, the success/failure rule, and the sources file."""

from __future__ import annotations

from pathlib import Path

import pytest

from geniac_baseline.data.trials import (
    SourcesError,
    classify,
    discover_trials,
    is_final_save,
    load_sources,
    parse_trial_dir_name,
)

from .conftest import config_text, make_trial


def test_parse_trial_dir_name() -> None:
    n = parse_trial_dir_name("trial_302_sandboxrc_0_reward_1.000_taskcompleted_1")
    assert n is not None and (n.trial, n.sandbox_rc, n.reward, n.task_completed) == (302, 0, 1.0, True)
    n = parse_trial_dir_name("trial_07_sandboxrc_-1_reward_0.000_taskcompleted_0")
    assert n is not None and (n.trial, n.sandbox_rc, n.task_completed) == (7, -1, False)
    assert parse_trial_dir_name("trial_07_sandboxrc_0_reward_1.000_taskcompleted_1.partial") is None
    assert parse_trial_dir_name("aaa_done_flag") is None


def _dir(tmp: Path, name: str, files: dict[str, str]) -> Path:
    d = tmp / name
    d.mkdir()
    for k, v in files.items():
        (d / k).write_text(v)
    return d


def test_classify(tmp_path: Path) -> None:
    ok = "trial_1_sandboxrc_0_reward_1.000_taskcompleted_1"
    d = _dir(tmp_path, ok, {"code.py": "x=1", "all_responses.json": "[]"})
    assert classify(d, parse_trial_dir_name(ok)) == "success"
    name = "trial_2_sandboxrc_0_reward_1.000_taskcompleted_1"
    d = _dir(tmp_path, name, {"code.py": "  \n"})  # empty program: not a sample
    assert classify(d, parse_trial_dir_name(name)) is None
    name = "trial_3_sandboxrc_1_reward_1.000_taskcompleted_1"  # completed but the sandbox failed: neither
    d = _dir(tmp_path, name, {"code.py": "x=1", "all_responses.json": "[]"})
    assert classify(d, parse_trial_dir_name(name)) is None
    name = "trial_4_sandboxrc_1_reward_0.000_taskcompleted_0"
    d = _dir(tmp_path, name, {"all_responses.json": "[]"})
    assert classify(d, parse_trial_dir_name(name)) == "failure"
    name = "trial_5_sandboxrc_1_reward_0.000_taskcompleted_0"  # died before any model answer
    d = _dir(tmp_path, name, {})
    assert classify(d, parse_trial_dir_name(name)) is None


def test_sources_validation(tmp_path: Path) -> None:
    (tmp_path / "t").mkdir()
    (tmp_path / "c.yaml").write_text(config_text())
    bad = tmp_path / "s.yaml"
    bad.write_text("sources:\n  - {run_tag: r, task: cube_lifting, trials_dir: t}\n")
    with pytest.raises(SourcesError, match="lacks"):
        load_sources(bad)
    bad.write_text("sources:\n  - {run_tag: r, task: cube_lifting, trials_dir: t, config: c.yaml, extra: 1}\n")
    with pytest.raises(SourcesError, match="unknown keys"):
        load_sources(bad)
    bad.write_text("sources:\n  - {run_tag: a__b, task: cube_lifting, trials_dir: t, config: c.yaml}\n")
    with pytest.raises(SourcesError, match="separator"):
        load_sources(bad)
    bad.write_text("sources:\n  - {run_tag: r, task: cube_lifting, trials_dir: missing, config: c.yaml}\n")
    with pytest.raises(SourcesError, match="not a directory"):
        load_sources(bad)


def test_duplicate_task_seed_is_an_error(tmp_path: Path) -> None:
    for run in ("a", "b"):
        make_trial(tmp_path / run, 301, [{"code": "x=1"}, {"decision": "finish", "stdout": "", "stderr": ""}])
    (tmp_path / "c.yaml").write_text(config_text())
    s = tmp_path / "s.yaml"
    s.write_text(
        "sources:\n"
        "  - {run_tag: a, task: cube_lifting, trials_dir: a, config: c.yaml}\n"
        "  - {run_tag: b, task: cube_lifting, trials_dir: b, config: c.yaml}\n"
    )
    with pytest.raises(SourcesError, match="duplicate"):
        discover_trials(load_sources(s))


def _per_turn_save(root: Path, seed: int, *, task_completed: bool) -> Path:
    """A folder as a per-turn save leaves it: the result so far in the name, only the program in summary.txt."""
    done, reward = int(task_completed), 1.0 if task_completed else 0.0
    d = root / f"trial_{seed}_sandboxrc_0_reward_{reward:.3f}_taskcompleted_{done}"
    d.mkdir(parents=True)
    (d / "summary.txt").write_text("-" * 100 + "\nGenerated program:\nx=1\n")
    (d / "code.py").write_text("x=1\n")
    (d / "all_responses.json").write_text("[]")
    return d


def _one_source(tmp_path: Path) -> Path:
    (tmp_path / "c.yaml").write_text(config_text())
    s = tmp_path / "s.yaml"
    s.write_text("sources:\n  - {run_tag: r, task: cube_lifting, trials_dir: t, config: c.yaml}\n")
    return s


def test_per_turn_saves_give_way_to_the_end_of_trial_save(tmp_path: Path) -> None:
    turns = [{"code": "x=1"}, {"decision": "finish", "stdout": "", "stderr": ""}]
    # failed while running, succeeded at the end
    _per_turn_save(tmp_path / "t", 303, task_completed=False)
    final = make_trial(tmp_path / "t", 303, turns)
    assert is_final_save(final) and not is_final_save(next((tmp_path / "t").glob("*_taskcompleted_0")))
    # succeeded while running, failed at the end: the success folder must not be used
    _per_turn_save(tmp_path / "t", 304, task_completed=True)
    make_trial(tmp_path / "t", 304, turns, task_completed=False, sandbox_rc=1)
    s = _one_source(tmp_path)
    found, skipped = discover_trials(load_sources(s), include=("success", "failure"))
    assert [(t.seed, t.outcome, t.trial_dir.name) for t in found] == [
        (303, "success", final.name),
        (304, "failure", "trial_304_sandboxrc_1_reward_0.000_taskcompleted_0"),
    ]
    assert skipped["intermediate_save"] == 2
    found, _ = discover_trials(load_sources(s))
    assert [t.seed for t in found] == [303]


def test_a_seed_without_one_end_of_trial_save_is_an_error(tmp_path: Path) -> None:
    _per_turn_save(tmp_path / "t", 305, task_completed=False)
    _per_turn_save(tmp_path / "t", 305, task_completed=True)
    with pytest.raises(SourcesError, match="0 of them"):
        discover_trials(load_sources(_one_source(tmp_path)))


def test_include_filters_outcomes(tmp_path: Path) -> None:
    make_trial(tmp_path / "t", 301, [{"code": "x=1"}, {"decision": "finish", "stdout": "", "stderr": ""}])
    make_trial(
        tmp_path / "t",
        302,
        [{"code": "x=1"}, {"decision": "finish", "stdout": "", "stderr": ""}],
        task_completed=False,
        sandbox_rc=1,
    )
    (tmp_path / "c.yaml").write_text(config_text())
    s = tmp_path / "s.yaml"
    s.write_text("sources:\n  - {run_tag: r, task: cube_lifting, trials_dir: t, config: c.yaml}\n")
    found, skipped = discover_trials(load_sources(s))
    assert [t.seed for t in found] == [301] and skipped == {"not_included_failure": 1}
    found, _ = discover_trials(load_sources(s), include=("success", "failure"))
    assert [(t.seed, t.outcome) for t in found] == [(301, "success"), (302, "failure")]
    assert found[0].sample_id == "cube_lifting__r__trial301"


def test_sources_layouts(tmp_path: Path) -> None:
    (tmp_path / "t").mkdir()
    (tmp_path / "c.yaml").write_text(config_text())
    s = tmp_path / "s.yaml"
    s.write_text("sources:\n  - {run_tag: r, task: libero_goal_drawer, trials_dir: t, config: c.yaml}\n")
    with pytest.raises(SourcesError, match="init states"):
        load_sources(s)
    s.write_text("sources:\n  - {run_tag: r, task: cube_lifting, trials_dir: t, config: c.yaml, layouts: 50}\n")
    with pytest.raises(SourcesError, match="only for LIBERO-PRO"):
        load_sources(s)
    s.write_text("sources:\n  - {run_tag: r, task: libero_goal_drawer, trials_dir: t, config: c.yaml, layouts: 50}\n")
    assert load_sources(s)[0].layouts == 50
