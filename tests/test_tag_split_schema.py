"""code_outcome tags, the split rules (layout and seed guards), and the row schema."""

from __future__ import annotations

import collections
import copy
from pathlib import Path

import pytest

from geniac_baseline.data import (
    EvalSeedError,
    LayoutError,
    SchemaError,
    assign_split,
    discover_trials,
    load_sources,
    split_rows,
    tag_rows,
    validate_row,
    validate_rows,
    write_rows,
)
from geniac_baseline.data.split import write_split
from geniac_baseline.data.tag import TagError, read_jsonl, write_tagged


def _rows(root: Path, tmp: Path) -> list[dict]:
    trials, _ = discover_trials(load_sources(root / "sources.yaml"))
    write_rows(trials, tmp / "rows")
    return read_jsonl(tmp / "rows" / "rows.jsonl")


def test_code_outcomes(synthetic_run: Path, tmp_path: Path) -> None:
    counts: collections.Counter = collections.Counter()
    tagged = {
        r["id"]: r["metadata"]["code_outcome"] for r in tag_rows(_rows(synthetic_run, tmp_path), set(), False, counts)
    }
    assert tagged == {
        "cube_lifting__syn__trial301-t0": "replaced_error",  # the next prompt's stderr has a traceback
        "cube_lifting__syn__trial301-t1": "final",
        "cube_lifting__syn__trial301-t2": None,
        "cube_lifting__syn__trial302-t0": "replaced_clean",
        "cube_lifting__syn__trial302-t1": "replaced_clean",  # "warn" is stderr but not a traceback
        "cube_lifting__syn__trial302-t2": "final",
        "cube_lifting__syn__trial302-t3": None,
    }


def test_special_token_trials_are_dropped_whole(synthetic_run: Path, tmp_path: Path) -> None:
    rows = _rows(synthetic_run, tmp_path)
    for r in rows:
        if r["id"] == "cube_lifting__syn__trial302-t0":
            r["messages"][2]["content"][0]["text"] = "<tool_call>x = 1"
    counts: collections.Counter = collections.Counter()
    kept = tag_rows(rows, {"<tool_call>"}, False, counts)
    assert {r["metadata"]["sample_id"] for r in kept} == {"cube_lifting__syn__trial301"}
    assert counts["dropped_trials_special_token"] == 1 and counts["dropped_rows_special_token"] == 4
    assert len(tag_rows(rows, {"<tool_call>"}, True, collections.Counter())) == 7


def test_non_contiguous_turns_are_an_error(synthetic_run: Path, tmp_path: Path) -> None:
    rows = [r for r in _rows(synthetic_run, tmp_path) if r["id"] != "cube_lifting__syn__trial302-t1"]
    with pytest.raises(TagError, match="contiguous"):
        tag_rows(rows, set(), False, collections.Counter())


def test_real_tags(real_fixture: Path, tmp_path: Path) -> None:
    rows = _rows(real_fixture, tmp_path)
    write_tagged(tmp_path / "rows", tmp_path / "tagged", specials={"<tool_call>"})
    tagged = read_jsonl(tmp_path / "tagged" / "rows.jsonl")
    assert len(tagged) == len(rows) == 10
    validate_rows(tagged)
    finals = [r for r in tagged if r["metadata"]["code_outcome"] == "final"]
    assert len(finals) == 3  # each trial's last program before FINISH


@pytest.mark.parametrize(
    ("task", "seed", "split"),
    [
        ("libero_spatial_on_stove", 301, "train"),  # layout 0
        ("libero_spatial_on_stove", 340, "train"),  # layout 39
        ("libero_spatial_on_stove", 341, "drop_eval_layout"),  # layout 40
        ("libero_spatial_on_stove", 350, "drop_eval_layout"),  # layout 49
        ("libero_spatial_on_stove", 351, "train"),
        ("libero_spatial_on_stove", 380, "train"),
        ("libero_spatial_on_stove", 381, "val"),  # layout 30
        ("libero_spatial_on_stove", 390, "val"),
        ("libero_spatial_on_stove", 391, "drop_eval_layout"),  # layout 40, even though it is in the Robosuite val band
        ("libero_spatial_on_stove", 300, "drop_eval_layout"),  # layout 49
        ("cube_lifting", 390, "train"),
        ("cube_lifting", 391, "val"),
        ("cube_lifting", 400, "val"),
        ("cube_lifting", 341, "train"),  # Robosuite has no layout rule
    ],
)
def test_assign_split(task: str, seed: int, split: str) -> None:
    assert assign_split(task, seed, 50 if task.startswith("libero_") else None) == split


def test_layout_count_comes_from_the_task() -> None:
    # 30 init states: layouts 20-29 are evaluation-only, and seed s and s + 30 are the same scene
    assert assign_split("libero_goal_drawer", 21, 30) == "drop_eval_layout"  # layout 20
    assert assign_split("libero_goal_drawer", 51, 30) == "drop_eval_layout"  # layout 20 again
    assert assign_split("libero_goal_drawer", 341, 30) == "train"  # layout 0; with 50 layouts it would be 40
    with pytest.raises(LayoutError, match="init states"):
        assign_split("libero_goal_drawer", 301)
    with pytest.raises(LayoutError, match="leave none"):
        assign_split("libero_goal_drawer", 301, 10)
    with pytest.raises(LayoutError, match="only LIBERO-PRO"):
        assign_split("cube_lifting", 301, 50)


@pytest.mark.parametrize(("task", "layouts"), [("cube_lifting", None), ("libero_object_bbq_sauce", 50)])
@pytest.mark.parametrize("seed", [401, 441, 650])
def test_eval_band_is_a_hard_error(task: str, layouts: int | None, seed: int) -> None:
    with pytest.raises(EvalSeedError):
        assign_split(task, seed, layouts)


def test_split_rows_and_guard(synthetic_run: Path, tmp_path: Path) -> None:
    rows = _rows(synthetic_run, tmp_path)
    train, val, count = split_rows(rows)
    assert len(train) == 7 and not val and count == {("robosuite", "train"): 7}
    bad = copy.deepcopy(rows[0])
    bad["metadata"]["seed"] = 401
    with pytest.raises(EvalSeedError, match="trial301-t0"):
        split_rows([*rows, bad])


def test_write_split(synthetic_run: Path, tmp_path: Path) -> None:
    _rows(synthetic_run, tmp_path)
    man = write_split(tmp_path / "rows", tmp_path / "split")
    assert (tmp_path / "split" / "sft.jsonl").is_file() and (tmp_path / "split" / "val.jsonl").is_file()
    assert man["chat_template_kwargs"] == {"enable_thinking": False}
    with pytest.raises(FileExistsError):
        write_split(tmp_path / "rows", tmp_path / "split")


def test_schema_rejects(real_fixture: Path, tmp_path: Path) -> None:
    rows = _rows(real_fixture, tmp_path)
    good = rows[1]
    validate_row(good)

    def broken(fn):
        r = copy.deepcopy(good)
        fn(r)
        return r

    cases = [
        lambda r: r.__setitem__("images", ["a.png"]),
        lambda r: r["messages"][0].__setitem__("role", "user"),
        lambda r: r["messages"][1]["content"].append({"type": "image"}),
        lambda r: r["metadata"].__setitem__("seed", True),
        lambda r: r["metadata"].__setitem__("turn", 5),  # id no longer matches
        lambda r: r["metadata"].__setitem__("kind", "other"),
        lambda r: r["messages"][2]["content"][0].__setitem__("text", "REGENERATE no code"),
        lambda r: r["metadata"].__setitem__("feedback", "rebuilt"),
    ]
    for c in cases:
        with pytest.raises(SchemaError):
            validate_row(broken(c))
    with pytest.raises(SchemaError, match="duplicate"):
        validate_rows([good, good])
