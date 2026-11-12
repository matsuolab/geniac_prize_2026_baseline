"""DMPO loss math, the pair builder and the DMPO config. CPU only, no torch."""

from __future__ import annotations

import copy
import math
from pathlib import Path

import pytest

from geniac_baseline.data.pairs import PairError, build_pairs, load_trajectories, write_pairs
from geniac_baseline.data.rows import write_rows
from geniac_baseline.data.split import EvalSeedError, write_split
from geniac_baseline.data.tag import read_jsonl
from geniac_baseline.data.trials import discover_trials, load_sources
from geniac_baseline.trainer.config import ConfigError, DMPOConfig, config_from_dict, load_config
from geniac_baseline.trainer.dmpo_trainer import PairDataError, load_pairs, steps_per_epoch
from geniac_baseline.trainer.losses.dmpo import margin, mean_loss, neg_log_sigmoid, phi, weighted_reward

from .conftest import FIXTURES, build_pair_fixture

REPO = Path(__file__).resolve().parents[1]


def test_phi_values() -> None:
    assert [round(phi(k, 3, 0.9), 3) for k in range(3)] == [1.0, 0.631, 0.299]
    assert phi(0, 1, 0.9) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        phi(3, 3, 0.9)
    with pytest.raises(ValueError):
        phi(0, 3, 1.0)


def test_margin_sign_and_loss() -> None:
    ref_w, ref_l = [-10.0, -20.0], [-5.0, -6.0, -7.0]
    up = margin([-9.0, -19.0], ref_w, [-5.0, -6.0, -7.0], ref_l, beta=0.5, gamma=0.9)
    assert up > 0  # the chosen side became more likely than under the reference
    down = margin([-10.0, -20.0], ref_w, [-4.0, -6.0, -7.0], ref_l, beta=0.5, gamma=0.9)
    assert down < 0  # the rejected side became more likely
    assert margin(ref_w, ref_w, ref_l, ref_l, 0.5, 0.9) == 0.0
    assert mean_loss([0.0]) == pytest.approx(math.log(2))  # policy == reference at step 0
    assert weighted_reward([1.0, 1.0, 1.0], [0.0, 0.0, 0.0], 0.9) == pytest.approx(1.0 + 0.631 + 0.299, abs=1e-3)
    assert neg_log_sigmoid(-1000.0) == pytest.approx(1000.0) and neg_log_sigmoid(1000.0) == 0.0


def test_last_step_of_an_epoch_can_hold_one_pair() -> None:
    assert steps_per_epoch(521, 32) == 17  # the evaluated baseline run: 16 full steps and one step with 9 pairs
    assert 521 - 16 * 32 == 9
    assert steps_per_epoch(521, 32) * 3 == 51


def _rows_and_fail(tmp: Path) -> tuple[Path, Path]:
    sources = load_sources(build_pair_fixture(tmp / "fx"))
    every, _ = discover_trials(sources, include=("success", "failure"))
    succ, _ = discover_trials(sources)
    fail, _ = discover_trials(sources, include=("failure",))
    write_rows(succ, tmp / "rows", check_trials=every)
    write_rows(fail, tmp / "fail", check_trials=every)
    write_split(tmp / "rows", tmp / "split")
    return tmp / "split", tmp / "fail"


def test_pairs_match_the_original_builder(tmp_path: Path) -> None:
    """expected_pairs.jsonl was written by the pair builder of the baseline DMPO run on the same inputs."""
    split, fail = _rows_and_fail(tmp_path)
    man = write_pairs(split, fail, tmp_path / "pairs", seed=0)
    assert read_jsonl(tmp_path / "pairs" / "pairs.jsonl") == read_jsonl(FIXTURES / "expected_pairs.jsonl")
    assert man["pair_counts"] == {"dropped_eval_layout": 1, "train|layout": 1, "train|task": 3, "val|task": 1}
    rows = read_jsonl(tmp_path / "pairs" / "rows.jsonl")
    traj = load_pairs(rows, read_jsonl(tmp_path / "pairs" / "pairs.jsonl"))
    assert len(traj) == 10


def test_failure_rows_use_the_rule_learnt_from_both_sets(tmp_path: Path) -> None:
    _split, fail = _rows_and_fail(tmp_path)
    rows = read_jsonl(fail / "rows.jsonl")
    assert rows and not any(r["metadata"]["success"] for r in rows)
    assert {r["metadata"]["seed"] for r in rows} == {352, 360, 395}


def test_pair_split_rules(tmp_path: Path) -> None:
    split, fail = _rows_and_fail(tmp_path)
    S, F = load_trajectories(split), load_trajectories(fail)
    pairs, _ = build_pairs(S, F)
    for p in pairs:  # no cross-split pair: both seeds fall on the pair's side
        seeds = [int(p[k].rsplit("trial", 1)[1]) for k in ("chosen", "rejected")]
        assert all((391 <= x <= 400) == (p["split"] == "val") for x in seeds)
    bad = copy.deepcopy(F)
    for r in bad["cube_lifting__fixture__trial352"]:
        r["metadata"]["seed"] = 401
    with pytest.raises(EvalSeedError):
        build_pairs(S, bad)
    bad = copy.deepcopy(F)
    bad["cube_lifting__fixture__trial352"][0]["metadata"]["success"] = True
    with pytest.raises(PairError, match="marked success"):
        build_pairs(S, bad)
    bad = copy.deepcopy(F)
    del bad["cube_lifting__fixture__trial360"][1]  # a dropped row: the trajectory is dropped whole
    pairs2, count = build_pairs(S, bad)
    assert count["dropped_incomplete_trajectory"] == 1
    assert all(p["rejected"] != "cube_lifting__fixture__trial360" for p in pairs2)
    bad = copy.deepcopy(F)
    for r in bad["libero_fixture__fixture__trial352"]:
        r["metadata"]["seed"] = 341  # layout 40: evaluation-only
    _, count = build_pairs(S, bad)
    assert count["dropped_eval_layout"] == 2


def test_load_pairs_checks_sides(tmp_path: Path) -> None:
    split, fail = _rows_and_fail(tmp_path)
    write_pairs(split, fail, tmp_path / "pairs")
    rows = read_jsonl(tmp_path / "pairs" / "rows.jsonl")
    pairs = read_jsonl(tmp_path / "pairs" / "pairs.jsonl")
    swapped = [{**pairs[0], "chosen": pairs[0]["rejected"], "rejected": pairs[0]["chosen"]}]
    with pytest.raises(PairDataError, match="chosen is a failure"):
        load_pairs(rows, swapped)


def test_dmpo_config() -> None:
    cfg = load_config(REPO / "configs" / "train_dmpo.yaml", cls=DMPOConfig)
    assert (cfg.beta, cfg.gamma, cfg.lr, cfg.pairs_per_step, cfg.epochs) == (0.5, 0.9, 1e-5, 32, 3)
    assert cfg.init_adapter.is_absolute()
    base = {"data": "d", "model": "m", "init_adapter": "a", "output_dir": "o"}
    with pytest.raises(ConfigError, match="gamma"):
        config_from_dict({**base, "gamma": 1.0}, cls=DMPOConfig)
    with pytest.raises(ConfigError, match="missing"):
        config_from_dict({"data": "d", "model": "m", "output_dir": "o"}, cls=DMPOConfig)
