"""SFT loss plug-in, trainer wiring and config loading. No torch needed (the trainers import it lazily)."""

from __future__ import annotations

import dataclasses
import inspect
import math
import sys
from pathlib import Path

import pytest

from geniac_baseline.trainer import loss as loss_mod
from geniac_baseline.trainer import sft_trainer as trainer
from geniac_baseline.trainer.config import ConfigError, config_from_dict, load_config
from geniac_baseline.trainer.data_source import parse_hf_location, resolve_data_dir
from geniac_baseline.trainer.loss import LossInputs, LossOutput, compute_loss

REPO = Path(__file__).resolve().parents[1]


def _dummy_inputs() -> LossInputs:
    return LossInputs(
        logits=None,
        target_ids=None,
        answer_mask=None,
        decision_prefix_len=0,
        input_ids=None,  # type: ignore[arg-type]
        n_prompt_tokens=0,
        metadata={"kind": "first_turn"},
        row_id="x-t0",
    )


def test_importing_train_does_not_import_torch() -> None:
    if "torch" in sys.modules:
        pytest.skip("torch already imported by another test")
    import geniac_baseline.trainer.dmpo_trainer
    import geniac_baseline.trainer.sft_trainer  # noqa: F401

    assert "torch" not in sys.modules


def test_compute_loss_is_the_completion_ce() -> None:
    import inspect as _inspect

    assert "completion_ce(inputs)" in _inspect.getsource(compute_loss)
    from geniac_baseline.trainer.losses import dmpo, sft_ce  # noqa: F401  (both import without torch)


def test_compute_loss_interface() -> None:
    sig = inspect.signature(compute_loss)
    assert list(sig.parameters) == ["inputs"]
    assert sig.parameters["inputs"].annotation in ("LossInputs", LossInputs)
    assert sig.return_annotation in ("LossOutput", LossOutput)
    assert [f.name for f in dataclasses.fields(LossInputs)] == [
        "logits",
        "target_ids",
        "answer_mask",
        "decision_prefix_len",
        "input_ids",
        "n_prompt_tokens",
        "metadata",
        "row_id",
    ]
    assert [f.name for f in dataclasses.fields(LossOutput)] == ["loss", "normalizer", "n_loss_tokens"]


def test_training_loop_calls_the_plug_in_point() -> None:
    assert "compute_row_loss(" in inspect.getsource(trainer.train)
    assert "loss_mod.compute_loss(" in inspect.getsource(trainer.compute_row_loss)


def test_trainer_checks_the_loss_output(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loss_mod, "compute_loss", lambda inputs: 0.0)
    with pytest.raises(TypeError, match="LossOutput"):
        trainer.compute_row_loss(_dummy_inputs())
    monkeypatch.setattr(loss_mod, "compute_loss", lambda inputs: LossOutput(loss=None, normalizer=3))
    with pytest.raises(ValueError, match="normalizer 0"):
        trainer.compute_row_loss(_dummy_inputs())
    monkeypatch.setattr(loss_mod, "compute_loss", lambda inputs: LossOutput(loss=None, normalizer=0))
    assert trainer.compute_row_loss(_dummy_inputs()).normalizer == 0


def test_schedule_matches_the_baseline() -> None:
    assert trainer.total_steps(3711, 3, 8, 0) == 1392  # the 3-epoch all-task run: 1392 steps
    assert trainer.total_steps(3711, 3, 8, 10) == 10
    assert trainer.lr_lambda(0, 3, 100) == pytest.approx(1 / 3)
    assert trainer.lr_lambda(2, 3, 100) == pytest.approx(0.5 * (1 + math.cos(math.pi * 2 / 100)))
    assert trainer.lr_lambda(100, 3, 100) == pytest.approx(0.0)


def test_example_config_loads() -> None:
    cfg = load_config(REPO / "configs" / "train_sft.yaml")
    assert cfg.lr == 1e-4 and cfg.grad_accum == 8 and cfg.lora_rank == 32 and cfg.lora_alpha == 32
    assert cfg.target_modules == ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
    assert cfg.wandb.enabled is False
    assert cfg.output_dir.is_absolute()


def test_config_errors(tmp_path: Path) -> None:
    base = {"data": "d", "model": "m", "output_dir": "o"}
    with pytest.raises(ConfigError, match="unknown"):
        config_from_dict({**base, "lerning_rate": 1})
    with pytest.raises(ConfigError, match="missing"):
        config_from_dict({"data": "d", "model": "m"})
    with pytest.raises(ConfigError, match="credential"):
        config_from_dict({**base, "wandb": {"enabled": True, "project": "p", "api_key": "x"}})
    with pytest.raises(ConfigError, match="credential"):
        config_from_dict({**base, "hf_token": "x"})
    with pytest.raises(ConfigError, match=r"wandb\.project"):
        config_from_dict({**base, "wandb": {"enabled": True}})
    with pytest.raises(ConfigError, match="save_every"):
        config_from_dict({**base, "save_every": 0})
    cfg = config_from_dict({**base, "data": "hf://org/name@abc123/sub"}, base_dir=tmp_path)
    assert cfg.data == "hf://org/name@abc123/sub"
    cfg = config_from_dict(base, base_dir=tmp_path)
    assert cfg.data == str(tmp_path / "d")


def test_data_locations(tmp_path: Path) -> None:
    loc = parse_hf_location("hf://org/name@0123abc/split_v1")
    assert (loc.repo_id, loc.revision, loc.subdir) == ("org/name", "0123abc", "split_v1")
    assert parse_hf_location("hf://org/name").revision is None
    with pytest.raises(ValueError):
        parse_hf_location("hf://just-a-name")
    with pytest.raises(FileNotFoundError, match="manifest"):
        resolve_data_dir(str(tmp_path))
    (tmp_path / "manifest.json").write_text("{}")
    assert resolve_data_dir(str(tmp_path)) == tmp_path


@pytest.mark.parametrize(("name", "cls"), [("train_sft.yaml", "TrainConfig"), ("train_dmpo.yaml", "DMPOConfig")])
def test_dataclass_defaults_equal_the_example_configs(name: str, cls: str) -> None:
    import yaml

    from geniac_baseline.trainer import config as C

    raw = yaml.safe_load((REPO / "configs" / name).read_text())
    for f in dataclasses.fields(getattr(C, cls)):
        if f.name in raw and f.default is not dataclasses.MISSING and f.name != "resume":
            got = raw[f.name]
            assert (tuple(got) if isinstance(got, list) else got) == f.default, f.name


def test_best_epoch_reads_the_whole_log(tmp_path: Path) -> None:
    import json

    log = tmp_path / "train_log.jsonl"
    recs = [
        {"step": 0, "val/loss_all": 0.17},
        {"step": 464, "epoch": 1, "val/loss_all": 0.1182},
        {"step": 928, "epoch": 2, "val/loss_all": 0.1151},
        {"step": 1392, "epoch": 3, "val/loss_all": 0.1162},
    ]
    log.write_text("".join(json.dumps(r) + "\n" for r in recs))
    assert trainer.best_epoch(log)["epoch"] == 2
    assert trainer.best_epoch(tmp_path / "none.jsonl") is None


def test_output_dir_guard(tmp_path: Path) -> None:
    from geniac_baseline.trainer.config import check_output_dir

    check_output_dir(tmp_path / "new", None)
    (tmp_path / "used").mkdir()
    (tmp_path / "used" / "train_log.jsonl").write_text("")
    with pytest.raises(FileExistsError, match="--resume"):
        check_output_dir(tmp_path / "used", None)
    check_output_dir(tmp_path / "used", tmp_path / "used" / "ckpt-last")


def test_reference_key_changes_with_adapter_and_data(tmp_path: Path) -> None:
    from geniac_baseline.trainer.dmpo_trainer import reference_key

    a, d = tmp_path / "adapter", tmp_path / "pairs"
    a.mkdir()
    d.mkdir()
    (a / "adapter_model.safetensors").write_bytes(b"w1")
    (d / "rows.jsonl").write_text("r")
    (d / "pairs.jsonl").write_text("p")
    k = reference_key(a, d)
    (a / "adapter_model.safetensors").write_bytes(b"w2")
    k2 = reference_key(a, d)
    (d / "pairs.jsonl").write_text("p2")
    assert len({k, k2, reference_key(a, d)}) == 3


def test_init_adapter_resolves_best_epoch(tmp_path: Path) -> None:
    import json

    from geniac_baseline.trainer.dmpo_trainer import resolve_init_adapter

    run = tmp_path / "train_sft"
    (run / "epoch-2" / "adapter").mkdir(parents=True)
    (run / "best_epoch.json").write_text(json.dumps({"epoch": 2}))
    assert resolve_init_adapter(run) == run / "epoch-2" / "adapter"
    (run / "adapter").mkdir()
    (run / "adapter" / "adapter_config.json").write_text("{}")
    assert resolve_init_adapter(run / "adapter") == run / "adapter"
    with pytest.raises(FileNotFoundError):
        resolve_init_adapter(tmp_path)
