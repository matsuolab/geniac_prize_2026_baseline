"""``geniac-baseline train sft|dmpo --config CONFIG.yaml [--dry-run] [--resume RUN/ckpt-last]``."""

from __future__ import annotations

import argparse
from pathlib import Path

from geniac_baseline.trainer.config import DMPOConfig, TrainConfig, load_config


def _overrides(a: argparse.Namespace) -> dict[str, str]:
    return {"resume": str(a.resume.resolve())} if a.resume else {}


def _sft(a: argparse.Namespace) -> int:
    cfg = load_config(a.config, overrides=_overrides(a), cls=TrainConfig)
    from geniac_baseline.trainer.sft_trainer import train

    train(cfg, dry_run=a.dry_run)
    return 0


def _dmpo(a: argparse.Namespace) -> int:
    cfg = load_config(a.config, overrides=_overrides(a), cls=DMPOConfig)
    from geniac_baseline.trainer.dmpo_trainer import train_dmpo

    train_dmpo(cfg, dry_run=a.dry_run)
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("train", help="LoRA training (GPU): SFT warm start, then DMPO")
    s = p.add_subparsers(dest="train_cmd", required=True)
    for name, func, cfg_help in (
        ("sft", _sft, "SFT config (configs/train_sft.yaml)"),
        ("dmpo", _dmpo, "DMPO config (configs/train_dmpo.yaml)"),
    ):
        q = s.add_parser(name, help=f"{name.upper()} stage")
        q.add_argument("--config", type=Path, required=True, help=cfg_help)
        q.add_argument("--resume", type=Path, help="checkpoint directory to resume from (RUN/ckpt-last)")
        q.add_argument("--dry-run", action="store_true", help="load the tokenizer, encode and report; no model, no GPU")
        q.set_defaults(func=func)
