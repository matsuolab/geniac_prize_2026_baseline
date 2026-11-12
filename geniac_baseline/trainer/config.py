"""Training configuration (YAML -> dataclass), with no credentials anywhere.

Defaults are the settings of the baseline runs (team trainer ``heuristic_sft_train.py`` and our port of it);
see docs/train.md. Example: ``configs/train_sft.yaml``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# LoRA on the language model's attention and MLP projections only (team trainer; vision modules untouched).
TARGET_MODULES = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class ConfigError(ValueError):
    """The training config is invalid."""


@dataclass
class WandbConfig:
    """Optional experiment tracking. Off by default. Credentials are never read from or written to the config:
    wandb uses its own login / environment (``WANDB_API_KEY``) when enabled."""

    enabled: bool = False
    project: str | None = None
    name: str | None = None
    # "offline" writes runs locally only; set "online" deliberately. WANDB_MODE in the environment wins.
    mode: str = "offline"


@dataclass
class TrainConfig:
    """All trainer settings.

    ``data`` is a local directory with ``sft.jsonl`` / ``val.jsonl`` / ``manifest.json`` (output of
    ``geniac-baseline convert split``), or ``hf://<repo_id>[@<revision>][/<subdir>]`` for a Hugging Face dataset
    repo (resolved with ``huggingface_hub``; no repo name is built in).
    ``model`` is a local model snapshot directory or a Hugging Face model id; loading is offline by default.
    """

    data: str
    model: str
    output_dir: Path
    train_file: str = "sft.jsonl"
    val_file: str = "val.jsonl"
    epochs: float = 3.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_accum: int = 8
    grad_clip: float = 1.0
    warmup_steps: int = 3
    lora_rank: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.0
    target_modules: tuple[str, ...] = TARGET_MODULES
    max_steps: int = 0
    seed: int = 42
    save_every: int = 25
    val_every: int = 0
    val_max_rows: int = 0
    require_val: bool = True
    # rows longer than this (prompt + answer tokens) are dropped and listed; 0 = keep all. The baseline dropped one
    # row of 159,470 tokens that ran out of memory on one B300 (next longest: under 85k).
    max_row_tokens: int = 100000
    stop_after_min: float = 0.0
    resume: Path | None = None
    local_files_only: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ConfigError("epochs must be > 0")
        if self.lr <= 0:
            raise ConfigError("lr must be > 0")
        for name in ("grad_accum", "lora_rank", "lora_alpha", "warmup_steps"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")
        for name in ("max_steps", "save_every", "val_every", "val_max_rows", "max_row_tokens"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be >= 0")
        if self.save_every == 0:
            raise ConfigError("save_every must be >= 1 (checkpoints are what makes resume possible)")
        if not self.target_modules:
            raise ConfigError("target_modules is empty")
        if self.wandb.enabled and not self.wandb.project:
            raise ConfigError("wandb.enabled needs wandb.project")


@dataclass
class DMPOConfig:
    """Settings of the DMPO stage (``geniac-baseline train dmpo``); defaults = the baseline DMPO run.

    ``data`` is the output of ``geniac-baseline pairs`` (``pairs.jsonl``, ``rows.jsonl``, ``manifest.json``), local
    or ``hf://...``. ``init_adapter`` is the SFT adapter: the policy starts from it and the frozen reference is it.
    Provenance of the defaults: docs/train.md.
    """

    data: str
    model: str
    init_adapter: Path
    output_dir: Path
    beta: float = 0.5
    gamma: float = 0.9
    lr: float = 1e-5
    weight_decay: float = 0.0
    epochs: int = 3
    pairs_per_step: int = 32
    warmup_steps: int = 1
    grad_clip: float = 1.0
    max_steps: int = 0
    max_pairs: int = 0
    seed: int = 42
    stop_after_min: float = 0.0
    resume: Path | None = None
    local_files_only: bool = True
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def validate(self) -> None:
        if self.beta <= 0:
            raise ConfigError("beta must be > 0")
        if not 0 < self.gamma < 1:
            raise ConfigError("gamma must be in (0, 1)")
        if self.lr <= 0:
            raise ConfigError("lr must be > 0")
        for name in ("epochs", "pairs_per_step", "warmup_steps"):
            if getattr(self, name) < 1:
                raise ConfigError(f"{name} must be >= 1")
        for name in ("max_steps", "max_pairs"):
            if getattr(self, name) < 0:
                raise ConfigError(f"{name} must be >= 0")
        if self.wandb.enabled and not self.wandb.project:
            raise ConfigError("wandb.enabled needs wandb.project")


_PATH_FIELDS = {"output_dir", "resume", "init_adapter"}


def check_output_dir(out: Path, resume: Path | None) -> None:
    """A new run needs an empty (or missing) output_dir; only ``--resume`` continues in a used one."""
    if resume is None and out.is_dir() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty; choose a new output_dir or resume with --resume {out}/ckpt-last")


_FORBIDDEN_KEYS = {"api_key", "wandb_api_key", "hf_token", "token", "password", "secret"}


def _check_no_credentials(d: Any, where: str = "") -> None:
    if isinstance(d, dict):
        for k, v in d.items():
            if str(k).lower() in _FORBIDDEN_KEYS:
                raise ConfigError(f"credential-like key '{where}{k}' in the config; use the tool's own login instead")
            _check_no_credentials(v, f"{where}{k}.")


def config_from_dict(raw: dict[str, Any], base_dir: Path | None = None, cls: type = TrainConfig) -> Any:
    """Build a ``TrainConfig`` (or ``DMPOConfig``) from a mapping; relative paths are resolved against
    ``base_dir``."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping")
    _check_no_credentials(raw)
    known = {f.name for f in dataclasses.fields(cls)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    required = [
        f.name
        for f in dataclasses.fields(cls)
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING
    ]
    for req in required:
        if req not in raw:
            raise ConfigError(f"missing required key '{req}'")
    kw = dict(raw)
    wb = kw.pop("wandb", None) or {}
    wb_known = {f.name for f in dataclasses.fields(WandbConfig)}
    if set(wb) - wb_known:
        raise ConfigError(f"unknown wandb keys: {sorted(set(wb) - wb_known)}")

    def resolve(p: str) -> Path:
        q = Path(p).expanduser()
        return q if q.is_absolute() or base_dir is None else (base_dir / q)

    for name in _PATH_FIELDS:
        if kw.get(name) is not None:
            kw[name] = resolve(str(kw[name]))
    if not str(kw["data"]).startswith("hf://"):
        kw["data"] = str(resolve(str(kw["data"])))
    if "target_modules" in kw:
        kw["target_modules"] = tuple(kw["target_modules"])
    cfg = cls(**kw, wandb=WandbConfig(**wb))
    cfg.validate()
    return cfg


def load_config(path: Path, overrides: dict[str, Any] | None = None, cls: type = TrainConfig) -> Any:
    """Load a YAML config; ``overrides`` (e.g. from the command line) replace top-level keys."""
    raw = yaml.safe_load(path.read_text()) or {}
    if overrides:
        raw = {**raw, **overrides}
    return config_from_dict(raw, base_dir=path.parent, cls=cls)


def config_to_dict(cfg: Any) -> dict[str, Any]:
    """JSON-serialisable view (paths as strings), for ``train_config.json`` in the run directory."""
    d = dataclasses.asdict(cfg)
    for k, v in list(d.items()):
        if isinstance(v, Path):
            d[k] = str(v)
        elif isinstance(v, tuple):
            d[k] = list(v)
    return d
