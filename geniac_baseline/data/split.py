"""Train / val split by seed, layout-aware for LIBERO-PRO, with the evaluation guards of the protocol.

Why LIBERO-PRO needs a layout-aware split
-----------------------------------------
CaP-X starts a LIBERO task from ``init_states[(seed - 1) % len(init_states)]``
(``capx/envs/simulators/libero.py``, ``FrankaLiberoEnv.reset``). In the competition CaP-X (geniac_prize_2026_code) that
state is kept (``set_init_state`` is not followed by another reset), so a task has only ``len(init_states)``
distinct start scenes and seeds ``s`` and ``s + len(init_states)`` are the *same* scene. A split by seed alone
would then put identical scenes into training and evaluation. The split is therefore by *layout* =
``(seed - 1) % n_layouts``, where ``n_layouts`` is the task's number of init states, recorded per task in the
sources file and carried in each row as ``metadata.n_layouts``. Whether seeds alias like this depends on the CaP-X
tree (docs/reproduce_baseline.md, "LIBERO の初期状態").

Rules (defaults of ``SplitRules``)
----------------------------------
LIBERO-PRO (task names starting with ``libero_``):

    evaluation layouts  the last ``libero_eval_layout_count`` (10) layout indices of the task: rows there are DROPPED
    val                 seeds 381-390 (unless they fall on an evaluation layout)
    train               every other seed

Robosuite (every other task): placements are drawn per seed, so val = seeds 391-400, train = the rest.

Any seed >= 401 is a hard error: 401 and above is the evaluation band.
"""

from __future__ import annotations

import collections
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from geniac_baseline.data.tag import read_jsonl

Split = Literal["train", "val", "drop_eval_layout"]


class EvalSeedError(ValueError):
    """A row comes from a seed in the evaluation band."""


class LayoutError(ValueError):
    """A LIBERO-PRO task without a usable layout count."""


@dataclass(frozen=True)
class SplitRules:
    """Seed rules of the protocol. Change only together with the evaluation protocol."""

    eval_seed_min: int = 401
    # collection seeds of the baseline: 301-400 (lower seeds are refused so a run cannot drift from the protocol)
    collect_seed_min: int = 301
    # LIBERO-PRO evaluation seeds stop here (five passes over 50 layouts: 401-650)
    libero_eval_seed_max: int = 650
    libero_eval_layout_count: int = 10
    libero_val_seeds: range = range(381, 391)
    robosuite_val_seeds: range = range(391, 401)


DEFAULT_RULES = SplitRules()


def is_libero(task: str) -> bool:
    return task.startswith("libero_")


def check_layouts(task: str, n_layouts: int | None, rules: SplitRules = DEFAULT_RULES) -> int:
    """The layout count of a LIBERO-PRO task, validated."""
    if not isinstance(n_layouts, int) or isinstance(n_layouts, bool) or n_layouts < 1:
        raise LayoutError(f"{task}: LIBERO-PRO tasks need their number of init states (layouts), got {n_layouts!r}")
    if n_layouts <= rules.libero_eval_layout_count:
        raise LayoutError(
            f"{task}: {n_layouts} layouts leave none for training after reserving "
            f"{rules.libero_eval_layout_count} for evaluation"
        )
    return n_layouts


def libero_layout(seed: int, n_layouts: int) -> int:
    """The init-state index CaP-X uses for a LIBERO seed: ``(seed - 1) % n_layouts``."""
    return (seed - 1) % n_layouts


def eval_layouts(n_layouts: int, rules: SplitRules = DEFAULT_RULES) -> range:
    """Layout indices reserved for evaluation: the last ``libero_eval_layout_count`` of the task."""
    return range(n_layouts - rules.libero_eval_layout_count, n_layouts)


def assign_split(task: str, seed: int, n_layouts: int | None = None, rules: SplitRules = DEFAULT_RULES) -> Split:
    """``train`` / ``val`` / ``drop_eval_layout`` for one (task, seed); raises on an evaluation seed."""
    if seed >= rules.eval_seed_min:
        raise EvalSeedError(f"{task} seed {seed} is in the evaluation band (>= {rules.eval_seed_min})")
    if is_libero(task):
        n = check_layouts(task, n_layouts, rules)
        if libero_layout(seed, n) in eval_layouts(n, rules):
            return "drop_eval_layout"
        return "val" if seed in rules.libero_val_seeds else "train"
    if n_layouts is not None:
        raise LayoutError(f"{task}: only LIBERO-PRO tasks have layouts")
    return "val" if seed in rules.robosuite_val_seeds else "train"


def split_rows(
    rows: list[dict[str, Any]], rules: SplitRules = DEFAULT_RULES
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], collections.Counter[tuple[str, str]]]:
    """Split rows into (train, val); rows on evaluation layouts are dropped and counted."""
    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    count: collections.Counter[tuple[str, str]] = collections.Counter()
    for r in rows:
        m = r["metadata"]
        task, seed = m["task"], int(m["seed"])
        suite = "libero" if is_libero(task) else "robosuite"
        try:
            dest = assign_split(task, seed, m.get("n_layouts"), rules)
        except (EvalSeedError, LayoutError) as e:
            raise type(e)(f"{r['id']}: {e}") from None
        count[(suite, dest)] += 1
        if dest == "train":
            train.append(r)
        elif dest == "val":
            val.append(r)
    return train, val, count


def write_split(src: Path, out: Path, rules: SplitRules = DEFAULT_RULES) -> dict[str, Any]:
    """Split ``<src>/rows.jsonl`` into a new directory with ``sft.jsonl`` (train), ``val.jsonl``, ``manifest.json``."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    rows = read_jsonl(src / "rows.jsonl")
    train, val, count = split_rows(rows, rules)
    out.mkdir(parents=True)
    for name, rs in (("sft.jsonl", train), ("val.jsonl", val)):
        with (out / name).open("w") as f:
            for r in rs:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = json.loads((src / "manifest.json").read_text())
    man.update(
        {
            "stage": "split",
            "split_from": str(src),
            "split_rules": {
                "eval_seed_min": rules.eval_seed_min,
                "libero_layout": "(seed - 1) % metadata.n_layouts",
                "libero_eval_layouts": f"last {rules.libero_eval_layout_count} layout indices, dropped",
                "libero_val_seeds": [rules.libero_val_seeds.start, rules.libero_val_seeds.stop - 1],
                "robosuite_val_seeds": [rules.robosuite_val_seeds.start, rules.robosuite_val_seeds.stop - 1],
            },
            "files": {"train": "sft.jsonl", "val": "val.jsonl"},
            "split_counts": {"|".join(k): v for k, v in sorted(count.items())},
            "tasks_train": dict(collections.Counter(r["metadata"]["task"] for r in train)),
            "tasks_val": dict(collections.Counter(r["metadata"]["task"] for r in val)),
        }
    )
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    return man
