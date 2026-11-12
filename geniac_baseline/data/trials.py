"""Find CaP-X trial folders and classify them as success / failure.

A CaP-X run writes trial folders named by ``capx/utils/launch_utils.py`` (``_save_trial_artifacts``)::

    trial_{trial:02d}_sandboxrc_{sandbox_rc}_reward_{reward:.3f}_taskcompleted_{int(task_completed)}

It saves after every code block with the result so far, so a trial whose result changed leaves several folders;
the one holding the end-of-trial save is the trial (``is_final_save``). It seeds trial ``t`` with
``env.reset(options={"trial": t}, seed=t)`` (``capx/envs/trial.py``), so the trial index in the folder name is
the seed.

Which trials are used is declared explicitly in a *sources file* (YAML), because the converter needs, for every
group of trials, the exact resolved config the run used (the multi-turn template is read from it)::

    sources:
      - run_tag: my_run_01                       # free label, becomes part of sample_id
        task: cube_lifting                       # task name (LIBERO-PRO tasks must start with "libero_")
        trials_dir: runs/my_run_01/outputs/Qwen_Qwen3.8-27B/cube_lifting   # folder holding trial_* folders
        config: runs/my_run_01/cube_lifting.yaml                            # the config that run was launched with
      - run_tag: my_run_01
        task: libero_spatial_on_stove
        trials_dir: ...
        config: ...
        layouts: 50          # LIBERO-PRO only, required: the task's number of init states (see data/split.py)

Relative paths are resolved against the folder of the sources file.
"""

from __future__ import annotations

import collections
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml

from geniac_baseline.data.prompts import summary_feedback
from geniac_baseline.data.split import LayoutError, check_layouts, is_libero

Outcome = Literal["success", "failure"]

# Folder name written by capx/utils/launch_utils.py::_save_trial_artifacts. sandbox_rc can be -1 (the initial
# value in capx/envs/trial.py when no block ran), hence the optional minus (as in our eval scorer).
TRIAL_DIR_RE = re.compile(
    r"^trial_(?P<trial>\d+)_sandboxrc_(?P<rc>-?\d+)_reward_(?P<reward>[0-9.]+)_taskcompleted_(?P<done>[01])$"
)


class SourcesError(ValueError):
    """The sources file is malformed or names trials that cannot be used together."""


@dataclass(frozen=True)
class Source:
    """One group of trial folders that share a task and a resolved config."""

    run_tag: str
    task: str
    trials_dir: Path
    config: Path
    layouts: int | None = None


@dataclass(frozen=True)
class TrialRef:
    """One trial folder, classified."""

    trial_dir: Path
    task: str
    seed: int
    run_tag: str
    config: Path
    outcome: Outcome
    sandbox_rc: int
    reward: float
    n_layouts: int | None = None

    @property
    def sample_id(self) -> str:
        """``<task>__<run_tag>__trial<seed:03d>``, the id format of the collection store (collect_sft.py)."""
        return f"{self.task}__{self.run_tag}__trial{self.seed:03d}"


@dataclass(frozen=True)
class TrialName:
    trial: int
    sandbox_rc: int
    reward: float
    task_completed: bool


def parse_trial_dir_name(name: str) -> TrialName | None:
    """Parse a CaP-X trial folder name; ``None`` if the name is not a trial folder."""
    m = TRIAL_DIR_RE.match(name)
    if not m:
        return None
    return TrialName(int(m["trial"]), int(m["rc"]), float(m["reward"]), m["done"] == "1")


def classify(trial_dir: Path, name: TrialName) -> Outcome | None:
    """Success / failure rule of the baseline data.

    success: ``taskcompleted_1`` and ``sandboxrc_0`` and a non-empty ``code.py`` (the store filter of the
             collection pipeline; the eval scorer uses the same first two conditions).
    failure: ``taskcompleted_0`` and ``all_responses.json`` exists (the trial got as far as a model answer).
    Anything else (``taskcompleted_1`` with a non-zero sandbox rc, or a folder without responses) is neither and
    is not used.
    """
    if name.task_completed:
        if name.sandbox_rc != 0:
            return None
        code = trial_dir / "code.py"
        if not code.is_file() or not code.read_text(errors="replace").strip():
            return None
        return "success"
    if not (trial_dir / "all_responses.json").is_file():
        return None
    return "failure"


def is_final_save(trial_dir: Path) -> bool:
    """Whether a trial folder holds the end-of-trial save rather than a per-turn one.

    CaP-X saves a trial after every code block, naming the folder by the result so far
    (``capx/envs/trial.py``), so a trial whose result changed leaves one folder per result. The per-turn saves
    write only the program to ``summary.txt``; the end-of-trial save writes its Stdout / Stderr / Reward block.
    """
    summary = trial_dir / "summary.txt"
    return summary.is_file() and summary_feedback(summary.read_text(errors="replace")) is not None


def load_sources(path: Path) -> list[Source]:
    """Read a sources file (format in the module docstring)."""
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict) or not isinstance(raw.get("sources"), list) or not raw["sources"]:
        raise SourcesError(f"{path}: expected a mapping with a non-empty 'sources' list")
    base = path.parent
    out: list[Source] = []
    for i, entry in enumerate(raw["sources"]):
        if not isinstance(entry, dict):
            raise SourcesError(f"{path}: sources[{i}] is not a mapping")
        missing = {"run_tag", "task", "trials_dir", "config"} - set(entry)
        if missing:
            raise SourcesError(f"{path}: sources[{i}] lacks {sorted(missing)}")
        extra = set(entry) - {"run_tag", "task", "trials_dir", "config", "layouts"}
        if extra:
            raise SourcesError(f"{path}: sources[{i}] has unknown keys {sorted(extra)}")
        src = Source(
            run_tag=str(entry["run_tag"]),
            task=str(entry["task"]),
            trials_dir=(base / str(entry["trials_dir"])).resolve(),
            config=(base / str(entry["config"])).resolve(),
            layouts=entry.get("layouts"),
        )
        if is_libero(src.task):
            try:
                check_layouts(src.task, src.layouts)
            except LayoutError as e:
                raise SourcesError(f"{path}: sources[{i}]: {e}") from None
        elif src.layouts is not None:
            raise SourcesError(f"{path}: sources[{i}]: 'layouts' is only for LIBERO-PRO tasks")
        if "__" in src.run_tag or "__" in src.task:
            raise SourcesError(f"{path}: sources[{i}]: '__' is the sample_id separator and cannot appear in names")
        if not src.trials_dir.is_dir():
            raise SourcesError(f"{path}: sources[{i}]: trials_dir {src.trials_dir} is not a directory")
        if not src.config.is_file():
            raise SourcesError(f"{path}: sources[{i}]: config {src.config} is not a file")
        out.append(src)
    return out


def discover_trials(
    sources: list[Source], include: tuple[Outcome, ...] = ("success",)
) -> tuple[list[TrialRef], collections.Counter[str]]:
    """List the trials of every source whose outcome is in ``include``.

    Returns the trials (sorted by task, seed) and counts of what was skipped. Several folders of one seed in one
    ``trials_dir`` are one trial saved per turn: the end-of-trial save is used (``is_final_save``). Two trials with
    the same (task, seed) from different sources are a hard error: they would be the same initial state twice,
    and silently keeping one would make the split depend on folder order.
    """
    found: list[TrialRef] = []
    skipped: collections.Counter[str] = collections.Counter()
    for src in sources:
        by_seed: dict[int, list[tuple[Path, TrialName]]] = collections.defaultdict(list)
        for d in sorted(p for p in src.trials_dir.iterdir() if p.is_dir()):
            name = parse_trial_dir_name(d.name)
            if name is not None:
                by_seed[name.trial].append((d, name))
        for seed, folders in sorted(by_seed.items()):
            # Resolved before the outcome filter: a trial that succeeded part-way and failed at the end leaves a
            # success folder too, and only the end-of-trial save says how it ended.
            if len(folders) > 1:
                final = [(d, n) for d, n in folders if is_final_save(d)]
                if len(final) != 1:
                    raise SourcesError(
                        f"{src.trials_dir}: seed {seed} has {len(folders)} trial folders and {len(final)} of them "
                        "hold an end-of-trial summary.txt; expected exactly one"
                    )
                skipped["intermediate_save"] += len(folders) - 1
                folders = final
            d, name = folders[0]
            outcome = classify(d, name)
            if outcome is None:
                skipped["neither_success_nor_failure"] += 1
                continue
            if outcome not in include:
                skipped[f"not_included_{outcome}"] += 1
                continue
            found.append(
                TrialRef(
                    d, src.task, name.trial, src.run_tag, src.config, outcome, name.sandbox_rc, name.reward, src.layouts
                )
            )
    seen: dict[tuple[str, int], TrialRef] = {}
    for t in found:
        key = (t.task, t.seed)
        if key in seen:
            raise SourcesError(
                f"duplicate (task, seed) {key}: {seen[key].trial_dir} and {t.trial_dir}; "
                "remove one of them from the sources before converting"
            )
        seen[key] = t
    found.sort(key=lambda t: (t.task, t.seed))
    return found, skipped
