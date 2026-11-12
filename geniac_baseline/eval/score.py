"""Score CaP-X evaluation runs from their trial folders.

Success = ``taskcompleted_1`` and ``sandboxrc_0`` in the trial folder name (the rule of the baseline collection
and evaluation; CaP-X writes the name in ``capx/utils/launch_utils.py`` ``_save_trial_artifacts``).

Not valid for the competition gateway path: there ``task_completed`` never crosses the boundary and the folder
name always says ``taskcompleted_0`` (``capx/envs/trial.py`` / ``launch_utils.py``); the score of that
path is the gateway's ``gateway-results.json``.

Expected seeds come from each run's ``plan.json`` (written by ``eval plan``): a seed with no trial folder (a crash
before CaP-X saved anything), or with only per-turn saves (killed before the end-of-trial save), counts as a
failure and is reported as ``missing``. Re-run those seeds as a new
``eval plan`` with only them (a refill) and score both run directories together.

Per task: n (expected seeds, or found folders when there is no plan.json), missing, success,
first_try (success without any REGENERATE), mean model calls among successes,
finish_unsolved (FINISH while the task was not completed), no_finish (no FINISH: turn limit or crash),
code_no_keyword (a decision answer with a code block but no REGENERATE; CaP-X counts it as FINISH,
``parse_multi_turn_decision``, so it is a format failure, not a decision; geniac_prize_2026_code also counts
an answer whose REGENERATE does not start a line as FINISH, which this count does not include).
"""

from __future__ import annotations

import collections
import json
from pathlib import Path
from typing import Any

from geniac_baseline.data.trials import is_final_save, parse_trial_dir_name


def score_trial(d: Path) -> dict[str, Any]:
    name = parse_trial_dir_name(d.name)
    if name is None:
        raise ValueError(f"not a trial folder: {d}")
    f = d / "all_responses.json"
    # a retried trial may rewrite the file without truncating it: the first JSON value is the latest attempt
    calls = json.JSONDecoder().raw_decode(f.read_text().lstrip())[0] if f.exists() else []
    dec = [c.get("decision") for c in calls]
    # CaP-X stores a FINISH answer's raw text in "reasoning" when the server returned no separate reasoning
    # (capx/envs/trial.py, finish branch)
    fin_text = [c.get("reasoning") or "" for c in calls if c.get("decision") == "finish"]
    return {
        "seed": name.trial,
        "success": name.task_completed and name.sandbox_rc == 0,
        "calls": len(calls),
        "regenerates": dec.count("regenerate"),
        "finished": "finish" in dec,
        "code_no_keyword": sum(1 for t in fin_text if "```" in t and "REGENERATE" not in t),
    }


def find_trials(run_dir: Path) -> tuple[list[Path], set[tuple[str, int]]]:
    """Trial folders of a run written by ``eval plan`` (``seg_*/outputs/<model>/<task>/trial_*``) or a plain
    CaP-X output tree (``outputs/<model>/<task>/trial_*``), one per trial, and the (task, seed) of trials that
    never reached their end-of-trial save.

    CaP-X saves a trial after every code block, naming the folder by the result so far, so one seed can have
    several folders; the end-of-trial save is the trial (``is_final_save``, as in ``discover_trials``). A seed
    whose folders are all per-turn saves (the trial was killed) is returned as unfinished and scored as missing.
    """
    pats = ("outputs/*/*/trial_*", "seg_*/outputs/*/*/trial_*")
    by_seed: dict[tuple[Path, int], list[Path]] = collections.defaultdict(list)
    for p in sorted(p for pat in pats for p in run_dir.glob(pat) if p.is_dir()):
        name = parse_trial_dir_name(p.name)
        if name is not None:
            by_seed[(p.parent, name.trial)].append(p)
    trials: list[Path] = []
    unfinished: set[tuple[str, int]] = set()
    for (task_dir, seed), folders in sorted(by_seed.items()):
        if len(folders) == 1:
            trials.append(folders[0])
            continue
        final = [d for d in folders if is_final_save(d)]
        if len(final) > 1:
            raise ValueError(f"{task_dir}: seed {seed} has {len(final)} end-of-trial saves: {[d.name for d in final]}")
        if final:
            trials.append(final[0])
        else:
            unfinished.add((task_dir.name, seed))
    return trials, unfinished


def expected_seeds(run_dir: Path) -> dict[str, set[int]]:
    """Task -> seeds the run was planned for (empty if the run has no plan.json)."""
    from geniac_baseline.collect.preflight import parse_segment

    f = run_dir / "plan.json"
    if not f.is_file():
        return {}
    out: dict[str, set[int]] = collections.defaultdict(set)
    for t in json.loads(f.read_text()).get("tasks", []):
        for seg in t["seeds"]:
            a, b = parse_segment(seg)
            out[t["task"]].update(range(a, b + 1))
    return out


def score_runs(run_dirs: list[Path]) -> dict[str, Any]:
    per: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    expected: dict[str, set[int]] = collections.defaultdict(set)
    for r in run_dirs:
        for task, seeds in expected_seeds(r).items():
            expected[task] |= seeds
        trials, unfinished = find_trials(r)
        for td in trials:
            per[td.parent.name].append(score_trial(td))
        for task, seed in unfinished:
            expected[task].add(seed)
    rows: dict[str, Any] = {}
    for task in sorted(set(per) | set(expected)):
        ts = per.get(task, [])
        seeds = [t["seed"] for t in ts]
        if len(set(seeds)) != len(seeds):
            dup = sorted(s for s, c in collections.Counter(seeds).items() if c > 1)
            raise ValueError(f"{task}: duplicate seeds {dup}")
        missing = sorted(expected.get(task, set()) - set(seeds))
        ok = [t for t in ts if t["success"]]
        rows[task] = {
            "n": len(ts) + len(missing),
            "missing": len(missing),
            "missing_seeds": missing,
            "success": len(ok),
            "first_try": sum(1 for t in ok if t["regenerates"] == 0),
            "turns_success": round(sum(t["calls"] for t in ok) / len(ok), 2) if ok else None,
            "finish_unsolved": sum(1 for t in ts if t["finished"] and not t["success"]),
            "no_finish": sum(1 for t in ts if not t["finished"]) + len(missing),
            "code_no_keyword": sum(t["code_no_keyword"] for t in ts),
        }
    keys = ("n", "missing", "success", "first_try", "finish_unsolved", "no_finish", "code_no_keyword")
    tot: dict[str, Any] = {k: sum(r[k] for r in rows.values()) for k in keys}
    tot["mean_task_success"] = (
        round(sum(r["success"] / r["n"] for r in rows.values() if r["n"]) / len(rows), 4) if rows else None
    )
    return {"tasks": rows, "total": tot}
