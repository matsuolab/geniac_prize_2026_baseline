"""Success / failure trajectory pairs for DMPO.

A trajectory is all rows of one trial (one per model call), ordered by turn. Pairs are drawn within a task:

* LIBERO-PRO: first within the same layout (``(seed - 1) % n_layouts``), then the leftovers within the task.
* Robosuite: within the task.

Each trajectory is used at most once. Both trials of a pair are on the same side of the train / val split, using
the rules of ``data/split.py`` (val = LIBERO-PRO seeds 381-390, Robosuite 391-400; trajectories on LIBERO-PRO
evaluation layouts are dropped; a seed >= 401 is an error). A trajectory whose turns are not exactly ``0..T-1`` (a row
was dropped upstream) is dropped whole.

Inputs: the success rows (output of ``convert split``: ``sft.jsonl`` + ``val.jsonl``) and the failure rows (output of
``convert rows --include failure`` + ``convert tag``: ``rows.jsonl``); every ``rows.jsonl`` / ``sft.jsonl`` /
``val.jsonl`` present in each directory is read, and the split is re-derived per trajectory.
Output: ``pairs.jsonl``, ``rows.jsonl`` (the rows of every paired trajectory) and ``manifest.json``.

Ported from the pair builder of the baseline DMPO run; with the same inputs and ``seed`` it draws the same pairs.
"""

from __future__ import annotations

import collections
import json
import random
from pathlib import Path
from typing import Any

from geniac_baseline.data.split import DEFAULT_RULES, SplitRules, assign_split, is_libero, libero_layout
from geniac_baseline.data.tag import read_jsonl

ROW_FILES = ("rows.jsonl", "sft.jsonl", "val.jsonl")

Trajectories = dict[str, list[dict[str, Any]]]


class PairError(ValueError):
    """The inputs cannot be paired."""


def load_trajectories(d: Path) -> Trajectories:
    traj: Trajectories = collections.defaultdict(list)
    found = False
    for name in ROW_FILES:
        p = d / name
        if p.exists():
            found = True
            for r in read_jsonl(p):
                traj[r["metadata"]["sample_id"]].append(r)
    if not found:
        raise PairError(f"{d}: none of {ROW_FILES}")
    for v in traj.values():
        v.sort(key=lambda r: r["metadata"]["turn"])
    return dict(traj)


def _meta(tr: list[dict[str, Any]]) -> tuple[str, int, int | None]:
    m = tr[0]["metadata"]
    return m["task"], int(m["seed"]), m.get("n_layouts")


def build_pairs(
    success: Trajectories, failure: Trajectories, seed: int = 0, rules: SplitRules = DEFAULT_RULES
) -> tuple[list[dict[str, Any]], collections.Counter[Any]]:
    """Return the pairs (dicts as written to ``pairs.jsonl``) and counts."""
    rng = random.Random(seed)
    count: collections.Counter[Any] = collections.Counter()
    for sid, tr in failure.items():
        if tr[0]["metadata"]["success"]:
            raise PairError(f"{sid}: a failure trajectory is marked success")
    for sid, tr in success.items():
        if not tr[0]["metadata"]["success"]:
            raise PairError(f"{sid}: a success trajectory is marked failure")
    overlap = set(success) & set(failure)
    if overlap:
        raise PairError(f"trajectories on both sides: {sorted(overlap)[:3]}")

    split: dict[str, str] = {}

    def keep(sid: str, tr: list[dict[str, Any]]) -> bool:
        if [r["metadata"]["turn"] for r in tr] != list(range(len(tr))):
            count["dropped_incomplete_trajectory"] += 1
            return False
        task, s, n = _meta(tr)
        dest = assign_split(task, s, n, rules)  # raises on seed >= 401 or a missing layout count
        if dest == "drop_eval_layout":
            count["dropped_eval_layout"] += 1
            return False
        split[sid] = dest
        return True

    S = {k: v for k, v in success.items() if keep(k, v)}
    F = {k: v for k, v in failure.items() if keep(k, v)}
    by_task: dict[tuple[str, str], tuple[list[str], list[str]]] = collections.defaultdict(lambda: ([], []))
    for k, v in S.items():
        by_task[(_meta(v)[0], split[k])][0].append(k)
    for k, v in F.items():
        by_task[(_meta(v)[0], split[k])][1].append(k)

    def layout(k: str, D: Trajectories) -> int:
        _, s, n = _meta(D[k])
        return libero_layout(s, int(n))  # type: ignore[arg-type]

    raw: list[tuple[str, str, str, str]] = []
    for (task, _split), (s_ids, f_ids) in sorted(by_task.items()):
        s_ids, f_ids = sorted(s_ids), sorted(f_ids)
        rng.shuffle(s_ids)
        rng.shuffle(f_ids)
        used_s: set[str] = set()
        used_f: set[str] = set()
        if is_libero(task):
            fl: dict[int, list[str]] = collections.defaultdict(list)
            for k in f_ids:
                fl[layout(k, F)].append(k)
            for k in s_ids:
                cand = [x for x in fl[layout(k, S)] if x not in used_f]
                if cand:
                    raw.append((task, "layout", k, cand[0]))
                    used_s.add(k)
                    used_f.add(cand[0])
        rest_s = [k for k in s_ids if k not in used_s]
        rest_f = [k for k in f_ids if k not in used_f]
        for k, j in zip(rest_s, rest_f, strict=False):  # leftovers beyond the shorter list stay unpaired
            raw.append((task, "task", k, j))

    pairs = []
    for i, (task, match, s, j) in enumerate(raw):
        if split[s] != split[j]:
            raise PairError(f"cross-split pair {s} / {j}")
        pairs.append(
            {
                "pair_id": i,
                "task": task,
                "match": match,
                "split": split[s],
                "chosen": s,
                "rejected": j,
                "chosen_turns": len(S[s]),
                "rejected_turns": len(F[j]),
            }
        )
        count[(split[s], match)] += 1
    return pairs, count


def write_pairs(success_dir: Path, failure_dir: Path, out: Path, seed: int = 0) -> dict[str, Any]:
    """Build pairs and write ``pairs.jsonl``, ``rows.jsonl`` and ``manifest.json`` into a new ``out``."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    S, F = load_trajectories(success_dir), load_trajectories(failure_dir)
    pairs, count = build_pairs(S, F, seed)
    out.mkdir(parents=True)
    seen: set[str] = set()
    with (out / "pairs.jsonl").open("w") as fp, (out / "rows.jsonl").open("w") as fr:
        for p in pairs:
            fp.write(json.dumps(p) + "\n")
            for k, D in ((p["chosen"], S), (p["rejected"], F)):
                if k not in seen:
                    seen.add(k)
                    for r in D[k]:
                        fr.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = json.loads((success_dir / "manifest.json").read_text())
    man.update(
        {
            "stage": "pairs",
            "pairs_from": {"success": str(success_dir), "failure": str(failure_dir)},
            "pair_seed": seed,
            "pair_counts": {
                "|".join(map(str, k)) if isinstance(k, tuple) else k: v for k, v in sorted(count.items(), key=str)
            },
            "pairs_per_task": dict(collections.Counter(p["task"] for p in pairs)),
            "files": {"pairs": "pairs.jsonl", "rows": "rows.jsonl"},
        }
    )
    (out / "manifest.json").write_text(json.dumps(man, indent=1, ensure_ascii=False) + "\n")
    return man
