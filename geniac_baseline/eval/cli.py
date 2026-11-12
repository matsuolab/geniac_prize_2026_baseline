"""``geniac-baseline eval plan --spec SPEC.yaml`` and ``geniac-baseline eval score RUN_DIR... [--json OUT]``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geniac_baseline.collect.cli import plan_cmd
from geniac_baseline.eval.score import score_runs


def _score(a: argparse.Namespace) -> int:
    res = score_runs(a.runs)
    print(json.dumps(res["total"]))
    for task, r in res["tasks"].items():
        print(
            f"  {task:34s} {r['success']:3d}/{r['n']:<3d} first_try {r['first_try']:3d}  "
            f"turns {r['turns_success']}  finish_unsolved {r['finish_unsolved']:3d}  "
            f"no_finish {r['no_finish']:3d}  code_no_kw {r['code_no_keyword']}"
            + (f"  missing {r['missing']} {r['missing_seeds']}" if r["missing"] else "")
        )
    if a.json:
        a.json.write_text(json.dumps(res, indent=1) + "\n")
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("eval", help="write a CaP-X evaluation script (runs nothing), or score finished runs")
    s = p.add_subparsers(dest="eval_cmd", required=True)
    pl = s.add_parser("plan", help="preflight + write run.sh and per-segment configs")
    pl.add_argument("--spec", type=Path, required=True, help="run spec YAML (configs/eval_base.yaml)")
    pl.set_defaults(func=plan_cmd("eval"))
    sc = s.add_parser("score", help="success rate and multi-turn behaviour per task")
    sc.add_argument("runs", type=Path, nargs="+", help="run directories (all trials are pooled; duplicate seeds error)")
    sc.add_argument("--json", type=Path)
    sc.set_defaults(func=_score)
