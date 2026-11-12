"""``geniac-baseline collect plan --spec SPEC.yaml`` (and the shared ``plan`` handler used by ``eval plan``)."""

from __future__ import annotations

import argparse
from pathlib import Path

from geniac_baseline.collect.plan import load_spec, write_plan


def plan_cmd(expected_mode: str):
    def run(a: argparse.Namespace) -> int:
        spec = load_spec(a.spec)
        if spec.mode != expected_mode:
            raise SystemExit(f"{a.spec}: mode is {spec.mode!r}, expected {expected_mode!r}")
        script = write_plan(spec)
        print(f"[plan] wrote {script}")
        if spec.mode == "collect":
            print(f"[plan] after the run: geniac-baseline convert rows --sources {script.parent / 'sources.yaml'} ...")
        print(f"[plan] run it on a GPU node: bash {script}")
        return 0

    return run


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("collect", help="write a CaP-X data-collection script (runs nothing)")
    s = p.add_subparsers(dest="collect_cmd", required=True)
    pl = s.add_parser("plan", help="preflight + write run.sh, per-segment configs and sources.yaml")
    pl.add_argument("--spec", type=Path, required=True, help="run spec YAML (configs/collect.yaml)")
    pl.set_defaults(func=plan_cmd("collect"))
