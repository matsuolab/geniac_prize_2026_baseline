"""``geniac-baseline convert {rows,tag,split,validate}``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from geniac_baseline.data.rows import RebuildCheckError, check_rebuild_rule, write_rows
from geniac_baseline.data.schema import validate_rows
from geniac_baseline.data.split import write_split
from geniac_baseline.data.tag import read_jsonl, tokenizer_special_tokens, write_tagged
from geniac_baseline.data.trials import discover_trials, load_sources


def _rows(a: argparse.Namespace) -> int:
    include = ("success", "failure") if a.include == "all" else (a.include,)
    sources = load_sources(a.sources)
    trials, skipped = discover_trials(sources, include=include)
    # the rebuild rule is learnt from and checked on every success and failure trial of the sources
    every, _ = discover_trials(sources, include=("success", "failure"))
    print(f"[rows] {len(trials)} trials ({dict(skipped) or 'nothing skipped'}); rule checked on {len(every)}")
    try:
        if a.check_only:
            rule = check_rebuild_rule(every)
            print(f"[check] OK {rule.counts}")
            return 0
        if a.out is None:
            raise SystemExit("--out is required unless --check-only")
        man = write_rows(trials, a.out, sources_file=a.sources, check_trials=every)
    except RebuildCheckError as e:
        print(f"[check] {e}\n[check] report: {json.dumps(e.report)}")
        return 1
    print(f"[check] OK {man['rule_check']}")
    for k, v in man["counts"].items():
        print(f"[rows] {k}: {v}")
    return 0


def _tag(a: argparse.Namespace) -> int:
    specials = tokenizer_special_tokens(a.tokenizer)
    man = write_tagged(a.src, a.out, specials, keep_special=a.keep_special_token_trials)
    print(json.dumps(man["tag_counts"], indent=1))
    return 0


def _split(a: argparse.Namespace) -> int:
    man = write_split(a.src, a.out)
    print(json.dumps(man["split_counts"], indent=1))
    return 0


def _validate(a: argparse.Namespace) -> int:
    for p in a.files:
        print(f"{p}: {validate_rows(read_jsonl(p))} rows OK")
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("convert", help="CaP-X trial folders -> training rows")
    s = p.add_subparsers(dest="convert_cmd", required=True)

    r = s.add_parser("rows", help="trials -> rows.jsonl (refuses unless every saved prompt rebuilds exactly)")
    r.add_argument("--sources", type=Path, required=True, help="sources YAML (see geniac_baseline.data.trials)")
    r.add_argument("--out", type=Path, help="new output directory (required unless --check-only)")
    r.add_argument("--include", choices=["success", "failure", "all"], default="success")
    r.add_argument("--check-only", action="store_true", help="only run the rebuild check")
    r.set_defaults(func=_rows)

    t = s.add_parser("tag", help="add metadata.code_outcome; drop trials with tokenizer special tokens")
    t.add_argument("--src", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--tokenizer", type=Path, required=True, help="local model snapshot (tokenizer only, offline)")
    t.add_argument("--keep-special-token-trials", action="store_true")
    t.set_defaults(func=_tag)

    sp = s.add_parser("split", help="train/val split by seed (LIBERO-PRO by layout; seeds >= 401 are an error)")
    sp.add_argument("--src", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    sp.set_defaults(func=_split)

    v = s.add_parser("validate", help="validate JSONL rows against the schema")
    v.add_argument("files", type=Path, nargs="+")
    v.set_defaults(func=_validate)


def _pairs(a: argparse.Namespace) -> int:
    from geniac_baseline.data.pairs import write_pairs

    man = write_pairs(a.success, a.failure, a.out, seed=a.seed)
    print(json.dumps(man["pair_counts"], indent=1))
    return 0


def add_pairs_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("pairs", help="success / failure trajectory pairs for DMPO")
    p.add_argument("--success", type=Path, required=True, help="success rows (output of convert split)")
    p.add_argument("--failure", type=Path, required=True, help="failure rows (convert rows --include failure + tag)")
    p.add_argument("--out", type=Path, required=True, help="new output directory")
    p.add_argument("--seed", type=int, default=0, help="shuffle seed of the pairing")
    p.set_defaults(func=_pairs)
