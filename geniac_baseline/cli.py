"""``geniac-baseline`` command line: collect, convert, pairs, train, eval, template."""

from __future__ import annotations

import argparse
import os
import sys

from geniac_baseline import __version__
from geniac_baseline.collect import cli as collect_cli
from geniac_baseline.collect import template as template_cli
from geniac_baseline.data import cli as convert_cli
from geniac_baseline.eval import cli as eval_cli
from geniac_baseline.trainer import cli as train_cli


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="geniac-baseline", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    convert_cli.add_parser(sub)
    convert_cli.add_pairs_parser(sub)
    train_cli.add_parser(sub)
    collect_cli.add_parser(sub)
    eval_cli.add_parser(sub)
    template_cli.add_parser(sub)
    return ap


DOCS = {
    "collect": "docs/collect.md",
    "eval": "docs/eval.md",
    "convert": "docs/data_format.md",
    "pairs": "docs/data_format.md",
    "train": "docs/train.md",
    "template": "docs/installation.md",
}
TRAIN_EXTRA = {"torch", "transformers", "peft", "accelerate"}


def main(argv: list[str] | None = None) -> int:
    """Errors a user can fix print one line and the page to read; GENIAC_BASELINE_DEBUG=1 shows the traceback."""
    a = build_parser().parse_args(argv)
    try:
        return int(a.func(a) or 0)
    except ModuleNotFoundError as e:
        if os.environ.get("GENIAC_BASELINE_DEBUG") or (e.name or "").split(".")[0] not in TRAIN_EXTRA:
            raise
        print(
            f"error: {e.name} is not installed; install the training stack: uv pip install -e '.[train]'",
            file=sys.stderr,
        )
        return 1
    except (ValueError, OSError, RuntimeError) as e:
        if os.environ.get("GENIAC_BASELINE_DEBUG"):
            raise
        print(f"error: {type(e).__name__}: {e}", file=sys.stderr)
        print(f"see {DOCS.get(a.cmd, 'README.md')}; GENIAC_BASELINE_DEBUG=1 shows the traceback", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
