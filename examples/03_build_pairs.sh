#!/usr/bin/env bash
# Build success / failure trajectory pairs for DMPO from the rows written by examples/00_collect_data.sh (CPU).
#
#   bash examples/03_build_pairs.sh
#
# Pairs are drawn within a task (LIBERO-PRO: same layout first), both trials of a pair on the same side of the
# train / val split. Docs: docs/data_format.md.
set -euo pipefail
cd "$(dirname "$0")/.."
# the package's command; falls back to the repo venv when it is not activated
GB=geniac-baseline; command -v "$GB" >/dev/null || GB=.venv/bin/geniac-baseline

DATA=${DATA:-runs/data}

"$GB" pairs --success "$DATA/split" --failure "$DATA/fail_tagged" --out "$DATA/pairs"
