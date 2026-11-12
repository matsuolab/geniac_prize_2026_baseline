#!/usr/bin/env bash
# Evaluate the base model with CaP-X (baseline settings, docs/eval.md), before any training.
#
#   bash examples/01_eval_base_cap_agent0.sh             # plan only (no GPU)
#   RUN_GPU=1 bash examples/01_eval_base_cap_agent0.sh   # also run [GPU] and score
#
# Config: configs/eval_base.yaml. Docs: docs/eval.md.
set -euo pipefail
cd "$(dirname "$0")/.."
# the package's command; falls back to the repo venv when it is not activated
GB=geniac-baseline; command -v "$GB" >/dev/null || GB=.venv/bin/geniac-baseline

SPEC=${SPEC:-configs/eval_base.yaml}

# 1. Preflight + write <output_dir>/run.sh (CPU, runs nothing).
RUN_SH=$("$GB" eval plan --spec "$SPEC" | sed -n 's/^\[plan\] wrote //p')
echo "[example] evaluation script: $RUN_SH"

if [ "${RUN_GPU:-0}" != "1" ]; then
  echo "[example] set RUN_GPU=1 on a GPU node to run it"
  exit 0
fi

# 2. [GPU] VDM server, code-writer server (thinking off), Molmo; then CaP-X's envs/launch.py per seed segment.
bash "$RUN_SH"

# 3. Score from the trial folders (success = taskcompleted_1 and sandboxrc_0).
RUN_DIR=$(dirname "$RUN_SH")
"$GB" eval score "$RUN_DIR" --json "$RUN_DIR/score.json"
