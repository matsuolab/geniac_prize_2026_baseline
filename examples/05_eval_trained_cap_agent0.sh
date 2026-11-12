#!/usr/bin/env bash
# Evaluate the DMPO adapter with CaP-X (baseline settings, docs/eval.md), on the same seeds as the base evaluation.
#
#   bash examples/05_eval_trained_cap_agent0.sh             # plan only (no GPU)
#   RUN_GPU=1 bash examples/05_eval_trained_cap_agent0.sh   # also run [GPU] and score
#
# Config: configs/eval_trained.yaml (adapter + lora_name). Docs: docs/eval.md.
set -euo pipefail
cd "$(dirname "$0")/.."
# the package's command; falls back to the repo venv when it is not activated
GB=geniac-baseline; command -v "$GB" >/dev/null || GB=.venv/bin/geniac-baseline

SPEC=${SPEC:-configs/eval_trained.yaml}

# 1. Preflight (checks the adapter directory) + write <output_dir>/run.sh (CPU, runs nothing).
RUN_SH=$("$GB" eval plan --spec "$SPEC" | sed -n 's/^\[plan\] wrote //p')
echo "[example] evaluation script: $RUN_SH"

if [ "${RUN_GPU:-0}" != "1" ]; then
  echo "[example] set RUN_GPU=1 on a GPU node to run it"
  exit 0
fi

# 2. [GPU] Same servers as the base evaluation; the code-writer server also serves the LoRA under lora_name.
bash "$RUN_SH"

# 3. Score, then compare with the base run (same seeds).
RUN_DIR=$(dirname "$RUN_SH")
"$GB" eval score "$RUN_DIR" --json "$RUN_DIR/score.json"
