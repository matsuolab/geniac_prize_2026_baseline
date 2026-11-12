#!/usr/bin/env bash
# SFT warm start: train a LoRA adapter on the success rows written by examples/00_collect_data.sh
# (completion-only cross-entropy on every answer token).
#
#   bash examples/02_train_sft.sh                # dry run: encode the rows and report (needs the [train] extra, no GPU)
#   RUN_GPU=1 bash examples/02_train_sft.sh      # train [GPU]
#
# Config: configs/train_sft.yaml. Docs: docs/train.md.
set -euo pipefail
cd "$(dirname "$0")/.."
# the package's command; falls back to the repo venv when it is not activated
GB=geniac-baseline; command -v "$GB" >/dev/null || GB=.venv/bin/geniac-baseline

CONFIG=${CONFIG:-configs/train_sft.yaml}

# 1. Encode every row with the model's chat template, print token counts and the step count (CPU).
"$GB" train sft --config "$CONFIG" --dry-run

if [ "${RUN_GPU:-0}" != "1" ]; then
  echo "[example] set RUN_GPU=1 on a GPU node to train"
  exit 0
fi

# 2. [GPU] Train. Adapters: <output_dir>/epoch-N/adapter after every epoch, <output_dir>/adapter at the end.
#    Resume an interrupted run with: geniac-baseline train sft --config "$CONFIG" --resume <output_dir>/ckpt-last
"$GB" train sft --config "$CONFIG"
# The epoch with the lowest val/loss_all is the one to evaluate and to start DMPO from (best_epoch.json);
# configs/train_dmpo.yaml points at the SFT run directory and picks it up.
