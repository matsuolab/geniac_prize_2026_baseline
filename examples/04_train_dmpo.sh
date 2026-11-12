#!/usr/bin/env bash
# DMPO on top of the SFT adapter: policy initialised from it, frozen reference = it.
#
#   bash examples/04_train_dmpo.sh                # dry run: encode the pairs and report (needs the [train] extra, no GPU)
#   RUN_GPU=1 bash examples/04_train_dmpo.sh      # train [GPU]
#
# Config: configs/train_dmpo.yaml (init_adapter = the SFT adapter). Docs: docs/train.md.
set -euo pipefail
cd "$(dirname "$0")/.."
# the package's command; falls back to the repo venv when it is not activated
GB=geniac-baseline; command -v "$GB" >/dev/null || GB=.venv/bin/geniac-baseline

CONFIG=${CONFIG:-configs/train_dmpo.yaml}

# 1. Encode every turn of every paired trajectory and report pair, turn and step counts (CPU).
"$GB" train dmpo --config "$CONFIG" --dry-run

if [ "${RUN_GPU:-0}" != "1" ]; then
  echo "[example] set RUN_GPU=1 on a GPU node to train"
  exit 0
fi

# 2. [GPU] Score every turn once with the SFT adapter (cached in <output_dir>/ref_logp.json), then train.
#    Adapters: <output_dir>/epoch-N/adapter after every epoch, <output_dir>/adapter at the end.
#    Resume an interrupted run with: geniac-baseline train dmpo --config "$CONFIG" --resume <output_dir>/ckpt-last
"$GB" train dmpo --config "$CONFIG"
