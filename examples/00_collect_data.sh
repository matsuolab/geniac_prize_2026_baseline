#!/usr/bin/env bash
# Collect multi-turn CaP-X trials and convert them into training rows: successes for SFT, failures for the DMPO
# pairs.
#
#   bash examples/00_collect_data.sh                 # plan only: preflight + write run.sh (no GPU)
#   RUN_GPU=1 bash examples/00_collect_data.sh       # also run the collection [GPU] and convert
#
# Config: configs/collect.yaml (CaP-X checkout, model paths, tasks, seeds). Docs: docs/collect.md.
# TOKENIZER must point at a local model snapshot for the tagging step (tokenizer files only).
set -euo pipefail
cd "$(dirname "$0")/.."
# the package's command; falls back to the repo venv when it is not activated
GB=geniac-baseline; command -v "$GB" >/dev/null || GB=.venv/bin/geniac-baseline

SPEC=${SPEC:-configs/collect.yaml}
DATA=${DATA:-runs/data}
TOKENIZER=${TOKENIZER:-}

# 1. Preflight + write <output_dir>/run.sh, per-segment configs and sources.yaml (CPU, runs nothing).
RUN_SH=$("$GB" collect plan --spec "$SPEC" | sed -n 's/^\[plan\] wrote //p')
RUN_DIR=$(dirname "$RUN_SH")
echo "[example] collection script: $RUN_SH"

if [ "${RUN_GPU:-0}" != "1" ]; then
  echo "[example] set RUN_GPU=1 on a GPU node to run it"
  exit 0
fi

# Checked before the GPU run: the tagging step after it needs a local tokenizer.
: "${TOKENIZER:?set TOKENIZER to a local model snapshot directory}"

# 2. [GPU] Start the model servers and run CaP-X's envs/launch.py for every seed segment.
bash "$RUN_SH"

# 3. Trials -> rows, successes and failures separately. The rebuild rule is learnt from both sets; nothing is
#    written unless every saved decision prompt rebuilds exactly.
"$GB" convert rows --sources "$RUN_DIR/sources.yaml" --out "$DATA/rows"
"$GB" convert rows --sources "$RUN_DIR/sources.yaml" --include failure --out "$DATA/fail_rows"

# 4. Tag code outcomes; drop trials containing tokenizer special tokens (needs transformers + tokenizer files).
"$GB" convert tag --src "$DATA/rows" --out "$DATA/tagged" --tokenizer "$TOKENIZER"
"$GB" convert tag --src "$DATA/fail_rows" --out "$DATA/fail_tagged" --tokenizer "$TOKENIZER"

# 5. Train / val split of the successes (layout-aware for LIBERO-PRO; seeds >= 401 are an error), then validate.
#    The failures are not split here: the pair builder (examples/03_build_pairs.sh) applies the same rule.
"$GB" convert split --src "$DATA/tagged" --out "$DATA/split"
"$GB" convert validate "$DATA/split/sft.jsonl" "$DATA/split/val.jsonl"
