#!/usr/bin/env bash
# Serve Qwen3.8-27B (bf16, OpenAI API) on 127.0.0.1:8000 of a B300 compute node, for ASPIRE on a separate client
# (docs/remote_qwen.md). Run it inside a Slurm job: as the batch job of remote_qwen/submit.sh, or directly in an
# interactive job (--mem=64G or more). The client reaches it through remote_qwen/tunnel.sh.
#
# Environment: SERVE_PY (Python with vLLM; default the .venv-serve of docs/installation.md), QWEN_MODEL_PATH
# (local snapshot; default the Hub id at the baseline's revision), QWEN_PORT (8000), QWEN_MAX_MODEL_LEN (122880),
# QWEN_GPU_UTIL (0.64). HF_HOME selects where the weights are downloaded on first start.
set -euo pipefail
root="${GENIAC_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
serve_py="${SERVE_PY:-$root/geniac_prize_2026_code/.venv-serve/bin/python}"
model="${QWEN_MODEL_PATH:-Qwen/Qwen3.8-27B}"

[[ -x "$serve_py" ]] || { echo "error: no vLLM Python at $serve_py; set SERVE_PY" >&2; exit 1; }
# vLLM needs host RAM besides the GPU: a 1 GiB allocation was OOM-killed while loading the bf16 weights.
mem="${SLURM_MEM_PER_NODE:-0}"
if [[ "$mem" =~ ^[0-9]+$ ]] && ((mem > 0 && mem < 65536)); then
  echo "error: this job has $mem MiB of host RAM; request --mem=64G or more" >&2
  exit 1
fi

args=(
  --host 127.0.0.1 --port "${QWEN_PORT:-8000}"
  --model "$model" --served-model-name Qwen/Qwen3.8-27B --dtype bfloat16
  --max-model-len "${QWEN_MAX_MODEL_LEN:-122880}" --gpu-memory-utilization "${QWEN_GPU_UTIL:-0.64}"
  --reasoning-parser qwen3
  # B300 workarounds (geniac_baseline/collect/settings.py B300_VLLM_FLAGS): the default FLASH_ATTN kernel fails to JIT
  --attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA
)
[[ "$model" == Qwen/Qwen3.8-27B ]] && args+=(--revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

echo "[serve] $(hostname) job ${SLURM_JOB_ID:-none}: Qwen/Qwen3.8-27B on 127.0.0.1:${QWEN_PORT:-8000}"
exec "$serve_py" -m vllm.entrypoints.openai.api_server "${args[@]}"
