#!/usr/bin/env bash
# Serve Qwen3.8-27B (bf16, OpenAI API) on a Unix socket of a B300 compute node, for ASPIRE on a separate client
# (docs/remote_qwen.md). Run it inside a Slurm job: as the batch job of remote_qwen/submit.sh, or directly in an
# interactive job (--mem=64G or more). The client reaches it through remote_qwen/tunnel.sh.
#
# The node is shared with other users, and a server on 127.0.0.1:<port> accepts connections from all of them. So
# the API is on no TCP port: vLLM listens on /tmp/qwen-<user>/qwen.sock, in a directory only this user can enter.
#
# Environment: SERVE_PY (Python with vLLM; default the .venv-serve of docs/installation.md), QWEN_MODEL_PATH
# (local snapshot; default the Hub id at the baseline's revision), QWEN_SOCKET (the socket path; give tunnel.sh the
# same value), QWEN_MAX_MODEL_LEN (122880),
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

# /tmp is node-local, so two jobs on different nodes do not share the path; anyone can create names in it, so
# only a directory this user owns, with mode 700, is used.
sock="${QWEN_SOCKET:-/tmp/qwen-$(id -un)/qwen.sock}"
dir="$(dirname "$sock")"
mkdir -p -m 700 "$dir"
if [[ ! -O "$dir" || -L "$dir" ]]; then
  echo "error: $dir is not a directory of $(id -un); set QWEN_SOCKET to a path in a directory you own" >&2
  exit 1
fi
chmod 700 "$dir"
if [[ -e "$sock" ]]; then
  if curl -sf --max-time 5 --unix-socket "$sock" http://127.0.0.1/health >/dev/null 2>&1; then
    echo "error: a server is already answering on $sock; stop it first" >&2
    exit 1
  fi
  rm -f "$sock"  # left by a server that is gone; vLLM does not replace an existing file
fi

args=(
  --uds "$sock"
  --model "$model" --served-model-name Qwen/Qwen3.8-27B --dtype bfloat16
  --max-model-len "${QWEN_MAX_MODEL_LEN:-122880}" --gpu-memory-utilization "${QWEN_GPU_UTIL:-0.64}"
  --reasoning-parser qwen3
  # B300 workarounds (geniac_baseline/collect/settings.py B300_VLLM_FLAGS): the default FLASH_ATTN kernel fails to JIT
  --attention-backend TRITON_ATTN --mm-encoder-attn-backend TORCH_SDPA
)
[[ "$model" == Qwen/Qwen3.8-27B ]] && args+=(--revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0)
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-10.0}"

# vLLM's engine opens TCP ports for its own processes even with --uds. In a network namespace of its own they exist
# only there, so nobody on the node can connect to them; the socket file is unaffected. The namespace has no
# network, so this needs local weights (QWEN_MODEL_PATH).
isolate=()
if [[ -d "$model" ]] && unshare -r -n sh -c 'ip link set lo up' 2>/dev/null; then
  isolate=(env HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 unshare -r -n sh -c 'ip link set lo up; exec "$@"' sh)
else
  echo "[serve] WARNING: no network namespace (it needs QWEN_MODEL_PATH to be a local snapshot, and unshare -r -n):" >&2
  echo "[serve] WARNING: the engine's internal TCP ports stay reachable from this node; the API is on the socket only" >&2
fi

echo "[serve] $(hostname) job ${SLURM_JOB_ID:-none}: Qwen/Qwen3.8-27B on $sock"
# The socket file takes its mode from the umask: owner only, like the directory it is in.
umask 077
exec${isolate[@]+"${isolate[@]}"} "$serve_py" -m vllm.entrypoints.openai.api_server "${args[@]}"
