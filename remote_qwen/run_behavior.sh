#!/usr/bin/env bash
# Run ASPIRE's BEHAVIOR-1K tasks (R1Pro, OmniGibson 3.9 / Isaac Sim 5.1) on this machine with the Qwen3.8-27B of
# remote_qwen/serve.sh, reached through remote_qwen/tunnel.sh (docs/remote_qwen.md). ASPIRE's own launcher
# (aspire/sim/cap/envs/launch_b1k.py) runs the trials and starts SAM3 and Contact-GraspNet; nothing is re-implemented.
#
#   bash remote_qwen/run_behavior.sh <task> <seeds> [output_dir]
#     task:   pick_up_trash (the blue soda can) | pick_up_radio      -> configs/capx/aspire/behavior/<task>.yaml
#     seeds:  26-35, or 26,28,30 (a seed is a BEHAVIOR task instance; ASPIRE uses 26-35 to develop, 1-25 to evaluate)
#     output: default runs/behavior/<task>-<UTC time>; score it with  geniac-baseline eval score <output_dir>
#
# Environment: ASPIRE_ROOT (default ./ASPIRE, from patches/aspire/setup.sh), SIM_PY (Python with OmniGibson 3.9, Isaac
# Sim 5.1, SAM3 and Contact-GraspNet), OMNIGIBSON_DATA_PATH (dataset root with 2026-challenge-task-instances),
# QWEN_URL (default http://127.0.0.1:8000/v1), CAPX_MOLMO_URL (default http://127.0.0.1:8122/v1; Molmo is only
# called when the generated code uses point_prompt_molmo), ASPIRE_TRIAL_TIMEOUT_S (default 2400).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
die() { echo "error: $*" >&2; exit 1; }
(($# == 2 || $# == 3)) || { sed -n '6,9p' "$0" >&2; exit 2; }
task="$1"
cfg="$root/configs/capx/aspire/behavior/$task.yaml"
out="$(realpath -m "${3:-$root/runs/behavior/$task-$(date -u +%Y%m%dT%H%M%SZ)}")"
aspire="$(realpath -m "${ASPIRE_ROOT:-$root/ASPIRE}")"
sim_py="${SIM_PY:-$root/geniac_prize_2026_code/capx/third_party/b1k/.venv-isaac5/bin/python}"
qwen="${QWEN_URL:-http://127.0.0.1:8000/v1}"
qwen="${qwen%/}"
model=Qwen/Qwen3.8-27B

[[ -f "$cfg" ]] || die "no config $cfg"
[[ -f "$aspire/aspire/sim/cap/envs/launch_b1k.py" ]] || die "no ASPIRE tree at $aspire (bash patches/aspire/setup.sh, or set ASPIRE_ROOT)"
[[ -x "$sim_py" ]] || die "no simulator Python at $sim_py; set SIM_PY"
[[ -f "${OMNIGIBSON_DATA_PATH:-}/2026-challenge-task-instances/metadata/available_tasks.yaml" ]] ||
  die "set OMNIGIBSON_DATA_PATH to a BEHAVIOR dataset root that has 2026-challenge-task-instances"
[[ ! -e "$out" ]] || die "$out already exists"

trials=()
IFS=, read -r -a parts <<<"$2"
for p in "${parts[@]}"; do
  if [[ "$p" =~ ^([0-9]+)-([0-9]+)$ ]] && ((10#${BASH_REMATCH[1]} <= 10#${BASH_REMATCH[2]})); then
    mapfile -t -O "${#trials[@]}" trials < <(seq "${BASH_REMATCH[1]}" "${BASH_REMATCH[2]}")
  elif [[ "$p" =~ ^[0-9]+$ ]]; then
    trials+=("$p")
  else
    die "bad seeds '$2' (e.g. 26-35 or 26,28,30)"
  fi
done

curl -sf -m 10 "$qwen/models" | grep -qF "\"id\":\"$model\"" ||
  die "$model is not answering at $qwen/models (is the tunnel or the Slurm job down?)"
export CAPX_MOLMO_URL="${CAPX_MOLMO_URL:-http://127.0.0.1:8122/v1}"
curl -sf -m 5 "$CAPX_MOLMO_URL/models" >/dev/null ||
  echo "warning: no Molmo at $CAPX_MOLMO_URL; point_prompt_molmo calls in the generated code will fail" >&2

export OMNI_KIT_ACCEPT_EULA=YES OMNIGIBSON_HEADLESS=1
export ASPIRE_TRIAL_TIMEOUT_S="${ASPIRE_TRIAL_TIMEOUT_S:-2400}"
# Isaac Sim needs libGLU; cap-x's BEHAVIOR setup keeps one next to its venv when the system has none.
glu="$(dirname "$(dirname "$(dirname "$sim_py")")")/.local-libs/usr/lib/x86_64-linux-gnu"
[[ -f "$glu/libGLU.so.1" ]] && export LD_LIBRARY_PATH="$glu${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
ulimit -c 0

mkdir -p "$out"
echo "[behavior] $task seeds ${trials[*]} -> $out/outputs/${model//\//_}/$task"
cd "$aspire/aspire/sim"
# ASPIRE inserts the model name before the last part of --output-dir (runner._setup_output_dir).
PYTHONPATH="$aspire${PYTHONPATH:+:$PYTHONPATH}" "$sim_py" "$aspire/aspire/sim/cap/envs/launch_b1k.py" \
  --config-path "$cfg" --model "$model" --server-url "$qwen/chat/completions" \
  --visual-differencing-model "$model" --visual-differencing-model-server-url "$qwen/chat/completions" \
  --trial-ids "${trials[@]}" --output-dir "$out/outputs/$task" 2>&1 | tee "$out/$task.log"
