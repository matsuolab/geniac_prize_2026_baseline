#!/usr/bin/env bash
# Submit remote_qwen/serve.sh as a Slurm batch job, from the cluster's login node (docs/remote_qwen.md).
# Arguments go to sbatch after the defaults below and override them:
#   bash remote_qwen/submit.sh --partition=<partition> [--time=2-00:00:00] [--account=<account>]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
logs="$root/runs/qwen_server"
command -v sbatch >/dev/null || { echo "error: sbatch not found; run this on the cluster's login node" >&2; exit 1; }
mkdir -p "$logs"

job=$(sbatch --parsable --job-name=qwen-server --nodes=1 --ntasks=1 --gpus=1 --mem=64G --time=1-00:00:00 \
  --no-requeue --chdir="$root" --output="$logs/%x-%j.out" --export="ALL,GENIAC_ROOT=$root" \
  "$@" "$root/remote_qwen/serve.sh")
job="${job%%;*}"
echo "submitted job $job; log: $logs/qwen-server-$job.out"
echo "node: squeue -h -j $job -o %N    stop: scancel $job"
