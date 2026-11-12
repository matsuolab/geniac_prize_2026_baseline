#!/usr/bin/env bash
# Put the baseline's skill library and task prompts into a CaP-Agent0 submission image (docs/tutorial/10_upload.md).
# Run from %post of the image definition, with this directory copied into the image:
#   bash /opt/baseline-submission/install.sh [capx_root]          (default: /workspace/capx)
# It
#   1. applies skill-library.patch (patches/geniac_prize/, copied next to this script): ASPIRE's 14 LIBERO skills
#   2. applies trial-state.patch: each trial starts with the APIs' attributes as they were built (rulebook §8.4)
#   3. copies gateway/*.yaml into env_configs/gateway/ (the baseline's APIs and prompts)
#   4. adds task_map.yaml's entries to env_configs/gateway/runner.yaml
# A patch that does not apply stops the build: the image's CaP-X has moved past what the patch was made for.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="${1:-/workspace/capx}"

for patch in "$here/skill-library.patch" "$here/trial-state.patch"; do
  (cd "$root" && git apply --check "$patch" && git apply "$patch")
  echo "[baseline] applied $(basename "$patch")"
done

cp "$here"/gateway/*.yaml "$root/env_configs/gateway/"
echo "[baseline] copied $(cd "$here/gateway" && echo *.yaml) to env_configs/gateway/"

python - "$root/env_configs/gateway/runner.yaml" "$here/task_map.yaml" <<'PY'
import sys

import yaml

runner_path, extra_path = sys.argv[1:]
with open(runner_path) as f:
    runner = yaml.safe_load(f)
with open(extra_path) as f:
    extra = yaml.safe_load(f)
ids = {e["task_id"] for e in extra}
runner["task_map"] = [e for e in runner.get("task_map") or [] if e["task_id"] not in ids] + extra
with open(runner_path, "w") as f:
    yaml.safe_dump(runner, f, sort_keys=False)
print("[baseline] runner.yaml task_map:", ", ".join(f"{e['task_id']} -> {e['config']}" for e in runner["task_map"]))
PY
