#!/usr/bin/env bash
# Build the CaP-X tree for collection and evaluation: the competition code
# (matsuolab/geniac_prize_2026_code at the commit below) + the two patches in patches/geniac_prize/, with the
# submodules the baseline uses:
#   geniac-prize.patch    fixes the runs depend on (trial time limit, VDM text files, LIBERO path, server URLs)
#   skill-library.patch   ASPIRE's 14 LIBERO skills, which the baseline's LIBERO data calls
#   bash patches/geniac_prize/setup.sh [target_dir]          (default: ./geniac_prize_2026_code)
# GENIAC_PRIZE_URL overrides the clone URL (SSH by default while the repository is private; switch to HTTPS once
# it is public). The venvs are then made per docs/installation.md.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patches=("$here/geniac-prize.patch" "$here/skill-library.patch")
target="${1:-geniac_prize_2026_code}"
url="${GENIAC_PRIZE_URL:-git@github.com:matsuolab/geniac_prize_2026_code.git}"
commit=ac680843fb60c464aa74c87a2815cb5917865be4
# BEHAVIOR (b1k) runs on a separate server at evaluation time; private is not public.
submodules=(
  capx/third_party/robosuite
  capx/third_party/libero_dependencies/robosuite
  capx/third_party/LIBERO-PRO
  capx/third_party/verl
  capx/third_party/contact_graspnet_pytorch
  capx/third_party/sam3
  capx/third_party/curobo
)

if [ -e "$target" ]; then
  echo "error: $target already exists; give a new directory" >&2
  exit 1
fi
git clone "$url" "$target"
git -C "$target" checkout --quiet "$commit"
for patch in "${patches[@]}"; do
  git -C "$target" apply --check "$patch"
  git -C "$target" apply "$patch"
  echo "applied $(basename "$patch") on geniac_prize_2026_code @ ${commit:0:7}"
done
git -C "$target" diff --name-only | while read -r f; do
  if command -v sha256sum >/dev/null; then sha256sum "$target/$f"; else shasum -a 256 "$target/$f"; fi
done
git -C "$target" submodule update --init --recursive -- "${submodules[@]}"
