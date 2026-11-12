#!/usr/bin/env bash
# Build the ASPIRE tree the baseline data and adapters were made with: NVlabs/ASPIRE @ f4c8939 + patches/aspire/aspire-f4c8939-local-changes.patch.
#   bash patches/aspire/setup.sh [target_dir]          (default: ./ASPIRE)
# ASPIRE_URL overrides the clone URL. Submodules and venvs are then set up per the ASPIRE README.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
patch="$here/aspire-f4c8939-local-changes.patch"
target="${1:-ASPIRE}"
url="${ASPIRE_URL:-https://github.com/NVlabs/ASPIRE.git}"
commit=f4c8939

if [ -e "$target" ]; then
  echo "error: $target already exists; give a new directory" >&2
  exit 1
fi
git clone "$url" "$target"
git -C "$target" checkout --quiet "$commit"
git -C "$target" apply --check "$patch"
git -C "$target" apply "$patch"
echo "applied $(basename "$patch") on ASPIRE @ $commit"
if command -v sha256sum >/dev/null; then sum=(sha256sum); else sum=(shasum -a 256); fi
(cd "$target" && "${sum[@]}" -c "$here/SHA256SUMS")
