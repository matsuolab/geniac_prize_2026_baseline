"""Where the training rows come from: a local directory or a Hugging Face dataset repo.

No dataset repo is built in. The location is always given by the user, as a local path or as
``hf://<repo_id>[@<revision>][/<subdir>]`` (``huggingface_hub`` resolves it; pin ``@<revision>`` to a commit so
the data cannot change under a run).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_HF_RE = re.compile(r"^hf://(?P<repo>[^@/\s]+/[^@/\s]+)(?:@(?P<rev>[^/\s]+))?(?:/(?P<sub>\S+))?$")


@dataclass(frozen=True)
class HfLocation:
    repo_id: str
    revision: str | None
    subdir: str | None


def parse_hf_location(s: str) -> HfLocation:
    m = _HF_RE.match(s)
    if not m:
        raise ValueError(f"not an hf:// location: {s!r} (expected hf://<owner>/<name>[@<revision>][/<subdir>])")
    return HfLocation(m["repo"], m["rev"], m["sub"])


def resolve_data_dir(data: str, local_files_only: bool = True) -> Path:
    """Local directory holding ``manifest.json`` and the row files."""
    if data.startswith("hf://"):
        loc = parse_hf_location(data)
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:  # pragma: no cover - depends on the optional extra
            raise RuntimeError("hf:// data needs the 'hub' extra (pip install 'geniac-baseline[hub]')") from e
        root = Path(
            snapshot_download(
                loc.repo_id, repo_type="dataset", revision=loc.revision, local_files_only=local_files_only
            )
        )
        d = root / loc.subdir if loc.subdir else root
    else:
        d = Path(data)
    if not (d / "manifest.json").is_file():
        raise FileNotFoundError(f"{d}: no manifest.json (expected the output of 'geniac-baseline convert split')")
    return d


def read_manifest(d: Path) -> dict[str, Any]:
    return json.loads((d / "manifest.json").read_text())


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]
