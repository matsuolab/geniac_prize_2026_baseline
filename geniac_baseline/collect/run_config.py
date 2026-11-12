"""Write the per-run copy of a CaP-X config: only the trial range, worker count and server ports change.

The edits are done on the text, so the prompt and every other field stay byte-identical to the source config
(the converter later reads the template from this copy):

* ``trials: <end>`` and ``resume_idx: <start>``: CaP-X runs trials ``range(resume_idx, trials + 1)``
  (``capx/envs/runner.py`` L150-152) and seeds trial t with ``seed=t`` (``capx/envs/trial.py`` L779).
  Existing ``resume_idx`` lines are removed and one is appended (no duplicate key).
* ``port: 811x`` lines of ``api_servers`` moved by ``port_offset`` (only allowed when the checkout reads the URLs
  from the environment, see ``preflight.check_port_env``).
* ``num_workers`` optionally lowered (trial concurrency only; e.g. to re-run trials whose every LLM request
  timed out).
"""

from __future__ import annotations

import re
from pathlib import Path


class RunConfigError(ValueError):
    """The source config does not have the lines the rewrite expects."""


def rewrite_config_text(
    text: str, *, start: int, end: int, port_offset: int = 0, num_workers: int | None = None
) -> str:
    if port_offset:
        text = re.sub(
            r"(?m)^(\s*port:\s*)(811[34567])\s*$", lambda m: f"{m.group(1)}{int(m.group(2)) + port_offset}", text
        )
    text, n = re.subn(r"(?m)^trials:\s*\d+\s*$", f"trials: {end}", text)
    if n != 1:
        raise RunConfigError(f"expected exactly one top-level 'trials:' line, found {n}")
    if num_workers is not None:
        text, n = re.subn(r"(?m)^num_workers:\s*\d+\s*$", f"num_workers: {num_workers}", text)
        if n != 1:
            raise RunConfigError(f"expected exactly one top-level 'num_workers:' line, found {n}")
    text = re.sub(r"(?m)^resume_idx:.*\n?", "", text)
    return text.rstrip("\n") + f"\nresume_idx: {start}\n"


def write_run_config(
    src: Path, dst: Path, *, start: int, end: int, port_offset: int = 0, num_workers: int | None = None
) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(
        rewrite_config_text(src.read_text(), start=start, end=end, port_offset=port_offset, num_workers=num_workers)
    )
    return dst
