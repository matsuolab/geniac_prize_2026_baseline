"""Pieces of a CaP-X multi-turn decision prompt, and the rule that rebuilds one from raw ingredients.

The decision prompt the model sees after code block k has run is built by CaP-X as (``capx/envs/trial.py``
``_handle_multi_turn_step`` + ``capx/utils/launch_utils.py`` ``_build_multi_turn_decision_prompt``)::

    copy of the initial prompt's last user message content
      + multi_turn_prompt.format(executed_code, console_stdout, console_stderr)
      + [VDM header + "\\n" + VDM text]                (when image differencing produced feedback)
      + the fixed "Based on the code output ... REGENERATE ... FINISH ..." instruction
    then every run of consecutive text parts is joined with "\\n" after each part (collapse_text_image_inputs).

The rule used here rebuilds the *user text* of that prompt as::

    user = PREFIX(task) + TEMPLATE.format(executed_code, console_stdout, console_stderr) + "\\n\\n"
           + VDM_HEADER + "\\n" + vdm_file_text + TAIL

It was not derived from the CaP-X source by hand: ``rows.check_rebuild_rule`` re-verifies it against every saved
prompt of the data being converted and refuses to write rows if a single one does not rebuild exactly.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

# First line of every multi_turn_prompt template in the CaP-X configs (env_configs/*/*multiturn*.yaml).
MARK = "The following code blocks were just executed:"
# Start of the image-differencing header in capx/utils/launch_utils.py (_build_multi_turn_decision_prompt).
VDM_HEAD = "Included below is the observed differences"
STDOUT_OPEN = "The current console stdout is:\n```\n"
STDERR_SEP = "\n```\nThe current console stderr is:\n```\n"


def text_of(content: Any) -> str:
    """Concatenate the text parts of an OpenAI-style message content (a string is returned as is)."""
    if isinstance(content, str):
        return content
    return "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")


def yaml_template(cfg: Path) -> str:
    """The ``multi_turn_prompt: |`` block of a CaP-X config, exactly as written (indentation removed).

    Read with a regex rather than a YAML parser on purpose: the rebuild check compares strings byte for byte,
    so the template is taken verbatim from the file the run used.
    """
    t = cfg.read_text()
    m = re.search(r"\n(\s*)multi_turn_prompt: \|\n((?:\1  .*\n|\s*\n)+)", t)
    if not m:
        raise ValueError(f"no 'multi_turn_prompt: |' block in {cfg}")
    ind = len(m.group(1)) + 2
    return "\n".join(line[ind:] if line.strip() else "" for line in m.group(2).split("\n")).rstrip("\n")


def split_saved(user: str) -> dict[str, str]:
    """Split a saved decision prompt's user text into prefix / template part / stdout / stderr / VDM part."""
    i, j = user.index(MARK), user.find(VDM_HEAD)
    mid = user[i:j] if j >= 0 else user[i:]
    a1 = mid.index(STDOUT_OPEN) + len(STDOUT_OPEN)
    b1 = mid.index(STDERR_SEP)
    a2 = b1 + len(STDERR_SEP)
    b2 = mid.index("\n```\n", a2) if "\n```\n" in mid[a2:] else len(mid)
    return {
        "prefix": user[:i],
        "mid": mid,
        "stdout": mid[a1:b1],
        "stderr": mid[a2:b2],
        "vdm": user[j:] if j >= 0 else "",
    }


def summary_feedback(summary: str) -> tuple[str, str] | None:
    """(stdout, stderr) recorded in a trial's summary.txt, or None if the block is not there.

    CaP-X writes the last executed block's stdout, and as stderr the join of every non-empty stderr of the trial
    (``capx/envs/trial.py``: ``"\\n\\n".join(stderr_history)``).
    """
    m = re.search(r"\n  Stdout: (.*?)\n  Stderr: (.*?)\n  Reward:", summary, re.S)
    return (m[1], m[2]) if m else None


def build(
    task_prefix: str, tpl: str, executed: str, stdout: str, stderr: str, vdm_header: str | None, vdm: str | None
) -> str:
    """Rebuild a decision prompt's user text with the rule in the module docstring."""
    u = task_prefix + tpl.format(executed_code=executed, console_stdout=stdout, console_stderr=stderr) + "\n\n"
    if vdm is not None:
        if vdm_header is None:
            raise ValueError("a VDM text needs its header")
        u += vdm_header + "\n" + vdm
    return u
