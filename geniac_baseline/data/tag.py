"""Tag every code answer with what happened to that code, and drop trials that contain tokenizer special tokens.

``metadata.code_outcome`` for rows whose answer contains code (first turn, REGENERATE):

    final           the trial's next answer is FINISH, or there is no later row: this code ended the trial
    replaced_error  the next answer is REGENERATE and its prompt's stderr has a Python traceback
    replaced_clean  the next answer is REGENERATE and the code raised nothing

FINISH rows get ``None``. The traceback is read from the next row's saved prompt (its "The current console stderr
is:" block), which is the stderr of the code in question.

Trials whose system/user/assistant text contains a special token of the tokenizer (e.g. Qwen's ``<tool_call>``)
are dropped whole unless ``keep_special_token_trials``: such a string is tokenized as the special token, so the
target and every later prompt that quotes the executed code would not round-trip as text (found in the baseline
data: models occasionally wrote their first program in tool-call markup).

The tags are inputs to a loss; this module does not decide which code gets loss.
"""

from __future__ import annotations

import collections
import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from geniac_baseline.data.prompts import split_saved, text_of

TRACEBACK = "Traceback (most recent call last)"


class TagError(ValueError):
    """The rows of a trial are not a contiguous turn sequence starting at 0."""


def _text(message: dict[str, Any]) -> str:
    return text_of(message["content"])


def tag_rows(
    rows: Iterable[dict[str, Any]], specials: set[str], keep_special: bool, counts: collections.Counter[Any]
) -> list[dict[str, Any]]:
    """Return tagged copies of ``rows`` grouped by trial (input is not modified)."""
    by: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by[r["metadata"]["sample_id"]].append(json.loads(json.dumps(r)))
    out: list[dict[str, Any]] = []
    for sid, v in by.items():
        v.sort(key=lambda r: r["metadata"]["turn"])
        turns = [r["metadata"]["turn"] for r in v]
        if turns != list(range(len(v))):
            raise TagError(f"{sid}: turns not contiguous from 0: {turns}")
        found = sorted({s for r in v for m in r["messages"] for s in specials if s in _text(m)})
        if found and not keep_special:
            counts["dropped_trials_special_token"] += 1
            counts["dropped_rows_special_token"] += len(v)
            continue
        for i, r in enumerate(v):
            ans = _text(r["messages"][2])
            if ans == "FINISH":
                outcome = None
            elif i + 1 == len(v) or _text(v[i + 1]["messages"][2]) == "FINISH":
                outcome = "final"
            elif _text(v[i + 1]["messages"][2]).startswith("REGENERATE"):
                stderr = split_saved(_text(v[i + 1]["messages"][1]))["stderr"]
                outcome = "replaced_error" if TRACEBACK in stderr else "replaced_clean"
            else:
                raise TagError(f"{r['id']}: next answer is neither FINISH nor REGENERATE")
            r["metadata"]["code_outcome"] = outcome
            head = "REGENERATE" if ans.startswith("REGENERATE") else ans[:6]
            counts[(r["metadata"]["kind"], head, outcome)] += 1
            out.append(r)
    return out


def tokenizer_special_tokens(tokenizer_path: Path) -> set[str]:
    """Added-vocabulary and special tokens of a local tokenizer (needs ``transformers``; no download)."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    return set(tok.get_added_vocab()) | set(tok.all_special_tokens)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().split("\n") if line.strip()]


def write_tagged(src: Path, out: Path, specials: set[str], keep_special: bool = False) -> dict[str, Any]:
    """Tag ``<src>/rows.jsonl`` into a new directory ``out`` (``rows.jsonl`` + ``manifest.json``)."""
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    rows = read_jsonl(src / "rows.jsonl")
    counts: collections.Counter[Any] = collections.Counter()
    tagged = tag_rows(rows, specials, keep_special, counts)
    out.mkdir(parents=True)
    with (out / "rows.jsonl").open("w") as f:
        for r in tagged:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = json.loads((src / "manifest.json").read_text())
    manifest.update(
        {
            "stage": "tagged",
            "tagged_from": str(src),
            "keep_special_token_trials": keep_special,
            "tag_counts": {
                "rows_in": len(rows),
                "rows_out": len(tagged),
                **{"|".join(map(str, k)) if isinstance(k, tuple) else k: n for k, n in sorted(counts.items(), key=str)},
            },
        }
    )
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest
