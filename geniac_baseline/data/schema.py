"""Schema of a training row, and a validator.

One row = one model call of one trial::

    {
      "id": "<sample_id>-t<turn>",
      "messages": [
        {"role": "system",    "content": [{"type": "text", "text": "..."}]},
        {"role": "user",      "content": [{"type": "text", "text": "<the prompt the model saw>"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "<the answer>"}]}
      ],
      "metadata": {
        "source": "capx_multiturn", "sample_id": "<task>__<run_tag>__trial<seed:03d>", "task": "...",
        "condition": "<task>", "seed": 302, "run_tag": "...", "n_llm_calls": 3, "success": true,
        "turn": 1, "kind": "first_turn" | "decision", "feedback": null | "saved" | "rebuilt",
        "code_outcome": null | "final" | "replaced_error" | "replaced_clean"      (present after tagging)
        "n_layouts": 50                                   (LIBERO-PRO rows only: the task's number of init states)
      }
    }

Text only: no "images" key (the decision prompt's images are not part of the rows; the VDM text is).
Answers: turn 0 is the program text; a decision answer is exactly ``FINISH`` or
``REGENERATE`` + newline + one fenced python block (``REGENERATE\\n```python\\n<code>\\n```).

Note on ensemble turns: with ``use_parallel_ensemble`` the recorded answer is the *synthesis* output, while the
user text is the decision prompt before synthesis (the prompt each candidate saw). The rows pair the two as if a
single call had produced the answer; this is inherited from the baseline data, not a CaP-X behaviour.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

ROLES = ("system", "user", "assistant")
KINDS = {"first_turn", "decision"}
FEEDBACK = {None, "saved", "rebuilt"}
CODE_OUTCOMES = {None, "final", "replaced_error", "replaced_clean"}
REQUIRED_METADATA: dict[str, type | tuple[type, ...]] = {
    "source": str,
    "sample_id": str,
    "task": str,
    "condition": str,
    "seed": int,
    "run_tag": str,
    "n_llm_calls": int,
    "success": bool,
    "turn": int,
    "kind": str,
}


class SchemaError(ValueError):
    """A row does not follow the schema."""


def _answer(row: dict[str, Any]) -> str:
    return "".join(p["text"] for p in row["messages"][2]["content"])


def validate_row(row: Any) -> None:
    """Raise ``SchemaError`` naming the first problem in ``row``."""
    if not isinstance(row, dict):
        raise SchemaError("row is not an object")
    rid = row.get("id", "<no id>")
    unknown = set(row) - {"id", "messages", "metadata"}
    if unknown:
        raise SchemaError(f"{rid}: unknown top-level keys {sorted(unknown)} (rows are text only)")
    msgs = row.get("messages")
    if not isinstance(msgs, list) or len(msgs) != 3:
        raise SchemaError(f"{rid}: messages must be a list of 3")
    for m, role in zip(msgs, ROLES, strict=True):
        if not isinstance(m, dict) or m.get("role") != role:
            raise SchemaError(f"{rid}: expected roles {ROLES}")
        content = m.get("content")
        if not isinstance(content, list) or not content:
            raise SchemaError(f"{rid}: {role} content must be a non-empty list of parts")
        for p in content:
            if not isinstance(p, dict) or p.get("type") != "text" or not isinstance(p.get("text"), str):
                raise SchemaError(f"{rid}: {role} has a non-text part")
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        raise SchemaError(f"{rid}: metadata missing")
    for key, typ in REQUIRED_METADATA.items():
        val = meta.get(key)
        # bool is an int subclass; seed / turn / n_llm_calls must not be booleans
        if not isinstance(val, typ) or (typ is int and isinstance(val, bool)):
            raise SchemaError(f"{rid}: metadata.{key} must be {typ}")
    if meta["kind"] not in KINDS:
        raise SchemaError(f"{rid}: metadata.kind {meta['kind']!r} not in {sorted(KINDS)}")
    if meta.get("feedback") not in FEEDBACK:
        raise SchemaError(f"{rid}: metadata.feedback {meta.get('feedback')!r} invalid")
    if "code_outcome" in meta and meta["code_outcome"] not in CODE_OUTCOMES:
        raise SchemaError(f"{rid}: metadata.code_outcome {meta['code_outcome']!r} invalid")
    if meta["task"].startswith("libero_"):
        n = meta.get("n_layouts")
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise SchemaError(f"{rid}: LIBERO-PRO rows need metadata.n_layouts (the task's number of init states)")
    elif "n_layouts" in meta:
        raise SchemaError(f"{rid}: metadata.n_layouts is only for LIBERO-PRO rows")
    if row.get("id") != f"{meta['sample_id']}-t{meta['turn']}":
        raise SchemaError(f"{rid}: id must be <sample_id>-t<turn>")
    if meta["condition"] != meta["task"]:
        raise SchemaError(f"{rid}: metadata.condition must equal metadata.task")
    ans = _answer(row)
    if meta["kind"] == "first_turn":
        if meta["turn"] != 0 or meta.get("feedback") is not None:
            raise SchemaError(f"{rid}: a first_turn row has turn 0 and feedback null")
        if not ans:
            raise SchemaError(f"{rid}: empty first program")
    else:
        if meta["turn"] < 1:
            raise SchemaError(f"{rid}: a decision row has turn >= 1")
        if ans == "FINISH":
            if meta.get("feedback") != "rebuilt":
                raise SchemaError(f"{rid}: FINISH rows are rebuilt (CaP-X never saves their prompt)")
        elif ans.startswith("REGENERATE\n```python\n") and ans.endswith("\n```"):
            if meta.get("feedback") != "saved":
                raise SchemaError(f"{rid}: REGENERATE rows use the saved prompt")
        else:
            raise SchemaError(f"{rid}: a decision answer is FINISH or REGENERATE + one fenced python block")
        if "code_outcome" in meta and (ans == "FINISH") != (meta["code_outcome"] is None):
            raise SchemaError(f"{rid}: code_outcome is null exactly for FINISH rows")


def validate_rows(rows: Iterable[Any]) -> int:
    """Validate every row; also checks that ids are unique. Returns the number of rows."""
    seen: set[str] = set()
    n = 0
    for row in rows:
        validate_row(row)
        if row["id"] in seen:
            raise SchemaError(f"duplicate id {row['id']}")
        seen.add(row["id"])
        n += 1
    return n
