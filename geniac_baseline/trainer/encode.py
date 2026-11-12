"""Render a row with the model's chat template and locate the answer tokens.

Unchanged from the baseline trainer: rendering with the ``chat_template_kwargs`` recorded in the data manifest
(thinking off), answer = everything after the generation prompt up to and including the final ``<|im_end|>``,
and the asserts that the prompt rendering is a string and token prefix of the full rendering. What used to be
the code mask now lives in the loss (``LossInputs.metadata``), so this module does no masking by content.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

DECISION_WORDS = ("REGENERATE", "FINISH")


@dataclass
class EncodedRow:
    """One row ready for the model (tensors are 1-D, batch dimension added at forward time)."""

    row_id: str
    input_ids: Any  # torch.LongTensor (seq_len,)
    n_prompt: int
    answer_mask: Any  # torch.BoolTensor (seq_len - n_prompt,)
    decision_prefix_len: int
    group: str
    decision: str | None
    metadata: dict[str, Any]

    @property
    def keep(self) -> int:
        """Logits to keep: from the position that predicts the first answer token to the end."""
        return int(self.input_ids.shape[0]) - self.n_prompt + 1


def text_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    msgs = []
    for m in row["messages"]:
        parts = [{k: v for k, v in p.items() if v is not None} for p in m["content"]]
        if any(p["type"] != "text" for p in parts):
            raise ValueError(f"{row['id']}: non-text content")
        msgs.append({"role": m["role"], "content": parts})
    return msgs


def answer_text(row: dict[str, Any]) -> str:
    return "".join(p.get("text", "") for p in row["messages"][-1]["content"])


def group_of(row: dict[str, Any]) -> str:
    """Validation group: finish / first_final / first_replaced / regen_final / regen_replaced."""
    ans, oc = answer_text(row), row["metadata"].get("code_outcome")
    if ans == "FINISH":
        return "finish"
    kind = "first" if row["metadata"]["kind"] == "first_turn" else "regen"
    return f"{kind}_{'final' if oc == 'final' else 'replaced'}"


def decision_word_ids(tokenizer: Any) -> dict[str, int]:
    """First token id of each decision word; the two must differ for the decision accuracy to be readable."""
    ids = {w: tokenizer(w, add_special_tokens=False)["input_ids"][0] for w in DECISION_WORDS}
    if ids["REGENERATE"] == ids["FINISH"]:
        raise ValueError(f"REGENERATE and FINISH share their first token: {ids}")
    return ids


def encode_row(
    processor: Any, row: dict[str, Any], template_kwargs: dict[str, Any], word_ids: dict[str, int]
) -> EncodedRow:
    import torch

    tok = processor.tokenizer
    msgs = text_messages(row)
    prompt = processor.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True, **template_kwargs)
    full = processor.apply_chat_template(msgs, tokenize=False, **template_kwargs)
    if not full.startswith(prompt):
        raise AssertionError(f"{row['id']}: prompt rendering is not a prefix of the full rendering")
    ids = tok(full, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    p_ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
    n_prompt = len(p_ids)
    if not torch.equal(ids[:n_prompt], p_ids):
        raise AssertionError(f"{row['id']}: token prefix mismatch")
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    last = int((ids == im_end).nonzero()[-1].item())
    if last < n_prompt:
        raise AssertionError(f"{row['id']}: no <|im_end|> after the generation prompt")
    answer_mask = torch.zeros(len(ids) - n_prompt, dtype=torch.bool)
    answer_mask[: last + 1 - n_prompt] = True  # the trailing newline after the final <|im_end|> gets no loss
    ans = answer_text(row)
    decision_prefix_len = 0
    for w, wid in word_ids.items():  # the decision accuracy reads exactly this token
        if ans.startswith(w):
            if int(ids[n_prompt].item()) != wid:
                raise AssertionError(f"{row['id']}: first answer token is not {w}'s first token")
            if row["metadata"].get("kind") == "decision":
                k_ids = tok(prompt + w, return_tensors="pt", add_special_tokens=False)["input_ids"][0]
                if not torch.equal(ids[: len(k_ids)], k_ids):
                    raise AssertionError(f"{row['id']}: prompt + {w} is not a token prefix")
                decision_prefix_len = len(k_ids) - n_prompt
    decision = ans.split("\n", 1)[0] if ans.startswith(DECISION_WORDS) else None
    return EncodedRow(
        row["id"], ids, n_prompt, answer_mask, decision_prefix_len, group_of(row), decision, row["metadata"]
    )
