"""The row-level loss plug-in of the SFT trainer.

``compute_loss`` is the completion-only cross-entropy of the SFT warm start
(``geniac_baseline.trainer.losses.sft_ce``). The second stage, DMPO, is pair-level and has its own trainer
(``geniac_baseline.trainer.dmpo_trainer``, loss in ``geniac_baseline.trainer.losses.dmpo``).

Contract with the SFT trainer (``geniac_baseline.trainer.sft_trainer``):

* The trainer encodes one row at a time (batch size 1, as in the baseline runs), runs the model, and calls
  ``compute_loss(inputs)`` once per row with the tensors described in ``LossInputs``.
* It back-propagates ``LossOutput.loss`` immediately (a 27B model cannot hold the graphs of several long rows),
  sums ``LossOutput.normalizer`` over the ``grad_accum`` rows of one optimizer step, and divides the accumulated
  gradients by that sum before clipping and stepping. ``normalizer = number of tokens that carry loss`` gives a
  token-weighted mean over the window; ``normalizer = 1`` gives a per-row mean.
* A row that should not contribute returns ``LossOutput(loss=None, normalizer=0)``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # torch is an optional dependency; this module must import without it
    import torch


@dataclass(frozen=True)
class LossInputs:
    """Everything the trainer knows about one row at loss time.

    Positions: the model is run on the full rendered sequence (system + user + generation prompt + answer +
    ``<|im_end|>``), but logits are kept only from the position that predicts the first answer token onward
    (``logits_to_keep``), so ``logits[i]`` predicts ``target_ids[i]``.

    Attributes:
        logits: float32 tensor ``(n, vocab)``, logits for the ``n`` answer positions.
        target_ids: int64 tensor ``(n,)``, the answer tokens those positions predict: the assistant text followed
            by ``<|im_end|>``.
        answer_mask: bool tensor ``(n,)``. True for every assistant token the baseline would allow loss on
            (everything after the generation prompt up to and including the final ``<|im_end|>``; the trailing
            newline after it is False). The trainer applies no other masking: which code gets loss (e.g. by
            ``metadata["code_outcome"]``) is the loss's decision.
        decision_prefix_len: number of leading answer tokens that spell the decision word (``REGENERATE`` /
            ``FINISH``) on decision rows, 0 on first-turn rows. Lets a loss treat the decision word separately.
        input_ids: int64 tensor ``(seq_len,)``, the full rendered sequence.
        n_prompt_tokens: number of tokens up to and including the generation prompt.
        metadata: the row's ``metadata`` object (task, seed, turn, kind, feedback, code_outcome, success, ...).
        row_id: the row's ``id``.
    """

    logits: torch.Tensor
    target_ids: torch.Tensor
    answer_mask: torch.Tensor
    decision_prefix_len: int
    input_ids: torch.Tensor
    n_prompt_tokens: int
    metadata: Mapping[str, Any]
    row_id: str


@dataclass(frozen=True)
class LossOutput:
    """``loss``: scalar tensor to back-propagate (or None to skip the row); ``normalizer``: see module docstring."""

    loss: torch.Tensor | None
    normalizer: float
    n_loss_tokens: int = 0


def compute_loss(inputs: LossInputs) -> LossOutput:
    """Loss of one row: completion-only cross-entropy on every answer token (``losses.sft_ce``)."""
    from geniac_baseline.trainer.losses.sft_ce import completion_ce

    return completion_ce(inputs)
