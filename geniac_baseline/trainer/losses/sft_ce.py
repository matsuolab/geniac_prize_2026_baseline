"""Completion-only cross-entropy: the SFT warm-start loss.

Loss on every answer token (assistant text and the closing ``<|im_end|>``), nothing on the prompt; summed per row
and normalised by the number of answer tokens over the accumulation window, i.e. a token-weighted mean. This is the
team recipe (``heuristic_sft_train.py``) the baseline SFT adapters were trained with.
"""

from __future__ import annotations

from geniac_baseline.trainer.loss import LossInputs, LossOutput


def completion_ce(inputs: LossInputs) -> LossOutput:
    """Sum of token cross-entropies over ``inputs.answer_mask``; ``normalizer`` = number of those tokens."""
    import torch

    per_tok = torch.nn.functional.cross_entropy(inputs.logits, inputs.target_ids, reduction="none")
    n = int(inputs.answer_mask.sum())
    if n == 0:
        return LossOutput(loss=None, normalizer=0, n_loss_tokens=0)
    return LossOutput(loss=per_tok[inputs.answer_mask].sum(), normalizer=float(n), n_loss_tokens=n)
