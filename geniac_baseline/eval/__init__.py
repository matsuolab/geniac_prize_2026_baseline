"""Evaluation of the base model or base + LoRA with CaP-X, and scoring.

``eval plan`` shares the run-spec machinery of ``geniac_baseline.collect`` with mode ``eval``: the VDM server keeps
the collection settings, the code writer gets its own server (thinking off, optional LoRA), seeds must be
evaluation seeds. ``eval score`` reads trial folder names (success = taskcompleted_1 and sandboxrc_0).
"""

from geniac_baseline.eval.score import score_runs, score_trial

__all__ = ["score_runs", "score_trial"]
