"""The two losses of the baseline pipeline.

``sft_ce``  completion-only cross-entropy on every answer token (SFT warm start, row-level).
``dmpo``    DMPO preference loss over whole trajectories (second stage, pair-level).
"""
