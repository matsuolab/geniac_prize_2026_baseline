"""geniac-baseline: data conversion, LoRA training and CaP-X run wrappers for the GENIAC CaP competition.

Subpackages:
    data     CaP-X trial folders -> one JSONL row per model call (with an exact prompt-rebuild check),
             code-outcome tags, train/val split, success / failure trajectory pairs.
    trainer  LoRA training: SFT warm start (completion-only cross-entropy), then DMPO on top of the SFT adapter.
    collect  Command builders and preflight checks for running CaP-X data collection.
    eval     Command builders for evaluating a base model or base + LoRA with CaP-X, and trial scoring.

See docs/data_format.md for the data formats. CaP-X-facing behaviour cites the CaP-X file and line where it is used.
"""

__version__ = "0.1.0.dev0"
