"""CaP-X trial folders -> training rows.

Pipeline (each stage writes a new directory and refuses to overwrite one):

    sources.yaml --rows--> rows.jsonl --tag--> rows.jsonl (+ code_outcome) --split--> sft.jsonl / val.jsonl

``rows`` refuses to write anything unless every saved decision prompt rebuilds exactly (``RebuildCheckError``).
"""

from geniac_baseline.data.rows import RebuildCheckError, check_rebuild_rule, iter_rows, write_rows
from geniac_baseline.data.schema import SchemaError, validate_row, validate_rows
from geniac_baseline.data.split import (
    DEFAULT_RULES,
    EvalSeedError,
    LayoutError,
    SplitRules,
    assign_split,
    split_rows,
)
from geniac_baseline.data.tag import tag_rows
from geniac_baseline.data.trials import Source, SourcesError, TrialRef, discover_trials, load_sources

__all__ = [
    "DEFAULT_RULES",
    "EvalSeedError",
    "LayoutError",
    "RebuildCheckError",
    "SchemaError",
    "Source",
    "SourcesError",
    "SplitRules",
    "TrialRef",
    "assign_split",
    "check_rebuild_rule",
    "discover_trials",
    "iter_rows",
    "load_sources",
    "split_rows",
    "tag_rows",
    "validate_row",
    "validate_rows",
    "write_rows",
]
