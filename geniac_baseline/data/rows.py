"""CaP-X multi-turn trials -> one training row per model call.

Row layout (schema in ``geniac_baseline.data.schema``; example in docs/data_format.md)::

    turn 0   user = the initial prompt the model saw        -> assistant = the first program
    turn k   user = the decision prompt the model saw at k  -> assistant = "FINISH" or
                                                               "REGENERATE\\n```python\\n...\\n```"

Decision prompts come from the saved ``multi_turn_prompt`` of each REGENERATE call (CaP-X stores it in
``all_responses.json`` when the config sets ``save_multiturn_prompts: true``). CaP-X never stores the prompt of
a FINISH call, so that one is rebuilt with the rule in ``geniac_baseline.data.prompts``:

    PREFIX(task)   everything before "The following code blocks were just executed:" in any saved prompt of the
                   task (exactly one variant per task, or the check fails)
    TEMPLATE       multi_turn_prompt from the run's resolved config
    executed_code  all code blocks executed so far, joined with "\\n"
    stdout         summary.txt "Stdout:" (the last executed block's own stdout)
    stderr         summary.txt "Stderr:" is the JOIN of every non-empty stderr so far, so the last turn's stderr is
                   whatever that join adds beyond the earlier turns' stderr ("" if nothing)
    vdm_file_text  differencing_feedback_{k-1:02d}.txt in the trial folder, verbatim
    TAIL           the fixed instruction after the VDM text (exactly one variant, or the check fails)

Before any row is written, every saved prompt of every input trial is rebuilt from those raw ingredients and
compared byte for byte. One mismatch and nothing is written (``RebuildCheckError``).

Ported from the conversion script that produced the baseline training data (it reported 1511/1511 saved prompts
rebuilt exactly on the full store); the behaviour is unchanged except where marked "stricter".
"""

from __future__ import annotations

import collections
import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from geniac_baseline.data.prompts import build, split_saved, summary_feedback, text_of, yaml_template
from geniac_baseline.data.trials import TrialRef

# metadata.source of every row this module writes.
SOURCE_TAG = "capx_multiturn"
# The rows are rendered with thinking off at training time (the team trainer's convention); recorded in the
# manifest so the trainer reads it from the data instead of assuming it.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False}

_NO_VDM_HINT = (
    "A saved decision prompt has no matching differencing_feedback_XX.txt in its trial folder. The FINISH-turn "
    "rebuild needs that file. The ASPIRE-vendored CaP-X used for the baseline data and geniac_prize_2026_code "
    "with patches/geniac_prize/geniac-prize.patch write it (docs/setup.md); unpatched geniac_prize_2026_code "
    "writes neither the VDM text nor the FINISH-turn prompt. Rows are not written, because FINISH rows cannot be "
    "rebuilt exactly from this data."
)


class RebuildCheckError(RuntimeError):
    """The prompt-rebuild rule does not hold on the input data; no rows were written."""

    def __init__(self, message: str, report: dict[str, Any]) -> None:
        super().__init__(message)
        self.report = report


@dataclass
class RebuildRule:
    """What the check learned from the saved prompts and needs for rebuilding FINISH prompts."""

    prefix: dict[str, str]
    vdm_header: str
    tail: str
    counts: dict[str, int] = field(default_factory=dict)


def _calls(t: TrialRef) -> list[dict[str, Any]]:
    calls = json.loads((t.trial_dir / "all_responses.json").read_text())
    if not isinstance(calls, list) or not calls:
        raise ValueError(f"{t.trial_dir}: all_responses.json is not a non-empty list")
    return calls


def _blocks(calls: list[dict[str, Any]]) -> list[str]:
    # as recorded: the rule check needs the exact text
    return ["\n".join(c.get("code_blocks") or []) for c in calls]


def check_rebuild_rule(trials: list[TrialRef]) -> RebuildRule:
    """Rebuild every saved decision prompt of ``trials`` from raw ingredients; raise if any one differs."""
    prefixes: dict[str, set[str]] = collections.defaultdict(set)
    headers: set[str] = set()
    tails: set[str] = set()
    stats: collections.Counter[str] = collections.Counter()
    tpl_cache: dict[Path, str] = {}
    first_mismatch: str | None = None
    for t in trials:
        calls = _calls(t)
        tpl = tpl_cache.setdefault(t.config, yaml_template(t.config))
        blocks = _blocks(calls)
        for k, c in enumerate(calls):
            p = c.get("multi_turn_prompt")
            if not p:
                continue
            if k == 0:
                raise RebuildCheckError(f"{t.trial_dir}: call 0 carries a multi_turn_prompt", {})
            user = text_of(p[-1]["content"])
            try:
                sp = split_saved(user)
            except ValueError:  # stricter: a prompt without the template markers cannot be checked
                stats["saved_unparseable"] += 1
                continue
            prefixes[t.task].add(sp["prefix"])
            vdm = sp["vdm"]
            if not vdm or "\n" not in vdm:  # stricter: a header without a body counts as a missing VDM text
                stats["saved_without_matching_vdm_file"] += 1
                continue
            vh, vdm_body = vdm.split("\n", 1)
            headers.add(vh)
            dfp = t.trial_dir / f"differencing_feedback_{k - 1:02d}.txt"
            if not dfp.exists() or not vdm_body.startswith(dfp.read_text()):
                stats["saved_without_matching_vdm_file"] += 1
                continue
            vfile = dfp.read_text()
            tails.add(vdm_body[len(vfile) :])
            executed = "\n".join(b for b in blocks[:k] if b)
            rebuilt = build(sp["prefix"], tpl, executed, sp["stdout"], sp["stderr"], vh, vfile + vdm_body[len(vfile) :])
            if rebuilt == user:
                stats["saved_rebuild_exact"] += 1
            else:
                stats["saved_rebuild_MISMATCH"] += 1
                first_mismatch = first_mismatch or f"{t.trial_dir} call {k}"
    bad_prefix = {task: len(v) for task, v in prefixes.items() if len(v) != 1}
    report: dict[str, Any] = {
        "counts": dict(stats),
        "tasks_with_not_exactly_one_prefix": bad_prefix,
        "vdm_headers": len(headers),
        "tails_after_vdm": len(tails),
        "first_mismatch": first_mismatch,
    }
    problems = []
    if stats["saved_rebuild_MISMATCH"]:
        problems.append(
            f"{stats['saved_rebuild_MISMATCH']} saved prompt(s) do not rebuild exactly (first: {first_mismatch})"
        )
    if stats["saved_unparseable"]:
        problems.append(f"{stats['saved_unparseable']} saved prompt(s) without the multi_turn_prompt markers")
    if stats["saved_without_matching_vdm_file"]:
        problems.append(
            f"{stats['saved_without_matching_vdm_file']} saved prompt(s) without a matching VDM file. {_NO_VDM_HINT}"
        )
    if bad_prefix:
        problems.append(f"tasks without exactly one prompt prefix: {bad_prefix}")
    if len(headers) != 1 or len(tails) != 1:
        problems.append(
            f"expected exactly one VDM header and one tail, found {len(headers)} and {len(tails)}"
            + (" (no saved decision prompt to learn the rule from)" if not headers else "")
        )
    if problems:
        raise RebuildCheckError("rebuild check FAILED; refusing to write rows: " + "; ".join(problems), report)
    return RebuildRule(
        prefix={task: next(iter(v)) for task, v in prefixes.items()},
        vdm_header=next(iter(headers)),
        tail=next(iter(tails)),
        counts=dict(stats),
    )


def iter_rows(
    trials: list[TrialRef], rule: RebuildRule, counts: collections.Counter[tuple[str, str, str | None]]
) -> Iterator[dict[str, Any]]:
    """Yield the rows of ``trials`` (call ``check_rebuild_rule`` first); ``counts`` records rows and skips."""
    tpl_cache: dict[Path, str] = {}
    for t in trials:
        calls = _calls(t)
        tpl = tpl_cache.setdefault(t.config, yaml_template(t.config))
        init = calls[0]["initial_prompt"]
        system = text_of(init[0]["content"])
        blocks = _blocks(calls)
        decisions = [c.get("decision") for c in calls]
        base = {
            "source": SOURCE_TAG,
            "sample_id": t.sample_id,
            "task": t.task,
            "condition": t.task,
            "seed": t.seed,
            "run_tag": t.run_tag,
            "n_llm_calls": len(calls),
            "success": t.outcome == "success",
        }
        if t.n_layouts is not None:
            base["n_layouts"] = t.n_layouts

        def row(
            user: str,
            answer: str,
            turn: int,
            kind: str,
            feedback: str | None,
            t: TrialRef = t,
            system: str = system,
            base: dict[str, Any] = base,
        ) -> dict[str, Any]:
            counts[(kind, "row", feedback)] += 1
            return {
                "id": f"{t.sample_id}-t{turn}",
                "messages": [
                    {"role": "system", "content": [{"type": "text", "text": system}]},
                    {"role": "user", "content": [{"type": "text", "text": user}]},
                    {"role": "assistant", "content": [{"type": "text", "text": answer}]},
                ],
                "metadata": {**base, "turn": turn, "kind": kind, "feedback": feedback},
            }

        if blocks[0]:
            yield row(text_of(init[-1]["content"]), blocks[0], 0, "first_turn", None)
        stderr_seen: list[str] = []  # non-empty stderr of the turns whose prompt we know
        prompt_missing = False
        for k in range(1, len(calls)):
            d = decisions[k]
            p = calls[k].get("multi_turn_prompt")
            if d == "regenerate" and p:
                user = text_of(p[-1]["content"])
                sp = split_saved(user)
                if sp["stderr"]:
                    stderr_seen.append(sp["stderr"])
                yield row(user, f"REGENERATE\n```python\n{blocks[k]}\n```", k, "decision", "saved")
            elif d == "finish":
                if prompt_missing:  # stricter: the stderr split below needs every earlier prompt
                    counts[("decision", "skip_finish", "earlier_prompt_missing")] += 1
                    continue
                fb = summary_feedback((t.trial_dir / "summary.txt").read_text(errors="replace"))
                if fb is None:
                    counts[("decision", "skip_finish", "no_summary")] += 1
                    continue
                stdout, joined = fb
                earlier = "\n\n".join(stderr_seen)
                if joined == earlier:
                    stderr = ""
                elif earlier and joined.startswith(earlier + "\n\n"):
                    stderr = joined[len(earlier) + 2 :]
                elif not earlier:
                    stderr = joined
                else:
                    counts[("decision", "skip_finish", "stderr_ambiguous")] += 1
                    continue
                dfp = t.trial_dir / f"differencing_feedback_{k - 1:02d}.txt"
                if not dfp.exists():  # the rule is only verified for turns with a VDM text
                    counts[("decision", "skip_finish", "no_vdm_file")] += 1
                    continue
                if t.task not in rule.prefix:
                    counts[("decision", "skip_finish", "no_prefix_for_task")] += 1
                    continue
                executed = "\n".join(b for b in blocks[:k] if b)
                user = build(
                    rule.prefix[t.task], tpl, executed, stdout, stderr, rule.vdm_header, dfp.read_text() + rule.tail
                )
                yield row(user, "FINISH", k, "decision", "rebuilt")
            else:
                prompt_missing = True
                counts[("decision", f"skip_{d}", "no_saved_prompt")] += 1


def write_rows(
    trials: list[TrialRef],
    out_dir: Path,
    sources_file: Path | None = None,
    check_trials: list[TrialRef] | None = None,
) -> dict[str, Any]:
    """Check the rebuild rule, then write the rows of ``trials`` (``rows.jsonl`` + ``manifest.json``) into a new
    ``out_dir``.

    ``check_trials`` (default: ``trials``) is the set the rule is learnt from and checked on. The CLI passes every
    success and failure trial of the sources, so a task whose failures never stored a decision prompt still gets its
    prefix from the successes, and every stored prompt of both sets is checked.
    """
    if out_dir.exists():
        raise FileExistsError(f"refusing to overwrite {out_dir}")
    if not trials:
        raise ValueError("no trials to convert")
    rule = check_rebuild_rule(check_trials if check_trials is not None else trials)  # raises before writing
    counts: collections.Counter[tuple[str, str, str | None]] = collections.Counter()
    out_dir.mkdir(parents=True)
    with (out_dir / "rows.jsonl").open("w") as f:
        for r in iter_rows(trials, rule, counts):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    manifest = {
        "chat_template_kwargs": CHAT_TEMPLATE_KWARGS,
        "stage": "rows",
        "sources_file": str(sources_file) if sources_file else None,
        "trials": len(trials),
        "trials_by_outcome": dict(collections.Counter(t.outcome for t in trials)),
        "files": {"rows": "rows.jsonl"},
        "counts": {"|".join(str(x) for x in k): v for k, v in sorted(counts.items(), key=str)},
        "rule_check": rule.counts,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    return manifest
