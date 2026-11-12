"""``geniac-baseline template --model-dir SNAPSHOT --out FILE``: the low-effort Qwen3.8 chat template.

The baseline servers used the model's own chat template with two edits (``settings.CHAT_TEMPLATE_SHA256``):

* the default reasoning effort is ``low`` instead of ``xhigh``;
* the two checks that refuse images / videos inside a system message are removed.

The template itself is not redistributed here; this derives it from the snapshot (``chat_template.jinja``, or
``chat_template`` in ``tokenizer_config.json``) and checks the result against the pinned hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from geniac_baseline.collect import settings as S

_EFFORT = ("reasoning_effort|default('xhigh')", "reasoning_effort|default('low')")
_SYSTEM_CHECKS = tuple(
    "                {%- if is_system_content %}\n"
    f"                    {{{{- raise_exception('System message cannot contain {kind}.') }}}}\n"
    "                {%- endif %}\n"
    for kind in ("images", "videos")
)


class TemplateError(ValueError):
    """The snapshot's template is not the one the baseline template was derived from."""


def read_model_template(model_dir: Path) -> str:
    f = model_dir / "chat_template.jinja"
    if f.is_file():
        return f.read_text()
    tc = model_dir / "tokenizer_config.json"
    if tc.is_file():
        t = json.loads(tc.read_text()).get("chat_template")
        if isinstance(t, str):
            return t
    raise TemplateError(f"no chat_template.jinja or tokenizer_config.json chat_template in {model_dir}")


def derive(text: str) -> str:
    if text.count(_EFFORT[0]) != 1:
        raise TemplateError(f"expected one {_EFFORT[0]!r} in the model template")
    text = text.replace(*_EFFORT)
    for block in _SYSTEM_CHECKS:
        if text.count(block) != 1:
            raise TemplateError("the system-message image/video checks are not where the derivation expects them")
        text = text.replace(block, "")
    return text


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _run(a: argparse.Namespace) -> int:
    out = derive(read_model_template(a.model_dir))
    h = sha256(out)
    if h != S.CHAT_TEMPLATE_SHA256 and not a.allow_mismatch:
        raise TemplateError(
            f"derived template sha256 {h} != baseline {S.CHAT_TEMPLATE_SHA256}: the snapshot differs from the one "
            "the baseline used (Qwen/Qwen3.8-27B revision 1d4bf0f). Use that revision, or --allow-mismatch."
        )
    if a.out.exists():
        raise FileExistsError(f"refusing to overwrite {a.out}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out)
    print(f"[template] wrote {a.out} (sha256 {h}{'' if h == S.CHAT_TEMPLATE_SHA256 else ', NOT the baseline hash'})")
    return 0


def add_parser(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("template", help="write the low-effort Qwen3.8 chat template the baseline servers used")
    p.add_argument("--model-dir", type=Path, required=True, help="local Qwen/Qwen3.8-27B snapshot directory")
    p.add_argument("--out", type=Path, required=True, help="output .jinja path (the run spec's chat_template)")
    p.add_argument("--allow-mismatch", action="store_true", help="write even if the hash differs from the baseline")
    p.set_defaults(func=_run)
