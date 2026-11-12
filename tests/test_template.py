import json

import pytest

from geniac_baseline.collect import template as T

CHECK = (
    "                {%- if is_system_content %}\n"
    "                    {{- raise_exception('System message cannot contain {k}.') }}\n"
    "                {%- endif %}\n"
)
MODEL = (
    "{%- for item in content %}\n"
    + CHECK.replace("{k}", "images")
    + "  image\n"
    + CHECK.replace("{k}", "videos")
    + "  video\n"
    "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}\n"
)


def test_derive_edits():
    out = T.derive(MODEL)
    assert "default('low')" in out and "xhigh" not in out
    assert "raise_exception" not in out
    assert out == "{%- for item in content %}\n  image\n  video\n" + (
        "{%- set resolved_reasoning_effort = reasoning_effort|default('low') %}\n"
    )


def test_derive_refuses_other_template():
    with pytest.raises(T.TemplateError):
        T.derive("{{ messages }}")


def test_reads_tokenizer_config(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({"chat_template": MODEL}))
    assert T.read_model_template(tmp_path) == MODEL


def test_hash_mismatch_refused(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(MODEL)
    import argparse

    a = argparse.Namespace(model_dir=tmp_path, out=tmp_path / "t.jinja", allow_mismatch=False)
    with pytest.raises(T.TemplateError):
        T._run(a)
    a.allow_mismatch = True
    T._run(a)
    assert (tmp_path / "t.jinja").read_text() == T.derive(MODEL)
