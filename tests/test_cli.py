"""The command line wires the stages together."""

from __future__ import annotations

from pathlib import Path

import pytest

from geniac_baseline.cli import main
from geniac_baseline.data.tag import read_jsonl


def test_convert_pipeline_via_cli(real_fixture: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["convert", "rows", "--sources", str(real_fixture / "sources.yaml"), "--check-only"]) == 0
    assert "saved_rebuild_exact" in capsys.readouterr().out
    assert main(["convert", "rows", "--sources", str(real_fixture / "sources.yaml"), "--out", str(tmp_path / "r")]) == 0
    assert main(["convert", "split", "--src", str(tmp_path / "r"), "--out", str(tmp_path / "s")]) == 0
    assert len(read_jsonl(tmp_path / "s" / "sft.jsonl")) == 10  # seeds 302-307 are Robosuite train seeds
    assert main(["convert", "validate", str(tmp_path / "s" / "sft.jsonl")]) == 0


def test_rows_cli_reports_a_failed_check(real_fixture: Path, tmp_path: Path) -> None:
    for f in (real_fixture / "trials").rglob("differencing_feedback_01.txt"):
        f.unlink()
    assert main(["convert", "rows", "--sources", str(real_fixture / "sources.yaml"), "--out", str(tmp_path / "r")]) == 1
    assert not (tmp_path / "r").exists()


def test_help() -> None:
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0


def test_example_scripts_parse() -> None:
    import re
    import shutil
    import subprocess

    repo = Path(__file__).resolve().parents[1]
    scripts = sorted((repo / "examples").glob("*.sh"))
    assert [s.name for s in scripts] == [
        "00_collect_data.sh",
        "01_eval_base_cap_agent0.sh",
        "02_train_sft.sh",
        "03_build_pairs.sh",
        "04_train_dmpo.sh",
        "05_eval_trained_cap_agent0.sh",
    ]
    for s in scripts:
        for cfg in re.findall(r"configs/[\w.]+\.yaml", s.read_text()):
            assert (repo / cfg).is_file(), (s.name, cfg)
        if shutil.which("bash"):
            subprocess.run(["bash", "-n", str(s)], check=True)


def test_doc_links_resolve() -> None:
    import re

    repo = Path(__file__).resolve().parents[1]
    for md in [repo / "README.md", *sorted((repo / "docs").rglob("*.md"))]:
        for link in re.findall(r"\]\(((?!https?://)[^)#]+)\)", md.read_text()):
            assert (md.parent / link).exists(), (md.name, link)


def test_user_errors_print_one_line(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    spec = tmp_path / "spec.yaml"
    spec.write_text("mode: collect\n")
    assert main(["collect", "plan", "--spec", str(spec)]) == 1
    err = capsys.readouterr().err
    assert "error: SpecError" in err and "docs/collect.md" in err and "Traceback" not in err


def test_example_commands_match_the_cli() -> None:
    import re
    import shlex

    from geniac_baseline.cli import build_parser

    repo = Path(__file__).resolve().parents[1]
    n = 0
    for s in sorted((repo / "examples").glob("*.sh")):
        for line in s.read_text().splitlines():
            m = re.search(r'"\$GB" ([^|)]*)', line)
            if not m or line.lstrip().startswith(("#", "GB=")):
                continue
            argv = shlex.split(re.sub(r"\$\{?\w+\}?", "x", m.group(1)))
            build_parser().parse_args(argv)  # SystemExit on an unknown flag or subcommand
            n += 1
    assert n >= 15
