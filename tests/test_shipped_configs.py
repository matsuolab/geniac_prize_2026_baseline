"""The run specs and per-task CaP-X configs shipped in configs/."""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from geniac_baseline.collect.plan import _segments, load_spec
from geniac_baseline.collect.preflight import check_collect_config, load_yaml

REPO = Path(__file__).resolve().parents[1]
SPECS = sorted(p for p in (REPO / "configs").glob("*.yaml") if not p.name.startswith("train"))
# APIs registered in geniac_prize_2026_code (capx/integrations/__init__.py) that the fixed configs use
CAPX_GENIAC_APIS = {
    "FrankaControlApiReducedSkillLibrary",
    "FrankaControlApiReducedSkillLibraryBimanual",
    "FrankaControlApiReducedSkillLibraryBimanualHandover",
    "FrankaControlApiReducedSkillLibrarySpillWipe",
    "FrankaLiberoApiReducedSkillLibrary",
}


@pytest.mark.parametrize("spec_path", SPECS, ids=lambda p: p.name)
def test_spec_loads_and_seeds_follow_the_protocol(spec_path: Path) -> None:
    spec = load_spec(spec_path)
    _segments(spec)  # collection 301-400 off the evaluation layouts; evaluation seeds only
    for t in spec.tasks:
        assert t.config.is_file(), t.config
        if spec.mode == "collect":
            check_collect_config(load_yaml(t.config), t.config)


@pytest.mark.parametrize(
    "cfg", sorted((REPO / "configs" / "capx" / "geniac_prize").rglob("*.yaml")), ids=lambda p: p.stem
)
def test_fixed_config_is_the_aspire_config_mapped(cfg: Path) -> None:
    src = REPO / "configs" / "capx" / "aspire" / cfg.parent.name / cfg.name
    fixed = yaml.safe_load(cfg.read_text())
    aspire = yaml.safe_load(src.read_text())
    apis = fixed["env"]["cfg"]["apis"]
    assert set(apis) <= CAPX_GENIAC_APIS
    assert [a + "Traced" for a in apis] == aspire["env"]["cfg"]["apis"]
    mapped = re.sub(r"(- Franka\w+?)Traced\b", r"\1", src.read_text().replace("aspire.sim.cap.", "capx."))
    assert cfg.read_text().split("\n", 2)[2] == mapped
