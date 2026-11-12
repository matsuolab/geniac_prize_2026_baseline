"""Checks that run before a collection / evaluation script is written.

Each check guards a failure that actually happened or that the source comparison showed would happen silently:

* ``check_visual_models``: in a tree whose ``needs_visual`` (``capx/envs/trial.py``) reads ``VLM_MODELS``, CaP-X
  only captures camera frames (and so only produces the VDM text of the decision prompt) when the model is in that
  list; otherwise the multi-turn loop would run with no visual feedback and nothing in the log would say so. The
  ASPIRE-vendored copy works this way (Qwen3.8-27B is in its list); geniac_prize_2026_code decides from
  the config alone, so the check has nothing to do there.
* ``check_port_env``: moving the perception servers to other ports only works if the CaP-X client modules read
  their URL from the environment; in unpatched geniac_prize_2026_code SAM3 / PyRoKi / SAM2 / OWL-ViT URLs
  are hard-coded.
* ``check_patched``: the run exports ``CAPX_TRIAL_TIMEOUT_S`` and collection needs the VDM text files; an
  unpatched geniac_prize_2026_code would ignore the first and not write the second, both silently.
* ``check_skill_library``: the baseline's LIBERO code calls ASPIRE's 14 added skills; a tree without them
  (geniac_prize_2026_code without ``skill-library.patch``) fails those calls inside every trial, and the run
  goes on with lower success and data that names skills the tree does not have.
* ``check_seeds``: collection seeds must stay out of the evaluation band; evaluation seeds must be in it.
* ``check_collect_config``: collection needs ``save_multiturn_prompts: true`` or no decision prompt is saved.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from geniac_baseline.data.split import (
    DEFAULT_RULES,
    LayoutError,
    SplitRules,
    check_layouts,
    eval_layouts,
    is_libero,
    libero_layout,
)

# Robosuite evaluation seeds of the protocol.
ROBOSUITE_EVAL_SEEDS = range(401, 451)


class PreflightError(RuntimeError):
    """A run would not do what it claims; nothing was written."""


def package_dir(capx_root: Path, launch_script: str) -> Path:
    """The CaP-X package directory, derived from the launch script (``<pkg>/envs/launch.py``)."""
    p = capx_root / launch_script
    if not p.is_file():
        raise PreflightError(f"launch script not found: {p}")
    if p.parent.name != "envs":
        raise PreflightError(f"expected the launch script at <package>/envs/launch.py, got {launch_script}")
    return p.parent.parent


def read_vlm_models(pkg: Path) -> list[str]:
    """``VLM_MODELS`` of a CaP-X checkout, read with ``ast`` from ``<pkg>/llm/client.py`` (nothing is imported)."""
    src = (pkg / "llm" / "client.py").read_text()
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "VLM_MODELS" for t in node.targets):
            return [str(v) for v in ast.literal_eval(node.value)]
    raise PreflightError(f"no VLM_MODELS list in {pkg / 'llm' / 'client.py'}")


def load_yaml(path: Path) -> dict[str, Any]:
    d = yaml.safe_load(path.read_text())
    if not isinstance(d, dict):
        raise PreflightError(f"{path}: not a YAML mapping")
    return d


def check_visual_models(pkg: Path, config: dict[str, Any], code_model: str, vdm_model: str) -> None:
    """Mirror of ``needs_visual`` in ``capx/envs/trial.py``; error if a visual feature is on but would not run."""
    if "VLM_MODELS" not in (pkg / "envs" / "trial.py").read_text():
        return  # this tree turns the visual features on from the config alone
    vlm = set(read_vlm_models(pkg))
    wants_feedback = bool(config.get("use_visual_feedback", False))
    wants_diff = bool(config.get("use_img_differencing", False))
    needs_visual = (wants_feedback and code_model in vlm) or (wants_diff and vdm_model in vlm)
    if (wants_feedback or wants_diff) and not needs_visual:
        raise PreflightError(
            f"the config turns on visual feedback/differencing, but neither the code model {code_model!r} nor the "
            f"VDM model {vdm_model!r} is in VLM_MODELS of {pkg / 'llm' / 'client.py'}; CaP-X would silently skip "
            "every image and every VDM text (capx/envs/trial.py, needs_visual). Add the model to VLM_MODELS in "
            "that checkout, or use the checkout the data was collected with."
        )


def check_patched(pkg: Path, mode: str) -> None:
    """Error if the checkout lacks a change the run depends on (patches/geniac_prize/, or the ASPIRE tree's own)."""
    runner = (pkg / "envs" / "runner.py").read_text()
    if "CAPX_TRIAL_TIMEOUT_S" not in runner and "ASPIRE_TRIAL_TIMEOUT_S" not in runner:
        raise PreflightError(
            f"{pkg / 'envs' / 'runner.py'} does not read CAPX_TRIAL_TIMEOUT_S, so trial_timeout_s would be ignored "
            "and trials cut at CaP-X's own limit. Build the tree with bash patches/geniac_prize/setup.sh "
            "(docs/setup.md)."
        )
    if mode == "collect" and "differencing_feedback" not in (pkg / "utils" / "launch_utils.py").read_text():
        raise PreflightError(
            f"{pkg / 'utils' / 'launch_utils.py'} does not write differencing_feedback_XX.txt, so convert rows "
            "could not rebuild the FINISH rows of this collection. Build the tree with "
            "bash patches/geniac_prize/setup.sh (docs/setup.md)."
        )


# The skills ASPIRE added to FrankaLiberoApiReducedSkillLibrary (patches/geniac_prize/skill-library.patch).
LIBERO_ADDED_SKILLS = (
    "goto_pose_settled",
    "point_at_object",
    "locate_object_points",
    "top_down_grasp_for",
    "bowl_rim_grasp_for",
    "place_pose_on_top_of",
    "bowl_place_pose_on",
    "survey_from_above",
    "yaw_quat",
    "open_drawer_by_handle",
    "place_pose_in_container",
    "wine_rack_place_pose",
    "turn_stove_knob",
    "refine_bowl_centre_from_above",
)


def check_skill_library(pkg: Path, config: dict[str, Any], path: Path) -> None:
    """Error if the config uses the LIBERO skill-library API but the checkout lacks ASPIRE's added skills."""
    apis = ((config.get("env") or {}).get("cfg") or {}).get("apis") or []
    if not any(str(a).startswith("FrankaLiberoApiReducedSkillLibrary") for a in apis):
        return
    lib = pkg / "integrations" / "franka" / "libero_reduced_skill_library.py"
    src = lib.read_text() if lib.is_file() else ""
    missing = [n for n in LIBERO_ADDED_SKILLS if f'fns["{n}"]' not in src]
    if missing:
        raise PreflightError(
            f"{path} uses FrankaLiberoApiReducedSkillLibrary, but {lib} does not register {', '.join(missing)}; "
            "the baseline's LIBERO code calls them. Build the tree with bash patches/geniac_prize/setup.sh "
            "(docs/setup.md)."
        )


# launcher -> (client module, URL variable, default port of that client in capx/integrations/*)
_URL_ENV = {
    "launch_sam3_server": ("vision/sam3.py", "SAM3_SERVICE_URL", 8114),
    "launch_sam2_server": ("vision/sam2.py", "SAM2_SERVICE_URL", 8113),
    "launch_contact_graspnet_server": ("vision/graspnet.py", "GRASPNET_SERVICE_URL", 8115),
    "launch_pyroki_server": ("motion/pyroki.py", "PYROKI_SERVICE_URL", 8116),
    "launch_owlvit_server": ("vision/owlvit.py", "OWLVIT_SERVICE_URL", 8117),
}


def check_port_env(pkg: Path, config: dict[str, Any], port_offset: int, molmo_port_moved: bool) -> None:
    """Only allow moved ports when the client side of that server reads its URL from the environment."""
    problems = []
    for srv in config.get("api_servers") or []:
        target = str(srv.get("_target_", ""))
        for key, (rel, env, default) in _URL_ENV.items():
            if key not in target:
                continue
            port = int(srv.get("port", default))
            effective = port + port_offset if 8113 <= port <= 8117 else port  # the rewrite moves only 8113-8117
            if effective != default + port_offset:
                raise PreflightError(
                    f"{target} listens on {effective} after the rewrite, but its client expects "
                    f"{default + port_offset}; use a config with CaP-X's default ports (8113-8117) and set port_offset"
                )
            if port_offset and env not in (pkg / "integrations" / rel).read_text():
                problems.append(f"{rel} does not read {env}")
    if molmo_port_moved and "CAPX_MOLMO_URL" not in (pkg / "integrations" / "vision" / "molmo.py").read_text():
        problems.append("vision/molmo.py does not read CAPX_MOLMO_URL")
    if problems:
        raise PreflightError(
            "ports were moved, but this CaP-X checkout hard-codes the client URLs (" + "; ".join(problems) + "); "
            "the clients would talk to the default ports. Run with port offsets 0."
        )


def parse_segment(spec: str) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)-(\d+)", spec)
    if not m or int(m[1]) > int(m[2]):
        raise PreflightError(f"bad seed range {spec!r} (expected a-b with a <= b)")
    return int(m[1]), int(m[2])


def check_seeds(
    mode: str, task: str, seeds: Iterable[int], n_layouts: int | None = None, rules: SplitRules = DEFAULT_RULES
) -> None:
    """collect: every seed in 301..400, and LIBERO-PRO seeds never on the task's evaluation layouts.
    eval: Robosuite 401-450; LIBERO-PRO 401-650 on the evaluation layouts only (``data/split.py``)."""
    if is_libero(task):
        try:
            n = check_layouts(task, n_layouts, rules)
        except LayoutError as e:
            raise PreflightError(f"{e} (set 'layouts' for this task)") from None
        reserved = eval_layouts(n, rules)
    elif n_layouts is not None:
        raise PreflightError(f"{task}: 'layouts' is only for LIBERO-PRO tasks")
    for s in seeds:
        if mode == "collect":
            if s < rules.collect_seed_min or s >= rules.eval_seed_min:
                raise PreflightError(
                    f"{task}: collection seed {s} is outside {rules.collect_seed_min}..{rules.eval_seed_min - 1}"
                )
            if is_libero(task) and libero_layout(s, n) in reserved:
                raise PreflightError(f"{task}: collection seed {s} maps to evaluation layout {libero_layout(s, n)}")
        elif mode == "eval":
            if is_libero(task):
                ok = rules.eval_seed_min <= s <= rules.libero_eval_seed_max and libero_layout(s, n) in reserved
            else:
                ok = s in ROBOSUITE_EVAL_SEEDS
            if not ok:
                raise PreflightError(f"{task}: seed {s} is not an evaluation seed of the protocol")
        else:
            raise ValueError(mode)


def check_collect_config(config: dict[str, Any], path: Path) -> None:
    if config.get("save_multiturn_prompts") is not True:
        raise PreflightError(
            f"{path}: collection needs 'save_multiturn_prompts: true' (otherwise CaP-X stores no decision prompt, "
            "capx/envs/trial.py, and the converter has nothing to train the decisions on)"
        )
