"""Collection / evaluation plans (preflight, generated script, config copies) and scoring."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from geniac_baseline.collect import PreflightError, load_spec, write_plan
from geniac_baseline.collect.preflight import LIBERO_ADDED_SKILLS, check_skill_library, read_vlm_models
from geniac_baseline.collect.run_config import RunConfigError, rewrite_config_text
from geniac_baseline.data.prompts import yaml_template
from geniac_baseline.eval.score import score_runs

from .conftest import REAL, make_trial

REPO = Path(__file__).resolve().parents[1]


def fake_checkout(
    root: Path, vlm: list[str], url_env: bool = False, vlm_gate: bool = True, patched: bool = True
) -> Path:
    """A directory shaped like a CaP-X checkout, with just the files the preflight reads.

    ``vlm_gate``: ``needs_visual`` in trial.py reads ``VLM_MODELS`` (ASPIRE) or the config alone (geniac_prize).
    ``patched``: runner.py reads CAPX_TRIAL_TIMEOUT_S and launch_utils.py writes the VDM text files.
    """
    pkg = root / "capx"
    for rel in ("envs", "llm", "utils", "integrations/vision", "integrations/motion"):
        (pkg / rel).mkdir(parents=True, exist_ok=True)
    (pkg / "envs" / "launch.py").write_text("# launch\n")
    gate = "args.model in VLM_MODELS" if vlm_gate else 'config["use_visual_feedback"]'
    (pkg / "envs" / "trial.py").write_text(f"needs_visual = {gate}\n")
    (pkg / "envs" / "runner.py").write_text("os.environ.get('CAPX_TRIAL_TIMEOUT_S')\n" if patched else "1000\n")
    (pkg / "utils" / "launch_utils.py").write_text("differencing_feedback_\n" if patched else "\n")
    (pkg / "llm" / "client.py").write_text(f"GPT_MODELS = []\nVLM_MODELS = {vlm!r}\n")
    env = (lambda n: f"SERVICE_URL = os.environ.get('{n}', 'x')\n") if url_env else (lambda n: "SERVICE_URL = 'x'\n")
    for f, n in (
        ("vision/sam3.py", "SAM3_SERVICE_URL"),
        ("vision/sam2.py", "SAM2_SERVICE_URL"),
        ("vision/graspnet.py", "GRASPNET_SERVICE_URL"),
        ("vision/owlvit.py", "OWLVIT_SERVICE_URL"),
        ("motion/pyroki.py", "PYROKI_SERVICE_URL"),
        ("vision/molmo.py", "CAPX_MOLMO_URL"),
    ):
        (pkg / "integrations" / f).write_text(env(n))
    return root


def default_port_config(tmp: Path) -> Path:
    """The real collection config with CaP-X's default perception ports (the run's copy had them moved)."""
    p = tmp / "cube_lifting_default_ports.yaml"
    if not p.exists():
        p.write_text((REAL / "cube_lifting.yaml").read_text().replace("port: 581", "port: 81"))
    return p


def write_spec(tmp: Path, **over) -> Path:
    cfg = str(default_port_config(tmp))
    tpl = tmp / "tpl.jinja"
    tpl.write_text("{{ messages }}")
    spec = {
        "mode": "collect",
        "capx_root": str(tmp / "capx_root"),
        "output_dir": str(tmp / "out"),
        "chat_template": str(tpl),
        "tasks": [{"task": "cube_lifting", "config": cfg, "seeds": ["301-340", "351-360"]}],
    }
    spec.update(over)
    p = tmp / "spec.yaml"
    p.write_text(yaml.safe_dump(spec))
    return p


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    return fake_checkout(tmp_path / "capx_root", ["Qwen/Qwen3.8-27B"])


def test_collect_plan(tmp_path: Path, checkout: Path) -> None:
    script = write_plan(load_spec(write_spec(tmp_path, b300_workarounds=True)))
    text = script.read_text()
    # one Qwen server with the collection settings, then Molmo; started one at a time
    assert text.count("vllm.entrypoints.openai.api_server") == 2
    for flag in (
        "--reasoning-parser qwen3",
        "--max-model-len 262144",
        "--gpu-memory-utilization 0.62",
        "--max-num-seqs 8",
        "--dtype bfloat16",
        "--chat-template",
        "--attention-backend TRITON_ATTN",
        "--max-tokens 12288",
        "--temperature 1.0",
        "--reasoning-effort low",
        "--visual-differencing-model Qwen/Qwen3.8-27B",
    ):
        assert flag in text, flag
    assert "enable_thinking" not in text
    assert "export CAPX_VLM_URL=http://127.0.0.1:8110/v1" in text and "export CAPX_VLM_MODEL=Qwen/Qwen3.8-27B" in text
    assert text.index("wait_up qwen") < text.index("starting molmo")
    # one config per segment, template byte-identical, a single resume_idx
    for start, end in ((301, 340), (351, 360)):
        cfg = tmp_path / "out" / f"seg_cube_lifting_{start}" / "cube_lifting.yaml"
        body = cfg.read_text()
        assert (
            f"\ntrials: {end}\n" in body and body.count("resume_idx:") == 1 and body.endswith(f"resume_idx: {start}\n")
        )
        assert yaml_template(cfg) == yaml_template(REAL / "cube_lifting.yaml")
    assert "port: 8114" in body
    src = yaml.safe_load((tmp_path / "out" / "sources.yaml").read_text())["sources"]
    assert [s["trials_dir"].endswith("outputs/Qwen_Qwen3.8-27B/cube_lifting") for s in src] == [True, True]
    if shutil.which("bash"):
        subprocess.run(["bash", "-n", str(script)], check=True)
    with pytest.raises(FileExistsError):
        write_plan(load_spec(write_spec(tmp_path)))


def test_vlm_membership_is_required(tmp_path: Path) -> None:
    fake_checkout(tmp_path / "capx_root", ["Qwen/Qwen3.6-27B"])
    with pytest.raises(PreflightError, match="VLM_MODELS"):
        write_plan(load_spec(write_spec(tmp_path)))
    assert not (tmp_path / "out").exists()


def test_config_gated_tree_skips_the_vlm_check(tmp_path: Path) -> None:
    fake_checkout(tmp_path / "capx_root", ["Qwen/Qwen3.6-27B"], vlm_gate=False)
    write_plan(load_spec(write_spec(tmp_path)))
    assert (tmp_path / "out").exists()


def test_libero_skill_library_needs_the_added_skills(tmp_path: Path) -> None:
    pkg = fake_checkout(tmp_path / "capx_root", ["Qwen/Qwen3.8-27B"]) / "capx"
    cfg = yaml.safe_load((REPO / "configs/capx/geniac_prize/libero/libero_goal_drawer.yaml").read_text())
    lib = pkg / "integrations" / "franka" / "libero_reduced_skill_library.py"
    lib.parent.mkdir(parents=True)
    lib.write_text('fns["select_top_down_grasp"] = self.select_top_down_grasp\n')
    with pytest.raises(PreflightError, match="goto_pose_settled"):
        check_skill_library(pkg, cfg, Path("libero_goal_drawer.yaml"))
    lib.write_text("".join(f'        fns["{n}"] = self.{n}\n' for n in LIBERO_ADDED_SKILLS))
    check_skill_library(pkg, cfg, Path("libero_goal_drawer.yaml"))
    # Robosuite configs use other APIs; nothing to check
    check_skill_library(pkg, yaml.safe_load((REAL / "cube_lifting.yaml").read_text()), Path("cube_lifting.yaml"))


def test_unpatched_tree_is_refused(tmp_path: Path) -> None:
    fake_checkout(tmp_path / "capx_root", ["Qwen/Qwen3.8-27B"], patched=False)
    with pytest.raises(PreflightError, match="CAPX_TRIAL_TIMEOUT_S"):
        write_plan(load_spec(write_spec(tmp_path)))
    assert not (tmp_path / "out").exists()


def test_read_vlm_models(checkout: Path) -> None:
    assert read_vlm_models(checkout / "capx") == ["Qwen/Qwen3.8-27B"]


def test_collect_needs_saved_prompts(tmp_path: Path, checkout: Path) -> None:
    cfg = tmp_path / "c.yaml"
    cfg.write_text(default_port_config(tmp_path).read_text().replace("save_multiturn_prompts: true", ""))
    spec = write_spec(tmp_path, tasks=[{"task": "cube_lifting", "config": str(cfg), "seeds": ["301-302"]}])
    with pytest.raises(PreflightError, match="save_multiturn_prompts"):
        write_plan(load_spec(spec))


@pytest.mark.parametrize(
    ("task", "seeds"),
    [("cube_lifting", "395-401"), ("libero_spatial_on_stove", "335-345"), ("cube_lifting", "0-3")],
)
def test_collect_seed_guard(tmp_path: Path, checkout: Path, task: str, seeds: str) -> None:
    t = {"task": task, "config": str(default_port_config(tmp_path)), "seeds": [seeds]}
    if task.startswith("libero_"):
        t["layouts"] = 50
    spec = write_spec(tmp_path, tasks=[t])
    with pytest.raises(PreflightError, match="seed"):
        write_plan(load_spec(spec))


def test_chat_template_is_required(tmp_path: Path, checkout: Path) -> None:
    spec = write_spec(tmp_path, chat_template=None)
    with pytest.raises(PreflightError, match="chat_template"):
        write_plan(load_spec(spec))


def test_config_ports_must_match_the_clients(tmp_path: Path, checkout: Path) -> None:
    spec = write_spec(
        tmp_path, tasks=[{"task": "cube_lifting", "config": str(REAL / "cube_lifting.yaml"), "seeds": ["301-302"]}]
    )
    with pytest.raises(PreflightError, match="client expects 8114"):
        write_plan(load_spec(spec))


def test_moved_ports_need_env_urls(tmp_path: Path, checkout: Path) -> None:
    with pytest.raises(PreflightError, match="hard-codes"):
        write_plan(load_spec(write_spec(tmp_path, port_offset=50000)))
    fake_checkout(tmp_path / "capx_root", ["Qwen/Qwen3.8-27B"], url_env=True)
    text = write_plan(load_spec(write_spec(tmp_path, port_offset=50000, vllm_port_offset=50000))).read_text()
    assert "SAM3_SERVICE_URL=http://127.0.0.1:58114" in text and "--port 58110" in text


def _eval_spec(tmp: Path, **over) -> Path:
    cfg = str(default_port_config(tmp))
    base = {
        "mode": "eval",
        "tasks": [
            {"task": "cube_lifting", "config": cfg, "seeds": ["401-450"]},
            {"task": "libero_spatial_on_stove", "config": cfg, "seeds": ["441-450", "641-650"], "layouts": 50},
        ],
    }
    base.update(over)
    return write_spec(tmp, **base)


def test_eval_plan_base_and_lora(tmp_path: Path, checkout: Path) -> None:
    text = write_plan(load_spec(_eval_spec(tmp_path))).read_text()
    vdm, code, molmo = [ln for ln in text.splitlines() if "api_server" in ln]
    assert "--reasoning-parser qwen3" in vdm and "--port 8110" in vdm and "--gpu-memory-utilization 0.3" in vdm
    assert "--reasoning-parser" not in code and "--port 8111" in code
    assert '{"enable_thinking": false}' in code and "--enable-lora" not in code
    assert "--server-url http://127.0.0.1:8111/v1/chat/completions" in text
    assert "--visual-differencing-model-server-url http://127.0.0.1:8110/v1/chat/completions" in text
    assert "Molmo2-8B" in molmo
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    out2 = tmp_path / "out2"
    text = write_plan(
        load_spec(_eval_spec(tmp_path, adapter=str(adapter), lora_name="my-lora", output_dir=str(out2)))
    ).read_text()
    code = [ln for ln in text.splitlines() if "api_server" in ln][1]
    assert f"--enable-lora --max-lora-rank 32 --max-loras 1 --lora-modules my-lora={adapter}" in code
    assert "--model my-lora" in text and "--visual-differencing-model Qwen/Qwen3.8-27B" in text
    assert not (out2 / "sources.yaml").exists()


@pytest.mark.parametrize(
    ("task", "seeds"),
    [
        ("cube_lifting", "400-410"),
        ("cube_lifting", "445-451"),
        ("libero_spatial_on_stove", "401-410"),
        ("libero_spatial_on_stove", "651-660"),
        ("libero_spatial_on_stove", "440-450"),
    ],
)
def test_eval_seed_guard(tmp_path: Path, checkout: Path, task: str, seeds: str) -> None:
    t = {"task": task, "config": str(default_port_config(tmp_path)), "seeds": [seeds]}
    if task.startswith("libero_"):
        t["layouts"] = 50
    spec = _eval_spec(tmp_path, tasks=[t])
    with pytest.raises(PreflightError, match="seed"):
        write_plan(load_spec(spec))


def test_eval_adapter_needs_name(tmp_path: Path, checkout: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}")
    with pytest.raises(PreflightError, match="lora_name"):
        write_plan(load_spec(_eval_spec(tmp_path, adapter=str(adapter))))


def test_rewrite_config_text() -> None:
    src = (
        "api_servers:\n  - _target_: a\n    port: 8114\n"
        "trials: 400\nnum_workers: 8\nresume_idx: 101\nx: 1\nresume_idx: 301\n"
    )
    out = rewrite_config_text(src, start=401, end=450, port_offset=1000, num_workers=2)
    assert out == "api_servers:\n  - _target_: a\n    port: 9114\ntrials: 450\nnum_workers: 2\nx: 1\nresume_idx: 401\n"
    with pytest.raises(RunConfigError):
        rewrite_config_text("x: 1\n", start=1, end=2)


def test_example_specs_parse() -> None:
    for name, mode in (("collect.yaml", "collect"), ("eval_base.yaml", "eval"), ("eval_trained.yaml", "eval")):
        assert load_spec(REPO / "configs" / name).mode == mode


def test_score(tmp_path: Path) -> None:
    t = tmp_path / "run" / "seg_cube_lifting_401" / "outputs" / "m" / "cube_lifting"
    make_trial(t, 401, [{"code": "a"}, {"decision": "finish", "stdout": "", "stderr": ""}])
    make_trial(
        t,
        402,
        [
            {"code": "a"},
            {"decision": "regenerate", "code": "b", "stdout": "", "stderr": ""},
            {"decision": "finish", "stdout": "", "stderr": ""},
        ],
    )
    make_trial(t, 403, [{"code": "a"}, {"decision": "finish", "stdout": "", "stderr": ""}], task_completed=False)
    d = make_trial(t, 404, [{"code": "a"}, {"decision": "finish", "stdout": "", "stderr": ""}], sandbox_rc=1)
    calls = json.loads((d / "all_responses.json").read_text())
    calls[-1]["reasoning"] = "```python\nx\n```"  # code but no REGENERATE keyword
    (d / "all_responses.json").write_text(json.dumps(calls))
    res = score_runs([tmp_path / "run"])
    r = res["tasks"]["cube_lifting"]
    assert (r["n"], r["success"], r["first_try"], r["finish_unsolved"], r["code_no_keyword"]) == (4, 2, 1, 2, 1)
    assert r["turns_success"] == 2.5
    make_trial(
        tmp_path / "run2" / "outputs" / "m" / "cube_lifting",
        401,
        [{"code": "a"}, {"decision": "finish", "stdout": "", "stderr": ""}],
    )
    with pytest.raises(ValueError, match="duplicate seeds"):
        score_runs([tmp_path / "run", tmp_path / "run2"])


def test_libero_tasks_need_layouts(tmp_path: Path, checkout: Path) -> None:
    cfg = str(default_port_config(tmp_path))
    spec = write_spec(tmp_path, tasks=[{"task": "libero_goal_drawer", "config": cfg, "seeds": ["301-310"]}])
    with pytest.raises(PreflightError, match="layouts"):
        write_plan(load_spec(spec))
    spec = write_spec(tmp_path, tasks=[{"task": "cube_lifting", "config": cfg, "seeds": ["301-310"], "layouts": 50}])
    with pytest.raises(PreflightError, match="only for LIBERO-PRO"):
        write_plan(load_spec(spec))
    # with 30 init states, seeds 321-330 sit on evaluation layouts 20-29 and are refused for collection
    spec = write_spec(
        tmp_path, tasks=[{"task": "libero_goal_drawer", "config": cfg, "seeds": ["321-321"], "layouts": 30}]
    )
    with pytest.raises(PreflightError, match="evaluation layout 20"):
        write_plan(load_spec(spec))
    spec = write_spec(
        tmp_path, tasks=[{"task": "libero_goal_drawer", "config": cfg, "seeds": ["301-320"], "layouts": 30}]
    )
    write_plan(load_spec(spec))
    src = yaml.safe_load((tmp_path / "out" / "sources.yaml").read_text())["sources"]
    assert src[0]["layouts"] == 30


def test_gpu_settings_from_spec(tmp_path: Path, checkout: Path) -> None:
    spec = write_spec(tmp_path, max_model_len=65536, tensor_parallel_size=2, qwen_gpu_util=0.8, molmo_gpu_util=0.1)
    text = write_plan(load_spec(spec)).read_text()
    for flag in (
        "--max-model-len 65536",
        "--tensor-parallel-size 2",
        "--gpu-memory-utilization 0.8",
        "--gpu-memory-utilization 0.1",
    ):
        assert flag in text, flag
    assert "TRITON_ATTN" not in text and "TORCH_CUDA_ARCH_LIST" not in text  # B300 workarounds are opt-in


def test_lora_rank_from_adapter(tmp_path: Path) -> None:
    from geniac_baseline.collect.plan import adapter_rank

    (tmp_path / "adapter_config.json").write_text(json.dumps({"r": 64}))
    assert adapter_rank(tmp_path) == 64
    assert adapter_rank(tmp_path / "missing") == 32


def test_score_counts_missing_seeds(tmp_path: Path) -> None:
    run = tmp_path / "run"
    t = run / "seg_cube_lifting_401" / "outputs" / "m" / "cube_lifting"
    d = make_trial(t, 401, [{"code": "a"}, {"decision": "finish", "stdout": "", "stderr": ""}])
    # a retried trial leaves an older JSON value after the latest one
    (d / "all_responses.json").write_text((d / "all_responses.json").read_text() + "\n[]")
    (run / "plan.json").write_text(json.dumps({"tasks": [{"task": "cube_lifting", "seeds": ["401-403"]}]}))
    r = score_runs([run])["tasks"]["cube_lifting"]
    assert (r["n"], r["success"], r["missing"], r["missing_seeds"], r["no_finish"]) == (3, 1, 2, [402, 403], 2)
    refill = tmp_path / "refill"
    make_trial(refill / "outputs" / "m" / "cube_lifting", 402, [{"code": "a"}])
    r = score_runs([run, refill])["tasks"]["cube_lifting"]
    assert (r["n"], r["missing_seeds"]) == (3, [403])


def test_score_uses_the_end_of_trial_save(tmp_path: Path) -> None:
    run = tmp_path / "run"
    t = run / "seg_cube_lifting_401" / "outputs" / "m" / "cube_lifting"
    turns = [{"code": "a"}, {"decision": "finish", "stdout": "", "stderr": ""}]

    def per_turn(seed: int, done: int) -> None:  # the result so far in the name, only the program in summary.txt
        d = t / f"trial_{seed}_sandboxrc_0_reward_{float(done):.3f}_taskcompleted_{done}"
        d.mkdir(parents=True)
        (d / "summary.txt").write_text("-" * 100 + "\nGenerated program:\na\n")
        (d / "all_responses.json").write_text("[]")

    per_turn(401, 0)
    make_trial(t, 401, turns)  # failed part-way, succeeded at the end
    per_turn(402, 1)
    make_trial(t, 402, turns, task_completed=False)  # succeeded part-way, failed at the end
    per_turn(403, 0)
    per_turn(403, 1)  # killed before the end-of-trial save
    r = score_runs([run])["tasks"]["cube_lifting"]
    assert (r["n"], r["success"], r["missing_seeds"], r["finish_unsolved"]) == (3, 1, [403], 1)


def test_run_sh_fails_on_failed_segment(tmp_path: Path, checkout: Path) -> None:
    text = write_plan(load_spec(write_spec(tmp_path))).read_text()
    assert text.rstrip().endswith('exit "$FAILED"')
