"""Turn a run spec (YAML) into a shell script that runs CaP-X's own entry point, plus the per-segment configs.

The script starts the model servers one at a time (two Qwen servers started together each count the other's
weights during vLLM's memory profiling and fail with negative KV-cache memory), waits for ``/health``, runs
``<package>/envs/launch.py`` once per seed segment, and stops the servers on exit. The agent loop, prompts,
seeding, scoring and artifact layout are CaP-X's; nothing here re-implements them.

Run spec (``configs/collect.yaml``, ``configs/eval_base.yaml``)::

    mode: collect | eval
    capx_root: <CaP-X checkout>
    launch_script: capx/envs/launch.py        # relative to capx_root; ASPIRE: aspire/sim/cap/envs/launch.py
    sim_python: .venv/bin/python              # simulator venv (LIBERO-PRO needs its own, e.g. .venv-libero)
    serve_python: .venv-serve/bin/python      # environment with vLLM
    pythonpath: null                          # extra PYTHONPATH for the launch (ASPIRE: its repo root)
    model: Qwen/Qwen3.8-27B                   # served model name
    model_path: null                          # local weights directory (default: the model id)
    molmo_path: null                          # local Molmo2-8B weights (default: the model id); molmo: false to skip
    chat_template: <path to the low-effort Qwen3.8 chat template>
    output_dir: runs/<name>
    tasks:
      - {task: cube_lifting, config: <CaP-X config used for this task>, seeds: ["301-400"]}
    # eval only
    adapter: null                             # LoRA adapter directory (null = base model)
    lora_name: null                           # name the adapter is served under
    code_thinking: false                      # true = code writer exactly like collection (control run)
"""

from __future__ import annotations

import dataclasses
import json
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from geniac_baseline.collect import settings as S
from geniac_baseline.collect.preflight import (
    PreflightError,
    check_collect_config,
    check_patched,
    check_port_env,
    check_seeds,
    check_skill_library,
    check_visual_models,
    load_yaml,
    package_dir,
    parse_segment,
)
from geniac_baseline.collect.run_config import write_run_config


@dataclass
class TaskSpec:
    task: str
    config: Path
    seeds: list[str]
    layouts: int | None = None  # LIBERO-PRO only: the task's number of init states


@dataclass
class RunSpec:
    mode: str
    capx_root: Path
    output_dir: Path
    tasks: list[TaskSpec]
    launch_script: str = "capx/envs/launch.py"
    sim_python: str = ".venv/bin/python"
    serve_python: str = ".venv-serve/bin/python"
    pythonpath: str | None = None
    model: str = S.MODEL
    model_path: str | None = None
    molmo: bool = True
    molmo_path: str | None = None
    chat_template: Path | None = None
    allow_stock_chat_template: bool = False
    port_offset: int = 0
    vllm_port_offset: int = 0
    b300_workarounds: bool = False
    # wall-clock limit per trial, exported as CAPX_TRIAL_TIMEOUT_S (geniac_prize) and ASPIRE_TRIAL_TIMEOUT_S (ASPIRE);
    # 2400 s = the ASPIRE default the baseline ran with
    trial_timeout_s: int = 2400
    # directory the launch runs in, relative to capx_root; null = aspire/sim for the ASPIRE launch script, else .
    launch_cwd: str | None = None
    max_model_len: int = S.MAX_MODEL_LEN
    tensor_parallel_size: int = 1
    qwen_gpu_util: float | None = None  # null = 0.62 (collect) / 0.30 per server (eval)
    molmo_gpu_util: float = S.MOLMO_GPU_UTIL
    num_workers: int | None = None
    max_tokens: int = S.MAX_TOKENS
    temperature: float = S.TEMPERATURE
    reasoning_effort: str = S.REASONING_EFFORT
    adapter: Path | None = None
    lora_name: str | None = None
    code_thinking: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)


class SpecError(ValueError):
    """The run spec is malformed."""


def load_spec(path: Path) -> RunSpec:
    raw = yaml.safe_load(path.read_text())
    if not isinstance(raw, dict):
        raise SpecError(f"{path}: not a mapping")
    known = {f.name for f in dataclasses.fields(RunSpec)}
    if set(raw) - known:
        raise SpecError(f"{path}: unknown keys {sorted(set(raw) - known)}")
    base = path.parent

    def p(v: Any) -> Path:
        q = Path(str(v)).expanduser()
        return q if q.is_absolute() else (base / q).resolve()

    kw = dict(raw)
    for req in ("mode", "capx_root", "output_dir", "tasks"):
        if req not in kw:
            raise SpecError(f"{path}: missing '{req}'")
    if kw["mode"] not in ("collect", "eval"):
        raise SpecError(f"{path}: mode must be collect or eval")
    for key in ("capx_root", "output_dir", "chat_template", "adapter"):
        if kw.get(key) is not None:
            kw[key] = p(kw[key])
    tasks = []
    for i, t in enumerate(kw["tasks"] or []):
        if not isinstance(t, dict) or not {"task", "config", "seeds"} <= set(t) <= {
            "task",
            "config",
            "seeds",
            "layouts",
        }:
            raise SpecError(f"{path}: tasks[{i}] needs task, config, seeds (and layouts for LIBERO-PRO)")
        seeds = t["seeds"] if isinstance(t["seeds"], list) else [t["seeds"]]
        tasks.append(TaskSpec(str(t["task"]), p(t["config"]), [str(s) for s in seeds], t.get("layouts")))
    if not tasks:
        raise SpecError(f"{path}: no tasks")
    kw["tasks"] = tasks
    return RunSpec(**kw)


def launch_cwd(spec: RunSpec) -> str:
    """ASPIRE's launch ran from ``<ASPIRE root>/aspire/sim`` in the baseline runs; CaP-X's from its root."""
    if spec.launch_cwd is not None:
        return spec.launch_cwd
    return "aspire/sim" if spec.launch_script.startswith("aspire/sim/") else "."


def adapter_rank(adapter: Path) -> int:
    """LoRA rank ``r`` of the adapter (vLLM needs --max-lora-rank >= r); 32 if the file cannot be read."""
    try:
        return int(json.loads((adapter / "adapter_config.json").read_text())["r"])
    except (OSError, KeyError, ValueError, TypeError):
        return 32


def _ports(spec: RunSpec) -> dict[str, int]:
    off = spec.vllm_port_offset
    return {"llm": S.LLM_PORT + off, "code": S.CODE_PORT + off, "molmo": S.MOLMO_PORT + off}


def servers(spec: RunSpec) -> tuple[list[S.VllmServer], str, str]:
    """Servers to start (in this order), and the launch's code-model name and server port key."""
    ports = _ports(spec)
    tpl = str(spec.chat_template) if spec.chat_template else None
    weights = spec.model_path or spec.model
    b300 = spec.b300_workarounds
    big = {"max_model_len": spec.max_model_len, "tensor_parallel_size": spec.tensor_parallel_size}
    if spec.mode == "collect":
        # one server for the code writer and the VDM, thinking on, util 0.62
        srv = [
            S.qwen_server(
                name="qwen",
                model_path=weights,
                port=ports["llm"],
                gpu_util=spec.qwen_gpu_util or S.COLLECT_QWEN_GPU_UTIL,
                chat_template=tpl,
                thinking=True,
                b300=b300,
                served_model_name=spec.model,
                **big,
            )
        ]
        code_model, code_port = spec.model, "llm"
    else:
        # VDM server keeps the collection settings; separate code-writer server
        vdm = S.qwen_server(
            name="vdm",
            model_path=weights,
            port=ports["llm"],
            gpu_util=spec.qwen_gpu_util or S.EVAL_VDM_GPU_UTIL,
            chat_template=tpl,
            thinking=True,
            b300=b300,
            served_model_name=spec.model,
            **big,
        )
        code = S.qwen_server(
            name="code",
            model_path=weights,
            port=ports["code"],
            gpu_util=spec.qwen_gpu_util or S.EVAL_CODE_GPU_UTIL,
            chat_template=tpl,
            thinking=spec.code_thinking,
            b300=b300,
            served_model_name=spec.model,
            **big,
        )
        code_model = spec.model
        if spec.adapter:
            code.lora_modules = {str(spec.lora_name): str(spec.adapter)}
            code.max_lora_rank = adapter_rank(spec.adapter)
            code_model = str(spec.lora_name)
        srv = [vdm, code]
        code_port = "code"
    if spec.molmo:
        srv.append(
            S.molmo_server(
                model_path=spec.molmo_path or S.MOLMO_MODEL,
                port=ports["molmo"],
                b300=b300,
                gpu_util=spec.molmo_gpu_util,
            )
        )
    return srv, code_model, code_port


def _segments(spec: RunSpec) -> list[tuple[TaskSpec, int, int]]:
    out = []
    for t in spec.tasks:
        for s in t.seeds:
            a, b = parse_segment(s)
            check_seeds(spec.mode, t.task, range(a, b + 1), t.layouts)
            out.append((t, a, b))
    return out


def preflight(spec: RunSpec) -> Path:
    """All checks; returns the CaP-X package directory."""
    pkg = package_dir(spec.capx_root, spec.launch_script)
    if spec.chat_template is None and not spec.allow_stock_chat_template:
        raise PreflightError(
            "no chat_template given. The baseline servers used a Qwen3.8 template with default reasoning effort "
            f"'low' (sha256 {S.CHAT_TEMPLATE_SHA256}); without it the thinking instructions differ. Set "
            "chat_template, or allow_stock_chat_template: true to run with the model's own template knowingly."
        )
    if spec.chat_template is not None and not spec.chat_template.is_file():
        raise PreflightError(f"chat_template {spec.chat_template} not found")
    if spec.mode == "eval" and spec.adapter is not None:
        if not (spec.adapter / "adapter_config.json").is_file():
            raise PreflightError(f"no adapter_config.json in {spec.adapter}")
        if not spec.lora_name:
            raise PreflightError("lora_name is required with adapter")
    if spec.mode == "collect" and spec.adapter is not None:
        raise PreflightError("collection runs the base model; adapter is an eval setting")
    check_patched(pkg, spec.mode)
    _, code_model, _ = servers(spec)
    for t in spec.tasks:
        cfg = load_yaml(t.config)
        if spec.mode == "collect":
            check_collect_config(cfg, t.config)
        check_visual_models(pkg, cfg, code_model=code_model, vdm_model=spec.model)
        check_skill_library(pkg, cfg, t.config)
        check_port_env(pkg, cfg, spec.port_offset, molmo_port_moved=spec.molmo and spec.vllm_port_offset != 0)
    _segments(spec)
    return pkg


def _q(x: object) -> str:
    return shlex.quote(str(x))


def write_plan(spec: RunSpec) -> Path:
    """Run the preflight, write configs + ``run.sh`` (+ ``sources.yaml`` for collection); return run.sh."""
    preflight(spec)
    out = spec.output_dir
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    srv, code_model, code_port = servers(spec)
    ports = _ports(spec)
    segs = _segments(spec)
    out.mkdir(parents=True)
    model_slug = code_model.replace("/", "_")  # capx/envs/runner.py _setup_output_dir inserts it
    sources = []
    lines = [
        "#!/usr/bin/env bash",
        f"# Generated by geniac-baseline ({spec.mode}). Runs CaP-X's own entry point (docs/collect.md, docs/eval.md).",
        "set -uo pipefail",
        f"OUT={_q(out)}",
        f"CAPX_ROOT={_q(spec.capx_root)}",
        f"SIM_PY={_q(spec.capx_root / spec.sim_python)}",
        f"SERVE_PY={_q(spec.capx_root / spec.serve_python)}",
        "",
    ]
    env = {**S.RUN_ENV, **(S.B300_ENV if spec.b300_workarounds else {})}
    # endpoint of the VLM some APIs query themselves (ASPIRE's LIBERO skill-library API reads
    # CAPX_VLM_URL / CAPX_VLM_MODEL; capx/utils/eval_utils.py reads LLM_BASE_URL). Points at the Qwen VDM server.
    env["LLM_BASE_URL"] = f"http://127.0.0.1:{ports['llm']}/"
    env["CAPX_VLM_URL"] = f"http://127.0.0.1:{ports['llm']}/v1"
    env["CAPX_VLM_MODEL"] = spec.model
    env["CAPX_TRIAL_TIMEOUT_S"] = str(spec.trial_timeout_s)
    env["ASPIRE_TRIAL_TIMEOUT_S"] = str(spec.trial_timeout_s)
    env.update(spec.extra_env)
    if spec.molmo:
        env["CAPX_MOLMO_URL"] = f"http://127.0.0.1:{ports['molmo']}/v1"  # read by both trees' molmo.py
        env["CAPX_MOLMO_MODEL"] = S.MOLMO_MODEL
    if spec.port_offset:
        for name, port in zip(("SAM2", "SAM3", "GRASPNET", "PYROKI", "OWLVIT"), S.PERCEPTION_PORTS, strict=True):
            env[f"{name}_SERVICE_URL"] = f"http://127.0.0.1:{port + spec.port_offset}"
    lines += [f"export {k}={_q(v)}" for k, v in env.items()]
    lines += [
        "",
        "PIDS=()",
        'cleanup() { for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null; done; sleep 5;'
        ' for p in "${PIDS[@]}"; do kill -9 "$p" 2>/dev/null; done; }',
        "trap cleanup EXIT",
        "wait_up() {  # name pid port log -- poll /health every 10 s, up to 30 min",
        '  for i in $(seq 1 180); do curl -sf "http://127.0.0.1:$3/health" >/dev/null 2>&1 &&'
        ' { echo "[run] $1 ready after $((i*10))s"; return 0; };',
        '    kill -0 "$2" 2>/dev/null || { echo "[run] FATAL: $1 died"; tail -40 "$4"; return 1; }; sleep 10; done;',
        '  echo "[run] FATAL: $1 never became ready"; tail -40 "$4"; return 1; }',
        "",
    ]
    for s in srv:  # one at a time
        log = out / f"vllm_{s.name}.log"
        lines += [
            f"echo '[run] starting {s.name} on :{s.port}'",
            '"$SERVE_PY" ' + " ".join(_q(x) for x in s.argv("python")[1:]) + f" > {_q(log)} 2>&1 &",
            "PIDS+=($!)",
            f'wait_up {s.name} "${{PIDS[-1]}}" {s.port} {_q(log)} || exit 1',
            "",
        ]
    launch = spec.capx_root / spec.launch_script
    pyenv = f"PYTHONPATH={_q(spec.pythonpath)} " if spec.pythonpath else ""
    lines.append(f"cd {_q(spec.capx_root / launch_cwd(spec))}")
    lines.append("declare -A RC")
    for t, a, b in segs:
        seg = out / f"seg_{t.task}_{a}"
        cfg = write_run_config(
            t.config, seg / f"{t.task}.yaml", start=a, end=b, port_offset=spec.port_offset, num_workers=spec.num_workers
        )
        argv = [
            "--config-path",
            cfg,
            "--model",
            code_model,
            "--server-url",
            f"http://127.0.0.1:{ports[code_port]}/v1/chat/completions",
            "--temperature",
            spec.temperature,
            "--reasoning-effort",
            spec.reasoning_effort,
            "--max-tokens",
            spec.max_tokens,
            "--visual-differencing-model",
            spec.model,
            "--visual-differencing-model-server-url",
            f"http://127.0.0.1:{ports['llm']}/v1/chat/completions",
            "--output-dir",
            seg / "outputs" / t.task,
        ]
        key = f"{t.task}:{a}-{b}"
        lines += [
            f"echo '[run] {key}'",
            f'{pyenv}"$SIM_PY" {_q(launch)} '
            + " ".join(_q(x) for x in argv)
            + f" > {_q(seg / (t.task + '.log'))} 2>&1",
            f"RC[{_q(key)}]=$?",
            f'echo "[run] {key} rc=${{RC[{_q(key)}]}}"',
        ]
        sources.append(
            {
                "run_tag": f"{out.name}-seg{a}",
                "task": t.task,
                "trials_dir": str(seg / "outputs" / model_slug / t.task),
                "config": str(cfg),
                **({"layouts": t.layouts} if t.layouts is not None else {}),
            }
        )
    lines += [
        "",
        'echo "[run] done; trial folders under $OUT/seg_*/outputs/"',
        "FAILED=0",
        'for k in "${!RC[@]}"; do [ "${RC[$k]}" = 0 ] ||',
        '  { echo "[run] FAILED segment $k rc=${RC[$k]}"; FAILED=1; }; done',
        'exit "$FAILED"',
    ]
    script = out / "run.sh"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    plan = {k: (str(v) if isinstance(v, Path) else v) for k, v in dataclasses.asdict(spec).items()}
    plan["tasks"] = [
        {"task": t.task, "config": str(t.config), "seeds": t.seeds, "layouts": t.layouts} for t in spec.tasks
    ]
    plan["servers"] = [s.argv("<serve_python>") for s in srv]
    (out / "plan.json").write_text(json.dumps(plan, indent=1, default=str) + "\n")
    if spec.mode == "collect":
        (out / "sources.yaml").write_text(yaml.safe_dump({"sources": sources}, sort_keys=False))
    return script
