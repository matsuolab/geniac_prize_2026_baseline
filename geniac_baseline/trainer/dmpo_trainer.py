"""DMPO stage: trajectory-level preference training on top of the SFT adapter.

Ported from the trainer of the baseline DMPO run. The policy starts from the SFT adapter (``init_adapter``) and the
frozen reference *is* that adapter: before the first step every turn of every used trajectory is scored once with
it, and the result is cached in ``<output_dir>/ref_logp.json`` together with a hash of the adapter and the pair
data (a resumed job reads it back; a different adapter or data is an error). Loss and exact
two-pass gradient: ``geniac_baseline.trainer.losses.dmpo``.

Rows are rendered exactly as in the SFT stage (``encode_row``); a turn's log-probability is the sum over all its
answer tokens. One optimizer step takes ``pairs_per_step`` pairs; the last step of an epoch takes whatever pairs
are left (it can be a single pair), as in the baseline run. Validation (step 0 and after every epoch) reports
preference accuracy, mean margin and mean loss on the val pairs. A checkpoint (``ckpt-last``) is written after
every step, an adapter after every epoch (``epoch-N/adapter``), the final adapter to ``adapter/``.

Two behaviours fixed in the baseline run and kept here:

* ``model.train()`` right before the loop: a resumed job skips the reference pass, which is what put the model
  back into train mode, and in eval mode gradient checkpointing is off (the resumed job ran out of memory).
* ``stop_after_min`` counts from job start and keeps room for one more step (a step can take more than an hour).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from geniac_baseline.data.schema import validate_rows
from geniac_baseline.trainer.config import DMPOConfig, check_output_dir, config_to_dict
from geniac_baseline.trainer.data_source import read_manifest, read_rows, resolve_data_dir
from geniac_baseline.trainer.encode import EncodedRow, decision_word_ids, encode_row
from geniac_baseline.trainer.losses.dmpo import Pair, margin, mean_loss, phi, two_pass_backward
from geniac_baseline.trainer.sft_trainer import DEFAULT_TEMPLATE_KWARGS, _forward, lr_lambda


class PairDataError(ValueError):
    """The pairs directory is inconsistent."""


def turn_logp(model: Any, ex: EncodedRow, device: Any, grad: bool) -> Any:
    """Summed log-probability of every answer token of one turn (a scalar tensor)."""
    import torch

    with torch.set_grad_enabled(grad):
        logits = _forward(model, ex, device)
        tgt = ex.input_ids[ex.n_prompt :].to(device)
        per_tok = torch.nn.functional.cross_entropy(logits[:-1], tgt, reduction="none")
        return -per_tok[ex.answer_mask.to(device)].sum()


def load_pairs(rows: list[dict[str, Any]], pairs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Group rows by trajectory and check every pair against them (pure Python; used by tests)."""
    traj: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        traj.setdefault(r["metadata"]["sample_id"], []).append(r)
    for sid, rs in traj.items():
        rs.sort(key=lambda r: r["metadata"]["turn"])
        if [r["metadata"]["turn"] for r in rs] != list(range(len(rs))):
            raise PairDataError(f"{sid}: turns are not 0..T-1")
    for p in pairs:
        for side in ("chosen", "rejected"):
            if p[side] not in traj:
                raise PairDataError(f"pair {p['pair_id']}: {side} {p[side]} has no rows")
        if not traj[p["chosen"]][0]["metadata"]["success"]:
            raise PairDataError(f"pair {p['pair_id']}: chosen is a failure")
        if traj[p["rejected"]][0]["metadata"]["success"]:
            raise PairDataError(f"pair {p['pair_id']}: rejected is a success")
        if p["split"] not in ("train", "val"):
            raise PairDataError(f"pair {p['pair_id']}: split {p['split']!r}")
    return traj


def steps_per_epoch(n_pairs: int, pairs_per_step: int) -> int:
    """Steps of one epoch; the last one holds the remainder (the evaluated baseline run: 521 train pairs at 32 per step
    = 17 steps, the last with 9 pairs; 3 epochs = 51 steps)."""
    return math.ceil(n_pairs / pairs_per_step)


def resolve_init_adapter(path: Path) -> Path:
    """An adapter directory, or an SFT run directory: then the epoch in its best_epoch.json (lowest val/loss_all)."""
    if (path / "adapter_config.json").is_file():
        return path
    best = path / "best_epoch.json"
    if best.is_file():
        return path / f"epoch-{json.loads(best.read_text())['epoch']}" / "adapter"
    raise FileNotFoundError(f"{path}: neither an adapter (adapter_config.json) nor an SFT run with best_epoch.json")


def reference_key(init_adapter: Path, data_dir: Path) -> str:
    """SHA-256 over the starting adapter's files and the pair data: what the cached reference log-probs depend on."""
    h = hashlib.sha256()
    for f in [
        *sorted(p for p in init_adapter.iterdir() if p.is_file()),
        data_dir / "rows.jsonl",
        data_dir / "pairs.jsonl",
    ]:
        h.update(f.name.encode())
        with f.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def train_dmpo(cfg: DMPOConfig, dry_run: bool = False) -> dict[str, Any]:
    import torch
    from transformers import AutoProcessor

    t0 = time.time()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data_dir = resolve_data_dir(cfg.data, local_files_only=cfg.local_files_only)
    manifest = read_manifest(data_dir)
    template_kwargs = manifest.get("chat_template_kwargs", DEFAULT_TEMPLATE_KWARGS)
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=cfg.local_files_only)
    word_ids = decision_word_ids(processor.tokenizer)
    rows = read_rows(data_dir / "rows.jsonl")
    validate_rows(rows)
    pairs = read_rows(data_dir / "pairs.jsonl")
    traj = load_pairs(rows, pairs)
    train = [p for p in pairs if p["split"] == "train"]
    val = [p for p in pairs if p["split"] == "val"]
    if cfg.max_pairs:
        train = train[: cfg.max_pairs]
    used = sorted({p[k] for p in train + val for k in ("chosen", "rejected")})
    ex: dict[str, list[EncodedRow]] = {}
    for s in used:
        ex[s] = [encode_row(processor, r, template_kwargs, word_ids) for r in traj[s]]
        for e in ex[s]:
            if int(e.answer_mask.sum()) == 0:
                raise PairDataError(f"{e.row_id}: no answer tokens")
    n_turns = {s: len(v) for s, v in ex.items()}
    spe = steps_per_epoch(len(train), cfg.pairs_per_step)
    total = spe * cfg.epochs
    if cfg.max_steps:
        total = min(total, cfg.max_steps)
    summary = {
        "train_pairs": len(train),
        "val_pairs": len(val),
        "trajectories": len(used),
        "turns": sum(n_turns.values()),
        "tokens_per_pass": sum(int(e.input_ids.shape[0]) for s in used for e in ex[s]),
        "steps_per_epoch": spe,
        "total_steps": total,
        "phi_example_T3": [round(phi(k, 3, cfg.gamma), 4) for k in range(3)],
        "encode_sec": round(time.time() - t0),
    }
    print(f"[data] {json.dumps(summary)}", flush=True)
    if dry_run:
        return summary
    if not train:
        raise PairDataError("no train pairs")

    from peft import PeftModel
    from transformers import AutoModelForImageTextToText

    out = cfg.output_dir
    check_output_dir(out, cfg.resume)
    out.mkdir(parents=True, exist_ok=True)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model, dtype=torch.bfloat16, device_map={"": 0}, local_files_only=cfg.local_files_only
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    init_adapter = resolve_init_adapter(cfg.init_adapter)
    print(f"[init] SFT adapter {init_adapter}", flush=True)
    start = cfg.resume / "adapter" if cfg.resume else init_adapter
    model = PeftModel.from_pretrained(model, start, is_trainable=True)
    device = model.device
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] loaded from {start} in {time.time() - t0:.0f}s", flush=True)

    ref_path = out / "ref_logp.json"
    key = reference_key(init_adapter, data_dir)
    if ref_path.exists():
        cached = json.loads(ref_path.read_text())
        if cached.get("key") != key:
            raise RuntimeError(f"{ref_path} was computed for another init_adapter or pair data; use a new output_dir")
        ref = cached["ref"]
    else:
        if cfg.resume:
            raise RuntimeError("resume needs the ref_logp.json written by the first job")
        model.eval()
        ref = {s: [float(turn_logp(model, e, device, False)) for e in ex[s]] for s in used}
        ref_path.write_text(json.dumps({"key": key, "ref": ref}))
        print(f"[ref] {len(ref)} trajectories scored with the starting adapter ({time.time() - t0:.0f}s)", flush=True)

    def logp_fn(s: str, k: int, grad: bool) -> Any:
        v = turn_logp(model, ex[s][k], device, grad)
        return v if grad else float(v)

    def validate() -> dict[str, float]:
        model.eval()
        ms = []
        for p in val:
            lp = {s: [logp_fn(s, k, False) for k in range(n_turns[s])] for s in (p["chosen"], p["rejected"])}
            ms.append(
                margin(lp[p["chosen"]], ref[p["chosen"]], lp[p["rejected"]], ref[p["rejected"]], cfg.beta, cfg.gamma)
            )
        model.train()
        if not ms:
            return {}
        return {
            "val/pref_acc": sum(m > 0 for m in ms) / len(ms),
            "val/margin": sum(ms) / len(ms),
            "val/loss": mean_loss(ms),
        }

    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: lr_lambda(s, cfg.warmup_steps, total))
    step, order = 0, []
    wandb_id = None
    if cfg.resume:
        st = torch.load(cfg.resume / "state.pt", weights_only=False)
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        step, order = st["step"], st["order"]
        random.setstate(st["py_rng"])
        if st["total_steps"] != total or st.get("train_pairs") != len(train):
            raise RuntimeError("resume with different data or schedule settings")
        wandb_id = json.loads((cfg.resume / "state.json").read_text()).get("wandb_id")

    run = None
    if cfg.wandb.enabled:
        import wandb

        os.environ.setdefault("WANDB_MODE", cfg.wandb.mode)
        run = wandb.init(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            config={**config_to_dict(cfg), "data_summary": summary},
            dir=str(out),
            id=wandb_id,
            resume="allow",
        )
    (out / "train_config.json").write_text(
        json.dumps(
            {
                **config_to_dict(cfg),
                "data_summary": summary,
                "chat_template_kwargs": template_kwargs,
                "loss": "DMPO (EMNLP 2024) arXiv 2406.14868 Eq. 16",
                "wandb_id": run.id if run else None,
            },
            indent=1,
        )
        + "\n"
    )
    log = (out / "train_log.jsonl").open("a")

    def emit(rec: dict[str, Any]) -> None:
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(f"[log] {json.dumps(rec)}", flush=True)
        if run:
            run.log(rec)

    def save(tag: str) -> None:
        d = out / tag
        d.mkdir(exist_ok=True)
        model.save_pretrained(d / "adapter")
        torch.save(
            {
                "optim": optim.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "order": order,
                "py_rng": random.getstate(),
                "total_steps": total,
                "train_pairs": len(train),
            },
            d / "state.pt",
        )
        (d / "state.json").write_text(json.dumps({"step": step, "wandb_id": run.id if run else None}) + "\n")

    if step >= total:
        print("[done] nothing to resume", flush=True)
        if run:
            run.finish()
        return summary
    if step == 0:
        emit({"step": 0, **validate()})
    model.train()  # also on resume: the skipped reference pass is what set train mode before
    while step < total:
        if not order:
            order = random.sample(range(len(train)), len(train))
        batch = [train[order.pop()] for _ in range(min(cfg.pairs_per_step, len(order)))]
        t_step = time.time()
        ms = two_pass_backward(
            [Pair(p["chosen"], p["rejected"]) for p in batch], n_turns, ref, logp_fn, cfg.beta, cfg.gamma
        )
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        step += 1
        sec = time.time() - t_step
        emit(
            {
                "step": step,
                "train/loss": mean_loss(ms),
                "train/margin": sum(ms) / len(ms),
                "train/pref_acc": sum(m > 0 for m in ms) / len(ms),
                "train/pairs": len(ms),
                "train/grad_norm": gnorm,
                "train/lr": sched.get_last_lr()[0],
                "train/epoch": step / spe,
                "perf/sec": round(sec, 1),
                "perf/max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
        )
        save("ckpt-last")
        if step % spe == 0 or step == total:
            model.save_pretrained(out / f"epoch-{math.ceil(step / spe)}" / "adapter")
            emit({"step": step, **validate()})
        if cfg.stop_after_min and step < total and (time.time() - t0 + sec) / 60 > cfg.stop_after_min:
            print(
                f"[stop] time budget reached at step {step}/{total}; resume with resume: {out / 'ckpt-last'}",
                flush=True,
            )
            if run:
                run.finish()
            return summary
    model.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "adapter")
    if run:
        run.finish()
    print(f"[done] adapter -> {out / 'adapter'} ({time.time() - t0:.0f}s total)", flush=True)
    return summary
