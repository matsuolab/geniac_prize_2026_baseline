"""SFT warm start: LoRA trainer for multi-turn CaP-X rows (one row per model call).

Ported from the trainer that produced the baseline adapters, itself based on the team's ``heuristic_sft_train.py``.
Kept from them: rendering with the data manifest's ``chat_template_kwargs`` (thinking off), LoRA on the language
model's q/k/v/o/gate/up/down, bf16 on one GPU, batch size 1 with ``grad_accum`` rows per optimizer step,
AdamW, linear warmup then cosine decay, gradient clipping at 1.0, logits only for the answer positions
(``logits_to_keep``), checkpoints with optimizer / scheduler / RNG state and the row order (exact resume), an
adapter per finished epoch (``epoch-N/adapter``), and validation at step 0 (= the base model: LoRA B starts at
zero), at the end of every epoch, and every ``val_every`` steps if set.

Epoch selection (fixed before any evaluation result of the baseline): evaluate, and start DMPO from, the
epoch-end adapter with the lowest ``val/loss_all``. The trainer writes it to ``<output_dir>/best_epoch.json``
(baseline: epoch 2).

Every row goes through ``geniac_baseline.trainer.loss.compute_loss`` (completion-only cross-entropy).
Validation reports loss-independent metrics: teacher-forced NLL of every answer token per row group, and
REGENERATE-vs-FINISH accuracy of the first answer token.

torch / transformers / peft are imported inside functions so the package imports without them.
"""

from __future__ import annotations

import collections
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any

from geniac_baseline.data.schema import validate_rows
from geniac_baseline.trainer import loss as loss_mod
from geniac_baseline.trainer.config import TrainConfig, check_output_dir, config_to_dict
from geniac_baseline.trainer.data_source import read_manifest, read_rows, resolve_data_dir
from geniac_baseline.trainer.encode import EncodedRow, decision_word_ids, encode_row
from geniac_baseline.trainer.loss import LossInputs, LossOutput

DEFAULT_TEMPLATE_KWARGS = {"enable_thinking": False}


def compute_row_loss(inputs: LossInputs) -> LossOutput:
    """The trainer's only call into the loss. Checks the returned object, nothing else."""
    out = loss_mod.compute_loss(inputs)
    if not isinstance(out, LossOutput):
        raise TypeError(f"compute_loss must return LossOutput, got {type(out).__name__}")
    if out.loss is None and out.normalizer != 0:
        raise ValueError(f"{inputs.row_id}: a skipped row (loss=None) must have normalizer 0")
    if out.normalizer < 0:
        raise ValueError(f"{inputs.row_id}: negative normalizer")
    return out


def lr_lambda(step: int, warmup: int, total: int) -> float:
    """Linear warmup over ``warmup`` steps times half-cosine decay to 0 at ``total`` (baseline schedule)."""
    return min(1.0, (step + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(step, total) / total))


def total_steps(n_rows: int, epochs: float, grad_accum: int, max_steps: int) -> int:
    t = math.ceil(n_rows * epochs / grad_accum)
    return min(t, max_steps) if max_steps else t


def _forward(model: Any, ex: EncodedRow, device: Any) -> Any:
    import torch

    ids = ex.input_ids[None].to(device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), logits_to_keep=ex.keep, use_cache=False)
    return out.logits[0].float()  # (keep, V): logits[0] predicts the first answer token


def _loss_inputs(ex: EncodedRow, logits: Any, device: Any) -> LossInputs:
    return LossInputs(
        logits=logits[:-1],
        target_ids=ex.input_ids[ex.n_prompt :].to(device),
        answer_mask=ex.answer_mask.to(device),
        decision_prefix_len=ex.decision_prefix_len,
        input_ids=ex.input_ids,
        n_prompt_tokens=ex.n_prompt,
        metadata=ex.metadata,
        row_id=ex.row_id,
    )


def validate(model: Any, val: list[EncodedRow], device: Any, word_ids: dict[str, int]) -> dict[str, float]:
    """Loss-independent validation metrics (every answer token, no content mask)."""
    import torch

    model.eval()
    s: collections.Counter[str] = collections.Counter()
    n: collections.Counter[str] = collections.Counter()
    hit = tot = 0
    with torch.no_grad():
        for ex in val:
            logits = _forward(model, ex, device)
            tgt = ex.input_ids[ex.n_prompt :].to(device)
            m = ex.answer_mask.to(device)
            per_tok = torch.nn.functional.cross_entropy(logits[:-1], tgt, reduction="none")
            s[ex.group] += per_tok[m].sum().item()
            n[ex.group] += int(m.sum())
            if ex.decision:
                first = logits[0]
                pred = "REGENERATE" if first[word_ids["REGENERATE"]] > first[word_ids["FINISH"]] else "FINISH"
                hit += pred == ex.decision
                tot += 1
    model.train()
    rec = {f"val/loss_{g}": s[g] / n[g] for g in sorted(n)}
    rec["val/loss_all"] = sum(s.values()) / max(sum(n.values()), 1)
    rec["val/decision_acc"] = hit / max(tot, 1)
    rec["val/decision_n"] = tot
    return rec


def best_epoch(log_path: Path) -> dict[str, Any] | None:
    """The epoch-end validation with the lowest val/loss_all in train_log.jsonl (all jobs of a resumed run)."""
    recs = [json.loads(x) for x in log_path.read_text().splitlines() if x.strip()] if log_path.exists() else []
    ep = {r["epoch"]: r["val/loss_all"] for r in recs if "epoch" in r and "val/loss_all" in r}
    if not ep:
        return None
    e = min(ep, key=lambda k: ep[k])
    return {"epoch": e, "val/loss_all": ep[e], "by_epoch": ep}


def train(cfg: TrainConfig, dry_run: bool = False) -> dict[str, Any]:
    """Run training (or, with ``dry_run``, only load and encode the data and report)."""
    import torch
    from transformers import AutoProcessor

    t_job = time.time()
    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    data_dir = resolve_data_dir(cfg.data, local_files_only=cfg.local_files_only)
    manifest = read_manifest(data_dir)
    template_kwargs = manifest.get("chat_template_kwargs", DEFAULT_TEMPLATE_KWARGS)
    processor = AutoProcessor.from_pretrained(cfg.model, local_files_only=cfg.local_files_only)
    word_ids = decision_word_ids(processor.tokenizer)

    dropped: list[dict[str, Any]] = []

    def load(name: str) -> list[EncodedRow]:
        rows = read_rows(data_dir / name)
        validate_rows(rows)
        enc = [encode_row(processor, r, template_kwargs, word_ids) for r in rows]
        if not cfg.max_row_tokens:
            return enc
        keep = []
        for e in enc:
            n = int(e.input_ids.shape[0])
            if n > cfg.max_row_tokens:
                dropped.append({"file": name, "row_id": e.row_id, "tokens": n})
            else:
                keep.append(e)
        return keep

    examples = load(cfg.train_file)
    if (data_dir / cfg.val_file).is_file():
        val = load(cfg.val_file)
    elif cfg.require_val:
        raise FileNotFoundError(
            f"{data_dir / cfg.val_file} not found: the epoch is chosen by val/loss_all (docs/train.md). "
            "Set require_val: false to train without validation."
        )
    else:
        val = []
    if cfg.val_max_rows:
        val = random.Random(0).sample(val, min(cfg.val_max_rows, len(val)))
    if not examples:
        raise ValueError("no training rows")
    lengths = [int(e.input_ids.shape[0]) for e in examples]
    steps = total_steps(len(examples), cfg.epochs, cfg.grad_accum, cfg.max_steps)
    summary = {
        "rows": len(examples),
        "groups": dict(collections.Counter(e.group for e in examples)),
        "answer_tokens": int(sum(int(e.answer_mask.sum()) for e in examples)),
        "tokens_total": sum(lengths),
        "max_len": max(lengths),
        "val_rows": len(val),
        "dropped_long_rows": dropped,
        "word_ids": word_ids,
        "steps_per_epoch": math.ceil(len(examples) / cfg.grad_accum),
        "total_steps": steps,
        "encode_sec": round(time.time() - t_job),
    }
    print(f"[data] {json.dumps(summary)}", flush=True)
    if dry_run:
        return summary

    from peft import LoraConfig, PeftModel, get_peft_model
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
    if cfg.resume:
        model = PeftModel.from_pretrained(model, cfg.resume / "adapter", is_trainable=True)
    else:
        lora = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias="none",
            target_modules=list(cfg.target_modules),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora)
    wrapped = [n for n, _ in model.named_modules() if n.endswith(".lora_A")]
    if any("visual" in n for n in wrapped):
        raise RuntimeError("LoRA unexpectedly wrapped vision modules")
    model.print_trainable_parameters()
    device = model.device
    print(f"[model] loaded in {time.time() - t_job:.0f}s; {len(wrapped)} LoRA modules", flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda s: lr_lambda(s, cfg.warmup_steps, steps))
    step, micro, order = 0, 0, []
    wandb_id = None
    if cfg.resume:
        st = torch.load(cfg.resume / "state.pt", weights_only=False)
        optim.load_state_dict(st["optim"])
        sched.load_state_dict(st["sched"])
        step, micro, order = st["step"], st["micro"], st["order"]
        random.setstate(st["py_rng"])
        torch.set_rng_state(st["torch_rng"])
        if st["total_steps"] != steps or st.get("n_rows") != len(examples):
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
                "micro": micro,
                "order": order,
                "py_rng": random.getstate(),
                "torch_rng": torch.get_rng_state(),
                "total_steps": steps,
                "n_rows": len(examples),
            },
            d / "state.pt",
        )
        (d / "state.json").write_text(json.dumps({"step": step, "wandb_id": run.id if run else None}) + "\n")
        print(f"[ckpt] {d} at step {step}", flush=True)

    if step >= steps:
        print(f"[done] already at step {step}/{steps}; nothing to resume", flush=True)
        if run:
            run.finish()
        return summary
    if step == 0 and val:
        emit({"step": 0, **validate(model, val, device, word_ids)})
    model.train()  # also on resume, where no step-0 validation ran (eval mode turns gradient checkpointing off)
    acc_loss, acc_norm, acc_tok, acc_seq = 0.0, 0.0, 0, 0
    t_step = time.time()
    last_step_sec = 0.0
    while step < steps:
        if not order:
            order = random.sample(range(len(examples)), len(examples))
        ex = examples[order.pop()]
        logits = _forward(model, ex, device)
        res = compute_row_loss(_loss_inputs(ex, logits, device))
        if res.loss is not None:
            res.loss.backward()
            acc_loss += float(res.loss.item())
        acc_norm += res.normalizer
        acc_tok += res.n_loss_tokens
        acc_seq += int(ex.input_ids.shape[0])
        micro += 1
        if micro % cfg.grad_accum:
            continue
        if acc_norm > 0:
            for p in params:
                if p.grad is not None:
                    p.grad /= acc_norm
        gnorm = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip).item()
        optim.step()
        sched.step()
        optim.zero_grad(set_to_none=True)
        step += 1
        last_step_sec = time.time() - t_step
        emit(
            {
                "step": step,
                "train/loss": acc_loss / acc_norm if acc_norm else 0.0,
                "train/grad_norm": gnorm,
                "train/lr": sched.get_last_lr()[0],
                "train/epoch": micro / len(examples),
                "train/loss_tokens": acc_tok,
                "perf/sec": round(last_step_sec, 1),
                "perf/tokens_per_sec": round(acc_seq / max(last_step_sec, 1e-9), 1),
                "perf/max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1),
            }
        )
        acc_loss, acc_norm, acc_tok, acc_seq = 0.0, 0.0, 0, 0
        t_step = time.time()
        ep_done = micro // len(examples)
        if ep_done > (micro - cfg.grad_accum) // len(examples):  # an epoch ended inside this step
            model.save_pretrained(out / f"epoch-{ep_done}" / "adapter")
            print(f"[epoch] {ep_done} done at step {step}; adapter saved", flush=True)
            if val:
                emit({"step": step, "epoch": ep_done, **validate(model, val, device, word_ids)})
        elif cfg.val_every and step % cfg.val_every == 0 and step < steps and val:
            emit({"step": step, **validate(model, val, device, word_ids)})
        if step % cfg.save_every == 0 or step == steps:
            save("ckpt-last")
        # stop before the job's time limit: time since process start plus one more step must fit the budget
        if cfg.stop_after_min and step < steps and (time.time() - t_job + last_step_sec) / 60 > cfg.stop_after_min:
            save("ckpt-last")
            print(
                f"[stop] time budget reached at step {step}/{steps}; resume with resume: {out / 'ckpt-last'}",
                flush=True,
            )
            if run:
                run.finish()
            return summary
    if val and micro % len(examples):  # the last step did not end an epoch (max_steps / fractional epochs)
        emit({"step": step, **validate(model, val, device, word_ids)})
    best = best_epoch(out / "train_log.jsonl")
    if best:
        (out / "best_epoch.json").write_text(json.dumps(best) + "\n")
        ep_dir = out / f"epoch-{best['epoch']}" / "adapter"
        print(f"[epoch] lowest val/loss_all at epoch {best['epoch']}: {ep_dir} (use it for eval and DMPO)", flush=True)
    model.save_pretrained(out / "adapter")
    processor.save_pretrained(out / "adapter")
    if run:
        run.finish()
    print(f"[done] adapter -> {out / 'adapter'} ({time.time() - t_job:.0f}s total)", flush=True)
    return summary


def train_from_path(config_path: Path, dry_run: bool = False) -> dict[str, Any]:
    from geniac_baseline.trainer.config import load_config

    return train(load_config(config_path), dry_run=dry_run)
