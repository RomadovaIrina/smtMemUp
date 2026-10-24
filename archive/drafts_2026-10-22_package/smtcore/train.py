"""5. Training: one optimizer step per batch; stages SMT -> DMT, BPTT, LM; checkpoint selection; run identity.

Families (configs/*.json, field "family" in a run spec):
    smt           SMT (teacher + RNN jointly) then DMT (RNN only, teacher frozen)       -> final model: the RNN
    teacher_full  teacher only, L_dec only, decoder reads all encoder outputs (diagnostic)
    bptt_full     the same RNN from scratch, direct CE, full BPTT
    bptt_1        the same RNN from scratch, direct CE, BPTT-1
    transformer   MQAR causal LM, direct CE
Checkpoint selection (every stage): lowest CE on LONG answers of the validation set (stream 501 / 511); for SMT the
teacher's j = 0 CE (as archived), for DMT / BPTT / LM the model's own. The last weights are kept too, and both are
evaluated on the independent set (stream 901) at the end: summary.json has *_best_* and *_last_* numbers.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from .config import Config, load_config
from .data import evaluation_set, sample, validation_set
from .evaluate import answer_logits, long_metrics
from .losses import bptt_loss, dmt_loss, lm_loss, smt_loss, teacher_full_access_loss
from .models import copy_readout, load_context_detector, make_lm, make_models
from .upstream import PINNED, ROOT, derive_seed, seeded


def make_optimizer(params, lr: float, cfg: Config, steps: int):
    """AdamW (betas 0.9 / 0.95, wd 0.01); LR: linear warm-up over min(200, steps/10) steps, then constant."""
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    warm = max(1, min(cfg.warmup_steps, steps // 10))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 1.0)


def update(loss, params, opt, sched, cfg: Config) -> float:
    """(zero_grad was called before the forward) backward -> clip global norm to 1.0 -> AdamW.step -> scheduler.step"""
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
    opt.step(); sched.step()
    return float(norm)


def code_sha256() -> str:
    h = hashlib.sha256()
    for f in sorted((ROOT / "smtcore").glob("*.py")):
        h.update(f.name.encode()); h.update(f.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def identity(cfg: Config, spec: dict, device) -> dict:
    return dict(config=asdict(cfg), spec=spec, code_sha256=code_sha256(), upstream=PINNED, torch=torch.__version__,
                x_transformers=importlib.metadata.version("x-transformers"), python=platform.python_version(),
                device=device.type, gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)


def run_stage(name: str, steps: int, lr: float, params, step_fn, monitor_fn, models: dict, out: Path, cfg: Config,
              ident: dict, log, stop_key: str | None = None) -> dict:
    """step_fn(step) -> (loss, parts); every eval_every steps monitor_fn() -> (score, metrics). Keeps <name>_best.pt
    (lowest score) and <name>_last.pt (resumable), writes <name>_history.csv (train loss every step, validation every
    eval_every steps). Optional early stop: metrics[stop_key] >= cfg.stop_accuracy (rule set in the config)."""
    opt, sched = make_optimizer(params, lr, cfg, steps)
    last, best_path = out / f"{name}_last.pt", out / f"{name}_best.pt"
    state = dict(step=0, history=[], best=float("inf"), seconds=0.0, stopped=False)
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        assert saved["identity"] == json.loads(json.dumps(ident)), f"{last}: identity mismatch"
        for k, m in models.items():
            m.load_state_dict(saved["models"][k])
        opt.load_state_dict(saved["optimizer"]); sched.load_state_dict(saved["scheduler"])
        state = saved["state"]
        log(f"resume {out.name}/{name} at {state['step']}/{steps}")
    tic = time.perf_counter()
    step = state["step"]
    while step < steps and not state["stopped"]:
        opt.zero_grad(set_to_none=True)
        loss, parts = step_fn(step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} step {step}: {parts}")
        norm = update(loss, params, opt, sched, cfg)
        step += 1
        row = dict(step=step, lr=opt.param_groups[0]["lr"], grad_norm=norm,
                   **{f"train_{k}": float(v.detach()) for k, v in parts.items()})
        if step % cfg.eval_every == 0 or step == steps:
            state["seconds"] += time.perf_counter() - tic
            score, metrics = monitor_fn()
            row.update(metrics, train_seconds=state["seconds"])
            if score < state["best"]:
                state["best"] = score
                torch.save(dict(step=step, models={k: m.state_dict() for k, m in models.items()}), best_path)
            state["stopped"] = bool(stop_key and cfg.stop_accuracy > 0 and metrics[stop_key] >= cfg.stop_accuracy)
            log(f"{out.name}/{name} {step}/{steps} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k.endswith("acc")))
            state["history"].append(row); state["step"] = step
            torch.save(dict(identity=ident, models={k: m.state_dict() for k, m in models.items()},
                            optimizer=opt.state_dict(), scheduler=sched.state_dict(), state=state), last)
            pd.DataFrame(state["history"]).to_csv(out / f"{name}_history.csv", index=False)
            tic = time.perf_counter()
        else:
            state["history"].append(row)
    last_models = copy.deepcopy({k: m.state_dict() for k, m in models.items()})
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    for k, m in models.items():
        m.load_state_dict(best["models"][k])
    return dict(best_step=best["step"], last_step=state["step"], seconds=state["seconds"], last_models=last_models,
                stopped_early=state["stopped"])


def _both(summary: dict, prefix: str, r: dict, models: dict, evaluate_fn):
    """Evaluate the validation-selected ('best') and the last weights on the independent set."""
    best = {k: copy.deepcopy(m.state_dict()) for k, m in models.items()}
    summary.update({f"{prefix}_best_step": r["best_step"], f"{prefix}_last_step": r["last_step"],
                    f"{prefix}_train_seconds": r["seconds"], f"{prefix}_stopped_early": r["stopped_early"]})
    summary.update({k.replace("eval_", "eval_best_"): v for k, v in evaluate_fn().items()})
    for k, m in models.items():
        m.load_state_dict(r["last_models"][k])
    summary.update({k.replace("eval_", "eval_last_"): v for k, v in evaluate_fn().items()})
    for k, m in models.items():
        m.load_state_dict(best[k])


def run(spec: dict, out_root: Path, device, log=print) -> dict:
    """spec = {"name", "family", "seed", "config": {fields that differ from Config defaults}}. Writes to
    <out_root>/<name>/: identity.json, <stage>_history.csv, <stage>_best.pt, <stage>_last.pt, summary.json."""
    cfg = load_config(spec.get("config", {}))
    seed, family = int(spec["seed"]), spec["family"]
    out = out_root / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    ident = identity(cfg, spec, device)
    p = out / "identity.json"
    assert not p.exists() or json.loads(p.read_text()) == json.loads(json.dumps(ident)), f"{p}: another run lives here"
    p.write_text(json.dumps(ident, indent=2))
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    val, ev_set = validation_set(cfg), evaluation_set(cfg)
    teacher, rnn = make_models(cfg, seed, device)
    detector = load_context_detector(cfg, device) if cfg.far == "ctx_far" else None
    summary, tic = dict(name=spec["name"], family=family, seed=seed), time.perf_counter()

    def rnn_monitor():
        ev = answer_logits("rnn", rnn, val, cfg, device)
        return float(ev[ev.delay >= cfg.long_delay].ce.mean()), long_metrics(ev, cfg, "val_rnn")

    rnn_eval = lambda tag: lambda: long_metrics(answer_logits("rnn", rnn, ev_set, cfg, device), cfg, f"eval_rnn_{tag}")

    if family in ("smt", "teacher_full"):
        params = [*teacher.parameters(), *rnn.parameters()] if family == "smt" else list(teacher.parameters())
        teacher_kind = "teacher_full" if family == "teacher_full" else "teacher"

        def smt_step(step):
            teacher.train(); rnn.train()
            batch = sample(cfg, cfg.batch_size, derive_seed(301, seed, step))
            gen = torch.Generator().manual_seed(derive_seed(303, seed, step))       # top-K tie breaking
            with seeded(derive_seed(304, seed, step)):                              # L_unif's random subset
                if family == "teacher_full":
                    return teacher_full_access_loss(teacher, batch, cfg, device)
                return smt_loss(teacher, rnn, batch, cfg, device, gen, detector)

        def smt_monitor():
            copy_readout(teacher, rnn)
            ev = answer_logits(teacher_kind, teacher, val, cfg, device)
            metrics = long_metrics(ev, cfg, "val_teacher")
            if family == "smt":
                metrics.update(long_metrics(answer_logits("rnn", rnn, val, cfg, device), cfg, "val_rnn"))
                if cfg.task == "mqar":
                    metrics.update(long_metrics(answer_logits("teacher_pre", teacher, val, cfg, device, "after_pairs"),
                                                cfg, "val_teacher_pre"))
            return float(ev[ev.delay >= cfg.long_delay].ce.mean()), metrics

        def smt_eval():
            copy_readout(teacher, rnn)
            kinds = [(teacher_kind, teacher, "query")]
            if cfg.task == "mqar" and family == "smt":
                kinds += [("teacher_pre", teacher, "after_pairs"), ("teacher_pre", teacher, "before_query")]
            if family == "smt":
                kinds.append(("rnn", rnn, "query"))       # "RNN before DMT": readout copied from the teacher
            out_ = {}
            for kind, model, anchor in kinds:
                tag = kind if anchor == "query" else f"{kind}_{anchor}"
                out_.update(long_metrics(answer_logits(kind, model, ev_set, cfg, device, anchor), cfg, f"eval_{tag}_smt"))
            return out_

        r = run_stage("smt", cfg.smt_steps, cfg.smt_lr, params, smt_step, smt_monitor,
                      dict(teacher=teacher, rnn=rnn), out, cfg, ident, log, "val_teacher_long_acc")
        _both(summary, "smt", r, dict(teacher=teacher, rnn=rnn), smt_eval)
        copy_readout(teacher, rnn)
        if family == "smt" and cfg.dmt_steps > 0:
            teacher.requires_grad_(False); teacher.eval()
            frozen = copy.deepcopy(teacher.state_dict())

            def dmt_step(step):
                rnn.train()
                batch = sample(cfg, cfg.batch_size, derive_seed(302, seed, step))
                with seeded(derive_seed(305, seed, step)):
                    return dmt_loss(teacher, rnn, batch, cfg, device)

            r = run_stage("dmt", cfg.dmt_steps, cfg.dmt_lr, list(rnn.parameters()), dmt_step, rnn_monitor,
                          dict(rnn=rnn), out, cfg, ident, log, "val_rnn_long_acc")
            assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in frozen.items()), "teacher changed in DMT"
            _both(summary, "dmt", r, dict(rnn=rnn), rnn_eval("dmt"))
    elif family in ("bptt_full", "bptt_1"):
        def bptt_step(step):
            rnn.train()
            batch = sample(cfg, cfg.batch_size, derive_seed(301, seed, step))
            with seeded(derive_seed(306, seed, step)):
                return bptt_loss(rnn, batch, cfg, device, truncate=family == "bptt_1")

        r = run_stage("bptt", cfg.bptt_steps, cfg.bptt_lr, list(rnn.parameters()), bptt_step, rnn_monitor,
                      dict(rnn=rnn), out, cfg, ident, log, "val_rnn_long_acc")
        _both(summary, "bptt", r, dict(rnn=rnn), rnn_eval("bptt"))
    elif family == "transformer":
        model = make_lm(cfg, seed, device)

        def lm_step(step):
            model.train()
            return lm_loss(model, sample(cfg, cfg.batch_size, derive_seed(301, seed, step)), device)

        def lm_monitor():
            model.eval()
            ev = answer_logits("lm", model, val, cfg, device)
            return float(ev[ev.delay >= cfg.long_delay].ce.mean()), long_metrics(ev, cfg, "val_lm")

        def lm_eval():
            model.eval()
            return long_metrics(answer_logits("lm", model, ev_set, cfg, device), cfg, "eval_lm")

        r = run_stage("lm", cfg.lm_steps, cfg.lm_lr, list(model.parameters()), lm_step, lm_monitor,
                      dict(model=model), out, cfg, ident, log, "val_lm_long_acc")
        _both(summary, "lm", r, dict(model=model), lm_eval)
    else:
        raise ValueError(family)
    summary["wall_seconds"] = time.perf_counter() - tic
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary
