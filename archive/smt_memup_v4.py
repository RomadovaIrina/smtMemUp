"""SMT x MemUP, version 4 (Stack, no noise): the advisor's target setting plus recurrent BPTT references.

Target setting (approved by the advisor, 16.10.2026): keep SMT with a SHORT future window Tf and add MemUP-like
prediction of FAR targets beyond the window from the memory m_t and the decoder's intermediate inputs. This is the
loss of version 2 (smt_memup_window.smt_window_loss, variant P2 of proposals/smt_window_memup_draft.md):

    L = coef_dec * (L_dec^{Tf} + alpha * L_far) + coef_dyn * L_dyn + coef_unif * L_unif

Two windows must not be confused:
  * Tf (cfg.future_window = 8): horizon of the base SMT predictive loss;
  * r  (cfg.local_window  = 8): local context of the window surprise detector (surprise_far only).
A short Tf does NOT truncate the decoder inputs: a far target k >= t + Tf is predicted from m_t and the causal
inputs x_{t+1..k}.

New in version 4:
  * BPTT references. The SAME recurrent model as the final SMT -> DMT model (author RNN class, same updater, state
    size 8 x 64 and readout head), trained FROM SCRATCH directly on the target answers (CE on pops), without a teacher
    and without DMT:
      - bptt_full: one optimizer step per batch; the gradient flows through all T = 64 updates. The loss equals the
                   authors' forward_bptt (upstream src/model.py; checked in checks/test_smt_memup_v4.py);
      - bptt_1   : the state is carried forward (identical forward values), but the incoming state of every update is
                   detached, so the loss at t trains only the update applied at t (and the readout). The state is NOT
                   reset to zero. One optimizer step per batch, after the losses of all T steps are summed.
    Where the detach is: S.rnn_rollout(..., detach=True) calls rnn.rnn_update(state.detach(), x_t).
    Order inside a step: zero_grad -> rollout over the whole sequence -> masked mean CE -> backward -> clip ->
    optimizer.step -> scheduler.step. These are empirical references, not mathematical bounds.
  * Diagnostic phase on the tuning seed (107) with LONG maximum budgets, constant LR after warm-up (authors' SMT
    default) and sufficiency rules stated before the runs (proposals/preregistration_v4.md). Its outcome fixes the
    budgets of the final comparison by a rule written in advance.
  * Cost accounting per stage: optimizer steps, training sequences, wall time (train and evaluation separately),
    peak allocated / reserved CUDA memory. SMT cost = SMT + DMT + (amortised) detector training.

smt_memup_stack.py and smt_memup_window.py are imported, not modified (their hashes identify versions 1 and 2).
Importing this file never trains anything.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import smt_memup_stack as S
import smt_memup_window as W

MODULE_PATH = Path(__file__).resolve()
DEFAULT_ROOT = MODULE_PATH.parent
SMT_VARIANTS = ("base_window", "uniform_far", "random_far", "surprise_far", "ctx_far")
SMT_OPTIONAL = ("oracle_far",)                       # ORACLE diagnostic (generator metadata restricts the pool)
BPTT_VARIANTS = ("bptt_full", "bptt_1")
ALL_VARIANTS = SMT_VARIANTS + SMT_OPTIONAL + BPTT_VARIANTS
DIAG_SMT = ("base_window", "ctx_far")                # diagnostic SMT runs (tuning seed)


@dataclass(frozen=True)
class V4Config(W.WindowConfig):
    # SMT and DMT: maximum budgets of the diagnostic; the final budgets are fixed by the diagnostic rule
    smt_steps: int = 20000
    dmt_steps: int = 12000
    smt_lr_schedule: str = "warmup_constant"     # authors' default (train_smt.py); also version 3
    dmt_lr_schedule: str = "warmup_constant"     # constant LR: a curve point = "stopped here" (cosine ties it to N)
    # BPTT references (same RNN, from scratch)
    bptt_steps: int = 20000
    bptt_batch_size: int = 32
    bptt_lr: float = 3e-4                        # replaced by the LR chosen in the diagnostic
    bptt_lr_grid: tuple = (1e-4, 3e-4, 1e-3)
    bptt_lr_schedule: str = "warmup_constant"
    # protocol
    eval_every: int = 500
    seeds: tuple = (67, 77, 87)                  # new final seeds (7/17/27: v1, v2; 37/47/57: v3)
    test_stream: int = 703                       # v1/v2 test: 503, v3 test: 603
    # sufficiency rule (stated before the diagnostic): gain of the best validation long-answer accuracy over the last
    # `plateau_fraction` of the budget below `plateau_pp`, or level >= `ceiling_acc`
    plateau_fraction: float = 0.25
    plateau_pp: float = 0.01
    ceiling_acc: float = 0.99


def code_sha256() -> str:
    return hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()


def run_identity(cfg: V4Config, device: torch.device) -> dict:
    ident = W.run_identity(cfg, device)                 # config, stack + window code hashes, upstream, versions, device
    ident.update(task="stack_v4_short_window_bptt", v4_code_sha256=code_sha256())
    return ident


def lr_lambda(cfg: V4Config, steps: int, schedule: str):
    warm = max(1, min(cfg.warmup_steps, steps // 10))
    if schedule == "warmup_constant":
        return lambda s: (s + 1) / warm if s < warm else 1.0
    if schedule == "cosine":
        return S.lr_lambda(cfg, steps)
    raise ValueError(schedule)


# --------------------------------------------------------------------------------------
# BPTT references
# --------------------------------------------------------------------------------------

def bptt_loss(rnn, batch: dict, cfg: V4Config, device, truncate: bool):
    """Masked mean CE over pops of the RNN rolled out from the zero state. truncate=False: full BPTT over the
    sequence; truncate=True (BPTT-1): the same forward values, the incoming state of every update is detached."""
    x = batch["x"].to(device)
    states = S.rnn_rollout(rnn, x, detach=truncate)                          # (B, T, Z)
    logits = rnn.rnn_readout(states.flatten(0, 1)).reshape(*x.shape, -1)     # same head as the SMT -> DMT model
    ce = S.sources().model.calc_cross_entropy_loss(logits, batch["y"].to(device).clamp(min=0))
    mask = batch["target_mask"].to(device)
    loss = S.pair_mean(ce, mask)
    return loss, dict(loss=loss, readout=loss, state_norm=states.detach().norm(dim=-1).mean())


@torch.no_grad()
def evaluate_rnn(rnn, batch: dict, cfg: V4Config, device, **labels) -> pd.DataFrame:
    """Same rows/columns as S.evaluate for model='rnn' (own head). No teacher: shared_* and memory_mse are NaN."""
    rnn.eval()
    frames = []
    for lo in range(0, len(batch["x"]), cfg.eval_batch_size):
        part = S.slice_batch(batch, lo, lo + cfg.eval_batch_size)
        rm = S.rnn_rollout(rnn, part["x"].to(device))
        y = part["y"].to(device)
        own = rnn.rnn_readout(rm.flatten(0, 1)).reshape(*y.shape, -1).float()
        ce = F.cross_entropy(own.flatten(0, 1), y.clamp(min=0).flatten(), reduction="none").reshape(y.shape).cpu()
        pred = own.argmax(-1).cpu()
        b, t = torch.where(part["target_mask"])
        n = len(b)
        frames.append(pd.DataFrame(dict(
            **labels, model="rnn", sequence=(b + lo).numpy(), t=t.numpy(), delay=part["delay"][b, t].numpy(),
            depth_after=part["depth"][b, t].numpy(), target=part["y"][b, t].numpy(),
            correct=(pred[b, t] == part["y"][b, t]).numpy(), ce=ce[b, t].numpy(),
            shared_correct=np.full(n, np.nan), shared_ce=np.full(n, np.nan), memory_mse=np.full(n, np.nan))))
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------------------
# Training (resumable at every evaluation; best checkpoint by validation CE on long answers)
# --------------------------------------------------------------------------------------

def _cuda_peaks(device) -> dict:
    if device.type != "cuda":
        return {}
    return dict(peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)


def _long_metrics(ev: pd.DataFrame, cfg: V4Config, model: str) -> dict:
    lg = ev[(ev.model == model) & (ev.delay >= cfg.long_delay)]
    return {f"val_{model}_long_ce": float(lg.ce.mean()), f"val_{model}_long_acc": float(lg.correct.mean()),
            f"val_{model}_all_acc": float(ev[ev.model == model].correct.mean())}


def _run_stage(stage: str, steps: int, lr: float, schedule: str, params, step_fn, monitor_fn, save_models, load_models,
               identity: dict, initial_hash: str, out: Path, cfg: V4Config, device, log) -> dict:
    """Generic resumable loop. step_fn(step) -> (loss, parts); monitor_fn() -> (score, row extras)."""
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, steps, schedule))
    last, best_path = out / f"{stage}_last.pt", out / f"{stage}_best.pt"
    start, history, best, cost = 0, [], float("inf"), dict(train_seconds=0.0, eval_seconds=0.0)
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity and saved["initial_hash"] == initial_hash
        load_models(saved["models"])
        opt.load_state_dict(saved["optimizer"]); sched.load_state_dict(saved["scheduler"])
        start, history, best, cost = saved["step"], saved["history"], saved["best"], saved["cost"]
        log(f"resume {out.name} {stage} at {start}/{steps}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    tic = time.perf_counter()
    for step in range(start, steps):
        opt.zero_grad(set_to_none=True)
        loss, parts = step_fn(step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{out.name} {stage} step {step}: {parts}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
        opt.step(); sched.step()
        row = dict(step=step + 1, lr=opt.param_groups[0]["lr"], grad_norm=float(norm),
                   **{k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()})
        monitor = (step + 1) % cfg.eval_every == 0 or step + 1 == steps
        if monitor:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_eval = time.perf_counter()
            cost["train_seconds"] += t_eval - tic
            score, extras = monitor_fn()
            row.update(extras)
            if score < best:
                best = score
                torch.save(dict(identity=identity, step=step + 1, models=save_models(), score=score), best_path)
            tic = time.perf_counter()
            cost["eval_seconds"] += tic - t_eval
            for k, v in _cuda_peaks(device).items():          # peaks include evaluation; kept across resumes
                cost[k] = max(cost.get(k, 0.0), v)
            row.update(train_seconds=cost["train_seconds"], eval_seconds=cost["eval_seconds"], **_cuda_peaks(device))
            log(f"{out.parent.name}/{out.name} {stage} {step + 1}/{steps} select={score:.4f} "
                + " ".join(f"{k}={v:.3f}" for k, v in extras.items() if k.endswith("long_acc") or k.endswith("memory_acc"))
                + f" {cost['train_seconds'] / max(1, step + 1):.3f}s/step(train)")
        history.append(row)
        if monitor:
            torch.save(dict(identity=identity, initial_hash=initial_hash, step=step + 1, models=save_models(),
                            optimizer=opt.state_dict(), scheduler=sched.state_dict(), history=history, best=best,
                            cost=cost), last)
            pd.DataFrame(history).to_csv(out / f"{stage}_history.csv", index=False)
    final_models = copy.deepcopy(save_models())
    best_saved = torch.load(best_path, map_location="cpu", weights_only=True)
    load_models(best_saved["models"])
    return dict(best_step=best_saved["step"], last_models=final_models, steps=steps,
                cost=dict(cost, optimizer_steps=steps))


def fit_smt(cfg: V4Config, variant: str, seed: int, out: Path, detectors: dict, validation: dict, device,
            log=print) -> dict:
    """SMT (short window + far targets of `variant`) then DMT. Same data streams, initial weights, detectors and
    evaluation as version 2; schedules and budgets of version 4."""
    assert variant in SMT_VARIANTS + SMT_OPTIONAL
    out.mkdir(parents=True, exist_ok=True)
    det_hash = {k: S.state_hash(v) for k, v in detectors.items()
                if (k == "window" and variant == "surprise_far") or (k == "context" and variant == "ctx_far")}
    identity = dict(run=run_identity(cfg, device), family="smt", variant=variant, seed=int(seed), detectors=det_hash,
                    validation=S.split_hash(validation))
    done = out / "completed.pt"
    if done.exists():
        saved = torch.load(done, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, f"{done}: identity mismatch"
        return saved
    teacher, rnn = S.make_models(cfg, seed, device)
    initial_hash = S.state_hash(teacher, rnn)
    result = dict(identity=identity, initial_hash=initial_hash, cost={})
    save = lambda: dict(teacher=teacher.state_dict(), rnn=rnn.state_dict())

    def load(m):
        teacher.load_state_dict(m["teacher"]); rnn.load_state_dict(m["rnn"])

    for stage in ("smt", "dmt"):
        is_smt = stage == "smt"
        teacher.requires_grad_(is_smt)
        if not is_smt:
            S.copy_readout(teacher, rnn)
            frozen = S.state_hash(teacher)
        params = [*teacher.parameters(), *rnn.parameters()] if is_smt else list(rnn.parameters())
        bs = cfg.batch_size if is_smt else cfg.dmt_batch_size

        def step_fn(step, is_smt=is_smt, bs=bs):
            teacher.train(is_smt); rnn.train()
            batch = S.sample_stack(cfg, bs, S.derive_seed(301 if is_smt else 302, seed, step))
            gen = torch.Generator().manual_seed(S.derive_seed(303, seed, step))
            with S.isolated_seed(S.derive_seed(304 if is_smt else 305, seed, step)):
                return (W.smt_window_loss(teacher, rnn, batch, cfg, variant, detectors, gen, device) if is_smt
                        else S.dmt_loss(teacher, rnn, batch, cfg, device))

        def monitor_fn(is_smt=is_smt):
            if is_smt:
                S.copy_readout(teacher, rnn)
            ev = S.evaluate(teacher, rnn, validation, cfg, device)
            extras = {**_long_metrics(ev, cfg, "teacher"), **_long_metrics(ev, cfg, "rnn")}
            if is_smt:
                fp = W.far_probe(teacher, S.slice_batch(validation, 0, 128), cfg, device)
                extras["val_far_memory_acc"] = float(fp.correct.mean())
            return S.long_ce(ev, cfg, "teacher" if is_smt else "rnn"), extras

        r = _run_stage(stage, cfg.smt_steps if is_smt else cfg.dmt_steps, cfg.smt_lr if is_smt else cfg.dmt_lr,
                       cfg.smt_lr_schedule if is_smt else cfg.dmt_lr_schedule, params, step_fn, monitor_fn, save, load,
                       identity, initial_hash, out, cfg, device, log)
        result[f"{stage}_last_teacher"], result[f"{stage}_last_rnn"] = r["last_models"]["teacher"], r["last_models"]["rnn"]
        result[f"{stage}_best_step"] = r["best_step"]
        result["cost"][stage] = dict(r["cost"], sequences=r["steps"] * bs)
        if is_smt:
            S.copy_readout(teacher, rnn)
            result["rnn_smt"] = copy.deepcopy(rnn.state_dict())
        else:
            assert S.state_hash(teacher) == frozen, "teacher changed during DMT"
            result["rnn_dmt"] = copy.deepcopy(rnn.state_dict())
    if variant in ("surprise_far", "ctx_far"):     # detectors were trained once (version 1) and are reused: amortised
        result["cost"]["detector"] = dict(optimizer_steps=cfg.detector_steps,
                                          sequences=cfg.detector_steps * cfg.detector_batch_size, shared_across_fits=True)
    result["teacher"] = teacher.state_dict()
    (out / "cost.json").write_text(json.dumps(result["cost"], indent=2))
    torch.save(result, done)
    return result


def fit_bptt(cfg: V4Config, variant: str, seed: int, lr: float, out: Path, validation: dict, device, log=print) -> dict:
    """The RNN of the SMT pipeline (same class, size, head and initial weights for the same seed), trained from scratch
    on the target answers with full BPTT or BPTT-1. Training data: the SMT stream (same sequences step by step)."""
    assert variant in BPTT_VARIANTS
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(run=run_identity(cfg, device), family="bptt", variant=variant, seed=int(seed), lr=float(lr),
                    validation=S.split_hash(validation))
    done = out / "completed.pt"
    if done.exists():
        saved = torch.load(done, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, f"{done}: identity mismatch"
        return saved
    _, rnn = S.make_models(cfg, seed, device)          # the teacher is created only to reproduce the RNN init
    initial_hash = S.state_hash(rnn)
    truncate = variant == "bptt_1"
    save = lambda: dict(rnn=rnn.state_dict())
    load = lambda m: rnn.load_state_dict(m["rnn"])

    def step_fn(step):
        rnn.train()
        batch = S.sample_stack(cfg, cfg.bptt_batch_size, S.derive_seed(301, seed, step))
        with S.isolated_seed(S.derive_seed(306, seed, step)):
            return bptt_loss(rnn, batch, cfg, device, truncate)

    def monitor_fn():
        ev = evaluate_rnn(rnn, validation, cfg, device)
        return S.long_ce(ev, cfg, "rnn"), _long_metrics(ev, cfg, "rnn")

    r = _run_stage("bptt", cfg.bptt_steps, lr, cfg.bptt_lr_schedule, list(rnn.parameters()), step_fn, monitor_fn,
                   save, load, identity, initial_hash, out, cfg, device, log)
    result = dict(identity=identity, initial_hash=initial_hash, rnn=rnn.state_dict(), bptt_best_step=r["best_step"],
                  bptt_last_rnn=r["last_models"]["rnn"],
                  cost=dict(bptt=dict(r["cost"], sequences=r["steps"] * cfg.bptt_batch_size)))
    (out / "cost.json").write_text(json.dumps(result["cost"], indent=2))
    torch.save(result, done)
    return result


def load_bptt(cfg: V4Config, saved: dict, device, which: str = "best"):
    _, rnn = S.make_models(cfg, 0, device)
    rnn.load_state_dict(saved["rnn"] if which == "best" else saved["bptt_last_rnn"])
    return rnn.eval()


# --------------------------------------------------------------------------------------
# Diagnostic: sufficiency of the budgets (CPU-safe: reads histories only)
# --------------------------------------------------------------------------------------

def sufficiency(history: pd.DataFrame, column: str, cfg: V4Config) -> dict:
    """Pre-stated rule. best(s) = best validation long-answer accuracy up to step s (running max).
    sufficient  <=>  best(N) - best((1 - plateau_fraction) * N) < plateau_pp   or   best(N) >= ceiling_acc.
    steps_to_plateau = first evaluated step with best(s) >= best(N) - plateau_pp."""
    h = history.dropna(subset=[column])
    if not len(h):
        return dict(evaluated_steps=0)
    steps, acc = h.step.to_numpy(), np.maximum.accumulate(h[column].to_numpy())
    n = int(steps.max())
    cut = (1 - cfg.plateau_fraction) * n
    before = acc[steps <= cut].max() if (steps <= cut).any() else 0.0
    gain = float(acc[-1] - before)
    reach = steps[acc >= acc[-1] - cfg.plateau_pp]
    return dict(evaluated_steps=n, best_acc=float(acc[-1]), gain_last_fraction=gain,
                sufficient=bool(gain < cfg.plateau_pp or acc[-1] >= cfg.ceiling_acc),
                steps_to_plateau=int(reach[0]) if len(reach) else None)


def final_budget(steps_to_plateau: list[int], maximum: int, round_to: int = 1000) -> int:
    """Pre-stated rule: 1.25 x the largest steps-to-plateau among the diagnostic runs of a family, rounded up to
    `round_to`, never above the diagnostic maximum."""
    s = max(steps_to_plateau)
    return int(min(maximum, math.ceil(1.25 * s / round_to) * round_to))


def diagnostic_summary(run_dir: Path, cfg: V4Config) -> tuple[pd.DataFrame, dict]:
    rows = []
    for d in sorted((Path(run_dir) / "diag").glob("*/")):
        for stage, col in (("smt", "val_teacher_long_acc"), ("dmt", "val_rnn_long_acc"), ("bptt", "val_rnn_long_acc")):
            f = d / f"{stage}_history.csv"
            if f.exists():
                h = pd.read_csv(f)
                last = h.dropna(subset=[col]).iloc[-1] if len(h.dropna(subset=[col])) else {}
                rows.append(dict(run=d.name, stage=stage, metric=col, complete=(d / "completed.pt").exists(),
                                 **sufficiency(h, col, cfg),
                                 far_memory_acc_best=float(h["val_far_memory_acc"].max()) if "val_far_memory_acc" in h else np.nan,
                                 train_seconds=float(last.get("train_seconds", np.nan)) if len(last) else np.nan,
                                 eval_seconds=float(last.get("eval_seconds", np.nan)) if len(last) else np.nan,
                                 peak_allocated_mib=float(h["peak_allocated_mib"].max()) if "peak_allocated_mib" in h else np.nan,
                                 peak_reserved_mib=float(h["peak_reserved_mib"].max()) if "peak_reserved_mib" in h else np.nan))
    table = pd.DataFrame(rows)
    decision = {}
    if len(table):
        done = table[table.complete]
        for family, stage, runs in (("smt", "smt", list(DIAG_SMT)), ("dmt", "dmt", list(DIAG_SMT))):
            sub = done[(done.stage == stage) & done.run.isin(runs)]
            if len(sub) == len(runs):
                decision[family] = dict(sufficient=bool(sub.sufficient.all()),
                                        budget=final_budget(list(sub.steps_to_plateau), cfg.smt_steps if stage == "smt" else cfg.dmt_steps))
        for v in BPTT_VARIANTS:
            sub = done[(done.stage == "bptt") & done.run.str.startswith(v + "_lr")]
            if len(sub) == len(cfg.bptt_lr_grid):
                decision[v] = dict(lr_selection="tuning split, RNN long-answer CE (see diag_lr_selection.csv)")
    return table, decision


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

class Experiment:
    def __init__(self, cfg: V4Config, root: Path | str = DEFAULT_ROOT, device=None, log=print):
        self.cfg, self.root, self.log = cfg, Path(root), log
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        S.sources(self.root)
        ident = run_identity(cfg, self.device)
        key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = self.root / "runs" / f"stackv4_{key}"
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "identity.json"
        if p.exists():
            assert json.loads(p.read_text()) == ident
        else:
            p.write_text(json.dumps(ident, indent=2))
        self.validation = S.sample_stack(cfg, cfg.validation_examples, S.derive_seed(501, 0))   # as versions 1-3
        self.tuning = S.sample_stack(cfg, cfg.tuning_examples, S.derive_seed(502, 0))
        self.detectors = W.load_v1_detectors(cfg, self.root, self.device)

    # -- diagnostic (tuning seed): base_window, ctx_far, bptt_full and bptt_1 over the LR grid
    def diag_job(self, name: str) -> dict:
        cfg, seed = self.cfg, self.cfg.tuning_seed
        out = self.dir / "diag" / name
        if name in DIAG_SMT:
            return fit_smt(cfg, name, seed, out, self.detectors, self.validation, self.device, self.log)
        variant, lr = name.split("_lr")
        assert variant in BPTT_VARIANTS and float(lr) in cfg.bptt_lr_grid, name
        return fit_bptt(cfg, variant, seed, float(lr), out, self.validation, self.device, self.log)

    def diag_jobs(self) -> list[str]:
        return list(DIAG_SMT) + [f"{v}_lr{lr:g}" for v in BPTT_VARIANTS for lr in self.cfg.bptt_lr_grid]

    def diag_lr_selection(self) -> pd.DataFrame:
        """Pre-stated: for each BPTT variant, the LR whose validation-selected checkpoint has the lowest RNN CE on long
        answers of the TUNING split (as alpha in version 1). Ties -> the smaller LR."""
        rows = []
        for v in BPTT_VARIANTS:
            for lr in self.cfg.bptt_lr_grid:
                d = self.dir / "diag" / f"{v}_lr{lr:g}" / "completed.pt"
                if not d.exists():
                    continue
                saved = torch.load(d, map_location="cpu", weights_only=True)
                ev = evaluate_rnn(load_bptt(self.cfg, saved, self.device), self.tuning, self.cfg, self.device)
                lg = ev[ev.delay >= self.cfg.long_delay]
                rows.append(dict(variant=v, lr=lr, tuning_rnn_long_ce=float(lg.ce.mean()),
                                 tuning_rnn_long_acc=float(lg.correct.mean()), best_step=saved["bptt_best_step"]))
        table = pd.DataFrame(rows)
        if len(table):
            table["selected"] = False
            for v, g in table.groupby("variant"):
                if len(g) == len(self.cfg.bptt_lr_grid):
                    table.loc[g.sort_values(["tuning_rnn_long_ce", "lr"]).index[0], "selected"] = True
            table.to_csv(self.dir / "diag_lr_selection.csv", index=False)
        return table

    def diag_report(self) -> dict:
        table, decision = diagnostic_summary(self.dir, self.cfg)
        table.to_csv(self.dir / "diag_summary.csv", index=False)
        lr = self.diag_lr_selection()
        for v in BPTT_VARIANTS:
            sel = lr[(lr.variant == v) & lr.selected] if len(lr) else lr
            if len(sel):
                chosen = float(sel.lr.iloc[0])
                row = table[(table.run == f"{v}_lr{chosen:g}") & (table.stage == "bptt")].iloc[0]
                decision[v] = dict(lr=chosen, sufficient=bool(row.sufficient),
                                   budget=final_budget([int(row.steps_to_plateau)], self.cfg.bptt_steps))
        (self.dir / "diag_decision.json").write_text(json.dumps(decision, indent=2))
        return dict(table=table, lr=lr, decision=decision)

    # -- final comparison: only after the protocol is frozen from the diagnostic (proposals/preregistration_v4.md)
    def final_config(self) -> V4Config:
        dec = json.loads((self.dir / "diag_decision.json").read_text())
        missing = [k for k in ("smt", "dmt", *BPTT_VARIANTS) if k not in dec]
        assert not missing, f"diagnostic incomplete: {missing}"
        return replace(self.cfg, smt_steps=dec["smt"]["budget"], dmt_steps=dec["dmt"]["budget"],
                       bptt_steps=max(dec[v]["budget"] for v in BPTT_VARIANTS))

    def final_job(self, seed: int, variant: str) -> dict:
        cfg = self.final_config()
        assert seed in cfg.seeds
        out = self.dir / "final" / f"seed{seed}" / variant
        if variant in BPTT_VARIANTS:
            lr = json.loads((self.dir / "diag_decision.json").read_text())[variant]["lr"]
            return fit_bptt(cfg, variant, seed, lr, out, self.validation, self.device, self.log)
        return fit_smt(cfg, variant, seed, out, self.detectors, self.validation, self.device, self.log)

    def final_report(self, variants, confirm_test: bool = False) -> dict:
        """The only place where the version-4 test split is generated (stream 703; disjoint from validation, tuning and
        the version-1/2 and version-3 tests). The final RNNs of all families are compared on the same sequences."""
        assert confirm_test, "pass confirm_test=True only with the user's permission; never tune after this"
        import h2_retention as H
        cfg = self.final_config()
        test = S.sample_stack(cfg, cfg.test_examples, S.derive_seed(cfg.test_stream, 0))
        seen = set(S.fingerprints(self.validation)) | set(S.fingerprints(self.tuning))
        for stream in (503, 603):
            seen |= set(S.fingerprints(S.sample_stack(cfg, cfg.test_examples, S.derive_seed(stream, 0))))
        assert not set(S.fingerprints(test)) & seen, "version-4 test overlaps an earlier split"
        frames, probes, h2, costs = [], [], [], []
        for seed in cfg.seeds:
            for v in variants:
                saved = self.final_job(seed, v)
                costs.append(dict(seed=seed, variant=v, **{f"{st}_{k}": val for st, c in saved["cost"].items()
                                                           for k, val in c.items()}))
                for which in ("best", "last"):
                    if v in BPTT_VARIANTS:
                        frames.append(evaluate_rnn(load_bptt(cfg, saved, self.device, which), test, cfg, self.device,
                                                   seed=seed, variant=v, stage="final", split="test", checkpoint=which))
                        continue
                    for stage in ("smt", "dmt"):
                        t, r = S.load_fitted(cfg, saved, self.device, stage, which)
                        ev = S.evaluate(t, r, test, cfg, self.device, seed=seed, variant=v, split="test", checkpoint=which)
                        ev["stage"] = "final" if stage == "dmt" else stage
                        if stage == "dmt":
                            ev = ev[ev.model == "rnn"]              # teacher is frozen in DMT: rows equal stage smt
                        frames.append(ev)
                if v not in BPTT_VARIANTS:
                    t, r = S.load_fitted(cfg, saved, self.device, "smt", "best")
                    fp = W.far_probe(t, test, cfg, self.device)
                    probes.append(dict(seed=seed, variant=v, far_pairs=len(fp), far_memory_acc=float(fp.correct.mean()),
                                       far_memory_ce=float(fp.ce.mean())))
                    t, r = S.load_fitted(cfg, saved, self.device, "dmt", "best")
                    pr = H.retention_by_source(t, r, test, cfg, self.device)
                    pr["seed"], pr["variant"], pr["checkpoint"] = seed, v, "best"
                    h2.append(H.summarize_retention(pr, cfg))
                self.log(f"evaluated seed{seed}/{v}")
        events = pd.concat(frames, ignore_index=True)
        events["shared_correct"] = events["shared_correct"].astype(float)     # NaN for BPTT rows (no teacher head)
        comps = [c for c in COMPARISONS if c[0] in variants and c[1] in variants]
        intervals, per_seed = S.paired_deltas(events[events.checkpoint == "best"], cfg, comps,
                                              metrics=("ce", "correct"))
        summary = S.summarize(events, cfg, keys=("checkpoint", "seed", "variant", "stage", "model"))
        events.to_csv(self.dir / "test_events.csv.gz", index=False, compression="gzip")
        for name, df in (("test_summary", summary), ("test_intervals", intervals), ("test_per_seed", per_seed),
                         ("test_far_memory", pd.DataFrame(probes)), ("cost", pd.DataFrame(costs)),
                         ("h2_test_summary", pd.concat(h2, ignore_index=True) if h2 else pd.DataFrame())):
            df.to_csv(self.dir / f"{name}.csv", index=False)
        report = dict(variants=list(variants), seeds=list(cfg.seeds), test_hash=S.split_hash(test),
                      budgets=dict(smt=cfg.smt_steps, dmt=cfg.dmt_steps, bptt=cfg.bptt_steps),
                      primary="final RNN (stage 'final'), CE on pops with delay >= 8; see proposals/preregistration_v4.md")
        (self.dir / "test_report.json").write_text(json.dumps(report, indent=2))
        return dict(events=events, intervals=intervals, per_seed=per_seed, report=report)


# Comparisons of the final protocol (reference, variant): variant - reference on the same test sequences.
COMPARISONS = (
    [("base_window", v) for v in ("uniform_far", "random_far", "surprise_far", "ctx_far")]     # far supervision
    + [("random_far", "surprise_far"), ("random_far", "ctx_far"), ("surprise_far", "ctx_far")]  # selection
    + [("bptt_1", v) for v in SMT_VARIANTS] + [(v, "bptt_full") for v in SMT_VARIANTS]          # BPTT references
    + [("bptt_1", "bptt_full")]
    + [(r, "oracle_far") for r in ("random_far", "ctx_far")]                                     # ORACLE diagnostic
)


def main(argv=None):
    p = argparse.ArgumentParser(description="SMT x MemUP version 4: short window + far targets + BPTT references (Stack)")
    p.add_argument("--phase", required=True, choices=["diag", "diag_report", "final", "report", "list"])
    p.add_argument("--job", help="diagnostic job name (see --phase list)")
    p.add_argument("--seed", type=int)
    p.add_argument("--variant")
    p.add_argument("--variants", default=",".join(SMT_VARIANTS + BPTT_VARIANTS))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-memory-fraction", type=float, default=None)
    p.add_argument("--confirm-test", action="store_true")
    p.add_argument("--override", default="{}", help="JSON Config overrides (a NEW run directory)")
    a = p.parse_args(argv)
    device = S.setup_runtime(a.threads, a.gpu_memory_fraction)
    cfg = replace(V4Config(), **{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.override).items()})
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    ex = Experiment(cfg, device=device, log=log)
    log(f"run directory: {ex.dir}")
    if a.phase == "list":
        print("\n".join(ex.diag_jobs()))
    elif a.phase == "diag":
        ex.diag_job(a.job)
    elif a.phase == "diag_report":
        r = ex.diag_report()
        pd.set_option("display.width", 250)
        print(r["table"].to_string()); print(r["lr"].to_string()); print(json.dumps(r["decision"], indent=2))
    elif a.phase == "final":
        ex.final_job(a.seed, a.variant)
    elif a.phase == "report":
        r = ex.final_report(tuple(v for v in a.variants.split(",") if v), confirm_test=a.confirm_test)
        print(r["intervals"].to_string())


if __name__ == "__main__":
    main()
