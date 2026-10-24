"""SMT x MemUP, version 2 (Stack, no noise): short SMT predictive window + MemUP supervision of far targets.

Agreed with the advisor on 14.10.2026 (proposals/smt_window_memup_draft.md). Importing never trains anything.
smt_memup_stack.py (version 1) is imported and NOT modified; its hash stays part of the version-1 run identity.

Version 1 reweighted targets that the full predictive window (Tf = T) already supervised. Here the authors' SMT loss
is used with a finite predictive window Tf (their default for long sequences); targets beyond the window
(k >= t + Tf) are NOT supervised by the base loss, and the extra term supervises only a selected subset of them:

    L = coef_dec * (L_dec^{Tf} + alpha * L_far) + coef_dyn * L_dyn + coef_unif * L_unif

  * L_dec^{Tf}: author masked mean CE over valid pairs (anchor t, target t + j) with j < Tf, all anchors;
  * L_far = sum_t N_t * mean_{k in S_t} CE_{t,k} / sum_t N_t over the far pool P_t = {k >= t + Tf, y_k defined},
    N_t = |P_t|, |S_t| = min(K, N_t) - the same normalisation as version 1 (E[random] = mean CE over far pairs);
  * predictor of a far target: the SMT decoder with the memory m_t and teacher-forced inputs x_{t+1..k}
    (variant P2 of the draft). In Stack the popped element depends on the operations between t and k, so a
    MemUP-style local predictor q(y_k | m_t, x_{k-r+1..k}) is information-deficient (~52% ceiling, see the draft);
  * gradients: L_far -> decoder and, through m_t, the teacher encoder; the RNN updater learns only from L_dyn;
    detectors are frozen, scores are computed under no_grad and top-K has no gradient.
Data stream, initial weights, anchors, dynamics/uniformity terms, DMT and evaluation are those of version 1, so the
version-1 full-window baseline (same seeds and budget) is the extended-window reference.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import smt_memup_stack as S

MODULE_PATH = Path(__file__).resolve()
DEFAULT_ROOT = MODULE_PATH.parent
V1_RUN = "stack02_4c69a2819ab8"
VARIANTS_AGREED = ("base_window", "uniform_far", "random_far", "surprise_far")
VARIANTS_OPTIONAL = ("ctx_far", "oracle_far")      # diagnostics: decoder-visible surprise; ORACLE (push <= t)
ALL_VARIANTS = VARIANTS_AGREED + VARIANTS_OPTIONAL


@dataclass(frozen=True)
class WindowConfig(S.Config):
    future_window: int = 8      # Tf: decoder targets y_t .. y_{t+Tf-1} are supervised by the base SMT loss
    far_K: int = 2              # far targets selected per anchor
    alpha_far: float = 1.0      # fixed in advance: version 1 selected alpha = 1.0 for every variant; no alpha search here


def code_sha256() -> str:
    return hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()


def run_identity(cfg: WindowConfig, device: torch.device) -> dict:
    ident = S.run_identity(cfg, device)              # config, stack code hash, upstream, versions, device type
    ident.update(task="stack_window_v2", window_code_sha256=code_sha256())
    return ident


# --------------------------------------------------------------------------------------
# Detectors: the version-1 detectors are reused unchanged (same data, same settings)
# --------------------------------------------------------------------------------------

def load_v1_detectors(cfg: WindowConfig, root: Path, device) -> dict:
    """Window detector (r = local_window) and context detector trained in the version-1 run. Checked against
    the version-1 identity so that a changed environment cannot silently swap them."""
    run = Path(root) / "runs" / V1_RUN
    v1_ident = json.loads((run / "identity.json").read_text())
    base = S.Config(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in v1_ident["config"].items()})
    shared = ("seq_len", "n_elements", "max_stack_size", "local_window", "detector_hidden", "detector_embed")
    assert all(getattr(base, k) == getattr(cfg, k) for k in shared), "data/detector settings differ from version 1"
    dets = {}
    for kind, cls in (("window", S.WindowDetector), ("context", S.ContextDetector)):
        saved = torch.load(run / f"detector_{kind}.pt", map_location="cpu", weights_only=True)
        assert saved["identity"]["run"] == v1_ident and saved["step"] == base.detector_steps
        m = cls(base).to(device)
        m.load_state_dict(saved["best_model"])
        dets[kind] = m.eval().requires_grad_(False)
    return dets


# --------------------------------------------------------------------------------------
# Loss
# --------------------------------------------------------------------------------------

def far_pool(w: dict, cfg: WindowConfig, device) -> tuple[torch.Tensor, torch.Tensor]:
    """(in-window pairs, far pool). Output j of anchor t predicts y_{t+j}."""
    valid = w["valid"].to(device)
    j = torch.arange(valid.shape[1], device=device)
    return valid & (j < cfg.future_window), valid & (j >= cfg.future_window)


def smt_window_loss(teacher, rnn, batch: dict, cfg: WindowConfig, variant: str, detectors: dict,
                    generator: torch.Generator, device):
    m_model = S.sources().model
    B, T = batch["x"].shape
    w = S.windows(batch)
    memory = S.encode_chunked(teacher, w["x_ctx"].to(device)).reshape(B, T, -1)        # m_t for every t
    logits = teacher.decode(memory.reshape(B * T, -1), w["x_fut"].to(device))         # causal: output j sees x_fut[:j]
    ce = m_model.calc_cross_entropy_loss(logits, w["y_fut"].to(device).clamp(min=0))
    in_win, far = far_pool(w, cfg, device)
    dec = S.pair_mean(ce, in_win)
    audit = {}
    if variant == "base_window":
        aux = dec * 0
    elif variant == "uniform_far":
        aux = S.weighted_selected_loss(ce, far, far)
    else:
        if variant == "random_far":
            pool, scores = far, torch.zeros_like(ce)
        elif variant == "oracle_far":                    # ORACLE: generator metadata restricts the pool
            pool, scores = far & w["prefix"].to(device), torch.zeros_like(ce)
        elif variant in ("surprise_far", "ctx_far"):
            pool = far
            with torch.no_grad():
                scores = S.selection_scores("surprise_k" if variant == "surprise_far" else "surprise_ctx_k",
                                            w, batch, detectors, device)
        else:
            raise ValueError(variant)
        chosen = S.topk_mask(scores, pool, cfg.far_K, generator)
        aux = S.weighted_selected_loss(ce, pool, chosen)
        c = chosen.cpu()
        if c.any():
            audit = dict(sel_prefix=float(w["prefix"][c].float().mean()),
                         sel_long_prefix=float((w["prefix"] & w["delay"].ge(cfg.long_delay))[c].float().mean()),
                         sel_future=float(w["future"][c].float().mean()))
    prev = torch.cat([torch.zeros_like(memory[:, :1]), memory[:, :-1]], 1)
    pred = rnn.rnn_update(prev.reshape(B * T, -1), batch["x"].to(device).reshape(B * T, 1))
    dyn = m_model.calc_mse_loss(pred, memory.reshape(B * T, -1)).mean()
    unif = m_model.calc_uniformity_loss(memory.reshape(B * T * cfg.mem_tokens, cfg.d_model))
    loss = cfg.coef_dec * (dec + cfg.alpha_far * aux) + cfg.coef_dyn * dyn + cfg.coef_unif * unif
    return loss, dict(loss=loss, decoder=dec, auxiliary=aux, dynamics=dyn, uniformity=unif, **audit)


# --------------------------------------------------------------------------------------
# Far-memory probe: does the teacher memory m_t hold what far memory-dependent pops need?
# --------------------------------------------------------------------------------------

@torch.no_grad()
def far_probe(teacher, batch: dict, cfg: WindowConfig, device, chunk: int = 32) -> pd.DataFrame:
    """For every far pair (t, k >= t + Tf) whose element was pushed at or before t, the frozen SMT decoder predicts
    y_k from m_t and x_{t+1..k}. High accuracy = the intermediate memory m_t retains the element."""
    teacher.eval()
    rows = []
    for lo in range(0, len(batch["x"]), chunk):
        part = S.slice_batch(batch, lo, lo + chunk)
        B, T = part["x"].shape
        w = S.windows(part)
        memory = S.encode_chunked(teacher, w["x_ctx"].to(device))
        logits = teacher.decode(memory, w["x_fut"].to(device)).float()
        _, far = far_pool(w, cfg, device)
        sel = (far.cpu() & w["prefix"])
        r, j = torch.where(sel)
        y = w["y_fut"][r, j]
        lg = logits[r.to(device), j.to(device)].cpu()
        rows.append(pd.DataFrame(dict(sequence=(w["seq"][r] + lo).numpy(), anchor=w["anchor"][r].numpy(),
                                      target_t=w["target_t"][r, j].numpy(), distance=j.numpy(),
                                      delay=w["delay"][r, j].numpy(), correct=(lg.argmax(-1) == y).numpy(),
                                      ce=torch.nn.functional.cross_entropy(lg, y, reduction="none").numpy())))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# Training: version-1 fit() with the window loss (same seeds, data streams, resume, checkpoint rule)
# --------------------------------------------------------------------------------------

def fit(cfg: WindowConfig, variant: str, seed: int, out: Path, detectors: dict, validation: dict, device,
        log=print) -> dict:
    assert variant in ALL_VARIANTS
    out.mkdir(parents=True, exist_ok=True)
    det_hash = {k: S.state_hash(v) for k, v in detectors.items()
                if (k == "window" and variant == "surprise_far") or (k == "context" and variant == "ctx_far")}
    identity = dict(run=run_identity(cfg, device), variant=variant, seed=int(seed), detectors=det_hash,
                    validation=S.split_hash(validation))
    done = out / "completed.pt"
    if done.exists():
        saved = torch.load(done, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, f"{done}: identity mismatch"
        return saved
    teacher, rnn = S.make_models(cfg, seed, device)
    initial_hash = S.state_hash(teacher, rnn)
    result = dict(identity=identity, initial_hash=initial_hash)
    for stage, steps in (("smt", cfg.smt_steps), ("dmt", cfg.dmt_steps)):
        is_smt = stage == "smt"
        teacher.requires_grad_(is_smt)
        if not is_smt:
            S.copy_readout(teacher, rnn)
            frozen = S.state_hash(teacher)
        params = [*teacher.parameters(), *rnn.parameters()] if is_smt else list(rnn.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.smt_lr if is_smt else cfg.dmt_lr, betas=(0.9, 0.95),
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, S.lr_lambda(cfg, steps))
        last, best_path = out / f"{stage}_last.pt", out / f"{stage}_best.pt"
        start, history, best = 0, [], float("inf")
        if last.exists():
            saved = torch.load(last, map_location="cpu", weights_only=True)
            assert saved["identity"] == identity and saved["initial_hash"] == initial_hash
            teacher.load_state_dict(saved["teacher"]); rnn.load_state_dict(saved["rnn"])
            opt.load_state_dict(saved["optimizer"]); sched.load_state_dict(saved["scheduler"])
            start, history, best = saved["step"], saved["history"], saved["best"]
            log(f"resume {out.name} {stage} at {start}/{steps}")
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        tic = time.perf_counter()
        for step in range(start, steps):
            teacher.train(is_smt); rnn.train()
            bs = cfg.batch_size if is_smt else cfg.dmt_batch_size
            batch = S.sample_stack(cfg, bs, S.derive_seed(301 if is_smt else 302, seed, step))
            gen = torch.Generator().manual_seed(S.derive_seed(303, seed, step))
            opt.zero_grad(set_to_none=True)
            with S.isolated_seed(S.derive_seed(304 if is_smt else 305, seed, step)):
                loss, parts = (smt_window_loss(teacher, rnn, batch, cfg, variant, detectors, gen, device)
                               if is_smt else S.dmt_loss(teacher, rnn, batch, cfg, device))
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{variant} {stage} step {step}: {parts}")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
            opt.step(); sched.step()
            row = dict(step=step + 1, lr=opt.param_groups[0]["lr"], grad_norm=float(norm),
                       **{k: float(v.detach()) if torch.is_tensor(v) else float(v) for k, v in parts.items()})
            monitor = (step + 1) % cfg.eval_every == 0 or step + 1 == steps
            if monitor:
                if is_smt:
                    S.copy_readout(teacher, rnn)
                ev = S.evaluate(teacher, rnn, validation, cfg, device)
                score = S.long_ce(ev, cfg, "teacher" if is_smt else "rnn")
                for m in ("teacher", "rnn"):
                    lg = ev[(ev.model == m) & (ev.delay >= cfg.long_delay)]
                    row[f"val_{m}_long_ce"], row[f"val_{m}_long_acc"] = float(lg.ce.mean()), float(lg.correct.mean())
                    row[f"val_{m}_all_acc"] = float(ev[ev.model == m].correct.mean())
                if is_smt:
                    fp = far_probe(teacher, S.slice_batch(validation, 0, 128), cfg, device)
                    row["val_far_memory_acc"] = float(fp.correct.mean())
                row["seconds_per_step"] = (time.perf_counter() - tic) / max(1, (step + 1 - start))
                if device.type == "cuda":
                    row["peak_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
                if score < best:
                    best = score
                    torch.save(dict(identity=identity, step=step + 1, teacher=teacher.state_dict(),
                                    rnn=rnn.state_dict(), score=score), best_path)
                log(f"{out.parent.name}/{out.name} {stage} {step + 1}/{steps} "
                    f"teacher long acc={row['val_teacher_long_acc']:.3f} rnn long acc={row['val_rnn_long_acc']:.3f} "
                    f"select={score:.4f} {row['seconds_per_step']:.3f}s/step")
            history.append(row)
            if monitor:
                torch.save(dict(identity=identity, initial_hash=initial_hash, step=step + 1,
                                teacher=teacher.state_dict(), rnn=rnn.state_dict(), optimizer=opt.state_dict(),
                                scheduler=sched.state_dict(), history=history, best=best), last)
                pd.DataFrame(history).to_csv(out / f"{stage}_history.csv", index=False)
        result[f"{stage}_last_teacher"] = copy.deepcopy(teacher.state_dict())
        result[f"{stage}_last_rnn"] = copy.deepcopy(rnn.state_dict())
        best_saved = torch.load(best_path, map_location="cpu", weights_only=True)
        teacher.load_state_dict(best_saved["teacher"]); rnn.load_state_dict(best_saved["rnn"])
        result[f"{stage}_best_step"] = best_saved["step"]
        if is_smt:
            S.copy_readout(teacher, rnn)
            result["rnn_smt"] = copy.deepcopy(rnn.state_dict())
        else:
            assert S.state_hash(teacher) == frozen, "teacher changed during DMT"
            result["rnn_dmt"] = copy.deepcopy(rnn.state_dict())
    result["teacher"] = teacher.state_dict()
    torch.save(result, done)
    return result


# --------------------------------------------------------------------------------------
# Selection audit on the far pool (before any training)
# --------------------------------------------------------------------------------------

@torch.no_grad()
def far_selection_audit(batch: dict, cfg: WindowConfig, detectors: dict, device, seed: int = 401) -> pd.DataFrame:
    w = S.windows(batch)
    _, far = far_pool(w, cfg, device)
    far = far.cpu()
    g = torch.Generator().manual_seed(seed)
    rows = []
    for rule in ("random_far", "surprise_far", "ctx_far", "oracle_far"):
        pool = far & w["prefix"] if rule == "oracle_far" else far
        if rule == "surprise_far":
            scores = S.selection_scores("surprise_k", w, batch, detectors, device).cpu()
        elif rule == "ctx_far":
            scores = S.selection_scores("surprise_ctx_k", w, batch, detectors, device).cpu()
        else:
            scores = torch.zeros(far.shape)
        chosen = S.topk_mask(scores, pool, cfg.far_K, g)
        d = w["delay"][chosen]
        rows.append(dict(rule=rule, pairs=int(chosen.sum()), anchors_with_pool=float((pool.sum(1) > 0).float().mean()),
                         memory_dependent=float(w["prefix"][chosen].float().mean()),
                         long_memory_dependent=float((w["prefix"] & w["delay"].ge(cfg.long_delay))[chosen].float().mean()),
                         pushed_after_anchor=float(w["future"][chosen].float().mean()),
                         mean_delay=float(d.float().mean()),
                         mean_distance=float((w["target_t"] - w["anchor"][:, None])[chosen].float().mean())))
    n = far.sum(1).float()
    k = n.clamp(max=cfg.far_K)
    has = n > 0
    frac = lambda m: float(((k * (m & far).sum(1) / n.clamp(min=1))[has]).sum() / k[has].sum())
    rows.append(dict(rule="random_far (exact expectation)", pairs=int(k.sum()), anchors_with_pool=float(has.float().mean()),
                     memory_dependent=frac(w["prefix"]), long_memory_dependent=frac(w["prefix"] & w["delay"].ge(cfg.long_delay)),
                     pushed_after_anchor=frac(w["future"]), mean_delay=float("nan"), mean_distance=float("nan")))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------------------

class Experiment:
    def __init__(self, cfg: WindowConfig, root: Path | str = DEFAULT_ROOT, device=None, log=print):
        self.cfg, self.root, self.log = cfg, Path(root), log
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        S.sources(self.root)
        ident = run_identity(cfg, self.device)
        key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = self.root / "runs" / f"stackwin_{key}"
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "identity.json"
        if p.exists():
            assert json.loads(p.read_text()) == ident
        else:
            p.write_text(json.dumps(ident, indent=2))
        # identical validation / tuning splits as version 1 (same seeds)
        self.validation = S.sample_stack(cfg, cfg.validation_examples, S.derive_seed(501, 0))
        self.tuning = S.sample_stack(cfg, cfg.tuning_examples, S.derive_seed(502, 0))
        self.detectors = load_v1_detectors(cfg, self.root, self.device)

    def audit(self) -> pd.DataFrame:
        sel = far_selection_audit(S.slice_batch(self.tuning, 0, 128), self.cfg, self.detectors, self.device)
        sel.to_csv(self.dir / "far_selection_audit.csv", index=False)
        return sel

    def baseline(self) -> dict:
        """base_window on the tuning seed; the same gate thresholds as version 1 (sanity, tuning split)."""
        cfg = self.cfg
        saved = fit(cfg, "base_window", cfg.tuning_seed, self.dir / "tuning" / "base_window", self.detectors,
                    self.validation, self.device, self.log)
        frames = []
        for stage in ("smt", "dmt"):
            t, r = S.load_fitted(cfg, saved, self.device, stage)
            frames.append(S.evaluate(t, r, self.tuning, cfg, self.device, stage=stage))
        ev = pd.concat(frames, ignore_index=True)
        chance = 1 / cfg.n_elements
        lt = ev[(ev.model == "teacher") & (ev.stage == "smt") & (ev.delay >= cfg.long_delay)].correct.mean()
        lr_ = ev[(ev.model == "rnn") & (ev.stage == "dmt") & (ev.delay >= cfg.long_delay)].correct.mean()
        gate = dict(teacher_long_accuracy=float(lt), rnn_dmt_long_accuracy=float(lr_), chance=chance,
                    required_teacher=chance + cfg.gate_teacher_margin, required_rnn=chance + cfg.gate_rnn_margin,
                    smt_best_step=saved["smt_best_step"], dmt_best_step=saved["dmt_best_step"])
        gate["passed"] = bool(lt >= gate["required_teacher"] and lr_ >= gate["required_rnn"])
        (self.dir / "baseline_gate.json").write_text(json.dumps(gate, indent=2))
        return gate

    def final_fits(self, variants=ALL_VARIANTS) -> dict:
        gate = self.dir / "baseline_gate.json"
        assert gate.exists() and json.loads(gate.read_text())["passed"], "run the gate first (phase baseline)"
        fitted = {}
        for seed in self.cfg.seeds:
            for v in variants:
                fitted[seed, v] = fit(self.cfg, v, seed, self.dir / "final" / f"seed{seed}" / v, self.detectors,
                                      self.validation, self.device, self.log)
            assert len({fitted[seed, v]["initial_hash"] for v in variants}) == 1, "initial weights differ"
        return fitted


    def final_report(self, variants=ALL_VARIANTS, confirm_test: bool = False) -> dict:
        """The only place where the test split is generated: the SAME test sequences as version 1. The version-1
        full-window baseline fits (same seeds, budget and initial weights) are evaluated as 'full_window'."""
        assert confirm_test, "pass confirm_test=True once the protocol is frozen; never tune after this"
        cfg = self.cfg
        fitted = self.final_fits(variants)
        test = S.sample_stack(cfg, cfg.test_examples, S.derive_seed(503, 0))
        assert not set(S.fingerprints(test)) & (set(S.fingerprints(self.validation)) | set(S.fingerprints(self.tuning)))
        v1 = self.root / "runs" / V1_RUN / "final"
        refs = {}
        for seed in cfg.seeds:
            saved = torch.load(v1 / f"seed{seed}" / "baseline" / "completed.pt", map_location="cpu", weights_only=True)
            assert saved["initial_hash"] == fitted[seed, variants[0]]["initial_hash"], "full-window reference: other init"
            refs[seed, "full_window"] = saved
        frames, probes = [], []
        for (seed, v), saved in {**fitted, **refs}.items():
            for which in ("best", "last"):
                for stage in ("smt", "dmt"):
                    t, r = S.load_fitted(cfg, saved, self.device, stage, which)
                    frames.append(S.evaluate(t, r, test, cfg, self.device, seed=seed, variant=v, stage=stage,
                                             split="test", checkpoint=which))
            t, _ = S.load_fitted(cfg, saved, self.device, "smt", "best")
            fp = far_probe(t, test, cfg, self.device)
            fp["seed"], fp["variant"] = seed, v
            probes.append(fp.groupby(["seed", "variant"]).agg(far_pairs=("correct", "size"), far_memory_acc=("correct", "mean"),
                                                             far_memory_ce=("ce", "mean")).reset_index())
        events = pd.concat(frames, ignore_index=True)
        comps = [("random_far", "surprise_far")]                                                   # primary
        comps += [("base_window", v) for v in ("uniform_far", "random_far", "surprise_far")]        # far supervision
        comps += [("uniform_far", "surprise_far"), ("base_window", "full_window"), ("full_window", "surprise_far")]
        comps += [(ref, v) for v in ("ctx_far", "oracle_far") for ref in ("random_far", "surprise_far")]
        names = set(variants) | {"full_window"}
        comps = [(a, b) for a, b in comps if a in names and b in names]
        intervals, per_seed = S.paired_deltas(events[events.checkpoint == "best"], cfg, comps)
        summary = S.summarize(events, cfg, keys=("checkpoint", "seed", "variant", "stage", "model"))
        far = pd.concat(probes, ignore_index=True)
        for name, df in (("test_events", events), ("test_summary", summary), ("test_intervals", intervals),
                         ("test_per_seed", per_seed), ("test_far_memory", far)):
            df.to_csv(self.dir / f"{name}.csv", index=False)
        report = dict(variants=list(variants), reference="full_window = version-1 baseline", seeds=list(cfg.seeds),
                      test_hash=S.split_hash(test), primary="RNN after DMT, CE on pops with delay>=long_delay, surprise_far vs random_far")
        (self.dir / "test_report.json").write_text(json.dumps(report, indent=2))
        return dict(events=events, intervals=intervals, per_seed=per_seed, far=far, report=report)


def main(argv=None):
    p = argparse.ArgumentParser(description="SMT x MemUP version 2: short window + far-target supervision (Stack)")
    p.add_argument("--phase", required=True, choices=["audit", "baseline", "final", "report"])
    p.add_argument("--confirm-test", action="store_true")
    p.add_argument("--variants", default=",".join(ALL_VARIANTS))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-memory-fraction", type=float, default=None)
    p.add_argument("--override", default="{}")
    a = p.parse_args(argv)
    device = S.setup_runtime(a.threads, a.gpu_memory_fraction)
    cfg = replace(WindowConfig(), **{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.override).items()})
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    ex = Experiment(cfg, device=device, log=log)
    log(f"run directory: {ex.dir}")
    if a.phase == "audit":
        print(ex.audit().round(3).to_string())
    elif a.phase == "baseline":
        print(json.dumps(ex.baseline(), indent=2))
    elif a.phase == "final":
        ex.final_fits(tuple(v for v in a.variants.split(",") if v))
    elif a.phase == "report":
        r = ex.final_report(tuple(v for v in a.variants.split(",") if v), confirm_test=a.confirm_test)
        print(r["intervals"].to_string())


if __name__ == "__main__":
    main()
