"""SMT x MemUP, version 3 (Stack, no noise): the version-1 comparison in a CONVERGED regime.

Why (results of 14.10.2026): versions 1 and 2 compared variants with 3000 SMT steps (cosine LR to 0.1x) and 1500 DMT
steps. The convergence check showed this is far from convergence: with longer training the full-window SMT teacher
reaches 100% on long pops (outputs_convergence/README.md), and DMT keeps improving well beyond 1500 steps. Differences
between variants at 3000 steps mostly reflect WHEN the long-memory skill switches on, hence the seed-to-seed sign flips.

What changes against version 1 (everything else is identical: author SMT loss and coefficients, all anchors with
Tc = Tf = T, model sizes, data generator, validation / tuning splits, detectors, K, alpha = 1.0, selection rules,
evaluation, checkpoint rule):
  * SMT 10 000 steps with the authors' default LR schedule (linear warm-up, then constant 3e-4);
  * DMT 6000 steps (cosine, as in the convergence checks);
  * NEW seeds 37 / 47 / 57 and a NEW test split (stream 603, disjoint from the version-1 test): the decoder-aware
    selection (surprise_ctx_k) found in version 1 is tested here as a confirmatory, pre-registered hypothesis;
  * test events are written gzip-compressed; the H2 retention analysis runs inside the report.
smt_memup_stack.py and smt_memup_window.py are imported, not modified. Importing this file never trains anything.
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
VARIANTS_MAIN = ("baseline", "uniform", "random_k", "surprise_k", "surprise_ctx_k")
VARIANTS_OPTIONAL = ("oracle_prefix_k",)          # diagnostic only, run last if time permits (decided before any result)
ALL_VARIANTS = VARIANTS_MAIN + VARIANTS_OPTIONAL


@dataclass(frozen=True)
class V3Config(S.Config):
    smt_steps: int = 10000
    dmt_steps: int = 6000
    smt_lr_schedule: str = "warmup_constant"   # authors' default (train_smt.py)
    dmt_lr_schedule: str = "cosine"            # version-1 schedule, as in the convergence checks
    alpha: float = 1.0                         # version 1 selected 1.0 for every variant; fixed, no search
    seeds: tuple = (37, 47, 57)
    test_stream: int = 603                     # version 1 used 503


def code_sha256() -> str:
    return hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()


def run_identity(cfg: V3Config, device: torch.device) -> dict:
    ident = S.run_identity(cfg, device)
    ident.update(task="stack_v3_converged", v3_code_sha256=code_sha256())
    return ident


def lr_lambda(cfg: V3Config, steps: int, schedule: str):
    warm = max(1, min(cfg.warmup_steps, steps // 10))
    if schedule == "warmup_constant":
        return lambda s: (s + 1) / warm if s < warm else 1.0
    if schedule == "cosine":
        return S.lr_lambda(cfg, steps)
    raise ValueError(schedule)


def load_v1_detectors(cfg: V3Config, root: Path, device) -> dict:
    """Same detectors as versions 1 and 2 (same data and detector settings), checked against the version-1 identity."""
    import smt_memup_window as W
    return W.load_v1_detectors(cfg, root, device)


def saved_run_dir(cfg: V3Config = None, root: Path | str = DEFAULT_ROOT) -> Path:
    """Directory of the GPU run, computed from the CUDA identity WITHOUT creating anything (safe on CPU; an
    Experiment on CPU would derive a different, empty directory because the device type is part of the identity)."""
    S.sources(root)
    ident = run_identity(cfg or V3Config(), torch.device("cuda"))
    return Path(root) / "runs" / f"stackv3_{hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]}"


def speed_table(run_dir: Path, seeds, variants) -> pd.DataFrame:
    """Learning speed from the validation histories: first evaluation step at a threshold (secondary metric)."""
    rows = []
    for seed in seeds:
        for v in variants:
            d = Path(run_dir) / "final" / f"seed{seed}" / v
            if not (d / "smt_history.csv").exists():
                continue
            smt = pd.read_csv(d / "smt_history.csv")
            dmt = pd.read_csv(d / "dmt_history.csv") if (d / "dmt_history.csv").exists() else pd.DataFrame(columns=["step", "val_rnn_long_acc"])
            rows.append(dict(seed=seed, variant=v, complete=(d / "completed.pt").exists(),
                             smt_steps_done=int(smt.step.max()), dmt_steps_done=int(dmt.step.max()) if len(dmt) else 0,
                             teacher_steps_to_80=steps_to_threshold(smt, "val_teacher_long_acc", .80),
                             teacher_steps_to_90=steps_to_threshold(smt, "val_teacher_long_acc", .90),
                             teacher_steps_to_95=steps_to_threshold(smt, "val_teacher_long_acc", .95),
                             rnn_dmt_steps_to_80=steps_to_threshold(dmt, "val_rnn_long_acc", .80),
                             rnn_dmt_steps_to_90=steps_to_threshold(dmt, "val_rnn_long_acc", .90),
                             teacher_last_val_long_acc=float(smt.val_teacher_long_acc.dropna().iloc[-1]),
                             rnn_dmt_last_val_long_acc=(float(dmt.val_rnn_long_acc.dropna().iloc[-1])
                                                        if len(dmt.val_rnn_long_acc.dropna()) else float("nan"))))
    return pd.DataFrame(rows)


def steps_to_threshold(history: pd.DataFrame, column: str, threshold: float) -> float:
    h = history.dropna(subset=[column])
    hit = h[h[column] >= threshold]
    return float(hit.step.iloc[0]) if len(hit) else float("nan")


# --------------------------------------------------------------------------------------
# Training: version-1 fit() with the schedules above (resume, data streams, checkpoint rule unchanged)
# --------------------------------------------------------------------------------------

def fit(cfg: V3Config, variant: str, seed: int, out: Path, detectors: dict, validation: dict, device, log=print) -> dict:
    assert variant in ALL_VARIANTS
    alpha = 0.0 if variant == "baseline" else cfg.alpha
    out.mkdir(parents=True, exist_ok=True)
    det_hash = {k: S.state_hash(v) for k, v in detectors.items()
                if (k == "window" and variant == "surprise_k") or (k == "context" and variant == "surprise_ctx_k")}
    identity = dict(run=run_identity(cfg, device), variant=variant, alpha=float(alpha), seed=int(seed),
                    detectors=det_hash, validation=S.split_hash(validation))
    done = out / "completed.pt"
    if done.exists():
        saved = torch.load(done, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, f"{done}: identity mismatch"
        return saved
    teacher, rnn = S.make_models(cfg, seed, device)
    initial_hash = S.state_hash(teacher, rnn)
    result = dict(identity=identity, initial_hash=initial_hash)
    for stage, steps, schedule in (("smt", cfg.smt_steps, cfg.smt_lr_schedule), ("dmt", cfg.dmt_steps, cfg.dmt_lr_schedule)):
        is_smt = stage == "smt"
        teacher.requires_grad_(is_smt)
        if not is_smt:
            S.copy_readout(teacher, rnn)
            frozen = S.state_hash(teacher)
        params = [*teacher.parameters(), *rnn.parameters()] if is_smt else list(rnn.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.smt_lr if is_smt else cfg.dmt_lr, betas=(0.9, 0.95),
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, steps, schedule))
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
                loss, parts = (S.smt_loss(teacher, rnn, batch, cfg, variant, alpha, detectors, gen, device)
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
# Orchestration
# --------------------------------------------------------------------------------------

PRIMARY_CONFIRMATORY = [("baseline", "surprise_ctx_k"), ("random_k", "surprise_ctx_k"), ("uniform", "surprise_ctx_k")]
PRIMARY_AGREED = [("random_k", "surprise_k"), ("uniform", "surprise_k")]
CONTEXT = [("baseline", v) for v in ("uniform", "random_k", "surprise_k")] + [("surprise_k", "surprise_ctx_k")]
DIAGNOSTIC = [(r, "oracle_prefix_k") for r in ("baseline", "random_k", "surprise_ctx_k")]


class Experiment:
    def __init__(self, cfg: V3Config, root: Path | str = DEFAULT_ROOT, device=None, log=print):
        self.cfg, self.root, self.log = cfg, Path(root), log
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        S.sources(self.root)
        ident = run_identity(cfg, self.device)
        key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = self.root / "runs" / f"stackv3_{key}"
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "identity.json"
        if p.exists():
            assert json.loads(p.read_text()) == ident
        else:
            p.write_text(json.dumps(ident, indent=2))
        self.validation = S.sample_stack(cfg, cfg.validation_examples, S.derive_seed(501, 0))   # as version 1
        self.tuning = S.sample_stack(cfg, cfg.tuning_examples, S.derive_seed(502, 0))
        self.detectors = load_v1_detectors(cfg, self.root, self.device)

    def fit_one(self, seed: int, variant: str) -> dict:
        assert seed in self.cfg.seeds
        return fit(self.cfg, variant, seed, self.dir / "final" / f"seed{seed}" / variant, self.detectors,
                   self.validation, self.device, self.log)

    def final_fits(self, variants) -> dict:
        fitted = {}
        for seed in self.cfg.seeds:
            for v in variants:
                fitted[seed, v] = self.fit_one(seed, v)
            assert len({fitted[seed, v]["initial_hash"] for v in variants}) == 1, "initial weights differ"
        return fitted

    def speed_table(self, variants) -> pd.DataFrame:
        return speed_table(self.dir, self.cfg.seeds, variants)

    def final_report(self, variants, confirm_test: bool = False) -> dict:
        """The only place where the version-3 test split is generated (stream 603, disjoint from validation, tuning
        and the version-1 test). Events are gzip-compressed; H2 retention is computed on the same test sequences."""
        assert confirm_test, "pass confirm_test=True only with the user's permission; never tune after this"
        import h2_retention as H
        cfg = self.cfg
        fitted = self.final_fits(variants)
        test = S.sample_stack(cfg, cfg.test_examples, S.derive_seed(cfg.test_stream, 0))
        v1_test = S.sample_stack(cfg, cfg.test_examples, S.derive_seed(503, 0))
        seen = set(S.fingerprints(self.validation)) | set(S.fingerprints(self.tuning)) | set(S.fingerprints(v1_test))
        assert not set(S.fingerprints(test)) & seen, "version-3 test overlaps an earlier split"
        frames, h2 = [], []
        for (seed, v), saved in fitted.items():
            for which in ("best", "last"):
                for stage in ("smt", "dmt"):
                    t, r = S.load_fitted(cfg, saved, self.device, stage, which)
                    frames.append(S.evaluate(t, r, test, cfg, self.device, seed=seed, variant=v, stage=stage,
                                             split="test", checkpoint=which))
            t, r = S.load_fitted(cfg, saved, self.device, "dmt", "best")
            pr = H.retention_by_source(t, r, test, cfg, self.device)
            pr["seed"], pr["variant"], pr["checkpoint"] = seed, v, "best"
            h2.append(H.summarize_retention(pr, cfg))
            self.log(f"evaluated seed{seed}/{v}")
        events = pd.concat(frames, ignore_index=True)
        comps = PRIMARY_CONFIRMATORY + PRIMARY_AGREED + CONTEXT + DIAGNOSTIC
        comps = [(a, b) for a, b in comps if a in variants and b in variants]
        intervals, per_seed = S.paired_deltas(events[events.checkpoint == "best"], cfg, comps)
        summary = S.summarize(events, cfg, keys=("checkpoint", "seed", "variant", "stage", "model"))
        events.to_csv(self.dir / "test_events.csv.gz", index=False, compression="gzip")
        for name, df in (("test_summary", summary), ("test_intervals", intervals), ("test_per_seed", per_seed),
                         ("h2_test_summary", pd.concat(h2, ignore_index=True)), ("speed", self.speed_table(variants))):
            df.to_csv(self.dir / f"{name}.csv", index=False)
        report = dict(variants=list(variants), seeds=list(cfg.seeds), test_hash=S.split_hash(test),
                      primary_confirmatory="RNN after DMT, CE on pops with delay>=8: surprise_ctx_k vs baseline, random_k, uniform",
                      primary_agreed="surprise_k vs random_k and uniform")
        (self.dir / "test_report.json").write_text(json.dumps(report, indent=2))
        return dict(events=events, intervals=intervals, per_seed=per_seed, report=report)


def main(argv=None):
    p = argparse.ArgumentParser(description="SMT x MemUP version 3: converged regime (Stack)")
    p.add_argument("--phase", required=True, choices=["fit", "final", "speed", "report"])
    p.add_argument("--seed", type=int)
    p.add_argument("--variant")
    p.add_argument("--variants", default=",".join(VARIANTS_MAIN))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-memory-fraction", type=float, default=None)
    p.add_argument("--confirm-test", action="store_true")
    a = p.parse_args(argv)
    variants = tuple(v for v in a.variants.split(",") if v)
    if a.phase == "speed":                        # CPU-safe: reads histories, creates nothing
        pd.set_option("display.width", 250)
        print(speed_table(saved_run_dir(), V3Config().seeds, variants).to_string()); return
    device = S.setup_runtime(a.threads, a.gpu_memory_fraction)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    ex = Experiment(V3Config(), device=device, log=log)
    log(f"run directory: {ex.dir}")
    if a.phase == "fit":
        ex.fit_one(a.seed, a.variant)
    elif a.phase == "final":
        ex.final_fits(variants)
    elif a.phase == "report":
        r = ex.final_report(variants, confirm_test=a.confirm_test)
        print(r["intervals"].to_string())


if __name__ == "__main__":
    main()
