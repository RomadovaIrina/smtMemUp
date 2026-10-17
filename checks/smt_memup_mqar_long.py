#!/usr/bin/env python3
"""MQAR pilot01 with a LONG budget (17.10.2026): is the failure of the 3000-step SMT pilot a budget problem or a method
problem? (The analogue of the Stack convergence check C1.) Context from 16-17.10: a plain causal Transformer solves
pilot01 (100%), our RNN with full BPTT stays at chance.

  python checks/smt_memup_mqar_long.py --variant baseline          # author SMT loss, Tc = Tf = T (as pilot01)
  python checks/smt_memup_mqar_long.py --variant surprise_ctx_k    # + selected future targets by decoder-input surprise

smt_memup_mqar.py is imported, not modified (its hash identifies the pilot01 run); the budget is part of the config,
so these runs get their own run directory runs/mqar03_pilot01long_<key>/. Same data, splits, initial weights (tuning
seed 107), evaluation and checkpoint rule as pilot01; alpha = 1.0 fixed (Stack: every variant chose 1.0). Only
validation / tuning are used. The evaluation at the end reports, per stage, the pilot01 gate metrics (accuracy among
the written values on long answers, original and shuffled query order) and full-vocabulary accuracy.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import pandas as pd  # noqa: E402
import torch  # noqa: E402

import smt_memup_mqar as M  # noqa: E402
import smt_memup_stack as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["baseline", "surprise_ctx_k"])
    ap.add_argument("--smt-steps", type=int, default=20000)
    ap.add_argument("--dmt-steps", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    device = S.setup_runtime(a.threads)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    real_copy = S.copy_readout                     # memory hygiene only (no effect on numerics)

    def copy_and_free(teacher, rnn):
        real_copy(teacher, rnn)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    S.copy_readout = copy_and_free
    cfg = M.candidate("pilot01", smt_steps=a.smt_steps, dmt_steps=a.dmt_steps)
    ex = M.Experiment(cfg, "pilot01long", root=ROOT, device=device, log=log)
    log(f"run directory: {ex.dir}")
    alpha = 0.0 if a.variant == "baseline" else a.alpha
    if a.variant != "baseline":
        ex._need_detectors([a.variant])            # trains (or loads) the context detector in this run directory
    out = ex.dir / "long" / a.variant
    saved = M.fit(cfg, a.variant, alpha, cfg.tuning_seed, out, ex.detectors, ex.validation, device, log)
    shuffled = M.shuffle_query_order(ex.tuning, cfg, S.derive_seed(514, 0))
    frames = []
    for stage in ("smt", "dmt"):
        t, r = S.load_fitted(cfg, saved, device, stage)
        for split, data in (("tuning", ex.tuning), ("tuning_shuffled_order", shuffled)):
            frames.append(M.evaluate(t, r, data, cfg, device, variant=a.variant, stage=stage, split=split))
    ev = pd.concat(frames, ignore_index=True)
    long = ev[ev.delay >= cfg.long_delay]
    rows = []
    for (stage, split, model), g in long.groupby(["stage", "split", "model"]):
        rows.append(dict(stage=stage, split=split, model=model, long_answers=len(g), full_vocab_acc=g.correct.mean(),
                         cand_acc=g.cand_correct.mean(), ce=g.ce.mean(), pred_in_written=g.pred_in_written.mean()))
    table = pd.DataFrame(rows)
    table.to_csv(out / "tuning_summary.csv", index=False)
    (out / "result.json").write_text(json.dumps(dict(variant=a.variant, alpha=alpha, smt_steps=a.smt_steps,
                                                     dmt_steps=a.dmt_steps, smt_best_step=saved["smt_best_step"],
                                                     dmt_best_step=saved["dmt_best_step"],
                                                     chance_written=cfg.chance_structured), indent=2))
    print(table.round(3).to_string())
    log(f"done {a.variant}")


if __name__ == "__main__":
    main()
