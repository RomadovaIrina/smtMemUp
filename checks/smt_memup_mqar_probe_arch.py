#!/usr/bin/env python3
"""MQAR pilot01: WHY does SMT not learn key->value binding? Short teacher-centred diagnostics (17.10.2026).

Context: a causal Transformer solves pilot01 (100%), but the SMT teacher, the SMT RNN and our RNN with full BPTT all
stay at chance. The common element of the failures is the memory: 8 memory tokens x 64 with RMSNorm. Each run here
changes ONE factor of the pilot01 baseline (author SMT loss, Tc = Tf = T, seed 107) and trains SMT only for a short
budget (DMT is a token stage of 50 steps; the question is whether the TEACHER can bind keys to values):
  --name overfit256   : the training stream is replaced by a FIXED set of 256 sequences (can it fit them at all?)
  --name mem16        : mem_tokens 16 (state 16 x 64)
  --name d128         : d_model 128 (state 8 x 128)
  --name lr1e-3       : smt_lr 1e-3
smt_memup_mqar.py is imported, not modified; every run has its own run directory (the config differs).
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

RUNS = {"overfit256": {}, "mem16": dict(mem_tokens=16), "d128": dict(d_model=128), "lr1e-3": dict(smt_lr=1e-3),
        "kv2": dict(num_kv_pairs=2, K=1), "kv4": dict(num_kv_pairs=4, K=1)}     # pairs-count sweep (17.10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(RUNS))
    ap.add_argument("--smt-steps", type=int, default=6000)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    device = S.setup_runtime(a.threads)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    cfg = M.candidate("pilot01", smt_steps=a.smt_steps, dmt_steps=50, **RUNS[a.name])
    ex = M.Experiment(cfg, f"pilot01arch_{a.name}", root=ROOT, device=device, log=log)
    log(f"run directory: {ex.dir}")
    fixed = None
    if a.name == "overfit256":
        real = M.sample_mqar
        fixed = real(cfg, 256, S.derive_seed(515, 107))
        assert not set(S.fingerprints(fixed)) & (set(S.fingerprints(ex.validation)) | set(S.fingerprints(ex.tuning)))

        def from_fixed(c, n, seed):          # training batches only (validation/tuning are already built)
            idx = torch.randint(0, 256, (n,), generator=torch.Generator().manual_seed(int(seed) % 2**31))
            return {k: v[idx] for k, v in fixed.items()}
        M.sample_mqar = from_fixed
    out = ex.dir / "arch" / a.name
    saved = M.fit(cfg, "baseline", 0.0, cfg.tuning_seed, out, {}, ex.validation, device, log)
    t, r = S.load_fitted(cfg, saved, device, "smt")
    rows = []
    for split, data in (("tuning", ex.tuning), ("fixed_train", fixed)):
        if data is None:
            continue
        ev = M.evaluate(t, r, data, cfg, device, split=split)
        g = ev[(ev.model == "teacher") & (ev.delay >= cfg.long_delay)]
        rows.append(dict(name=a.name, split=split, teacher_long_full_vocab=g.correct.mean(), teacher_long_cand=g.cand_correct.mean(),
                         teacher_all_cand=ev[ev.model == "teacher"].cand_correct.mean(), pred_in_written=g.pred_in_written.mean(),
                         smt_best_step=saved["smt_best_step"]))
    pd.DataFrame(rows).to_csv(out / "teacher_summary.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    log(f"done {a.name}")


if __name__ == "__main__":
    main()
