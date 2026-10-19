#!/usr/bin/env python3
"""Cross-check (19.10.2026): is the failure of SMT on MQAR a bug of the MQAR-specific SMT loss?

The same author models, the same MQAR data (short: T = 16, 2 pairs - the whole example fits Tc = Tf = T), the same
optimiser; only the loss function differs:
  --loss stack : smt_memup_stack.smt_loss (all anchors, full decoder; the code path that learns Stack)
  --loss mqar  : smt_memup_mqar.smt_loss (answer-position logits, trimmed windows; the pilot01 code path)
Reports the teacher accuracy among the written values on validation every 250 steps (chance 1/N).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

import smt_memup_mqar as M  # noqa: E402
import smt_memup_stack as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", required=True, choices=["stack", "mqar"])
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    device = S.setup_runtime(4)
    S.sources(ROOT)
    cfg = M.candidate("pilot01", seq_len=a.T, num_kv_pairs=a.pairs, K=1, long_delay=a.T // 4, local_window=a.T // 4)
    val = M.sample_mqar(cfg, 256, S.derive_seed(511, 0))
    teacher, rnn = S.make_models(cfg, 107, device)
    params = [*teacher.parameters(), *rnn.parameters()]
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    for step in range(a.steps):
        teacher.train(); rnn.train()
        batch = M.sample_mqar(cfg, 32, S.derive_seed(301, 107, step))
        gen = torch.Generator().manual_seed(step)
        if a.loss == "stack":
            loss, parts = S.smt_loss(teacher, rnn, batch, cfg, "baseline", 0.0, {}, gen, device)
        else:
            loss, parts = M.smt_loss(teacher, rnn, batch, cfg, "baseline", 0.0, {}, gen, device)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if (step + 1) % 250 == 0:
            S.copy_readout(teacher, rnn)
            ev = M.evaluate(teacher, rnn, val, cfg, device)
            t = ev[ev.model == "teacher"]
            print(time.strftime("%H:%M:%S"), f"{a.loss} step {step + 1} train_dec={float(parts['decoder']):.3f} "
                  f"teacher cand_acc={t.cand_correct.mean():.3f} full_vocab={t.correct.mean():.3f} (chance 1/{a.pairs})", flush=True)


if __name__ == "__main__":
    main()
