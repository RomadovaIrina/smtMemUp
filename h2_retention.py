"""H2: does the RNN keep the information that the encoder (teacher) state holds? Analysis only, no training.

Separate from smt_memup_stack.py (that file is not modified: its hash identifies the Stack runs).

The probe in smt_memup_stack.retention_probe starts the RNN from the teacher state m_s, applies L = q - s updates
on the true inputs and reads y_q. It mixes two different abilities:
  * source <= s: the popped element was already in the stack at the start state, so the answer must be CARRIED
    by the RNN through L of its own updates (retention of encoder information);
  * source  > s: the element is pushed during the rollout, so the RNN must WRITE it itself (the same skill as a
    rollout from the zero state).
Here every probe row carries source_pos and the flag source_le_start, and the comparison over horizons is also
reported on a FIXED set of answers (delay > 16, horizons L <= 16): for them source <= s holds at every L, so a
change with L is not a change in which answers are counted.

Readouts on the same state: the frozen teacher head ('shared', the probe of the original code) and the RNN's own
head after DMT ('own'). The teacher state at q (same head) is the reference; the RNN rolled out from the zero
state (the deployed model) is reported too.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import smt_memup_stack as S

HORIZONS = (0, 1, 2, 4, 8, 16, 32)


@torch.no_grad()
def retention_by_source(teacher, rnn, batch: dict, cfg: S.Config, device, horizons=HORIZONS,
                        chunk: int = 128) -> pd.DataFrame:
    """One row per (answer q, horizon L, model, head). model: rnn_from_teacher (start m_{q-L}), teacher (m_q),
    rnn_from_zero (RNN rollout from m_{-1} = 0; horizon column = -1)."""
    teacher.eval(); rnn.eval()
    rows = []
    heads = (("shared", teacher.teacher_readout), ("own", rnn.rnn_readout))
    for lo in range(0, len(batch["x"]), chunk):
        part = S.slice_batch(batch, lo, lo + chunk)
        tm = S.teacher_states(teacher, part, device)               # (B, T, Z)
        rm = S.rnn_rollout(rnn, part["x"].to(device))              # (B, T, Z), from the zero state
        x = part["x"].to(device)
        B, T, Z = tm.shape
        b_all, q_all = torch.where(part["target_mask"])
        y_all, src_all = part["y"][b_all, q_all], part["source_pos"][b_all, q_all]
        delay_all = part["delay"][b_all, q_all]

        def emit(model, L, keep, z):
            b, q, y = b_all[keep], q_all[keep], y_all[keep]
            for head_name, head in heads:
                logits = head(z).float().cpu()
                rows.append(pd.DataFrame(dict(
                    model=model, readout=head_name, horizon=L, sequence=(b + lo).numpy(), t=q.numpy(),
                    source=src_all[keep].numpy(), delay=delay_all[keep].numpy(),
                    source_le_start=(src_all[keep] <= q - L).numpy() if L >= 0 else np.ones(int(keep.sum()), bool),
                    correct=(logits.argmax(-1) == y).numpy(),
                    ce=F.cross_entropy(logits, y, reduction="none").numpy())))

        everyone = torch.ones_like(q_all, dtype=torch.bool)
        emit("teacher", -1, everyone, tm[b_all.to(device), q_all.to(device)])
        emit("rnn_from_zero", -1, everyone, rm[b_all.to(device), q_all.to(device)])
        for L in horizons:
            keep = q_all >= L
            b, q = b_all[keep], q_all[keep]
            state = tm[b.to(device), (q - L).to(device)]
            for step in range(1, L + 1):
                state = rnn.rnn_update(state, x[b.to(device), (q - L + step).to(device)][:, None])
            emit("rnn_from_teacher", L, keep, state)
    return pd.concat(rows, ignore_index=True)


LMAX_FIXED = 16   # fixed set: delay > 16 and L <= 16  ->  source = q - delay < q - L = start at every horizon


def summarize_retention(probe: pd.DataFrame, cfg: S.Config) -> pd.DataFrame:
    """Accuracy/CE of rnn_from_teacher per horizon, with the SAME answers read from the teacher state at q
    (reference) and from the RNN rolled out from zero. Answer sets:
      'delay>=long, by source': long answers with q >= L, split by source<=start / source>start
                                (which answers fall in each part changes with L: context only);
      'fixed: delay>16'       : identical answers at every L <= 16, all with source <= start (pure retention)."""
    keys = [k for k in ("seed", "variant", "checkpoint") if k in probe.columns]
    on = keys + ["readout", "sequence", "t"]
    ref = probe[probe.horizon == -1].pivot_table(index=on, columns="model", values=["correct", "ce"])
    ref.columns = [f"{m}_{'acc' if v == 'correct' else 'ce'}" for v, m in ref.columns]
    roll = probe[probe.model == "rnn_from_teacher"].merge(ref.reset_index(), on=on, how="left")
    roll = roll.rename(columns={"correct": "acc"})
    roll["source"] = np.where(roll.source_le_start, "source<=start", "source>start")
    cols = ["acc", "ce", "teacher_acc", "teacher_ce", "rnn_from_zero_acc", "rnn_from_zero_ce"]
    parts = []
    a = roll[roll.delay >= cfg.long_delay]
    t = a.groupby(keys + ["readout", "horizon", "source"])[cols].mean()
    t["n"] = a.groupby(keys + ["readout", "horizon", "source"]).size()
    parts.append(t.reset_index().assign(set=f"delay>={cfg.long_delay}, by source"))
    f = roll[(roll.delay > LMAX_FIXED) & (roll.horizon <= LMAX_FIXED)]
    assert f.source_le_start.all()
    t = f.groupby(keys + ["readout", "horizon", "source"])[cols].mean()
    t["n"] = f.groupby(keys + ["readout", "horizon", "source"]).size()
    parts.append(t.reset_index().assign(set=f"fixed: delay>{LMAX_FIXED}"))
    out = pd.concat(parts, ignore_index=True)
    out["gap_vs_teacher_acc"] = out.acc - out.teacher_acc
    out["gap_vs_teacher_ce"] = out.ce - out.teacher_ce
    return out
