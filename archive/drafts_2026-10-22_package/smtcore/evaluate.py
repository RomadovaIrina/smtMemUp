"""6. Inference and metrics. One row per answer position (Stack pop / MQAR query). Metric = accuracy of the answer ID
(argmax of the logits == y_t); "long" answers have delay >= long_delay. MQAR also: accuracy among the N values written
in the sequence (cand_correct, chance 1/N).

What actually runs at inference for each model:
    'rnn'          final SMT->DMT model and BPTT models: roll the RNN from m_{-1} = 0 over x_0..x_t, read out m^rnn_t.
                   No teacher, no encoder. This is the recurrent inference the method is for.
    'teacher'      diagnostic of the teacher: m_t = encode(x_0..x_t) (the encoder HAS seen the pop / the query key),
                   decoder output j = 0. All archived teacher numbers are this mode.
    'teacher_pre'  MQAR diagnostic (new, 21.10.2026): memory m_a built AFTER all pairs and BEFORE the query
                   (a = 2N-1 'after_pairs' or a = t-1 'before_query'); decoder reads x_{a+1..t}, which end with the
                   query key; output j = t - a. Not yet run on trained models.
    'teacher_full' as 'teacher' but the decoder reads all encoder outputs.
    'lm'           causal LM: logits[:, t].
"""
from __future__ import annotations

import pandas as pd
import torch
import torch.nn.functional as F

from .config import Config
from .models import decode_full, encode_full, rnn_rollout
from .windows import windows


@torch.no_grad()
def answer_logits(kind: str, model, batch: dict, cfg: Config, device, anchor: str = "query") -> pd.DataFrame:
    rows = []
    for lo in range(0, len(batch["x"]), cfg.eval_batch_size):
        part = {k: v[lo:lo + cfg.eval_batch_size] for k, v in batch.items()}
        x, y = part["x"], part["y"]                                     # (B, T)
        B, T = x.shape
        b, t = torch.where(part["target_mask"])                         # answer positions, (A,) each
        if kind == "rnn":
            states = rnn_rollout(model, x.to(device))                   # (B, T, Z)
            lg = model.rnn_readout(states[b.to(device), t.to(device)])  # (A, V)
        elif kind == "lm":
            lg = model(x.to(device))[b.to(device), t.to(device)]        # (A, V)
        else:
            w = windows(part)
            if kind == "teacher_pre":
                a = torch.full_like(t, 2 * cfg.num_kv_pairs - 1) if anchor == "after_pairs" else t - 1
                assert (a < t).all() and (a >= 2 * cfg.num_kv_pairs - 1).all(), "memory must be built after all pairs"
                rows_, j = b * T + a, t - a                             # anchor row, decoder output index
            else:
                rows_, j = b * T + t, torch.zeros_like(t)
            xc, xf = w["x_ctx"][rows_].to(device), w["x_fut"][rows_].to(device)          # (A, T) each
            if kind == "teacher_full":
                h, mask = encode_full(model, xc)
                full = decode_full(model, h, mask, xf)                  # (A, T, V)
            else:
                full = model.decode(model.encode(xc), xf)               # (A, T, V)
            lg = full[torch.arange(len(j), device=device), j.to(device)]                # (A, V)
            assert torch.equal(w["y_fut"][rows_, j], y[b, t]), "output j of anchor a must be aligned with y_t"
        lg = lg.float().cpu()
        target = y[b, t]
        row = dict(sequence=(b + lo).numpy(), t=t.numpy(), delay=part["delay"][b, t].numpy(), target=target.numpy(),
                   pred=lg.argmax(-1).numpy(), correct=(lg.argmax(-1) == target).numpy(),
                   ce=F.cross_entropy(lg, target, reduction="none").numpy())
        if cfg.task == "mqar":
            written = x[:, 1:2 * cfg.num_kv_pairs:2][b]                 # (A, N) values written in the sequence
            cand_y = (written == target[:, None]).float().argmax(1)
            row["cand_correct"] = (lg.gather(1, written).argmax(1) == cand_y).numpy()
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def long_metrics(ev: pd.DataFrame, cfg: Config, prefix: str) -> dict:
    lg = ev[ev.delay >= cfg.long_delay]
    out = {f"{prefix}_long_acc": float(lg.correct.mean()), f"{prefix}_long_ce": float(lg.ce.mean()),
           f"{prefix}_all_acc": float(ev.correct.mean()), f"{prefix}_long_n": int(len(lg))}
    if "cand_correct" in ev:
        out[f"{prefix}_long_cand_acc"] = float(lg.cand_correct.mean())
        out[f"{prefix}_all_cand_acc"] = float(ev.cand_correct.mean())
    return out
