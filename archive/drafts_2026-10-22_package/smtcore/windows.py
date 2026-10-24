"""2. SMT windows. Every position t of every sequence is an anchor ("all anchors"; the SMT paper's synthetic setting
Tc = Tf = T, every timestep). For anchor t (one row r = b*T + t of every tensor below):

    x_ctx[r] = x[t-T+1 .. t]   left-padded with -1   -> encoder input. Since T = sequence length, m_t sees x_0..x_t.
    x_fut[r] = x[t+1 .. t+T]   right-padded with -1  -> decoder inputs. Decoder output j sees x_fut[r, :j] = x_{t+1..t+j}.
    y_fut[r] = y[t .. t+T-1]   right-padded with -1  -> target of decoder output j is y_{t+j}.
                                                        j = 0 is y_t: the answer AT the anchor = what the RNN reads out.
    valid[r] = y_fut[r] != -1                         -> mask of (anchor, target) pairs that have an answer.

Why t+1 for inputs but t for targets: the memory m_t already contains x_t; the answer y_t belongs to position t (it is
the answer to x_t itself), so output j = 0 predicts y_t from m_t alone, and output j predicts y_{t+j} after reading
the j inputs x_{t+1..t+j} that come after the anchor.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .upstream import IGNORE


def windows(batch: dict) -> dict:
    x, y = batch["x"], batch["y"]                               # (B, T)
    B, T = x.shape
    unfold = lambda z, left, right: F.pad(z, (left, right), value=IGNORE).unfold(1, T, 1)    # -> (B, T, T)
    w = dict(x_ctx=unfold(x, T - 1, 0),                         # [b, t, :] = x[b, t-T+1 .. t]
             x_fut=unfold(x, 0, T)[:, 1:T + 1],                 # [b, t, :] = x[b, t+1 .. t+T]
             y_fut=unfold(y, 0, T - 1),                         # [b, t, :] = y[b, t .. t+T-1]
             source=unfold(batch["source_pos"], 0, T - 1))
    w["valid"] = w["y_fut"].ne(IGNORE)
    # metadata (audit / probes only): the answer was written at or before the anchor, i.e. it must be IN m_t
    w["prefix"] = w["valid"] & (w["source"] <= torch.arange(T).view(1, T, 1))
    return {k: v.reshape(B * T, T) for k, v in w.items()}       # every tensor: (B*T, T), row = b*T + t
