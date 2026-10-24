"""4. Losses. B sequences, T positions, R = B*T anchors (rows of windows.py), V vocabulary, Z = K*d.

SMT  : L = coef_dec * (L_dec^Tf + alpha_far * L_far) + coef_dyn * L_dyn + coef_unif * L_unif
DMT  : L = dmt_coef_dyn * MSE(m^rnn_t, m_t) + CE(readout(sg[m^rnn_t]), y_t)          (teacher frozen)
BPTT : L = CE(readout(m^rnn_t), y_t)                                                  (no teacher)
LM   : L = CE(logits_t, y_t)                                                          (MQAR Transformer)
Every CE is a masked mean over the positions / pairs that have an answer. sg = stop-gradient (detach).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .config import Config
from .models import decode_full, encode_all, encode_full, rnn_rollout
from .upstream import IGNORE, author
from .windows import windows


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """sum(values * mask) / number of True entries (authors' masked_mean, model.py:55-59, guarded against 0)."""
    return (values * mask).sum() / mask.sum().clamp(min=1)


def topk_mask(scores, pool, K: int, generator) -> torch.Tensor:
    """(R, T) -> (R, T) bool: per row the K highest scores inside `pool` (ties broken by a seeded random order)."""
    s = torch.where(pool, scores.float(), torch.full_like(scores, -torch.inf, dtype=torch.float32))
    perm = torch.rand(s.shape, generator=generator, device="cpu").to(s.device).argsort(1)
    order = torch.sort(s.gather(1, perm), dim=1, descending=True, stable=True).indices
    return torch.zeros_like(pool).scatter(1, perm.gather(1, order)[:, :K], True) & pool


def smt_loss(teacher, rnn, batch: dict, cfg: Config, device, generator=None, detector=None):
    m_author = author("model")
    B, T = batch["x"].shape
    w = windows(batch)                                                                   # each (R, T)
    memory = encode_all(teacher, w["x_ctx"].to(device)).reshape(B, T, -1)                # m_t: (B, T, Z)
    logits = teacher.decode(memory.reshape(B * T, -1), w["x_fut"].to(device))            # (R, T, V)
    ce = m_author.calc_cross_entropy_loss(logits, w["y_fut"].to(device).clamp(min=0))    # (R, T); -1 -> 0, masked below
    valid = w["valid"].to(device)                                                        # (R, T)
    j = torch.arange(T, device=device)                                                   # decoder output index
    near, far = valid & (j < cfg.future_window), valid & (j >= cfg.future_window)
    # L_dec^Tf = sum_{t, j<Tf} valid * CE(y_{t+j} | m_t, x_{t+1..t+j}) / #valid pairs with j < Tf
    dec = masked_mean(ce, near)
    if cfg.far == "ctx_far":
        # far pool P_t = {j >= Tf, valid}; S_t = top far_K by detector CE (no gradient through the choice)
        with torch.no_grad():
            scores = torch.cat([F.cross_entropy(detector(xf.to(device)).flatten(0, 1), yf.clamp(min=0).flatten().to(device),
                                                reduction="none").reshape(yf.shape) * yf.ne(IGNORE).to(device)
                                for xf, yf in zip(w["x_fut"].split(4096), w["y_fut"].split(4096))])   # (R, T)
        chosen = topk_mask(scores, far, cfg.far_K, generator)                            # (R, T)
        # L_far = sum_t N_t * mean_{k in S_t} CE_{t,k} / sum_t N_t,  N_t = |P_t|  (E over random S_t = mean far CE)
        n, k = far.sum(1).to(ce.dtype), chosen.sum(1).to(ce.dtype)                       # (R,), (R,)
        aux = (n * (ce * chosen).sum(1) / k.clamp(min=1)).sum() / n.sum().clamp(min=1)
    else:
        aux = dec * 0
    # L_dyn = mean (f(m_{t-1}, x_t) - m_t)^2 over B*T*Z numbers; m_{-1} = 0 (first transition: ours)
    prev = torch.cat([torch.zeros_like(memory[:, :1]), memory[:, :-1]], 1)                # m_{t-1}: (B, T, Z)
    if cfg.dyn_detach == "both":                       # authors' detach_rnn=True: L_dyn trains only f
        prev = prev.detach()
    pred = rnn.rnn_update(prev.reshape(B * T, -1), batch["x"].to(device).reshape(B * T, 1))   # m^_t: (R, Z)
    target = memory.reshape(B * T, -1)                                                  # m_t: (R, Z)
    # dyn_detach="target" (archived ablation "detached" = target_detached): the gradient still reaches the
    # encoder through `prev`, which is an encoder output with grad.
    dyn = m_author.calc_mse_loss(pred, target.detach() if cfg.dyn_detach in ("target", "both") else target).mean()
    # L_unif (model.py:41-53): log mean_{pairs} exp(-2 ||z_i - z_j||^2) over 512 random normalised memory tokens
    unif = m_author.calc_uniformity_loss(memory.reshape(B * T * cfg.mem_tokens, cfg.d_model))
    loss = cfg.coef_dec * (dec + cfg.alpha_far * aux) + cfg.coef_dyn * dyn + cfg.coef_unif * unif
    return loss, dict(loss=loss, decoder=dec, auxiliary=aux, dynamics=dyn, uniformity=unif)


def teacher_full_access_loss(teacher, batch: dict, cfg: Config, device):
    """Diagnostic: L_dec over all valid pairs (Tf = T), the decoder reads all encoder outputs (models.encode_full).
    Its bottleneck counterpart is smt_loss with coef_dyn = coef_unif = 0 and future_window = T."""
    w = windows(batch)
    h, mask = encode_full(teacher, w["x_ctx"].to(device))                                # (R, T+K, d)
    logits = decode_full(teacher, h, mask, w["x_fut"].to(device))                        # (R, T, V)
    y = w["y_fut"].to(device)
    ce = F.cross_entropy(logits.flatten(0, 1).float(), y.clamp(min=0).flatten(), reduction="none").reshape(y.shape)
    loss = masked_mean(ce, w["valid"].to(device))
    return loss, dict(loss=loss, decoder=loss)


def dmt_loss(teacher, rnn, batch: dict, cfg: Config, device):
    """Teacher frozen. The RNN runs on its OWN states from m_{-1} = 0 with the gradient cut between steps, so the MSE at
    step t trains only the update at t (one-step DMT); the readout CE sees sg[states] and trains only the readout."""
    B, T = batch["x"].shape
    with torch.no_grad():
        target = encode_all(teacher, windows(batch)["x_ctx"].to(device)).reshape(B, T, -1)   # m_t: (B, T, Z)
    x = batch["x"].to(device)
    states = rnn_rollout(rnn, x, detach=True)                                            # m^rnn_t: (B, T, Z)
    dyn = (states - target).square().mean()
    logits = rnn.rnn_readout(states.detach().flatten(0, 1)).reshape(B, T, -1)            # (B, T, V)
    ce = author("model").calc_cross_entropy_loss(logits, batch["y"].to(device).clamp(min=0))
    readout = masked_mean(ce, batch["target_mask"].to(device))
    loss = cfg.dmt_coef_dyn * dyn + readout
    return loss, dict(loss=loss, dynamics=dyn, readout=readout)


def bptt_loss(rnn, batch: dict, cfg: Config, device, truncate: bool):
    """The same RNN from scratch on the answers. truncate=False: full BPTT through all T updates (= authors'
    forward_bptt, model.py:233-253). truncate=True (BPTT-1): same forward values (state NOT reset), but the incoming
    state of every update is detached, so CE at t trains only the update at t and the readout."""
    x = batch["x"].to(device)
    states = rnn_rollout(rnn, x, detach=truncate)                                        # (B, T, Z)
    logits = rnn.rnn_readout(states.flatten(0, 1)).reshape(*x.shape, -1)                 # (B, T, V)
    ce = author("model").calc_cross_entropy_loss(logits, batch["y"].to(device).clamp(min=0))
    loss = masked_mean(ce, batch["target_mask"].to(device))
    return loss, dict(loss=loss, readout=loss)


def lm_loss(model, batch: dict, device):
    logits = model(batch["x"].to(device))                                               # (B, T, V)
    ce = F.cross_entropy(logits.flatten(0, 1).float(), batch["y"].clamp(min=0).flatten().to(device),
                         reduction="none").reshape(batch["y"].shape)
    loss = masked_mean(ce, batch["target_mask"].to(device))
    return loss, dict(loss=loss)
