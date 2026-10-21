"""SMT x MemUP: the minimal reading path (Stack and MQAR) in ONE file.

Read top to bottom: 1 data -> 2 tokens/labels/windows -> 3 models -> 4 forward -> 5 losses -> 6 update -> 7 evaluation.
Nothing here is a second implementation of SMT: the encoder, decoder, recurrent updater and readout are the AUTHORS'
classes Teacher and RNN (upstream/smt/src/model.py, commit cf0d0f0), used unchanged; the data come from the authors'
generators (Stack: upstream/smt/src/my_datasets/synthetic.py; MQAR: upstream/zoology, commit 1ad20d1). What is ours is
the thin glue below. checks/test_minimal_equivalence.py checks on fixed small examples that every function here gives
the same numbers as the archived experiment code (smt_memup_stack.py, smt_memup_window.py, smt_memup_v4.py,
smt_memup_mqar.py, mqar_baselines.py, checks/mqar_bottleneck_test.py), which is NOT modified (its hash is part of the
identity of the archived runs).

Two settings must not be confused:
  * Stack, version 1 (archived): FULL predictive window Tf = T = 64 (the SMT paper's synthetic setting);
  * Stack, versions 2/4 and this file: SHORT window Tf = 8 (the target setting), optionally + far targets (ctx_far).

CLI (CPU is enough for the demo and the traces):
  python smt_minimal.py trace --task stack      # one Stack sequence: positions, tokens, labels, masks, windows
  python smt_minimal.py trace --task mqar       # one MQAR sequence (T = 16, N = 2): the same + both teacher read modes
  python smt_minimal.py demo --task stack       # tiny CPU training SMT -> DMT (~2 min), one pop traced to its loss
  python smt_minimal.py demo --task mqar        # the same for one MQAR query, plus the pre-query read modes
  python smt_minimal.py train --spec <json>     # one run of the rerun plan (GPU), see reruns/plan.json
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

ROOT = Path(__file__).resolve().parent
IGNORE = -1                                          # label of positions without a target (Stack pushes, MQAR filler)
UPSTREAM = {"smt": "cf0d0f04be606ab9d97b1a449a6846450d656ad0", "memup": "d27657137f6e427e30fcd4ae885b20b071b2917d",
            "zoology": "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"}


# ======================================================================================================================
# 0. Author code and seeds
# ======================================================================================================================

def _git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_AUTHOR = {}


def author(name: str):
    """'model' -> upstream/smt/src/model.py (Teacher, RNN, losses); 'stack' -> synthetic.py; 'mqar' -> Zoology."""
    if name not in _AUTHOR:
        repo = "zoology" if name == "mqar" else "smt"
        if _git_head(ROOT / "upstream" / repo) != UPSTREAM[repo]:
            raise RuntimeError(f"upstream/{repo} is not at the pinned commit {UPSTREAM[repo]}")
        if name == "model":                                    # same module name as the archived code uses
            _AUTHOR[name] = sys.modules.get("smt_author_model") or _load("smt_author_model", ROOT / "upstream/smt/src/model.py")
        elif name == "stack":
            _AUTHOR[name] = (sys.modules.get("smt_author_synthetic")
                             or _load("smt_author_synthetic", ROOT / "upstream/smt/src/my_datasets/synthetic.py"))
        else:
            sys.path.insert(0, str(ROOT / "upstream" / "zoology"))
            from zoology.data.multiquery_ar import multiquery_ar
            _AUTHOR[name] = multiquery_ar
    return _AUTHOR[name]


def derive_seed(*parts: int) -> int:
    """One integer seed per (stream, run seed, step): batches and random draws never depend on what ran before."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


@contextmanager
def seeded(seed: int):
    """Seed numpy / python / torch inside the block, restore the global RNG state afterwards."""
    states = np.random.get_state(), random.getstate(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(int(seed) % 2**32); random.seed(int(seed)); torch.manual_seed(int(seed))
        yield
    finally:
        np.random.set_state(states[0]); random.setstate(states[1]); torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


# ======================================================================================================================
# Config (one dataclass for both tasks; defaults = Stack, version 4 / ablation reference "full")
# ======================================================================================================================

@dataclass(frozen=True)
class Config:
    task: str = "stack"              # stack | mqar
    seq_len: int = 64                # T
    # Stack (author StackDataset)
    n_elements: int = 4              # push tokens 1..4, pop token 0 -> vocabulary 5
    max_stack_size: int = 8
    # MQAR (Zoology multiquery_ar, random_non_queries=False: no noise)
    vocab: int = 128                 # keys 1..63, values 64..127, filler 0
    num_kv_pairs: int = 2            # N
    power_a: float = 0.01
    long_delay: int = 8              # "long" answers: source written >= long_delay positions before the answer
    # model sizes (author classes)
    d_model: int = 64
    mem_tokens: int = 8              # K memory tokens -> state m_t has K x d_model = 512 numbers
    encoder_depth: int = 3
    decoder_depth: int = 2
    rnn_depth: int = 2
    n_heads: int = 4
    # SMT loss
    future_window: int = 8           # Tf: the decoder predicts y_t .. y_{t+Tf-1} from m_t (Stack v1: Tf = T = 64)
    far: str = "none"                # none | ctx_far (Stack only: + top-far_K far targets k >= t + Tf, CE weight alpha_far)
    far_K: int = 2
    alpha_far: float = 1.0
    coef_dec: float = 1.0
    coef_dyn: float = 0.1            # L_dyn; 0 = ablation no_dyn
    coef_unif: float = 0.001
    dyn_detach: str = "none"         # none (joint, as all archived runs) | target (= archived ablation "detached")
                                     # | both (= authors' detach_rnn=True: no L_dyn gradient reaches the encoder)
    # optimisation
    batch_size: int = 32
    smt_steps: int = 11000
    smt_lr: float = 3e-4
    dmt_steps: int = 5000
    dmt_lr: float = 1e-4
    dmt_coef_dyn: float = 10.0
    bptt_steps: int = 20000
    bptt_lr: float = 3e-4
    lm_steps: int = 3000             # MQAR causal Transformer
    lm_lr: float = 1e-3
    warmup_steps: int = 200
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    # evaluation
    eval_every: int = 500
    eval_batch_size: int = 128
    validation_examples: int = 512   # checkpoint selection (stream 501 Stack / 511 MQAR)
    eval_examples: int = 1024        # independent evaluation of reruns (stream 901), never used for selection

    @property
    def vocab_size(self) -> int:
        return self.n_elements + 1 if self.task == "stack" else self.vocab

    @property
    def mem_size(self) -> int:
        return self.d_model * self.mem_tokens


# ======================================================================================================================
# 1. Data: author generators
# ======================================================================================================================

def sample_stack(cfg: Config, batch_size: int, seed: int) -> dict:
    """x[b, t] in 1..4 = push that element, 0 = pop.  y[b, t] = popped element at a pop, -1 at a push.
    The answer to a pop sits AT the pop position: after reading x_t = 0 the model must output y_t."""
    ds = author("stack").StackDataset(n_elements=cfg.n_elements, max_stack_size=cfg.max_stack_size)
    with seeded(seed):
        raw = ds.sample_batch(batch_size=batch_size, ctx_len=cfg.seq_len)
    x, y = raw["x"].long(), raw["y"].long()
    source = torch.full_like(x, IGNORE)                 # where the popped element was pushed (replay, metadata only)
    for b in range(len(x)):
        stack = []
        for t, token in enumerate(x[b].tolist()):
            if token > 0:
                stack.append(t)
            else:
                source[b, t] = stack.pop()
    return _annotate(x, y, source)


def sample_mqar(cfg: Config, batch_size: int, seed: int) -> dict:
    """Zoology layout (T = 16, N = 2):  k1 v1 k2 v2 | 0 ... 0 k? 0 ... k? 0 ...
    keys in 1..vocab/2-1, values in vocab/2..vocab-1, both unique within a sequence; each key is queried exactly once
    at an even position 2N + 2*gap; the label (the value) sits AT the query position; inputs never contain a value
    after the pairs, so nothing after the query reveals the answer."""
    with seeded(seed):
        seg = author("mqar")(vocab_size=cfg.vocab, num_examples=batch_size, input_seq_len=cfg.seq_len,
                             seed=int(seed) % 2**32, power_a=cfg.power_a, num_kv_pairs=cfg.num_kv_pairs, num_passes=1,
                             random_non_queries=False)
    x, y = seg.inputs.long().clone(), seg.labels.long().clone()
    y[y == -100] = IGNORE
    N = cfg.num_kv_pairs
    keys = x[:, 0:2 * N:2]
    slot = (x[:, :, None] == keys[:, None, :]).float().argmax(-1)       # which written pair a query asks for
    source = torch.where(y.ne(IGNORE), 2 * slot + 1, torch.full_like(x, IGNORE))
    return _annotate(x, y, source)


def _annotate(x, y, source) -> dict:
    mask = y.ne(IGNORE)                                  # loss / metric mask: positions with an answer
    pos = torch.arange(x.shape[1]).expand_as(x)
    return dict(x=x, y=y, target_mask=mask, source_pos=source,
                delay=torch.where(mask, pos - source, torch.full_like(x, IGNORE)))


def sample(cfg: Config, batch_size: int, seed: int) -> dict:
    return (sample_stack if cfg.task == "stack" else sample_mqar)(cfg, batch_size, seed)


# Data streams (identical to the archived code): train batch of step s = stream 301 (SMT, BPTT) / 302 (DMT);
# validation = 501 (Stack) / 511 (MQAR); independent evaluation of the reruns = 901 (new, never used before).
def validation_set(cfg: Config) -> dict:
    return sample(cfg, cfg.validation_examples, derive_seed(501 if cfg.task == "stack" else 511, 0))


def evaluation_set(cfg: Config) -> dict:
    return sample(cfg, cfg.eval_examples, derive_seed(901, 0))


# ======================================================================================================================
# 2. Tokens -> SMT windows (every timestep t is an anchor)
# ======================================================================================================================

def windows(batch: dict) -> dict:
    """For each sequence b and anchor t (rows flattened as b * T + t):
        x_ctx[t] = x[t-T+1 .. t]  -> encoder input, left padding -1 (masked)          memory m_t sees x_0..x_t
        x_fut[t] = x[t+1 .. t+T]  -> decoder inputs; output j sees x_fut[:j] = x_{t+1..t+j} (causal)
        y_fut[t] = y[t .. t+T-1]  -> output j is trained to predict y_{t+j};  j = 0 is the RNN readout target
        valid    = y_fut != -1    -> loss mask (pairs (anchor, target) that have an answer)"""
    x, y = batch["x"], batch["y"]
    B, T = x.shape
    unfold = lambda z, left, right: F.pad(z, (left, right), value=IGNORE).unfold(1, T, 1)     # (B, T, T)
    w = dict(x_ctx=unfold(x, T - 1, 0), x_fut=unfold(x, 0, T)[:, 1:T + 1], y_fut=unfold(y, 0, T - 1),
             source=unfold(batch["source_pos"], 0, T - 1))
    w["valid"] = w["y_fut"].ne(IGNORE)
    w["prefix"] = w["valid"] & (w["source"] <= torch.arange(T).view(1, T, 1))   # element written at or before t
    return {k: v.reshape(B * T, T) for k, v in w.items()}


# ======================================================================================================================
# 3. Models: the authors' Teacher (encoder + decoder) and RNN (updater + readout), unchanged
# ======================================================================================================================

def make_models(cfg: Config, seed: int, device):
    """Teacher = embedding + bidirectional Transformer encoder over [x_ctx, K memory registers] -> m_t (K x d), RMS-norm;
               causal Transformer decoder over [m_t, embed(x_fut)] + linear head.
    RNN     = updater f: Transformer encoder over [m_{t-1} (K tokens), embed(x_t)] -> first K outputs = m_t, RMS-norm
              (NOT a GRU); readout = its own causal decoder + head, initialised as a COPY of the teacher's.
    Same seed -> same initial weights (teacher first, then RNN), as in all archived runs."""
    m = author("model")
    common = dict(vocab_size=cfg.vocab_size, mem_size=cfg.mem_size, mem_tokens=cfg.mem_tokens, d_model=cfg.d_model,
                  decoder_depth=cfg.decoder_depth, n_heads=cfg.n_heads)
    with seeded(seed):
        teacher = m.Teacher(m.TeacherConfig(**common, encoder_depth=cfg.encoder_depth)).to(device)
        rnn = m.RNN(m.RNNConfig(**common, rnn_depth=cfg.rnn_depth)).to(device)
    copy_readout(teacher, rnn)
    return teacher, rnn


def copy_readout(teacher, rnn) -> None:
    """The RNN readout (decoder + head) is a copy of the teacher's; repeated before DMT and at every SMT evaluation."""
    rnn.decoder.load_state_dict(teacher.decoder.state_dict())
    rnn.to_logits.load_state_dict(teacher.to_logits.state_dict())


class ContextDetector(torch.nn.Module):
    """ctx_far only (a SEPARATE small GRU, not the SMT updater): q(y_{t+j} | x_{t+1..t+j}) WITHOUT memory. High CE =
    the far target cannot be inferred from the decoder's own inputs -> it needs m_t. Trained once in version 1 and
    frozen (runs/stack02_4c69a2819ab8/detector_context.pt): embed 6 -> 16, GRU 16 -> 64, linear 64 -> 5."""

    def __init__(self, vocab_size: int, embed: int = 16, hidden: int = 64):
        super().__init__()
        self.bos = vocab_size + 1
        self.embed = torch.nn.Embedding(vocab_size + 2, embed, padding_idx=0)
        self.gru = torch.nn.GRU(embed, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x_fut):                            # (N, T) -> logits (N, T, V); output j has seen x_fut[:j]
        inp = torch.cat([torch.full_like(x_fut[:, :1], self.bos - 1), x_fut[:, :-1]], 1)
        h, _ = self.gru(self.embed(inp + 1))
        return self.head(h)


def load_context_detector(cfg: Config, device) -> ContextDetector:
    saved = torch.load(ROOT / "runs/stack02_4c69a2819ab8/detector_context.pt", map_location="cpu", weights_only=True)
    det = ContextDetector(cfg.vocab_size)
    det.load_state_dict(saved["best_model"])
    return det.to(device).eval().requires_grad_(False)


class CausalTransformer(torch.nn.Module):
    """MQAR reference: an ordinary causal LM (x-transformers TransformerWrapper + Decoder, d 64, depth 2, 4 heads,
    learned absolute positions). logits[:, t] predicts y[:, t] (the value at the query key x_t)."""

    def __init__(self, cfg: Config, d_model: int = 64, depth: int = 2, heads: int = 4):
        super().__init__()
        from x_transformers import Decoder, TransformerWrapper
        self.net = TransformerWrapper(num_tokens=cfg.vocab_size, max_seq_len=cfg.seq_len,
                                      attn_layers=Decoder(dim=d_model, depth=depth, heads=heads))

    def forward(self, x):
        return self.net(x)


# ======================================================================================================================
# 4. Forward passes
# ======================================================================================================================

def encode_all(teacher, x_ctx: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """Teacher.encode for every window: (B*T, T) tokens -> (B*T, K*d) memories m_t."""
    return torch.cat([teacher.encode(p) for p in x_ctx.split(chunk)])


def rnn_rollout(rnn, x: torch.Tensor, init=None, detach: bool = False) -> torch.Tensor:
    """m_t = f(m_{t-1}, x_t), m_{-1} = 0. detach=True cuts the gradient between steps (DMT, BPTT-1). (B,T) -> (B,T,K*d)."""
    state = torch.zeros(x.shape[0], rnn.cfg.mem_size, device=x.device) if init is None else init
    out = []
    for t in range(x.shape[1]):
        state = rnn.rnn_update(state.detach() if detach else state, x[:, t:t + 1])
        out.append(state)
    return torch.stack(out, 1)


def encode_full(teacher, x_ctx):
    """Diagnostic 'full access' (archived checks/mqar_bottleneck_test.py): the authors' encoder, but ALL its outputs
    (T context positions + K registers) are returned instead of the last K. Changes TWO things at once: how many
    positions the decoder can read (T + K instead of K) and how it reads them (attention to per-position outputs)."""
    B = x_ctx.shape[0]
    mask = torch.cat([x_ctx.ne(IGNORE), torch.ones(B, teacher.cfg.mem_tokens, dtype=torch.bool, device=x_ctx.device)], 1)
    regs = repeat(teacher.encoder_mem_register(torch.arange(teacher.cfg.mem_tokens, device=x_ctx.device)), "K D -> B K D", B=B)
    return teacher.encoder(torch.cat([teacher.embed(x_ctx.clamp(min=0)), regs], 1), mask=mask), mask


def decode_full(teacher, h, mask, x_fut):
    """Causal decoder over [all encoder outputs, embed(x_fut)]; output j = position n-1+j predicts y_{t+j}."""
    n = h.shape[1]
    inp = torch.cat([h, teacher.embed(x_fut.clamp(min=0))], 1)
    m = torch.cat([mask, torch.ones(x_fut.shape, dtype=torch.bool, device=x_fut.device)], 1)
    return teacher.to_logits(teacher.decoder(inp, mask=m))[:, n - 1:-1]


# ======================================================================================================================
# 5. Losses
# ======================================================================================================================

def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp(min=1)


def topk_mask(scores, pool, K: int, generator) -> torch.Tensor:
    """Per row: the K highest scores inside the pool (exact ties broken by a seeded random order)."""
    s = torch.where(pool, scores.float(), torch.full_like(scores, -torch.inf, dtype=torch.float32))
    perm = torch.rand(s.shape, generator=generator, device="cpu").to(s.device).argsort(1)
    order = torch.sort(s.gather(1, perm), dim=1, descending=True, stable=True).indices
    return torch.zeros_like(pool).scatter(1, perm.gather(1, order)[:, :K], True) & pool


def smt_loss(teacher, rnn, batch: dict, cfg: Config, device, generator=None, detector=None):
    """L = coef_dec * (L_dec^{Tf} + alpha_far * L_far) + coef_dyn * L_dyn + coef_unif * L_unif     (all anchors t)
    L_dec^{Tf}: masked mean CE of decoder output j < Tf  (near targets y_t .. y_{t+Tf-1});
    L_far     : ctx_far only - CE of the far_K far targets (j >= Tf) with the highest detector CE, weighted by |pool|;
    L_dyn     : MSE(f(m_{t-1}, x_t), m_t): the updater imitates one encoder step (m_{-1} = 0);
    L_unif    : authors' uniformity of the memory tokens (log mean exp(-2 * squared distance))."""
    m_author = author("model")
    B, T = batch["x"].shape
    w = windows(batch)
    memory = encode_all(teacher, w["x_ctx"].to(device)).reshape(B, T, -1)                # m_t, (B, T, K*d)
    logits = teacher.decode(memory.reshape(B * T, -1), w["x_fut"].to(device))            # (B*T, T, V)
    ce = m_author.calc_cross_entropy_loss(logits, w["y_fut"].to(device).clamp(min=0))    # (B*T, T)
    valid = w["valid"].to(device)
    j = torch.arange(T, device=device)
    near, far = valid & (j < cfg.future_window), valid & (j >= cfg.future_window)
    dec = masked_mean(ce, near)
    if cfg.far == "ctx_far":
        with torch.no_grad():                            # selection has no gradient; the detector is frozen
            scores = torch.cat([F.cross_entropy(detector(xf.to(device)).flatten(0, 1), yf.clamp(min=0).flatten().to(device),
                                                reduction="none").reshape(yf.shape) * yf.ne(IGNORE).to(device)
                                for xf, yf in zip(w["x_fut"].split(4096), w["y_fut"].split(4096))])
        chosen = topk_mask(scores, far, cfg.far_K, generator)
        n, k = far.sum(1).to(ce.dtype), chosen.sum(1).to(ce.dtype)
        aux = (n * (ce * chosen).sum(1) / k.clamp(min=1)).sum() / n.sum().clamp(min=1)
    else:
        aux = dec * 0
    prev = torch.cat([torch.zeros_like(memory[:, :1]), memory[:, :-1]], 1)                # m_{t-1}, m_{-1} = 0
    if cfg.dyn_detach == "both":
        prev = prev.detach()
    pred = rnn.rnn_update(prev.reshape(B * T, -1), batch["x"].to(device).reshape(B * T, 1))
    target = memory.reshape(B * T, -1)
    dyn = m_author.calc_mse_loss(pred, target.detach() if cfg.dyn_detach in ("target", "both") else target).mean()
    unif = m_author.calc_uniformity_loss(memory.reshape(B * T * cfg.mem_tokens, cfg.d_model))
    loss = cfg.coef_dec * (dec + cfg.alpha_far * aux) + cfg.coef_dyn * dyn + cfg.coef_unif * unif
    return loss, dict(loss=loss, decoder=dec, auxiliary=aux, dynamics=dyn, uniformity=unif)


def teacher_full_access_loss(teacher, batch: dict, cfg: Config, device):
    """Diagnostic (archived bottleneck test, mode 'full'): only L_dec over all valid pairs, decoder reads all encoder
    outputs. The matching bottleneck run uses smt_loss with coef_dyn = coef_unif = 0 and Tf = T."""
    w = windows(batch)
    h, mask = encode_full(teacher, w["x_ctx"].to(device))
    logits = decode_full(teacher, h, mask, w["x_fut"].to(device))
    y = w["y_fut"].to(device)
    ce = F.cross_entropy(logits.flatten(0, 1).float(), y.clamp(min=0).flatten(), reduction="none").reshape(y.shape)
    loss = masked_mean(ce, w["valid"].to(device))
    return loss, dict(loss=loss, decoder=loss)


def dmt_loss(teacher, rnn, batch: dict, cfg: Config, device):
    """DMT (teacher frozen): the RNN runs on its OWN states from m_{-1} = 0, with the gradient cut between steps;
    L = dmt_coef_dyn * MSE(m^rnn_t, m^teacher_t) + CE(readout(m^rnn_t).detach-input, y_t) on answer positions.
    The readout CE trains only the readout (its input is detached); the updater learns only from the MSE."""
    B, T = batch["x"].shape
    with torch.no_grad():
        target = encode_all(teacher, windows(batch)["x_ctx"].to(device)).reshape(B, T, -1)
    x = batch["x"].to(device)
    states = rnn_rollout(rnn, x, detach=True)
    dyn = (states - target).square().mean()
    logits = rnn.rnn_readout(states.detach().flatten(0, 1)).reshape(B, T, -1)
    ce = author("model").calc_cross_entropy_loss(logits, batch["y"].to(device).clamp(min=0))
    readout = masked_mean(ce, batch["target_mask"].to(device))
    loss = cfg.dmt_coef_dyn * dyn + readout
    return loss, dict(loss=loss, dynamics=dyn, readout=readout)


def bptt_loss(rnn, batch: dict, cfg: Config, device, truncate: bool):
    """Direct task CE of the SAME RNN from scratch, no teacher: masked mean CE of readout(m_t) vs y_t on answers.
    truncate=False: full BPTT through all T updates; truncate=True (BPTT-1): identical forward values, but the incoming
    state of each update is detached, so the loss at t trains only the update at t (and the readout)."""
    x = batch["x"].to(device)
    states = rnn_rollout(rnn, x, detach=truncate)
    logits = rnn.rnn_readout(states.flatten(0, 1)).reshape(*x.shape, -1)
    ce = author("model").calc_cross_entropy_loss(logits, batch["y"].to(device).clamp(min=0))
    loss = masked_mean(ce, batch["target_mask"].to(device))
    return loss, dict(loss=loss, readout=loss)


def lm_loss(model, batch: dict, device):
    """MQAR Transformer (and RNN wrapped as an LM): masked mean CE of logits[:, t] vs y_t on the query positions."""
    logits = model(batch["x"].to(device))
    ce = F.cross_entropy(logits.flatten(0, 1).float(), batch["y"].clamp(min=0).flatten().to(device),
                         reduction="none").reshape(batch["y"].shape)
    loss = masked_mean(ce, batch["target_mask"].to(device))
    return loss, dict(loss=loss)


# ======================================================================================================================
# 6. Weight update: one optimizer step per batch (AdamW, warm-up then constant LR, clip 1.0)
# ======================================================================================================================

def make_optimizer(params, lr: float, cfg: Config, steps: int):
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    warm = max(1, min(cfg.warmup_steps, steps // 10))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 1.0)


def update(loss, params, opt, sched, cfg: Config) -> float:
    """zero_grad was called before the forward; backward -> clip -> AdamW.step -> scheduler.step."""
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
    opt.step(); sched.step()
    return float(norm)


# ======================================================================================================================
# 7. Evaluation (the metric is accuracy of the answer token; "long" = delay >= long_delay)
# ======================================================================================================================

@torch.no_grad()
def answer_logits(kind: str, model, batch: dict, cfg: Config, device, anchor: str = "query") -> pd.DataFrame:
    """One row per answer position t (a Stack pop / an MQAR query) with the logits' argmax and CE.
    kind = 'teacher'    : teacher_readout(m_t) = decoder output j = 0 from the memory of x_0..x_t (the encoder has seen
                          the pop / the query key itself). model = Teacher.
           'teacher_pre': MQAR only. Memory m_a is built BEFORE the query, at anchor a = 2N-1 (right after the last pair;
                          anchor='after_pairs') or a = t-1 (anchor='before_query'); the decoder reads m_a and the inputs
                          x_{a+1..t} (they include the query key) and output j = t - a predicts y_t. model = Teacher.
           'teacher_full': as 'teacher' but the decoder reads all encoder outputs (diagnostic). model = Teacher.
           'rnn'        : readout(m^rnn_t), the RNN rolled out from zero over x_0..x_t. model = RNN.
           'lm'         : logits[:, t] of an ordinary causal model. model = CausalTransformer / RNN-as-LM."""
    rows = []
    for lo in range(0, len(batch["x"]), cfg.eval_batch_size):
        part = {k: v[lo:lo + cfg.eval_batch_size] for k, v in batch.items()}
        x, y = part["x"], part["y"]
        B, T = x.shape
        b, t = torch.where(part["target_mask"])
        if kind == "rnn":
            states = rnn_rollout(model, x.to(device))
            lg = model.rnn_readout(states[b.to(device), t.to(device)])
        elif kind == "lm":
            lg = model(x.to(device))[b.to(device), t.to(device)]
        else:
            w = windows(part)
            if kind == "teacher_pre":
                a = torch.full_like(t, 2 * cfg.num_kv_pairs - 1) if anchor == "after_pairs" else t - 1
                assert (a < t).all() and (a >= 2 * cfg.num_kv_pairs - 1).all(), "memory must be built after all pairs"
                rows_, j = b * T + a, t - a
            else:
                rows_, j = b * T + t, torch.zeros_like(t)
            xc, xf = w["x_ctx"][rows_].to(device), w["x_fut"][rows_].to(device)
            if kind == "teacher_full":
                h, mask = encode_full(model, xc)
                full = decode_full(model, h, mask, xf)
            else:
                full = model.decode(model.encode(xc), xf)
            lg = full[torch.arange(len(j), device=device), j.to(device)]
            assert torch.equal(w["y_fut"][rows_, j], y[b, t]), "output j of anchor a must be aligned with y_t"
        lg = lg.float().cpu()
        target = y[b, t]
        row = dict(sequence=(b + lo).numpy(), t=t.numpy(), delay=part["delay"][b, t].numpy(), target=target.numpy(),
                   pred=lg.argmax(-1).numpy(), correct=(lg.argmax(-1) == target).numpy(),
                   ce=F.cross_entropy(lg, target, reduction="none").numpy())
        if cfg.task == "mqar":                           # accuracy among the N values written in this sequence
            written = x[:, 1:2 * cfg.num_kv_pairs:2][b]
            cand_y = (written == target[:, None]).float().argmax(1)
            row["cand_correct"] = (lg.gather(1, written).argmax(1) == cand_y).numpy()
        rows.append(pd.DataFrame(row))
    return pd.concat(rows, ignore_index=True)


def long_metrics(ev: pd.DataFrame, cfg: Config, prefix: str) -> dict:
    lg = ev[ev.delay >= cfg.long_delay]
    out = {f"{prefix}_long_acc": float(lg.correct.mean()), f"{prefix}_long_ce": float(lg.ce.mean()),
           f"{prefix}_all_acc": float(ev.correct.mean())}
    if "cand_correct" in ev:
        out[f"{prefix}_long_cand_acc"] = float(lg.cand_correct.mean())
    return out


# ======================================================================================================================
# 8. Runs (new reruns only; archived runs stay in runs/ with their own code). Resumable, best checkpoint by validation.
# ======================================================================================================================

def identity(cfg: Config, spec: dict, device) -> dict:
    return dict(config=asdict(cfg), spec=spec, code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                upstream=UPSTREAM, torch=torch.__version__, x_transformers=importlib.metadata.version("x-transformers"),
                python=platform.python_version(), device=device.type)


def run_stage(name: str, steps: int, lr: float, params, step_fn, monitor_fn, models: dict, out: Path, cfg: Config,
              ident: dict, log) -> dict:
    """Generic loop: step_fn(step) -> (loss, parts); every eval_every steps monitor_fn() -> (selection score, metrics);
    the checkpoint with the lowest validation score is kept (score = CE on long answers of the validation set)."""
    opt, sched = make_optimizer(params, lr, cfg, steps)
    last, best_path = out / f"{name}_last.pt", out / f"{name}_best.pt"
    state = dict(step=0, history=[], best=float("inf"), seconds=0.0)
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        assert saved["identity"] == ident, f"{last}: identity mismatch"
        for k, m in models.items():
            m.load_state_dict(saved["models"][k])
        opt.load_state_dict(saved["optimizer"]); sched.load_state_dict(saved["scheduler"])
        state = saved["state"]
        log(f"resume {out.name}/{name} at {state['step']}/{steps}")
    tic = time.perf_counter()
    for step in range(state["step"], steps):
        opt.zero_grad(set_to_none=True)
        loss, parts = step_fn(step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{name} step {step}: {parts}")
        norm = update(loss, params, opt, sched, cfg)
        row = dict(step=step + 1, lr=opt.param_groups[0]["lr"], grad_norm=norm,
                   **{k: float(v.detach()) for k, v in parts.items()})
        if (step + 1) % cfg.eval_every == 0 or step + 1 == steps:
            state["seconds"] += time.perf_counter() - tic
            score, metrics = monitor_fn()
            row.update(metrics, train_seconds=state["seconds"])
            if score < state["best"]:
                state["best"] = score
                torch.save(dict(step=step + 1, models={k: m.state_dict() for k, m in models.items()}), best_path)
            log(f"{out.name}/{name} {step + 1}/{steps} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if "acc" in k))
            state["history"].append(row); state["step"] = step + 1
            torch.save(dict(identity=ident, models={k: m.state_dict() for k, m in models.items()},
                            optimizer=opt.state_dict(), scheduler=sched.state_dict(), state=state), last)
            pd.DataFrame(state["history"]).to_csv(out / f"{name}_history.csv", index=False)
            tic = time.perf_counter()
        else:
            state["history"].append(row)
    best = torch.load(best_path, map_location="cpu", weights_only=True)
    for k, m in models.items():
        m.load_state_dict(best["models"][k])
    return dict(best_step=best["step"], seconds=state["seconds"])


def run(spec: dict, out_root: Path, device, log=print) -> dict:
    """spec = {"name", "family": smt | bptt_full | bptt_1 | teacher_full | transformer, "seed", "config": {...}}.
    Writes <out>/identity.json, *_history.csv, summary.json (validation-selected checkpoint, independent eval set)."""
    cfg = replace(Config(), **spec.get("config", {}))
    seed, family = int(spec["seed"]), spec["family"]
    out = out_root / spec["name"]
    out.mkdir(parents=True, exist_ok=True)
    ident = identity(cfg, spec, device)
    p = out / "identity.json"
    assert not p.exists() or json.loads(p.read_text()) == json.loads(json.dumps(ident)), f"{p}: another run lives here"
    p.write_text(json.dumps(ident, indent=2))
    if (out / "summary.json").exists():
        return json.loads((out / "summary.json").read_text())
    val, ev_set = validation_set(cfg), evaluation_set(cfg)
    teacher, rnn = make_models(cfg, seed, device)
    detector = load_context_detector(cfg, device) if cfg.far == "ctx_far" else None
    summary, tic = dict(name=spec["name"], family=family, seed=seed), time.perf_counter()

    def rnn_monitor():
        ev = answer_logits("rnn", rnn, val, cfg, device)
        return float(ev[ev.delay >= cfg.long_delay].ce.mean()), long_metrics(ev, cfg, "val_rnn")

    if family in ("smt", "teacher_full"):
        params = [*teacher.parameters(), *rnn.parameters()] if family == "smt" else list(teacher.parameters())

        def smt_step(step):
            teacher.train(); rnn.train()
            batch = sample(cfg, cfg.batch_size, derive_seed(301, seed, step))
            gen = torch.Generator().manual_seed(derive_seed(303, seed, step))
            with seeded(derive_seed(304, seed, step)):
                if family == "teacher_full":
                    return teacher_full_access_loss(teacher, batch, cfg, device)
                return smt_loss(teacher, rnn, batch, cfg, device, gen, detector)

        def smt_monitor():
            copy_readout(teacher, rnn)
            kind = "teacher_full" if family == "teacher_full" else "teacher"
            ev = answer_logits(kind, teacher, val, cfg, device)
            metrics = long_metrics(ev, cfg, "val_teacher")
            if family == "smt":
                metrics.update(long_metrics(answer_logits("rnn", rnn, val, cfg, device), cfg, "val_rnn"))
            if cfg.task == "mqar" and family == "smt":
                metrics.update(long_metrics(answer_logits("teacher_pre", teacher, val, cfg, device, "after_pairs"),
                                            cfg, "val_teacher_pre"))
            return float(ev[ev.delay >= cfg.long_delay].ce.mean()), metrics

        r = run_stage("smt", cfg.smt_steps, cfg.smt_lr, params, smt_step, smt_monitor,
                      dict(teacher=teacher, rnn=rnn), out, cfg, ident, log)
        summary.update(smt_best_step=r["best_step"], smt_seconds=r["seconds"])
        copy_readout(teacher, rnn)
        kinds = [("teacher_full" if family == "teacher_full" else "teacher", teacher, "query")]
        if cfg.task == "mqar" and family == "smt":
            kinds += [("teacher_pre", teacher, "after_pairs"), ("teacher_pre", teacher, "before_query")]
        if family == "smt":
            kinds.append(("rnn", rnn, "query"))           # "RNN before DMT" (readout copied from the teacher)
        for kind, model, anchor in kinds:
            ev = answer_logits(kind, model, ev_set, cfg, device, anchor)
            summary.update(long_metrics(ev, cfg, f"eval_{kind}{'' if anchor == 'query' else '_' + anchor}_smt"))
        if family == "smt" and cfg.dmt_steps > 0:
            teacher.requires_grad_(False); teacher.eval()
            frozen = copy.deepcopy(teacher.state_dict())

            def dmt_step(step):
                rnn.train()
                batch = sample(cfg, cfg.batch_size, derive_seed(302, seed, step))
                with seeded(derive_seed(305, seed, step)):
                    return dmt_loss(teacher, rnn, batch, cfg, device)

            r = run_stage("dmt", cfg.dmt_steps, cfg.dmt_lr, list(rnn.parameters()), dmt_step, rnn_monitor,
                          dict(rnn=rnn), out, cfg, ident, log)
            assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in frozen.items()), "teacher changed in DMT"
            summary.update(dmt_best_step=r["best_step"], dmt_seconds=r["seconds"])
            summary.update(long_metrics(answer_logits("rnn", rnn, ev_set, cfg, device), cfg, "eval_rnn_dmt"))
    elif family in ("bptt_full", "bptt_1"):
        def bptt_step(step):
            rnn.train()
            batch = sample(cfg, cfg.batch_size, derive_seed(301, seed, step))
            with seeded(derive_seed(306, seed, step)):
                return bptt_loss(rnn, batch, cfg, device, truncate=family == "bptt_1")

        r = run_stage("bptt", cfg.bptt_steps, cfg.bptt_lr, list(rnn.parameters()), bptt_step, rnn_monitor,
                      dict(rnn=rnn), out, cfg, ident, log)
        summary.update(bptt_best_step=r["best_step"], bptt_seconds=r["seconds"])
        summary.update(long_metrics(answer_logits("rnn", rnn, ev_set, cfg, device), cfg, "eval_rnn_bptt"))
    elif family == "transformer":
        with seeded(seed):
            model = CausalTransformer(cfg).to(device)

        def lm_step(step):
            model.train()
            return lm_loss(model, sample(cfg, cfg.batch_size, derive_seed(301, seed, step)), device)

        def lm_monitor():
            model.eval()
            ev = answer_logits("lm", model, val, cfg, device)
            return float(ev[ev.delay >= cfg.long_delay].ce.mean()), long_metrics(ev, cfg, "val_lm")

        r = run_stage("lm", cfg.lm_steps, cfg.lm_lr, list(model.parameters()), lm_step, lm_monitor,
                      dict(model=model), out, cfg, ident, log)
        model.eval()
        summary.update(lm_best_step=r["best_step"], lm_seconds=r["seconds"])
        summary.update(long_metrics(answer_logits("lm", model, ev_set, cfg, device), cfg, "eval_lm"))
    else:
        raise ValueError(family)
    summary["wall_seconds"] = time.perf_counter() - tic
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


# ======================================================================================================================
# CLI: traces (token tables) and a tiny CPU demo
# ======================================================================================================================

def trace_table(cfg: Config, batch: dict, b: int = 0) -> pd.DataFrame:
    """Position table of one sequence: meaning, token, model input, label, loss mask, delay."""
    x, y = batch["x"][b], batch["y"][b]
    rows = []
    for t in range(len(x)):
        tok, lab = int(x[t]), int(y[t])
        if cfg.task == "stack":
            meaning = f"push {tok}" if tok > 0 else "pop"
        else:
            N = cfg.num_kv_pairs
            meaning = (f"key of pair {t // 2}" if t < 2 * N and t % 2 == 0 else f"value of pair {t // 2}" if t < 2 * N
                       else "query key" if lab != IGNORE else "filler 0")
        rows.append(dict(t=t, meaning=meaning, token=tok, input=tok, label=lab, loss_mask=int(lab != IGNORE),
                         source=int(batch["source_pos"][b, t]), delay=int(batch["delay"][b, t])))
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["trace", "demo", "train"])
    p.add_argument("--task", default="stack", choices=["stack", "mqar"])
    p.add_argument("--spec", help="train: JSON file with one run spec, or a plan file + --index")
    p.add_argument("--index", type=int)
    p.add_argument("--out", default=str(ROOT / "reruns" / "runs"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    pd.set_option("display.width", 200); pd.set_option("display.max_rows", 100)
    device = torch.device("cuda" if torch.cuda.is_available() and a.command == "train" else "cpu")
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.command == "trace":
        cfg = Config() if a.task == "stack" else Config(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4,
                                                        future_window=16)
        batch = validation_set(cfg)
        print(trace_table(cfg, batch, 0).to_string(index=False))
    elif a.command == "demo":
        demo(a.task, log)
    else:
        spec = json.loads(Path(a.spec).read_text())
        if isinstance(spec, dict) and "runs" in spec:
            spec = spec["runs"][a.index]
        print(json.dumps(run(spec, Path(a.out), device, log), indent=2))


DEMO = {   # tiny CPU settings (about 2 min each); not results, only a walk through the code
    "stack": Config(seq_len=16, max_stack_size=4, long_delay=4, d_model=32, n_heads=2, smt_steps=150, dmt_steps=100,
                    eval_every=50, validation_examples=64, eval_examples=64, batch_size=16, warmup_steps=20),
    "mqar": Config(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4, future_window=16, d_model=32, n_heads=2,
                   smt_steps=150, dmt_steps=50, eval_every=50, validation_examples=64, eval_examples=64,
                   batch_size=16, warmup_steps=20),
}


@torch.no_grad()
def demo(task: str = "stack", log=print):
    """Train the tiny model, then follow ONE answer (the last pop / the last query of an evaluation sequence) through
    the windows, the SMT loss term and the evaluation rows."""
    cfg = DEMO[task]
    out = ROOT / "reruns" / "demo"
    spec = dict(name=f"demo_{task}_smt", family="smt", seed=1,
                config={k: v for k, v in asdict(cfg).items() if getattr(Config(), k) != v})
    with torch.enable_grad():
        print(json.dumps(run(spec, out, torch.device("cpu"), log), indent=2))
    teacher, rnn = make_models(cfg, 1, torch.device("cpu"))
    teacher.load_state_dict(torch.load(out / spec["name"] / "smt_best.pt", weights_only=True)["models"]["teacher"])
    rnn.load_state_dict(torch.load(out / spec["name"] / "dmt_best.pt", weights_only=True)["models"]["rnn"])
    teacher.eval(); rnn.eval()
    batch = sample(cfg, 1, derive_seed(901, 0))            # sequence 0 of the independent evaluation set
    table = trace_table(cfg, batch)
    print(table.to_string(index=False))
    t = int(table[table.label != IGNORE].t.iloc[-1])
    T, y_t = cfg.seq_len, int(batch["y"][0, t])
    w = windows(batch)
    print(f"\nAnswer at t = {t}: input x_t = {int(batch['x'][0, t])}, label y_t = {y_t}, delay {int(batch['delay'][0, t])}.")
    print(f"windows row t: x_ctx = {w['x_ctx'][t].tolist()}  (encoder sees x_0..x_t; -1 = padding)")
    m_t = teacher.encode(w["x_ctx"][t:t + 1])
    ce = author("model").calc_cross_entropy_loss(teacher.decode(m_t, w["x_fut"][t:t + 1]), w["y_fut"][t:t + 1].clamp(min=0))
    print(f"SMT loss, anchor t, output j = 0: y_fut[t, 0] = {int(w['y_fut'][t, 0])}, CE = {float(ce[0, 0]):.3f} "
          f"(enters L_dec^Tf because j = 0 < Tf = {cfg.future_window} and valid[t, 0] = {bool(w['valid'][t, 0])})")
    for a in range(max(0, t - 3), t):
        print(f"  the same y_t is also output j = {t - a} of anchor a = {a}: decoder sees m_a and x_fut[a, :{t - a}] = "
              f"{w['x_fut'][a, :t - a].tolist()}  ({'near' if t - a < cfg.future_window else 'far'} target)")
    for kind, model in (("teacher", teacher), ("rnn", rnn)):
        row = answer_logits(kind, model, batch, cfg, torch.device("cpu"))
        row = row[row.t == t].iloc[0]
        print(f"evaluation row ({kind}): pred = {int(row.pred)}, target = {int(row.target)}, correct = {bool(row.correct)}, "
              f"CE = {row.ce:.3f}")
    if task == "mqar":
        for anchor in ("after_pairs", "before_query"):
            row = answer_logits("teacher_pre", teacher, batch, cfg, torch.device("cpu"), anchor)
            row = row[row.t == t].iloc[0]
            a = 2 * cfg.num_kv_pairs - 1 if anchor == "after_pairs" else t - 1
            print(f"evaluation row (teacher_pre, {anchor}: memory m_{a}, decoder inputs x_{a + 1}..x_{t}): "
                  f"pred = {int(row.pred)}, correct = {bool(row.correct)}, among written = {bool(row.cand_correct)}")


if __name__ == "__main__":
    main()
