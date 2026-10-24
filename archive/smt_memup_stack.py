"""SMT x MemUP on the author's Stack task (no noise).

Experiment code for ``02_smt_memup_stack.ipynb``. Importing this module never
trains anything. Long runs are started explicitly from the notebook or from the
command line (``python smt_memup_stack.py --help``).

What comes from the authors (pinned commits, imported without modification):
  * ``StackDataset`` (data), ``Teacher``/``RNN`` and loss helpers (SMT repo);
  * the MemUP principle: frozen detector score -> top-K future targets.
What is ours (documented in the notebook):
  * replay annotation of the Stack generator (source position, delay, depth);
  * the all-anchor SMT forward for synthetic data (the authors' synthetic
    forward is not published; the paper states Tc=Tf=T and all timesteps);
  * supervision of the first transition f(0, x_0) -> E(x_0);
  * one-step DMT started from the zero state;
  * auxiliary selected-target CE, detectors, selection audit, statistics.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

IGNORE = -1
MODULE_PATH = Path(__file__).resolve()
DEFAULT_ROOT = MODULE_PATH.parent
EXPECTED_REVISIONS = {
    "smt": "cf0d0f04be606ab9d97b1a449a6846450d656ad0",
    "memup": "d27657137f6e427e30fcd4ae885b20b071b2917d",
}
VARIANTS_AGREED = ("baseline", "uniform", "random_k", "surprise_k")
# Optional, pre-declared extensions. They are never part of the agreed four.
#   surprise_ctx_k : non-oracle score that conditions on exactly the inputs the
#                    SMT decoder sees for this anchor (x_{t+1..k}), no memory.
#   oracle_prefix_k: ORACLE control. Uses generator metadata (source <= t) to
#                    restrict the pool. Diagnostic only, not a MemUP algorithm.
VARIANTS_OPTIONAL = ("surprise_ctx_k", "oracle_prefix_k")
ALL_VARIANTS = VARIANTS_AGREED + VARIANTS_OPTIONAL

# --------------------------------------------------------------------------------------
# Runtime: be a polite neighbour on a shared server.
# --------------------------------------------------------------------------------------

def setup_runtime(threads: int = 4, gpu_memory_fraction: float | None = None,
                  device: str | None = None) -> torch.device:
    """Limit CPU threads and (optionally) the share of GPU memory this process may use."""
    torch.set_num_threads(max(1, int(threads)))
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    dev = torch.device(device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    if dev.type == "cuda" and gpu_memory_fraction:
        torch.cuda.set_per_process_memory_fraction(float(gpu_memory_fraction), dev)
    return dev


# --------------------------------------------------------------------------------------
# Author sources
# --------------------------------------------------------------------------------------

def repository_revision(path: Path) -> str:
    out = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"],
                         check=True, capture_output=True, text=True)
    return out.stdout.strip()


def _load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Sources:
    """Pinned author code. Loads single files to skip my_datasets/__init__ (MNIST/TinyStories)."""

    def __init__(self, root: Path | str = DEFAULT_ROOT, check_revisions: bool = True):
        self.root = Path(root)
        self.smt_root = self.root / "upstream" / "smt"
        self.memup_root = self.root / "upstream" / "memup"
        self.revisions = {k: repository_revision(self.root / "upstream" / k) for k in EXPECTED_REVISIONS}
        if check_revisions and self.revisions != EXPECTED_REVISIONS:
            raise RuntimeError(f"Upstream revisions changed: {self.revisions}")
        self.data = _load_file("smt_author_synthetic", self.smt_root / "src/my_datasets/synthetic.py")
        self.model = _load_file("smt_author_model", self.smt_root / "src/model.py")


SRC: Sources | None = None


def sources(root: Path | str = DEFAULT_ROOT) -> Sources:
    global SRC
    if SRC is None:
        SRC = Sources(root)
    return SRC


def code_sha256() -> str:
    """Hash of this file: any code change creates a new run directory (no stale resume)."""
    return hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------
# Seeds
# --------------------------------------------------------------------------------------

def derive_seed(*parts: int) -> int:
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


@contextmanager
def isolated_seed(seed: int):
    """Seed global numpy/python/torch RNGs inside the block and restore them afterwards."""
    np_state, py_state, th_state = np.random.get_state(), random.getstate(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(int(seed) % 2**32)
        random.seed(int(seed))
        torch.manual_seed(int(seed))
        yield
    finally:
        np.random.set_state(np_state)
        random.setstate(py_state)
        torch.set_rng_state(th_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Config:
    # data: author StackDataset; seq_len/depth are our choices (paper does not publish them)
    seq_len: int = 64
    n_elements: int = 4
    max_stack_size: int = 8
    long_delay: int = 8              # primary group: pops with delay >= long_delay
    # model (author classes; sizes are ours)
    d_model: int = 64
    mem_tokens: int = 8
    encoder_depth: int = 3
    decoder_depth: int = 2
    rnn_depth: int = 2
    n_heads: int = 4
    # SMT (author defaults for coefficients; paper: all anchors, batch 32 for synthetic)
    batch_size: int = 32
    anchors_per_sequence: int = 0    # 0 = all anchors (paper); >0 = Monte Carlo subsample
    smt_steps: int = 3000
    smt_lr: float = 3e-4
    coef_dec: float = 1.0
    coef_dyn: float = 0.1
    coef_unif: float = 0.001
    # DMT (one-step imitation from the zero state; readout trained on detached states)
    dmt_batch_size: int = 32
    dmt_steps: int = 1500
    dmt_lr: float = 1e-4
    dmt_coef_dyn: float = 10.0
    # optimisation
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    # auxiliary loss / MemUP adaptation
    K: int = 4
    alpha_grid: tuple = (0.1, 0.3, 1.0)
    local_window: int = 8            # r of the agreed window detector
    detector_steps: int = 2000
    detector_batch_size: int = 32
    detector_lr: float = 1e-3
    detector_hidden: int = 64
    detector_embed: int = 16
    detector_seed: int = 911
    # protocol
    eval_every: int = 250
    eval_batch_size: int = 128
    validation_examples: int = 512   # checkpoint selection
    tuning_examples: int = 512       # baseline gate and alpha selection
    test_examples: int = 2048        # created only in the final report
    tuning_seed: int = 107
    seeds: tuple = (7, 17, 27)
    bootstrap_repeats: int = 2000
    gate_teacher_margin: float = 0.10   # teacher long accuracy >= chance + margin
    gate_rnn_margin: float = 0.05       # RNN (after DMT) long accuracy >= chance + margin
    amp_bf16: bool = False

    @property
    def vocab_size(self) -> int:
        return self.n_elements + 1

    @property
    def mem_size(self) -> int:
        return self.d_model * self.mem_tokens


def config_json(cfg: Config) -> dict:
    return json.loads(json.dumps(asdict(cfg)))


def run_identity(cfg: Config, device: torch.device) -> dict:
    src = sources()
    return dict(config=config_json(cfg), code_sha256=code_sha256(), repositories=src.revisions,
                torch=str(torch.__version__), x_transformers=importlib.metadata.version("x-transformers"),
                device=device.type, python=platform.python_version())


# --------------------------------------------------------------------------------------
# Data: author Stack generator + replay annotation (ours)
# --------------------------------------------------------------------------------------

def annotate_stack(raw: dict) -> dict:
    """Replay pushes/pops; add source position of every popped element, its delay and depth."""
    x, y = raw["x"].long(), raw["y"].long()
    B, T = x.shape
    source = torch.full_like(x, IGNORE)
    depth = torch.zeros_like(x)
    xs, ys = x.tolist(), y.tolist()
    for b in range(B):
        stack = []
        for t in range(T):
            token = xs[b][t]
            if token > 0:
                stack.append((token, t))
                if ys[b][t] != IGNORE:
                    raise AssertionError("push must not have a target")
            elif token == 0:
                if not stack:
                    raise AssertionError("pop from empty stack")
                value, s = stack.pop()
                if ys[b][t] != value:
                    raise AssertionError("pop target differs from replayed top of stack")
                source[b, t] = s
            else:
                raise AssertionError("author Stack data has no padding")
            depth[b, t] = len(stack)
    target_mask = y.ne(IGNORE)
    positions = torch.arange(T).expand(B, T)
    delay = torch.where(target_mask, positions - source, torch.full_like(x, IGNORE))
    return dict(x=x, y=y, target_mask=target_mask, source_pos=source, delay=delay, depth=depth)


def sample_stack(cfg: Config, batch_size: int, seed: int) -> dict:
    ds = sources().data.StackDataset(n_elements=cfg.n_elements, max_stack_size=cfg.max_stack_size)
    with isolated_seed(seed):
        raw = ds.sample_batch(batch_size=batch_size, ctx_len=cfg.seq_len)
    batch = annotate_stack(raw)
    check_batch(batch, cfg)
    return batch


def check_batch(batch: dict, cfg: Config) -> bool:
    x, y, m = batch["x"], batch["y"], batch["target_mask"]
    assert x.shape == y.shape and x.dtype == y.dtype == torch.long
    assert ((x >= 0) & (x < cfg.vocab_size)).all()
    assert ((y == IGNORE) | ((y >= 1) & (y < cfg.vocab_size))).all()
    assert torch.equal(m, x.eq(0)), "every pop (x=0) has exactly one target"
    src = batch["source_pos"]
    pos = torch.arange(x.shape[1]).expand_as(x)
    assert (src[m] >= 0).all() and (src[m] < pos[m]).all()
    assert torch.equal(x.gather(1, src.clamp(min=0))[m], y[m])
    assert int(batch["depth"].max()) <= cfg.max_stack_size
    return True


def fingerprints(batch: dict) -> list[str]:
    return [hashlib.sha256(torch.stack((a, b)).numpy().tobytes()).hexdigest()
            for a, b in zip(batch["x"], batch["y"])]


def split_hash(batch: dict) -> str:
    return hashlib.sha256("".join(fingerprints(batch)).encode()).hexdigest()


def slice_batch(batch: dict, lo: int, hi: int) -> dict:
    return {k: v[lo:hi] for k, v in batch.items()}


def data_audit(batch: dict, cfg: Config) -> dict:
    d = batch["delay"][batch["target_mask"]].float()
    return dict(sequences=len(batch["x"]), pops=len(d), long_pops=int((d >= cfg.long_delay).sum()),
                long_fraction=float((d >= cfg.long_delay).float().mean()),
                ge16_fraction=float((d >= 16).float().mean()),
                delay_median=float(d.median()), delay_p95=float(torch.quantile(d, .95)),
                unique_sequences=len(set(fingerprints(batch))))


# --------------------------------------------------------------------------------------
# All-anchor SMT windows (ours; paper: Tc = Tf = T, every timestep)
# --------------------------------------------------------------------------------------

def windows(batch: dict, anchors: torch.Tensor | None = None) -> dict:
    """For each (sequence b, anchor t):
        x_ctx = x[t-T+1 .. t]   (left padded with -1, encoder masks it)
        x_fut = x[t+1 .. t+T]   (right padded; decoder output j sees x_fut[:j] only)
        y_fut = y[t .. t+T-1]   (output j predicts y_{t+j}; j = 0 is the RNN readout target)
    Returns tensors flattened over (b, t) in row-major order.
    """
    x, y = batch["x"], batch["y"]
    B, T = x.shape
    unf = lambda z, left, right: F.pad(z, (left, right), value=IGNORE).unfold(1, T, 1)
    x_ctx = unf(x, T - 1, 0)                     # (B, T, T)
    x_fut = unf(x, 0, T)[:, 1:T + 1]             # (B, T, T)
    y_fut = unf(y, 0, T - 1)                     # (B, T, T)
    src = unf(batch["source_pos"], 0, T - 1)
    delay = unf(batch["delay"], 0, T - 1)
    t = torch.arange(T)
    anchor = t.view(1, T, 1).expand(B, T, T)
    target_t = anchor + torch.arange(T).view(1, 1, T)
    seq = torch.arange(B).view(B, 1, 1).expand(B, T, T)
    valid = y_fut.ne(IGNORE)
    w = dict(x_ctx=x_ctx, x_fut=x_fut, y_fut=y_fut, valid=valid, source=src, delay=delay,
             target_t=target_t, anchor=anchor[..., 0], seq=seq[..., 0],
             prefix=valid & (src <= anchor),          # memory m_t must carry the element
             future=valid & (src > anchor))           # element is pushed inside x_fut
    flat = {k: v.reshape(B * T, *v.shape[2:]) for k, v in w.items()}
    if anchors is not None:
        flat = {k: v[anchors] for k, v in flat.items()}
    return flat


def sample_anchor_rows(cfg: Config, B: int, generator: torch.Generator) -> torch.Tensor | None:
    if cfg.anchors_per_sequence <= 0 or cfg.anchors_per_sequence >= cfg.seq_len:
        return None
    T = cfg.seq_len
    picks = torch.rand(B, T, generator=generator).argsort(1)[:, :cfg.anchors_per_sequence]
    return (torch.arange(B)[:, None] * T + picks).flatten().sort().values


# --------------------------------------------------------------------------------------
# Models (author classes)
# --------------------------------------------------------------------------------------

def make_models(cfg: Config, seed: int, device: torch.device):
    m = sources().model
    common = dict(vocab_size=cfg.vocab_size, mem_size=cfg.mem_size, mem_tokens=cfg.mem_tokens,
                  d_model=cfg.d_model, decoder_depth=cfg.decoder_depth, n_heads=cfg.n_heads)
    with isolated_seed(seed):
        teacher = m.Teacher(m.TeacherConfig(**common, encoder_depth=cfg.encoder_depth)).to(device)
        rnn = m.RNN(m.RNNConfig(**common, rnn_depth=cfg.rnn_depth)).to(device)
    copy_readout(teacher, rnn)
    return teacher, rnn


def copy_readout(teacher, rnn) -> None:
    """As in train_smt.py: teacher decoder weights become the RNN readout."""
    rnn.decoder.load_state_dict(teacher.decoder.state_dict())
    rnn.to_logits.load_state_dict(teacher.to_logits.state_dict())


def state_hash(*models) -> str:
    h = hashlib.sha256()
    for i, model in enumerate(models):
        for k, v in model.state_dict().items():
            h.update(f"{i}:{k}".encode())
            h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def encode_chunked(teacher, x_ctx: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    return torch.cat([teacher.encode(p) for p in x_ctx.split(chunk)])


@torch.no_grad()
def teacher_states(teacher, batch: dict, device, chunk: int = 4096) -> torch.Tensor:
    B, T = batch["x"].shape
    w = windows(batch)
    return encode_chunked(teacher, w["x_ctx"].to(device), chunk).reshape(B, T, -1)


def rnn_rollout(rnn, x: torch.Tensor, init: torch.Tensor | None = None, detach: bool = False):
    """m_t = f(m_{t-1}, x_t) with m_{-1} = 0 (or init). Returns (B, T, Z)."""
    B, T = x.shape
    state = torch.zeros(B, rnn.cfg.mem_size, device=x.device) if init is None else init
    out = []
    for t in range(T):
        state = rnn.rnn_update(state.detach() if detach else state, x[:, t:t + 1])
        out.append(state)
    return torch.stack(out, 1)


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------

def pair_mean(ce: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Author masked_mean over all valid (anchor, target) pairs; 0 if there are none."""
    return (ce * valid).sum() / valid.sum().clamp(min=1)


def weighted_selected_loss(ce: torch.Tensor, pool: torch.Tensor, selected: torch.Tensor) -> torch.Tensor:
    """sum_t N_t * mean_{k in S_t} ce / sum_t N_t,  N_t = |pool_t|.

    With pool = all valid targets, E[random-K] equals the author decoder loss and
    selecting the whole pool reproduces it exactly, so 'uniform' == (1+alpha)*L_dec.
    """
    assert not (selected & ~pool).any()
    n = pool.sum(1).to(ce.dtype)
    k = selected.sum(1).to(ce.dtype)
    row = (ce * selected).sum(1) / k.clamp(min=1)
    return (n * row).sum() / n.sum().clamp(min=1)


def topk_mask(scores: torch.Tensor, pool: torch.Tensor, K: int, generator: torch.Generator) -> torch.Tensor:
    """Top-K of `scores` inside `pool` per row; exact ties are broken by a seeded random order.
    Equivalent to MemUP TopKSelector(time_dependent_selection=True) restricted to the
    pool, but never repeats an index when the pool has fewer than K targets."""
    s = torch.where(pool, scores.float(), torch.full_like(scores, -torch.inf, dtype=torch.float32))
    perm = torch.rand(s.shape, generator=generator, device="cpu").to(s.device).argsort(1)
    order = torch.sort(s.gather(1, perm), dim=1, descending=True, stable=True).indices
    idx = perm.gather(1, order)[:, :K]
    chosen = torch.zeros_like(pool).scatter(1, idx, True) & pool
    assert torch.equal(chosen.sum(1), pool.sum(1).clamp(max=K))
    return chosen


class WindowDetector(torch.nn.Module):
    """Agreed score: q(y_k | x_{k-r+1..k}); r = cfg.local_window. GRU from zero state."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embed = torch.nn.Embedding(cfg.vocab_size + 1, cfg.detector_embed, padding_idx=0)
        self.gru = torch.nn.GRU(cfg.detector_embed, cfg.detector_hidden, batch_first=True)
        self.head = torch.nn.Linear(cfg.detector_hidden, cfg.vocab_size)

    def forward(self, x_windows):  # (N, r) with -1 padding -> logits (N, V)
        _, h = self.gru(self.embed(x_windows + 1))
        return self.head(h[-1])

    def scores(self, batch: dict, device) -> torch.Tensor:
        """(B, T) surprise of y_k; +inf-free; non-targets get 0 (they are never in a pool)."""
        x, y = batch["x"], batch["y"]
        r = self.cfg.local_window
        win = F.pad(x, (r - 1, 0), value=IGNORE).unfold(1, r, 1)  # (B, T, r)
        logits = self(win.reshape(-1, r).to(device)).reshape(*x.shape, -1)
        ce = F.cross_entropy(logits.flatten(0, 1), y.clamp(min=0).flatten().to(device), reduction="none")
        return (ce.reshape(x.shape) * batch["target_mask"].to(device)).detach()


class ContextDetector(torch.nn.Module):
    """Optional score: q(y_{t+j} | x_{t+1..t+j}) - the inputs the SMT decoder sees for anchor t,
    WITHOUT memory. High surprise = target not inferable from decoder-visible inputs."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.bos = cfg.vocab_size + 1
        self.embed = torch.nn.Embedding(cfg.vocab_size + 2, cfg.detector_embed, padding_idx=0)
        self.gru = torch.nn.GRU(cfg.detector_embed, cfg.detector_hidden, batch_first=True)
        self.head = torch.nn.Linear(cfg.detector_hidden, cfg.vocab_size)

    def forward(self, x_fut):  # (N, T) future inputs of each anchor -> logits (N, T, V)
        inp = torch.cat([torch.full_like(x_fut[:, :1], self.bos - 1), x_fut[:, :-1]], 1)
        h, _ = self.gru(self.embed(inp + 1))
        return self.head(h)

    def pair_scores(self, w: dict, device, chunk: int = 4096) -> torch.Tensor:
        out = []
        for xf, yf in zip(w["x_fut"].split(chunk), w["y_fut"].split(chunk)):
            logits = self(xf.to(device))
            ce = F.cross_entropy(logits.flatten(0, 1), yf.clamp(min=0).flatten().to(device), reduction="none")
            out.append(ce.reshape(yf.shape) * yf.ne(IGNORE).to(device))
        return torch.cat(out).detach()


def selection_scores(variant: str, w: dict, batch: dict, detectors: dict, device) -> torch.Tensor | None:
    if variant == "surprise_k":
        local = detectors["window"].scores(batch, device)                       # (B, T)
        return local[w["seq"].to(device)[:, None], w["target_t"].clamp(max=batch["x"].shape[1] - 1).to(device)]
    if variant == "surprise_ctx_k":
        return detectors["context"].pair_scores(w, device)
    return None


def smt_loss(teacher, rnn, batch: dict, cfg: Config, variant: str, alpha: float,
             detectors: dict, generator: torch.Generator, device):
    """Author SMT loss (all anchors, pair-averaged decoder CE, joint dynamics, uniformity)
    + coef_dec * alpha * L_selected. Scores are detached; the selection has no gradient."""
    m_model = sources().model
    B, T = batch["x"].shape
    rows = sample_anchor_rows(cfg, B, generator)
    w_all = windows(batch)
    w = w_all if rows is None else {k: v[rows] for k, v in w_all.items()}
    ctx = w_all["x_ctx"].to(device)
    memory = encode_chunked(teacher, ctx).reshape(B, T, -1)          # m_t for every t
    mem_rows = memory.reshape(B * T, -1) if rows is None else memory.reshape(B * T, -1)[rows.to(device)]
    logits = teacher.decode(mem_rows, w["x_fut"].to(device))
    target = w["y_fut"].to(device)
    valid = w["valid"].to(device)
    ce = m_model.calc_cross_entropy_loss(logits, target.clamp(min=0))
    dec = pair_mean(ce, valid)
    audit = {}
    if variant == "baseline":
        aux = dec * 0
    elif variant == "uniform":
        aux = dec
    else:
        if variant == "random_k":
            pool = valid
            scores = torch.zeros_like(ce)
        elif variant == "oracle_prefix_k":   # ORACLE: generator metadata restricts the pool
            pool = w["prefix"].to(device)
            scores = torch.zeros_like(ce)
        elif variant in ("surprise_k", "surprise_ctx_k"):
            pool = valid
            with torch.no_grad():
                scores = selection_scores(variant, w, batch, detectors, device)
        else:
            raise ValueError(variant)
        chosen = topk_mask(scores, pool, cfg.K, generator)
        aux = weighted_selected_loss(ce, pool, chosen)
        c = chosen.cpu()
        if c.any():
            audit = dict(sel_prefix=float(w["prefix"][c].float().mean()),
                         sel_long_prefix=float((w["prefix"] & w["delay"].ge(cfg.long_delay))[c].float().mean()),
                         sel_future=float(w["future"][c].float().mean()))
    # dynamics: f(m_{t-1}, x_t) -> m_t for all t, with m_{-1} = 0 (first transition is ours).
    prev = torch.cat([torch.zeros_like(memory[:, :1]), memory[:, :-1]], 1)
    pred = rnn.rnn_update(prev.reshape(B * T, -1), batch["x"].to(device).reshape(B * T, 1))
    dyn = m_model.calc_mse_loss(pred, memory.reshape(B * T, -1)).mean()   # detach_rnn=False (author)
    unif = m_model.calc_uniformity_loss(memory.reshape(B * T * cfg.mem_tokens, cfg.d_model))
    loss = cfg.coef_dec * (dec + alpha * aux) + cfg.coef_dyn * dyn + cfg.coef_unif * unif
    return loss, dict(loss=loss, decoder=dec, auxiliary=aux, dynamics=dyn, uniformity=unif, **audit)


def dmt_loss(teacher, rnn, batch: dict, cfg: Config, device):
    """One-step DMT from the zero state: teacher frozen; updater learns MSE to teacher states
    on its own (detached) states; readout CE on detached states trains only the head."""
    target = teacher_states(teacher, batch, device)
    x = batch["x"].to(device)
    states = rnn_rollout(rnn, x, detach=True)
    dyn = (states - target).square().mean()
    logits = rnn.rnn_readout(states.detach().flatten(0, 1)).reshape(*x.shape, -1)
    readout = pair_mean(sources().model.calc_cross_entropy_loss(logits, batch["y"].to(device).clamp(min=0)),
                        batch["target_mask"].to(device))
    loss = cfg.dmt_coef_dyn * dyn + readout
    return loss, dict(loss=loss, dynamics=dyn, readout=readout)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(teacher, rnn, batch: dict, cfg: Config, device, **labels) -> pd.DataFrame:
    """One row per (pop, model). model in {teacher, rnn}. ce/correct use the model's own head;
    shared_* use the frozen teacher head on the same state (separates memory from readout)."""
    teacher.eval(); rnn.eval()
    frames = []
    for lo in range(0, len(batch["x"]), cfg.eval_batch_size):
        part = slice_batch(batch, lo, lo + cfg.eval_batch_size)
        tm = teacher_states(teacher, part, device)
        rm = rnn_rollout(rnn, part["x"].to(device))
        b, t = torch.where(part["target_mask"])
        y = part["y"].to(device)
        for name, mem, head in (("teacher", tm, teacher.teacher_readout), ("rnn", rm, rnn.rnn_readout)):
            z = mem.flatten(0, 1)
            own = head(z).reshape(*y.shape, -1)
            shared = teacher.teacher_readout(z).reshape(*y.shape, -1)
            ce = F.cross_entropy(own.flatten(0, 1), y.clamp(min=0).flatten(), reduction="none").reshape(y.shape).cpu()
            sce = F.cross_entropy(shared.flatten(0, 1), y.clamp(min=0).flatten(), reduction="none").reshape(y.shape).cpu()
            pred, spred = own.argmax(-1).cpu(), shared.argmax(-1).cpu()
            frames.append(pd.DataFrame(dict(
                **labels, model=name, sequence=(b + lo).numpy(), t=t.numpy(),
                delay=part["delay"][b, t].numpy(), depth_after=part["depth"][b, t].numpy(),
                target=part["y"][b, t].numpy(), correct=(pred[b, t] == part["y"][b, t]).numpy(),
                ce=ce[b, t].numpy(), shared_correct=(spred[b, t] == part["y"][b, t]).numpy(),
                shared_ce=sce[b, t].numpy(),
                memory_mse=(mem - tm).square().mean(-1).cpu()[b, t].numpy())))
    return pd.concat(frames, ignore_index=True)


def summarize(events: pd.DataFrame, cfg: Config, keys=("seed", "variant", "stage", "split", "model")) -> pd.DataFrame:
    keys = [k for k in keys if k in events.columns]
    out = []
    groups = {"all": events, f"delay>={cfg.long_delay}": events[events.delay >= cfg.long_delay],
              "delay>=16": events[events.delay >= 16], f"delay<{cfg.long_delay}": events[events.delay < cfg.long_delay]}
    for name, data in groups.items():
        t = data.groupby(keys, observed=True).agg(
            pops=("correct", "size"), accuracy=("correct", "mean"), ce=("ce", "mean"),
            shared_accuracy=("shared_correct", "mean"), shared_ce=("shared_ce", "mean"),
            memory_mse=("memory_mse", "mean")).reset_index()
        t.insert(0, "group", name)
        out.append(t)
    return pd.concat(out, ignore_index=True)


@torch.no_grad()
def retention_probe(teacher, rnn, batch: dict, cfg: Config, device,
                    horizons=(0, 1, 2, 4, 8, 16, 32)) -> pd.DataFrame:
    """H2 diagnostic. Start the RNN from the TEACHER state m_s and apply L updates on the true
    inputs x_{s+1..s+L}; read y_{s+L} with the frozen teacher head. Compare with the teacher
    state at s+L (same head). Accuracy falling with L while the teacher stays high means the
    updater loses information that the encoder had."""
    teacher.eval(); rnn.eval()
    rows = []
    for lo in range(0, len(batch["x"]), cfg.eval_batch_size):
        part = slice_batch(batch, lo, lo + cfg.eval_batch_size)
        tm = teacher_states(teacher, part, device)
        x = part["x"].to(device)
        B, T, Z = tm.shape
        for L in horizons:
            if L >= T:
                continue
            s_idx = torch.arange(T - L)
            state = tm[:, s_idx].reshape(-1, Z)
            for step in range(1, L + 1):
                state = rnn.rnn_update(state, x[:, s_idx + step].reshape(-1, 1))
            q = s_idx + L
            target = part["y"][:, q]
            mask = target.ne(IGNORE)
            for name, z in (("rnn_from_teacher", state), ("teacher", tm[:, q].reshape(-1, Z))):
                logits = teacher.teacher_readout(z).reshape(B, len(s_idx), -1).cpu()
                ce = F.cross_entropy(logits.flatten(0, 1), target.clamp(min=0).flatten(), reduction="none").reshape(target.shape)
                b, j = torch.where(mask)
                rows.append(pd.DataFrame(dict(model=name, horizon=L, sequence=(b + lo).numpy(),
                                              t=q[j].numpy(), delay=part["delay"][:, q][b, j].numpy(),
                                              correct=(logits.argmax(-1)[b, j] == target[b, j]).numpy(),
                                              ce=ce[b, j].numpy())))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# Detectors: training, quality and selection audit
# --------------------------------------------------------------------------------------

def _auc(scores: np.ndarray, positive: np.ndarray) -> float:
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def fit_detector(kind: str, cfg: Config, directory: Path, validation: dict, device, log=print):
    """kind in {'window', 'context'}; resumable; best checkpoint by validation CE."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"detector_{kind}.pt"
    identity = dict(kind=kind, run=run_identity(cfg, device))
    with isolated_seed(cfg.detector_seed):
        model = (WindowDetector if kind == "window" else ContextDetector)(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.detector_lr, weight_decay=cfg.weight_decay)
    start, history, best, best_state = 0, [], float("inf"), None
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, f"{path} belongs to another configuration"
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"])
        start, history, best, best_state = saved["step"], saved["history"], saved["best"], saved["best_model"]
    for step in range(start, cfg.detector_steps):
        batch = sample_stack(cfg, cfg.detector_batch_size, derive_seed(101, cfg.detector_seed, step))
        model.train(); opt.zero_grad(set_to_none=True)
        loss = detector_train_loss(model, kind, batch, cfg, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
        opt.step()
        row = dict(step=step + 1, train_ce=float(loss.detach()))
        if (step + 1) % 200 == 0 or step + 1 == cfg.detector_steps:
            q = detector_quality(model, kind, validation, cfg, device)
            row.update(q)
            if q["ce"] < best:
                best, best_state = q["ce"], copy.deepcopy(model.state_dict())
            log(f"detector[{kind}] {step + 1}/{cfg.detector_steps} {q}")
            history.append(row)
            torch.save(dict(identity=identity, step=step + 1, history=history, best=best,
                            best_model=best_state, model=model.state_dict(), optimizer=opt.state_dict()), path)
        else:
            history.append(row)
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, pd.DataFrame(history)


def detector_train_loss(model, kind, batch, cfg, device):
    if kind == "window":
        r = cfg.local_window
        win = F.pad(batch["x"], (r - 1, 0), value=IGNORE).unfold(1, r, 1)[batch["target_mask"]]
        return F.cross_entropy(model(win.to(device)), batch["y"][batch["target_mask"]].to(device))
    w = windows(batch)
    logits = model(w["x_fut"].to(device))
    return pair_mean(F.cross_entropy(logits.flatten(0, 1), w["y_fut"].clamp(min=0).flatten().to(device),
                                     reduction="none").reshape(w["y_fut"].shape), w["valid"].to(device))


@torch.no_grad()
def detector_quality(model, kind, batch, cfg, device) -> dict:
    model.eval()
    if kind == "window":
        s = model.scores(batch, device).cpu()[batch["target_mask"]].numpy()
        delay = batch["delay"][batch["target_mask"]].numpy()
        outside = delay >= cfg.local_window
        r = cfg.local_window
        win = F.pad(batch["x"], (r - 1, 0), value=IGNORE).unfold(1, r, 1)[batch["target_mask"]]
        pred = model(win.to(device)).argmax(-1).cpu().numpy()
        y = batch["y"][batch["target_mask"]].numpy()
        return dict(ce=float(s.mean()), acc_inside_window=float((pred == y)[~outside].mean()),
                    acc_outside_window=float((pred == y)[outside].mean()),
                    auc_outside_window=_auc(s, outside), n=len(s))
    w = windows(batch)
    s = model.pair_scores(w, device).cpu()
    v = w["valid"]
    sv, prefix = s[v].numpy(), w["prefix"][v].numpy()
    return dict(ce=float(sv.mean()), mean_ce_future_source=float(sv[~prefix].mean()),
                mean_ce_prefix_source=float(sv[prefix].mean()), auc_prefix_source=_auc(sv, prefix), n=len(sv))


@torch.no_grad()
def selection_audit(batch: dict, cfg: Config, detectors: dict, device, seed: int = 401) -> pd.DataFrame:
    """Which targets each rule selects, per (anchor, target) pair. Metadata is read only AFTER
    selection. 'prefix' = source pushed at or before the anchor -> must be in m_t."""
    w = windows(batch)
    valid = w["valid"].to(device)
    rows = []
    g = torch.Generator().manual_seed(seed)
    eligible = w["valid"].sum(1) > cfg.K
    for variant in ("random_k", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"):
        if variant == "surprise_k" and "window" not in detectors:
            continue
        if variant == "surprise_ctx_k" and "context" not in detectors:
            continue
        pool = w["prefix"].to(device) if variant == "oracle_prefix_k" else valid
        scores = selection_scores(variant, w, batch, detectors, device)
        scores = torch.zeros(valid.shape, device=device) if scores is None else scores
        chosen = topk_mask(scores, pool, cfg.K, g).cpu()
        for group, mask in (("all anchors", chosen), ("pool > K", chosen & eligible[:, None])):
            if not mask.any():
                continue
            d = w["delay"][mask]
            rows.append(dict(rule=variant, anchors=group, pairs=int(mask.sum()),
                             memory_dependent=float(w["prefix"][mask].float().mean()),
                             long_memory_dependent=float((w["prefix"] & w["delay"].ge(cfg.long_delay))[mask].float().mean()),
                             future_source=float(w["future"][mask].float().mean()),
                             mean_delay=float(d.float().mean()), mean_offset=float((w["target_t"] - w["anchor"][:, None])[mask].float().mean())))
    # exact expectation of uniform random-K within each anchor
    n = w["valid"].sum(1).float()
    k = n.clamp(max=cfg.K)
    for group, sel in (("all anchors", n > 0), ("pool > K", eligible)):
        frac = lambda m: float(((k * m.sum(1) / n.clamp(min=1))[sel]).sum() / k[sel].sum())
        rows.append(dict(rule="random_k (exact expectation)", anchors=group, pairs=int(k[sel].sum()),
                         memory_dependent=frac(w["prefix"]),
                         long_memory_dependent=frac(w["prefix"] & w["delay"].ge(cfg.long_delay)),
                         future_source=frac(w["future"]), mean_delay=float("nan"), mean_offset=float("nan")))
    out = pd.DataFrame(rows)
    out["anchors_with_pool_le_K"] = float((~eligible & (w["valid"].sum(1) > 0)).float().mean())
    return out


# --------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------

def lr_lambda(cfg: Config, steps: int):
    warm = max(1, min(cfg.warmup_steps, steps // 10))

    def f(s):
        if s < warm:
            return (s + 1) / warm
        p = min(1.0, (s - warm) / max(1, steps - warm))
        return cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return f


def long_ce(events: pd.DataFrame, cfg: Config, model: str) -> float:
    sel = events[(events.model == model) & (events.delay >= cfg.long_delay)]
    assert len(sel), "no long pops in the monitoring set"
    return float(sel.ce.mean())


def fit(cfg: Config, variant: str, alpha: float, seed: int, out: Path, detectors: dict,
        validation: dict, device, log=print) -> dict:
    """SMT then DMT for one (variant, alpha, seed). Resumable at every eval_every steps.
    Same seed -> same initial weights, train batches, anchors and random draws for all variants."""
    assert variant in ALL_VARIANTS
    out.mkdir(parents=True, exist_ok=True)
    det_hash = {k: state_hash(v) for k, v in detectors.items()
                if (k == "window" and variant == "surprise_k") or (k == "context" and variant == "surprise_ctx_k")}
    identity = dict(run=run_identity(cfg, device), variant=variant, alpha=float(alpha), seed=int(seed),
                    detectors=det_hash, validation=split_hash(validation))
    done = out / "completed.pt"
    if done.exists():
        saved = torch.load(done, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, f"{done}: identity mismatch"
        return saved
    teacher, rnn = make_models(cfg, seed, device)
    initial_hash = state_hash(teacher, rnn)
    result = dict(identity=identity, initial_hash=initial_hash)
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp_bf16 and device.type == "cuda")
    for stage, steps in (("smt", cfg.smt_steps), ("dmt", cfg.dmt_steps)):
        is_smt = stage == "smt"
        teacher.requires_grad_(is_smt)
        if not is_smt:
            copy_readout(teacher, rnn)
            frozen = state_hash(teacher)
        params = [*teacher.parameters(), *rnn.parameters()] if is_smt else list(rnn.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.smt_lr if is_smt else cfg.dmt_lr,
                                betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg, steps))
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
            batch = sample_stack(cfg, bs, derive_seed(301 if is_smt else 302, seed, step))
            gen = torch.Generator().manual_seed(derive_seed(303, seed, step))
            opt.zero_grad(set_to_none=True)
            with isolated_seed(derive_seed(304 if is_smt else 305, seed, step)), amp:
                loss, parts = (smt_loss(teacher, rnn, batch, cfg, variant, alpha, detectors, gen, device)
                               if is_smt else dmt_loss(teacher, rnn, batch, cfg, device))
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
                    copy_readout(teacher, rnn)
                ev = evaluate(teacher, rnn, validation, cfg, device)
                score = long_ce(ev, cfg, "teacher" if is_smt else "rnn")
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
        # keep the final ('last') weights too: used as a robustness check of checkpoint selection
        result[f"{stage}_last_teacher"] = copy.deepcopy(teacher.state_dict())
        result[f"{stage}_last_rnn"] = copy.deepcopy(rnn.state_dict())
        best_saved = torch.load(best_path, map_location="cpu", weights_only=True)
        teacher.load_state_dict(best_saved["teacher"]); rnn.load_state_dict(best_saved["rnn"])
        result[f"{stage}_best_step"] = best_saved["step"]
        if is_smt:
            copy_readout(teacher, rnn)
            result["rnn_smt"] = copy.deepcopy(rnn.state_dict())
        else:
            assert state_hash(teacher) == frozen, "teacher changed during DMT"
            result["rnn_dmt"] = copy.deepcopy(rnn.state_dict())
    result["teacher"] = teacher.state_dict()
    torch.save(result, done)
    return result


def load_fitted(cfg: Config, saved: dict, device, stage: str, which: str = "best"):
    teacher, rnn = make_models(cfg, 0, device)
    if which == "best":
        teacher.load_state_dict(saved["teacher"])
        rnn.load_state_dict(saved["rnn_" + stage])
    else:  # final weights of the stage; teacher after SMT is fixed in DMT
        teacher.load_state_dict(saved["smt_last_teacher"] if stage == "smt" else saved["dmt_last_teacher"])
        rnn.load_state_dict(saved[f"{stage}_last_rnn"])
        if stage == "smt":
            copy_readout(teacher, rnn)
    return teacher.eval(), rnn.eval()


# --------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------

def paired_deltas(events: pd.DataFrame, cfg: Config, comparisons, metrics=("ce", "correct", "shared_ce", "shared_correct"),
                  group_min_delay: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Differences variant - reference on the SAME test sequences, per training seed.
    Unit of resampling: sequence (pops in one sequence are dependent). Seeds are resampled
    too, but with <5 seeds that interval is not reliable: read the per-seed table first."""
    d0 = cfg.long_delay if group_min_delay is None else group_min_delay
    data = events[events.delay >= d0]
    per_seed, rows = [], []
    rng = np.random.default_rng(derive_seed(402, 0))
    for (model, stage), sub in data.groupby(["model", "stage"]):
        if model == "teacher" and stage == "dmt":
            continue   # teacher is frozen during DMT: identical to stage 'smt'
        for reference, variant in comparisons:
            for metric in metrics:
                arrays = []
                for seed in sorted(sub.seed.unique()):
                    s = sub[sub.seed == seed].groupby(["variant", "sequence"])[metric].agg(["sum", "count"])
                    if reference not in s.index.get_level_values(0) or variant not in s.index.get_level_values(0):
                        continue
                    a, b = s.loc[reference].sort_index(), s.loc[variant].sort_index()
                    assert a.index.equals(b.index) and np.array_equal(a["count"], b["count"])
                    dif, cnt = (b["sum"] - a["sum"]).to_numpy(float), a["count"].to_numpy(float)
                    arrays.append((dif, cnt))
                    per_seed.append(dict(model=model, stage=stage, reference=reference, variant=variant,
                                         metric=metric, seed=seed, delta=dif.sum() / cnt.sum(), pops=int(cnt.sum())))
                if not arrays:
                    continue
                est = [d.sum() / c.sum() for d, c in arrays]
                nseq = len(arrays[0][1])
                boot_seq, boot_both = [], []
                for _ in range(cfg.bootstrap_repeats):
                    idx = rng.integers(0, nseq, nseq)
                    per = [d[idx].sum() / c[idx].sum() for d, c in arrays]
                    boot_seq.append(np.mean(per))
                    boot_both.append(np.mean([per[i] for i in rng.integers(0, len(per), len(per))]))
                rows.append(dict(model=model, stage=stage, reference=reference, variant=variant, metric=metric,
                                 delta_mean=float(np.mean(est)), seeds=len(est),
                                 seeds_same_sign=int(max((np.array(est) > 0).sum(), (np.array(est) < 0).sum())),
                                 seed_sd=float(np.std(est, ddof=1)) if len(est) > 1 else float("nan"),
                                 ci_sequences_low=float(np.quantile(boot_seq, .025)),
                                 ci_sequences_high=float(np.quantile(boot_seq, .975)),
                                 ci_seeds_and_sequences_low=float(np.quantile(boot_both, .025)),
                                 ci_seeds_and_sequences_high=float(np.quantile(boot_both, .975)),
                                 min_delay=d0))
    return pd.DataFrame(rows), pd.DataFrame(per_seed)


# --------------------------------------------------------------------------------------
# Orchestration (explicit phases; test exists only in the final report)
# --------------------------------------------------------------------------------------

class Experiment:
    def __init__(self, cfg: Config, root: Path | str = DEFAULT_ROOT, device=None, log=print):
        self.cfg, self.root, self.log = cfg, Path(root), log
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        sources(self.root)
        ident = run_identity(cfg, self.device)
        key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = self.root / "runs" / f"stack02_{key}"
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "identity.json"
        if p.exists():
            assert json.loads(p.read_text()) == ident
        else:
            p.write_text(json.dumps(ident, indent=2))
        self.validation = sample_stack(cfg, cfg.validation_examples, derive_seed(501, 0))
        self.tuning = sample_stack(cfg, cfg.tuning_examples, derive_seed(502, 0))
        assert not set(fingerprints(self.validation)) & set(fingerprints(self.tuning))
        self.detectors: dict = {}

    # -- phase 0: data + detectors + selection audit (no SMT training)
    def audit(self, kinds=("window", "context")) -> dict:
        cfg = self.cfg
        audit = pd.DataFrame([dict(split=n, **data_audit(b, cfg)) for n, b in
                              (("validation", self.validation), ("tuning", self.tuning))])
        audit.to_csv(self.dir / "data_audit.csv", index=False)
        assert audit.long_pops.min() >= 100, "too few long pops for monitoring; enlarge validation/tuning"
        quality = {}
        for kind in kinds:
            self.detectors[kind], _ = fit_detector(kind, cfg, self.dir, self.validation, self.device, self.log)
            quality[kind] = detector_quality(self.detectors[kind], kind, self.tuning, cfg, self.device)
        sel = selection_audit(self.tuning, cfg, self.detectors, self.device)
        sel.to_csv(self.dir / "selection_audit.csv", index=False)
        (self.dir / "detector_quality.json").write_text(json.dumps(quality, indent=2))
        return dict(data=audit, detector_quality=quality, selection=sel)

    def _need_detectors(self, variants):
        need = {"surprise_k": "window", "surprise_ctx_k": "context"}
        kinds = [need[v] for v in variants if v in need and need[v] not in self.detectors]
        for kind in kinds:
            self.detectors[kind], _ = fit_detector(kind, self.cfg, self.dir, self.validation, self.device, self.log)

    # -- phase 1: baseline on the tuning seed + gate (tuning split)
    def baseline(self) -> dict:
        cfg = self.cfg
        saved = fit(cfg, "baseline", 0.0, cfg.tuning_seed, self.dir / "tuning" / "baseline",
                    {}, self.validation, self.device, self.log)
        frames = []
        for stage in ("smt", "dmt"):
            t, r = load_fitted(cfg, saved, self.device, stage)
            frames.append(evaluate(t, r, self.tuning, cfg, self.device, seed=cfg.tuning_seed,
                                   variant="baseline", stage=stage, split="tuning"))
        ev = pd.concat(frames, ignore_index=True)
        chance = 1 / cfg.n_elements
        lt = ev[(ev.model == "teacher") & (ev.stage == "smt") & (ev.delay >= cfg.long_delay)].correct.mean()
        lr_ = ev[(ev.model == "rnn") & (ev.stage == "dmt") & (ev.delay >= cfg.long_delay)].correct.mean()
        gate = dict(teacher_long_accuracy=float(lt), rnn_dmt_long_accuracy=float(lr_), chance=chance,
                    required_teacher=chance + cfg.gate_teacher_margin, required_rnn=chance + cfg.gate_rnn_margin,
                    smt_best_step=saved["smt_best_step"], dmt_best_step=saved["dmt_best_step"])
        gate["passed"] = bool(lt >= gate["required_teacher"] and lr_ >= gate["required_rnn"])
        (self.dir / "baseline_gate.json").write_text(json.dumps(gate, indent=2))
        ev.to_csv(self.dir / "baseline_tuning_events.csv", index=False)
        t, r = load_fitted(cfg, saved, self.device, "dmt")
        probe = retention_probe(t, r, self.tuning, cfg, self.device)
        probe.to_csv(self.dir / "baseline_retention_probe.csv", index=False)
        return dict(gate=gate, summary=summarize(ev, cfg), events=ev, retention=probe)

    # -- phase 2: alpha search, identical grid and budget for every non-baseline variant
    def tune(self, variants=VARIANTS_AGREED, require_gate: bool = True) -> dict:
        cfg = self.cfg
        gate_file = self.dir / "baseline_gate.json"
        if require_gate:
            assert gate_file.exists() and json.loads(gate_file.read_text())["passed"], \
                "baseline gate not passed: inspect validation curves and change the config in a NEW run"
        self._need_detectors(variants)
        rows = []
        for variant in variants:
            if variant == "baseline":
                continue
            for alpha in cfg.alpha_grid:
                saved = fit(cfg, variant, alpha, cfg.tuning_seed, self.dir / "tuning" / f"{variant}_a{alpha}",
                            self.detectors, self.validation, self.device, self.log)
                t, r = load_fitted(cfg, saved, self.device, "dmt")
                ev = evaluate(t, r, self.tuning, cfg, self.device)
                rows.append(dict(variant=variant, alpha=alpha, tuning_rnn_long_ce=long_ce(ev, cfg, "rnn"),
                                 tuning_teacher_long_ce=long_ce(ev, cfg, "teacher")))
        table = pd.DataFrame(rows)
        table.to_csv(self.dir / "alpha_tuning.csv", index=False)
        alphas = {"baseline": 0.0}
        for v, g in table.groupby("variant"):
            alphas[v] = float(g.sort_values(["tuning_rnn_long_ce", "alpha"]).iloc[0].alpha)
        lock = self.dir / "locked_alphas.json"
        if lock.exists():
            old = json.loads(lock.read_text())["alphas"]
            assert all(old.get(k) == v for k, v in alphas.items() if k in old), "alpha lock would change"
            alphas = {**old, **alphas}
        lock.write_text(json.dumps(dict(alphas=alphas, criterion="tuning split, RNN after DMT, CE on pops "
                                        f"with delay>={cfg.long_delay}; checkpoints selected on validation"), indent=2))
        return dict(table=table, alphas=alphas)

    # -- phase 3: final fits (no test yet)
    def final_fits(self, variants=VARIANTS_AGREED) -> dict:
        cfg = self.cfg
        alphas = json.loads((self.dir / "locked_alphas.json").read_text())["alphas"]
        missing = [v for v in variants if v not in alphas]
        assert not missing, f"alpha not tuned for {missing}"
        self._need_detectors(variants)
        fitted = {}
        for seed in cfg.seeds:
            for v in variants:
                fitted[seed, v] = fit(cfg, v, alphas[v], seed, self.dir / "final" / f"seed{seed}" / v,
                                      self.detectors, self.validation, self.device, self.log)
            assert len({fitted[seed, v]["initial_hash"] for v in variants}) == 1, "initial weights differ"
        return fitted

    # -- phase 4: the only place where the test split is generated
    def final_report(self, variants=VARIANTS_AGREED, confirm_test: bool = False) -> dict:
        assert confirm_test, "pass confirm_test=True once the protocol is frozen; never tune after this"
        cfg = self.cfg
        fitted = self.final_fits(variants)
        test = sample_stack(cfg, cfg.test_examples, derive_seed(503, 0))
        assert not set(fingerprints(test)) & (set(fingerprints(self.validation)) | set(fingerprints(self.tuning)))
        frames = []
        for (seed, v), saved in fitted.items():
            for which in ("best", "last"):
                for stage in ("smt", "dmt"):
                    t, r = load_fitted(cfg, saved, self.device, stage, which)
                    frames.append(evaluate(t, r, test, cfg, self.device, seed=seed, variant=v, stage=stage,
                                           split="test", checkpoint=which))
        events = pd.concat(frames, ignore_index=True)
        comps = [("random_k", "surprise_k"), ("uniform", "surprise_k"), ("baseline", "surprise_k"),
                 ("baseline", "uniform"), ("baseline", "random_k")]
        comps += [(ref, v) for v in variants if v in VARIANTS_OPTIONAL for ref in ("random_k", "uniform", "baseline")]
        comps = [(a, b) for a, b in comps if a in variants and b in variants]
        best = events[events.checkpoint == "best"]
        intervals, per_seed = paired_deltas(best, cfg, comps)
        summary = summarize(events, cfg, keys=("checkpoint", "seed", "variant", "stage", "model"))
        for name, df in (("test_events", events), ("test_summary", summary), ("test_intervals", intervals),
                         ("test_per_seed", per_seed)):
            df.to_csv(self.dir / f"{name}.csv", index=False)
        report = dict(variants=list(variants), seeds=list(cfg.seeds), test_hash=split_hash(test),
                      validation_hash=split_hash(self.validation), tuning_hash=split_hash(self.tuning),
                      primary="RNN after DMT, CE on pops with delay>=long_delay, surprise_k vs random_k and uniform",
                      caveat="Few training seeds; the per-seed table is the primary evidence.")
        (self.dir / "test_report.json").write_text(json.dumps(report, indent=2))
        return dict(events=events, summary=summary, intervals=intervals, per_seed=per_seed, report=report)


# --------------------------------------------------------------------------------------
# Cost measurement (a few optimizer steps on throw-away models)
# --------------------------------------------------------------------------------------

def benchmark(cfg: Config, device, steps: int = 10, variant: str = "surprise_k") -> dict:
    teacher, rnn = make_models(cfg, 0, device)
    dets = {"window": WindowDetector(cfg).to(device).eval().requires_grad_(False),
            "context": ContextDetector(cfg).to(device).eval().requires_grad_(False)}
    params = [*teacher.parameters(), *rnn.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4)
    out = {}
    for stage in ("smt", "dmt"):
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        times = []
        for s in range(steps + 2):
            batch = sample_stack(cfg, cfg.batch_size if stage == "smt" else cfg.dmt_batch_size, 10_000 + s)
            tic = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss, _ = (smt_loss(teacher, rnn, batch, cfg, variant, 0.3, dets, torch.Generator().manual_seed(s), device)
                       if stage == "smt" else dmt_loss(teacher, rnn, batch, cfg, device))
            loss.backward(); opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if s >= 2:
                times.append(time.perf_counter() - tic)
        out[f"{stage}_s_per_step"] = float(np.median(times))
        if device.type == "cuda":
            out[f"{stage}_peak_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    val = sample_stack(cfg, cfg.validation_examples, 7)
    tic = time.perf_counter()
    evaluate(teacher, rnn, val, cfg, device)
    out["eval_s"] = time.perf_counter() - tic
    return out


def plan(cfg: Config, bench: dict | None = None, variants=VARIANTS_AGREED) -> pd.DataFrame:
    n_nonbase = len([v for v in variants if v != "baseline"])
    fits = dict(baseline_gate=1, alpha_tuning=n_nonbase * len(cfg.alpha_grid), final=len(cfg.seeds) * len(variants))
    rows = []
    for phase, n in fits.items():
        r = dict(phase=phase, fits=n, smt_steps=n * cfg.smt_steps, dmt_steps=n * cfg.dmt_steps)
        if bench:
            evals = n * (cfg.smt_steps + cfg.dmt_steps) / cfg.eval_every
            r["hours_est"] = (n * (cfg.smt_steps * bench["smt_s_per_step"] + cfg.dmt_steps * bench["dmt_s_per_step"])
                              + evals * bench["eval_s"]) / 3600
        rows.append(r)
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description="SMT x MemUP Stack experiment (phases run only on request)")
    p.add_argument("--phase", required=True, choices=["benchmark", "audit", "baseline", "tune", "final", "report"])
    p.add_argument("--variants", default=",".join(VARIANTS_AGREED))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-memory-fraction", type=float, default=None)
    p.add_argument("--confirm-test", action="store_true")
    p.add_argument("--override", default="{}", help='JSON dict of Config overrides, e.g. {"smt_steps": 500}')
    a = p.parse_args(argv)
    device = setup_runtime(a.threads, a.gpu_memory_fraction)
    cfg = replace(Config(), **{k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.override).items()})
    variants = tuple(v for v in a.variants.split(",") if v)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.phase == "benchmark":
        sources()
        b = benchmark(cfg, device)
        print(json.dumps(b, indent=2)); print(plan(cfg, b, variants).to_string()); return
    ex = Experiment(cfg, device=device, log=log)
    log(f"run directory: {ex.dir}")
    if a.phase == "audit":
        r = ex.audit(); print(r["data"].to_string()); print(json.dumps(r["detector_quality"], indent=2)); print(r["selection"].to_string())
    elif a.phase == "baseline":
        r = ex.baseline(); print(json.dumps(r["gate"], indent=2))
    elif a.phase == "tune":
        r = ex.tune(variants); print(r["table"].to_string()); print(r["alphas"])
    elif a.phase == "final":
        ex.final_fits(variants)
    elif a.phase == "report":
        r = ex.final_report(variants, confirm_test=a.confirm_test); print(r["intervals"].to_string())


if __name__ == "__main__":
    main()
