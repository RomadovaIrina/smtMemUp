"""SMT x MemUP on MQAR (Zoology multiquery_ar, no noise) - code for 03_smt_memup_mqar.ipynb.

Importing never trains anything. Long runs start only from the notebook or the CLI.

Reused unchanged from smt_memup_stack.py (that file is NOT modified; its hash identifies the
Stack runs): seeds, author model loading, all-anchor windows, top-K selection, loss normalisation,
detectors, statistics. MQAR-specific here:
  * data: author Zoology generator (pinned commit), random_non_queries=False, plus annotation of
    where each queried value was written (source position) - metadata for audit/oracle only;
  * an SMT forward for long sequences: optional Monte Carlo subset of anchors for BOTH decoder and
    dynamics terms, decoder logits only at answer positions (vocab 8192 otherwise does not fit),
    encoder windows without the leading padding (it is masked anyway);
  * DMT with an optional truncated rollout started from the teacher state (as in the author DMT);
  * chunked evaluation and the H2 retention probe at answer positions;
  * run identity = hash of this file + hash of smt_memup_stack.py + Zoology revision.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import platform
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from einops import rearrange

import smt_memup_stack as S

IGNORE = S.IGNORE
MODULE_PATH = Path(__file__).resolve()
DEFAULT_ROOT = MODULE_PATH.parent
ZOOLOGY_REVISION = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
VARIANTS_AGREED, VARIANTS_OPTIONAL, ALL_VARIANTS = S.VARIANTS_AGREED, S.VARIANTS_OPTIONAL, S.ALL_VARIANTS

# --------------------------------------------------------------------------------------
# Config and candidate data settings
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class MQARConfig:
    # data: Zoology multiquery_ar, random_non_queries=False (no noise), num_passes=1
    vocab: int = 256
    seq_len: int = 64
    num_kv_pairs: int = 4
    power_a: float = 0.01
    long_delay: int = 8              # primary group: answers whose value was written >= long_delay tokens ago
    # model (author classes; sizes as in the Stack experiment)
    d_model: int = 64
    mem_tokens: int = 8
    encoder_depth: int = 3
    decoder_depth: int = 2
    rnn_depth: int = 2
    n_heads: int = 4
    # SMT
    batch_size: int = 32
    anchors_per_sequence: int = 0    # 0 = all anchors (paper); >0 = random subset per sequence and step
    smt_steps: int = 3000
    smt_lr: float = 3e-4
    coef_dec: float = 1.0
    coef_dyn: float = 0.1
    coef_unif: float = 0.001
    # DMT
    dmt_batch_size: int = 32
    dmt_steps: int = 1500
    dmt_lr: float = 1e-4
    dmt_coef_dyn: float = 10.0
    dmt_rollout: int = 0             # 0 = whole sequence from the zero state; >0 = this many steps from the teacher state
    # optimisation
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    warmup_steps: int = 200
    min_lr_ratio: float = 0.1
    # auxiliary loss
    K: int = 4
    alpha_grid: tuple = (0.3, 1.0, 3.0)   # Stack: every variant chose 1.0, the edge of (0.1, 0.3, 1.0); fixed before any MQAR fit
    local_window: int = 8
    # detectors: same GRU family as Stack, larger and longer. With the Stack setting (16/64, 2000 steps) the MQAR window
    # detector ranked pair-outside answers with AUC 0.79 only; 64/128 reaches ~0.94 by 4000 steps (CPU trial on validation).
    # A local GRU stays at the value-set level (~18% when the pair is inside the window): it does not learn binding,
    # but the RANKING (pair outside window -> high surprise), which is what top-K uses, is close to the ideal one.
    detector_steps: int = 20000
    detector_batch_size: int = 32
    detector_lr: float = 1e-3
    detector_hidden: int = 128
    detector_embed: int = 64
    detector_seed: int = 911
    # protocol
    eval_every: int = 250
    eval_batch_size: int = 64
    validation_examples: int = 512
    tuning_examples: int = 512
    test_examples: int = 2048
    tuning_seed: int = 107
    seeds: tuple = (7, 17, 27)
    bootstrap_repeats: int = 2000
    gate_teacher_margin: float = 0.10   # over the structured chance 1/num_kv_pairs
    gate_rnn_margin: float = 0.05
    encode_chunk: int = 256          # windows per encoder/decoder call; smaller = less padding computed
    amp_bf16: bool = False

    @property
    def vocab_size(self) -> int:
        return self.vocab

    @property
    def mem_size(self) -> int:
        return self.d_model * self.mem_tokens

    @property
    def chance_structured(self) -> float:
        """Guessing uniformly among the values written in this sequence."""
        return 1 / self.num_kv_pairs

    @property
    def chance_vocab(self) -> float:
        """Guessing uniformly among all value tokens."""
        return 1 / (self.vocab - self.vocab // 2)


# Zoology references: basic example (examples/basic.py) and the noise-free sweep
# (mqar_example_configs/mqar_configs_random_false.py, vocab 8192, train mix 64/4 ... 256/64).
# K = num_kv_pairs / 4 (at least 1): with K >= num_kv_pairs every rule selects all future answers and
# surprise-K, random-K and uniform coincide.
# local_window r: the smallest r on WINDOW_GRID at which the queried pair lies inside the last r tokens for at
# least half of the answers (a data property, fixed before any training; see choose_window). long_delay = r, as
# in the Stack experiment where both were 8. All three are our choices, not advisor requirements.
WINDOW_GRID = (4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
CANDIDATES = {
    "zoology_basic": dict(vocab=256, seq_len=64, num_kv_pairs=4, K=1, local_window=16, long_delay=16),
    "pilot01": dict(vocab=128, seq_len=64, num_kv_pairs=8, K=2, local_window=24, long_delay=24),
    "zoology_64": dict(vocab=8192, seq_len=64, num_kv_pairs=4, K=1, local_window=16, long_delay=16),
    "zoology_128": dict(vocab=8192, seq_len=128, num_kv_pairs=8, K=2, local_window=32, long_delay=32),
    "zoology_256": dict(vocab=8192, seq_len=256, num_kv_pairs=16, K=4, local_window=64, long_delay=64),
    "zoology_256_kv64": dict(vocab=8192, seq_len=256, num_kv_pairs=64, K=16, local_window=128, long_delay=128),
}


def candidate(name: str, **override) -> MQARConfig:
    return replace(MQARConfig(), **{**CANDIDATES[name], **override})


def code_sha256() -> str:
    return hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()


def run_identity(cfg: MQARConfig, device: torch.device) -> dict:
    src = S.sources()
    return dict(task="mqar", config=json.loads(json.dumps(asdict(cfg))), code_sha256=code_sha256(),
                stack_code_sha256=S.code_sha256(), zoology=ZOOLOGY_REVISION, repositories=src.revisions,
                torch=str(torch.__version__), x_transformers=importlib.metadata.version("x-transformers"),
                device=device.type, python=platform.python_version())


# --------------------------------------------------------------------------------------
# Data: author Zoology generator + annotation (ours)
# --------------------------------------------------------------------------------------

_ZOO = None


def zoology_generator(root: Path | str = DEFAULT_ROOT):
    global _ZOO
    if _ZOO is None:
        zroot = Path(root) / "upstream" / "zoology"
        rev = S.repository_revision(zroot)
        if rev != ZOOLOGY_REVISION:
            raise RuntimeError(f"Zoology revision changed: {rev}")
        if str(zroot) not in sys.path:
            sys.path.insert(0, str(zroot))
        from zoology.data.multiquery_ar import multiquery_ar
        _ZOO = multiquery_ar
    return _ZOO


def annotate_mqar(x: torch.Tensor, y: torch.Tensor, n_pairs: int) -> dict:
    """Prefix = n_pairs (key, value) pairs at positions (2i, 2i+1); every answer y_t sits at a query
    key x_t. source_pos = position of the queried value, key_pos = source_pos - 1."""
    B, T = x.shape
    m = y.ne(IGNORE)
    keys, vals = x[:, 0:2 * n_pairs:2], x[:, 1:2 * n_pairs:2]
    match = x[:, :, None].eq(keys[:, None, :]) & m[:, :, None]
    if not (match.sum(-1)[m] == 1).all():
        raise AssertionError("each query key must match exactly one written key")
    idx = match.float().argmax(-1)
    if not torch.equal(vals.gather(1, idx)[m], y[m]):
        raise AssertionError("answer differs from the value written for the queried key")
    source = torch.where(m, 2 * idx + 1, torch.full_like(x, IGNORE))
    pos = torch.arange(T).expand(B, T)
    return dict(x=x, y=y, target_mask=m, source_pos=source,
                delay=torch.where(m, pos - source, torch.full_like(x, IGNORE)),
                key_pos=torch.where(m, source - 1, torch.full_like(x, IGNORE)),
                depth=torch.zeros_like(x))


def sample_mqar(cfg: MQARConfig, batch_size: int, seed: int) -> dict:
    gen = zoology_generator()
    with S.isolated_seed(seed):
        seg = gen(vocab_size=cfg.vocab, num_examples=batch_size, input_seq_len=cfg.seq_len, seed=int(seed) % 2**32,
                  power_a=cfg.power_a, num_kv_pairs=cfg.num_kv_pairs, num_passes=1, random_non_queries=False)
    x = seg.inputs.long().clone()
    y = seg.labels.long().clone()
    y[y == -100] = IGNORE                      # only the ignore-index convention changes
    batch = annotate_mqar(x, y, cfg.num_kv_pairs)
    check_batch(batch, cfg)
    return batch


def shuffle_query_order(batch: dict, cfg: MQARConfig, seed: int) -> dict:
    """Binding control (evaluation only): the same written pairs and query positions, but the keys are assigned
    to the query positions by a uniformly random permutation (labels follow the keys). The generator places the
    keys of earlier pairs at earlier queries more often (power-law gaps drawn without replacement), so a model
    could exceed 1/N among the written values by query order alone; here the order carries no information."""
    x, m = batch["x"].clone(), batch["target_mask"]
    N = cfg.num_kv_pairs
    keys, vals = x[:, 0:2 * N:2], x[:, 1:2 * N:2]
    g = torch.Generator().manual_seed(int(seed))
    y = torch.full_like(x, IGNORE)
    for b in range(len(x)):
        q = torch.where(m[b])[0]
        perm = torch.randperm(N, generator=g)
        x[b, q], y[b, q] = keys[b, perm], vals[b, perm]
    out = annotate_mqar(x, y, N)
    check_batch(out, cfg)
    return out


def check_batch(batch: dict, cfg: MQARConfig) -> bool:
    x, y, m = batch["x"], batch["y"], batch["target_mask"]
    N, T = cfg.num_kv_pairs, cfg.seq_len
    assert x.shape[1] == T and ((x >= 0) & (x < cfg.vocab)).all()
    assert (m.sum(1) == N).all(), "one query per written pair"
    keys = x[:, 0:2 * N:2]
    assert ((keys >= 1) & (keys < cfg.vocab // 2)).all() and ((x[:, 1:2 * N:2] >= cfg.vocab // 2)).all()
    assert all(len(set(r.tolist())) == N for r in keys), "keys are unique within a sequence"
    assert not m[:, :2 * N].any(), "queries come after the written pairs"
    filler = ~m.clone(); filler[:, :2 * N] = False
    assert (x[filler] == 0).all(), "no noise: non-query positions after the prefix are the filler 0"
    assert (batch["source_pos"][m] < torch.arange(T).expand_as(x)[m]).all()
    return True


def data_audit(batch: dict, cfg: MQARConfig) -> dict:
    m = batch["target_mask"]
    d = batch["delay"][m].float()
    T, N, r = cfg.seq_len, cfg.num_kv_pairs, cfg.local_window
    pos = torch.arange(T).expand_as(batch["x"])
    visible = (batch["key_pos"] >= pos - r + 1)[m]
    w = S.windows(S.slice_batch(batch, 0, min(64, len(batch["x"]))))
    pool = w["valid"].sum(1).float()
    has = pool > 0
    return dict(sequences=len(batch["x"]), answers=int(m.sum()), answers_per_sequence=N,
                delay_median=float(d.median()), delay_p95=float(torch.quantile(d, .95)),
                long_fraction=float((d >= cfg.long_delay).float().mean()),
                pair_inside_last_r_tokens=float(visible.float().mean()),
                anchors_inside_written_prefix=2 * N / T,
                anchors_with_future_answers=float(has.float().mean()),
                mean_pool=float(pool[has].mean()), anchors_pool_gt_K=float((pool[has] > cfg.K).float().mean()),
                chance_structured=cfg.chance_structured, chance_vocab=cfg.chance_vocab,
                unique_sequences=len(set(S.fingerprints(batch))))


def pair_visible_fraction(batch: dict, r: int) -> float:
    """Share of answers whose (key, value) pair lies inside the last r tokens before the query."""
    m = batch["target_mask"]
    pos = torch.arange(batch["x"].shape[1]).expand_as(batch["x"])
    return float((batch["key_pos"] >= pos - r + 1)[m].float().mean())


def choose_window(batch: dict, grid=WINDOW_GRID) -> int:
    """Smallest r on the grid with the pair visible for at least half of the answers (data property only)."""
    for r in grid:
        if r <= batch["x"].shape[1] and pair_visible_fraction(batch, r) >= 0.5:
            return r
    return grid[-1]


def window_sweep(cfg: "MQARConfig", batch: dict, grid=WINDOW_GRID) -> pd.DataFrame:
    """For each r: share of answers visible to an r-token window, and the share of selected answers that are
    long (delay >= r) AND need memory, for random-K and for the idealised r-token surprise-K."""
    rows = []
    sub = S.slice_batch(batch, 0, min(64, len(batch["x"])))
    for r in grid:
        if r > cfg.seq_len:
            continue
        c = replace(cfg, local_window=r, long_delay=r)
        sel = S.selection_audit(sub, c, {"window": IdealWindow(c), "context": IdealContext(c)}, torch.device("cpu"))
        a = sel[sel.anchors == "all anchors"].set_index("rule")
        rows.append(dict(r=r, pair_visible=pair_visible_fraction(batch, r),
                         long_share_random=a.loc["random_k (exact expectation)", "long_memory_dependent"],
                         long_share_window=a.loc["surprise_k", "long_memory_dependent"]))
    t = pd.DataFrame(rows)
    t["lift"] = t.long_share_window / t.long_share_random
    return t


# --------------------------------------------------------------------------------------
# Encoder / decoder helpers for long sequences and large vocabularies
# --------------------------------------------------------------------------------------

def encode_anchors(teacher, x: torch.Tensor, b_idx: torch.Tensor, t_idx: torch.Tensor, chunk: int, device) -> torch.Tensor:
    """m_{b,t} = E(x_{b, 0..t}) for each requested (b, t), returned in request order. Windows are
    processed in chunks sorted by t; a chunk keeps only its last (max t + 1) context columns -
    the dropped columns are left padding, which the encoder masks."""
    n = len(t_idx)
    order = torch.argsort(t_idx, stable=True)
    parts = []
    for part in order.split(chunk):
        tt, bb = t_idx[part], b_idx[part]
        L = int(tt.max()) + 1
        pos = tt[:, None] + torch.arange(-L + 1, 1)
        ctx = torch.where(pos >= 0, x[bb[:, None], pos.clamp(min=0)], torch.full_like(pos, IGNORE))
        parts.append(teacher.encode(ctx.to(device)))
    enc = torch.cat(parts)
    inv = torch.empty_like(order)
    inv[order] = torch.arange(n)
    return enc[inv.to(enc.device)]


def all_states(teacher, x: torch.Tensor, cfg: MQARConfig, device) -> torch.Tensor:
    B, T = x.shape
    return encode_anchors(teacher, x, torch.arange(B).repeat_interleave(T), torch.arange(T).repeat(B),
                          cfg.encode_chunk, device).reshape(B, T, -1)


def decoder_hidden(teacher, z: torch.Tensor, x_fut: torch.Tensor) -> torch.Tensor:
    """Hidden states of Teacher.decode before to_logits; output j <-> position mem_tokens - 1 + j."""
    K = teacher.cfg.mem_tokens
    zz = rearrange(z, "N (K d) -> N K d", K=K)
    return teacher.decoder(torch.cat([zz, teacher.embed(x_fut.clamp(min=0))], dim=1))


# --------------------------------------------------------------------------------------
# Losses
# --------------------------------------------------------------------------------------

def smt_loss(teacher, rnn, batch: dict, cfg: MQARConfig, variant: str, alpha: float, detectors: dict,
             generator: torch.Generator, device):
    """Author SMT loss + coef_dec * alpha * L_selected, as in smt_memup_stack.smt_loss.
    With anchors_per_sequence = 0 it is the same objective (all anchors, pair-averaged decoder CE,
    joint dynamics with the first transition, uniformity over memory tokens). With a subset of
    anchors A, the decoder term uses those anchors and the dynamics term the transitions into them
    (both are Monte Carlo estimates of the all-anchor terms)."""
    m_model = S.sources().model
    x = batch["x"]
    B, T = x.shape
    K = teacher.cfg.mem_tokens
    A = cfg.anchors_per_sequence
    if 0 < A < T:
        picks = torch.rand(B, T, generator=generator).argsort(1)[:, :A].sort(1).values
        b_a, t_a = torch.arange(B).repeat_interleave(A), picks.flatten()
    else:
        b_a, t_a = torch.arange(B).repeat_interleave(T), torch.arange(T).repeat(B)
    rows = b_a * T + t_a
    need = torch.unique(torch.cat([rows, rows[t_a > 0] - 1]))           # sorted, row-major
    mem_need = encode_anchors(teacher, x, need // T, need % T, cfg.encode_chunk, device)
    m_rows = mem_need[torch.searchsorted(need, rows).to(device)]

    w_all = S.windows(batch)
    w = {k: v[rows] for k, v in w_all.items()}
    r_idx = torch.where(w["valid"].any(1))[0]
    audit = {}
    if len(r_idx):
        last = (w["valid"][r_idx] * (torch.arange(T) + 1)).max(1).values      # last answer offset + 1 per row
        tmax = int(last.max())
        wd = {k: (v[r_idx][:, :tmax] if v.ndim == 2 else v[r_idx]) for k, v in w.items()}
        valid = wd["valid"].to(device)
        ce = torch.zeros(valid.shape, device=device)
        # decoder rows in chunks sorted by their last answer; a chunk drops inputs after its last answer
        # (the causal decoder never uses them for earlier outputs)
        for part in torch.argsort(last, stable=True).split(cfg.encode_chunk):
            L = int(last[part].max())
            h = decoder_hidden(teacher, m_rows[r_idx[part].to(device)], wd["x_fut"][part, :L].to(device))
            vr, vc = torch.where(wd["valid"][part, :L])
            logits = teacher.to_logits(h[vr.to(device), K - 1 + vc.to(device)])
            ce_v = F.cross_entropy(logits.float(), wd["y_fut"][part[vr], vc].to(device), reduction="none")
            ce = ce.index_put((part[vr].to(device), vc.to(device)), ce_v)
        dec = S.pair_mean(ce, valid)
        if variant == "baseline":
            aux = dec * 0
        elif variant == "uniform":
            aux = dec
        else:
            if variant == "random_k":
                pool, scores = valid, torch.zeros_like(ce)
            elif variant == "oracle_prefix_k":            # ORACLE: generator metadata restricts the pool
                pool, scores = wd["prefix"].to(device), torch.zeros_like(ce)
            elif variant in ("surprise_k", "surprise_ctx_k"):
                pool = valid
                with torch.no_grad():
                    scores = S.selection_scores(variant, wd, batch, detectors, device)
            else:
                raise ValueError(variant)
            chosen = S.topk_mask(scores, pool, cfg.K, generator)
            aux = S.weighted_selected_loss(ce, pool, chosen)
            c = chosen.cpu()
            if c.any():
                audit = dict(sel_prefix=float(wd["prefix"][c].float().mean()),
                             sel_long_prefix=float((wd["prefix"] & wd["delay"].ge(cfg.long_delay))[c].float().mean()),
                             sel_future=float(wd["future"][c].float().mean()))
    else:
        dec = aux = m_rows.sum() * 0
    prev_idx = torch.searchsorted(need, (rows - 1).clamp(min=0)).to(device)
    m_prev = torch.where((t_a > 0).to(device)[:, None], mem_need[prev_idx], torch.zeros_like(m_rows))
    pred = rnn.rnn_update(m_prev, x[b_a, t_a][:, None].to(device))
    dyn = m_model.calc_mse_loss(pred, m_rows).mean()                      # detach_rnn=False (author)
    unif = m_model.calc_uniformity_loss(mem_need.reshape(len(need) * cfg.mem_tokens, cfg.d_model))
    loss = cfg.coef_dec * (dec + alpha * aux) + cfg.coef_dyn * dyn + cfg.coef_unif * unif
    return loss, dict(loss=loss, decoder=dec, auxiliary=aux, dynamics=dyn, uniformity=unif, **audit)


def dmt_loss(teacher, rnn, batch: dict, cfg: MQARConfig, device, generator: torch.Generator | None = None):
    """One-step DMT (teacher frozen, detach between RNN steps, readout CE on detached states).
    dmt_rollout = 0: whole sequence from the zero state (as in the Stack experiment).
    dmt_rollout = L > 0: L steps from the teacher state at a random start (author-style DMT)."""
    x = batch["x"]
    B, T = x.shape
    L = T if cfg.dmt_rollout <= 0 else min(cfg.dmt_rollout, T)
    t0 = 0 if L == T else int(torch.randint(0, T - L + 1, (1,), generator=generator))
    with torch.no_grad():
        target = encode_anchors(teacher, x, torch.arange(B).repeat_interleave(L), (t0 + torch.arange(L)).repeat(B),
                                cfg.encode_chunk, device).reshape(B, L, -1)
        init = None if t0 == 0 else encode_anchors(teacher, x, torch.arange(B), torch.full((B,), t0 - 1),
                                                   cfg.encode_chunk, device)
    states = S.rnn_rollout(rnn, x[:, t0:t0 + L].to(device), init=init, detach=True)
    dyn = (states - target).square().mean()
    y = batch["y"][:, t0:t0 + L]
    m = y.ne(IGNORE)
    if m.any():
        logits = rnn.rnn_readout(states.detach()[m.to(device)])
        readout = F.cross_entropy(logits.float(), y[m].to(device))
    else:
        readout = dyn * 0
    loss = cfg.dmt_coef_dyn * dyn + readout
    return loss, dict(loss=loss, dynamics=dyn, readout=readout)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(teacher, rnn, batch: dict, cfg: MQARConfig, device, **labels) -> pd.DataFrame:
    """One row per (answer, model). Own head and frozen teacher head ('shared_*') on the same state.
    pred_in_written: the predicted token is one of the values written in this sequence ('knows the value set').
    cand_*: the same logits restricted to the N values written in this sequence (candidates come from the
    inputs only). Chance among candidates is 1/N; clearly above it = the key selects its value (binding). The
    query order leaks a little (see shuffle_query_order), so the gate reads the shuffled-order split.
    query_rank: 0 for the first query of the sequence, ...; pair_slot: which written pair is asked."""
    teacher.eval(); rnn.eval()
    N = cfg.num_kv_pairs
    frames = []
    for lo in range(0, len(batch["x"]), cfg.eval_batch_size):
        part = S.slice_batch(batch, lo, lo + cfg.eval_batch_size)
        x = part["x"]
        tm = all_states(teacher, x, cfg, device)
        rm = S.rnn_rollout(rnn, x.to(device))
        b, t = torch.where(part["target_mask"])
        y = part["y"][b, t]
        written = x[:, 1:2 * N:2][b]
        slot = (part["source_pos"][b, t] - 1) // 2
        rank = part["target_mask"].long().cumsum(1)[b, t] - 1
        cand_y = (written == y[:, None]).float().argmax(1)          # index of the answer among the candidates
        for name, mem, head in (("teacher", tm, teacher.teacher_readout), ("rnn", rm, rnn.rnn_readout)):
            z = mem[b.to(device), t.to(device)]
            own, shared = head(z).float().cpu(), teacher.teacher_readout(z).float().cpu()
            pred, spred = own.argmax(-1), shared.argmax(-1)
            cand, scand = own.gather(1, written), shared.gather(1, written)
            frames.append(pd.DataFrame(dict(
                **labels, model=name, sequence=(b + lo).numpy(), t=t.numpy(),
                delay=part["delay"][b, t].numpy(), target=y.numpy(), correct=(pred == y).numpy(),
                ce=F.cross_entropy(own, y, reduction="none").numpy(),
                shared_correct=(spred == y).numpy(), shared_ce=F.cross_entropy(shared, y, reduction="none").numpy(),
                pred_in_written=(pred[:, None] == written).any(1).numpy(),
                cand_correct=(cand.argmax(1) == cand_y).numpy(), cand_ce=F.cross_entropy(cand, cand_y, reduction="none").numpy(),
                shared_cand_correct=(scand.argmax(1) == cand_y).numpy(),
                shared_cand_ce=F.cross_entropy(scand, cand_y, reduction="none").numpy(),
                query_rank=rank.numpy(), pair_slot=slot.numpy(),
                memory_mse=(mem - tm)[b.to(device), t.to(device)].square().mean(-1).cpu().numpy())))
    return pd.concat(frames, ignore_index=True)


def summarize(events: pd.DataFrame, cfg: MQARConfig, keys=("seed", "variant", "stage", "split", "model")) -> pd.DataFrame:
    out = S.summarize(events, cfg, keys)
    keys = [k for k in keys if k in events.columns]
    groups = {"all": events, f"delay>={cfg.long_delay}": events[events.delay >= cfg.long_delay],
              "delay>=16": events[events.delay >= 16], f"delay<{cfg.long_delay}": events[events.delay < cfg.long_delay]}
    extra = []
    for name, data in groups.items():
        t = data.groupby(keys, observed=True).agg(
            pred_in_written=("pred_in_written", "mean"), cand_accuracy=("cand_correct", "mean"), cand_ce=("cand_ce", "mean"),
            shared_cand_accuracy=("shared_cand_correct", "mean"), shared_cand_ce=("shared_cand_ce", "mean")).reset_index()
        t.insert(0, "group", name)
        extra.append(t)
    return out.merge(pd.concat(extra, ignore_index=True), on=["group", *keys], how="left")


@torch.no_grad()
def retention_probe(teacher, rnn, batch: dict, cfg: MQARConfig, device, horizons=(0, 1, 2, 4, 8, 16, 32)) -> pd.DataFrame:
    """H2 at answer positions q: RNN starts from the TEACHER state m_{q-L} and makes L updates on the
    true inputs; y_q is read with the frozen teacher head and compared with the teacher state at q."""
    teacher.eval(); rnn.eval()
    rows = []
    for lo in range(0, len(batch["x"]), cfg.eval_batch_size):
        part = S.slice_batch(batch, lo, lo + cfg.eval_batch_size)
        x = part["x"]
        tm = all_states(teacher, x, cfg, device)
        b, q = torch.where(part["target_mask"])
        y = part["y"][b, q]
        for L in horizons:
            keep = q >= L
            bb, qq, yy = b[keep], q[keep], y[keep]
            state = tm[bb.to(device), (qq - L).to(device)]
            for step in range(1, L + 1):
                state = rnn.rnn_update(state, x[bb, qq - L + step][:, None].to(device))
            for name, z in (("rnn_from_teacher", state), ("teacher", tm[bb.to(device), qq.to(device)])):
                logits = teacher.teacher_readout(z).float().cpu()
                rows.append(pd.DataFrame(dict(model=name, horizon=L, sequence=(bb + lo).numpy(), t=qq.numpy(),
                                              delay=part["delay"][bb, qq].numpy(),
                                              source_le_start=(part["source_pos"][bb, qq] <= qq - L).numpy(),
                                              correct=(logits.argmax(-1) == yy).numpy(),
                                              ce=F.cross_entropy(logits, yy, reduction="none").numpy())))
    return pd.concat(rows, ignore_index=True)


# --------------------------------------------------------------------------------------
# Detectors: learned (as in Stack) and idealised rule-based references for the audit
# --------------------------------------------------------------------------------------

class IdealWindow:
    """What a perfect r-token local predictor can know: the answer is certain iff the queried key and
    its value are inside the last r tokens; otherwise it can only guess among the value tokens.
    A rule-based reference for the audit, not a trained detector."""

    def __init__(self, cfg: MQARConfig):
        self.cfg = cfg

    def scores(self, batch, device):
        T = batch["x"].shape[1]
        pos = torch.arange(T).expand_as(batch["x"])
        hidden = batch["target_mask"] & (batch["key_pos"] < pos - self.cfg.local_window + 1)
        return (hidden.float() * math.log(self.cfg.vocab - self.cfg.vocab // 2)).to(device)


class IdealContext:
    """Perfect predictor from the decoder inputs x_{t+1..k} without memory: certain iff the key was
    written after the anchor (pair visible to the decoder). Rule-based reference for the audit."""

    def __init__(self, cfg: MQARConfig):
        self.cfg = cfg

    def pair_scores(self, w, device):
        hidden = w["valid"] & (w["source"] - 1 <= w["anchor"][:, None])
        return (hidden.float() * math.log(self.cfg.vocab - self.cfg.vocab // 2)).to(device)


def detector_quality(model, kind, batch, cfg: MQARConfig, device) -> dict:
    if kind == "context":
        return S.detector_quality(model, kind, batch, cfg, device)
    model.eval()
    with torch.no_grad():
        m = batch["target_mask"]
        s = model.scores(batch, device).cpu()[m].numpy()
        T = batch["x"].shape[1]
        hidden = (batch["key_pos"] < torch.arange(T).expand_as(batch["x"]) - cfg.local_window + 1)[m].numpy()
        r = cfg.local_window
        win = F.pad(batch["x"], (r - 1, 0), value=IGNORE).unfold(1, r, 1)[m]
        pred = model(win.to(device)).argmax(-1).cpu().numpy()
        y = batch["y"][m].numpy()
    return dict(ce=float(s.mean()), acc_pair_in_window=float((pred == y)[~hidden].mean()) if (~hidden).any() else float("nan"),
                acc_pair_outside=float((pred == y)[hidden].mean()) if hidden.any() else float("nan"),
                auc_pair_outside=S._auc(s, hidden), n=len(s))


def fit_detector(kind: str, cfg: MQARConfig, directory: Path, validation: dict, device, log=print):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"detector_{kind}.pt"
    identity = dict(kind=kind, run=run_identity(cfg, device))
    with S.isolated_seed(cfg.detector_seed):
        model = (S.WindowDetector if kind == "window" else S.ContextDetector)(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.detector_lr, weight_decay=cfg.weight_decay)
    start, history, best, best_state = 0, [], float("inf"), None
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, f"{path} belongs to another configuration"
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"])
        start, history, best, best_state = saved["step"], saved["history"], saved["best"], saved["best_model"]
    for step in range(start, cfg.detector_steps):
        batch = sample_mqar(cfg, cfg.detector_batch_size, S.derive_seed(101, cfg.detector_seed, step))
        model.train(); opt.zero_grad(set_to_none=True)
        loss = S.detector_train_loss(model, kind, batch, cfg, device)
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
            torch.save(dict(identity=identity, step=step + 1, history=history, best=best, best_model=best_state,
                            model=model.state_dict(), optimizer=opt.state_dict()), path)
        else:
            history.append(row)
    model.load_state_dict(best_state)
    model.eval().requires_grad_(False)
    return model, pd.DataFrame(history)


# --------------------------------------------------------------------------------------
# Training (same protocol as the Stack experiment)
# --------------------------------------------------------------------------------------

def fit(cfg: MQARConfig, variant: str, alpha: float, seed: int, out: Path, detectors: dict,
        validation: dict, device, log=print) -> dict:
    assert variant in ALL_VARIANTS
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
    amp = torch.autocast(device.type, dtype=torch.bfloat16, enabled=cfg.amp_bf16 and device.type == "cuda")
    for stage, steps in (("smt", cfg.smt_steps), ("dmt", cfg.dmt_steps)):
        is_smt = stage == "smt"
        teacher.requires_grad_(is_smt)
        if not is_smt:
            S.copy_readout(teacher, rnn)
            frozen = S.state_hash(teacher)
        params = [*teacher.parameters(), *rnn.parameters()] if is_smt else list(rnn.parameters())
        opt = torch.optim.AdamW(params, lr=cfg.smt_lr if is_smt else cfg.dmt_lr, betas=(0.9, 0.95),
                                weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, S.lr_lambda(cfg, steps))
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
            batch = sample_mqar(cfg, bs, S.derive_seed(301 if is_smt else 302, seed, step))
            gen = torch.Generator().manual_seed(S.derive_seed(303 if is_smt else 306, seed, step))
            opt.zero_grad(set_to_none=True)
            with S.isolated_seed(S.derive_seed(304 if is_smt else 305, seed, step)), amp:
                loss, parts = (smt_loss(teacher, rnn, batch, cfg, variant, alpha, detectors, gen, device)
                               if is_smt else dmt_loss(teacher, rnn, batch, cfg, device, gen))
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
                ev = evaluate(teacher, rnn, validation, cfg, device)
                score = S.long_ce(ev, cfg, "teacher" if is_smt else "rnn")
                for mname in ("teacher", "rnn"):
                    lg = ev[(ev.model == mname) & (ev.delay >= cfg.long_delay)]
                    row[f"val_{mname}_long_ce"], row[f"val_{mname}_long_acc"] = float(lg.ce.mean()), float(lg.correct.mean())
                    row[f"val_{mname}_all_acc"] = float(ev[ev.model == mname].correct.mean())
                    row[f"val_{mname}_pred_in_written"] = float(ev[ev.model == mname].pred_in_written.mean())
                    row[f"val_{mname}_long_cand_acc"] = float(lg.cand_correct.mean())
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

class Experiment:
    def __init__(self, cfg: MQARConfig, name: str, root: Path | str = DEFAULT_ROOT, device=None, log=print):
        self.cfg, self.name, self.root, self.log = cfg, name, Path(root), log
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        S.sources(self.root)
        zoology_generator(self.root)
        ident = run_identity(cfg, self.device)
        key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = self.root / "runs" / f"mqar03_{name}_{key}"
        self.dir.mkdir(parents=True, exist_ok=True)
        p = self.dir / "identity.json"
        if p.exists():
            assert json.loads(p.read_text()) == ident
        else:
            p.write_text(json.dumps(ident, indent=2))
        self.validation = sample_mqar(cfg, cfg.validation_examples, S.derive_seed(511, 0))
        self.tuning = sample_mqar(cfg, cfg.tuning_examples, S.derive_seed(512, 0))
        assert not set(S.fingerprints(self.validation)) & set(S.fingerprints(self.tuning))
        self.detectors: dict = {}

    def audit(self, kinds=("window", "context")) -> dict:
        cfg = self.cfg
        audit = pd.DataFrame([dict(split=n, **data_audit(b, cfg)) for n, b in
                              (("validation", self.validation), ("tuning", self.tuning))])
        audit.to_csv(self.dir / "data_audit.csv", index=False)
        quality = {}
        for kind in kinds:
            self.detectors[kind], _ = fit_detector(kind, cfg, self.dir, self.validation, self.device, self.log)
            quality[kind] = detector_quality(self.detectors[kind], kind, self.tuning, cfg, self.device)
        sub = S.slice_batch(self.tuning, 0, 128)
        sel = S.selection_audit(sub, cfg, self.detectors, self.device)
        sel.to_csv(self.dir / "selection_audit.csv", index=False)
        (self.dir / "detector_quality.json").write_text(json.dumps(quality, indent=2))
        return dict(data=audit, detector_quality=quality, selection=sel)

    def _need_detectors(self, variants):
        need = {"surprise_k": "window", "surprise_ctx_k": "context"}
        for kind in [need[v] for v in variants if v in need and need[v] not in self.detectors]:
            self.detectors[kind], _ = fit_detector(kind, self.cfg, self.dir, self.validation, self.device, self.log)

    def baseline(self) -> dict:
        """Baseline on the tuning seed + gate. The gate asks for key->value BINDING, not the value set: accuracy
        among the N values written in the sequence (chance 1/N) on long answers (delay >= long_delay), read on
        the tuning split with a shuffled query order (shuffle_query_order: the order carries no information)."""
        cfg = self.cfg
        saved = fit(cfg, "baseline", 0.0, cfg.tuning_seed, self.dir / "tuning" / "baseline", {}, self.validation,
                    self.device, self.log)
        shuffled = shuffle_query_order(self.tuning, cfg, S.derive_seed(514, 0))
        frames = []
        for stage in ("smt", "dmt"):
            t, r = S.load_fitted(cfg, saved, self.device, stage)
            for split, data in (("tuning", self.tuning), ("tuning_shuffled_order", shuffled)):
                frames.append(evaluate(t, r, data, cfg, self.device, seed=cfg.tuning_seed, variant="baseline",
                                       stage=stage, split=split))
        ev = pd.concat(frames, ignore_index=True)
        chance = cfg.chance_structured
        sh = ev[(ev.split == "tuning_shuffled_order") & (ev.delay >= cfg.long_delay)]
        lt = sh[(sh.model == "teacher") & (sh.stage == "smt")].cand_correct.mean()
        lr_ = sh[(sh.model == "rnn") & (sh.stage == "dmt")].cand_correct.mean()
        orig = ev[(ev.split == "tuning") & (ev.delay >= cfg.long_delay)]
        gate = dict(criterion="accuracy among the written values, long answers, shuffled query order (tuning split)",
                    teacher_long_cand_accuracy=float(lt), rnn_dmt_long_cand_accuracy=float(lr_),
                    chance_structured=chance, chance_vocab=cfg.chance_vocab,
                    required_teacher=chance + cfg.gate_teacher_margin, required_rnn=chance + cfg.gate_rnn_margin,
                    teacher_long_cand_accuracy_original_order=float(orig[(orig.model == "teacher") & (orig.stage == "smt")].cand_correct.mean()),
                    rnn_dmt_long_cand_accuracy_original_order=float(orig[(orig.model == "rnn") & (orig.stage == "dmt")].cand_correct.mean()),
                    teacher_long_accuracy_full_vocab=float(orig[(orig.model == "teacher") & (orig.stage == "smt")].correct.mean()),
                    rnn_dmt_long_accuracy_full_vocab=float(orig[(orig.model == "rnn") & (orig.stage == "dmt")].correct.mean()),
                    teacher_first_query_cand_accuracy=float(sh[(sh.model == "teacher") & (sh.stage == "smt") & (sh.query_rank == 0)].cand_correct.mean()),
                    teacher_pred_in_written=float(ev[(ev.model == "teacher") & (ev.stage == "smt")].pred_in_written.mean()),
                    smt_best_step=saved["smt_best_step"], dmt_best_step=saved["dmt_best_step"])
        gate["passed"] = bool(lt >= gate["required_teacher"] and lr_ >= gate["required_rnn"])
        (self.dir / "baseline_gate.json").write_text(json.dumps(gate, indent=2))
        ev.to_csv(self.dir / "baseline_tuning_events.csv", index=False)
        t, r = S.load_fitted(cfg, saved, self.device, "dmt")
        probe = retention_probe(t, r, self.tuning, cfg, self.device)
        probe.to_csv(self.dir / "baseline_retention_probe.csv", index=False)
        return dict(gate=gate, summary=summarize(ev, cfg, keys=("split", "stage", "model")), events=ev, retention=probe)

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
                t, r = S.load_fitted(cfg, saved, self.device, "dmt")
                ev = evaluate(t, r, self.tuning, cfg, self.device)
                rows.append(dict(variant=variant, alpha=alpha, tuning_rnn_long_ce=S.long_ce(ev, cfg, "rnn"),
                                 tuning_teacher_long_ce=S.long_ce(ev, cfg, "teacher")))
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
        lock.write_text(json.dumps(dict(alphas=alphas, criterion="tuning split, RNN after DMT, CE on answers with "
                                        f"delay>={cfg.long_delay}; checkpoints selected on validation"), indent=2))
        return dict(table=table, alphas=alphas)

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

    def final_report(self, variants=VARIANTS_AGREED, confirm_test: bool = False) -> dict:
        assert confirm_test, "pass confirm_test=True once the protocol is frozen; never tune after this"
        cfg = self.cfg
        fitted = self.final_fits(variants)
        test = sample_mqar(cfg, cfg.test_examples, S.derive_seed(513, 0))
        assert not set(S.fingerprints(test)) & (set(S.fingerprints(self.validation)) | set(S.fingerprints(self.tuning)))
        test_shuffled = shuffle_query_order(test, cfg, S.derive_seed(515, 0))
        frames = []
        for (seed, v), saved in fitted.items():
            for which in ("best", "last"):
                for stage in ("smt", "dmt"):
                    t, r = S.load_fitted(cfg, saved, self.device, stage, which)
                    frames.append(evaluate(t, r, test, cfg, self.device, seed=seed, variant=v, stage=stage,
                                           split="test", checkpoint=which))
                    if which == "best":
                        frames.append(evaluate(t, r, test_shuffled, cfg, self.device, seed=seed, variant=v, stage=stage,
                                               split="test_shuffled_order", checkpoint=which))
        events = pd.concat(frames, ignore_index=True)
        comps = [("random_k", "surprise_k"), ("uniform", "surprise_k"), ("baseline", "surprise_k"),
                 ("baseline", "uniform"), ("baseline", "random_k")]
        comps += [(ref, v) for v in variants if v in VARIANTS_OPTIONAL for ref in ("random_k", "uniform", "surprise_k")]
        comps = [(a, b) for a, b in comps if a in variants and b in variants]
        metrics = ("ce", "correct", "shared_ce", "shared_correct", "cand_ce", "cand_correct")
        intervals, per_seed = S.paired_deltas(events[(events.checkpoint == "best") & (events.split == "test")], cfg, comps, metrics)
        summary = summarize(events, cfg, keys=("split", "checkpoint", "seed", "variant", "stage", "model"))
        for name, df in (("test_events", events), ("test_summary", summary), ("test_intervals", intervals),
                         ("test_per_seed", per_seed)):
            df.to_csv(self.dir / f"{name}.csv", index=False)
        report = dict(variants=list(variants), seeds=list(cfg.seeds), test_hash=S.split_hash(test))
        (self.dir / "test_report.json").write_text(json.dumps(report, indent=2))
        return dict(events=events, summary=summary, intervals=intervals, per_seed=per_seed, report=report)


# --------------------------------------------------------------------------------------
# Cost: exact work counts (no GPU) and a short measured benchmark (GPU, throw-away models)
# --------------------------------------------------------------------------------------

def work_counts(cfg: MQARConfig, seed: int = 0) -> dict:
    """Exact numbers of encoder/decoder tokens and attention-score entries per SMT step for one batch,
    as the code above processes it (chunked, trimmed windows; decoder only for anchors with answers)."""
    batch = sample_mqar(cfg, cfg.batch_size, seed)
    B, T, Km = cfg.batch_size, cfg.seq_len, cfg.mem_tokens
    A = cfg.anchors_per_sequence if 0 < cfg.anchors_per_sequence < T else T
    g = torch.Generator().manual_seed(seed)
    if A < T:
        t_a = torch.rand(B, T, generator=g).argsort(1)[:, :A].sort(1).values.flatten()
        b_a = torch.arange(B).repeat_interleave(A)
    else:
        b_a, t_a = torch.arange(B).repeat_interleave(T), torch.arange(T).repeat(B)
    rows = b_a * T + t_a
    need = torch.unique(torch.cat([rows, rows[t_a > 0] - 1]))
    tt = (need % T).sort().values
    enc_tok = enc_att = 0
    for part in tt.split(cfg.encode_chunk):
        L = int(part.max()) + 1 + Km
        enc_tok += len(part) * L
        enc_att += len(part) * L * L
    w = S.windows(batch)
    v = w["valid"][rows]
    has = v.any(1)
    last = (v[has] * (torch.arange(T) + 1)).max(1).values.sort().values
    dec_tok = dec_att = 0
    for part in last.split(cfg.encode_chunk):
        L = Km + int(part.max())
        dec_tok += len(part) * L
        dec_att += len(part) * L * L
    return dict(encoder_windows=len(need), encoder_tokens=enc_tok, encoder_attention=enc_att * cfg.n_heads,
                decoder_rows=int(has.sum()), decoder_tokens=dec_tok, decoder_attention=dec_att * cfg.n_heads,
                answer_logits=int(v.sum()), logit_floats=int(v.sum()) * cfg.vocab)


def benchmark(cfg: MQARConfig, device, steps: int = 5, variant: str = "surprise_k") -> dict:
    teacher, rnn = S.make_models(cfg, 0, device)
    dets = {"window": S.WindowDetector(cfg).to(device).eval().requires_grad_(False),
            "context": S.ContextDetector(cfg).to(device).eval().requires_grad_(False)}
    opt = torch.optim.AdamW([*teacher.parameters(), *rnn.parameters()], lr=1e-4)
    out = {}
    for stage in ("smt", "dmt"):
        if device.type == "cuda":
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
        times = []
        for s in range(steps + 1):
            batch = sample_mqar(cfg, cfg.batch_size if stage == "smt" else cfg.dmt_batch_size, 20_000 + s)
            g = torch.Generator().manual_seed(s)
            tic = time.perf_counter()
            opt.zero_grad(set_to_none=True)
            loss, _ = (smt_loss(teacher, rnn, batch, cfg, variant, 0.3, dets, g, device) if stage == "smt"
                       else dmt_loss(teacher, rnn, batch, cfg, device, g))
            loss.backward(); opt.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if s >= 1:
                times.append(time.perf_counter() - tic)
        out[f"{stage}_s_per_step"] = float(np.median(times))
        if device.type == "cuda":
            out[f"{stage}_peak_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    val = sample_mqar(cfg, cfg.validation_examples, 7)
    if device.type == "cuda":
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(device)
    tic = time.perf_counter()
    evaluate(teacher, rnn, val, cfg, device)
    out["eval_s"] = time.perf_counter() - tic
    if device.type == "cuda":
        out["eval_peak_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
    return out


def plan(cfg: MQARConfig, bench: dict | None = None, variants=VARIANTS_AGREED) -> pd.DataFrame:
    return S.plan(cfg, bench, variants)


def main(argv=None):
    p = argparse.ArgumentParser(description="SMT x MemUP on MQAR (phases run only on request)")
    p.add_argument("--phase", required=True, choices=["benchmark", "audit", "baseline", "tune", "final", "report"])
    p.add_argument("--config", required=True, choices=sorted(CANDIDATES))
    p.add_argument("--variants", default=",".join(VARIANTS_AGREED))
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-memory-fraction", type=float, default=None)
    p.add_argument("--confirm-test", action="store_true")
    p.add_argument("--override", default="{}", help='JSON dict of MQARConfig overrides, e.g. {"anchors_per_sequence": 32}')
    a = p.parse_args(argv)
    device = S.setup_runtime(a.threads, a.gpu_memory_fraction)
    over = {k: tuple(v) if isinstance(v, list) else v for k, v in json.loads(a.override).items()}
    cfg = candidate(a.config, **over)
    variants = tuple(v for v in a.variants.split(",") if v)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.phase == "benchmark":
        S.sources(); zoology_generator()
        b = benchmark(cfg, device)
        print(json.dumps(b, indent=2)); print(plan(cfg, b, variants).to_string()); return
    ex = Experiment(cfg, a.config, device=device, log=log)
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
