"""1. Data. A 'token' here is an integer ID that indexes an embedding table (nn.Embedding(V, d_model)); there is no
text tokenisation. Every batch is a dict of (B, T) LongTensors:
    x            input ID at each position              Stack: 0 = pop, 1..4 = push that element
                                                        MQAR : 0 = filler, 1..V/2-1 = key, V/2..V-1 = value
    y            correct answer ID at each position, IGNORE (-1) where there is nothing to answer
    target_mask  y != -1: the ONLY positions that enter a loss or a metric
    source_pos   where the answer was written (Stack: its push; MQAR: the value of the queried pair) - metadata
    delay        t - source_pos ("how far back"), -1 where there is no answer - metadata for metrics
Answers sit AT the position of the token that asks for them (Stack pop, MQAR query key): no shift by one.
"""
from __future__ import annotations

import torch

from .config import Config
from .upstream import IGNORE, author, derive_seed, seeded


def sample_stack(cfg: Config, batch_size: int, seed: int) -> dict:
    """Authors' StackDataset.sample_batch (upstream/smt/src/my_datasets/synthetic.py:40-64), unchanged:
    at every step push (prob 1/2, forced if the stack is empty) a uniform element 1..4, otherwise pop
    (forced if the stack holds max_stack_size). y = popped element at a pop, -1 at a push."""
    ds = author("stack").StackDataset(n_elements=cfg.n_elements, max_stack_size=cfg.max_stack_size)
    with seeded(seed):
        raw = ds.sample_batch(batch_size=batch_size, ctx_len=cfg.seq_len)
    x, y = raw["x"].long(), raw["y"].long()                     # (B, T), (B, T)
    source = torch.full_like(x, IGNORE)                         # replay of the stack: where each popped element came from
    for b in range(len(x)):
        stack = []
        for t, token in enumerate(x[b].tolist()):
            if token > 0:
                stack.append(t)
            else:
                source[b, t] = stack.pop()
    return _annotate(x, y, source)


def sample_mqar(cfg: Config, batch_size: int, seed: int) -> dict:
    """Zoology multiquery_ar (upstream/zoology/zoology/data/multiquery_ar.py:18-141), random_non_queries=False.
    2 tokens per pair: positions 2i (key) and 2i+1 (value), i < N. Keys and values are drawn WITHOUT replacement inside
    a sequence (unique), keys from 1..V/2-1, values from V/2..V-1. Each key is queried once, at position 2N + 2*gap,
    gaps drawn without replacement from a power law over 0..(T-2N)/2-1. Every other position after the pairs is the
    filler 0. The label (the value) is at the query position. No value ID occurs after position 2N-1. There is no
    padding inside a sequence; -1 appears only when windows are built (windows.py)."""
    with seeded(seed):
        seg = author("mqar")(vocab_size=cfg.vocab, num_examples=batch_size, input_seq_len=cfg.seq_len,
                             seed=int(seed) % 2**32, power_a=cfg.power_a, num_kv_pairs=cfg.num_kv_pairs, num_passes=1,
                             random_non_queries=False)
    x, y = seg.inputs.long().clone(), seg.labels.long().clone()      # (B, T)
    y[y == -100] = IGNORE                                             # Zoology's ignore index -> ours
    N = cfg.num_kv_pairs
    keys = x[:, 0:2 * N:2]                                            # (B, N)
    slot = (x[:, :, None] == keys[:, None, :]).float().argmax(-1)     # (B, T): which pair a query key asks for
    source = torch.where(y.ne(IGNORE), 2 * slot + 1, torch.full_like(x, IGNORE))
    return _annotate(x, y, source)


def _annotate(x, y, source) -> dict:
    mask = y.ne(IGNORE)
    pos = torch.arange(x.shape[1]).expand_as(x)
    return dict(x=x, y=y, target_mask=mask, source_pos=source,
                delay=torch.where(mask, pos - source, torch.full_like(x, IGNORE)))


def sample(cfg: Config, batch_size: int, seed: int) -> dict:
    return (sample_stack if cfg.task == "stack" else sample_mqar)(cfg, batch_size, seed)


# Data streams, identical to the archived code. Training batch of step s: derive_seed(301, seed, s) for SMT / BPTT /
# LM, derive_seed(302, seed, s) for DMT. Validation (checkpoint selection): 501 (Stack) / 511 (MQAR).
# Independent evaluation of the new runs: 901 (never used before 21.10.2026, never used for a choice).
def validation_set(cfg: Config) -> dict:
    return sample(cfg, cfg.validation_examples, derive_seed(501 if cfg.task == "stack" else 511, 0))


def evaluation_set(cfg: Config) -> dict:
    return sample(cfg, cfg.eval_examples, derive_seed(901, 0))


def stack_states(x_row: torch.Tensor) -> list[tuple[list, list]]:
    """For tables only: the stack (bottom -> top) before and after every step of one Stack sequence."""
    stack, out = [], []
    for token in x_row.tolist():
        before = list(stack)
        stack.append(token) if token > 0 else stack.pop()
        out.append((before, list(stack)))
    return out
