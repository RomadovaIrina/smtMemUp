#!/usr/bin/env python3
"""Indexing, masks and information leaks - checked against hand-written expectations and by perturbation, not by
re-running the implementation. CPU, ~1 min.   python tests/test_masks_and_leaks.py"""
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from smtcore.config import Config  # noqa: E402
from smtcore.data import _annotate, sample_mqar, sample_stack  # noqa: E402
from smtcore.evaluate import answer_logits  # noqa: E402
from smtcore.losses import smt_loss  # noqa: E402
from smtcore.models import make_lm, make_models, rnn_rollout  # noqa: E402
from smtcore.upstream import seeded  # noqa: E402
from smtcore.windows import windows  # noqa: E402

CPU = torch.device("cpu")
TINY = dict(d_model=32, n_heads=2)
torch.set_num_threads(4)


def hand_stack():
    """push 1, push 2, pop -> 2, push 3, pop -> 3, pop -> 1   (T = 6)"""
    x = torch.tensor([[1, 2, 0, 3, 0, 0]])
    y = torch.tensor([[-1, -1, 2, -1, 3, 1]])
    src = torch.tensor([[-1, -1, 1, -1, 3, 0]])
    return _annotate(x, y, src)


def test_hand_windows():
    """Windows of the hand example, written out by hand for anchor t = 2 and t = 5."""
    w = windows(hand_stack())
    assert w["x_ctx"].shape == (6, 6)
    assert w["x_ctx"][2].tolist() == [-1, -1, -1, 1, 2, 0]         # x_0..x_2, left padded: m_2 has seen the pop x_2
    assert w["x_fut"][2].tolist() == [3, 0, 0, -1, -1, -1]          # x_3..x_5, right padded
    assert w["y_fut"][2].tolist() == [2, -1, 3, 1, -1, -1]          # y_2..y_5: output 0 = answer to the pop AT t = 2
    assert w["valid"][2].tolist() == [True, False, True, True, False, False]
    assert w["prefix"][2].tolist() == [True, False, False, True, False, False]   # y_4 = 3 pushed at 3 > 2: future
    assert w["x_ctx"][5].tolist() == [1, 2, 0, 3, 0, 0] and w["y_fut"][5].tolist() == [1, -1, -1, -1, -1, -1]
    d = hand_stack()["delay"][0].tolist()
    assert d == [-1, -1, 1, -1, 1, 5]                               # Stack delays are odd: t - push


def test_near_far_split_by_hand():
    """With Tf = 2 the near targets of anchor 2 are j in {0} (y_2), far are j in {2, 3} (y_4, y_5)."""
    w = windows(hand_stack())
    j = torch.arange(6)
    near, far = w["valid"] & (j < 2), w["valid"] & (j >= 2)
    assert near[2].nonzero().flatten().tolist() == [0] and far[2].nonzero().flatten().tolist() == [2, 3]


def test_decoder_loss_matches_loop():
    """L_dec^Tf recomputed with an explicit loop over (b, t, j) calling encode/decode on raw slices of x - catches any
    off-by-one in windows / output alignment / masking / normalisation."""
    cfg = Config(seq_len=6, max_stack_size=3, future_window=3, coef_dyn=0.0, coef_unif=0.0, **TINY)
    batch = sample_stack(cfg, 3, 5)
    teacher, rnn = make_models(cfg, 0, CPU)
    with torch.no_grad(), seeded(1):
        loss, parts = smt_loss(teacher, rnn, batch, cfg, CPU)
        total, count = 0.0, 0
        x, y = batch["x"], batch["y"]
        T = cfg.seq_len
        for b in range(len(x)):
            for t in range(T):
                ctx = torch.cat([torch.full((T - 1 - t,), -1), x[b, :t + 1]])[None]           # x_0..x_t padded
                m = teacher.encode(ctx)
                for j in range(cfg.future_window):
                    if t + j >= T or y[b, t + j] < 0:
                        continue
                    fut = torch.cat([x[b, t + 1:t + 1 + j], torch.full((T - j,), -1)])[None]      # x_{t+1..t+j}
                    logits = teacher.decode(m, fut)[0, j]
                    total += float(F.cross_entropy(logits[None], y[b, t + j][None])); count += 1
    assert abs(total / count - float(parts["decoder"])) < 1e-5, (total / count, float(parts["decoder"]))


def test_teacher_readout_is_output_j0():
    cfg = Config(seq_len=8, max_stack_size=4, **TINY)
    teacher, _ = make_models(cfg, 0, CPU)
    w = windows(sample_stack(cfg, 2, 3))
    with torch.no_grad():
        m = teacher.encode(w["x_ctx"])
        assert torch.allclose(teacher.teacher_readout(m), teacher.decode(m, w["x_fut"])[:, 0], atol=1e-5)


def test_no_future_reaches_memory_or_decoder():
    """Perturb inputs after a position and check what must NOT change:
    m_t vs x_{>t};  decoder output j of anchor t vs x_{>t+j};  RNN readout at t vs x_{>t};  causal LM logits at t."""
    cfg = Config(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4, future_window=16, **TINY)
    batch = sample_mqar(cfg, 4, 11)
    teacher, rnn = make_models(cfg, 0, CPU)
    lm = make_lm(cfg, 0, CPU)
    t, j = 6, 3
    x2 = batch["x"].clone()
    x2[:, t + j + 1:] = torch.randint(0, cfg.vocab, x2[:, t + j + 1:].shape)
    x3 = batch["x"].clone()
    x3[:, t + 1:] = torch.randint(0, cfg.vocab, x3[:, t + 1:].shape)
    with torch.no_grad():
        w1, w2, w3 = (windows(dict(batch, x=x)) for x in (batch["x"], x2, x3))
        rows = torch.arange(4) * 16 + t
        m1, m3 = teacher.encode(w1["x_ctx"][rows]), teacher.encode(w3["x_ctx"][rows])
        assert torch.allclose(m1, m3, atol=1e-6), "memory m_t depends on x after t"
        d1 = teacher.decode(m1, w1["x_fut"][rows])[:, j]
        d2 = teacher.decode(teacher.encode(w2["x_ctx"][rows]), w2["x_fut"][rows])[:, j]
        assert torch.allclose(d1, d2, atol=1e-5), "decoder output j depends on x after t + j"
        r1, r3 = rnn_rollout(rnn, batch["x"])[:, t], rnn_rollout(rnn, x3)[:, t]
        assert torch.allclose(rnn.rnn_readout(r1), rnn.rnn_readout(r3), atol=1e-5), "RNN at t depends on x after t"
        lm.eval()
        assert torch.allclose(lm(batch["x"])[:, t], lm(x3)[:, t], atol=1e-5), "causal LM at t depends on x after t"
        # and the check is not vacuous: changing x at t+1..t+j DOES change decoder output j
        x4 = batch["x"].clone(); x4[:, t + 1] = (x4[:, t + 1] + 7) % cfg.vocab
        d4 = teacher.decode(m1, windows(dict(batch, x=x4))["x_fut"][rows])[:, j]
        assert not torch.allclose(d1, d4), "perturbation test is vacuous"


def test_mqar_layout_against_generator_contract():
    """For many generated sequences: 2 tokens per pair at 0..2N-1 (key range, value range), keys/values unique,
    each key queried exactly once at an even position >= 2N, label = the value written right after that key,
    filler 0 everywhere else, no value ID after the pairs (nothing after the pairs reveals an answer)."""
    for T, N in ((16, 2), (32, 4), (64, 8)):
        cfg = Config(task="mqar", seq_len=T, num_kv_pairs=N, future_window=T)
        bt = sample_mqar(cfg, 300, 2024)
        for x, y in zip(bt["x"].tolist(), bt["y"].tolist()):
            keys, vals = x[0:2 * N:2], x[1:2 * N:2]
            assert all(1 <= k < 64 for k in keys) and all(64 <= v < 128 for v in vals)
            assert len(set(keys)) == N and len(set(vals)) == N
            q = [t for t in range(T) if y[t] != -1]
            assert len(q) == N and all(t >= 2 * N and t % 2 == 0 for t in q)
            assert sorted(x[t] for t in q) == sorted(keys)
            assert all(y[t] == vals[keys.index(x[t])] for t in q)
            assert all(x[t] == 0 for t in range(2 * N, T) if t not in q)
            assert not any(v >= 64 for v in x[2 * N:])


def test_pre_query_read_mode_sees_no_query_in_memory():
    """teacher_pre: the memory window ends before the query (a < t) and contains all pairs; the query key arrives as
    the LAST decoder input; prediction is unchanged if everything after the query is perturbed."""
    cfg = Config(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4, future_window=16, **TINY)
    batch = sample_mqar(cfg, 8, 5)
    teacher, _ = make_models(cfg, 0, CPU)
    a1 = answer_logits("teacher_pre", teacher, batch, cfg, CPU, "after_pairs")
    x2 = batch["x"].clone()
    for b in range(8):
        last_q = int(batch["target_mask"][b].nonzero().max())
        x2[b, last_q + 1:] = torch.randint(0, 128, (15 - last_q,))
    a2 = answer_logits("teacher_pre", teacher, dict(batch, x=x2), cfg, CPU, "after_pairs")
    last = a1.groupby("sequence").t.transform("max") == a1.t
    assert (a1[last].pred.values == a2[last].pred.values).all()


TESTS = [test_hand_windows, test_near_far_split_by_hand, test_decoder_loss_matches_loop, test_teacher_readout_is_output_j0,
         test_no_future_reaches_memory_or_decoder, test_mqar_layout_against_generator_contract,
         test_pre_query_read_mode_sees_no_query_in_memory]

if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        tic = time.time()
        try:
            test(); print(f"PASS {test.__name__} ({time.time() - tic:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1; print(f"FAIL {test.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
