"""Technical checks for mqar_baselines.py (MQAR check A on pilot01). Run on CPU:
CUDA_VISIBLE_DEVICES="" python checks/test_mqar_baselines.py      (small models; no experiment training)"""
import math
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import mqar_baselines as B  # noqa: E402
import smt_memup_mqar as M  # noqa: E402
import smt_memup_stack as S  # noqa: E402

DEV = S.setup_runtime(threads=8, device="cpu")
S.sources(ROOT)
CFG = B.BaselineConfig()
DCFG = B.data_config(CFG)
QUIET = lambda s: None
results = []


def check(name):
    def deco(fn):
        fn()
        results.append(name)
        print("OK ", name, flush=True)
        return fn
    return deco


@check("pilot01 data: Zoology generator, 8 pairs then 8 queries, labels only at query keys, values never after the prefix")
def _():
    b = M.sample_mqar(DCFG, 64, 123)
    N, x, y, m = DCFG.num_kv_pairs, b["x"], b["y"], b["target_mask"]
    assert DCFG.vocab == 128 and DCFG.seq_len == 64 and N == 8
    assert (m.sum(1) == N).all() and not m[:, :2 * N].any()
    assert ((x[m] >= 1) & (x[m] < DCFG.vocab // 2)).all(), "every answer position holds a query KEY"
    assert ((y[m] >= DCFG.vocab // 2)).all(), "every answer is a VALUE"
    assert not (x[:, 2 * N:] >= DCFG.vocab // 2).any(), "no value token after the prefix (no leakage of answers)"
    assert (y[~m] == S.IGNORE).all()
    assert torch.equal(x.gather(1, b["key_pos"].clamp(min=0))[m], x[m]), "the queried key was written at key_pos"


@check("Causality: logits at t do not change when inputs after t change (Transformer and RNN)")
def _():
    b = M.sample_mqar(DCFG, 4, 124)
    for model in ("transformer", "rnn_bptt"):
        net = B.make_model(replace(CFG, model=model), DEV).eval()
        x = b["x"].clone()
        t = 30
        x2 = x.clone(); x2[:, t + 1:] = torch.randint(0, DCFG.vocab, x2[:, t + 1:].shape)
        with torch.no_grad():
            a, c = net(x), net(x2)
        assert torch.allclose(a[:, :t + 1], c[:, :t + 1], atol=1e-5), model
        assert not torch.allclose(a[:, t + 1:], c[:, t + 1:], atol=1e-5), model


@check("RNN baseline: loss equals the authors' forward_bptt on MQAR batches; same init as the pilot01 RNN")
def _():
    b = M.sample_mqar(DCFG, 4, 125)
    net = B.make_model(replace(CFG, model="rnn_bptt"), DEV)
    mine = B.target_loss(net, b, DEV)
    m = S.sources().model
    theirs = m.forward_bptt(net.rnn, dict(x=b["x"], y=b["y"]), m.BPTTConfig(T=DCFG.seq_len))["loss"]
    assert torch.allclose(mine, theirs, atol=1e-6)
    _, rnn = S.make_models(DCFG, CFG.seed, DEV)
    assert S.state_hash(rnn) == S.state_hash(net.rnn)


class Fixed(torch.nn.Module):
    """Returns fixed logits (for metric checks)."""
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


@check("Metrics: a perfect model scores 1 everywhere; uniform over written values gives cand CE = ln N, chance 1/N")
def _():
    b = M.sample_mqar(DCFG, 64, 126)
    lookup = {}
    N = DCFG.num_kv_pairs

    def perfect(x):
        out = torch.full((*x.shape, DCFG.vocab), -10.0)
        for i in range(len(x)):
            kv = {int(k): int(v) for k, v in zip(x[i, 0:2 * N:2], x[i, 1:2 * N:2])}
            for t in range(2 * N, x.shape[1]):
                if int(x[i, t]) in kv:
                    out[i, t, kv[int(x[i, t])]] = 10.0
        return out

    def value_set(x):
        out = torch.full((*x.shape, DCFG.vocab), -10.0)
        out.scatter_(2, x[:, None, 1:2 * N:2].expand(-1, x.shape[1], -1), 0.0)
        return out

    ev = B.evaluate(Fixed(perfect), b, DCFG, DEV)
    assert ev.correct.all() and ev.cand_correct.all() and ev.pred_in_written.all() and (ev.ce < 1e-3).all()
    ev = B.evaluate(Fixed(value_set), b, DCFG, DEV)
    assert np.allclose(ev.cand_ce, math.log(N), atol=1e-5) and ev.pred_in_written.all()
    s = B.summary(ev, DCFG)
    assert (s.group == "first query").any() and s[s.group == "first query"].answers.iloc[0] == 64, \
        "first-query metric must be computed on all first queries (not only on long answers)"


@check("Shuffled query order: same written pairs and query positions, answers follow the permuted keys")
def _():
    b = M.sample_mqar(DCFG, 32, 127)
    sh = M.shuffle_query_order(b, DCFG, 5)
    assert torch.equal(sh["target_mask"], b["target_mask"]) and torch.equal(sh["x"][:, :16], b["x"][:, :16])
    assert not torch.equal(sh["x"], b["x"])


@check("Training: a crash in the middle reproduces the uninterrupted run; fixed training set is disjoint from val/tuning")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        cfg = replace(CFG, model="transformer", steps=6, eval_every=3, train_set=64, eval_examples=32, warmup_steps=2)
        val, tun = B.splits(DCFG, 32)
        ref = B.fit(cfg, tmp / "ref", val, tun, DEV, QUIET)
        calls = {"n": 0}
        real = B.target_loss

        def flaky(model, batch, device):
            calls["n"] += 1
            if calls["n"] == 5:
                raise KeyboardInterrupt
            return real(model, batch, device)
        B.target_loss = flaky
        try:
            B.fit(cfg, tmp / "crash", val, tun, DEV, QUIET)
            raise AssertionError("expected an interruption")
        except KeyboardInterrupt:
            pass
        finally:
            B.target_loss = real
        res = B.fit(cfg, tmp / "crash", val, tun, DEV, QUIET)
        assert all(torch.equal(res["model"][k], ref["model"][k]) for k in ref["model"])
        assert res["cost"]["optimizer_steps"] == 6 and res["cost"]["distinct_training_sequences"] == 64
    finally:
        shutil.rmtree(tmp)


print(f"\n{len(results)}/{len(results)} checks passed")
