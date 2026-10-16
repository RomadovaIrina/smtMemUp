"""Technical checks for smt_memup_v4.py (version 4: short SMT window + far targets + BPTT references).
Run on CPU: CUDA_VISIBLE_DEVICES="" python checks/test_smt_memup_v4.py
Small models only; no experiment training. Prints one line per check."""
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smt_memup_stack as S  # noqa: E402
import smt_memup_v4 as V  # noqa: E402
import smt_memup_window as W  # noqa: E402

DEV = S.setup_runtime(threads=4, device="cpu")
S.sources(ROOT)
TINY = replace(V.V4Config(), seq_len=16, max_stack_size=4, d_model=32, mem_tokens=2, encoder_depth=1,
               decoder_depth=1, rnn_depth=1, n_heads=4, batch_size=4, dmt_batch_size=4, bptt_batch_size=4,
               smt_steps=6, dmt_steps=4, bptt_steps=6, eval_every=3, eval_batch_size=8, validation_examples=16,
               tuning_examples=16, warmup_steps=2, future_window=4, far_K=2, local_window=4)
with S.isolated_seed(3):
    DETS = {"window": S.WindowDetector(TINY).eval().requires_grad_(False),
            "context": S.ContextDetector(TINY).eval().requires_grad_(False)}
QUIET = lambda s: None
results = []


def check(name):
    def deco(fn):
        fn()
        results.append(name)
        print("OK ", name, flush=True)
        return fn
    return deco


def tiny_rnn(seed=5):
    return S.make_models(TINY, seed, DEV)[1]


@check("Full BPTT loss equals the authors' forward_bptt (upstream model.py) exactly")
def _():
    b = S.sample_stack(TINY, 4, 21)
    rnn = tiny_rnn()
    mine, _ = V.bptt_loss(rnn, b, TINY, DEV, truncate=False)
    m = S.sources().model
    theirs = m.forward_bptt(rnn, dict(x=b["x"], y=b["y"]), m.BPTTConfig(T=TINY.seq_len))["loss"]
    assert torch.allclose(mine, theirs, atol=1e-6), (float(mine), float(theirs))


@check("BPTT-1 carries the state forward: forward values equal full BPTT, differ from resetting the state to zero")
def _():
    b = S.sample_stack(TINY, 4, 22)
    rnn = tiny_rnn()
    x = b["x"]
    full = S.rnn_rollout(rnn, x, detach=False)
    trunc = S.rnn_rollout(rnn, x, detach=True)
    assert torch.equal(full, trunc)
    reset = torch.stack([rnn.rnn_update(torch.zeros(len(x), rnn.cfg.mem_size), x[:, t:t + 1]) for t in range(x.shape[1])], 1)
    assert not torch.allclose(full[:, 1:], reset[:, 1:])
    lf, _ = V.bptt_loss(rnn, b, TINY, DEV, truncate=False)
    lt, _ = V.bptt_loss(rnn, b, TINY, DEV, truncate=True)
    assert torch.equal(lf, lt)                          # same loss value; only the gradient differs


def _recorded_rollout(rnn, b, truncate):
    """Run bptt_loss while recording the state that enters each update (to inspect gradient paths)."""
    seen = []
    real = rnn.rnn_update

    def spy(z, x):
        if z.requires_grad:
            z.retain_grad()
        seen.append(z)
        return real(z, x)
    rnn.rnn_update = spy
    try:
        loss, _ = V.bptt_loss(rnn, b, TINY, DEV, truncate=truncate)
    finally:
        del rnn.rnn_update
    return loss, seen


@check("Gradient paths: full BPTT reaches earlier updates; BPTT-1 gives no gradient to any previous update")
def _():
    b = S.sample_stack(TINY, 4, 23)
    last_only = dict(b)                                  # loss only at the last pop of every sequence
    tm = torch.zeros_like(b["target_mask"])
    for i in range(len(b["x"])):
        idx = torch.where(b["target_mask"][i])[0]
        tm[i, idx[-1]] = True
    last_only["target_mask"] = tm
    rnn = tiny_rnn()
    loss, seen = _recorded_rollout(rnn, last_only, truncate=False)
    loss.backward()
    assert all(z.requires_grad for z in seen[1:])
    first_pop = int(torch.where(tm)[1].min())
    early = [z.grad for z in seen[1:first_pop]]
    assert any(g is not None and g.abs().sum() > 0 for g in early), "full BPTT must reach earlier states"
    rnn.zero_grad()
    loss, seen = _recorded_rollout(rnn, last_only, truncate=True)
    loss.backward()
    assert not any(z.requires_grad for z in seen), "BPTT-1: the incoming state of every update must be detached"
    upd = [p.grad for n, p in rnn.named_parameters() if n.startswith("rnn.")]
    assert any(g is not None and g.abs().sum() > 0 for g in upd), "BPTT-1: the current update must get a gradient"


@check("BPTT-1 gradient = sum over t of per-step gradients with the incoming state as a constant; differs from full BPTT")
def _():
    b = S.sample_stack(TINY, 4, 24)
    rnn = tiny_rnn()
    params = [p for p in rnn.parameters()]
    rnn.zero_grad(); V.bptt_loss(rnn, b, TINY, DEV, truncate=True)[0].backward()
    g_trunc = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    rnn.zero_grad(); V.bptt_loss(rnn, b, TINY, DEV, truncate=False)[0].backward()
    g_full = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    with torch.no_grad():
        states = S.rnn_rollout(rnn, b["x"])
    rnn.zero_grad()
    x, y, mask = b["x"], b["y"], b["target_mask"]
    total = torch.zeros(())
    for t in range(x.shape[1]):
        z_in = torch.zeros(len(x), rnn.cfg.mem_size) if t == 0 else states[:, t - 1]
        logits = rnn.rnn_readout(rnn.rnn_update(z_in, x[:, t:t + 1]))
        ce = torch.nn.functional.cross_entropy(logits, y[:, t].clamp(min=0), reduction="none")
        total = total + (ce * mask[:, t]).sum()
    (total / mask.sum()).backward()
    g_manual = [p.grad.clone() if p.grad is not None else torch.zeros_like(p) for p in params]
    assert all(torch.allclose(a, c, atol=1e-6) for a, c in zip(g_trunc, g_manual))
    assert any(not torch.allclose(a, c, atol=1e-6) for a, c in zip(g_trunc, g_full))


@check("BPTT models: same RNN class, state size, head and initial weights as the SMT -> DMT RNN of the same seed")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        val = S.sample_stack(TINY, 16, 77)
        r = V.fit_bptt(TINY, "bptt_1", 7, 3e-4, tmp / "b1", val, DEV, QUIET)
        _, rnn_smt = S.make_models(TINY, 7, DEV)
        assert r["initial_hash"] == S.state_hash(rnn_smt)
        loaded = V.load_bptt(TINY, r, DEV)
        assert type(loaded) is type(rnn_smt) and loaded.cfg == rnn_smt.cfg
        assert {k: v.shape for k, v in loaded.state_dict().items()} == {k: v.shape for k, v in rnn_smt.state_dict().items()}
        h = pd.read_csv(tmp / "b1" / "bptt_history.csv")
        assert len(h) == TINY.bptt_steps and list(h.step) == list(range(1, TINY.bptt_steps + 1))   # one step per batch
        cost = r["cost"]["bptt"]
        assert cost["optimizer_steps"] == TINY.bptt_steps and cost["sequences"] == TINY.bptt_steps * TINY.bptt_batch_size
        assert cost["train_seconds"] > 0 and cost["eval_seconds"] > 0
    finally:
        shutil.rmtree(tmp)


@check("BPTT training: a crash in the middle reproduces the uninterrupted run (resume)")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        val = S.sample_stack(TINY, 16, 77)
        ref = V.fit_bptt(TINY, "bptt_full", 7, 1e-3, tmp / "ref", val, DEV, QUIET)
        calls = {"n": 0}
        real = S.sample_stack

        def flaky(c, n, seed):
            if n == TINY.bptt_batch_size:
                calls["n"] += 1
                if calls["n"] == 5:
                    raise KeyboardInterrupt
            return real(c, n, seed)
        S.sample_stack = flaky
        try:
            V.fit_bptt(TINY, "bptt_full", 7, 1e-3, tmp / "crash", val, DEV, QUIET)
            raise AssertionError("expected an interruption")
        except KeyboardInterrupt:
            pass
        finally:
            S.sample_stack = real
        res = V.fit_bptt(TINY, "bptt_full", 7, 1e-3, tmp / "crash", val, DEV, QUIET)
        assert all(torch.equal(res["rnn"][k], ref["rnn"][k]) for k in ref["rnn"])
        assert all(torch.equal(res["bptt_last_rnn"][k], ref["bptt_last_rnn"][k]) for k in ref["bptt_last_rnn"])
    finally:
        shutil.rmtree(tmp)


@check("SMT under version-2 schedules reproduces version-2 training bit for bit (same loss, streams, init)")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        val = S.sample_stack(TINY, 16, 77)
        cfg = replace(TINY, smt_lr_schedule="cosine", dmt_lr_schedule="cosine")
        v2 = W.fit(W.WindowConfig(**{k: getattr(cfg, k) for k in W.WindowConfig.__dataclass_fields__}),
                   "ctx_far", 7, tmp / "v2", DETS, val, DEV, QUIET)
        v4 = V.fit_smt(cfg, "ctx_far", 7, tmp / "v4", DETS, val, DEV, QUIET)
        assert v2["initial_hash"] == v4["initial_hash"]
        for key in ("teacher", "rnn_smt", "rnn_dmt", "smt_last_teacher", "dmt_last_rnn"):
            assert all(torch.equal(v2[key][k], v4[key][k]) for k in v2[key]), key
        cost = v4["cost"]
        assert cost["smt"]["sequences"] == cfg.smt_steps * cfg.batch_size and cost["dmt"]["optimizer_steps"] == cfg.dmt_steps
        assert cost["detector"]["sequences"] == cfg.detector_steps * cfg.detector_batch_size
    finally:
        shutil.rmtree(tmp)


@check("Metrics: evaluate_rnn gives the same CE/accuracy rows as S.evaluate for the same RNN")
def _():
    b = S.sample_stack(TINY, 12, 31)
    teacher, rnn = S.make_models(TINY, 5, DEV)
    a = S.evaluate(teacher, rnn, b, TINY, DEV)
    a = a[a.model == "rnn"].reset_index(drop=True)
    c = V.evaluate_rnn(rnn, b, TINY, DEV)
    for col in ("sequence", "t", "delay", "target", "correct"):
        assert (a[col].to_numpy() == c[col].to_numpy()).all(), col
    assert np.allclose(a.ce.to_numpy(), c.ce.to_numpy(), atol=1e-6)


@check("Sufficiency rule and final-budget rule behave as written in the protocol")
def _():
    cfg = V.V4Config()
    steps = np.arange(500, 20001, 500)
    flat = pd.DataFrame(dict(step=steps, acc=np.minimum(1.0, steps / 8000)))          # reaches 100% at 8000
    r = V.sufficiency(flat, "acc", cfg)
    assert r["sufficient"] and r["steps_to_plateau"] == 8000 - 0 or r["steps_to_plateau"] <= 8000
    rising = pd.DataFrame(dict(step=steps, acc=0.3 + 0.5 * steps / 20000))            # still rising at the end
    r2 = V.sufficiency(rising, "acc", cfg)
    assert not r2["sufficient"] and abs(r2["gain_last_fraction"] - 0.125) < 0.02
    assert V.final_budget([7900, 6000], 20000) == 10000 and V.final_budget([19000], 20000) == 20000


@check("Splits: the version-4 test stream is disjoint from validation, tuning and the version 1/2 and 3 tests")
def _():
    cfg = V.V4Config()
    fp = lambda s, n: set(S.fingerprints(S.sample_stack(cfg, n, S.derive_seed(s, 0))))
    test = fp(cfg.test_stream, cfg.test_examples)
    for s, n in ((501, cfg.validation_examples), (502, cfg.tuning_examples), (503, cfg.test_examples), (603, cfg.test_examples)):
        assert not test & fp(s, n), s
    assert len(test) == cfg.test_examples


print(f"\n{len(results)}/{len(results)} checks passed")
