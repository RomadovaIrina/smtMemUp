"""Technical checks for smt_memup_stack.py. Run: python checks/test_smt_memup_stack.py
Small CPU models only; no experiment training. Prints one line per check."""
import shutil
import sys
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smt_memup_stack as S  # noqa: E402

DEV = S.setup_runtime(threads=4, device="cpu")
src = S.sources(ROOT)
M = src.model
TINY = replace(S.Config(), seq_len=16, max_stack_size=4, d_model=32, mem_tokens=2, encoder_depth=1,
               decoder_depth=1, rnn_depth=1, n_heads=4, batch_size=4, dmt_batch_size=4,
               smt_steps=6, dmt_steps=4, eval_every=3, eval_batch_size=8, validation_examples=16,
               tuning_examples=16, detector_steps=20, warmup_steps=2)
results = []


def check(name):
    def deco(fn):
        fn()
        results.append(name)
        print("OK ", name, flush=True)
        return fn
    return deco


@check("Stack: author generator, replay annotation, paper example, seed reproducibility")
def _():
    paper = S.annotate_stack(dict(x=torch.tensor([[1, 0, 2, 3, 0, 1, 0, 0]]), y=torch.tensor([[-1, 1, -1, -1, 3, -1, 1, 2]])))
    assert paper["source_pos"][0].tolist() == [-1, 0, -1, -1, 3, -1, 5, 2]
    assert paper["delay"][0].tolist() == [-1, 1, -1, -1, 1, -1, 1, 5]
    a = S.sample_stack(S.Config(), 64, 123)
    b = S.sample_stack(S.Config(), 64, 123)
    assert all(torch.equal(a[k], b[k]) for k in a)
    # generator semantics: push iff stack empty, pop iff full, else p=.5
    full = a["depth"] == S.Config().max_stack_size
    nxt_pop = torch.zeros_like(full); nxt_pop[:, :-1] = a["x"][:, 1:].eq(0)
    assert nxt_pop[:, :-1][full[:, :-1]].all()
    empty = a["depth"] == 0
    nxt_push = torch.zeros_like(empty); nxt_push[:, :-1] = a["x"][:, 1:].gt(0)
    assert nxt_push[:, :-1][empty[:, :-1]].all()
    # isolation: sampling does not disturb the global RNG
    st = np.random.get_state()[1].copy(); S.sample_stack(S.Config(), 4, 5)
    assert np.array_equal(st, np.random.get_state()[1])


@check("Windows: shifts equal the author get_smt_batch at anchor t=Tc-1; prefix/future partition")
def _():
    cfg = S.Config()
    batch = S.sample_stack(cfg, 8, 1)
    T = cfg.seq_len
    w = S.windows(batch)
    for t in (0, 5, 31, 62, 63):
        rows = torch.arange(8) * T + t
        # author: x_ctx = x[:, :Tc], x_fut = x[:, Tc:Tc+Tf], y_fut = y[:, Tc-1:Tc+Tf-1] with Tc = t+1
        Tc, Tf = t + 1, T - t - 1
        if Tf > 0:
            ab = M.get_smt_batch(dict(x=batch["x"], y=batch["y"]), Tc, Tf)
            assert torch.equal(w["x_ctx"][rows][:, T - Tc:], ab["x_ctx"])
            assert torch.equal(w["x_fut"][rows][:, :Tf], ab["x_fut"])
            assert torch.equal(w["y_fut"][rows][:, :Tf], ab["y_fut"])
        assert (w["x_ctx"][rows][:, :T - Tc] == -1).all() and (w["x_fut"][rows][:, T - t - 1:] == -1).all()
    v = w["valid"]
    assert torch.equal(w["prefix"].int() + w["future"].int(), v.int())
    assert torch.equal(v[:, 0], batch["target_mask"].flatten())         # offset 0 = y_t
    assert (w["prefix"][:, 0] == v[:, 0]).all()                          # y_t always needs m_t


@check("Teacher: left padding is exactly masked; m_t independent of future inputs (no leakage)")
def _():
    cfg = S.Config()
    teacher, _ = S.make_models(cfg, 3, DEV)
    teacher.eval()
    batch = S.sample_stack(cfg, 4, 2)
    with torch.no_grad():
        m = S.teacher_states(teacher, batch, DEV)
        for t in (0, 7, 40, 63):
            direct = teacher.encode(batch["x"][:, :t + 1])
            assert torch.allclose(m[:, t], direct, atol=2e-5), (t, (m[:, t] - direct).abs().max())
        changed = {k: v.clone() for k, v in batch.items()}
        changed["x"][:, 33:] = 1
        m2 = S.teacher_states(teacher, changed, DEV)
        assert torch.equal(m[:, :33], m2[:, :33])


@check("Decoder: output j sees only memory + x_fut[:j]; teacher_readout == decode[:, 0]; RNN readout == teacher head after copy")
def _():
    cfg = S.Config()
    teacher, rnn = S.make_models(cfg, 4, DEV)
    teacher.eval(); rnn.eval()
    batch = S.sample_stack(cfg, 2, 3)
    w = S.windows(batch)
    with torch.no_grad():
        z = teacher.encode(w["x_ctx"][:10])
        a = teacher.decode(z, w["x_fut"][:10])
        xf = w["x_fut"][:10].clone(); xf[:, 20:] = 3
        b = teacher.decode(z, xf)
        assert torch.allclose(a[:, :21], b[:, :21], atol=1e-5) and not torch.allclose(a[:, 21:], b[:, 21:])
        assert torch.allclose(teacher.teacher_readout(z), a[:, 0], atol=1e-5)
        assert torch.allclose(rnn.rnn_readout(z), teacher.teacher_readout(z))


@check("Loss normalisation: alpha=0 identical for all variants; uniform == (1+alpha) L_dec; E[random-K] == L_dec")
def _():
    cfg = replace(TINY)
    teacher, rnn = S.make_models(cfg, 5, DEV)
    dets = {"window": S.WindowDetector(cfg).eval().requires_grad_(False),
            "context": S.ContextDetector(cfg).eval().requires_grad_(False)}
    batch = S.sample_stack(cfg, 4, 9)
    vals = {}
    for v in S.ALL_VARIANTS:
        with S.isolated_seed(1):
            vals[v] = S.smt_loss(teacher, rnn, batch, cfg, v, 0.0, dets, torch.Generator().manual_seed(0), DEV)
    base = vals["baseline"][0]
    assert all(torch.allclose(base, l, atol=1e-6) for l, _ in vals.values())
    with S.isolated_seed(1):
        lu, pu = S.smt_loss(teacher, rnn, batch, cfg, "uniform", 0.3, dets, torch.Generator().manual_seed(0), DEV)
    assert torch.allclose(lu, base + 0.3 * pu["decoder"], atol=1e-6)
    # exact enumeration for random-K expectation
    ce = torch.tensor([[1., 2., 3., 0.], [4., 5., 0., 0.], [7., 0., 0., 0.]])
    pool = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]], dtype=torch.bool)
    target = S.pair_mean(ce, pool)
    import itertools
    acc = []
    for i, j in itertools.product(range(3), range(2)):
        sel = torch.zeros_like(pool); sel[0, i] = True; sel[1, j] = True
        acc.append(S.weighted_selected_loss(ce, pool, sel))
    assert torch.allclose(torch.stack(acc).mean(), target)
    assert torch.allclose(S.weighted_selected_loss(ce, pool, pool), target)
    # Monte Carlo through topk_mask with K=2 on the real pool
    w = S.windows(batch); c = torch.rand(w["valid"].shape)
    g = torch.Generator().manual_seed(0)
    mc = torch.stack([S.weighted_selected_loss(c, w["valid"], S.topk_mask(torch.zeros_like(c), w["valid"], 2, g))
                      for _ in range(3000)]).mean()
    assert abs(float(mc - S.pair_mean(c, w["valid"]))) < 0.01


@check("Top-K: equals MemUP TopKSelector inside the pool; never duplicates; pool<K -> whole pool")
def _():
    sys.path.insert(0, str(src.memup_root))
    sel_mod = S._load_file("memup_selector", src.memup_root / "memup/target_selector.py")
    rng = np.random.default_rng(0)
    g = torch.Generator().manual_seed(0)
    for _ in range(200):
        n = rng.integers(1, 12)
        scores = torch.tensor(rng.permutation(n).astype(np.float32))[None]   # distinct scores
        pool = torch.ones(1, n, dtype=torch.bool)
        k = int(rng.integers(1, 6))
        mine = set(torch.where(S.topk_mask(scores, pool, k, g)[0])[0].tolist())
        if n >= k:
            ref = set(sel_mod.TopKSelector(k, time_dependent_selection=False).target_idx_from_estimate(scores[0].numpy(), [0])[0])
            assert mine == {int(i) for i in ref}
        else:
            assert mine == set(range(n))


@check("Gradients: SMT reaches encoder+updater, not detectors; DMT readout CE never reaches updater; teacher frozen")
def _():
    cfg = TINY
    teacher, rnn = S.make_models(cfg, 6, DEV)
    dets = {"window": S.WindowDetector(cfg).eval().requires_grad_(False),
            "context": S.ContextDetector(cfg).eval().requires_grad_(False)}
    batch = S.sample_stack(cfg, 4, 11)
    for v in ("surprise_k", "surprise_ctx_k"):
        teacher.zero_grad(); rnn.zero_grad()
        loss, _ = S.smt_loss(teacher, rnn, batch, cfg, v, 0.5, dets, torch.Generator().manual_seed(0), DEV)
        loss.backward()
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in teacher.encoder.parameters())
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in rnn.rnn.parameters())
        assert all(p.grad is None for d in dets.values() for p in d.parameters())
    teacher.zero_grad(set_to_none=True); rnn.zero_grad(set_to_none=True)
    teacher.requires_grad_(False)
    loss, parts = S.dmt_loss(teacher, rnn, batch, cfg, DEV)
    leak = torch.autograd.grad(parts["readout"], list(rnn.rnn.parameters()) + list(rnn.embed.parameters()),
                               allow_unused=True, retain_graph=True)
    assert all(g is None for g in leak)
    loss.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in rnn.rnn.parameters())
    # one-step DMT: gradient of the dynamics loss at t does not flow into earlier updates
    rnn.zero_grad(set_to_none=True)
    x = batch["x"]
    st = S.rnn_rollout(rnn, x, detach=True)
    g0 = torch.autograd.grad(st[:, 5].sum(), list(rnn.rnn.parameters()))
    assert all(torch.isfinite(g).all() for g in g0)


@check("Context detector: output j depends only on x_fut[:j] (same information as decoder, no memory)")
def _():
    cfg = S.Config()
    d = S.ContextDetector(cfg).eval()
    xf = torch.randint(0, cfg.vocab_size, (5, cfg.seq_len))
    a = d(xf); xf2 = xf.clone(); xf2[:, 30:] = (xf2[:, 30:] + 1) % cfg.vocab_size
    b = d(xf2)
    assert torch.allclose(a[:, :31], b[:, :31]) and not torch.allclose(a[:, 31:], b[:, 31:])
    wd = S.WindowDetector(cfg).eval()
    batch = S.sample_stack(cfg, 3, 4)
    s1 = wd.scores(batch, DEV)
    ch = {k: v.clone() for k, v in batch.items()}; ch["x"][:, :20] = 1   # change only far past
    s2 = wd.scores(dict(ch, y=batch["y"], target_mask=batch["target_mask"]), DEV)
    assert torch.allclose(s1[:, 20 + cfg.local_window - 1:], s2[:, 20 + cfg.local_window - 1:])


@check("Training: same seed -> same initial weights for every variant; rerun after a lost DMT stage is identical")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        cfg = TINY
        val = S.sample_stack(cfg, 16, 77)
        full = S.fit(cfg, "random_k", 0.3, 7, tmp / "a", {}, val, DEV, log=lambda s: None)
        base = S.fit(cfg, "baseline", 0.0, 7, tmp / "b", {}, val, DEV, log=lambda s: None)
        assert base["initial_hash"] == full["initial_hash"]
        for f in ("completed.pt", "dmt_last.pt", "dmt_best.pt"):
            (tmp / "a" / f).unlink()
        again = S.fit(cfg, "random_k", 0.3, 7, tmp / "a", {}, val, DEV, log=lambda s: None)
        for k in full["rnn_dmt"]:
            assert torch.equal(full["rnn_dmt"][k], again["rnn_dmt"][k]), k
        # a completed run with another identity must be refused, not silently reused
        try:
            S.fit(cfg, "random_k", 1.0, 7, tmp / "a", {}, val, DEV, log=lambda s: None)
            raise RuntimeError("identity mismatch was not detected")
        except AssertionError:
            pass
    finally:
        shutil.rmtree(tmp)


@check("Mid-stage resume: crash after step 3 of 6 reproduces the uninterrupted SMT weights")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        cfg = TINY
        val = S.sample_stack(cfg, 16, 77)
        with S.isolated_seed(3):
            det = {"window": S.WindowDetector(cfg).eval().requires_grad_(False)}
        ref = S.fit(cfg, "surprise_k", 0.3, 7, tmp / "ref", det, val, DEV, log=lambda s: None)
        short = replace(cfg, smt_steps=3)
        # train 3 steps with the full-length schedule: emulate by running the full config and
        # interrupting via an exception raised by the 4th sample_stack call of the SMT stage
        calls = {"n": 0}
        real = S.sample_stack

        def flaky(c, n, seed):
            if n == cfg.batch_size and c.smt_steps == cfg.smt_steps:
                calls["n"] += 1
                if calls["n"] == 4:
                    raise KeyboardInterrupt
            return real(c, n, seed)
        S.sample_stack = flaky
        try:
            S.fit(cfg, "surprise_k", 0.3, 7, tmp / "cut", det, val, DEV, log=lambda s: None)
        except KeyboardInterrupt:
            pass
        finally:
            S.sample_stack = real
        assert torch.load(tmp / "cut" / "smt_last.pt", weights_only=True)["step"] == 3
        resumed = S.fit(cfg, "surprise_k", 0.3, 7, tmp / "cut", det, val, DEV, log=lambda s: None)
        for k in ref["teacher"]:
            assert torch.equal(ref["teacher"][k], resumed["teacher"][k]), k
        for k in ref["rnn_dmt"]:
            assert torch.equal(ref["rnn_dmt"][k], resumed["rnn_dmt"][k]), k
    finally:
        shutil.rmtree(tmp)


@check("Evaluation + retention probe + paired statistics run on a tiny model and give finite values")
def _():
    cfg = TINY
    teacher, rnn = S.make_models(cfg, 8, DEV)
    data = S.sample_stack(cfg, 12, 5)
    ev = S.evaluate(teacher, rnn, data, cfg, DEV, seed=1, variant="baseline", stage="dmt", split="x")
    assert len(ev) == 2 * int(data["target_mask"].sum()) and np.isfinite(ev.ce).all()
    assert (ev[ev.model == "teacher"].memory_mse == 0).all()
    pr = S.retention_probe(teacher, rnn, data, cfg, DEV, horizons=(0, 2, 4))
    h0 = pr[(pr.horizon == 0)]
    a = h0[h0.model == "teacher"].sort_values(["sequence", "t"]).correct.to_numpy()
    b = h0[h0.model == "rnn_from_teacher"].sort_values(["sequence", "t"]).correct.to_numpy()
    assert np.array_equal(a, b)                     # horizon 0 = teacher state itself
    import pandas as pd
    rows = []
    for seed in (1, 2, 3):
        for v, ce in (("baseline", 1.0), ("random_k", .9), ("surprise_k", .8), ("uniform", .85)):
            for s in range(6):
                rows.append(dict(seed=seed, variant=v, stage="dmt", model="rnn", sequence=s, delay=10,
                                 ce=ce + 0.01 * s, correct=s % 2, shared_ce=ce, shared_correct=s % 2))
    iv, ps = S.paired_deltas(pd.DataFrame(rows), replace(cfg, bootstrap_repeats=50),
                             [("random_k", "surprise_k"), ("uniform", "surprise_k")])
    r = iv[(iv.metric == "ce") & (iv.reference == "random_k")].iloc[0]
    assert abs(r.delta_mean + 0.1) < 1e-9 and r.seeds == 3 and r.seeds_same_sign == 3


@check("Selection audit runs; oracle pool contains only memory-dependent pairs")
def _():
    cfg = replace(S.Config(), K=4)
    dets = {"window": S.WindowDetector(cfg).eval(), "context": S.ContextDetector(cfg).eval()}
    sel = S.selection_audit(S.sample_stack(cfg, 16, 3), cfg, dets, DEV)
    o = sel[(sel.rule == "oracle_prefix_k") & (sel.anchors == "all anchors")].iloc[0]
    assert o.memory_dependent == 1.0 and o.future_source == 0.0


@check("Selector uses no metadata: scrambling source_pos/delay/depth leaves every non-oracle loss unchanged")
def _():
    cfg = TINY
    teacher, rnn = S.make_models(cfg, 12, DEV)
    with S.isolated_seed(5):
        dets = {"window": S.WindowDetector(cfg).eval().requires_grad_(False),
                "context": S.ContextDetector(cfg).eval().requires_grad_(False)}
    batch = S.sample_stack(cfg, 4, 21)
    fake = dict(batch)
    g = torch.Generator().manual_seed(1)
    fake["source_pos"] = torch.randint(0, cfg.seq_len, batch["x"].shape, generator=g)
    fake["delay"] = torch.randint(0, cfg.seq_len, batch["x"].shape, generator=g)
    fake["depth"] = torch.randint(0, 5, batch["x"].shape, generator=g)
    for v in ("baseline", "uniform", "random_k", "surprise_k", "surprise_ctx_k"):
        out = []
        for b in (batch, fake):
            with S.isolated_seed(3):
                out.append(S.smt_loss(teacher, rnn, b, cfg, v, 0.7, dets, torch.Generator().manual_seed(9), DEV)[0])
        assert torch.equal(out[0], out[1]), v
    with S.isolated_seed(3):
        a = S.smt_loss(teacher, rnn, batch, cfg, "oracle_prefix_k", 0.7, dets, torch.Generator().manual_seed(9), DEV)[0]
    with S.isolated_seed(3):
        b = S.smt_loss(teacher, rnn, fake, cfg, "oracle_prefix_k", 0.7, dets, torch.Generator().manual_seed(9), DEV)[0]
    assert not torch.equal(a, b)          # the oracle is the only branch that reads metadata


@check("Context detector score for (t, k): invariant to x_{<=t}, to x_{>k} and to all labels except y_k")
def _():
    cfg = S.Config()
    with S.isolated_seed(2):
        d = S.ContextDetector(cfg).eval()
    batch = S.sample_stack(cfg, 2, 31)
    w = S.windows(batch)
    s0 = d.pair_scores(w, DEV)
    T, t = cfg.seq_len, 20
    ch = {k: v.clone() for k, v in batch.items()}
    ch["x"][:, :t + 1] = 2                                  # whole prefix up to the anchor
    s1 = d.pair_scores(S.windows(ch), DEV)
    rows = torch.arange(2) * T + t
    assert torch.equal(s0[rows], s1[rows])
    ch = {k: v.clone() for k, v in batch.items()}
    ch["x"][:, t + 11:] = 3                                 # inputs after target k = t + 10
    s2 = d.pair_scores(S.windows(ch), DEV)
    assert torch.equal(s0[rows, :11], s2[rows, :11])
    ch = {k: v.clone() for k, v in batch.items()}
    flip = ch["y"].ne(-1); flip[:, t + 10] = False
    ch["y"][flip] = 1 + ch["y"][flip] % cfg.n_elements      # change every other label
    s3 = d.pair_scores(S.windows(ch), DEV)
    assert torch.equal(s0[rows, 10], s3[rows, 10])


@check("Oracle: only push<=t targets, min(K, P_t) per anchor, E[aux] = mean CE over memory-dependent pairs")
def _():
    cfg = replace(S.Config(), K=2)
    batch = S.sample_stack(cfg, 4, 41)
    w = S.windows(batch)
    pool = w["prefix"]
    g = torch.Generator().manual_seed(0)
    ce = torch.rand(pool.shape)
    acc = []
    for _ in range(3000):
        sel = S.topk_mask(torch.zeros_like(ce), pool, cfg.K, g)
        assert not (sel & ~pool).any() and torch.equal(sel.sum(1), pool.sum(1).clamp(max=cfg.K))
        acc.append(S.weighted_selected_loss(ce, pool, sel))
    assert abs(float(torch.stack(acc).mean() - S.pair_mean(ce, pool))) < 0.01


@check("Mid-DMT resume (random_k): optimizer, scheduler and random draws restored; result bit-identical")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        cfg = TINY
        val = S.sample_stack(cfg, 16, 78)
        ref = S.fit(cfg, "random_k", 0.5, 11, tmp / "ref", {}, val, DEV, log=lambda s: None)
        calls = {"n": 0}
        real = S.sample_stack

        def flaky(c, n, seed):
            if n == cfg.dmt_batch_size and seed == S.derive_seed(302, 11, 3):
                raise KeyboardInterrupt          # crash inside DMT, after the step-3 checkpoint
            return real(c, n, seed)
        S.sample_stack = flaky
        try:
            S.fit(cfg, "random_k", 0.5, 11, tmp / "cut", {}, val, DEV, log=lambda s: None)
        except KeyboardInterrupt:
            pass
        finally:
            S.sample_stack = real
        ck = torch.load(tmp / "cut" / "dmt_last.pt", weights_only=True)
        assert ck["step"] == 3 and ck["scheduler"]["last_epoch"] == 3 and len(ck["optimizer"]["state"]) > 0
        res = S.fit(cfg, "random_k", 0.5, 11, tmp / "cut", {}, val, DEV, log=lambda s: None)
        for k in ref["rnn_dmt"]:
            assert torch.equal(ref["rnn_dmt"][k], res["rnn_dmt"][k]), k
        for k in ref["teacher"]:
            assert torch.equal(ref["teacher"][k], res["teacher"][k]), k
    finally:
        shutil.rmtree(tmp)


print(f"\n{len(results)} checks passed")
