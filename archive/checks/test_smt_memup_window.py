"""Technical checks for smt_memup_window.py (version 2). Run: python checks/test_smt_memup_window.py
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
import smt_memup_window as W  # noqa: E402

DEV = S.setup_runtime(threads=4, device="cpu")
S.sources(ROOT)
TINY = replace(W.WindowConfig(), seq_len=16, max_stack_size=4, d_model=32, mem_tokens=2, encoder_depth=1,
               decoder_depth=1, rnn_depth=1, n_heads=4, batch_size=4, dmt_batch_size=4, smt_steps=6, dmt_steps=4,
               eval_every=3, eval_batch_size=8, validation_examples=16, tuning_examples=16, warmup_steps=2,
               future_window=4, far_K=2, local_window=4)
with S.isolated_seed(3):
    DETS = {"window": S.WindowDetector(TINY).eval().requires_grad_(False),
            "context": S.ContextDetector(TINY).eval().requires_grad_(False)}
results = []


def check(name):
    def deco(fn):
        fn()
        results.append(name)
        print("OK ", name, flush=True)
        return fn
    return deco


def loss_of(cfg, variant, batch, seed=0, teacher=None, rnn=None):
    if teacher is None:
        teacher, rnn = S.make_models(cfg, 5, DEV)
    with S.isolated_seed(1):
        return W.smt_window_loss(teacher, rnn, batch, cfg, variant, DETS, torch.Generator().manual_seed(seed), DEV)


@check("Full window (Tf = T): base_window equals the version-1 SMT loss exactly")
def _():
    b = S.sample_stack(TINY, 4, 11)
    cfg = replace(TINY, future_window=TINY.seq_len)
    teacher, rnn = S.make_models(cfg, 5, DEV)
    l2, _ = loss_of(cfg, "base_window", b, teacher=teacher, rnn=rnn)
    with S.isolated_seed(1):
        l1, _ = S.smt_loss(teacher, rnn, b, cfg, "baseline", 0.0, {}, torch.Generator().manual_seed(0), DEV)
    assert torch.equal(l1, l2)


@check("Window loss: only pairs with j < Tf; far pool = valid pairs with j >= Tf; the two partition all valid pairs")
def _():
    b = S.sample_stack(TINY, 4, 12)
    w = S.windows(b)
    in_win, far = W.far_pool(w, TINY, DEV)
    assert not (in_win & far).any() and torch.equal(in_win | far, w["valid"])
    j = torch.arange(TINY.seq_len)
    assert (j[None].expand_as(far)[far] >= TINY.future_window).all()
    teacher, rnn = S.make_models(TINY, 5, DEV)
    _, parts = loss_of(TINY, "base_window", b, teacher=teacher, rnn=rnn)
    logits = teacher.decode(S.encode_chunked(teacher, w["x_ctx"]), w["x_fut"])
    ce = S.sources().model.calc_cross_entropy_loss(logits, w["y_fut"].clamp(min=0))
    assert torch.allclose(parts["decoder"], S.pair_mean(ce, in_win), atol=1e-6)


@check("Normalisation: uniform_far = mean CE over far pairs; E[random_far] = uniform_far; alpha scales only L_far")
def _():
    b = S.sample_stack(TINY, 4, 13)
    teacher, rnn = S.make_models(TINY, 5, DEV)
    w = S.windows(b)
    _, far = W.far_pool(w, TINY, DEV)
    logits = teacher.decode(S.encode_chunked(teacher, w["x_ctx"]), w["x_fut"])
    ce = S.sources().model.calc_cross_entropy_loss(logits, w["y_fut"].clamp(min=0)).detach()
    _, pu = loss_of(TINY, "uniform_far", b, teacher=teacher, rnn=rnn)
    assert torch.allclose(pu["auxiliary"], S.pair_mean(ce, far), atol=1e-6)
    est = np.mean([float(loss_of(TINY, "random_far", b, seed=s, teacher=teacher, rnn=rnn)[1]["auxiliary"]) for s in range(400)])
    assert abs(est - float(pu["auxiliary"])) < 0.02 * float(pu["auxiliary"]), (est, float(pu["auxiliary"]))
    lb, pb = loss_of(TINY, "base_window", b, teacher=teacher, rnn=rnn)
    l3, p3 = loss_of(replace(TINY, alpha_far=3.0), "uniform_far", b, teacher=teacher, rnn=rnn)
    assert torch.allclose(l3 - lb, TINY.coef_dec * 3.0 * p3["auxiliary"], atol=1e-5)


@check("Selection: chosen pairs lie in the far pool, min(K, N_t) per anchor; oracle only push <= t; no metadata otherwise")
def _():
    b = S.sample_stack(TINY, 6, 14)
    w = S.windows(b)
    _, far = W.far_pool(w, TINY, DEV)
    for rule in ("random_far", "surprise_far", "ctx_far", "oracle_far"):
        pool = far & w["prefix"] if rule == "oracle_far" else far
        scores = (torch.zeros(far.shape) if rule in ("random_far", "oracle_far") else
                  S.selection_scores("surprise_k" if rule == "surprise_far" else "surprise_ctx_k", w, b, DETS, DEV))
        chosen = S.topk_mask(scores, pool, TINY.far_K, torch.Generator().manual_seed(0))
        assert not (chosen & ~pool).any() and torch.equal(chosen.sum(1), pool.sum(1).clamp(max=TINY.far_K))
    teacher, rnn = S.make_models(TINY, 5, DEV)
    scr = dict(b); g = torch.Generator().manual_seed(9)
    for k in ("source_pos", "delay", "depth"):
        scr[k] = b[k][:, torch.randperm(TINY.seq_len, generator=g)]
    for v in ("base_window", "uniform_far", "random_far", "surprise_far", "ctx_far"):
        assert torch.equal(loss_of(TINY, v, b, teacher=teacher, rnn=rnn)[0], loss_of(TINY, v, scr, teacher=teacher, rnn=rnn)[0]), v


@check("Gradients: L_far reaches decoder and encoder, not the RNN updater; detectors get none")
def _():
    b = S.sample_stack(TINY, 4, 15)
    cfg = replace(TINY, coef_dyn=0.0, coef_unif=0.0)
    teacher, rnn = S.make_models(cfg, 5, DEV)
    for v in ("surprise_far", "ctx_far"):
        teacher.zero_grad(); rnn.zero_grad()
        loss, parts = loss_of(cfg, v, b, teacher=teacher, rnn=rnn)
        (parts["auxiliary"]).backward()
        enc = [p.grad for n, p in teacher.named_parameters() if n.startswith("encoder")]
        dec = [p.grad for n, p in teacher.named_parameters() if n.startswith("decoder")]
        assert any(g is not None and g.abs().sum() > 0 for g in enc) and any(g is not None and g.abs().sum() > 0 for g in dec)
        assert all(p.grad is None or p.grad.abs().sum() == 0 for p in rnn.parameters())
        assert all(p.grad is None for d in DETS.values() for p in d.parameters())


@check("Far-memory probe: exactly the far pairs with push <= t, with consistent anchor, distance and delay")
def _():
    b = S.sample_stack(TINY, 6, 16)
    teacher, _ = S.make_models(TINY, 5, DEV)
    fp = W.far_probe(teacher, b, TINY, DEV, chunk=4)
    assert len(fp) and (fp.distance >= TINY.future_window).all() and (fp.delay >= 1).all()
    w = S.windows(b)
    _, far = W.far_pool(w, TINY, DEV)
    assert len(fp) == int((far & w["prefix"]).sum())
    assert (fp.target_t - fp.anchor == fp.distance).all() and (fp.target_t - fp.delay <= fp.anchor).all()


@check("Training: same seed -> same initial weights as version 1; mid-SMT crash reproduces the uninterrupted run")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        val = S.sample_stack(TINY, 16, 77)
        ref = W.fit(TINY, "surprise_far", 7, tmp / "ref", DETS, val, DEV, log=lambda s: None)
        t1, r1 = S.make_models(S.Config(**{k: getattr(TINY, k) for k in S.Config.__dataclass_fields__}), 7, DEV)
        assert ref["initial_hash"] == S.state_hash(t1, r1)
        calls = {"n": 0}
        real = S.sample_stack

        def flaky(c, n, seed):
            if n == TINY.batch_size and c.smt_steps == TINY.smt_steps:
                calls["n"] += 1
                if calls["n"] == 4:
                    raise KeyboardInterrupt
            return real(c, n, seed)
        S.sample_stack = flaky
        try:
            W.fit(TINY, "surprise_far", 7, tmp / "cut", DETS, val, DEV, log=lambda s: None)
        except KeyboardInterrupt:
            pass
        finally:
            S.sample_stack = real
        assert torch.load(tmp / "cut" / "smt_last.pt", weights_only=True)["step"] == 3
        res = W.fit(TINY, "surprise_far", 7, tmp / "cut", DETS, val, DEV, log=lambda s: None)
        for k in ref["rnn_dmt"]:
            assert torch.equal(ref["rnn_dmt"][k], res["rnn_dmt"][k]), k
        h = __import__("pandas").read_csv(tmp / "ref" / "smt_history.csv")
        assert "val_far_memory_acc" in h.columns and h.val_far_memory_acc.notna().any()
    finally:
        shutil.rmtree(tmp)


@check("Version-1 detectors load unchanged and are tied to the saved version-1 identity")
def _():
    d = W.load_v1_detectors(W.WindowConfig(), ROOT, DEV)
    assert set(d) == {"window", "context"} and all(not p.requires_grad for m in d.values() for p in m.parameters())
    saved = torch.load(ROOT / "runs" / W.V1_RUN / "detector_window.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(saved["best_model"][k], v) for k, v in d["window"].state_dict().items())


print(f"\n{len(results)} checks passed")
