"""Technical checks for smt_memup_mqar.py. Run: python checks/test_smt_memup_mqar.py
Small CPU models only; no experiment training."""
import itertools
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
import smt_memup_mqar as Q  # noqa: E402

DEV = S.setup_runtime(threads=4, device="cpu")
S.sources(ROOT); Q.zoology_generator(ROOT)
M = S.sources().model
TINY = replace(Q.candidate("zoology_basic"), d_model=32, mem_tokens=2, encoder_depth=1, decoder_depth=1, rnn_depth=1,
               batch_size=4, dmt_batch_size=4, smt_steps=6, dmt_steps=4, eval_every=3, eval_batch_size=8,
               warmup_steps=2, encode_chunk=64)
results = []


def check(name):
    def deco(fn):
        fn()
        results.append(name)
        print("OK ", name, flush=True)
        return fn
    return deco


@check("Data: Zoology semantics (pairs, one query per pair, answer = written value, no noise), reproducible")
def _():
    for name in Q.CANDIDATES:
        cfg = Q.candidate(name)
        b = Q.sample_mqar(cfg, 8, 3)                                       # check_batch runs inside
        b2 = Q.sample_mqar(cfg, 8, 3)
        assert all(torch.equal(b[k], b2[k]) for k in b)
        x, y, m = b["x"], b["y"], b["target_mask"]
        bb, tt = torch.where(m)
        assert torch.equal(x[bb, b["key_pos"][bb, tt]], x[bb, tt])        # queried key equals written key
        assert torch.equal(x[bb, b["source_pos"][bb, tt]], y[bb, tt])     # answer equals written value
    st = np.random.get_state()[1].copy(); Q.sample_mqar(TINY, 4, 5)
    assert np.array_equal(st, np.random.get_state()[1])


@check("Same raw data as the direct Zoology call (only -100 -> -1)")
def _():
    gen = Q.zoology_generator()
    seg = gen(vocab_size=256, num_examples=6, input_seq_len=64, seed=11, power_a=0.01, num_kv_pairs=4,
              num_passes=1, random_non_queries=False)
    b = Q.sample_mqar(Q.candidate("zoology_basic"), 6, 11)
    assert torch.equal(b["x"], seg.inputs.long())
    lab = seg.labels.long().clone(); lab[lab == -100] = -1
    assert torch.equal(b["y"], lab)


@check("Encoder: trimmed chunked windows equal full padded windows (Stack code path)")
def _():
    cfg = replace(TINY, encode_chunk=7)
    teacher, _ = S.make_models(cfg, 3, DEV); teacher.eval()
    b = Q.sample_mqar(cfg, 3, 4)
    with torch.no_grad():
        full = S.teacher_states(teacher, b, DEV)
        mine = Q.all_states(teacher, b["x"], cfg, DEV)
    assert torch.allclose(full, mine, atol=2e-5), (full - mine).abs().max()


@check("Decoder: logits at answer positions equal Teacher.decode")
def _():
    cfg = replace(Q.candidate("zoology_64"), d_model=32, mem_tokens=2, encoder_depth=1, decoder_depth=1, rnn_depth=1)
    teacher, _ = S.make_models(cfg, 5, DEV); teacher.eval()
    b = Q.sample_mqar(cfg, 2, 6)
    w = S.windows(b)
    with torch.no_grad():
        z = teacher.encode(w["x_ctx"][:20])
        full = teacher.decode(z, w["x_fut"][:20])
        h = Q.decoder_hidden(teacher, z, w["x_fut"][:20])
        r, c = torch.where(w["valid"][:20])
        assert torch.allclose(teacher.to_logits(h[r, teacher.cfg.mem_tokens - 1 + c]), full[r, c], atol=1e-5)


@check("SMT loss with all anchors = smt_memup_stack.smt_loss (MQAR and Stack batches, all deterministic variants)")
def _():
    # K = N makes the oracle select its whole pool (no random draw), so both code paths are deterministic;
    # random selections are checked through their expectation in the next check
    cfg = replace(TINY, K=TINY.num_kv_pairs)
    teacher, rnn = S.make_models(cfg, 7, DEV)
    with S.isolated_seed(1):
        dets = {"window": S.WindowDetector(TINY).eval().requires_grad_(False),
                "context": S.ContextDetector(TINY).eval().requires_grad_(False)}
    stack_cfg = replace(S.Config(), seq_len=16, max_stack_size=4, d_model=32, mem_tokens=2, encoder_depth=1,
                        decoder_depth=1, rnn_depth=1)
    for batch in (Q.sample_mqar(TINY, 4, 8), S.sample_stack(stack_cfg, 4, 8)):
        for v in ("baseline", "uniform", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"):
            for alpha in (0.0, 0.7):
                with S.isolated_seed(2):
                    ref, rp = S.smt_loss(teacher, rnn, batch, cfg, v, alpha, dets, torch.Generator().manual_seed(0), DEV)
                with S.isolated_seed(2):
                    new, np_ = Q.smt_loss(teacher, rnn, batch, cfg, v, alpha, dets, torch.Generator().manual_seed(0), DEV)
                assert torch.allclose(ref, new, atol=1e-4), (v, alpha, float(ref), float(new))
                for k in ("decoder", "dynamics", "uniformity", "auxiliary"):
                    assert torch.allclose(rp[k], np_[k], atol=1e-4), (v, k)


@check("Random-K: E[L_selected] = L_dec; uniform = (1+alpha) L_dec; alpha=0 identical for all variants")
def _():
    teacher, rnn = S.make_models(TINY, 9, DEV)
    dets = {"window": S.WindowDetector(TINY).eval(), "context": S.ContextDetector(TINY).eval()}
    batch = Q.sample_mqar(TINY, 4, 12)
    vals = []
    for v in Q.ALL_VARIANTS:
        with S.isolated_seed(1):
            vals.append(Q.smt_loss(teacher, rnn, batch, TINY, v, 0.0, dets, torch.Generator().manual_seed(0), DEV)[0])
    assert all(torch.allclose(vals[0], x, atol=1e-6) for x in vals)
    with S.isolated_seed(1):
        lu, pu = Q.smt_loss(teacher, rnn, batch, TINY, "uniform", 0.4, dets, torch.Generator().manual_seed(0), DEV)
    assert torch.allclose(lu, vals[0] + 0.4 * pu["decoder"], atol=1e-6)
    with torch.no_grad():
        acc = []
        g = torch.Generator().manual_seed(1)
        for _ in range(400):
            with S.isolated_seed(1):
                _, p = Q.smt_loss(teacher, rnn, batch, TINY, "random_k", 1.0, dets, g, DEV)
            acc.append(p["auxiliary"])
        assert abs(float(torch.stack(acc).mean() - pu["decoder"])) < 0.02


@check("Anchor subset: only the requested windows are encoded; A = T reproduces the all-anchor loss")
def _():
    batch = Q.sample_mqar(TINY, 4, 13)
    teacher, rnn = S.make_models(TINY, 10, DEV)
    with S.isolated_seed(1):
        a, _ = Q.smt_loss(teacher, rnn, batch, TINY, "baseline", 0, {}, torch.Generator().manual_seed(0), DEV)
    with S.isolated_seed(1):
        b, _ = Q.smt_loss(teacher, rnn, batch, replace(TINY, anchors_per_sequence=TINY.seq_len), "baseline", 0, {},
                          torch.Generator().manual_seed(0), DEV)
    assert torch.allclose(a, b, atol=1e-6)
    sub = replace(TINY, anchors_per_sequence=5)
    calls = []
    real = Q.encode_anchors

    def spy(teacher_, x, b_idx, t_idx, chunk, device):
        calls.append(len(t_idx))
        return real(teacher_, x, b_idx, t_idx, chunk, device)
    Q.encode_anchors = spy
    try:
        with S.isolated_seed(1):
            loss, _ = Q.smt_loss(teacher, rnn, batch, sub, "random_k", 0.5, {}, torch.Generator().manual_seed(0), DEV)
    finally:
        Q.encode_anchors = real
    assert calls[0] <= 4 * 2 * 5 and torch.isfinite(loss)


@check("Gradients: SMT reaches encoder and updater, not detectors; DMT readout never reaches updater; truncated DMT")
def _():
    teacher, rnn = S.make_models(TINY, 6, DEV)
    dets = {"window": S.WindowDetector(TINY).eval().requires_grad_(False),
            "context": S.ContextDetector(TINY).eval().requires_grad_(False)}
    batch = Q.sample_mqar(TINY, 4, 14)
    loss, _ = Q.smt_loss(teacher, rnn, batch, TINY, "surprise_ctx_k", 0.5, dets, torch.Generator().manual_seed(0), DEV)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in teacher.encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in rnn.rnn.parameters())
    assert all(p.grad is None for d in dets.values() for p in d.parameters())
    for cfg in (TINY, replace(TINY, dmt_rollout=10)):
        teacher.zero_grad(set_to_none=True); rnn.zero_grad(set_to_none=True)
        teacher.requires_grad_(False)
        loss, parts = Q.dmt_loss(teacher, rnn, batch, cfg, DEV, torch.Generator().manual_seed(3))
        leak = torch.autograd.grad(parts["readout"], list(rnn.rnn.parameters()) + list(rnn.embed.parameters()),
                                   allow_unused=True, retain_graph=True)
        assert all(g is None for g in leak)
        loss.backward()
        assert all(p.grad is None for p in teacher.parameters())
        teacher.requires_grad_(True)
    # full-sequence DMT equals the Stack DMT objective on the same batch
    teacher.requires_grad_(False)
    a, _ = S.dmt_loss(teacher, rnn, batch, TINY, DEV)
    b, _ = Q.dmt_loss(teacher, rnn, batch, TINY, DEV)
    assert torch.allclose(a, b, atol=1e-4)


@check("Selector uses no metadata: scrambling source/delay/key positions changes only the oracle")
def _():
    teacher, rnn = S.make_models(TINY, 12, DEV)
    with S.isolated_seed(5):
        dets = {"window": S.WindowDetector(TINY).eval(), "context": S.ContextDetector(TINY).eval()}
    batch = Q.sample_mqar(TINY, 4, 21)
    fake = dict(batch)
    g = torch.Generator().manual_seed(1)
    for k in ("source_pos", "delay", "key_pos"):
        fake[k] = torch.randint(0, TINY.seq_len, batch["x"].shape, generator=g)
    for v in ("baseline", "uniform", "random_k", "surprise_k", "surprise_ctx_k"):
        out = []
        for b in (batch, fake):
            with S.isolated_seed(3):
                out.append(Q.smt_loss(teacher, rnn, b, TINY, v, 0.7, dets, torch.Generator().manual_seed(9), DEV)[0])
        assert torch.equal(out[0], out[1]), v


@check("Resume: crash in SMT and in DMT reproduces the uninterrupted run bit-exactly; identity mismatch refused")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        cfg = TINY
        val = Q.sample_mqar(cfg, 16, 77)
        ref = Q.fit(cfg, "random_k", 0.5, 11, tmp / "ref", {}, val, DEV, log=lambda s: None)
        for crash_stream, crash_step in ((301, 3), (302, 3)):
            real = Q.sample_mqar

            def flaky(c, n, seed, stream=crash_stream, st=crash_step):
                if seed == S.derive_seed(stream, 11, st):
                    raise KeyboardInterrupt
                return real(c, n, seed)
            out = tmp / f"cut{crash_stream}"
            Q.sample_mqar = flaky
            try:
                Q.fit(cfg, "random_k", 0.5, 11, out, {}, val, DEV, log=lambda s: None)
            except KeyboardInterrupt:
                pass
            finally:
                Q.sample_mqar = real
            res = Q.fit(cfg, "random_k", 0.5, 11, out, {}, val, DEV, log=lambda s: None)
            for k in ref["rnn_dmt"]:
                assert torch.equal(ref["rnn_dmt"][k], res["rnn_dmt"][k]), (crash_stream, k)
        try:
            Q.fit(cfg, "random_k", 1.0, 11, tmp / "ref", {}, val, DEV, log=lambda s: None)
            raise RuntimeError("identity mismatch not detected")
        except AssertionError:
            pass
    finally:
        shutil.rmtree(tmp)


@check("Evaluation, retention probe, idealised-detector selection audit run and are consistent")
def _():
    teacher, rnn = S.make_models(TINY, 8, DEV)
    data = Q.sample_mqar(TINY, 12, 5)
    ev = Q.evaluate(teacher, rnn, data, TINY, DEV, seed=1, variant="baseline", stage="dmt", split="x")
    assert len(ev) == 2 * int(data["target_mask"].sum()) and np.isfinite(ev.ce).all()
    assert (ev[ev.model == "teacher"].memory_mse == 0).all()
    pr = Q.retention_probe(teacher, rnn, data, TINY, DEV, horizons=(0, 3))
    h0 = pr[pr.horizon == 0].sort_values(["model", "sequence", "t"])
    assert np.array_equal(h0[h0.model == "teacher"].correct.to_numpy(), h0[h0.model == "rnn_from_teacher"].correct.to_numpy())
    cfg = Q.candidate("zoology_basic")
    sel = S.selection_audit(Q.sample_mqar(cfg, 16, 3), cfg, {"window": Q.IdealWindow(cfg), "context": Q.IdealContext(cfg)}, DEV)
    o = sel[(sel.rule == "oracle_prefix_k") & (sel.anchors == "all anchors")].iloc[0]
    assert o.memory_dependent == 1.0 and o.future_source == 0.0
    wc = Q.work_counts(replace(cfg, batch_size=4))
    assert wc["answer_logits"] > 0 and wc["encoder_windows"] == 4 * cfg.seq_len



@check("Window r of every candidate follows the pre-set rule (smallest grid r with the pair visible for >= half of answers)")
def _():
    for name in Q.CANDIDATES:
        cfg = Q.candidate(name)
        b = Q.sample_mqar(cfg, 256, S.derive_seed(512, 0))
        assert Q.choose_window(b) == cfg.local_window == cfg.long_delay, (name, Q.choose_window(b), cfg.local_window)


print(f"\n{len(results)} checks passed")
