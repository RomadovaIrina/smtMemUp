#!/usr/bin/env python3
"""smtcore == the archived experiment code (archive/), on small fixed examples (CPU).

Every check builds the same inputs for the minimal path and for the archived function and compares the outputs:
data and labels, SMT windows, initial weights, every loss (value AND gradients), evaluation rows, and a short complete
SMT -> DMT fit (final weights and selected checkpoints). Archived files are imported, not modified.
Run:  python tests/test_equivalence.py            (about 2-4 min on a laptop CPU)
"""
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "archive"
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ARCHIVE)); sys.path.insert(0, str(ARCHIVE / "checks"))
import torch  # noqa: E402

import mqar_baselines as B  # noqa: E402
import mqar_bottleneck_test as BN  # noqa: E402
import smt_memup_mqar as M  # noqa: E402
import smt_memup_stack as S  # noqa: E402
import smt_memup_v4 as V  # noqa: E402
import smt_memup_window as W  # noqa: E402
import types  # noqa: E402

from smtcore import config as Cc, data as Cd, evaluate as Ce, losses as Cl, models as Cm, train as Ct  # noqa: E402
from smtcore import upstream as Cu, windows as Cw  # noqa: E402

C = types.SimpleNamespace(**{k: getattr(m, k) for m in (Cc, Cd, Ce, Cl, Cm, Ct, Cu, Cw) for k in dir(m) if not k.startswith("_")})

CPU = torch.device("cpu")
S.sources(ARCHIVE)
torch.set_num_threads(4)

# the same settings, written once for the minimal Config and once for the archived configs
STACK_V4 = V.V4Config()
STACK_C = C.Config()
MQAR_M = M.candidate("pilot01", seq_len=16, num_kv_pairs=2, K=1, long_delay=4, local_window=4)   # archived T16kv2
MQAR_C = C.Config(task="mqar", seq_len=16, num_kv_pairs=2, vocab=128, long_delay=4, future_window=16)


def same(a, b, what, exact=True, tol=1e-5):
    if isinstance(a, dict):
        for k in b:
            same(a[k], b[k], f"{what}[{k}]", exact, tol)
        return
    a, b = torch.as_tensor(a), torch.as_tensor(b)
    ok = torch.equal(a, b) if exact else torch.allclose(a.double(), b.double(), atol=tol, rtol=tol)
    assert ok, f"{what}: max |diff| = {(a.double() - b.double()).abs().max():.3e}"


def grads(*models):
    return {f"{i}.{n}": (p.grad.clone() if p.grad is not None else torch.zeros_like(p))
            for i, m in enumerate(models) for n, p in m.named_parameters()}


def loss_and_grads(fn, models):
    for m in models:
        m.zero_grad(set_to_none=True)
    loss, parts = fn()
    loss.backward()
    return float(loss.detach()), grads(*models)


# ---------------------------------------------------------------------------------------------------------------------
def test_stack_data_and_windows():
    a, b = C.sample_stack(STACK_C, 8, 12345), S.sample_stack(STACK_V4, 8, 12345)
    same(a, {k: b[k] for k in ("x", "y", "target_mask", "source_pos", "delay")}, "stack batch")
    wa, wb = C.windows(a), S.windows(b)
    same(wa, {k: wb[k] for k in ("x_ctx", "x_fut", "y_fut", "valid", "prefix")}, "stack windows")
    same(C.validation_set(STACK_C)["x"], S.sample_stack(STACK_V4, 512, S.derive_seed(501, 0))["x"], "validation 501")


def test_mqar_data():
    a, b = C.sample_mqar(MQAR_C, 16, 777), M.sample_mqar(MQAR_M, 16, 777)
    same(a, {k: b[k] for k in ("x", "y", "target_mask", "source_pos", "delay")}, "mqar batch")
    same(C.validation_set(MQAR_C)["x"], M.sample_mqar(MQAR_M, 512, S.derive_seed(511, 0))["x"], "validation 511")


def test_initial_weights():
    t1, r1 = C.make_models(STACK_C, 107, CPU)
    t2, r2 = S.make_models(STACK_V4, 107, CPU)
    assert S.state_hash(t1, r1) == S.state_hash(t2, r2), "Stack initial weights"
    t1, r1 = C.make_models(MQAR_C, 107, CPU)
    t2, r2 = S.make_models(MQAR_M, 107, CPU)
    assert S.state_hash(t1, r1) == S.state_hash(t2, r2), "MQAR initial weights"


def _stack_small():
    cfg_c = replace(STACK_C, seq_len=24, max_stack_size=6)
    cfg_v = replace(STACK_V4, seq_len=24, max_stack_size=6)
    return cfg_c, cfg_v, C.sample_stack(cfg_c, 4, 99)


def test_smt_loss_base_window_and_ctx_far():
    cfg_c, cfg_v, batch = _stack_small()
    det_w = W.load_v1_detectors(STACK_V4, ARCHIVE, CPU)                          # archived loader (identity-checked)
    det_c = C.load_context_detector(STACK_C, CPU)
    assert S.state_hash(det_w["context"]) == S.state_hash(det_c), "context detector weights"
    for far, variant in (("none", "base_window"), ("ctx_far", "ctx_far")):
        t, r = C.make_models(cfg_c, 7, CPU)
        with C.seeded(5):
            la, ga = loss_and_grads(lambda: C.smt_loss(t, r, batch, replace(cfg_c, far=far), CPU,
                                                       torch.Generator().manual_seed(3), det_c), (t, r))
        t, r = C.make_models(cfg_c, 7, CPU)
        with C.seeded(5):
            lb, gb = loss_and_grads(lambda: W.smt_window_loss(t, r, batch, cfg_v, variant, det_w,
                                                              torch.Generator().manual_seed(3), CPU), (t, r))
        assert la == lb, f"{variant}: loss {la} != {lb}"
        same(ga, gb, f"{variant} gradients")


def test_dyn_detach_modes():
    """dyn_detach='target' == the archived `detached` ablation patch; 'both' == authors' detach_rnn (no L_dyn grad
    into the encoder)."""
    cfg_c, cfg_v, batch = _stack_small()
    only_dyn_c = replace(cfg_c, coef_dec=0.0, coef_unif=0.0, coef_dyn=1.0)
    only_dyn_v = replace(cfg_v, coef_dec=0.0, coef_unif=0.0, coef_dyn=1.0)
    m = S.sources().model
    real = m.calc_mse_loss
    t, r = C.make_models(cfg_c, 7, CPU)
    with C.seeded(5):
        _, ga = loss_and_grads(lambda: C.smt_loss(t, r, batch, replace(only_dyn_c, dyn_detach="target"), CPU,
                                                  torch.Generator().manual_seed(3)), (t, r))
    t, r = C.make_models(cfg_c, 7, CPU)
    m.calc_mse_loss = lambda pred, target: real(pred, target.detach())      # checks/smt_memup_ablation.py, `detached`
    try:
        with C.seeded(5):
            _, gb = loss_and_grads(lambda: W.smt_window_loss(t, r, batch, only_dyn_v, "base_window", {},
                                                             torch.Generator().manual_seed(3), CPU), (t, r))
    finally:
        m.calc_mse_loss = real
    same(ga, gb, "dyn_detach=target vs archived detached ablation")
    enc = [k for k in ga if k.startswith("0.") and not k.startswith(("0.decoder", "0.to_logits"))]
    assert sum(float(ga[k].abs().sum()) for k in enc) > 0, "the archived `detached` still sends L_dyn grad to the encoder"
    t, r = C.make_models(cfg_c, 7, CPU)
    with C.seeded(5):
        _, gc = loss_and_grads(lambda: C.smt_loss(t, r, batch, replace(only_dyn_c, dyn_detach="both"), CPU,
                                                  torch.Generator().manual_seed(3)), (t, r))
    assert sum(float(gc[k].abs().sum()) for k in enc) == 0, "dyn_detach=both: no L_dyn gradient into the encoder"
    upd = [k for k in ga if k.startswith("1.rnn.")]
    same({k: ga[k] for k in upd}, {k: gc[k] for k in upd}, "updater gradient (target vs both)", exact=False, tol=1e-6)


def test_dmt_and_bptt():
    cfg_c, cfg_v, batch = _stack_small()
    for name, fc, fv in (
            ("dmt", lambda t, r: C.dmt_loss(t, r, batch, cfg_c, CPU), lambda t, r: S.dmt_loss(t, r, batch, cfg_v, CPU)),
            ("bptt_full", lambda t, r: C.bptt_loss(r, batch, cfg_c, CPU, False),
             lambda t, r: V.bptt_loss(r, batch, cfg_v, CPU, False)),
            ("bptt_1", lambda t, r: C.bptt_loss(r, batch, cfg_c, CPU, True),
             lambda t, r: V.bptt_loss(r, batch, cfg_v, CPU, True))):
        res = []
        for f in (fc, fv):
            t, r = C.make_models(cfg_c, 11, CPU)
            t.requires_grad_(name != "dmt")
            res.append(loss_and_grads(lambda: f(t, r), (r,)))
        assert res[0][0] == res[1][0], f"{name}: loss"
        same(res[0][1], res[1][1], f"{name} gradients")


def test_mqar_losses():
    batch = C.sample_mqar(MQAR_C, 4, 31)
    t, r = C.make_models(MQAR_C, 7, CPU)
    with C.seeded(5):
        la, ga = loss_and_grads(lambda: C.smt_loss(t, r, batch, MQAR_C, CPU, torch.Generator().manual_seed(3)), (t, r))
    t, r = C.make_models(MQAR_C, 7, CPU)
    with C.seeded(5):
        lb, gb = loss_and_grads(lambda: M.smt_loss(t, r, batch, MQAR_M, "baseline", 0.0, {},
                                                   torch.Generator().manual_seed(3), CPU), (t, r))
    # the archived MQAR forward trims the masked left padding of the encoder windows: equal up to float rounding
    assert abs(la - lb) < 1e-5, f"mqar smt loss {la} vs {lb}"
    same(ga, gb, "mqar smt gradients", exact=False, tol=2e-4)
    t, r = C.make_models(MQAR_C, 7, CPU)
    la, _ = loss_and_grads(lambda: C.dmt_loss(t, r, batch, MQAR_C, CPU), (r,))
    t, r = C.make_models(MQAR_C, 7, CPU)
    lb, _ = loss_and_grads(lambda: M.dmt_loss(t, r, batch, MQAR_M, CPU), (r,))
    assert abs(la - lb) < 1e-5, f"mqar dmt loss {la} vs {lb}"
    # full-access teacher == archived bottleneck test, mode full (loss as in its training loop)
    t, _ = C.make_models(MQAR_C, 7, CPU)
    la, ga = loss_and_grads(lambda: C.teacher_full_access_loss(t, batch, MQAR_C, CPU), (t,))
    t, _ = C.make_models(MQAR_C, 7, CPU)

    def archived_full():
        w = S.windows(batch)
        lg = BN.logits_for(t, w, "full", CPU)
        y = w["y_fut"]
        ce = torch.nn.functional.cross_entropy(lg.flatten(0, 1).float(), y.clamp(min=0).flatten(), reduction="none")
        loss = S.pair_mean(ce.reshape(y.shape), w["valid"])
        return loss, {}
    lb, gb = loss_and_grads(archived_full, (t,))
    assert la == lb, "full-access loss"
    same(ga, gb, "full-access gradients")
    # causal Transformer and RNN full BPTT on MQAR == mqar_baselines.py
    bc = replace(B.BaselineConfig(), seq_len=16, kv_pairs=2)
    with C.seeded(107):
        tf = C.CausalTransformer(MQAR_C)
    ref = B.make_model(bc, CPU)
    assert S.state_hash(tf) == S.state_hash(ref), "transformer init"
    assert abs(float(C.lm_loss(tf, batch, CPU)[0]) - float(B.target_loss(ref, batch, CPU))) < 1e-6, "transformer loss"
    rnn_ref = B.make_model(replace(bc, model="rnn_bptt"), CPU)
    _, r = C.make_models(MQAR_C, 107, CPU)
    assert abs(float(C.bptt_loss(r, batch, MQAR_C, CPU, False)[0]) - float(B.target_loss(rnn_ref, batch, CPU))) < 1e-6


def test_evaluation_rows():
    val = S.sample_stack(STACK_V4, 16, S.derive_seed(501, 0))
    t, r = C.make_models(STACK_C, 3, CPU)
    old = S.evaluate(t, r, val, STACK_V4, CPU)
    for kind, model in (("teacher", t), ("rnn", r)):
        new = C.answer_logits(kind, model, val, STACK_C, CPU)
        o = old[old.model == kind].reset_index(drop=True)
        same(new.correct.values, o.correct.values, f"stack {kind} correct")
        same(new.ce.values, o.ce.values, f"stack {kind} ce", exact=False, tol=1e-6)
    val = M.sample_mqar(MQAR_M, 16, S.derive_seed(511, 0))
    t, r = C.make_models(MQAR_C, 3, CPU)
    old = M.evaluate(t, r, val, MQAR_M, CPU)
    for kind, model in (("teacher", t), ("rnn", r)):
        new = C.answer_logits(kind, model, val, MQAR_C, CPU)
        o = old[old.model == kind].reset_index(drop=True)
        same(new.cand_correct.values, o.cand_correct.values, f"mqar {kind} cand_correct")
        same(new.ce.values, o.ce.values, f"mqar {kind} ce", exact=False, tol=1e-4)
    # the two pre-query read modes are aligned with the answer and never see the answer token
    for anchor in ("after_pairs", "before_query"):
        ev = C.answer_logits("teacher_pre", t, val, MQAR_C, CPU, anchor)
        assert len(ev) == int(val["target_mask"].sum())


def test_short_fit_equals_archived_fit_smt():
    """A complete tiny SMT -> DMT fit: smt_minimal.run vs smt_memup_v4.fit_smt (same data streams, seeds, optimiser,
    schedule, evaluation and checkpoint rule). Compares selected steps and final weights."""
    small = dict(seq_len=24, max_stack_size=6, smt_steps=6, dmt_steps=4, eval_every=2, validation_examples=16,
                 batch_size=8, warmup_steps=2)
    cfg_c = replace(STACK_C, **small, eval_examples=16)
    cfg_v = replace(STACK_V4, **{k: v for k, v in small.items() if k != "batch_size"}, batch_size=8, dmt_batch_size=8)
    val = S.sample_stack(cfg_v, 16, S.derive_seed(501, 0))
    with tempfile.TemporaryDirectory() as d:
        old = V.fit_smt(cfg_v, "base_window", 107, Path(d) / "old", {}, val, CPU, log=lambda s: None)
        C.run(dict(name="new", family="smt", seed=107, config={k: getattr(cfg_c, k) for k in small} | {"eval_examples": 16}),
              Path(d), CPU, log=lambda s: None)
        new_smt = torch.load(Path(d) / "new" / "smt_best.pt", weights_only=True)
        new_dmt = torch.load(Path(d) / "new" / "dmt_best.pt", weights_only=True)
    assert new_smt["step"] == old["smt_best_step"] and new_dmt["step"] == old["dmt_best_step"], "selected steps"
    same(new_smt["models"]["teacher"], old["teacher"], "teacher after SMT (best)")
    same(new_dmt["models"]["rnn"], old["rnn_dmt"], "RNN after DMT (best)")


def test_mqar_visibility_and_no_leak():
    """For every query t (answer y_t = value v) and both read modes: what the encoder and the decoder may see.
    j = 0 at the query: encoder window x_{t-T+1..t} contains every pair and the query key; decoder sees no inputs.
    pre-query (anchor a = 2N-1 or t-1): encoder window ends at a < t (all pairs, no query key); the decoder inputs
    x_{a+1..t} contain the query key and NO value token. No value token appears anywhere after the pairs."""
    for cfg in (MQAR_C, replace(MQAR_C, seq_len=32, num_kv_pairs=4, long_delay=8, future_window=32)):
        batch = C.sample_mqar(cfg, 256, 4242)
        x, y, N, T = batch["x"], batch["y"], cfg.num_kv_pairs, cfg.seq_len
        is_value = x >= cfg.vocab // 2
        assert not is_value[:, 2 * N:].any(), "a value token after the pairs would leak answers"
        assert (x[:, 1:2 * N:2] >= cfg.vocab // 2).all() and (x[:, 0:2 * N:2] < cfg.vocab // 2).all()
        w = C.windows(batch)
        b, t = torch.where(batch["target_mask"])
        assert (t >= 2 * N).all(), "queries come after all pairs"
        for a in (torch.full_like(t, 2 * N - 1), t - 1):
            rows, j = b * T + a, t - a
            assert torch.equal(w["y_fut"][rows, j], y[b, t]), "output j = t - a is aligned with the answer y_t"
            ctx = w["x_ctx"][rows]
            for i in range(len(b)):
                assert torch.equal(ctx[i, T - 1 - a[i]:], x[b[i], :a[i] + 1]), "encoder window = x_0..x_a (all pairs)"
                assert not (ctx[i] == x[b[i], t[i]]).sum() > 1, "the query key appears in the window only as its pair key"
                seen = w["x_fut"][rows[i], :j[i]]       # decoder inputs of output j: x_fut[:j] = x_{a+1..t}
                assert torch.equal(seen, x[b[i], a[i] + 1:t[i] + 1]) and seen[-1] == x[b[i], t[i]]
                assert not (seen >= cfg.vocab // 2).any(), "decoder inputs contain no value token"
        ctx0 = w["x_ctx"][b * T + t]                    # j = 0: the encoder has seen all pairs AND the query key
        for i in range(len(b)):
            assert torch.equal(ctx0[i, T - 1 - t[i]:], x[b[i], :t[i] + 1]), "encoder window = x_0..x_t"
            assert ctx0[i, -1] == x[b[i], t[i]], "the last encoder input is the query key"


TESTS = [test_stack_data_and_windows, test_mqar_visibility_and_no_leak, test_mqar_data, test_initial_weights, test_smt_loss_base_window_and_ctx_far,
         test_dyn_detach_modes, test_dmt_and_bptt, test_mqar_losses, test_evaluation_rows,
         test_short_fit_equals_archived_fit_smt]

if __name__ == "__main__":
    failed = 0
    for test in TESTS:
        tic = time.time()
        try:
            test()
            print(f"PASS {test.__name__} ({time.time() - tic:.1f}s)", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {test.__name__}: {type(e).__name__}: {e}", flush=True)
    print(f"{len(TESTS) - failed}/{len(TESTS)} passed")
    sys.exit(1 if failed else 0)
