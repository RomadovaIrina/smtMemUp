"""Technical checks for smt_memup_v3.py (version 3, converged regime). Run: python checks/test_smt_memup_v3.py
Small CPU models only; no experiment training. Prints one line per check."""
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
import smt_memup_v3 as V  # noqa: E402

DEV = S.setup_runtime(threads=4, device="cpu")
S.sources(ROOT)
TINY = replace(V.V3Config(), seq_len=16, max_stack_size=4, d_model=32, mem_tokens=2, encoder_depth=1, decoder_depth=1,
               rnn_depth=1, n_heads=4, batch_size=4, dmt_batch_size=4, smt_steps=6, dmt_steps=4, eval_every=3,
               eval_batch_size=8, validation_examples=16, tuning_examples=16, warmup_steps=2, local_window=4)
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


@check("Schedules: warmup_constant = linear warm-up then 1.0 (authors); cosine = the version-1 schedule")
def _():
    cfg = V.V3Config()
    f = V.lr_lambda(cfg, cfg.smt_steps, "warmup_constant")
    assert abs(f(0) - 1 / 200) < 1e-12 and f(199) == 1.0 and f(200) == 1.0 and f(9999) == 1.0
    g, h = V.lr_lambda(cfg, 6000, "cosine"), S.lr_lambda(cfg, 6000)
    assert all(g(s) == h(s) for s in (0, 100, 600, 3000, 5999))


@check("Training loop: with the version-1 schedules, V3 fit reproduces smt_memup_stack.fit bit-exactly")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        cfg = replace(TINY, smt_lr_schedule="cosine", dmt_lr_schedule="cosine", alpha=0.3)
        val = S.sample_stack(cfg, 16, 77)
        for variant in ("baseline", "surprise_k", "surprise_ctx_k"):
            a = V.fit(cfg, variant, 7, tmp / f"v3_{variant}", DETS, val, DEV, log=lambda s: None)
            b = S.fit(cfg, variant, 0.0 if variant == "baseline" else 0.3, 7, tmp / f"v1_{variant}", DETS, val, DEV,
                      log=lambda s: None)
            assert a["initial_hash"] == b["initial_hash"]
            for k in b["rnn_dmt"]:
                assert torch.equal(a["rnn_dmt"][k], b["rnn_dmt"][k]), (variant, k)
            for k in b["teacher"]:
                assert torch.equal(a["teacher"][k], b["teacher"][k]), (variant, k)
    finally:
        shutil.rmtree(tmp)


@check("Resume: a crash in the middle of SMT (warmup_constant) reproduces the uninterrupted run")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        val = S.sample_stack(TINY, 16, 78)
        ref = V.fit(TINY, "random_k", 11, tmp / "ref", DETS, val, DEV, log=lambda s: None)
        calls, real = {"n": 0}, S.sample_stack

        def flaky(c, n, seed):
            if n == TINY.batch_size and c.smt_steps == TINY.smt_steps:
                calls["n"] += 1
                if calls["n"] == 4:
                    raise KeyboardInterrupt
            return real(c, n, seed)
        S.sample_stack = flaky
        try:
            V.fit(TINY, "random_k", 11, tmp / "cut", DETS, val, DEV, log=lambda s: None)
        except KeyboardInterrupt:
            pass
        finally:
            S.sample_stack = real
        assert torch.load(tmp / "cut" / "smt_last.pt", weights_only=True)["step"] == 3
        res = V.fit(TINY, "random_k", 11, tmp / "cut", DETS, val, DEV, log=lambda s: None)
        for k in ref["rnn_dmt"]:
            assert torch.equal(ref["rnn_dmt"][k], res["rnn_dmt"][k]), k
        h = pd.read_csv(tmp / "ref" / "smt_history.csv")
        assert np.allclose(h.lr.iloc[2:], TINY.smt_lr)               # constant after the 2-step warm-up
    finally:
        shutil.rmtree(tmp)


@check("Same seed -> same initial weights for every variant; baseline trains with alpha = 0, the others with alpha = 1")
def _():
    tmp = Path(tempfile.mkdtemp(dir=ROOT / "checks"))
    try:
        val = S.sample_stack(TINY, 16, 79)
        out = {v: V.fit(TINY, v, 5, tmp / v, DETS, val, DEV, log=lambda s: None) for v in ("baseline", "uniform")}
        assert out["baseline"]["initial_hash"] == out["uniform"]["initial_hash"]
        assert out["baseline"]["identity"]["alpha"] == 0.0 and out["uniform"]["identity"]["alpha"] == 1.0
    finally:
        shutil.rmtree(tmp)


@check("Splits: the version-3 test (stream 603) is disjoint from validation, tuning and the version-1 test; new seeds")
def _():
    cfg = V.V3Config()
    fp = lambda stream, n: set(S.fingerprints(S.sample_stack(cfg, n, S.derive_seed(stream, 0))))
    t3 = fp(603, cfg.test_examples)
    assert not t3 & (fp(501, cfg.validation_examples) | fp(502, cfg.tuning_examples) | fp(503, cfg.test_examples))
    assert len(t3) == cfg.test_examples and not set(cfg.seeds) & {7, 17, 27, 107}


@check("Speed metric: first evaluation step at or above the threshold; NaN if never reached")
def _():
    h = pd.DataFrame(dict(step=[250, 500, 750, 1000], acc=[.5, np.nan, .91, .95]))
    assert V.steps_to_threshold(h, "acc", .9) == 750 and np.isnan(V.steps_to_threshold(h, "acc", .99))


@check("Version-1 detectors load unchanged for version 3")
def _():
    d = V.load_v1_detectors(V.V3Config(), ROOT, DEV)
    saved = torch.load(ROOT / "runs" / "stack02_4c69a2819ab8" / "detector_context.pt", map_location="cpu", weights_only=True)
    assert all(torch.equal(saved["best_model"][k], v) for k, v in d["context"].state_dict().items())


print(f"\n{len(results)} checks passed")
