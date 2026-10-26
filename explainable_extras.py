"""Optional checks for SMT_MemUP_EXPLAINABLE.ipynb (kept out of its main path).

check_against_v1(ns): executes the definitions of SMT_MemUP_FROM_ZERO.ipynb - the code that trained every run in
results_3090/ - and compares them with the functions of the explainable notebook (namespace ns) on identical
weights and batches: data, windows, initial weights, every SMT loss component and all gradients for the six far-target
rules, DMT, full BPTT and BPTT-1. Returns a table of maximal absolute differences.
"""
import contextlib
import io
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
V1_DEFS = {"s0-setup", "s0-seeds", "s1-config", "s2-stack", "s2-mqar", "s3-windows", "s4-model", "s4-make", "s6-losses"}


def v1_namespace():
    import nbformat
    nb = nbformat.read(ROOT / "SMT_MemUP_FROM_ZERO.ipynb", as_version=4)
    g = {"display": lambda *a, **k: None, "__name__": "__v1__"}
    for c in nb.cells:
        if c.cell_type == "code" and set(c.metadata.get("tags", [])) & V1_DEFS:
            with contextlib.redirect_stdout(io.StringIO()):                    # the v1 cells print demonstrations
                exec(compile(c.source, c.metadata["tags"][0], "exec"), g)
    return g


def check_against_v1(ns):
    import torch
    from dataclasses import replace
    g = v1_namespace()
    assert g["DEVICE"].type == "cpu"
    to_v1 = lambda c: g["Config"](task=c.task, seq_len=c.T, n_elements=c.n_elements, max_stack_size=c.max_stack_size, vocab=c.vocab,
                                  num_kv_pairs=c.N, power_a=c.power_a, long_delay=c.long_delay, d_model=c.d, mem_tokens=c.K_mem,
                                  encoder_depth=c.enc_depth, decoder_depth=c.dec_depth, rnn_depth=c.rnn_depth, n_heads=c.heads,
                                  future_window=c.T_f, far=c.far, far_K=c.K_far, alpha_far=c.alpha_far, coef_dec=c.c_dec,
                                  coef_dyn=c.c_dyn, coef_unif=c.c_unif, dyn_detach=c.dyn_detach, dmt_coef_dyn=c.c_dmt_mse)
    rows = []

    def diff(a, b):
        if isinstance(a, dict):
            return max(diff(a[k], b[k]) for k in a)
        a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
        return float((a - b).abs().max()) if a.numel() else 0.0

    def check(what, d, tol=0.0):
        rows.append(dict(check=what, max_abs_diff=d, tolerance=tol, ok=d <= tol))

    def grads(*ms):
        return {f"{i}.{n}": (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for i, m in enumerate(ms) for n, p in m.named_parameters()}

    STACK, MQ = ns["STACK"], ns["MQAR_T16"]
    keys = ("x", "y", "source_pos", "delay")
    a, b = ns["sample_stack"](STACK, 8, 12345), g["sample_stack"](to_v1(STACK), 8, 12345)
    check("Stack batch: x, y, source_pos, delay", diff({k: a[k] for k in keys}, {k: b[k] for k in keys}))
    wa, wb = ns["windows"](a), g["windows"](b)
    check("windows: x_ctx, x_fut, y_fut, valid, prefix", diff({k: wa[k] for k in ("x_ctx", "x_fut", "y_fut", "valid", "prefix")}, wb))
    a, b = ns["sample_mqar"](MQ, 16, 777), g["sample_mqar"](to_v1(MQ), 16, 777)
    check("MQAR T=16 N=2 batch", diff({k: a[k] for k in keys}, {k: b[k] for k in keys}))
    t1, r1 = ns["make_models"](STACK, 201); t2, r2 = g["make_models"](to_v1(STACK), 201)
    check("initial weights, seed 201", diff(t1.state_dict(), t2.state_dict()) + diff(r1.state_dict(), r2.state_dict()))

    small = replace(STACK, T=24, max_stack_size=6)
    bs = ns["sample_stack"](small, 4, 99)
    dets1, dets2 = ns["load_detectors"](), g["load_detectors"]()
    names = dict(L_near="decoder", L_far="auxiliary", L_dyn="dynamics", L_unif="uniformity", loss="loss")
    for far in ("none", "ctx_far", "surprise_far", "random_far", "oracle_far", "uniform_far"):
        c = replace(small, far=far)
        t1, r1 = ns["make_models"](c, 7); t2, r2 = g["make_models"](to_v1(c), 7)
        with ns["seeded"](5):
            l1, p1 = ns["smt_loss"](t1, r1, bs, c, torch.Generator().manual_seed(3), dets1)
        with g["seeded"](5):
            l2, p2 = g["smt_loss"](t2, r2, bs, to_v1(c), torch.Generator().manual_seed(3), dets2)
        l1.backward(); l2.backward()
        check(f"SMT loss components [{far}]", diff({k: p1[k].detach() for k in names}, {k: p2[v].detach() for k, v in names.items()}))
        check(f"SMT gradients, all parameters [{far}]", diff(grads(t1, r1), grads(t2, r2)))
    for dd in ("target", "both"):
        c = replace(small, dyn_detach=dd)
        t1, r1 = ns["make_models"](c, 7); t2, r2 = g["make_models"](to_v1(c), 7)
        with ns["seeded"](5):
            ns["smt_loss"](t1, r1, bs, c)[0].backward()
        with g["seeded"](5):
            g["smt_loss"](t2, r2, bs, to_v1(c))[0].backward()
        check(f"SMT gradients, dyn_detach = {dd}", diff(grads(t1, r1), grads(t2, r2)))
    t1, r1 = ns["make_models"](small, 7); t2, r2 = g["make_models"](to_v1(small), 7)
    t1.requires_grad_(False); t2.requires_grad_(False)
    l1 = ns["dmt_loss"](t1, r1, bs, small)[0]; l2 = g["dmt_loss"](t2, r2, bs, to_v1(small))[0]
    l1.backward(); l2.backward()
    check("DMT loss + gradients", abs(float(l1) - float(l2)) + diff(grads(r1), grads(r2)))
    for trunc, lab in ((False, "full BPTT"), (True, "BPTT-1")):
        _, r1 = ns["make_models"](small, 7); _, r2 = g["make_models"](to_v1(small), 7)
        l1 = ns["bptt_loss"](r1, bs, trunc)[0]; l2 = g["bptt_loss"](r2, bs, to_v1(small), trunc)[0]
        l1.backward(); l2.backward()
        check(f"{lab} loss + gradients", abs(float(l1) - float(l2)) + diff(grads(r1), grads(r2)))
    df = pd.DataFrame(rows)
    assert df.ok.all(), df[~df.ok]
    return df
