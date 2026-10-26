#!/usr/bin/env python3
"""Gradient-norm decomposition of the SMT loss on saved checkpoints (CPU, no training).

    python diagnostics/grad_decomposition.py      # -> diagnostics/grad_decomposition.csv (+ printed tables)

For a checkpoint (teacher + RNN after SMT) and the TRAINING batches of the last steps of its own run, the gradient of every
loss term is computed separately, with the weight it has in the total loss:
  c_dec*L_near, c_dec*alpha*L_far, c_dyn*L_dyn, c_unif*L_unif, and the total.
The norm is taken over all trained parameters (teacher + RNN), exactly what clip_grad_norm_(.., 1.0) sees in training, and
split by module. The loss code is the one that trained the runs: the definitions of SMT_MemUP_FROM_ZERO.ipynb (v1).
Batches, the top-K tie-breaking generator and the L_unif subset use the same seed streams as training (301 / 303 / 304).
Caveat: smt_best.pt is the state AFTER step 11000, and these batches were just trained on, so norms come out lower than the
logged training norms (base 0.03 vs 0.20, ctx_far 0.80 vs 1.35); the split between terms is the point, not the absolute value.
Checkpoints: results_3090/checkpoints/<run>/smt_best.pt (copied from the GPU server; *.pt are not in git).
"""
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from explainable_extras import v1_namespace  # noqa: E402

g = v1_namespace()
STACK, smt_loss, make_models, sample, derive_seed, seeded = g["STACK"], g["smt_loss"], g["make_models"], g["sample"], g["derive_seed"], g["seeded"]
dets = g["load_detectors"](STACK.V)
RUNS = {"stack_smt_tf8_s202": ("none", 202), "stack_smt_tf8_ctx_far_s202": ("ctx_far", 202)}
STEPS = range(10990, 11000)                    # training steps 10991..11000 (0-based batch index s), the end of SMT
TERMS = {"L_near": ("decoder", "coef_dec"), "L_far": ("auxiliary", "alpha"), "L_dyn": ("dynamics", "coef_dyn"), "L_unif": ("uniformity", "coef_unif")}
GROUPS = {"teacher.embed": "0.embed", "teacher.encoder+registers": ("0.encoder", "0.encoder_mem_register"), "teacher.decoder+head": ("0.decoder", "0.to_logits"),
          "rnn.embed": "1.embed", "rnn.updater f": "1.rnn"}


def grad_vector(models, loss):
    for m in models:
        m.zero_grad(set_to_none=True)
    loss.backward(retain_graph=True)
    return {f"{i}.{n}": (p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)) for i, m in enumerate(models) for n, p in m.named_parameters()}


def norm(gr, prefix=None):
    keys = [k for k in gr if prefix is None or k.startswith(prefix if isinstance(prefix, tuple) else (prefix,))]
    return float(torch.sqrt(sum((gr[k] ** 2).sum() for k in keys))) if keys else 0.0


rows = []
for run, (far, seed) in RUNS.items():
    for far_eval in ("none", "ctx_far"):                            # also: what L_far would do on the base checkpoint
        cfg = replace(STACK, far=far_eval)
        teacher, rnn = make_models(cfg, seed)
        st = torch.load(ROOT / "results_3090/checkpoints" / run / "smt_best.pt", map_location="cpu", weights_only=True)
        teacher.load_state_dict(st["models"]["teacher"]); rnn.load_state_dict(st["models"]["rnn"])
        teacher.train(); rnn.train()
        for s in STEPS:
            batch = sample(cfg, cfg.batch_size, derive_seed(301, seed, s))
            gen = torch.Generator().manual_seed(derive_seed(303, seed, s))
            with seeded(derive_seed(304, seed, s)):
                total, parts = smt_loss(teacher, rnn, batch, cfg, gen, dets)
            coef = dict(coef_dec=cfg.coef_dec, alpha=cfg.coef_dec * cfg.alpha_far, coef_dyn=cfg.coef_dyn, coef_unif=cfg.coef_unif)
            grads = {"total": grad_vector((teacher, rnn), total)}
            for t, (key, c) in TERMS.items():
                if t == "L_far" and far_eval == "none":
                    continue
                grads[t] = grad_vector((teacher, rnn), coef[c] * parts[key])
            for t, gr in grads.items():
                rows.append(dict(checkpoint=run, loss_far=far_eval, step=s + 1, term=t,
                                 value=float(total) if t == "total" else float(parts[TERMS[t][0]]), grad_norm=norm(gr),
                                 **{grp: norm(gr, pre) for grp, pre in GROUPS.items()}))
        print("done", run, far_eval, flush=True)

df = pd.DataFrame(rows)
out = ROOT / "diagnostics" / "grad_decomposition.csv"
df.to_csv(out, index=False)
med = df.groupby(["checkpoint", "loss_far", "term"]).median(numeric_only=True).drop(columns="step").round(4)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(med)
logged = {}
for run in RUNS:
    d = next((ROOT / "results_3090/notebook_runs").glob(f"{run}_2026*"))
    h = pd.read_csv(d / "smt_history.csv")
    logged[run] = float(h[h.step.isin([s + 1 for s in STEPS])].grad_norm.median())
print("logged grad_norm (training, same steps, median):", logged)
