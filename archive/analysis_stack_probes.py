#!/usr/bin/env python3
"""How do the trained models store the stack? Linear probes on memory states (analysis only; no training of the
models, no test split).

For every time step t the true stack after x_t is replayed from the inputs: depth d_t and the value at each position
from the top (1 = top ... 8 = bottom; 0 = empty). A linear (multinomial logistic) probe is fitted on the TUNING split
(stream 502) and scored on VALIDATION (stream 501) for each memory:
  * teacher m_t (encoder of the SMT runs, best SMT checkpoint);
  * RNN m_t of the SMT runs after DMT (rolled out from the zero state, i.e. the deployed model), when DMT has finished;
  * RNN m_t of the BPTT references (best checkpoint).
High probe accuracy at position k = the information about the k-th element is linearly present in the state.
A majority-class baseline is reported for every target.

  python analysis_stack_probes.py [--runs runs/stackv4_*] [--threads 2]      # -> outputs_v4/probes/
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import smt_memup_stack as S
import smt_memup_v4 as V

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v4" / "probes"
DEPTH = 8


def stack_targets(x: torch.Tensor, depth_max: int = DEPTH) -> torch.Tensor:
    """(B, T, 1 + depth_max): [depth, value at position 1 from the top, ..., position depth_max]; 0 = empty."""
    B, T = x.shape
    out = torch.zeros(B, T, 1 + depth_max, dtype=torch.long)
    for b in range(B):
        st = []
        for t in range(T):
            tok = int(x[b, t])
            if tok > 0:
                st.append(tok)
            else:
                st.pop()
            out[b, t, 0] = len(st)
            for k in range(min(len(st), depth_max)):
                out[b, t, 1 + k] = st[-1 - k]
    return out


def config_of(run: Path) -> V.V4Config:
    conf = json.loads((run / "identity.json").read_text())["config"]
    return V.V4Config(**{k: tuple(v) if isinstance(v, list) else v for k, v in conf.items()})


def load_models(run: Path, name: str, cfg: V.V4Config) -> dict:
    """Name -> memory function x -> (B, T, Z). Uses completed.pt when present, else the best checkpoints so far."""
    d = run / "diag" / name
    out = {}
    if name.startswith("bptt"):
        f = d / "completed.pt"
        if f.exists():
            rnn = V.load_bptt(cfg, torch.load(f, map_location="cpu", weights_only=True), torch.device("cpu"))
        elif (d / "bptt_best.pt").exists():
            _, rnn = S.make_models(cfg, 0, torch.device("cpu"))
            rnn.load_state_dict(torch.load(d / "bptt_best.pt", map_location="cpu", weights_only=True)["models"]["rnn"])
        else:
            return out
        out[f"{name} | RNN (BPTT)"] = lambda b, rnn=rnn.eval(): S.rnn_rollout(rnn, b["x"])
        return out
    teacher, rnn = S.make_models(cfg, 0, torch.device("cpu"))
    if (d / "completed.pt").exists():
        saved = torch.load(d / "completed.pt", map_location="cpu", weights_only=True)
        teacher.load_state_dict(saved["teacher"]); rnn.load_state_dict(saved["rnn_dmt"])
        out[f"{name} | RNN after DMT"] = lambda b, rnn=rnn.eval(): S.rnn_rollout(rnn, b["x"])
    elif (d / "smt_best.pt").exists():
        teacher.load_state_dict(torch.load(d / "smt_best.pt", map_location="cpu", weights_only=True)["models"]["teacher"])
    else:
        return out
    out[f"{name} | teacher"] = lambda b, teacher=teacher.eval(): S.teacher_states(teacher, b, torch.device("cpu"))
    return out


def fit_probe(xtr, ytr, xva, yva, classes: int, steps: int = 300) -> float:
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-5
    xtr, xva = (xtr - mu) / sd, (xva - mu) / sd
    w = torch.zeros(xtr.shape[1], classes, requires_grad=True)
    b = torch.zeros(classes, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        loss = F.cross_entropy(xtr @ w + b, ytr) + 1e-4 * w.square().sum()
        loss.backward(); opt.step()
    with torch.no_grad():
        return float(((xva @ w + b).argmax(-1) == yva).float().mean())


@torch.no_grad()
def states(fn, batch, chunk=64):
    return torch.cat([fn(S.slice_batch(batch, i, i + chunk)) for i in range(0, len(batch["x"]), chunk)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=sorted(glob.glob(str(ROOT / "runs" / "stackv4_*"))))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from table_stack_probes.csv")
    a = ap.parse_args()
    if a.figure_only:
        figure(pd.read_csv(OUT / "table_stack_probes.csv"), OUT / "figP1_stack_probes.png"); return
    torch.set_num_threads(a.threads)
    S.sources(ROOT)
    cfg0 = V.V4Config()
    tun = S.sample_stack(cfg0, cfg0.tuning_examples, S.derive_seed(502, 0))
    val = S.sample_stack(cfg0, cfg0.validation_examples, S.derive_seed(501, 0))
    ytr, yva = stack_targets(tun["x"]).flatten(0, 1), stack_targets(val["x"]).flatten(0, 1)
    rows = []
    for k in range(ytr.shape[1]):
        maj = torch.bincount(ytr[:, k]).argmax()
        rows.append(dict(model="majority class", target=k, accuracy=float((yva[:, k] == maj).float().mean())))
    for run in map(Path, a.runs):
        ident = json.loads((run / "identity.json").read_text())
        if ident.get("device") != "cuda" or not (run / "diag").exists():
            continue
        cfg = config_of(run)
        for d in sorted((run / "diag").glob("*/")):
            for label, fn in load_models(run, d.name, cfg).items():
                tic = time.time()
                xtr, xva = states(fn, tun).flatten(0, 1), states(fn, val).flatten(0, 1)
                for k in range(ytr.shape[1]):
                    classes = DEPTH + 1 if k == 0 else cfg.vocab_size
                    rows.append(dict(model=label, run=run.name, target=k,
                                     accuracy=fit_probe(xtr, ytr[:, k], xva, yva[:, k], classes)))
                print(f"{label}: {time.time() - tic:.0f}s", flush=True)
    table = pd.DataFrame(rows)
    table["target_name"] = np.where(table.target == 0, "depth", "pos " + table.target.astype(str))
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "table_stack_probes.csv", index=False)
    wide = table.pivot_table(index="model", columns="target", values="accuracy")
    wide.columns = ["depth"] + [f"pos{k}" for k in wide.columns[1:]]
    wide.round(3).to_csv(OUT / "table_stack_probes_wide.csv")
    figure(table, OUT / "figP1_stack_probes.png")
    print(wide.round(3).to_string())


def figure(table: pd.DataFrame, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_v4_outputs as M
    fig, ax = plt.subplots(figsize=(9, 5.6), constrained_layout=True)
    M.style_axes(ax, "Что линейно читается из памяти: элемент стека на позиции k от вершины (validation)",
                 "точность линейной пробы", "позиция в стеке (1 = вершина)")
    for model, g in table[table.target > 0].groupby("model"):
        variant = model.split(" | ")[0]
        variant = "bptt_full" if variant.startswith("bptt_full") else "bptt_1" if variant.startswith("bptt_1") else variant
        if variant.startswith("bptt") and not model.split(" | ")[0].endswith("0.0003"):
            continue                                      # one LR per BPTT family in the figure (all in the table)
        style = "--" if "teacher" in model else "-"
        color = M.MUTED if model == "majority class" else M.COLOR.get(variant, M.MUTED)
        ax.plot(g.target, g.accuracy, color=color, linestyle=style if model != "majority class" else ":",
                linewidth=2, marker="o", markersize=4, label=model)
    M.pct(ax)
    ax.legend(fontsize=7, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
    fig.savefig(path, dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
