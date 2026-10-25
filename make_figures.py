#!/usr/bin/env python3
"""All figures and summary tables for the presentation, in one style, from saved data only.

    python make_figures.py            # -> figures/*.png (+ .pdf), results_3090/summary_new_runs*.csv

Sources:
  * NEW runs (seeds 201-203): results_3090/notebook_runs/<name>_s<seed>_*/ (synced from the 3090 by sync_results.sh).
    For each (experiment, seed) the folder with summary.json is used; if none finished yet, the newest folder with a
    history (marked "running" in the legend). Folders of crashed attempts (OOM) are kept but not plotted.
  * ARCHIVED runs (old code, seed 107): archive/runs/*.
Curves are unsmoothed; validation points every eval_every steps. Markers: o = checkpoint selected by the minimum
validation CE on long answers, x = last point. Bar charts show the independent evaluation set (stream 901) at the
selected checkpoint: bar = mean over seeds, dots = individual seeds.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parent
RUNS = ROOT / "results_3090" / "notebook_runs"
ARCH = ROOT / "archive" / "runs"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

# ---- one style for every figure ------------------------------------------------------------------------------------
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 200, "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
                     "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 1.6})
COLOR = {"teacher j=0": "#1f77b4", "teacher, memory before query": "#17becf", "teacher 8 tokens, L_dec only": "#9467bd",
         "teacher, all encoder outputs": "#2ca02c", "causal Transformer": "#d62728", "RNN after DMT": "#ff7f0e",
         "RNN before DMT": "#ffbb78", "RNN full BPTT": "#8c564b", "RNN BPTT-1": "#7f7f7f", "SMT Tf=8": "#1f77b4",
         "SMT Tf=8 + ctx_far": "#17becf", "no L_dyn": "#e377c2", "target_detached": "#bcbd22", "fully detached": "#9467bd",
         "archived": "#555555"}
SEED_STYLE = {201: "-", 202: "--", 203: ":"}
FOOT_NEW = "new runs (notebook v1 code, RTX 3090), seeds 201-203; validation curves, no smoothing; o = selected checkpoint (min val CE, long answers), x = last point"
FOOT_ARCH = "ARCHIVED runs (old code, GPU, 14-20.10.2026), seed 107, validation; no smoothing; o = selected checkpoint, x = last point"


def save(fig, name, foot):
    fig.text(0.01, -0.02, foot, fontsize=7, color="#555555", ha="left", va="top")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("figure", name)


# ---- new runs: pick one folder per (experiment, seed) ---------------------------------------------------------------
def new_runs() -> dict:
    groups = {}
    for d in sorted(RUNS.glob("*_s20[0-9]_*")):                       # sorted by name = by start time per key
        m = re.match(r"(.+)_s(\d{3})_\d{8}-\d{6}_\d+$", d.name)
        if m:
            groups.setdefault((m.group(1), int(m.group(2))), []).append(d)
    found = {}
    for key, ds in groups.items():
        done = [d for d in ds if (d / "summary.json").exists()]
        live = [d for d in ds if any(d.glob("*_history.csv"))]
        if done or live:
            found[key] = dict(dir=(done or live)[-1], done=bool(done))
    return found


R = new_runs()


def history(name, seed, stage):
    r = R.get((name, seed))
    f = r and r["dir"] / f"{stage}_history.csv"
    return (pd.read_csv(f), r["done"]) if f and f.exists() else (None, False)


def curve(ax, h, col, ce, label, color, ls="-"):
    v = h.dropna(subset=[col])
    if not len(v):
        return
    ax.plot(v.step, v[col], ls=ls, color=color, label=label)
    if ce in v:
        i = v[ce].idxmin()
        ax.scatter(v.loc[i, "step"], v.loc[i, col], color=color, marker="o", s=22, zorder=3)
    ax.scatter(v.step.iloc[-1], v[col].iloc[-1], color=color, marker="x", s=28, zorder=3)


def empty(ax, text="no data yet"):
    ax.text(0.5, 0.5, text, ha="center", va="center", transform=ax.transAxes, color="#999999")


def seeds_panel(ax, specs, title, ylabel, chance=None, chance_label=None):
    """specs = [(experiment, stage, column, ce column, legend label)] plotted for every seed."""
    any_line = False
    for name, stage, col, ce, label in specs:
        for seed, ls in SEED_STYLE.items():
            h, done = history(name, seed, stage)
            if h is None or col not in h:
                continue
            curve(ax, h, col, ce, f"{label}, s{seed}{'' if done else ' (running)'}", COLOR[label], ls)
            any_line = True
    if not any_line:
        empty(ax)
    if chance is not None:
        ax.axhline(chance, color="#999999", ls=":", lw=1)
        ax.text(ax.get_xlim()[1], chance, f"{chance_label} ", fontsize=7, color="#777777", va="top", ha="right")
    ax.set_title(title); ax.set_xlabel("optimizer step"); ax.set_ylabel(ylabel); ax.set_ylim(-0.02, 1.02)
    if any_line:
        ax.legend(ncol=1, loc="lower right", frameon=False)


# ---- summary tables of the new runs ---------------------------------------------------------------------------------
rows = []
for (name, seed), r in sorted(R.items()):
    s = json.loads((r["dir"] / "summary.json").read_text()) if r["done"] else {}
    rows.append(dict(experiment=name, seed=seed, finished=r["done"], folder=r["dir"].name,
                     **{k: v for k, v in s.items() if k.startswith(("eval_", "smt_", "dmt_", "bptt_", "lm_"))}))
summary = pd.DataFrame(rows)
(ROOT / "results_3090").mkdir(exist_ok=True)
summary.to_csv(ROOT / "results_3090" / "summary_new_runs.csv", index=False)
num = summary[summary.finished].drop(columns=["folder", "finished"])
if len(num):
    agg = num.groupby("experiment").agg(["mean", "min", "max", "count"])
    agg.columns = [f"{a}__{b}" for a, b in agg.columns]
    agg.drop(columns=[c for c in agg.columns if c.startswith("seed__")]).to_csv(ROOT / "results_3090" / "summary_new_runs_agg.csv")


def bars(ax, items, title, ylabel, chance=None, chance_label=None):
    """items = [(label, experiment, summary key)] -> bar = mean over finished seeds, dots = seeds."""
    for i, (label, exp, key) in enumerate(items):
        sub = summary[(summary.experiment == exp) & summary.finished]
        vals = sub[key].dropna().tolist() if key in sub else []
        if vals:
            ax.bar(i, sum(vals) / len(vals), color=COLOR.get(label, "#888888"), alpha=0.75, width=0.7)
            ax.scatter([i] * len(vals), vals, color="black", s=12, zorder=3)
            ax.text(i, sum(vals) / len(vals) + 0.02, f"{100 * sum(vals) / len(vals):.1f}%\nn={len(vals)}", ha="center", fontsize=7)
        else:
            ax.text(i, 0.05, "pending", ha="center", fontsize=7, color="#999999", rotation=90)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items], rotation=25, ha="right")
    if chance is not None:
        ax.axhline(chance, color="#999999", ls=":", lw=1)
        ax.text(len(items) - 0.5, chance, f"{chance_label} ", fontsize=7, color="#777777", va="bottom", ha="right")
    ax.set_ylim(0, 1.12); ax.set_title(title); ax.set_ylabel(ylabel)


# ===================================================================================================================
# NEW RUNS
# ===================================================================================================================
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
seeds_panel(axes[0], [("mqar_T16N2_smt", "smt", "val_teacher_long_acc", "val_teacher_long_ce", "teacher j=0")],
            "MQAR T=16, N=2: SMT teacher, j=0 at the query", "validation accuracy, long answers (full vocab)", 0.5, "1/N among written values")
seeds_panel(axes[1], [("mqar_T16N2_smt", "smt", "val_teacher_pre_long_acc", None, "teacher, memory before query")],
            "MQAR T=16, N=2: SMT teacher, memory before the query", "validation accuracy, long answers", 0.5, "1/N")
seeds_panel(axes[2], [("mqar_T16N2_teacher_bottleneck", "smt", "val_teacher_long_acc", "val_teacher_long_ce", "teacher 8 tokens, L_dec only"),
                      ("mqar_T16N2_teacher_full", "smt", "val_teacher_long_acc", "val_teacher_long_ce", "teacher, all encoder outputs")],
            "MQAR T=16, N=2: 8 tokens vs all encoder outputs", "validation accuracy, long answers", 0.5, "1/N")
save(fig, "N1_mqar_T16_teacher_curves", FOOT_NEW)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
seeds_panel(axes[0], [("mqar_T16N2_smt", "dmt", "val_rnn_long_acc", "val_rnn_long_ce", "RNN after DMT")],
            "MQAR T=16, N=2: final RNN during DMT", "validation accuracy, long answers", 0.5, "1/N")
seeds_panel(axes[1], [("mqar_T16N2_rnn_bptt_full", "bptt", "val_rnn_long_acc", "val_rnn_long_ce", "RNN full BPTT")],
            "MQAR T=16, N=2: the same RNN, full BPTT", "validation accuracy, long answers", 0.5, "1/N")
seeds_panel(axes[2], [("mqar_T16N2_transformer", "lm", "val_lm_long_acc", "val_lm_long_ce", "causal Transformer")],
            "MQAR T=16, N=2: causal Transformer (stops at 0.99)", "validation accuracy, long answers", 0.5, "1/N")
save(fig, "N2_mqar_T16_rnn_and_baselines_curves", FOOT_NEW)

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
seeds_panel(axes[0], [("mqar_T32N4_smt", "smt", "val_teacher_long_acc", "val_teacher_long_ce", "teacher j=0"),
                      ("mqar_T32N4_smt", "smt", "val_teacher_pre_long_acc", None, "teacher, memory before query")],
            "MQAR T=32, N=4: SMT teacher", "validation accuracy, long answers", 0.25, "1/N")
seeds_panel(axes[1], [("mqar_T32N4_smt", "dmt", "val_rnn_long_acc", "val_rnn_long_ce", "RNN after DMT")],
            "MQAR T=32, N=4: final RNN during DMT", "validation accuracy, long answers", 0.25, "1/N")
seeds_panel(axes[2], [("mqar_T32N4_transformer", "lm", "val_lm_long_acc", "val_lm_long_ce", "causal Transformer")],
            "MQAR T=32, N=4: causal Transformer", "validation accuracy, long answers", 0.25, "1/N")
save(fig, "N3_mqar_T32_curves", FOOT_NEW)

fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
bars(axes[0], [("teacher j=0", "mqar_T16N2_smt", "eval_best_teacher_smt_long_acc"),
               ("teacher, memory before query", "mqar_T16N2_smt", "eval_best_teacher_pre_after_pairs_smt_long_acc"),
               ("teacher 8 tokens, L_dec only", "mqar_T16N2_teacher_bottleneck", "eval_best_teacher_smt_long_acc"),
               ("teacher, all encoder outputs", "mqar_T16N2_teacher_full", "eval_best_teacher_full_smt_long_acc"),
               ("RNN before DMT", "mqar_T16N2_smt", "eval_best_rnn_smt_long_acc"),
               ("RNN after DMT", "mqar_T16N2_smt", "eval_best_rnn_dmt_long_acc"),
               ("RNN full BPTT", "mqar_T16N2_rnn_bptt_full", "eval_best_rnn_bptt_long_acc"),
               ("causal Transformer", "mqar_T16N2_transformer", "eval_best_lm_long_acc")],
     "MQAR T=16, N=2: independent evaluation (stream 901)", "accuracy, long answers (full vocab)", 0.5, "1/N")
bars(axes[1], [("teacher j=0", "mqar_T32N4_smt", "eval_best_teacher_smt_long_acc"),
               ("teacher, memory before query", "mqar_T32N4_smt", "eval_best_teacher_pre_after_pairs_smt_long_acc"),
               ("RNN after DMT", "mqar_T32N4_smt", "eval_best_rnn_dmt_long_acc"),
               ("causal Transformer", "mqar_T32N4_transformer", "eval_best_lm_long_acc")],
     "MQAR T=32, N=4: independent evaluation (stream 901)", "accuracy, long answers (full vocab)", 0.25, "1/N")
save(fig, "N4_mqar_summary_bars", "new runs, seeds 201-203; independent evaluation set (stream 901, 1024 sequences), validation-selected checkpoint; bar = mean, dots = seeds")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
seeds_panel(axes[0], [("stack_smt_tf8", "smt", "val_teacher_long_acc", "val_teacher_long_ce", "SMT Tf=8"),
                      ("stack_smt_tf8_ctx_far", "smt", "val_teacher_long_acc", "val_teacher_long_ce", "SMT Tf=8 + ctx_far")],
            "Stack: SMT teacher, j=0", "validation accuracy, long answers (delay >= 9)", 0.25, "chance 1/4")
seeds_panel(axes[1], [("stack_smt_tf8", "dmt", "val_rnn_long_acc", "val_rnn_long_ce", "SMT Tf=8"),
                      ("stack_smt_tf8_ctx_far", "dmt", "val_rnn_long_acc", "val_rnn_long_ce", "SMT Tf=8 + ctx_far")],
            "Stack: final RNN during DMT", "validation accuracy, long answers", 0.25, "chance")
seeds_panel(axes[2], [("stack_bptt_full", "bptt", "val_rnn_long_acc", "val_rnn_long_ce", "RNN full BPTT"),
                      ("stack_bptt_1", "bptt", "val_rnn_long_acc", "val_rnn_long_ce", "RNN BPTT-1")],
            "Stack: the same RNN trained directly", "validation accuracy, long answers", 0.25, "chance")
save(fig, "N5_stack_curves", FOOT_NEW)

fig, ax = plt.subplots(figsize=(11, 4.8))
bars(ax, [("SMT Tf=8", "stack_smt_tf8", "eval_best_rnn_dmt_long_acc"),
          ("SMT Tf=8 + ctx_far", "stack_smt_tf8_ctx_far", "eval_best_rnn_dmt_long_acc"),
          ("RNN full BPTT", "stack_bptt_full", "eval_best_rnn_bptt_long_acc"),
          ("RNN BPTT-1", "stack_bptt_1", "eval_best_rnn_bptt_long_acc"),
          ("no L_dyn", "stack_smt_tf8_no_dyn", "eval_best_rnn_dmt_long_acc"),
          ("target_detached", "stack_smt_tf8_target_detached", "eval_best_rnn_dmt_long_acc"),
          ("fully detached", "stack_smt_tf8_fully_detached", "eval_best_rnn_dmt_long_acc")],
     "Stack: final recurrent model, independent evaluation (stream 901)", "accuracy, long answers (delay >= 9)", 0.25, "chance 1/4")
save(fig, "N6_stack_summary_bars", "new runs; main comparison seeds 201-203, ablations (no L_dyn, target_detached, fully detached) seed 201 only; validation-selected checkpoint")

# ===================================================================================================================
# ARCHIVED RUNS, same style
# ===================================================================================================================
D = ARCH / "stackv4_d459c6921638" / "diag"
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
for ax, (title, spec, ylabel) in zip(axes, [
        ("Stack v4 (archived): SMT teacher, j=0", [("base_window/smt_history.csv", "SMT Tf=8", "val_teacher_long_acc", "val_teacher_long_ce"),
                                                   ("ctx_far/smt_history.csv", "SMT Tf=8 + ctx_far", "val_teacher_long_acc", "val_teacher_long_ce")], "validation accuracy, long answers"),
        ("Stack v4 (archived): final RNN during DMT", [("base_window/dmt_history.csv", "SMT Tf=8", "val_rnn_long_acc", "val_rnn_long_ce"),
                                                      ("ctx_far/dmt_history.csv", "SMT Tf=8 + ctx_far", "val_rnn_long_acc", "val_rnn_long_ce")], "validation accuracy, long answers"),
        ("Stack v4 (archived): the same RNN trained directly", [("bptt_full_lr0.0003/bptt_history.csv", "RNN full BPTT", "val_rnn_long_acc", "val_rnn_long_ce"),
                                                               ("bptt_1_lr0.0003/bptt_history.csv", "RNN BPTT-1", "val_rnn_long_acc", "val_rnn_long_ce")], "validation accuracy, long answers")]):
    for f, label, col, ce in spec:
        curve(ax, pd.read_csv(D / f), col, ce, label, COLOR[label])
    ax.axhline(0.25, color="#999999", ls=":", lw=1); ax.set_title(title); ax.set_xlabel("optimizer step"); ax.set_ylabel(ylabel)
    ax.set_ylim(-0.02, 1.02); ax.legend(loc="lower right", frameon=False)
save(fig, "A1_stack_v4_archived_curves", FOOT_ARCH)

abl = {"full (= SMT Tf=8)": "8318239d86e6/ablation/full", "target_detached": "8318239d86e6/ablation/detached",
       "no L_dyn": "93364b393f72/ablation/no_dyn", "no L_unif": "7af76cfa569f/ablation/no_unif", "Tf=1": "7c073f59ff92/ablation/tf1",
       "Tf=4": "03af753e0b10/ablation/tf4", "Tf=16": "b93af0011028/ablation/tf16", "Tf=64": "8067828308c1/ablation/tf64",
       "K=4 tokens": "7ccc3a06ed73/ablation/mem4", "K=16 tokens": "4cd5be1d3493/ablation/mem16"}
fig, ax = plt.subplots(figsize=(12, 4.8))
for i, (label, p) in enumerate(abl.items()):
    h = pd.read_csv(ARCH / f"stackv4_{p}" / "dmt_history.csv").dropna(subset=["val_rnn_long_ce"])
    best, last = h.loc[h.val_rnn_long_ce.idxmin(), "val_rnn_long_acc"], h.val_rnn_long_acc.iloc[-1]
    ax.bar(i - 0.18, best, width=0.36, color=COLOR["archived"], alpha=0.8, label="selected checkpoint" if i == 0 else None)
    ax.bar(i + 0.18, last, width=0.36, color="#bbbbbb", label="last point" if i == 0 else None)
    ax.text(i - 0.18, best + 0.02, f"{100 * best:.1f}", ha="center", fontsize=7)
ax.set_xticks(range(len(abl))); ax.set_xticklabels(list(abl), rotation=25, ha="right"); ax.set_ylim(0, 1.1)
ax.axhline(0.25, color="#999999", ls=":", lw=1); ax.legend(frameon=False)
ax.set_title("Stack ablations (archived, SMT 11k / DMT 5k): final RNN"); ax.set_ylabel("validation accuracy, long answers")
save(fig, "A2_stack_ablations_archived", FOOT_ARCH + "; 'full' is the same trajectory as v4 base_window, Tf=64 = v4 full window")

ls = pd.read_csv(ROOT / "archive" / "outputs_mqar" / "table_length_sweep.csv")
fig, ax = plt.subplots(figsize=(11, 4.6))
groups = list(dict.fromkeys(zip(ls["T"], ls.pairs)))
models = list(dict.fromkeys(ls.model))
cmap = {m: c for m, c in zip(models, ["#1f77b4", "#d62728", "#8c564b"])}
for gi, (T, N) in enumerate(groups):
    for mi, m in enumerate(models):
        v = ls[(ls["T"] == T) & (ls.pairs == N) & (ls.model == m)].long_cand_acc
        if len(v):
            ax.bar(gi + (mi - 1) * 0.27, float(v.iloc[0]), width=0.27, color=cmap[m], label=m if gi == 0 else None)
    ax.hlines(1 / N, gi - 0.45, gi + 0.45, color="#999999", ls=":", lw=1)
ax.set_xticks(range(len(groups))); ax.set_xticklabels([f"T={T}, N={N}" for T, N in groups]); ax.set_ylim(0, 1.1)
ax.set_title("MQAR length sweep (archived): accuracy among written values"); ax.set_ylabel("tuning accuracy, long answers")
ax.legend(frameon=False, loc="upper right")
save(fig, "A3_mqar_length_sweep_archived", "ARCHIVED (old code), seed 107, tuning split; teacher = j=0 at the query; SMT 6000 steps with cosine LR decay; dotted = chance 1/N")
print(f"summary: {int(summary.finished.sum())} finished / {len(summary)} (experiment, seed) pairs -> results_3090/summary_new_runs*.csv")
