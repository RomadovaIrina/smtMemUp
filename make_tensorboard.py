#!/usr/bin/env python3
"""All saved training histories -> TensorBoard, plus loss curves, bar charts and box plots as figures.

    python make_tensorboard.py                 # -> tb_logs/ (event files) and figures_tb/*.png
    tensorboard --logdir tb_logs               # open http://localhost:6006

Runs (TensorBoard "run" = one training):
  new/<task>/<experiment>/s<seed>   results_3090/notebook_runs (RTX 3090, seeds 201-203); unfinished runs end in "_не_завершён"
  old/stack_v2/<variant>/s<seed>    archive/runs/stackwin_26f6a039d62c (SMT 3k / DMT 1,5k, seeds 7/17/27)
  old/stack_v4_diag/<variant>       archive/runs/stackv4_d459c6921638/diag (seed 107)
  old/stack_ablation/<name>         archive/runs/stackv4_*/ablation (seed 107)
Tags: <stage>/train/<loss component> every step, <stage>/val/<metric> at every evaluation, <stage>/opt/{lr,grad_norm}.
Stages: SMT, DMT, BPTT, LM. The x-axis is the optimiser step of that stage. Nothing is smoothed when written; the
TensorBoard smoothing slider only changes the display. Figures (Images tab, run "summary"): loss curves, bars, box plots.
Crashed attempts (OOM, no summary.json, not the newest folder) are skipped; they are not results.
"""
from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

ROOT = Path(__file__).resolve().parent
NEW = ROOT / "results_3090" / "notebook_runs"
OLD = ROOT / "archive" / "runs"
TB = ROOT / "tb_logs"
FIG = ROOT / "figures_tb"

TRAIN = {"loss": "total", "decoder": "L_near", "auxiliary": "L_far", "dynamics": "L_dyn", "uniformity": "L_unif", "readout": "CE"}
DMT_TRAIN = {"loss": "total", "dynamics": "MSE (x10 в total)", "readout": "CE"}
VAL = {"val_teacher_long_acc": "teacher acc, дальние", "val_teacher_long_ce": "teacher CE, дальние", "val_teacher_all_acc": "teacher acc, все",
       "val_teacher_long_cand_acc": "teacher acc среди записанных", "val_teacher_pre_long_acc": "teacher до запросов acc, дальние",
       "val_rnn_long_acc": "RNN acc, дальние", "val_rnn_long_ce": "RNN CE, дальние", "val_rnn_all_acc": "RNN acc, все",
       "val_rnn_long_cand_acc": "RNN acc среди записанных", "val_far_memory_acc": "far-memory probe teacher",
       "val_lm_long_acc": "Transformer acc, дальние", "val_lm_long_ce": "Transformer CE, дальние", "val_lm_long_cand_acc": "Transformer acc среди записанных"}
NAMES = {"stack_smt_tf8": "SMT: короткое окно, без дальних целей", "stack_smt_tf8_ctx_far": "SMT + дальние цели по ошибке без памяти",
         "stack_smt_tf8_surprise_far": "SMT + дальние цели по локальной неожиданности", "stack_smt_tf8_random_far": "SMT + случайные дальние цели",
         "stack_smt_tf8_no_dyn": "SMT без L_dyn", "stack_smt_tf8_target_detached": "SMT, цель L_dyn отцеплена",
         "stack_smt_tf8_fully_detached": "SMT, L_dyn полностью отцеплен", "stack_bptt_full": "RNN, полный BPTT", "stack_bptt_1": "RNN, BPTT-1",
         "mqar_T16N2_smt": "SMT → DMT", "mqar_T16N2_teacher_bottleneck": "teacher 8 токенов, только L_dec",
         "mqar_T16N2_teacher_full": "диагностика: все выходы энкодера", "mqar_T16N2_transformer": "causal Transformer",
         "mqar_T16N2_rnn_bptt_full": "RNN, полный BPTT, 6000 шагов", "mqar_T16N2_rnn_bptt_full_long": "RNN, полный BPTT, до 60000 шагов",
         "mqar_T32N4_smt": "SMT → DMT", "mqar_T32N4_transformer": "causal Transformer", "mqar_T32N4_rnn_bptt_full_long": "RNN, полный BPTT, до 60000 шагов"}
STAGES = {"smt": "SMT", "dmt": "DMT", "bptt": "BPTT", "lm": "LM"}
COL = ["#1f77b4", "#17becf", "#7f7f7f", "#e377c2", "#bcbd22", "#9467bd", "#8c564b", "#ff7f0e", "#d62728", "#2ca02c"]

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.family": "Arial", "font.size": 12,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddddd", "savefig.dpi": 150})


TRAIN_STRIDE = 10                                   # training losses: every 10th step (TensorBoard shows <= 1000 points anyway)


def write_history(writer, csv, stage, info=""):
    h = pd.read_csv(csv)
    train_map = DMT_TRAIN if stage == "dmt" else TRAIN
    S = STAGES[stage]
    ht = h.iloc[::TRAIN_STRIDE]
    for col in h.columns:
        key = col[6:] if col.startswith("train_") else col
        if key in train_map:
            for st, v in zip(ht.step.tolist(), ht[col].tolist()):
                if v == v: writer.add_scalar(f"{S}/train/{train_map[key]}", v, st)
        elif col in VAL:
            hv = h[["step", col]].dropna()
            for st, v in zip(hv.step.tolist(), hv[col].tolist()):
                writer.add_scalar(f"{S}/val/{VAL[col]}", v, st)
        elif col in ("lr", "grad_norm"):
            for st, v in zip(ht.step.tolist(), ht[col].tolist()):
                writer.add_scalar(f"{S}/opt/{col}", v, st)
    if info:
        writer.add_text("описание", info)


def new_runs():
    groups = {}
    for d in sorted(NEW.glob("*_s20[0-9]_*")):
        m = re.match(r"(.+)_s(\d{3})_\d{8}-\d{6}_\d+$", d.name)
        if m: groups.setdefault((m.group(1), int(m.group(2))), []).append(d)
    out = {}
    for key, ds in groups.items():
        done = [d for d in ds if (d / "summary.json").exists()]
        out[key] = ((done or ds)[-1], bool(done))
    return out


def main():
    shutil.rmtree(TB, ignore_errors=True); FIG.mkdir(exist_ok=True)
    n = 0
    R = new_runs()
    for (name, seed), (d, done) in sorted(R.items()):
        task = "stack" if name.startswith("stack") else name.split("_")[1]
        run = TB / "new" / task / name / (f"s{seed}" + ("" if done else "_не_завершён"))
        cfg = json.loads((d / "config.json").read_text())
        info = (f"**{NAMES.get(name, name)}** · seed {seed} · {'завершён' if done else 'НЕ завершён'}  \n"
                f"family {cfg['family']}; источник `{d.relative_to(ROOT).as_posix()}`  \n"
                f"validation: поток {'501' if task == 'stack' else '511'}, 512 последовательностей")
        if (name, seed) in {("stack_smt_tf8", 203), ("stack_smt_tf8_ctx_far", 201)}:
            info += "  \nистория SMT сохранилась только до шага 6500 (сбой синхронизации); итоговые числа не затронуты"
        with SummaryWriter(run) as w:
            for stage in STAGES:
                f = d / f"{stage}_history.csv"
                if f.exists(): write_history(w, f, stage, info)
            if done:
                s = json.loads((d / "summary.json").read_text())
                for k, v in s.items():
                    if k.startswith("eval_best_") and k.endswith(("long_acc", "long_cand_acc")):
                        w.add_scalar(f"независимый набор (поток 901)/{k[10:]}", float(v), 0)
        n += 1
    for seed_dir in sorted((OLD / "stackwin_26f6a039d62c" / "final").glob("seed*")):
        for v in sorted(seed_dir.iterdir()):
            with SummaryWriter(TB / "old" / "stack_v2" / v.name / f"s{seed_dir.name[4:]}") as w:
                for stage in ("smt", "dmt"):
                    if (v / f"{stage}_history.csv").exists():
                        write_history(w, v / f"{stage}_history.csv", stage, f"СТАРЫЙ протокол v2: {v.name}, seed {seed_dir.name[4:]}, SMT 3000 / DMT 1500")
            n += 1
    for v in sorted((OLD / "stackv4_d459c6921638" / "diag").iterdir()):
        with SummaryWriter(TB / "old" / "stack_v4_diag" / v.name) as w:
            for stage in ("smt", "dmt", "bptt"):
                if (v / f"{stage}_history.csv").exists():
                    write_history(w, v / f"{stage}_history.csv", stage, f"СТАРЫЙ протокол v4-диагностика: {v.name}, seed 107")
        n += 1
    for v in sorted(OLD.glob("stackv4_*/ablation/*")):
        with SummaryWriter(TB / "old" / "stack_ablation" / v.name) as w:
            for stage in ("smt", "dmt"):
                if (v / f"{stage}_history.csv").exists():
                    write_history(w, v / f"{stage}_history.csv", stage, f"СТАРЫЙ протокол: абляция {v.name}, seed 107, SMT 11000 / DMT 5000")
        n += 1
    figs = make_figures(R)
    with SummaryWriter(TB / "summary") as w:
        for i, (name, fig) in enumerate(figs):
            w.add_figure(name, fig, 0)
    print(f"{n} runs -> {TB.relative_to(ROOT)}; figures -> {FIG.relative_to(ROOT)}: {', '.join(f for f, _ in figs)}")


# ---------------------------------------------------------------------------------------------------------------------
def hist(R, name, seed, stage):
    d, done = R.get((name, seed), (None, False))
    f = d and d / f"{stage}_history.csv"
    return pd.read_csv(f) if done and f.exists() else None


def loss_panel(ax, R, name, stage, cols, title, window=200):
    """Raw per-step training loss (faint) and its running median over `window` steps (solid), seeds 201-203."""
    for k, (col, lab) in enumerate(cols):
        for s, ls in ((201, "-"), (202, "--"), (203, ":")):
            h = hist(R, name, s, stage)
            if h is None or col not in h: continue
            ax.plot(h.step, h[col], color=COL[k], alpha=0.12, lw=0.6)
            ax.plot(h.step, h[col].rolling(window, min_periods=1).median(), color=COL[k], ls=ls, lw=1.6,
                    label=f"{lab}" if s == 201 else None)
    ax.set_title(title, loc="left", fontsize=12); ax.set_xlabel(f"шаг {STAGES[stage]}")
    ax.legend(frameon=False, fontsize=10)


def seed_values(R, name, key):
    out = []
    for s in (201, 202, 203):
        d, done = R.get((name, s), (None, False))
        if done:
            v = json.loads((d / "summary.json").read_text()).get(key)
            if v is not None: out.append(100 * v)
    return out


def box_panel(ax, items, title, chance=None):
    """Box plot over seeds + the individual seeds as dots; n is printed, n < 3 is not boxed."""
    for i, (lab, vals) in enumerate(items):
        if len(vals) >= 3:
            ax.boxplot([vals], positions=[i], widths=0.5, patch_artist=True, showfliers=False,
                       boxprops=dict(facecolor="#e8eef7", edgecolor="black"), medianprops=dict(color="black"))
        ax.scatter([i] * len(vals), vals, color="black", s=22, zorder=3)
        ax.text(i, 104, f"n={len(vals)}", ha="center", fontsize=10, color="#444444")
    if chance: ax.axhline(chance, color="#888888", ls=":", lw=1.2)
    ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items], rotation=30, ha="right", fontsize=10.5)
    ax.set_ylim(0, 110); ax.set_ylabel("accuracy, %"); ax.set_title(title, loc="left", fontsize=12)


def make_figures(R):
    figs = []
    note = ("НОВЫЕ запуски RTX 3090, обучающие батчи; бледно — значение на каждом шаге, линия — скользящая медиана 200 шагов; "
            "— seed 201, – – 202, ··· 203. Разные функции потерь по абсолютной шкале не сравнивать.")
    # loss curves, Stack
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    loss_panel(axes[0], R, "stack_smt_tf8", "smt", [("train_decoder", "L_near"), ("train_dynamics", "L_dyn")],
               "Stack, SMT: L_near и L_dyn (total < 0 из-за L_unif)")
    loss_panel(axes[1], R, "stack_smt_tf8", "dmt", [("train_dynamics", "MSE"), ("train_readout", "CE")], "Stack, DMT: MSE и CE")
    loss_panel(axes[2], R, "stack_bptt_full", "bptt", [("train_loss", "CE")], "Stack, полный BPTT: CE")
    axes[0].set_yscale("log"); axes[1].set_yscale("log"); axes[2].set_yscale("log")
    fig.text(0.005, 0.005, note, fontsize=10, color="#333333"); fig.subplots_adjust(bottom=0.2, wspace=0.25)
    fig.savefig(FIG / "L1_stack_losses.png"); figs.append(("L1_stack_losses", fig))
    # loss curves, MQAR T16
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    loss_panel(axes[0], R, "mqar_T16N2_smt", "smt", [("train_decoder", "L_near"), ("train_dynamics", "L_dyn")], "MQAR T=16, SMT: компоненты loss", 50)
    loss_panel(axes[1], R, "mqar_T16N2_smt", "dmt", [("train_dynamics", "MSE"), ("train_readout", "CE")], "MQAR T=16, DMT: MSE и CE", 50)
    loss_panel(axes[2], R, "mqar_T16N2_rnn_bptt_full_long", "bptt", [("train_loss", "CE")], "MQAR T=16, полный BPTT до 60k: CE", 500)
    for a in axes: a.set_yscale("log")
    fig.text(0.005, 0.005, note.replace("200 шагов", "50 шагов (BPTT: 500)"), fontsize=10, color="#333333"); fig.subplots_adjust(bottom=0.2, wspace=0.25)
    fig.savefig(FIG / "L2_mqar_T16_losses.png"); figs.append(("L2_mqar_T16_losses", fig))
    # diagnostics: gradient norm before clipping (clip = 1) and L_dyn around the MQAR teacher jump
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    for name, s, lab, c in (("stack_smt_tf8", 201, "без дальних целей", COL[0]), ("stack_smt_tf8_random_far", 201, "+ случайные", COL[2]),
                            ("stack_smt_tf8_surprise_far", 202, "+ локальная неожиданность (s202)", COL[3]),
                            ("stack_smt_tf8_ctx_far", 202, "+ ошибка без памяти (s202)", COL[1])):
        h = hist(R, name, s, "smt")
        if h is not None:
            axes[0].plot(h.step, h.grad_norm.rolling(200, min_periods=1).median(), color=c, label=lab, lw=2)
    axes[0].axhline(1.0, color="black", ls="--", lw=1.2); axes[0].text(200, 1.15, "порог обрезки clip = 1", fontsize=12)
    axes[0].set_yscale("log"); axes[0].set_xlabel("шаг SMT"); axes[0].set_ylabel("норма градиента до обрезки")
    axes[0].set_title("(а) Stack, SMT: норма градиента (медиана 200 шагов)", loc="left", fontsize=12); axes[0].legend(frameon=False, fontsize=11)
    for s, ls in ((201, "-"), (202, "--"), (203, ":")):
        h = hist(R, "mqar_T16N2_smt", s, "smt")
        if h is None: continue
        axes[1].plot(h.step, h.train_dynamics.rolling(50, min_periods=1).median(), color=COL[1], ls=ls, lw=2, label=f"L_dyn, s{s}")
        axes[1].plot(h.step, h.train_decoder.rolling(50, min_periods=1).median(), color=COL[0], ls=ls, lw=1.5, label=f"L_near, s{s}")
    axes[1].set_yscale("log"); axes[1].set_xlabel("шаг SMT"); axes[1].legend(frameon=False, fontsize=10, ncol=2)
    axes[1].set_title("(б) MQAR T=16, SMT: L_dyn растёт после скачка teacher", loc="left", fontsize=12)
    fig.text(0.005, 0.01, "НОВЫЕ запуски RTX 3090, обучающие батчи, скользящая медиана. grad_norm записан ДО обрезки "
                          "(clip_grad_norm_ возвращает исходную норму).", fontsize=10, color="#333333")
    fig.subplots_adjust(bottom=0.17, wspace=0.22)
    fig.savefig(FIG / "G1_grad_norm_and_ldyn.png"); figs.append(("G1_grad_norm_and_ldyn", fig))
    # bars + box plots over seeds
    SHORT = {"stack_smt_tf8": "SMT, короткое окно", "stack_smt_tf8_ctx_far": "+ ошибка без памяти", "stack_smt_tf8_random_far": "+ случайные",
             "stack_smt_tf8_surprise_far": "+ локальн. неожид.", "stack_smt_tf8_no_dyn": "без L_dyn", "stack_smt_tf8_target_detached": "цель L_dyn отцеплена",
             "stack_smt_tf8_fully_detached": "L_dyn полностью отцеплен", "stack_bptt_full": "полный BPTT", "stack_bptt_1": "BPTT-1"}
    stack_items = [(SHORT[n], n, k) for n, k in (("stack_smt_tf8", "eval_best_rnn_dmt_long_acc"), ("stack_smt_tf8_ctx_far", "eval_best_rnn_dmt_long_acc"),
                   ("stack_smt_tf8_random_far", "eval_best_rnn_dmt_long_acc"), ("stack_smt_tf8_surprise_far", "eval_best_rnn_dmt_long_acc"),
                   ("stack_smt_tf8_no_dyn", "eval_best_rnn_dmt_long_acc"), ("stack_smt_tf8_target_detached", "eval_best_rnn_dmt_long_acc"),
                   ("stack_smt_tf8_fully_detached", "eval_best_rnn_dmt_long_acc"), ("stack_bptt_full", "eval_best_rnn_bptt_long_acc"),
                   ("stack_bptt_1", "eval_best_rnn_bptt_long_acc"))]
    mq_items = [(lab, n, k) for lab, n, k in (("Transformer", "mqar_T16N2_transformer", "eval_best_lm_long_acc"),
                ("teacher j=0", "mqar_T16N2_smt", "eval_best_teacher_smt_long_acc"),
                ("teacher до запросов", "mqar_T16N2_smt", "eval_best_teacher_pre_after_pairs_smt_long_acc"),
                ("teacher 8 ток., только L_dec", "mqar_T16N2_teacher_bottleneck", "eval_best_teacher_smt_long_acc"),
                ("все выходы энкодера", "mqar_T16N2_teacher_full", "eval_best_teacher_full_smt_long_acc"),
                ("RNN до DMT", "mqar_T16N2_smt", "eval_best_rnn_smt_long_acc"), ("RNN после DMT", "mqar_T16N2_smt", "eval_best_rnn_dmt_long_acc"),
                ("полный BPTT 6k", "mqar_T16N2_rnn_bptt_full", "eval_best_rnn_bptt_long_acc"),
                ("полный BPTT ≤60k", "mqar_T16N2_rnn_bptt_full_long", "eval_best_rnn_bptt_long_acc"))]
    for fname, items, title, chance in (("B1_stack_box", stack_items, "Stack: итоговая RNN, независимый набор, delay ≥ 9", 25),
                                        ("B2_mqar_T16_box", mq_items, "MQAR T=16, N=2: независимый набор, весь словарь, delay ≥ 4", None)):
        fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
        vals = [(lab, seed_values(R, n, k)) for lab, n, k in items]
        ax = axes[0]
        for i, (lab, v) in enumerate(vals):
            if v:
                ax.bar(i, np.mean(v), color=COL[i % len(COL)], alpha=0.85, edgecolor="black", lw=0.5)
                ax.scatter([i] * len(v), v, color="black", s=16, zorder=3)
                ax.text(i, max(v) + 2, f"{np.mean(v):.1f}", ha="center", fontsize=10)
            else:
                ax.text(i, 3, "не готов", ha="center", rotation=90, fontsize=10, color="#666666")
        ax.set_xticks(range(len(vals))); ax.set_xticklabels([l for l, _ in vals], rotation=30, ha="right", fontsize=10.5)
        ax.set_ylim(0, 110); ax.set_ylabel("accuracy, %"); ax.set_title("среднее по готовым seeds (столбец) и seeds (точки)", loc="left", fontsize=12)
        if chance: ax.axhline(chance, color="#888888", ls=":", lw=1.2)
        box_panel(axes[1], vals, "box plot по seeds (ящик только при n = 3)", chance)
        fig.suptitle(title, x=0.01, ha="left", fontsize=13)
        fig.text(0.005, 0.005, "НОВЫЕ запуски RTX 3090, seeds 201–203, независимый набор (поток 901), чекпойнт по validation CE. "
                               "Box plot из 3 точек показывает только разброс seeds, не распределение.", fontsize=10, color="#333333")
        fig.subplots_adjust(bottom=0.36, wspace=0.18)
        fig.savefig(FIG / f"{fname}.png"); figs.append((fname, fig))
    return figs


if __name__ == "__main__":
    main()
