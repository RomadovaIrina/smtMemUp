#!/usr/bin/env python3
"""Figures and tables of the 26.10.2026 deck, built only from saved CSV / JSON.

    python deck/make_deck_figures.py        # -> deck/figures/*.png, deck/tables/*.csv

Two protocols are never mixed in one panel:
  * NEW  - results_3090/notebook_runs/ (notebook v1 code, RTX 3090, seeds 201-203; validation stream 501/511 for curves
           and checkpoint selection, independent evaluation stream 901 for the bars);
  * OLD  - archive/ (old code, seed 107 or seeds 7/17/27, validation / tuning only).
A run counts as finished only when its folder has summary.json. Unfinished runs are listed, never drawn as zero.
Curves are not smoothed and not extended past the last saved point.
"""
from __future__ import annotations

import json
import textwrap
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results_3090" / "notebook_runs"
ARCH = ROOT / "archive"
FIG = ROOT / "deck" / "figures"
TAB = ROOT / "deck" / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
                     "font.family": "Arial", "font.size": 13, "axes.titlesize": 14, "axes.labelsize": 13,
                     "legend.fontsize": 12, "xtick.labelsize": 12, "ytick.labelsize": 12, "axes.edgecolor": "black",
                     "axes.labelcolor": "black", "text.color": "black", "xtick.color": "black", "ytick.color": "black",
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#dddddd",
                     "grid.linewidth": 0.6, "lines.linewidth": 1.8, "savefig.dpi": 200})

# one Russian name and one colour per model / mode (the same in every figure and in the deck)
NAME = {"stack_smt_tf8": "SMT: короткое окно, без дальних целей",
        "stack_smt_tf8_ctx_far": "SMT + дальние цели по ошибке без памяти",
        "stack_smt_tf8_surprise_far": "SMT + дальние цели по локальной неожиданности",
        "stack_smt_tf8_random_far": "SMT + случайные дальние цели",
        "stack_smt_tf8_no_dyn": "SMT без L_dyn",
        "stack_smt_tf8_target_detached": "SMT, цель L_dyn отцеплена",
        "stack_smt_tf8_fully_detached": "SMT, L_dyn полностью отцеплен",
        "stack_bptt_full": "RNN, полный BPTT",
        "stack_bptt_1": "RNN, градиент на один переход (BPTT-1)"}
COL = {"teacher": "#1f77b4", "teacher_pre": "#17becf", "teacher_bneck": "#9467bd", "teacher_full": "#2ca02c",
       "lm": "#d62728", "rnn_dmt": "#ff7f0e", "rnn_smt": "#c49c94", "bptt_full": "#8c564b", "bptt_1": "#7f7f7f",
       "ctx_far": "#17becf", "smt": "#1f77b4", "base_window": "#1f77b4", "full_window": "#000000",
       "uniform_far": "#bcbd22", "random_far": "#7f7f7f", "surprise_far": "#e377c2", "oracle_far": "#9467bd",
       "no_dyn": "#e377c2", "detached": "#bcbd22", "no_unif": "#8c564b"}
SEED_LS = {201: "-", 202: "--", 203: ":"}


def foot(fig, text):
    width = int(fig.get_figwidth() * 11.6)                          # ~characters per line at 8.5 pt
    lines = [w for para in text.split("\n") for w in textwrap.wrap(para, width)]
    fig.text(0.005, 0.005, "\n".join(lines), fontsize=11.5, color="#333333", ha="left", va="bottom")


def legend_models_seeds(fig, models, x=0.8, y=0.85):
    """Two legends instead of one per line: colour = model, line style = seed."""
    from matplotlib.lines import Line2D
    l1 = fig.legend([Line2D([], [], color=c, lw=2.2) for _, c in models], [m for m, _ in models], loc="upper left",
                    bbox_to_anchor=(x, y), frameon=False, title="модель / режим", alignment="left")
    fig.add_artist(l1)
    fig.legend([Line2D([], [], color="black", ls=ls) for ls in SEED_LS.values()], [f"seed {s}" for s in SEED_LS],
               loc="upper left", bbox_to_anchor=(x, y - 0.06 * (len(models) + 2.2)), frameon=False, title="seed", alignment="left")


def save(fig, name, note, bottom=0.2, right=None):
    foot(fig, note)
    fig.subplots_adjust(bottom=bottom, **({"right": right} if right else {}))
    fig.savefig(FIG / f"{name}.png")
    plt.close(fig)
    print("figure", name)


# ---- NEW runs: one folder per (experiment, seed): the finished one, else the newest (reported as unfinished) ----------
def scan():
    groups = {}
    for d in sorted(RUNS.glob("*_s20[0-9]_*")):
        m = re.match(r"(.+)_s(\d{3})_\d{8}-\d{6}_\d+$", d.name)
        if m:
            groups.setdefault((m.group(1), int(m.group(2))), []).append(d)
    out = {}
    for key, ds in groups.items():
        done = [d for d in ds if (d / "summary.json").exists()]
        out[key] = dict(dir=(done or ds)[-1], done=bool(done), attempts=len(ds),
                        crashed=sum(1 for d in ds if not (d / "summary.json").exists() and d != (done or ds)[-1]))
    return out


R = scan()
TRUNCATED = {("stack_smt_tf8", 203), ("stack_smt_tf8_ctx_far", 201)}   # SMT history cut at step 6500 by a git pull (NOTES_RU.md)


def summ(name, seed):
    r = R.get((name, seed))
    return json.loads((r["dir"] / "summary.json").read_text()) if r and r["done"] else None


def hist(name, seed, stage):
    r = R.get((name, seed))
    f = r and r["dir"] / f"{stage}_history.csv"
    return pd.read_csv(f) if f and f.exists() else None


def cfg_of(name, seed):
    r = R.get((name, seed))
    return json.loads((r["dir"] / "config.json").read_text())["config"] if r else {}


# ---- status table of every NEW run (the audit table of the deck) ------------------------------------------------------
status = []
for (name, seed), r in sorted(R.items()):
    c = cfg_of(name, seed); s = summ(name, seed) or {}
    fam = json.loads((r["dir"] / "config.json").read_text())["family"]
    budget = {"smt": f"SMT {c.get('smt_steps')} / DMT {c.get('dmt_steps')}", "teacher_full": f"SMT {c.get('smt_steps')} (только teacher)",
              "bptt_full": f"BPTT {c.get('bptt_steps')}", "bptt_1": f"BPTT-1 {c.get('bptt_steps')}",
              "transformer": f"LM {c.get('lm_steps')}"}[fam]
    status.append(dict(experiment=name, seed=seed, family=fam, task=c.get("task"), T=c.get("seq_len"),
                       N=c.get("num_kv_pairs") if c.get("task") == "mqar" else None, Tf=c.get("future_window"),
                       far=c.get("far"), budget=budget, lr_schedule=c.get("schedule"), finished=r["done"],
                       attempts=r["attempts"], folder=str(r["dir"].relative_to(ROOT)).replace("\\", "/"),
                       history_truncated=(name, seed) in TRUNCATED,
                       **{k: v for k, v in s.items() if k.endswith(("_step", "long_acc", "long_cand_acc"))}))
ST = pd.DataFrame(status)
ST.to_csv(TAB / "new_runs_status.csv", index=False)
print(ST[["experiment", "seed", "finished", "budget"]].to_string())


def vals(name, key, seeds=(201, 202, 203)):
    out = {}
    for s in seeds:
        v = (summ(name, s) or {}).get(key)
        if v is not None:
            out[s] = v
    return out


def pending(name, seeds=(201, 202, 203)):
    return [s for s in seeds if (name, s) in R and not R[(name, s)]["done"]] + [s for s in seeds if (name, s) not in R]


def bar_panel(ax, items, title, ylabel="точность", chance=None, chance_label=None, ylim=1.05):
    """items = [(label, colour, {seed: value}, note)]: bar = mean over the seeds present, dots = seeds."""
    for i, (label, col, v, note) in enumerate(items):
        if v:
            m = sum(v.values()) / len(v)
            ax.bar(i, m, color=col, alpha=0.85, width=0.65, edgecolor="black", linewidth=0.5)
            ax.scatter([i] * len(v), list(v.values()), color="black", s=16, zorder=3)
            ax.text(i, max(v.values()) + 0.03, f"{100 * m:.1f}%" + ("" if len(v) == 3 else f"\n(n={len(v)})") + (f"\n{note}" if note else ""),
                    ha="center", va="bottom", fontsize=11.5 if not note else 10.5)
        elif note:
            ax.text(i, 0.04, note, ha="center", va="bottom", fontsize=10.5, rotation=90, color="#444444")
    ax.set_xticks(range(len(items))); ax.set_xticklabels([it[0] for it in items], fontsize=11.5)
    if chance is not None:
        ax.axhline(chance, color="#888888", ls=":", lw=1.2)
        ax.text(0.0 if not title else 1.0, 1.02, f"пунктир: {chance_label}", transform=ax.transAxes, fontsize=10.5, color="#555555", ha="left" if not title else "right", va="bottom")
    ax.set_ylim(0, ylim + 0.12); ax.set_title(title, loc="left"); ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)


def curve(ax, h, col_name, color, label, ls="-", ce=None):
    v = h.dropna(subset=[col_name])
    ax.plot(v.step, v[col_name], ls=ls, color=color, label=label)
    if ce is not None and ce in v:
        i = v[ce].idxmin(); ax.scatter(v.loc[i, "step"], v.loc[i, col_name], color=color, s=24, zorder=3, marker="o")
    ax.scatter(v.step.iloc[-1], v[col_name].iloc[-1], color=color, s=30, zorder=3, marker="x")


def legend_right(ax, **kw):
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False, **kw)


# =====================================================================================================================
# STACK, NEW runs
# =====================================================================================================================
fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.4))
ax = axes[0]
for s, ls in SEED_LS.items():
    h = hist("stack_smt_tf8", s, "smt")
    if h is not None and R[("stack_smt_tf8", s)]["done"]:
        curve(ax, h, "val_teacher_long_acc", COL["teacher"], f"seed {s}" + (" (история до 6500)" if ("stack_smt_tf8", s) in TRUNCATED else ""), ls, "val_teacher_long_ce")
ax.set_title("(а) SMT: teacher, выход j = 0", loc="left"); ax.set_xlabel("шаг SMT"); ax.set_ylabel("validation accuracy, delay ≥ 9")
ax.set_ylim(0, 1.03); ax.legend(loc="lower right", frameon=False, fontsize=11)
ax = axes[1]
for s, ls in SEED_LS.items():
    h = hist("stack_smt_tf8", s, "dmt")
    if h is not None and R[("stack_smt_tf8", s)]["done"]:
        curve(ax, h, "val_rnn_long_acc", COL["rnn_dmt"], f"seed {s}", ls, "val_rnn_long_ce")
ax.set_title("(б) DMT: итоговая RNN (только f + readout)", loc="left"); ax.set_xlabel("шаг DMT"); ax.set_ylim(0, 1.03)
ax.legend(loc="lower right", frameon=False, fontsize=11)
ax = axes[2]
for name, col in (("stack_bptt_full", COL["bptt_full"]), ("stack_bptt_1", COL["bptt_1"])):
    for s, ls in SEED_LS.items():
        h = hist(name, s, "bptt")
        if h is not None and R[(name, s)]["done"]:
            curve(ax, h, "val_rnn_long_acc", col, f"{'полный BPTT' if name.endswith('full') else 'BPTT-1'}, seed {s}", ls, "val_rnn_long_ce")
ax.set_title("(в) та же RNN с нуля: BPTT", loc="left"); ax.set_xlabel("шаг BPTT"); ax.set_ylim(0, 1.03)
ax.axhline(0.25, color="#888888", ls=":", lw=1.2); ax.text(ax.get_xlim()[1], 0.17, "угадывание 1/4", fontsize=10.5, color="#555555", ha="right")
ax.legend(loc="center right", frameon=False, fontsize=11)
save(fig, "S1_stack_new_curves",
     "Stack T = 64, Tf = 8, K_mem = 8. НОВЫЕ запуски RTX 3090; validation (поток 501, 512 последовательностей) каждые 500 шагов, без сглаживания; "
     "o = выбранный чекпойнт (мин. validation CE на дальних ответах), x = последняя точка. Бюджет SMT 11 000 / DMT 5 000, BPTT 20 000 шагов, batch 32.\n"
     "Рисуются только завершённые запуски. Seed 203: история SMT сохранилась до шага 6500 (сбой синхронизации), итоговые числа не затронуты.",
     bottom=0.24)

STACK_ITEMS = [("SMT → DMT\n(короткое\nокно)", COL["smt"], "stack_smt_tf8", "eval_best_rnn_dmt_long_acc", (201, 202, 203)),
               ("+ дальние\nпо ошибке\nбез памяти", COL["ctx_far"], "stack_smt_tf8_ctx_far", "eval_best_rnn_dmt_long_acc", (201, 202, 203)),
               ("+ случайные\nдальние", COL["random_far"], "stack_smt_tf8_random_far", "eval_best_rnn_dmt_long_acc", (201,)),
               ("+ дальние по\nлокальной\nнеожид.", COL["surprise_far"], "stack_smt_tf8_surprise_far", "eval_best_rnn_dmt_long_acc", (201,)),
               ("без L_dyn", COL["no_dyn"], "stack_smt_tf8_no_dyn", "eval_best_rnn_dmt_long_acc", (201,)),
               ("цель L_dyn\nотцеплена", COL["detached"], "stack_smt_tf8_target_detached", "eval_best_rnn_dmt_long_acc", (201,)),
               ("L_dyn\nполностью\nотцеплен", "#9467bd", "stack_smt_tf8_fully_detached", "eval_best_rnn_dmt_long_acc", (201,)),
               ("RNN,\nполный\nBPTT", COL["bptt_full"], "stack_bptt_full", "eval_best_rnn_bptt_long_acc", (201, 202, 203)),
               ("RNN,\nBPTT-1", COL["bptt_1"], "stack_bptt_1", "eval_best_rnn_bptt_long_acc", (201, 202, 203))]
items = [(lab, col, vals(exp, key, seeds), (lambda p_: None if not p_ else "не готов: " + ", ".join(map(str, p_)))(pending(exp, seeds)))
         for lab, col, exp, key, seeds in STACK_ITEMS]
fig, ax = plt.subplots(figsize=(13, 5.4))
bar_panel(ax, items, "Stack: итоговая RNN на независимом наборе (абляции и дальние цели — только seed 201)", "accuracy, дальние pop (delay ≥ 9)", 0.25, "угадывание 1/4")
save(fig, "S2_stack_new_bars",
     "НОВЫЕ запуски RTX 3090, независимый набор (поток 901, 1024 последовательности), чекпойнт выбран по validation CE; столбец = среднее по готовым seeds, точки = seeds 201–203.\n"
     "Бюджет: SMT 11 000 + DMT 5 000; BPTT 20 000; batch 32, LR константа после разогрева. «не готов» = запуск идёт или ждёт очереди, в среднее не входит.",
     bottom=0.3)

# =====================================================================================================================
# STACK, OLD: short vs full window and the Tf ablation
# =====================================================================================================================
v2 = pd.read_csv(ARCH / "outputs_v2_window" / "table_validation_by_seed.csv")
abl = pd.read_csv(ARCH / "outputs_ablation" / "table_ablation.csv")
fig, axes = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.5]})
ax = axes[0]
for i, (var, lab) in enumerate((("full_window", "полное будущее окно\nTf = T = 64"), ("base_window", "короткое окно\nTf = 8"))):
    sub = v2[v2.variant == var]
    ax.bar(i, sub.dmt_acc.mean(), color=COL[var] if var != "full_window" else "#555555", alpha=0.85, width=0.6, edgecolor="black", linewidth=0.5)
    ax.scatter([i] * len(sub), sub.dmt_acc, color="black", s=16, zorder=3)
    for _, r in sub.iterrows():
        ax.annotate(f"seed {int(r.seed)}: {100 * r.dmt_acc:.1f}", (i, r.dmt_acc), xytext=(-75 if int(r.seed) != 17 else 8, 0), textcoords="offset points", fontsize=10, va="center")
    ax.text(i, 0.05, f"среднее\n{100 * sub.dmt_acc.mean():.1f}%", ha="center", fontsize=12, color="white", weight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["полное будущее окно\nTf = T = 64", "короткое окно\nTf = 8"])
ax.set_ylim(0, 1.05); ax.set_ylabel("accuracy итоговой RNN, delay ≥ 9"); ax.grid(axis="x", visible=False)
ax.axhline(0.25, color="#888888", ls=":", lw=1.2)
ax.set_title("(а) v2: SMT 3 000 / DMT 1 500, seeds 7/17/27", loc="left")
ax = axes[1]
tf = abl[abl.name.isin(["tf1", "tf4", "full", "tf16", "tf64"])].copy()
tf["Tf"] = tf.name.map({"tf1": 1, "tf4": 4, "full": 8, "tf16": 16, "tf64": 64}); tf = tf.sort_values("Tf")
x = range(len(tf))
ax.plot(list(x), tf.teacher, "o-", color=COL["teacher"], label="teacher после SMT")
ax.plot(list(x), tf.rnn_after_dmt, "s-", color=COL["rnn_dmt"], label="итоговая RNN после DMT")
ax.plot(list(x), tf.rnn_before_dmt, "^--", color=COL["rnn_smt"], label="RNN до DMT")
for xi, (_, r) in zip(x, tf.iterrows()):
    ax.annotate(f"{100 * r.rnn_after_dmt:.1f}%", (xi, r.rnn_after_dmt), xytext=(14, -14), textcoords="offset points", ha="left", fontsize=10.5, color=COL["rnn_dmt"])
ax.set_xticks(list(x)); ax.set_xticklabels([f"Tf = {t}" + (" (= T)" if t == 64 else "") for t in tf.Tf]); ax.set_ylim(0, 1.05)
ax.axhline(0.25, color="#888888", ls=":", lw=1.2); ax.set_title("(б) абляция Tf: SMT 11 000 / DMT 5 000, seed 107", loc="left")
legend_right(ax)
save(fig, "S3_stack_window_old",
     "СТАРЫЙ протокол (архив, старый код). (а) outputs_v2_window/table_validation_by_seed.csv: validation, чекпойнт по validation CE, бюджет мал — все варианты недообучены.\n"
     "(б) outputs_ablation/table_ablation.csv: validation, один seed 107, одинаковый бюджет для всех Tf, чекпойнт по validation CE. Числа (а) и (б) не сравнивать между собой: разные бюджеты и seeds.",
     bottom=0.22, right=0.76)

# =====================================================================================================================
# STACK, OLD: far targets, three different metrics (v2, 3 seeds)
# =====================================================================================================================
order = [("base_window", "без дальних"), ("random_far", "случайные"), ("surprise_far", "локальн. неожид."),
         ("ctx_far", "ошибка без памяти"), ("uniform_far", "все дальние"), ("oracle_far", "диагн.: в памяти")]
fig, axes = plt.subplots(1, 3, figsize=(17, 5.4), sharey=True)
for ax, (col_name, title) in zip(axes, (("smt_acc", "(а) teacher после SMT, выход j = 0"),
                                        ("far_memory_acc", "(б) far-memory probe teacher"),
                                        ("dmt_acc", "(в) итоговая RNN после DMT"))):
    for i, (var, lab) in enumerate(order):
        sub = v2[v2.variant == var]
        ax.bar(i, sub[col_name].mean(), color=COL[var], alpha=0.85, width=0.65, edgecolor="black", linewidth=0.5)
        ax.scatter([i] * len(sub), sub[col_name], color="black", s=14, zorder=3)
        ax.text(i, sub[col_name].max() + 0.02, f"{100 * sub[col_name].mean():.0f}", ha="center", fontsize=11)
    ax.set_xticks(range(len(order))); ax.set_xticklabels([o[1] for o in order], fontsize=12, rotation=35, ha="right", rotation_mode="anchor")
    ax.set_title(title, loc="left"); ax.set_ylim(0, 1.05); ax.grid(axis="x", visible=False)
axes[0].set_ylabel("validation accuracy, %-доля")
save(fig, "S4_stack_far_three_metrics_old",
     "СТАРЫЙ протокол v2: Stack T = 64, Tf = 8, K_far = 2, SMT 3 000 / DMT 1 500 (недообучено), seeds 7/17/27, validation; столбец = среднее, точки = seeds, число = среднее в %.\n"
     "(а) teacher: ответ на позиции pop из памяти m_t; (б) probe: тот же замороженный декодер teacher на дальних парах (j ≥ 8), чей push уже в памяти; (в) RNN без teacher. Источник: archive/outputs_v2_window/table_validation_by_seed.csv.",
     bottom=0.3)

# OLD v4 diagnosis, seed 107: probe during SMT and RNN during DMT
fig, axes = plt.subplots(1, 2, figsize=(14, 4.4))
diag = ARCH / "runs" / "stackv4_d459c6921638" / "diag"
for var, lab, ls in (("base_window", "без дальних целей", "--"), ("ctx_far", "+ по ошибке без памяти", "-")):
    h = pd.read_csv(diag / var / "smt_history.csv"); curve(axes[0], h, "val_far_memory_acc", COL[var], lab, ls)
    h = pd.read_csv(diag / var / "dmt_history.csv"); curve(axes[1], h, "val_rnn_long_acc", COL[var], lab, ls, ce="val_rnn_long_ce")
axes[0].set_title("(а) far-memory probe teacher во время SMT", loc="left"); axes[0].set_xlabel("шаг SMT"); axes[0].set_ylabel("validation accuracy")
axes[1].set_title("(б) итоговая RNN во время DMT", loc="left"); axes[1].set_xlabel("шаг DMT")
for a in axes: a.set_ylim(0, 1.03)
legend_right(axes[1])
save(fig, "S5_stack_far_v4_old",
     "СТАРЫЙ протокол v4-диагностика: Stack T = 64, Tf = 8, seed 107, SMT 20 000 / DMT 12 000, validation каждые 500 шагов, без сглаживания; x = последняя точка, o = выбранный чекпойнт.\n"
     "Источник: archive/runs/stackv4_d459c6921638/diag/{base_window,ctx_far}/*_history.csv. Один seed: разница RNN в пределах шума.",
     bottom=0.22, right=0.8)

# OLD ablation of the SMT components (seed 107)
fig, ax = plt.subplots(figsize=(12, 4.6))
abl2 = abl.set_index("name").loc[["full", "no_dyn", "detached", "no_unif"]]
lab = {"full": "все компоненты\n(Tf = 8)", "no_dyn": "без L_dyn\n(c_dyn = 0)", "detached": "цель L_dyn\nотцеплена", "no_unif": "без L_unif\n(c_unif = 0)"}
w = 0.26
for k, (col_name, l, c) in enumerate((("teacher", "teacher после SMT", COL["teacher"]), ("rnn_before_dmt", "RNN до DMT", COL["rnn_smt"]),
                                      ("rnn_after_dmt", "итоговая RNN после DMT", COL["rnn_dmt"]))):
    xs = [i + (k - 1) * w for i in range(len(abl2))]
    ax.bar(xs, abl2[col_name], width=w, color=c, label=l, edgecolor="black", linewidth=0.5)
    for xx, v in zip(xs, abl2[col_name]):
        ax.text(xx, v + 0.015, f"{100 * v:.1f}", ha="center", fontsize=10.5)
ax.set_xticks(range(len(abl2))); ax.set_xticklabels([lab[n] for n in abl2.index]); ax.set_ylim(0, 1.1)
ax.set_ylabel("validation accuracy, delay ≥ 9"); ax.grid(axis="x", visible=False); legend_right(ax)
ax.set_title("Абляции компонентов SMT", loc="left")
save(fig, "S6_stack_ablation_old",
     "СТАРЫЙ протокол: Stack T = 64, seed 107, validation, SMT 11 000 / DMT 5 000, чекпойнт по validation CE. «Цель L_dyn отцеплена» = старое имя detached: "
     "градиент в энкодер через вход m_(t-1) остаётся.\nИсточник: archive/outputs_ablation/table_ablation.csv. Новые абляции (seed 201) на RTX 3090 ещё считаются.",
     bottom=0.2, right=0.8)

# =====================================================================================================================
# MQAR, NEW runs
# =====================================================================================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), gridspec_kw={"width_ratios": [1, 0.8, 1]})
ax = axes[0]
for name, col, lab in (("mqar_T16N2_smt", COL["teacher"], "SMT teacher, j = 0"),
                       ("mqar_T16N2_teacher_bottleneck", COL["teacher_bneck"], "teacher 8 токенов, только L_dec"),
                       ("mqar_T16N2_teacher_full", COL["teacher_full"], "диагностика: декодер читает все выходы энкодера")):
    for s, ls in SEED_LS.items():
        h = hist(name, s, "smt")
        if h is not None and R[(name, s)]["done"]:
            curve(ax, h, "val_teacher_long_acc", col, f"{lab}, s{s}", ls)
ax.axhline(0.5, color="#888888", ls=":", lw=1.2); ax.text(3500, 0.40, "плато 50% = 1 из 2", fontsize=11, color="#555555", ha="center")
ax.set_ylim(0, 1.03); ax.set_xlabel("шаг обучения teacher"); ax.set_ylabel("validation accuracy, весь словарь")
ax.set_title("(а) teacher: скачок после плато ~50%", loc="left")
ax = axes[1]
for name, stage, colname, col, lab in (("mqar_T16N2_smt", "dmt", "val_rnn_long_acc", COL["rnn_dmt"], "RNN во время DMT"),
                                       ("mqar_T16N2_transformer", "lm", "val_lm_long_acc", COL["lm"], "causal Transformer")):
    for s, ls in SEED_LS.items():
        h = hist(name, s, stage)
        if h is not None and R[(name, s)]["done"]:
            curve(ax, h, colname, col, f"{lab}, s{s}", ls)
ax.axhline(0.5, color="#888888", ls=":", lw=1.2); ax.set_ylim(0, 1.03); ax.set_xlabel("шаг своей стадии")
ax.set_title("(б) RNN в DMT и Transformer", loc="left")
ax = axes[2]
for s_, ls in SEED_LS.items():
    h = hist("mqar_T16N2_rnn_bptt_full_long", s_, "bptt")
    if h is not None and R[("mqar_T16N2_rnn_bptt_full_long", s_)]["done"]:
        curve(ax, h, "val_rnn_long_acc", COL["bptt_full"], f"s{s_}", ls)
ax.axhline(0.5, color="#888888", ls=":", lw=1.2); ax.set_ylim(0, 1.03); ax.set_xlabel("шаг BPTT")
ax.set_title("(в) RNN, полный BPTT до 60 000 шагов", loc="left")
legend_models_seeds(fig, [("SMT teacher, j = 0", COL["teacher"]), ("teacher 8 токенов,\nтолько L_dec", COL["teacher_bneck"]),
                          ("диагностика: декодер\nчитает все выходы\nэнкодера", COL["teacher_full"]), ("RNN во время DMT", COL["rnn_dmt"]),
                          ("RNN, полный BPTT", COL["bptt_full"]), ("causal Transformer", COL["lm"])], x=0.79, y=0.93)
save(fig, "M1_mqar_T16_curves",
     "MQAR T = 16, N = 2, V = 128. НОВЫЕ запуски RTX 3090, seeds 201 (—), 202 (– –), 203 (···); validation (поток 511, 512 посл.) каждые 250 шагов, без сглаживания, x = последняя точка. "
     "Teacher: SMT 6 000 шагов, LR константа 3e-4; DMT 2 000; Transformer и полный BPTT (LR 1e-3, до 60 000 шагов) останавливаются при validation ≥ 99%. "
     "Рисуются только завершённые запуски (полный BPTT seed 203 идёт). «Длинные» ответы: delay ≥ 4.",
     bottom=0.24, right=0.78)

def mq_items(prefix, has_full=True):
    T = prefix
    it = [("causal\nTransformer", COL["lm"], f"mqar_{T}_transformer", "eval_best_lm_long"),
          ("teacher\nj = 0", COL["teacher"], f"mqar_{T}_smt", "eval_best_teacher_smt_long"),
          ("teacher, память\nдо запросов", COL["teacher_pre"], f"mqar_{T}_smt", "eval_best_teacher_pre_after_pairs_smt_long")]
    if has_full:
        it += [("teacher 8 ток.,\nтолько L_dec", COL["teacher_bneck"], f"mqar_{T}_teacher_bottleneck", "eval_best_teacher_smt_long"),
               ("диагн.: все\nвыходы энкод.", COL["teacher_full"], f"mqar_{T}_teacher_full", "eval_best_teacher_full_smt_long")]
    it += [("RNN до DMT", COL["rnn_smt"], f"mqar_{T}_smt", "eval_best_rnn_smt_long"),
           ("RNN после\nDMT", COL["rnn_dmt"], f"mqar_{T}_smt", "eval_best_rnn_dmt_long")]
    if has_full:
        it += [("полный BPTT,\n6k шагов", COL["bptt_full"], f"mqar_{T}_rnn_bptt_full", "eval_best_rnn_bptt_long"),
               ("полный BPTT,\n≤ 60k шагов", COL["bptt_full"], f"mqar_{T}_rnn_bptt_full_long", "eval_best_rnn_bptt_long")]
    return it

for T, N, has_full in (("T16N2", 2, True), ("T32N4", 4, False)):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8.6), sharex=True)
    its = mq_items(T, has_full)
    for ax, suffix, ylab, chance, cl in ((axes[0], "_acc", "accuracy по всему словарю", None, None),
                                         (axes[1], "_cand_acc", "accuracy среди N записанных\nзначений (диагностика)", 1 / N, f"угадывание 1/N = {100 / N:.0f}%")):
        items = [(lab, col, vals(exp, key + suffix), None if not pending(exp) else "нет: " + ",".join(map(str, pending(exp))))
                 for lab, col, exp, key in its]
        bar_panel(ax, items, "", ylab, chance, cl)
    axes[0].set_title(f"MQAR T = {T[1:3]}, N = {N}: независимый набор", loc="left"); axes[0].tick_params(labelbottom=False); axes[1].tick_params(labelbottom=True)
    note = ("НОВЫЕ запуски RTX 3090, независимый набор (поток 901, 1024 посл.), чекпойнт по validation CE (у Transformer — остановка при 99%), столбец = среднее, точки = seeds 201–203. "
            f"Длинные ответы: delay ≥ {4 if N == 2 else 8}.\n" +
            ("Бюджеты: teacher/SMT 6 000 + DMT 2 000; teacher-диагностики 6 000; Transformer ≤ 3 000. Полный BPTT: 6 000 шагов не хватило; длинный бюджет — до 60 000 шагов с остановкой при 99%." if N == 2 else
             "Бюджеты: SMT 6 000 + DMT 2 000, Transformer ≤ 3 000. Полный BPTT на T = 32 ещё не запускался (в очереди, 60 000 шагов)."))
    save(fig, f"M2_mqar_{T}_bars", note, bottom=0.2)

# T32N4 curves
fig, ax = plt.subplots(figsize=(9.5, 4.8))
for name, stage, colname, col, lab in (("mqar_T32N4_smt", "smt", "val_teacher_long_acc", COL["teacher"], "SMT teacher, j = 0"),
                                       ("mqar_T32N4_smt", "dmt", "val_rnn_long_acc", COL["rnn_dmt"], "RNN во время DMT"),
                                       ("mqar_T32N4_transformer", "lm", "val_lm_long_acc", COL["lm"], "causal Transformer")):
    for s, ls in SEED_LS.items():
        h = hist(name, s, stage)
        if h is not None and R[(name, s)]["done"]:
            curve(ax, h, colname, col, f"{lab}, s{s}", ls)
ax.axhline(0.25, color="#888888", ls=":", lw=1.2); ax.set_ylim(0, 1.03); ax.set_xlabel("шаг своей стадии"); ax.set_ylabel("validation accuracy, весь словарь")
ax.set_title("MQAR T = 32, N = 4: кривые обучения", loc="left")
legend_models_seeds(fig, [("SMT teacher, j = 0", COL["teacher"]), ("RNN во время DMT", COL["rnn_dmt"]), ("causal Transformer", COL["lm"])], x=0.72, y=0.93)
save(fig, "M3_mqar_T32_curves",
     "НОВЫЕ запуски RTX 3090, seeds 201–203, validation (поток 511) каждые 250 шагов, без сглаживания; SMT 6 000 / DMT 2 000, LR константа; Transformer остановлен при 99%. Длинные ответы: delay ≥ 8.",
     bottom=0.27, right=0.7)


# T32N4 curves, large standalone version for a slide (three small panels instead of one crowded plot)
with plt.rc_context({"font.size": 15, "axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14}):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.6), sharey=True)
    for ax, (name, stage, colname, col, title) in zip(axes, (("mqar_T32N4_smt", "smt", "val_teacher_long_acc", COL["teacher"], "(а) SMT teacher, j = 0"),
                                                             ("mqar_T32N4_smt", "dmt", "val_rnn_long_acc", COL["rnn_dmt"], "(б) итоговая RNN в DMT"),
                                                             ("mqar_T32N4_transformer", "lm", "val_lm_long_acc", COL["lm"], "(в) causal Transformer"))):
        for s_, ls in SEED_LS.items():
            h = hist(name, s_, stage)
            if h is not None and R[(name, s_)]["done"]:
                curve(ax, h, colname, col, f"seed {s_}", ls)
        ax.axhline(0.25, color="#888888", ls=":", lw=1.4)
        ax.set_ylim(0, 1.03); ax.set_xlabel(f"шаг {stage.upper()}"); ax.set_title(title, loc="left")
        ax.legend(loc="upper left", frameon=False)
    axes[0].set_ylabel("validation accuracy, весь словарь")
    axes[0].text(150, 0.28, "угадывание 1/N = 25%", fontsize=13, color="#555555")
    fig.text(0.005, 0.01, "MQAR T = 32, N = 4. Новые запуски RTX 3090, validation (поток 511) каждые 250 шагов, без сглаживания, x = последняя точка. "
                          "\nSMT 6 000 / DMT 2 000, LR константа; Transformer остановлен при 99%. Длинные ответы: delay ≥ 8.", fontsize=12, color="#333333")
    fig.subplots_adjust(bottom=0.24, wspace=0.08, left=0.06, right=0.99)
    fig.savefig(FIG / "M3_mqar_T32_curves_big.png"); plt.close(fig); print("figure M3_mqar_T32_curves_big")

# =====================================================================================================================
# Schemes: SMT training / DMT / inference (exact data flow of the code)
# =====================================================================================================================
def box(ax, x, y, w, h, text, fc="white", ec="black", fs=10.5, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.08", fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")


def arrow(ax, a, b, text=None, color="black", ls="-", off=(0, 0.12)):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=12, color=color, lw=1.3, linestyle=ls))
    if text:
        ax.text((a[0] + b[0]) / 2 + off[0], (a[1] + b[1]) / 2 + off[1], text, ha="center", fontsize=10.5, color=color)


def scheme(task):
    stack = task == "stack"
    xs = "x_0..x_t (стек-операции)" if stack else "x_0..x_t (пары и запросы)"
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0))
    for ax in axes:
        ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")
    # (а) SMT
    ax = axes[0]; ax.set_title("(а) SMT: teacher и f учатся вместе", loc="left", fontsize=14.5)
    box(ax, 0.2, 8.2, 4.6, 1.1, f"x_ctx = {xs}\nслева паддинг −1 до длины T", fs=10)
    box(ax, 5.2, 8.2, 4.6, 1.1, "8 обучаемых регистров\nencoder_mem_register [8, 64]", fs=10)
    box(ax, 1.5, 6.3, 7, 1.1, "teacher.encoder (двунаправленный, 3 слоя)\nвход [T+8, 64] → выходы регистров [8, 64]", fc="#eaf2fb", fs=10)
    box(ax, 2.5, 4.7, 5, 1.0, "память m_t [8, 64] (RMS-норма)", fc="#eaf2fb", fs=10, bold=True)
    box(ax, 0.2, 2.6, 4.9, 1.4, "teacher.decoder (каузальный)\nвход [m_t, embed(x_{t+1..t+j})]\nвыход j → y_{t+j}", fc="#eaf2fb", fs=10)
    box(ax, 5.4, 2.6, 4.4, 1.4, "f(m_{t−1}, x_t) — updater RNN\nвход [m_{t−1}, embed(x_t)]\n[9, 64] → [8, 64]", fc="#fdf0e6", fs=10)
    box(ax, 0.2, 0.3, 4.9, 1.5, "L_near: CE по j < Tf\nL_far: CE по K_far выбранным\nдальним j ≥ Tf (если far ≠ none)", fs=10)
    box(ax, 5.4, 0.3, 4.4, 1.5, "L_dyn = MSE(f(m_{t−1}, x_t), m_t)\n(m_{t−1} тоже от энкодера)\nL_unif по токенам m_t", fs=10)
    arrow(ax, (2.5, 8.2), (3.5, 7.4)); arrow(ax, (7.5, 8.2), (6.5, 7.4)); arrow(ax, (5, 6.3), (5, 5.7))
    arrow(ax, (4, 4.7), (2.7, 4.0)); arrow(ax, (6, 4.7), (7.6, 4.0)); arrow(ax, (2.65, 2.6), (2.65, 1.8)); arrow(ax, (7.6, 2.6), (7.6, 1.8))
    ax.text(0.2, -0.4, "y берётся из генератора; детекторы дальних целей заморожены и\nработают только при обучении", fontsize=10.5, color="#333333", va="top")
    # (б) DMT
    ax = axes[1]; ax.set_title("(б) DMT: teacher заморожен, учится только RNN", loc="left", fontsize=14.5)
    box(ax, 0.3, 8.2, 9.4, 1.1, "teacher.encoder заморожен (no_grad):\nm_t для каждого t — только цель", fc="#eeeeee", fs=10)
    box(ax, 1.4, 5.6, 1.9, 1.4, "m^_{t−1}\nсвоё\n(detach)", fc="#fdf0e6", fs=10)
    box(ax, 3.6, 5.6, 2.8, 1.4, "f(m^_{t−1}, x_t)\nupdater", fc="#fdf0e6", fs=10, bold=True)
    box(ax, 6.9, 5.6, 2.8, 1.4, "m^_t\n[8, 64]", fc="#fdf0e6", fs=10)
    box(ax, 0.3, 2.7, 4.4, 1.8, "10 · MSE(m^_t, m_t)\nградиент → updater f\n(на один шаг: вход detach)", fs=10)
    box(ax, 5.3, 2.7, 4.4, 1.8, "CE(y_t | readout(sg[m^_t]))\nградиент → только readout\n(decoder + to_logits RNN)", fs=10)
    arrow(ax, (3.3, 6.3), (3.6, 6.3)); arrow(ax, (6.4, 6.3), (6.9, 6.3)); arrow(ax, (8.3, 5.6), (7.5, 4.5)); arrow(ax, (8.0, 5.6), (2.5, 4.5))
    arrow(ax, (0.9, 8.2), (0.9, 4.5), color="#555555", ls="--"); ax.text(1.0, 7.55, "m_t (цель)", fontsize=10.5, color="#555555")
    ax.text(5.55, 7.35, "m^_t → вход следующего шага", fontsize=10.5, color="#555555", ha="center")
    ax.text(0.3, 1.9, "detach режет градиент между шагами,\nно не обнуляет само состояние:\nm^_t реально переносится по всей последовательности", fontsize=10.5, va="top")
    # (в) inference
    ax = axes[2]; ax.set_title("(в) инференс итоговой модели", loc="left", fontsize=14.5)
    box(ax, 0.3, 7.6, 9.4, 1.3, "m^_{−1} = 0  →  для t = 0..T−1:  m^_t = f(m^_{t−1}, x_t)", fc="#fdf0e6", fs=11, bold=True)
    box(ax, 0.3, 5.0, 9.4, 1.3, "ответ на позиции pop / запроса:  argmax readout(m^_t)", fc="#fdf0e6", fs=11)
    box(ax, 0.3, 1.6, 9.4, 2.4, "НЕ используются:\n• teacher.encoder и teacher.decoder\n• правильные ответы y и x_fut (будущие входы)\n• детекторы и отбор дальних целей", fc="#f7f7f7", fs=10)
    arrow(ax, (5, 7.6), (5, 6.3))
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.13, wspace=0.08)
    fig.savefig(FIG / f"D_scheme_{task}.png"); plt.close(fig); print("figure", f"D_scheme_{task}")


scheme("stack"); scheme("mqar")

# OLD protocol, three panels on one slide: window length (v2), Tf ablation and component ablation (seed 107)
fig, axes = plt.subplots(1, 3, figsize=(16, 4.9), gridspec_kw={"width_ratios": [0.8, 1.2, 1.25]})
ax = axes[0]
for i, var in enumerate(("full_window", "base_window")):
    sub = v2[v2.variant == var]
    ax.bar(i, sub.dmt_acc.mean(), color="#555555" if var == "full_window" else COL["base_window"], alpha=0.85, width=0.6, edgecolor="black", linewidth=0.5)
    ax.scatter([i] * len(sub), sub.dmt_acc, color="black", s=16, zorder=3)
    ax.text(i, 0.05, f"{100 * sub.dmt_acc.mean():.1f}%", ha="center", fontsize=12, color="white", weight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["полное будущее\nокно, Tf = 64", "короткое окно\nTf = 8"])
ax.set_ylim(0, 1.05); ax.set_ylabel("validation accuracy итоговой RNN, delay ≥ 9"); ax.grid(axis="x", visible=False)
ax.axhline(0.25, color="#888888", ls=":", lw=1.2)
ax.set_title("(а) v2: SMT 3k / DMT 1,5k,\nseeds 7/17/27 (точки)", loc="left", fontsize=13.5)
ax = axes[1]
ax.plot(list(range(len(tf))), tf.teacher, "o-", color=COL["teacher"], label="teacher")
ax.plot(list(range(len(tf))), tf.rnn_after_dmt, "s-", color=COL["rnn_dmt"], label="итоговая RNN")
for xi, (_, r) in enumerate(tf.iterrows()):
    ax.annotate(f"{100 * r.rnn_after_dmt:.1f}", (xi, r.rnn_after_dmt), xytext=(10, -16), textcoords="offset points", ha="left", fontsize=11, color=COL["rnn_dmt"])
ax.set_xticks(list(range(len(tf)))); ax.set_xticklabels([str(t) + (" = T" if t == 64 else "") for t in tf.Tf]); ax.set_xlabel("T_f")
ax.set_ylim(0, 1.05); ax.axhline(0.25, color="#888888", ls=":", lw=1.2); ax.legend(loc="lower right", frameon=False)
ax.set_title("(б) горизонт T_f: SMT 11k / DMT 5k, seed 107", loc="left", fontsize=13.5)
ax = axes[2]
w_ = 0.38
for k, (col_name, l, c) in enumerate((("teacher", "teacher", COL["teacher"]), ("rnn_after_dmt", "итоговая RNN", COL["rnn_dmt"]))):
    xs = [i + (k - 0.5) * w_ for i in range(len(abl2))]
    ax.bar(xs, abl2[col_name], width=w_, color=c, label=l, edgecolor="black", linewidth=0.5)
    for xx, v in zip(xs, abl2[col_name]):
        ax.text(xx, v + 0.015, f"{100 * v:.1f}", ha="center", fontsize=10.5)
ax.set_xticks(range(len(abl2))); ax.set_xticklabels(["все\nкомпоненты", "без L_dyn", "цель L_dyn\nотцеплена", "без L_unif"])
ax.set_ylim(0, 1.3); ax.grid(axis="x", visible=False); ax.legend(loc="upper center", ncol=2, frameon=False)
ax.set_title("(в) компоненты SMT: 11k / 5k, seed 107, T_f = 8", loc="left", fontsize=13.5)
save(fig, "S7_stack_ablations_old",
     "СТАРЫЙ протокол (архив, старый код), validation, чекпойнт по validation CE; Stack T = 64, delay ≥ 9. (а) archive/outputs_v2_window/table_validation_by_seed.csv — бюджет мал, недообучено. "
     "(б, в) archive/outputs_ablation/table_ablation.csv — один seed 107, одинаковый бюджет внутри панели. Панели между собой не сравнивать. Новые абляции (seed 201, RTX 3090) ещё считаются.",
     bottom=0.2)

# ---- tables for the deck ---------------------------------------------------------------------------------------------
rows = []
for exp, key, lab in (("stack_smt_tf8", "eval_best_rnn_dmt_long_acc", "SMT → DMT, короткое окно"),
                      ("stack_smt_tf8_ctx_far", "eval_best_rnn_dmt_long_acc", "+ дальние цели по ошибке без памяти"),
                      ("stack_smt_tf8_surprise_far", "eval_best_rnn_dmt_long_acc", "+ дальние цели по локальной неожиданности"),
                      ("stack_smt_tf8_random_far", "eval_best_rnn_dmt_long_acc", "+ случайные дальние цели"),
                      ("stack_smt_tf8_no_dyn", "eval_best_rnn_dmt_long_acc", "без L_dyn"),
                      ("stack_smt_tf8_target_detached", "eval_best_rnn_dmt_long_acc", "цель L_dyn отцеплена"),
                      ("stack_smt_tf8_fully_detached", "eval_best_rnn_dmt_long_acc", "L_dyn полностью отцеплен"),
                      ("stack_bptt_full", "eval_best_rnn_bptt_long_acc", "RNN, полный BPTT"),
                      ("stack_bptt_1", "eval_best_rnn_bptt_long_acc", "RNN, BPTT-1")):
    v = vals(exp, key)
    rows.append(dict(model=lab, experiment=exp, **{f"s{s}": (f"{100 * v[s]:.1f}" if s in v else ("не завершён" if (exp, s) in R else "—")) for s in (201, 202, 203)}))
pd.DataFrame(rows).to_csv(TAB / "stack_new_table.csv", index=False)
print(pd.DataFrame(rows).to_string())
