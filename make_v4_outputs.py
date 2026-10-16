#!/usr/bin/env python3
"""Figures and tables of the diagnostic queue (16.10.2026) from the saved histories. CPU only; reads files, trains
nothing, creates no run directory, never touches a test split.

  python make_v4_outputs.py            # -> outputs_v4/ (Stack v4 part D) and outputs_mqar/ (MQAR A, Zoology B)

Everything here is VALIDATION (or the Zoology test set used by the authors' own protocol) and may be interim: the
tables say how many steps each run has done.
"""
from __future__ import annotations

import glob
import json
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import smt_memup_stack as S  # noqa: E402
import smt_memup_v4 as V  # noqa: E402

ROOT = Path(__file__).resolve().parent
OUT_V4, OUT_MQAR = ROOT / "outputs_v4", ROOT / "outputs_mqar"
CFG = V.V4Config()
# categorical slots in fixed order (reference palette, light mode); LR is a secondary encoding (line style)
COLOR = {"base_window": "#2a78d6", "ctx_far": "#eb6834", "bptt_full": "#1baf7a", "bptt_1": "#eda100",
         "random_far": "#e87ba4", "surprise_far": "#008300", "uniform_far": "#6250d6", "full_window": "#e34948",
         "transformer": "#2a78d6", "rnn_bptt": "#1baf7a"}
SMT_RUNS = ("base_window", "ctx_far", "random_far", "surprise_far", "uniform_far", "full_window", "oracle_far")
STYLE = {1e-4: ":", 3e-4: "-", 1e-3: "--"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
NAMES = {"base_window": "SMT короткое окно, без добавки", "ctx_far": "SMT + контекстный отбор (ctx_far)",
         "bptt_full": "full BPTT", "bptt_1": "BPTT-1", "random_far": "SMT + случайные дальние (random_far)",
         "surprise_far": "SMT + локальный surprise r=8 (surprise_far)", "uniform_far": "SMT + все дальние (uniform_far)",
         "full_window": "SMT полное окно Tf=T (бейзлайн статьи)", "oracle_far": "SMT + oracle (диагностика)"}


def style_axes(ax, title, ylabel, xlabel="шаг оптимизации"):
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel(xlabel, fontsize=9, color=MUTED); ax.set_ylabel(ylabel, fontsize=9, color=MUTED)
    ax.grid(True, color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)


def chance(ax, level, label):
    ax.axhline(level, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(ax.get_xlim()[1], level, f" {label}", va="bottom", ha="right", fontsize=7, color=MUTED)


def pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylim(0, 1.02)


# --------------------------------------------------------------------------------------
# Stack v4, part D
# --------------------------------------------------------------------------------------

def stack_run_dir(cfg=CFG) -> Path | None:
    """The GPU run of a V4Config (found by its identity; nothing is created)."""
    for d in sorted(glob.glob(str(ROOT / "runs" / "stackv4_*"))):
        ident = json.loads((Path(d) / "identity.json").read_text())
        if ident.get("device") == "cuda" and ident["config"] == S.config_json(cfg):
            return Path(d)
    return None


def load_stack(run: Path) -> pd.DataFrame:
    rows = []
    conf = json.loads((run / "identity.json").read_text())["config"]
    for d in sorted((run / "diag").glob("*/")):
        variant, lr = (d.name.split("_lr") + [None])[:2]
        for stage in ("smt", "dmt", "bptt"):
            f = d / f"{stage}_history.csv"
            if f.exists():
                h = pd.read_csv(f)
                h = h.dropna(subset=[c for c in h.columns if c.startswith("val_")][:1])
                h.insert(0, "stage", stage); h.insert(0, "job_lr", float(lr) if lr else np.nan); h.insert(0, "variant", variant)
                h.insert(0, "steps_max", {"smt": conf["smt_steps"], "dmt": conf["dmt_steps"], "bptt": conf["bptt_steps"]}[stage])
                h.insert(0, "smt_budget", conf["smt_steps"])
                h.insert(0, "run", d.name)
                rows.append(h)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def stack_progress(hist: pd.DataFrame, run: Path) -> pd.DataFrame:
    rows = []
    for (name, stage), h in hist.groupby(["run", "stage"], sort=False):
        col = "val_teacher_long_acc" if stage == "smt" else "val_rnn_long_acc"
        maximum = int(h.steps_max.iloc[0])
        s = V.sufficiency(h, col, CFG)
        best_i = h[col].idxmax()
        rows.append(dict(run=name, stage=stage, metric=col.replace("val_", "").replace("_long_acc", ""),
                         steps_done=int(h.step.max()), steps_max=maximum, complete=int(h.step.max()) >= maximum,
                         last_acc=float(h[col].iloc[-1]), best_acc=float(h[col].max()), best_step=int(h.step[best_i]),
                         gain_last_quarter_of_done=s.get("gain_last_fraction"), steps_to_plateau_so_far=s.get("steps_to_plateau"),
                         far_memory_acc_last=float(h["val_far_memory_acc"].iloc[-1]) if "val_far_memory_acc" in h else np.nan,
                         train_hours=float(h["train_seconds"].iloc[-1]) / 3600 if "train_seconds" in h else np.nan,
                         eval_hours=float(h["eval_seconds"].iloc[-1]) / 3600 if "eval_seconds" in h else np.nan,
                         sec_per_step=float(h["train_seconds"].iloc[-1]) / max(1, int(h.step.max())) if "train_seconds" in h else np.nan,
                         peak_allocated_mib=float(h["peak_allocated_mib"].max()) if "peak_allocated_mib" in h else np.nan))
    return pd.DataFrame(rows)


def stack_figure(hist: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    diag = SMT_RUNS
    panels = [(axes[0, 0], "smt", "val_teacher_long_acc", "а) SMT: teacher, дальние ответы", diag),
              (axes[0, 1], "smt", "val_far_memory_acc", "б) SMT: far-memory probe (память m_t teacher)", diag),
              (axes[1, 0], "dmt", "val_rnn_long_acc", "в) DMT: итоговая RNN, дальние ответы", diag),
              (axes[1, 1], "bptt", "val_rnn_long_acc", "г) BPTT-ориентиры: RNN, дальние ответы", ("bptt_full", "bptt_1"))]
    for ax, stage, col, title, variants in panels:
        style_axes(ax, title, "точность (validation)")
        sub = hist[(hist.stage == stage) & hist.variant.isin(variants)] if len(hist) else hist
        for (name, variant, lr), h in (sub.groupby(["run", "variant", "job_lr"], dropna=False) if len(sub) else []):
            if col not in h or h[col].isna().all():
                continue
            label = NAMES[variant] + (f", LR {lr:g}" if not np.isnan(lr) else "")
            ax.plot(h.step, h[col], color=COLOR.get(variant, MUTED), linewidth=2, linestyle=STYLE.get(lr, "-") if not np.isnan(lr) else "-",
                    label=label)
        pct(ax)
        if ax.lines:
            chance(ax, 0.25, "угадывание 25%")
            ax.legend(fontsize=7, frameon=False, loc="upper left" if stage == "bptt" else "lower right")
        else:
            ax.text(0.5, 0.5, "ещё нет данных", transform=ax.transAxes, ha="center", color=MUTED)
    fig.suptitle(f"Stack v4, диагностика D (seed 107, validation, промежуточно; {time.strftime('%d.%m %H:%M')})",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150); plt.close(fig)


def stack_cost_figure(hist: pd.DataFrame, path: Path):
    """Final-model accuracy against the training sequences consumed (SMT: SMT + DMT sequences; detectors excluded,
    trained once in version 1). One axis; the same quantity for every curve."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    style_axes(ax, "Итоговая RNN против числа обучающих последовательностей (validation, дальние ответы)",
               "точность RNN", "обучающих последовательностей, млн")
    for name, h in (hist.groupby("run") if len(hist) else []):
        variant = h.variant.iloc[0]
        if variant in SMT_RUNS:
            smt, dmt = h[h.stage == "smt"], h[h.stage == "dmt"]
            if not len(dmt):
                continue
            x = (int(h.smt_budget.iloc[0]) * CFG.batch_size + dmt.step * CFG.dmt_batch_size) / 1e6
            ax.plot(x, dmt.val_rnn_long_acc, color=COLOR.get(variant, MUTED), linewidth=2,
                    label=NAMES[variant] + f" (после SMT {int(h.smt_budget.iloc[0]) // 1000}k)")
        else:
            lr = h.job_lr.iloc[0]
            ax.plot(h.step * CFG.bptt_batch_size / 1e6, h.val_rnn_long_acc, color=COLOR.get(variant, MUTED), linewidth=2,
                    linestyle=STYLE.get(lr, "-"), label=f"{NAMES[variant]}, LR {lr:g}")
    pct(ax)
    if ax.lines:
        chance(ax, 0.25, "угадывание 25%")
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    else:
        ax.text(0.5, 0.5, "DMT ещё не начался", transform=ax.transAxes, ha="center", color=MUTED)
    fig.savefig(path, dpi=150); plt.close(fig)


# --------------------------------------------------------------------------------------
# MQAR A and Zoology B
# --------------------------------------------------------------------------------------

def mqar_a() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    hist, summ, rows = [], [], []
    for d in sorted(glob.glob(str(ROOT / "runs" / "mqar_baselines_*" / "*/"))):
        d = Path(d)
        if (d / "history.csv").exists():
            h = pd.read_csv(d / "history.csv")
            h = h.dropna(subset=["val_long_acc"]) if "val_long_acc" in h else h.iloc[0:0]
            h.insert(0, "job", d.name); hist.append(h)
            if len(h):
                last = h.iloc[-1]
                rows.append(dict(job=d.name, steps_done=int(last.step), complete=(d / "completed.pt").exists(),
                                 val_long_acc=last.val_long_acc, val_long_cand_acc=last.val_long_cand_acc,
                                 val_first_query_acc=last.val_first_query_acc, val_long_ce=last.val_long_ce,
                                 train_set_acc=last.get("train_set_acc", np.nan),
                                 train_hours=last.train_seconds / 3600 if "train_seconds" in last else np.nan))
        if (d / "summary.csv").exists():
            s = pd.read_csv(d / "summary.csv"); s.insert(0, "job", d.name); summ.append(s)
    cat = lambda x: pd.concat(x, ignore_index=True) if x else pd.DataFrame()
    return cat(hist), pd.DataFrame(rows), cat(summ)


def mqar_figure(hist: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), constrained_layout=True)
    specs = [(axes[0], "val_long_acc", "а) full-vocabulary, дальние ответы", 1 / 64, "шанс среди значений 1/64"),
             (axes[1], "val_long_cand_acc", "б) среди 8 записанных значений", 1 / 8, "шанс 1/8"),
             (axes[2], "train_set_acc", "в) переобучение: фиксированные 256 посл.", None, None)]
    for ax, col, title, ch, chl in specs:
        style_axes(ax, title, "точность")
        for job, h in (hist.groupby("job") if len(hist) else []):
            if col not in h or h[col].isna().all():
                continue
            model = "transformer" if job.startswith("tf_") else "rnn_bptt"
            lr = 1e-3 if ("lr1e-3" in job or "overfit" in job) else 3e-4 if "lr3e-4" in job else 1e-4
            ax.plot(h.step, h[col], color=COLOR[model], linewidth=2, linestyle=STYLE.get(lr, "-"), label=job)
        pct(ax)
        if ax.lines:
            if ch:
                chance(ax, ch, chl)
            ax.legend(fontsize=7, frameon=False, loc="lower right")
        else:
            ax.text(0.5, 0.5, "ещё нет данных", transform=ax.transAxes, ha="center", color=MUTED)
    fig.suptitle(f"MQAR pilot01, проверка A: Transformer (синий) и RNN full BPTT (бирюзовый); validation; {time.strftime('%d.%m %H:%M')}",
                 x=0.01, ha="left", fontsize=11, color=INK)
    fig.savefig(path, dpi=150); plt.close(fig)


def zoology_b() -> pd.DataFrame:
    rows = []
    for d in sorted(glob.glob(str(ROOT / "runs" / "zoology_repro" / "*" / "*/"))):
        d = Path(d)
        if (d / "summary.json").exists():
            rows.append(dict(cell=d.parent.name, **json.loads((d / "summary.json").read_text()), complete=True))
        elif (d / "metrics.jsonl").exists():
            ep = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines() if "valid/accuracy" in l]
            rows.append(dict(cell=d.parent.name, run_id=d.name, epochs_run=len(ep), complete=False,
                             final_valid_accuracy=ep[-1]["valid/accuracy"] if ep else np.nan))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------

def md(df: pd.DataFrame, cols, fmt=None) -> str:
    if not len(df):
        return "_нет данных_\n"
    fmt = fmt or {}
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    body = ""
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r.get(c)
            cells.append(fmt[c](v) if c in fmt and pd.notna(v) else ("—" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)))
        body += "| " + " | ".join(cells) + " |\n"
    return head + body


def main():
    OUT_V4.mkdir(exist_ok=True); OUT_MQAR.mkdir(exist_ok=True)
    stamp = time.strftime("%d.%m.%Y %H:%M")
    p = lambda x: f"{100 * x:.1f}%"
    run = stack_run_dir()
    # part D (default config) plus the extra diagnostic runs (short budget, full window): every GPU run directory
    dirs = [Path(d) for d in sorted(glob.glob(str(ROOT / "runs" / "stackv4_*")))
            if json.loads((Path(d) / "identity.json").read_text()).get("device") == "cuda"]
    parts = [load_stack(r) for r in dirs]
    parts = [p for p in parts if len(p)]
    hist = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    prog = stack_progress(hist, run) if len(hist) else pd.DataFrame()
    hist.to_csv(OUT_V4 / "table_diag_histories.csv", index=False)
    prog.to_csv(OUT_V4 / "table_diag_progress.csv", index=False)
    stack_figure(hist, OUT_V4 / "figD1_validation_curves.png")
    stack_cost_figure(hist, OUT_V4 / "figD2_rnn_vs_sequences.png")
    decision = (run / "diag_decision.json") if run else None
    mh, mp, ms = mqar_a()
    mh.to_csv(OUT_MQAR / "table_mqarA_histories.csv", index=False)
    mp.to_csv(OUT_MQAR / "table_mqarA_progress.csv", index=False)
    if len(ms):
        ms.to_csv(OUT_MQAR / "table_mqarA_summary.csv", index=False)
    mqar_figure(mh, OUT_MQAR / "figA1_mqar_baselines.png")
    zb = zoology_b()
    zb.to_csv(OUT_MQAR / "table_zooB.csv", index=False)
    text = f"""# Диагностическая очередь 16.10.2026 — графики и таблицы (обновляется автоматически)

**Обновлено {stamp}.** Только validation, промежуточные данные: пока задачи не завершены, это не результаты.
Собирает `python make_v4_outputs.py` из `runs/`. Протокол — `proposals/preregistration_v4.md`, статус — `status.md`.
Каталог запуска Stack: `{run.relative_to(ROOT) if run else '—'}`.

## Stack v4, часть D (seed 107)
![](figD1_validation_curves.png)

![](figD2_rnn_vs_sequences.png)

Как читать:
* (а) teacher после SMT на дальних ответах (давность ≥ 8);
* (б) far-memory probe — хранит ли промежуточная память teacher m_t элемент. **Это не итоговая RNN**;
* (в) итоговая RNN в DMT;
* (г) BPTT-ориентиры, та же RNN с нуля: стиль линии = LR (точки 1e-4, сплошная 3e-4, штрихи 1e-3).

Прогресс:
* «шаги до плато» и «прирост за последнюю четверть» считаются по уже пройденным шагам;
* правило достаточности применяется только к завершённым обучениям.

{md(prog, ["run", "stage", "steps_done", "steps_max", "best_acc", "best_step", "last_acc", "gain_last_quarter_of_done", "far_memory_acc_last", "sec_per_step", "train_hours", "peak_allocated_mib"],
    dict(best_acc=p, last_acc=p, gain_last_quarter_of_done=lambda x: f"{100 * x:+.1f} п.п.", far_memory_acc_last=p,
         sec_per_step=lambda x: f"{x:.2f}", train_hours=lambda x: f"{x:.2f}", peak_allocated_mib=lambda x: f"{x:.0f}"))}
**Время шага** измерено под нагрузкой: до 7 процессов на одной GPU. Чистая стоимость одного процесса — в
`benchmark_rtx5090_2026-10-16.json`.

Решение по бюджетам финала: {('`' + str(decision.relative_to(ROOT)) + '`') if decision and decision.exists() else 'ещё нет (после всех 8 обучений: `python smt_memup_v4.py --phase diag_report`)'}.

## MQAR, проверка A (pilot01) и Zoology, проверка Б
Графики и таблицы: `outputs_mqar/figA1_mqar_baselines.png`, `table_mqarA_progress.csv`, `table_mqarA_summary.csv`,
`table_zooB.csv`.

{md(mp, ["job", "steps_done", "complete", "val_long_acc", "val_long_cand_acc", "val_first_query_acc", "train_set_acc"],
    dict(val_long_acc=p, val_long_cand_acc=p, val_first_query_acc=p, train_set_acc=p))}
{md(zb, [c for c in ["cell", "run_id", "epochs_run", "final_valid_accuracy", "independent_test_accuracy", "early_stopped", "complete"] if c in zb.columns] if len(zb) else ["cell"],
    dict(final_valid_accuracy=p, independent_test_accuracy=p))}
## Файлы
* `table_diag_histories.csv` — все точки validation: run, stage, step, метрики, время, память.
* `table_diag_progress.csv` — сводка выше.
* Логи задач — `runs/jobs/*.log`, планировщик — `runs/gpu_jobs.log`.
* Истории и стоимость — `runs/stackv4_*/diag/*/`, `runs/mqar_baselines_*/*/`, `runs/zoology_repro/`. Checkpoints
  (`*.pt`) в git не входят.
"""
    (OUT_V4 / "README.md").write_text(text)
    print(f"written outputs_v4/ and outputs_mqar/ ({stamp}); stack runs: {hist.run.nunique() if len(hist) else 0}, "
          f"mqar jobs: {mh.job.nunique() if len(mh) else 0}, zoology runs: {len(zb)}")


if __name__ == "__main__":
    main()
