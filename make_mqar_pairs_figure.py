#!/usr/bin/env python3
"""MQAR (pilot01 family: vocab 128, T = 64): where do the models break as the number of key-value pairs grows?
Reads finished runs only (CPU; trains nothing). Metric: full-vocabulary accuracy on long answers (delay >= 24) of the
TUNING split, best checkpoint by validation. Budgets differ and are listed in the table (the 8-pair points come from
the longer runs of 16-17.10).

  python make_mqar_pairs_figure.py      # -> outputs_mqar/figA2_pairs_sweep.png, table_pairs_sweep.csv
"""
import glob
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

import make_v4_outputs as V  # noqa: E402  (shared styling)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_mqar"


def baseline_point(job: str) -> dict | None:
    for d in glob.glob(str(ROOT / "runs" / "mqar_baselines_*" / job / "summary.csv")):
        s = pd.read_csv(d)
        r = s[(s.group == "delay>=24") & (s.split == "tuning") & (s.checkpoint == "best")]
        cost = json.loads((Path(d).parent / "cost.json").read_text())
        return dict(accuracy=float(r.accuracy.iloc[0]), cand_accuracy=float(r.cand_accuracy.iloc[0]),
                    steps=int(cost["optimizer_steps"]))
    return None


def smt_point(n: int) -> dict | None:
    if n == 8:
        for f in glob.glob(str(ROOT / "runs" / "mqar03_pilot01long_*" / "long" / "baseline" / "tuning_summary.csv")):
            s = pd.read_csv(f)
            r = s[(s.stage == "smt") & (s.split == "tuning") & (s.model == "teacher")]
            q = s[(s.stage == "dmt") & (s.split == "tuning") & (s.model == "rnn")]
            res = json.loads((Path(f).parent / "result.json").read_text())
            return dict(accuracy=float(r.full_vocab_acc.iloc[0]), cand_accuracy=float(r.cand_acc.iloc[0]),
                        rnn_after_dmt=float(q.full_vocab_acc.iloc[0]) if len(q) else None, steps=res["smt_steps"])
        return None
    for f in glob.glob(str(ROOT / "runs" / f"mqar03_pilot01arch_kv{n}_*" / "arch" / f"kv{n}" / "teacher_summary.csv")):
        s = pd.read_csv(f)
        r = s[s.split == "tuning"]
        return dict(accuracy=float(r.teacher_long_full_vocab.iloc[0]), cand_accuracy=float(r.teacher_long_cand.iloc[0]),
                    steps=6000)
    return None


def main():
    rows = []
    for n in (2, 4, 8):
        for model, point in (("Transformer (causal)", baseline_point("tf_lr1e-3" if n == 8 else f"tf_kv{n}")),
                             ("RNN, full BPTT", baseline_point("rnn_lr1e-3" if n == 8 else f"rnn_kv{n}")),
                             ("SMT teacher", smt_point(n))):
            if point:
                rows.append(dict(pairs=n, model=model, **point))
    table = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    table.to_csv(OUT / "table_pairs_sweep.csv", index=False)
    color = {"Transformer (causal)": "#2a78d6", "RNN, full BPTT": "#1baf7a", "SMT teacher": "#eb6834"}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    for ax, col, title, chance in ((axes[0], "accuracy", "а) по всему словарю", None),
                                   (axes[1], "cand_accuracy", "б) среди записанных значений", "1/N")):
        V.style_axes(ax, title, "точность, дальние ответы (tuning)", "число пар ключ–значение N")
        for model, g in table.groupby("model"):
            ax.plot(g.pairs, g[col], color=color[model], linewidth=2, marker="o", markersize=6, label=model)
        if chance:
            ax.plot([2, 4, 8], [1 / 2, 1 / 4, 1 / 8], color=V.MUTED, linestyle=(0, (4, 3)), linewidth=1, label="угадывание 1/N")
        else:
            ax.axhline(1 / 64, color=V.MUTED, linestyle=(0, (4, 3)), linewidth=1, label="угадывание 1/64")
        ax.set_xticks([2, 4, 8]); V.pct(ax)
        ax.legend(fontsize=7, frameon=False, loc="center right")
    fig.suptitle("MQAR (словарь 128, T = 64): где модели перестают связывать ключ со значением", x=0.01, ha="left",
                 fontsize=11, color=V.INK)
    fig.savefig(OUT / "figA2_pairs_sweep.png", dpi=150); plt.close(fig)
    print(table.round(3).to_string())


if __name__ == "__main__":
    main()
