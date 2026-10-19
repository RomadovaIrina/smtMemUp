#!/usr/bin/env python3
"""Tables and figures of the 19.10.2026 runs (validation / tuning only; CPU; trains nothing):
  * Stack ablations of SMT (checks/smt_memup_ablation.py)  -> outputs_ablation/
  * MQAR length sweep (T = 16 / 32; checks/smt_memup_mqar_probe_arch.py, mqar_baselines.py) -> outputs_mqar/

  python make_ablation_outputs.py
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
OUT = ROOT / "outputs_ablation"
NAMES = {"full": "SMT, Tf = 8 (все компоненты)", "no_dyn": "без L_dyn", "detached": "L_dyn не идёт в энкодер (detached)",
         "no_unif": "без L_unif", "tf1": "окно Tf = 1", "tf4": "окно Tf = 4", "tf16": "окно Tf = 16",
         "tf64": "окно Tf = T = 64", "mem4": "память 4 токена", "mem16": "память 16 токенов"}
ORDER = list(NAMES)
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#6b6a64", "#1c5cab"]


def ablation_table() -> tuple[pd.DataFrame, dict]:
    rows, hist = [], {}
    for d in glob.glob(str(ROOT / "runs" / "stackv4_*" / "ablation" / "*/")):
        d = Path(d); name = d.name
        if not (d / "smt_history.csv").exists():
            continue
        s = pd.read_csv(d / "smt_history.csv").dropna(subset=["val_teacher_long_acc"])
        m = pd.read_csv(d / "dmt_history.csv").dropna(subset=["val_rnn_long_acc"]) if (d / "dmt_history.csv").exists() else None
        hist[name] = (s, m)
        i = s.val_teacher_long_ce.idxmin()
        first = lambda h, c, x: (int(h[h[c] >= x].step.min()) if (h[c] >= x).any() else None)
        rows.append(dict(name=name, what=NAMES.get(name, name), smt_steps_done=int(s.step.max()),
                         teacher=s.val_teacher_long_acc[i], teacher_to_80=first(s, "val_teacher_long_acc", .8),
                         teacher_to_95=first(s, "val_teacher_long_acc", .95), rnn_before_dmt=s.val_rnn_long_acc[i],
                         far_probe=s.val_far_memory_acc.max() if "val_far_memory_acc" in s else None,
                         rnn_after_dmt=(m.val_rnn_long_acc[m.val_rnn_long_ce.idxmin()] if m is not None and len(m) else None),
                         dmt_steps_done=int(m.step.max()) if m is not None and len(m) else 0))
    t = pd.DataFrame(rows)
    if len(t):
        t["order"] = t.name.map({n: i for i, n in enumerate(ORDER)}); t = t.sort_values("order").drop(columns="order")
    return t, hist


def ablation_figure(hist: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for ax, key, title, xl in ((axes[0], 0, "а) SMT: teacher, дальние ответы", "шаг SMT"),
                               (axes[1], 1, "б) DMT: итоговая RNN, дальние ответы", "шаг DMT")):
        V.style_axes(ax, title, "точность (validation)", xl)
        for i, name in enumerate(n for n in ORDER if n in hist):
            h = hist[name][key]
            if h is None or not len(h):
                continue
            col = "val_teacher_long_acc" if key == 0 else "val_rnn_long_acc"
            ax.plot(h.step, h[col], color=COLORS[ORDER.index(name)], linewidth=2, label=NAMES[name],
                    linestyle="-" if name in ("full", "no_dyn", "detached", "no_unif") else "--")
        V.pct(ax); ax.axhline(0.25, color=V.MUTED, linewidth=1, linestyle=(0, (4, 3)))
        ax.legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Stack: абляции SMT (seed 107, бюджет SMT 11k / DMT 5k, validation)", x=0.01, ha="left", fontsize=11, color=V.INK)
    fig.savefig(path, dpi=150); plt.close(fig)


def length_sweep() -> pd.DataFrame:
    rows = []
    for tag, T, N in (("T16kv2", 16, 2), ("T32kv2", 32, 2), ("T32kv4", 32, 4)):
        for f in glob.glob(str(ROOT / "runs" / f"mqar03_pilot01arch_{tag}_*" / "arch" / tag / "teacher_summary.csv")):
            s = pd.read_csv(f); r = s[s.split == "tuning"].iloc[0]
            rows.append(dict(T=T, pairs=N, model="SMT teacher", long_acc=r.teacher_long_full_vocab, long_cand_acc=r.teacher_long_cand,
                             all_cand_acc=r.teacher_all_cand, steps=6000))
        for prefix, model in (("tf", "Transformer (causal)"), ("rnn", "RNN, full BPTT")):
            for f in glob.glob(str(ROOT / "runs" / "mqar_baselines_*" / f"{prefix}_{tag}" / "summary.csv")):
                s = pd.read_csv(f)
                g = lambda grp: s[(s.group == grp) & (s.split == "tuning") & (s.checkpoint == "best")].iloc[0]
                lg = s[s.group.str.startswith("delay>=") & (s.split == "tuning") & (s.checkpoint == "best")].iloc[0]
                cost = json.loads((Path(f).parent / "cost.json").read_text())
                rows.append(dict(T=T, pairs=N, model=model, long_acc=lg.accuracy, long_cand_acc=lg.cand_accuracy,
                                 all_cand_acc=g("all").cand_accuracy, steps=cost["optimizer_steps"]))
    return pd.DataFrame(rows)


def md(df: pd.DataFrame, pct_cols) -> str:
    if not len(df):
        return "_ещё нет данных_\n"
    out = "| " + " | ".join(df.columns) + " |\n|" + "---|" * len(df.columns) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(f"{100 * r[c]:.1f}%" if c in pct_cols and pd.notna(r[c]) else ("—" if pd.isna(r[c]) else str(r[c]))
                                 for c in df.columns) + " |\n"
    return out


def main():
    OUT.mkdir(exist_ok=True)
    t, hist = ablation_table()
    t.to_csv(OUT / "table_ablation.csv", index=False)
    if hist:
        ablation_figure(hist, OUT / "figB1_ablation_curves.png")
    ls = length_sweep()
    ls.to_csv(ROOT / "outputs_mqar" / "table_length_sweep.csv", index=False)
    pc = ["teacher", "rnn_before_dmt", "far_probe", "rnn_after_dmt"]
    (OUT / "README.md").write_text(
        "# 19.10.2026: абляции SMT на Stack и свип длины MQAR (validation / tuning, seed 107)\n\n"
        "Собирает `python make_ablation_outputs.py`. Полные кривые — `tensorboard --logdir tensorboard`.\n\n"
        "## Stack: абляции (бюджет SMT 11k / DMT 5k)\n![](figB1_ablation_curves.png)\n\n"
        + md(t.drop(columns=["name"]) if len(t) else t, pc)
        + "\n## MQAR: длина примера (вся последовательность в окне энкодера-декодера, Tc = Tf = T)\n"
        + md(ls, ["long_acc", "long_cand_acc", "all_cand_acc"])
        + "\nУгадывание среди записанных значений = 1/N.\n")
    print(t.round(3).to_string()); print(ls.round(3).to_string())


if __name__ == "__main__":
    main()
