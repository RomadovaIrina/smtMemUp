"""Version 2 (short SMT window + MemUP far-target supervision): tables and figures.

  python make_v2_outputs.py --validation   # interim, from validation histories of finished fits (no test)
  python make_v2_outputs.py --test         # after `smt_memup_window.py --phase report --confirm-test` (user permission)

Writes outputs_v2_window/. The pre-registered comparisons and decision rule are in proposals/preregistration_v2.md.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import smt_memup_stack as S

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs_v2_window"
V1 = ROOT / "runs" / "stack02_4c69a2819ab8"
VARIANTS = ["full_window", "base_window", "uniform_far", "random_far", "surprise_far", "ctx_far", "oracle_far"]
LABEL = {"full_window": "полное окно (версия 1, baseline)", "base_window": "короткое окно, без добавки",
         "uniform_far": "+ все дальние цели", "random_far": "+ K случайных дальних", "surprise_far": "+ K неожиданных дальних (окно 8)",
         "ctx_far": "+ K неожиданных по входам декодера (доп.)", "oracle_far": "+ K дальних из памяти (oracle, доп.)"}
TICK = {"full_window": "полное\nокно", "base_window": "короткое\nокно", "uniform_far": "uniform\nfar", "random_far": "random\nfar",
        "surprise_far": "surprise\nfar", "ctx_far": "ctx far\n(доп.)", "oracle_far": "oracle far\n(доп.)"}
COLOR = {"full_window": "#52514e", "base_window": "#8a8984", "uniform_far": "#2a78d6", "random_far": "#eb6834",
         "surprise_far": "#1baf7a", "ctx_far": "#eda100", "oracle_far": "#e87ba4"}
SEED_MARK = {7: "o", 17: "s", 27: "^"}
PRIMARY = [("random_far", "surprise_far")]
FAR = [("base_window", v) for v in ("uniform_far", "random_far", "surprise_far")]
REF = [("uniform_far", "surprise_far"), ("base_window", "full_window"), ("full_window", "surprise_far")]
DIAG = [(r, v) for v in ("ctx_far", "oracle_far") for r in ("random_far", "surprise_far")]
COMPS = PRIMARY + FAR + REF + DIAG


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                         "lines.linewidth": 2, "axes.edgecolor": "#8a8984", "axes.labelcolor": "#52514e",
                         "xtick.color": "#52514e", "ytick.color": "#52514e", "figure.dpi": 130, "savefig.bbox": "tight",
                         "legend.frameon": False, "font.size": 10})


def md_table(df, fmt="{:.3f}"):
    f = lambda v: fmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    return ("| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
            + "".join("| " + " | ".join(f(v) for v in r) + " |\n" for r in df.itertuples(index=False)))


def run_dir():
    d = sorted((ROOT / "runs").glob("stackwin_*"))
    assert len(d) == 1, d
    return d[0]


def validation_table(run):
    rows = []
    for seed in (7, 17, 27):
        for v in VARIANTS:
            d = (V1 / "final" / f"seed{seed}" / "baseline") if v == "full_window" else run / "final" / f"seed{seed}" / v
            if not (d / "completed.pt").exists():
                continue
            r = dict(seed=seed, variant=v)
            for st, col in (("smt", "val_teacher_long"), ("dmt", "val_rnn_long")):
                h = pd.read_csv(d / f"{st}_history.csv").dropna(subset=[col + "_ce"])
                b = h.loc[h[col + "_ce"].idxmin()]
                r[f"{st}_acc"], r[f"{st}_ce"] = b[col + "_acc"], b[col + "_ce"]
                if st == "smt" and "val_far_memory_acc" in h:
                    r["far_memory_acc"] = b.get("val_far_memory_acc", np.nan)
            rows.append(r)
    return pd.DataFrame(rows)


def fig_validation(t):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for i, (col, title) in enumerate((("smt_acc", "teacher после SMT"), ("dmt_acc", "RNN после DMT (итоговая)"))):
        for j, v in enumerate(VARIANTS):
            d = t[t.variant == v]
            for _, r in d.iterrows():
                ax[i].scatter(j + (r.seed - 17) / 60, 100 * r[col], marker=SEED_MARK[r.seed], s=40, color=COLOR[v], zorder=3)
            if len(d):
                ax[i].hlines(100 * d[col].mean(), j - .28, j + .28, color=COLOR[v], lw=3)
        ax[i].set_xticks(range(len(VARIANTS)), [TICK[v] for v in VARIANTS], fontsize=8)
        ax[i].set_title(title, fontsize=10)
        ax[i].axhline(25, color="#52514e", ls=":", lw=1)
    ax[0].set_ylabel("точность на дальних ответах (давность ≥ 8), %")
    fig.suptitle("Версия 2, VALIDATION (набор выбора checkpoint; не test). Черта — среднее по готовым seeds.", x=.01, ha="left", fontsize=10.5)
    fig.savefig(OUT / "figV2_validation.png"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validation", action="store_true")
    ap.add_argument("--test", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    style()
    run = run_dir()
    cfg = S.Config()
    if a.validation:
        t = validation_table(run)
        t.to_csv(OUT / "table_validation_by_seed.csv", index=False)
        fig_validation(t)
        piv = t.pivot(index="variant", columns="seed", values="dmt_acc").reindex(VARIANTS) * 100
        piv["среднее"] = piv.mean(axis=1)
        piv = piv.reset_index().rename(columns={"variant": "вариант"})
        piv["вариант"] = piv["вариант"].map(LABEL)
        print(md_table(piv, "{:.1f}"))
    if a.test:
        ev = pd.read_csv(run / "test_events.csv")
        ev = ev[~((ev.model == "teacher") & (ev.stage == "dmt"))]
        iv = pd.read_csv(run / "test_intervals.csv")
        ps = pd.read_csv(run / "test_per_seed.csv")
        far = pd.read_csv(run / "test_far_memory.csv")
        sm = S.summarize(ev, cfg, keys=("checkpoint", "seed", "variant", "stage", "model"))
        for name, df in (("table_test_summary", sm), ("table_test_intervals", iv), ("table_test_per_seed", ps),
                         ("table_test_far_memory", far)):
            df.to_csv(OUT / f"{name}.csv", index=False)
        m = iv[(iv.model == "rnn") & (iv.stage == "dmt")]
        rows = []
        for ref, var in PRIMARY + [("base_window", "random_far")]:
            ce = m[(m.reference == ref) & (m.variant == var) & (m.metric == "ce")].iloc[0]
            acc = m[(m.reference == ref) & (m.variant == var) & (m.metric == "correct")].iloc[0]
            c = [ce.delta_mean < 0, ce.delta_mean < 0 and ce.seeds_same_sign == ce.seeds, ce.ci_sequences_high < 0, acc.delta_mean >= 0]
            rows.append(dict(comparison=f"{var} − {ref}", delta_ce=ce.delta_mean, ci=f"[{ce.ci_sequences_low:+.4f}, {ce.ci_sequences_high:+.4f}]",
                             delta_acc=acc.delta_mean, rule=" ".join("✓" if x else "✗" for x in c), passed=all(c)))
        pd.DataFrame(rows).to_csv(OUT / "table_test_decision_rule.csv", index=False)
        print(pd.DataFrame(rows).to_string())
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
