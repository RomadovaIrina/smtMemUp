"""Stack: tables and figures from the single TEST evaluation (after
`python smt_memup_stack.py --phase report --confirm-test --variants <all six>`). No training, nothing is selected here.

Reads runs/<run>/test_events.csv (written by final_report) and writes outputs_test/. Every number is TEST, best
checkpoints unless a column says 'last'. All pre-registered comparisons (preregistration.md, section
'Сравнения') are computed, including the secondary ones against surprise_k that final_report itself does not
write. Optional --h2: the H2 retention probe split by source<=start / source>start (h2_retention.py) on the same
test sequences; it needs the fitted models (GPU if visible, else CPU, slow).

Run: python make_test_outputs.py [--h2] [--gpu-memory-fraction 0.15]
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import smt_memup_stack as S

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs" / "stack02_4c69a2819ab8"
OUT = ROOT / "outputs_test"
VARIANTS = ["baseline", "uniform", "random_k", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"]
SHORT = {"baseline": "baseline", "uniform": "uniform", "random_k": "random-K", "surprise_k": "surprise-K (окно 8)",
         "surprise_ctx_k": "surprise-K по входам декодера (доп.)", "oracle_prefix_k": "oracle (разметка, доп.)"}
TICK = {"baseline": "baseline", "uniform": "uniform", "random_k": "random-K", "surprise_k": "surprise-K",
        "surprise_ctx_k": "surprise-K\nвходы декодера\n(доп.)", "oracle_prefix_k": "oracle\n(доп.)"}
COLOR = {"baseline": "#8a8984", "uniform": "#2a78d6", "random_k": "#eb6834", "surprise_k": "#1baf7a",
         "surprise_ctx_k": "#eda100", "oracle_prefix_k": "#e87ba4"}
MINI = {"baseline": "baseline", "uniform": "uniform", "random_k": "random-K", "surprise_k": "surprise-K",
        "surprise_ctx_k": "ctx (доп.)", "oracle_prefix_k": "oracle (доп.)"}
SEED_MARK = {7: "o", 17: "s", 27: "^"}
INK, MUTED = "#0b0b0b", "#52514e"
# pre-registered comparisons (reference, variant); delta = variant - reference
PRIMARY = [("random_k", "surprise_k"), ("uniform", "surprise_k")]
CONTEXT = [("baseline", v) for v in ("uniform", "random_k", "surprise_k")]
SECONDARY = [(r, v) for v in ("surprise_ctx_k", "oracle_prefix_k") for r in ("random_k", "uniform", "surprise_k")]
COMPARISONS = PRIMARY + CONTEXT + SECONDARY
BLOCK = {**{c: "основное" for c in PRIMARY}, **{c: "контекст" for c in CONTEXT}, **{c: "доп. диагностика" for c in SECONDARY}}
RULE = [("random_k", "surprise_k"), ("uniform", "surprise_k")]          # comparisons the decision rule is applied to


def use_preset_v3(variants_present):
    """Version 3 (smt_memup_v3.py, converged regime): decoder-aware selection is the confirmatory hypothesis."""
    global VARIANTS, PRIMARY, CONTEXT, SECONDARY, COMPARISONS, BLOCK, RULE, TAG
    VARIANTS = [v for v in ["baseline", "uniform", "random_k", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"]
                if v in variants_present]
    PRIMARY = [("baseline", "surprise_ctx_k"), ("random_k", "surprise_ctx_k"), ("uniform", "surprise_ctx_k"),
               ("random_k", "surprise_k"), ("uniform", "surprise_k")]
    CONTEXT = [("baseline", v) for v in ("uniform", "random_k", "surprise_k")] + [("surprise_k", "surprise_ctx_k")]
    SECONDARY = ([(r, "oracle_prefix_k") for r in ("baseline", "random_k", "surprise_ctx_k")]
                 if "oracle_prefix_k" in VARIANTS else [])
    COMPARISONS = PRIMARY + CONTEXT + SECONDARY
    BLOCK = {**{c: "основное" for c in PRIMARY}, **{c: "контекст" for c in CONTEXT}, **{c: "доп. диагностика" for c in SECONDARY}}
    RULE = PRIMARY
    TAG = "TEST версии 3 (сходящийся режим; 2048 новых последовательностей, один раз), 3 новых seeds"
    SHORT["surprise_ctx_k"] = "surprise-K по входам декодера"
    TICK["surprise_ctx_k"] = "surprise-K\nвходы декодера"
    MINI["surprise_ctx_k"] = "ctx"
STAGE_MODEL = {("smt", "teacher"): "teacher после SMT", ("smt", "rnn"): "RNN до DMT", ("dmt", "rnn"): "RNN после DMT"}
BINS, BIN_LABELS = [0, 2, 4, 8, 16, 64], ["1–2", "3–4", "5–8", "9–16", "17+"]
TAG = "TEST (2048 последовательностей, один раз), 3 seeds"


def groups(cfg):
    L = cfg.long_delay
    return {f"delay>={L} (первичная)": lambda e: e[e.delay >= L], "все ответы": lambda e: e,
            f"delay<{L}": lambda e: e[e.delay < L], "delay>=16": lambda e: e[e.delay >= 16]}


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": .8, "lines.linewidth": 2,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 130,
                         "savefig.bbox": "tight", "legend.frameon": False, "font.size": 10})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def md_table(df, floatfmt="{:.3f}"):
    def f(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(f(v) for v in row) + " |\n" for row in df.itertuples(index=False))


def load_events(run):
    f = run / "test_events.csv"
    ev = pd.read_csv(f if f.exists() else run / "test_events.csv.gz")
    assert set(ev.split) == {"test"}, "only the test split belongs here"
    assert set(ev.variant) == set(VARIANTS), f"all six pre-declared variants must be reported: {sorted(set(ev.variant))}"
    assert set(ev.checkpoint) == {"best", "last"}
    ev = ev[~((ev.model == "teacher") & (ev.stage == "dmt"))].copy()   # teacher is frozen in DMT: same as after SMT
    ev["model_stage"] = [STAGE_MODEL[(s, m)] for s, m in zip(ev.stage, ev.model)]
    return ev


def summary_table(ev, cfg):
    rows = []
    for gname, sel in groups(cfg).items():
        g = sel(ev).groupby(["checkpoint", "seed", "variant", "model_stage"]).agg(
            answers=("correct", "size"), accuracy=("correct", "mean"), ce=("ce", "mean"),
            teacher_head_accuracy=("shared_correct", "mean"), teacher_head_ce=("shared_ce", "mean")).reset_index()
        g.insert(0, "group", gname)
        rows.append(g)
    return pd.concat(rows, ignore_index=True)


def paired(ev, cfg):
    """S.paired_deltas (unchanged code from the experiment) on every group and checkpoint."""
    ivs, pss = [], []
    for ckpt in ("best", "last"):
        e = ev[ev.checkpoint == ckpt]
        for gname, sel in groups(cfg).items():
            iv, ps = S.paired_deltas(sel(e), cfg, COMPARISONS, group_min_delay=0)
            for t in (iv, ps):
                t["checkpoint"], t["group"] = ckpt, gname
                t["block"] = [BLOCK[(r, v)] for r, v in zip(t.reference, t.variant)]
            ivs.append(iv.drop(columns="min_delay")); pss.append(ps)
    return pd.concat(ivs, ignore_index=True), pd.concat(pss, ignore_index=True)


def decision_rule(iv, cfg, checkpoint="best"):
    """Pre-registered rule for H1 (both primary comparisons, RNN after DMT, delay >= long_delay)."""
    g = f"delay>={cfg.long_delay} (первичная)"
    m = iv[(iv.model == "rnn") & (iv.stage == "dmt") & (iv.group == g) & (iv.checkpoint == checkpoint)]
    rows = []
    for ref, var in RULE:
        ce = m[(m.reference == ref) & (m.variant == var) & (m.metric == "ce")].iloc[0]
        acc = m[(m.reference == ref) & (m.variant == var) & (m.metric == "correct")].iloc[0]
        c = {"1: средняя ΔCE < 0": ce.delta_mean < 0,
             "2: ΔCE < 0 у каждого seed": bool(ce.delta_mean < 0 and ce.seeds_same_sign == ce.seeds),
             "3: 95% ДИ (последовательности) не содержит 0": bool(ce.ci_sequences_high < 0),
             "4: средняя Δaccuracy ≥ 0": bool(acc.delta_mean >= 0)}
        rows.append(dict(checkpoint=checkpoint, comparison=f"{SHORT[var]} − {SHORT[ref]}", delta_ce=ce.delta_mean,
                         ci_ce=f"[{ce.ci_sequences_low:+.4f}, {ce.ci_sequences_high:+.4f}]",
                         delta_acc=acc.delta_mean, **{k: "да" if v else "нет" for k, v in c.items()},
                         passed=all(c.values())))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures
def fig_primary(sm, cfg):
    g = sm[(sm.group == f"delay>={cfg.long_delay} (первичная)") & (sm.model_stage == "RNN после DMT") & (sm.checkpoint == "best")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4))
    for i, (col, lab) in enumerate((("ce", "CE (меньше — лучше)"), ("accuracy", "точность"))):
        for j, v in enumerate(VARIANTS):
            d = g[g.variant == v]
            for _, r in d.iterrows():
                ax[i].scatter(j + (list(SEED_MARK).index(r.seed) - 1) / 60, r[col], marker=SEED_MARK[r.seed], s=40, color=COLOR[v],
                              edgecolor="white", zorder=3)
            ax[i].hlines(d[col].mean(), j - .28, j + .28, color=COLOR[v], lw=3)
        ax[i].set_xticks(range(len(VARIANTS)), [TICK[v] for v in VARIANTS], fontsize=8.5)
        ax[i].set_ylabel(lab)
    ax[1].axhline(1 / cfg.n_elements, color=MUTED, ls=":", lw=1)
    ax[1].text(-.4, 1 / cfg.n_elements + .01, "случайное угадывание 25%", color=MUTED, fontsize=8)
    for s, mk in SEED_MARK.items():
        ax[0].scatter([], [], marker=mk, color=MUTED, label=f"seed {s}")
    ax[0].legend(fontsize=8, loc="upper right")
    fig.suptitle(f"Первичная метрика: RNN после DMT, дальние ответы (давность ≥ {cfg.long_delay}). Черта — среднее по seeds. {TAG}",
                 x=.01, ha="left", fontsize=10.5)
    save(fig, "figT1_primary_per_seed")


def fig_forest(iv, ps, cfg):
    g = f"delay>={cfg.long_delay} (первичная)"
    sel = lambda t: t[(t.model == "rnn") & (t.stage == "dmt") & (t.group == g)]
    iv, ps = sel(iv), sel(ps)
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    labels = [f"{SHORT[v]} − {SHORT[r]}" for r, v in COMPARISONS]
    for i, (metric, lab, scale) in enumerate((("ce", "ΔCE (меньше 0 — variant лучше)", 1), ("correct", "Δточность, п.п.", 100))):
        for k, (r, v) in enumerate(COMPARISONS):
            y = len(COMPARISONS) - 1 - k
            for ckpt, dy, filled in (("best", .12, True), ("last", -.12, False)):
                row = iv[(iv.reference == r) & (iv.variant == v) & (iv.metric == metric) & (iv.checkpoint == ckpt)]
                if row.empty:
                    continue
                row = row.iloc[0]
                c = COLOR[v]
                ax[i].plot([row.ci_sequences_low * scale, row.ci_sequences_high * scale], [y + dy] * 2, color=c, lw=2)
                ax[i].scatter(row.delta_mean * scale, y + dy, s=45, color=c if filled else "white", edgecolor=c, lw=1.6, zorder=3)
                if ckpt == "best":
                    sd = ps[(ps.reference == r) & (ps.variant == v) & (ps.metric == metric) & (ps.checkpoint == ckpt)]
                    for _, s_ in sd.iterrows():
                        ax[i].scatter(s_.delta * scale, y + .32, marker=SEED_MARK[s_.seed], s=14, color=c, alpha=.8)
        ax[i].axvline(0, color=INK, lw=1)
        for yb in (len(COMPARISONS) - len(PRIMARY) - .5, len(COMPARISONS) - len(PRIMARY) - len(CONTEXT) - .5):
            ax[i].axhline(yb, color=MUTED, lw=.6, ls="--")
        ax[i].set_xlabel(lab)
    ax[0].set_yticks(range(len(COMPARISONS)), labels[::-1], fontsize=8.5)
    ax[0].scatter([], [], s=45, color=MUTED, label="best checkpoint: среднее по seeds, 95% ДИ по последовательностям")
    ax[0].scatter([], [], s=45, color="white", edgecolor=MUTED, label="last checkpoint (проверка устойчивости)")
    seeds_txt = ", ".join(f"{m} {sd}" for sd, m in zip(SEED_MARK, "○□△"))
    ax[0].scatter([], [], s=14, marker="o", color=MUTED, label=f"отдельные seeds ({seeds_txt}), best")
    fig.legend(loc="lower center", ncol=3, fontsize=8.5, bbox_to_anchor=(.5, -.04))
    fig.suptitle(f"Парные разности на одних и тех же test-последовательностях: RNN после DMT, давность ≥ {cfg.long_delay}.\n"
                 f"Сверху вниз: основные, контекст (против baseline), доп. диагностика. ДИ по последовательностям не учитывает "
                 f"разброс seeds. {TAG}", x=.01, ha="left", fontsize=10)
    save(fig, "figT2_paired_deltas")


def fig_stages(sm, cfg):
    g = sm[(sm.group == f"delay>={cfg.long_delay} (первичная)") & (sm.checkpoint == "best")]
    panels = [("teacher после SMT", "accuracy", "teacher после SMT"), ("RNN до DMT", "accuracy", "RNN до DMT (голова teacher)"),
              ("RNN после DMT", "accuracy", "RNN после DMT, своя голова"),
              ("RNN после DMT", "teacher_head_accuracy", "RNN после DMT, голова teacher")]
    fig, ax = plt.subplots(1, len(panels), figsize=(15, 4.2), sharey=True)
    for i, (ms, col, title) in enumerate(panels):
        d = g[g.model_stage == ms]
        for j, v in enumerate(VARIANTS):
            dv = d[d.variant == v]
            ax[i].scatter([j + (list(SEED_MARK).index(s) - 1) / 60 for s in dv.seed], dv[col], c=COLOR[v], s=22, zorder=3)
            ax[i].hlines(dv[col].mean(), j - .3, j + .3, color=COLOR[v], lw=3)
        ax[i].axhline(1 / cfg.n_elements, color=MUTED, ls=":", lw=1)
        ax[i].set_xticks(range(len(VARIANTS)), [MINI[v] for v in VARIANTS], fontsize=8, rotation=40, ha="right")
        ax[i].set_title(title, fontsize=10)
    ax[0].set_ylabel(f"точность, давность ≥ {cfg.long_delay}")
    fig.suptitle(f"Где появляется и теряется эффект: teacher → RNN до DMT → RNN после DMT; своя голова и голова teacher. {TAG}",
                 x=.01, ha="left", fontsize=10.5)
    save(fig, "figT3_stages")


def fig_by_delay(ev):
    e = ev[(ev.checkpoint == "best") & (ev.model_stage == "RNN после DMT")].copy()
    e["bin"] = pd.cut(e.delay, BINS, labels=BIN_LABELS)
    t = e.groupby(["variant", "seed", "bin"], observed=True).correct.mean().reset_index()
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for v in VARIANTS:
        d = t[t.variant == v].groupby("bin", observed=True).correct.agg(["mean", "min", "max"])
        x = np.arange(len(d))
        ax.plot(x, d["mean"], marker="o", color=COLOR[v], label=SHORT[v], ls="--" if v == "baseline" else "-")
        ax.fill_between(x, d["min"], d["max"], color=COLOR[v], alpha=.12)
    ax.set_xticks(range(len(BIN_LABELS)), BIN_LABELS)
    ax.set(xlabel="давность ответа (шаг pop − шаг push)", ylabel="точность RNN после DMT", ylim=(0, 1.02))
    ax.axhline(.25, color=MUTED, ls=":", lw=1)
    ax.legend(fontsize=8.5, loc="lower left")
    ax.set_title(f"Точность по давности (линия — среднее, заливка — min..max по seeds). {TAG}", fontsize=10)
    save(fig, "figT4_by_delay")


def fig_h2(h2):
    fx = h2[h2.set.str.startswith("fixed") & (h2.readout == "shared")]
    bs = h2[h2.set.str.contains("by source") & (h2.readout == "shared")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
    for v in VARIANTS:
        d = fx[fx.variant == v].groupby("horizon").gap_vs_teacher_acc.agg(["mean", "min", "max"])
        ax[0].plot(d.index, 100 * d["mean"], marker="o", color=COLOR[v], label=SHORT[v], ls="--" if v == "baseline" else "-")
        ax[0].fill_between(d.index, 100 * d["min"], 100 * d["max"], color=COLOR[v], alpha=.1)
        for src, ls in (("source<=start", "-"), ("source>start", ":")):
            d = bs[(bs.variant == v) & (bs.source == src)].groupby("horizon").gap_vs_teacher_acc.mean()
            ax[1].plot(d.index, 100 * d.values, marker="o", ms=4, color=COLOR[v], ls=ls)
    ax[1].plot([], [], color=MUTED, ls="-", label="элемент уже в стеке на старте (надо пронести)")
    ax[1].plot([], [], color=MUTED, ls=":", label="элемент положен во время rollout (надо записать)")
    for a in ax:
        a.axhline(0, color=INK, lw=1)
        a.set_xlabel("L — число собственных обновлений RNN от состояния teacher")
        a.set_xticks([0, 1, 2, 4, 8, 16, 32])
    ax[0].set_ylabel("точность RNN − точность teacher на тех же ответах, п.п.")
    ax[0].set_title("Одни и те же ответы при всех L (давность > 16): чистое удержание", fontsize=10)
    ax[1].set_title("Дальние ответы, разделены по моменту push (состав меняется с L)", fontsize=10)
    ax[0].legend(fontsize=8, loc="lower left")
    ax[1].legend(fontsize=8, loc="lower left")
    fig.suptitle(f"H2: теряет ли RNN информацию состояния teacher при последовательных обновлениях (голова teacher). {TAG}",
                 x=.01, ha="left", fontsize=10.5)
    save(fig, "figT5_h2_retention")


# ------------------------------------------------------------------ H2 on test (needs the fitted models)
def run_h2(cfg, args):
    import torch
    import h2_retention as H
    assert (RUN / "test_report.json").exists(), "H2 on test only after the single test report"
    dev = S.setup_runtime(args.threads, args.gpu_memory_fraction)
    S.sources(ROOT)
    test = S.sample_stack(cfg, cfg.test_examples, S.derive_seed(503, 0))
    rep = json.loads((RUN / "test_report.json").read_text())
    assert S.split_hash(test) == rep["test_hash"], "regenerated test differs from the reported one"
    cache = RUN / "h2_test_summary.csv"
    done = pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    for seed in cfg.seeds:
        for v in VARIANTS:
            if len(done) and ((done.seed == seed) & (done.variant == v)).any():
                continue
            saved = torch.load(RUN / "final" / f"seed{seed}" / v / "completed.pt", map_location="cpu", weights_only=True)
            t, r = S.load_fitted(cfg, saved, dev, "dmt", "best")
            pr = H.retention_by_source(t, r, test, cfg, dev)
            pr["seed"], pr["variant"], pr["checkpoint"] = seed, v, "best"
            done = pd.concat([done, H.summarize_retention(pr, cfg)], ignore_index=True)
            done.to_csv(cache, index=False)
            print("H2", seed, v, flush=True)
    return done


def main():
    global RUN, OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(RUN))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--h2", action="store_true")
    ap.add_argument("--preset", choices=["v1", "v3"], default="v1")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.15)
    a = ap.parse_args()
    RUN, OUT = Path(a.run), Path(a.out)
    OUT.mkdir(exist_ok=True)
    style()
    cfg = S.Config()
    if a.preset == "v3":
        f = RUN / "test_events.csv.gz"
        use_preset_v3(set(pd.read_csv(f, usecols=["variant"]).variant))
    ev = load_events(RUN)
    SEED_MARK.clear(); SEED_MARK.update({sd: m for sd, m in zip(sorted(ev.seed.unique()), "os^")})
    sm = summary_table(ev, cfg)
    sm.to_csv(OUT / "table_test_summary.csv", index=False)
    iv, ps = paired(ev, cfg)
    iv.to_csv(OUT / "table_test_paired_intervals.csv", index=False)
    ps.to_csv(OUT / "table_test_paired_per_seed.csv", index=False)
    rule = pd.concat([decision_rule(iv, cfg, c) for c in ("best", "last")], ignore_index=True)
    rule.to_csv(OUT / "table_test_decision_rule.csv", index=False)
    fig_primary(sm, cfg); fig_forest(iv, ps, cfg); fig_stages(sm, cfg); fig_by_delay(ev)
    h2 = run_h2(cfg, a) if a.h2 else (pd.read_csv(RUN / "h2_test_summary.csv") if (RUN / "h2_test_summary.csv").exists() else None)
    if h2 is not None:
        h2.to_csv(OUT / "table_test_h2_retention.csv", index=False)
        fig_h2(h2)
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    sys.exit(main())
