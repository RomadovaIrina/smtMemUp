"""Interim figures (14.10.2026): Stack final fits in progress, H2 analysis correction, MQAR detectors/selection/baseline.
Nothing here uses the test split. Every figure says which split it reads (validation = checkpoint-selection set,
tuning = alpha-selection set). Runs on CPU from saved files (+ one small H2 evaluation of seed7/baseline on 256 tuning
sequences). Re-run any time: python make_interim_figures.py  ->  interim_results_2026-10-14/
"""
import json
import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import smt_memup_stack as S

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "interim_results_2026-10-14"
RUN = ROOT / "runs" / "stack02_4c69a2819ab8"
MQ = sorted((ROOT / "runs").glob("mqar03_pilot01_*"))
MQ_OLD = ROOT / "runs" / "obsolete_mqar03_pilot01_037cb754085a_undertrained_detectors"
VARIANTS = ["baseline", "uniform", "random_k", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"]
LABEL = {"baseline": "baseline (α = 0)", "uniform": "uniform", "random_k": "random-K", "surprise_k": "surprise-K (окно 8)",
         "surprise_ctx_k": "surprise-K по входам декодера (доп.)", "oracle_prefix_k": "oracle (доп.)"}
TICK = {"baseline": "baseline", "uniform": "uniform", "random_k": "random-K", "surprise_k": "surprise-K",
        "surprise_ctx_k": "ctx (доп.)", "oracle_prefix_k": "oracle (доп.)"}
COLOR = {"baseline": "#8a8984", "uniform": "#2a78d6", "random_k": "#eb6834", "surprise_k": "#1baf7a",
         "surprise_ctx_k": "#eda100", "oracle_prefix_k": "#e87ba4"}
SEEDS = [107, 7, 17, 27]
SEED_MARK = {107: "D", 7: "o", 17: "s", 27: "^"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e0"


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": .8, "lines.linewidth": 2, "axes.edgecolor": "#8a8984",
                         "axes.labelcolor": MUTED, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "figure.dpi": 130, "savefig.bbox": "tight", "legend.frameon": False, "font.size": 10})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def stack_histories():
    alphas = json.loads((RUN / "locked_alphas.json").read_text())["alphas"]
    rows = []
    for v in VARIANTS:
        for seed in SEEDS:
            d = (RUN / "tuning" / ("baseline" if v == "baseline" else f"{v}_a{alphas[v]}")) if seed == 107 \
                else RUN / "final" / f"seed{seed}" / v
            for stage in ("smt", "dmt"):
                f = d / f"{stage}_history.csv"
                if not f.exists():
                    continue
                h = pd.read_csv(f).dropna(subset=["val_teacher_long_acc"])
                h = h[["step", "val_teacher_long_acc", "val_rnn_long_acc", "val_teacher_long_ce", "val_rnn_long_ce"]].copy()
                h["variant"], h["seed"], h["stage"] = v, seed, stage
                h["complete"] = (d / "completed.pt").exists() if seed != 107 else True
                rows.append(h)
    return pd.concat(rows, ignore_index=True)


def fig_stack_curves(h):
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.2), sharey="row")
    for j, seed in enumerate(SEEDS):
        for v in VARIANTS:
            for i, (stage, col) in enumerate((("smt", "val_teacher_long_acc"), ("dmt", "val_rnn_long_acc"))):
                d = h[(h.seed == seed) & (h.variant == v) & (h.stage == stage)]
                if len(d):
                    ax[i, j].plot(d.step, d[col], color=COLOR[v], ls="--" if v == "baseline" else "-", lw=1.8,
                                  label=LABEL[v] if (i, j) == (0, 0) else None)
        ax[0, j].set_title(("seed 107 — подбор α" if seed == 107 else f"seed {seed} — финал"), fontsize=10)
        ax[1, j].set_xlabel("шаг")
        for i in range(2):
            ax[i, j].axhline(.25, color=MUTED, ls=":", lw=1)
    ax[0, 0].set_ylabel("teacher (SMT): точность,\nдальние ответы")
    ax[1, 0].set_ylabel("RNN (DMT): точность,\nдальние ответы")
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="lower center", ncol=6, fontsize=8.5, bbox_to_anchor=(.5, -.03))
    fig.suptitle("Stack: ход обучения на VALIDATION (давность ≥ 8; пунктир — 25% случайное). Финал ещё идёт: "
                 "отсутствующие линии — обучения не начаты/не закончены. Не test.", x=.01, ha="left", fontsize=10.5)
    save(fig, "01_stack_validation_curves_by_seed")


def stack_best(h):
    rows = []
    for (v, seed, stage), d in h.groupby(["variant", "seed", "stage"]):
        steps = 3000 if stage == "smt" else 1500
        if d.step.max() < steps:
            continue                                       # stage not finished
        col = "val_teacher_long" if stage == "smt" else "val_rnn_long"
        b = d.loc[d[f"{col}_ce"].idxmin()]
        rows.append(dict(variant=v, seed=seed, stage=stage, best_step=int(b.step), acc=b[f"{col}_acc"], ce=b[f"{col}_ce"]))
    return pd.DataFrame(rows)


def fig_stack_best(b):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), sharey=True)
    for i, (stage, title) in enumerate((("smt", "teacher после SMT"), ("dmt", "RNN после DMT (итоговая модель)"))):
        for j, v in enumerate(VARIANTS):
            for seed in SEEDS:
                d = b[(b.variant == v) & (b.seed == seed) & (b.stage == stage)]
                if d.empty:
                    continue
                x = j + (SEEDS.index(seed) - 1.5) * .12
                ax[i].scatter(x, d.acc, marker=SEED_MARK[seed], s=48, color="white" if seed == 107 else COLOR[v],
                              edgecolor=COLOR[v], lw=1.6, zorder=3)
        ax[i].set_xticks(range(len(VARIANTS)), [TICK[v] for v in VARIANTS], fontsize=8.5, rotation=20)
        ax[i].set_title(title, fontsize=10)
        ax[i].axhline(.25, color=MUTED, ls=":", lw=1)
    ax[0].set_ylabel("точность на дальних ответах (давность ≥ 8),\nлучший checkpoint по validation CE")
    for seed, mk in SEED_MARK.items():
        ax[1].scatter([], [], marker=mk, s=40, color="white" if seed == 107 else MUTED, edgecolor=MUTED,
                      label="seed 107 (подбор α, другой набор для α)" if seed == 107 else f"seed {seed} (финал)")
    fig.legend(*ax[1].get_legend_handles_labels(), loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(.5, -.08))
    fig.suptitle("Stack, VALIDATION: разброс между seeds сравним с разницей вариантов. Показаны только законченные этапы. Не test.",
                 x=.01, ha="left", fontsize=10.5)
    save(fig, "02_stack_seed_variability")


def h2_demo():
    """seed7/baseline (final, best checkpoints) on the first 256 TUNING sequences, CPU."""
    import h2_retention as H
    cache = OUT / "table_h2_demo_seed7_baseline_tuning.csv"
    if cache.exists():
        return pd.read_csv(cache)
    S.sources(ROOT)
    cfg, dev = S.Config(), torch.device("cpu")
    torch.set_num_threads(4)
    saved = torch.load(RUN / "final" / "seed7" / "baseline" / "completed.pt", map_location="cpu", weights_only=True)
    t, r = S.load_fitted(cfg, saved, dev, "dmt")
    tun = S.slice_batch(S.sample_stack(cfg, cfg.tuning_examples, S.derive_seed(502, 0)), 0, 256)
    pr = H.retention_by_source(t, r, tun, cfg, dev)
    pr["seed"], pr["variant"], pr["checkpoint"] = 7, "baseline", "best"
    sm = H.summarize_retention(pr, cfg)
    # the original probe's view: all long answers with q >= L, no split, no reference on the same answers
    roll = pr[(pr.model == "rnn_from_teacher") & (pr.readout == "shared") & (pr.delay >= cfg.long_delay)]
    naive = roll.groupby("horizon").correct.mean().rename("acc").reset_index().assign(set="исходный probe: все дальние ответы с q ≥ L")
    sm = pd.concat([sm, naive.assign(readout="shared", source="all")], ignore_index=True)
    sm.to_csv(cache, index=False)
    return sm


def fig_h2(sm):
    sh = sm[sm.readout == "shared"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1.5, 1, 1]))
    nv = sh[sh.set.str.startswith("исходный")].sort_values("horizon")
    ax[0].plot(nv.horizon, 100 * nv.acc, marker="o", color=MUTED, label="исходный probe: все дальние ответы с q ≥ L")
    for src, ls, lab in (("source<=start", "-", "элемент уже в стеке на старте"), ("source>start", ":", "элемент положен во время rollout")):
        d = sh[sh.set.str.contains("by source") & (sh.source == src)].sort_values("horizon")
        ax[0].plot(d.horizon, 100 * d.acc, marker="o", color=COLOR["uniform"], ls=ls, label=f"RNN: {lab}")
        ax[0].plot(d.horizon, 100 * d.teacher_acc, marker="s", ms=4, color=COLOR["random_k"], ls=ls, lw=1.2,
                   label=f"teacher на тех же ответах: {lab}")
    ax[0].set(xlabel="L — обновлений RNN от состояния teacher", ylabel="точность, %",
              title="Состав ответов меняется с L")
    ax[0].legend(fontsize=7.5, loc="lower left")
    fx = sh[sh.set.str.startswith("fixed")].sort_values("horizon")
    n = int(fx.n.iloc[0])
    ax[1].plot(fx.horizon, 100 * fx.gap_vs_teacher_acc, marker="o", color=COLOR["uniform"])
    ax[1].set(xlabel="L", ylabel="точность RNN − teacher, п.п.", title=f"Те же {n} ответов при всех L: точность")
    ax[2].plot(fx.horizon, fx.gap_vs_teacher_ce, marker="o", color=COLOR["random_k"])
    ax[2].set(xlabel="L", ylabel="CE RNN − CE teacher", title=f"Те же {n} ответов: CE (больше 0 — потеря)")
    for a in ax[1:]:
        a.axhline(0, color=INK, lw=1)
        a.set_xticks([0, 1, 2, 4, 8, 16])
    ax[0].set_xticks([0, 1, 2, 4, 8, 16, 32])
    fig.suptitle("H2, демонстрация метода: seed7/baseline, 256 последовательностей TUNING-набора, голова teacher. "
                 "Один seed, не test. Справа — только ответы с давностью > 16: для них элемент в стеке на старте при любом L ≤ 16.",
                 x=.01, ha="left", fontsize=10)
    save(fig, "03_h2_method_seed7_baseline")


def detector_history(path):
    s = torch.load(path, map_location="cpu", weights_only=True)
    return pd.DataFrame([r for r in s["history"] if "ce" in r])


def fig_mqar_detectors():
    if not MQ:
        return
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for d, lab, ls in ((MQ_OLD, "настройки Stack: emb 16, GRU 64, 2000 шагов", ":"), (MQ[-1], "MQAR: emb 64, GRU 128, 20 000 шагов", "-")):
        if not (d / "detector_window.pt").exists():
            continue
        w = detector_history(d / "detector_window.pt")
        c = detector_history(d / "detector_context.pt")
        ax[0].plot(w.step, w.auc_pair_outside, color=COLOR["surprise_k"], ls=ls, label=lab)
        ax[1].plot(w.step, w.acc_pair_in_window, color=COLOR["surprise_k"], ls=ls, label=lab)
        ax[2].plot(c.step, c.auc_prefix_source, color=COLOR["surprise_ctx_k"], ls=ls, label=lab)
    ax[0].set(title="Окно 24: AUC «пара вне окна»", ylabel="AUC (идеал 1,0)", xlabel="шаг", ylim=(.45, 1.02))
    ax[1].set(title="Окно 24: точность при видимой паре", ylabel="точность (идеал ≈ 1,0)", xlabel="шаг", ylim=(0, 1))
    ax[1].text(.5, .3, "плато ≈ «одно из значений в окне»:\nдетектор не осваивает связывание,\nно ранжирует почти идеально",
               transform=ax[1].transAxes, color=MUTED, fontsize=8.5, ha="center")
    ax[2].set(title="Входы декодера: AUC «пара до t»", ylabel="AUC (идеал 1,0)", xlabel="шаг", ylim=(.45, 1.02))
    for a in ax:
        a.legend(fontsize=8, loc="lower right")
    fig.suptitle("MQAR pilot01: обучение детекторов (VALIDATION). Пунктир — первая попытка с настройками Stack (отброшена до обучения SMT).",
                 x=.01, ha="left", fontsize=10.5)
    save(fig, "04_mqar_detectors")


def fig_mqar_selection():
    if not MQ:
        return
    learned = pd.read_csv(MQ[-1] / "selection_audit.csv")
    learned = learned[learned.anchors == "all anchors"].set_index("rule")
    ideal = pd.read_csv(ROOT / "outputs_mqar" / "table_selection_audit_ideal.csv")
    ideal = ideal[(ideal["конфигурация"] == "pilot01") & (ideal.anchors == "all anchors")].set_index("rule")
    rules = [("random_k (exact expectation)", "random-K"), ("surprise_k", "surprise-K\nокно 24"),
             ("surprise_ctx_k", "surprise-K\nвходы декодера\n(доп.)"), ("oracle_prefix_k", "oracle\n(доп.)")]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    for i, (col, title) in enumerate((("memory_dependent", "ответу нужна память m_t"),
                                      ("long_memory_dependent", "нужна память и давность ≥ 24"))):
        x = np.arange(len(rules))
        lv = [learned.loc[r, col] for r, _ in rules]
        iv = [ideal.loc[r, col] if r in ideal.index else np.nan for r, _ in rules]
        ax[i].bar(x - .2, lv, .38, color=COLOR["uniform"], label="обученные детекторы")
        ax[i].bar(x + .2, iv, .38, color="#86b6ef", label="идеальные правила (до обучения)")
        for xx, val in zip(x - .2, lv):
            ax[i].text(xx, val + .015, f"{val:.2f}", ha="center", fontsize=8.5)
        ax[i].set_xticks(x, [n for _, n in rules], fontsize=8.5)
        ax[i].set_title(f"Доля выбранных пар: {title}", fontsize=10)
        ax[i].set_ylim(0, 1.1)
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8.5, bbox_to_anchor=(.5, -.06))
    fig.suptitle("MQAR pilot01 (128 последовательностей TUNING): каким ответам отбор даёт вес. Почти всем ответам нужна память, "
                 "поэтому surprise-K в основном сдвигает вес к давним ответам.", x=.01, ha="left", fontsize=10)
    save(fig, "05_mqar_selection")


def fig_mqar_baseline():
    if not MQ:
        return
    d = MQ[-1] / "tuning" / "baseline"
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for i, stage in enumerate(("smt", "dmt")):
        f = d / f"{stage}_history.csv"
        ax[i].set(title=("teacher, SMT" if stage == "smt" else "RNN, DMT") + ": дальние ответы (давность ≥ 24)",
                  xlabel="шаг", ylim=(0, .5))
        if not f.exists():
            ax[i].text(.5, .5, "этап ещё не начат", transform=ax[i].transAxes, ha="center", color=MUTED)
            continue
        h = pd.read_csv(f).dropna(subset=["val_teacher_long_acc"])
        m = "teacher" if stage == "smt" else "rnn"
        ax[i].plot(h.step, h[f"val_{m}_long_cand_acc"], color=COLOR["uniform"], marker="o", ms=4, label="среди 8 записанных значений")
        ax[i].plot(h.step, h[f"val_{m}_long_acc"], color=COLOR["random_k"], marker="o", ms=4, label="по всему словарю")
        ax[2].plot(h.step + (0 if stage == "smt" else 3000), h[f"val_{m}_pred_in_written"], marker="o", ms=4,
                   color=COLOR["uniform"] if stage == "smt" else COLOR["surprise_k"], label=f"{m} ({stage.upper()})")
        thr = .225 if stage == "smt" else .175
        for y, txt, c in ((.125, "12,5% = 1/8", MUTED), (.157, "15,7% порядок", COLOR["oracle_prefix_k"]),
                          (thr, f"порог gate {thr * 100:.1f}%".replace(".", ","), INK)):
            ax[i].axhline(y, color=c, ls=":" if y < thr else "--", lw=1)
            ax[i].text(1.0, y, " " + txt, transform=ax[i].get_yaxis_transform(), color=c, fontsize=7.5, va="center")
        ax[i].legend(fontsize=8, loc="upper left")
    ax[0].set_ylabel("точность на validation")
    ax[2].set(title="Ответ — одно из записанных значений", xlabel="шаг (SMT, затем DMT)", ylim=(0, 1.02))
    ax[2].legend(fontsize=8, loc="lower right")
    fig.subplots_adjust(wspace=.45)
    fig.suptitle("MQAR pilot01, baseline (seed 107), VALIDATION по ходу обучения. Gate — на tuning-наборе с перемешанным "
                 "порядком запросов (mqar_baseline_gate.json).", x=.01, ha="left", fontsize=10)
    save(fig, "06_mqar_baseline_binding")


def main():
    OUT.mkdir(exist_ok=True)
    style()
    h = stack_histories()
    h.to_csv(OUT / "table_stack_validation_histories.csv", index=False)
    fig_stack_curves(h)
    b = stack_best(h)
    b.to_csv(OUT / "table_stack_best_validation_by_seed.csv", index=False)
    fig_stack_best(b)
    fig_h2(h2_demo())
    fig_mqar_detectors()
    fig_mqar_selection()
    fig_mqar_baseline()
    if MQ and (MQ[-1] / "baseline_gate.json").exists():
        (OUT / "mqar_baseline_gate.json").write_text((MQ[-1] / "baseline_gate.json").read_text())
    print("written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
