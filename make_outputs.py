"""Builds outputs/: figures and tables for the advisor from saved runs (no training).

Evaluates every finished tuning fit on the TUNING split (the split used for alpha selection),
the H2 retention probe, and reads validation histories. Test is never touched.
Run: python make_outputs.py [--threads 4] [--gpu-memory-fraction 0.12]
"""
import argparse
import json
import re
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import smt_memup_stack as S

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
ORDER = ["baseline", "uniform", "random_k", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"]
LABEL = {"baseline": "baseline", "uniform": "uniform", "random_k": "random-K",
         "surprise_k": "surprise-K (окно r=8)", "surprise_ctx_k": "surprise-K (контекст декодера)*",
         "oracle_prefix_k": "oracle (push ≤ t)*"}
# fixed categorical order (reference palette slots 1-5); baseline is the neutral reference
COLOR = {"baseline": "#8a8984", "uniform": "#2a78d6", "random_k": "#eb6834", "surprise_k": "#1baf7a",
         "surprise_ctx_k": "#eda100", "oracle_prefix_k": "#e87ba4"}
INK, MUTED = "#0b0b0b", "#52514e"
BINS = [0, 2, 4, 8, 16, 64]
BIN_LABELS = ["1–2", "3–4", "5–8", "9–16", "17+"]
NOTE = "tuning-фаза, один seed (107); не результат test"


def style():
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e5e0", "grid.linewidth": .8, "lines.linewidth": 2,
                         "axes.edgecolor": "#8a8984", "axes.labelcolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "figure.dpi": 130,
                         "savefig.bbox": "tight", "legend.frameon": False, "font.size": 10})


def save(fig, name):
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    def f(v):
        return floatfmt.format(v) if isinstance(v, (float, np.floating)) and np.isfinite(v) else str(v)
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    return head + "".join("| " + " | ".join(f(v) for v in row) + " |\n" for row in df.itertuples(index=False))


def fits(run: Path):
    for d in sorted((run / "tuning").iterdir()):
        if not (d / "completed.pt").exists():
            continue
        if d.name == "baseline":
            yield d, "baseline", 0.0
        else:
            m = re.fullmatch(r"(.+)_a([0-9.]+)", d.name)
            yield d, m.group(1), float(m.group(2))


def evaluate_fits(ex, run: Path, cache: Path):
    if (cache / "events.csv.gz").exists():
        ev = pd.read_csv(cache / "events.csv.gz")
        pr = pd.read_csv(cache / "probe.csv")
        done = set(zip(ev.variant, ev.alpha))
    else:
        cache.mkdir(parents=True, exist_ok=True)
        ev, pr, done = pd.DataFrame(), pd.DataFrame(), set()
    cfg, dev = S.replace(ex.cfg, eval_batch_size=32), ex.device   # smaller eval batch: stays within a small GPU share
    new_ev, new_pr = [], []
    for d, variant, alpha in fits(run):
        if (variant, alpha) in done:
            continue
        saved = S.torch.load(d / "completed.pt", map_location="cpu", weights_only=True)
        for stage in ("smt", "dmt"):
            t, r = S.load_fitted(cfg, saved, dev, stage)
            new_ev.append(S.evaluate(t, r, ex.tuning, cfg, dev, variant=variant, alpha=alpha, stage=stage))
            if stage == "dmt":
                p = S.retention_probe(t, r, ex.tuning, cfg, dev, horizons=(0, 1, 2, 4, 8, 16, 32))
                p = p[p.delay >= cfg.long_delay].groupby(["model", "horizon"]).agg(
                    acc=("correct", "mean"), ce=("ce", "mean"), n=("correct", "size")).reset_index()
                p["variant"], p["alpha"] = variant, alpha
                new_pr.append(p)
        print("evaluated", variant, alpha, flush=True)
    if new_ev:
        ev = pd.concat([ev, *new_ev], ignore_index=True)
        pr = pd.concat([pr, *new_pr], ignore_index=True)
        ev.to_csv(cache / "events.csv.gz", index=False)
        pr.to_csv(cache / "probe.csv", index=False)
    return ev, pr


def histories(run: Path):
    rows = []
    for d, variant, alpha in fits(run):
        for stage in ("smt", "dmt"):
            h = pd.read_csv(d / f"{stage}_history.csv").dropna(subset=["val_teacher_long_acc"])
            h = h[["step", "val_teacher_long_acc", "val_rnn_long_acc", "val_teacher_long_ce", "val_rnn_long_ce"]].copy()
            h["variant"], h["alpha"], h["stage"] = variant, alpha, stage
            rows.append(h)
    return pd.concat(rows, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    a = ap.parse_args()
    dev = S.setup_runtime(a.threads, a.gpu_memory_fraction)
    style()
    OUT.mkdir(exist_ok=True)
    cfg = S.Config()
    ex = S.Experiment(cfg, root=ROOT, device=dev, log=lambda s: None)
    run = ex.dir
    long = cfg.long_delay
    readme = [f"# Результаты для обсуждения (собрано из `{run.name}`)\n",
              "Все числа ниже — **данные, аудит отбора, baseline и подбор alpha** на tuning-seed 107. "
              "Это не финальное сравнение: оно будет по трём seeds (7, 17, 27) на отдельном test по правилу "
              "из `preregistration.md`. Checkpoint выбирался по validation, alpha — по tuning-набору. "
              "Варианты со звёздочкой — вторичная диагностика (oracle использует metadata генератора). "
              f"«Дальние pop» — pop с delay ≥ {long} (в Stack давность нечётная, т.е. ≥ 9); случайный уровень 25%.\n"]

    # ---------------- data ----------------
    tun = ex.tuning
    audit = pd.read_csv(run / "data_audit.csv")
    audit.to_csv(OUT / "table_data_audit.csv", index=False)
    w = S.windows(S.slice_batch(tun, 0, 128))
    v = w["valid"]
    off = (w["target_t"] - w["anchor"][:, None])[v].numpy()
    pref = w["prefix"][v].numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12, 3.6))
    dl = tun["delay"][tun["target_mask"]].numpy()
    ax[0].hist(dl, bins=np.arange(1, 65) - .5, color=COLOR["uniform"], rwidth=.85)
    ax[0].axvline(long - .5, color=MUTED, ls="--", lw=1)
    ax[0].text(long + .5, ax[0].get_ylim()[1] * .3, f"дальние pop: {(dl >= long).mean():.0%}", color=MUTED)
    ax[0].set(title="Давность pop: t(pop) − t(push)", xlabel="давность", ylabel="число pop (лог. шкала)", yscale="log")
    share = pd.Series(pref).groupby(off).mean()
    ax[1].plot(share.index, share.values, color=COLOR["uniform"])
    ax[1].set(title="Доля целей, для которых нужна память m_t", xlabel="смещение цели от anchor, k − t",
              ylabel="доля (push ≤ t)", ylim=(0, 1.02))
    fig.suptitle("Stack (авторский генератор): T=64, глубина 8, 4 значения", x=0.01, ha="left", fontsize=11)
    save(fig, "fig1_data")
    readme += ["## 1. Данные\n", "![](fig1_data.png)\n",
               "Слева: большинство pop короткие, дальних около 15%. Справа: цель на смещении 0 (ответ RNN) "
               "всегда требует памяти; дальше по будущему растёт доля целей, чей элемент положен уже после anchor — "
               "их decoder видит во входах и без m_t.\n", md_table(audit) + "\n"]

    # ---------------- selection audit ----------------
    sel = pd.read_csv(run / "selection_audit.csv")
    sel.to_csv(OUT / "table_selection_audit.csv", index=False)
    q = json.loads((run / "detector_quality.json").read_text())
    pd.DataFrame(q).T.to_csv(OUT / "table_detector_quality.csv")
    s = sel[sel.anchors == "all anchors"].set_index("rule")
    rules = ["random_k (exact expectation)", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"]
    names = ["random-K", "surprise-K\n(окно r=8)", "surprise-K\n(контекст декодера)*", "oracle*\n(push ≤ t)"]
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    x = np.arange(len(rules))
    ax.bar(x - .2, s.loc[rules, "memory_dependent"], .38, color=COLOR["uniform"], label="нужна память m_t (push ≤ t)")
    ax.bar(x + .2, s.loc[rules, "long_memory_dependent"], .38, color=COLOR["random_k"], label=f"нужна m_t и delay ≥ {long}")
    for i, r in enumerate(rules):
        for dx, col in ((-.2, "memory_dependent"), (.2, "long_memory_dependent")):
            ax.text(i + dx, s.loc[r, col] + .02, f"{s.loc[r, col]:.0%}", ha="center", fontsize=9, color=INK)
    ceiling = s.loc["oracle_prefix_k", "pairs"] / s.loc["random_k", "pairs"]
    ax.axhline(ceiling, color=MUTED, ls="--", lw=1)
    ax.text(-.45, ceiling + .03, f"потолок доли при K=4: {ceiling:.0%}", color=MUTED, fontsize=9)
    ax.set_xticks(x, names); ax.set_ylim(0, 1.12); ax.set_ylabel("доля выбранных пар (anchor, цель)")
    ax.set_title("Какие цели получает дополнительный loss (tuning-набор, до обучения)")
    ax.legend(loc="upper left")
    save(fig, "fig2_selection")
    readme += ["## 2. Аудит отбора целей (до обучения SMT)\n", "![](fig2_selection.png)\n",
               f"Окно r=8 отделяет цели «вне окна» идеально (AUC {q['window']['auc_outside_window']:.3f}), но "
               f"{s.loc['surprise_k', 'future_source']:.0%} выбранных им пар имеют источник после anchor. "
               "Score по входам декодера (без памяти) почти достигает потолка доли memory-dependent целей.\n",
               md_table(sel[sel.anchors == "all anchors"][["rule", "pairs", "memory_dependent", "long_memory_dependent",
                                                          "future_source", "mean_delay"]]) + "\n"]

    # ---------------- evaluate all tuning fits on the tuning split ----------------
    ev, pr = evaluate_fits(ex, run, run / "analysis_tuning")
    ev["delay_bin"] = pd.cut(ev.delay, BINS, labels=BIN_LABELS)
    lg = ev[ev.delay >= long]
    piv = lg.groupby(["variant", "alpha", "stage", "model"]).agg(acc=("correct", "mean"), ce=("ce", "mean"),
                                                                 shared_acc=("shared_correct", "mean")).reset_index()
    allacc = ev.groupby(["variant", "alpha", "stage", "model"]).correct.mean().rename("acc_all").reset_index()
    piv = piv.merge(allacc, on=["variant", "alpha", "stage", "model"])

    def get(vv, aa, stage, model, col):
        r = piv[(piv.variant == vv) & (piv.alpha == aa) & (piv.stage == stage) & (piv.model == model)]
        return float(r[col].iloc[0]) if len(r) else np.nan

    lock_file = run / "locked_alphas.json"
    locked = json.loads(lock_file.read_text())["alphas"] if lock_file.exists() else {}
    if lock_file.exists():
        shutil.copy(lock_file, OUT / "locked_alphas.json")
    rows = []
    for vv in ORDER:
        for aa in sorted(piv[piv.variant == vv].alpha.unique()):
            rows.append({"вариант": LABEL[vv], "alpha": aa, "выбран": "✓" if locked.get(vv) == aa else "",
                         "teacher: acc дальн.": get(vv, aa, "smt", "teacher", "acc"),
                         "teacher: CE дальн.": get(vv, aa, "smt", "teacher", "ce"),
                         "RNN до DMT: acc": get(vv, aa, "smt", "rnn", "acc"),
                         "RNN после DMT: acc": get(vv, aa, "dmt", "rnn", "acc"),
                         "RNN после DMT: CE": get(vv, aa, "dmt", "rnn", "ce"),
                         "RNN после DMT, голова teacher: acc": get(vv, aa, "dmt", "rnn", "shared_acc"),
                         "RNN после DMT: acc все pop": get(vv, aa, "dmt", "rnn", "acc_all")})
    tune_tab = pd.DataFrame(rows)
    tune_tab.to_csv(OUT / "table_tuning_long_pops.csv", index=False)

    # fig3: alpha sweep
    base_rnn, base_t = get("baseline", 0.0, "dmt", "rnn", "acc"), get("baseline", 0.0, "smt", "teacher", "acc")
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4), sharey=True)
    for i, (stage, model, title, ref) in enumerate((("smt", "teacher", "Teacher (encoder+decoder) после SMT", base_t),
                                                     ("dmt", "rnn", "RNN после DMT (основная модель)", base_rnn))):
        for vv in ORDER[1:]:
            sub = piv[(piv.variant == vv) & (piv.stage == stage) & (piv.model == model)].sort_values("alpha")
            if sub.empty:
                continue
            ax[i].plot(sub.alpha, sub.acc, marker="o", ms=7, color=COLOR[vv], label=LABEL[vv])
            if vv in locked and locked[vv] in sub.alpha.values:
                y = float(sub[sub.alpha == locked[vv]].acc.iloc[0])
                ax[i].plot([locked[vv]], [y], marker="o", ms=13, mfc="none", mec=COLOR[vv], mew=2)
        ax[i].axhline(ref, color=COLOR["baseline"], ls="--", lw=1.5, label="baseline")
        ax[i].axhline(.25, color=MUTED, ls=":", lw=1)
        ax[i].set(title=title, xlabel="alpha (вес дополнительной CE)", xscale="log", ylim=(.2, 1))
        ax[i].set_xticks([.1, .3, 1.0], ["0,1", "0,3", "1,0"])
    ax[0].set_ylabel(f"accuracy на дальних pop (delay ≥ {long})")
    ax[1].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Подбор alpha на tuning-наборе ({NOTE}). Обведено — выбранный alpha.", x=.01, ha="left", fontsize=11)
    save(fig, "fig3_alpha_tuning")

    # fig4: validation learning curves at locked alpha (or best tuning alpha if not locked yet)
    hist = histories(run)
    hist.to_csv(OUT / "table_validation_histories.csv", index=False)
    chosen = dict(locked) if locked else {
        vv: float(piv[(piv.variant == vv) & (piv.stage == "dmt") & (piv.model == "rnn")].sort_values("ce").alpha.iloc[0])
        for vv in piv.variant.unique()}
    chosen["baseline"] = 0.0
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4), sharey=True)
    for vv in ORDER:
        if vv not in chosen:
            continue
        for i, (stage, col) in enumerate((("smt", "val_teacher_long_acc"), ("dmt", "val_rnn_long_acc"))):
            h = hist[(hist.variant == vv) & (hist.alpha == chosen[vv]) & (hist.stage == stage)]
            ax[i].plot(h.step, h[col], color=COLOR[vv], ls="--" if vv == "baseline" else "-",
                       label=f"{LABEL[vv]}, α={chosen[vv]:g}" if vv != "baseline" else "baseline")
    ax[0].set(title="SMT: teacher на validation", xlabel="шаг SMT", ylabel=f"accuracy, delay ≥ {long}", ylim=(.2, 1))
    ax[1].set(title="DMT: RNN на validation", xlabel="шаг DMT", ylim=(.2, 1))
    for a_ in ax:
        a_.axhline(.25, color=MUTED, ls=":", lw=1)
    ax[1].legend(loc="upper left", fontsize=9)
    fig.suptitle(f"Кривые обучения при выбранных alpha ({NOTE})", x=.01, ha="left", fontsize=11)
    save(fig, "fig4_learning_curves")

    # fig5: teacher vs RNN (H2 transfer) across all tuning fits
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    for vv in ORDER:
        for aa in sorted(piv[piv.variant == vv].alpha.unique()):
            x_, y_ = get(vv, aa, "smt", "teacher", "acc"), get(vv, aa, "dmt", "rnn", "acc")
            ax.scatter(x_, y_, s=40 + 120 * aa, color=COLOR[vv], edgecolor="white", lw=1.5, zorder=3,
                       label=LABEL[vv] if aa == max(piv[piv.variant == vv].alpha) else None)
    ax.plot([.2, 1], [.2, 1], color=MUTED, lw=1, ls="--")
    ax.text(.8, .83, "RNN = teacher", rotation=40, color=MUTED, fontsize=9)
    ax.set(xlim=(.3, .9), ylim=(.3, .9), xlabel=f"teacher после SMT: acc, delay ≥ {long}",
           ylabel=f"RNN после DMT: acc, delay ≥ {long}",
           title="H2: переносится ли память teacher в RNN\n(точка — одно обучение; размер — alpha)")
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig5_teacher_vs_rnn")

    # fig6: retention probe (H2) at chosen alphas
    pr.to_csv(OUT / "table_h2_probe.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    for vv in ORDER:
        if vv not in chosen:
            continue
        p = pr[(pr.variant == vv) & (pr.alpha == chosen[vv])].pivot(index="horizon", columns="model", values="acc")
        if p.empty:
            continue
        ax.plot(p.index, 100 * (p["rnn_from_teacher"] - p["teacher"]), marker="o", color=COLOR[vv],
                ls="--" if vv == "baseline" else "-", label=f"{LABEL[vv]}, α={chosen[vv]:g}")
    ax.axhline(0, color=MUTED, lw=1)
    ax.set(xscale="symlog", xlabel="L — число обновлений RNN, начиная с памяти teacher в момент s",
           ylabel="RNN − teacher, п.п.", title=f"H2: потери информации в updater (tuning-набор, delay ≥ {long})")
    ax.set_xticks([0, 1, 2, 4, 8, 16, 32], ["0", "1", "2", "4", "8", "16", "32"])
    ax.legend(fontsize=8, loc="lower left")
    save(fig, "fig6_h2_probe")

    # by-delay table at chosen alphas
    dt = ev[ev.apply(lambda r: chosen.get(r.variant) == r.alpha, axis=1)]
    bydelay = dt.groupby(["variant", "stage", "model", "delay_bin"], observed=True).correct.mean().unstack("delay_bin")
    bydelay = bydelay.reset_index()
    bydelay["variant"] = pd.Categorical(bydelay.variant, ORDER); bydelay = bydelay.sort_values(["variant", "stage", "model"])
    bydelay.to_csv(OUT / "table_accuracy_by_delay.csv", index=False)

    # baseline gate
    gate = json.loads((run / "baseline_gate.json").read_text())
    pd.DataFrame([gate]).to_csv(OUT / "table_baseline_gate.csv", index=False)

    readme += ["## 3. Baseline\n",
               f"Gate на tuning-наборе пройден: teacher {gate['teacher_long_accuracy']:.1%}, RNN после DMT "
               f"{gate['rnn_dmt_long_accuracy']:.1%} на дальних pop (пороги {gate['required_teacher']:.0%} / "
               f"{gate['required_rnn']:.0%}). Pop с delay ≤ 8 решены почти полностью; узкое место — дальние.\n",
               "## 4. Подбор alpha (одинаковая сетка и бюджет для всех вариантов)\n", "![](fig3_alpha_tuning.png)\n",
               md_table(tune_tab) + "\n",
               ("Зафиксированные alpha: `" + json.dumps(locked) + "`\n") if locked else
               "alpha ещё не зафиксированы (подбор не закончен); на графиках — лучшие по tuning CE.\n",
               "![](fig4_learning_curves.png)\n",
               "### Что видно на tuning (предварительно, один seed, не test)\n",
               "\n".join(f"* {LABEL[vv]}, α={chosen[vv]:g}: teacher {get(vv, chosen[vv], 'smt', 'teacher', 'acc'):.1%}, "
                         f"RNN после DMT {get(vv, chosen[vv], 'dmt', 'rnn', 'acc'):.1%} "
                         f"(CE {get(vv, chosen[vv], 'dmt', 'rnn', 'ce'):.3f})"
                         for vv in ORDER if vv in chosen) + "\n",
               "* Согласованный surprise-K (окно r=8) на tuning лучше random-K, но хуже uniform. Варианты, где больше "
               "выбранных целей реально требуют памяти m_t (контекстный score, oracle), дают больший прирост. "
               "Проверка этого — финальные три seeds на test по правилу из `preregistration.md`.\n"
               "* **Ограничение:** у всех вариантов выбран α=1,0 — край заранее заданной сетки; при большем α результат "
               "мог бы быть иным у всех вариантов. Сетка одинакова для всех и не менялась после просмотра.\n"
               "* uniform — сильный контроль: при α=1 он меняет соотношение decoder : dynamics : uniformity в SMT loss "
               "(Adam почти не чувствителен к общему масштабу loss).\n",
               "## 5. H2: перенос памяти teacher в RNN\n", "![](fig5_teacher_vs_rnn.png)\n", "![](fig6_h2_probe.png)\n",
               "Слева точки около диагонали — DMT переносит качество teacher в RNN. Справа: RNN стартует из памяти "
               "teacher в момент s и делает L шагов; ниже нуля — информация теряется в updater.\n",
               "## Таблица: accuracy по давности (выбранные alpha, tuning-набор)\n",
               md_table(bydelay.rename(columns=str)) + "\n",
               "## Файлы\n",
               "* `table_*.csv` — все таблицы; `table_validation_histories.csv` — validation по шагам для всех обучений.\n"
               "* `locked_alphas.json` — зафиксированные alpha. `preregistration.md` — правило вывода, записанное до подбора alpha.\n"
               "* Окно r=8, K=4, размеры моделей, шаги, seeds, пороги gate — наши настройки, научником не фиксировались.\n"]
    shutil.copy(run / "preregistration.md", OUT / "preregistration.md")
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("outputs written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
