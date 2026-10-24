"""Builds outputs/: figures and tables for the advisor from saved runs (no training).

Evaluates every finished tuning fit on the TUNING split (the split used for alpha selection),
the H2 retention probe, and reads validation histories. Test is never touched.
Run: python make_outputs.py [--threads 4] [--gpu-memory-fraction 0.12]
"""
import argparse
import hashlib
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
OUT = ROOT / "outputs_v2"
ORDER = ["baseline", "uniform", "random_k", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"]
LABEL = {"baseline": "baseline: исходный SMT (α = 0)",
         "uniform": "uniform: весь decoder loss × (1 + α)",
         "random_k": "random-K: K случайных будущих ответов",
         "surprise_k": "surprise-K: неожиданные по окну 8 токенов",
         "surprise_ctx_k": "surprise-K по входам декодера (доп.)",
         "oracle_prefix_k": "oracle: ответы из памяти по разметке (доп.)"}
SHORT = {"baseline": "baseline", "uniform": "uniform", "random_k": "random-K",
         "surprise_k": "surprise-K (окно 8)", "surprise_ctx_k": "surprise-K (входы декодера), доп.",
         "oracle_prefix_k": "oracle (разметка), доп."}
# fixed categorical order (reference palette slots 1-5); baseline is the neutral reference
COLOR = {"baseline": "#8a8984", "uniform": "#2a78d6", "random_k": "#eb6834", "surprise_k": "#1baf7a",
         "surprise_ctx_k": "#eda100", "oracle_prefix_k": "#e87ba4"}
INK, MUTED = "#0b0b0b", "#52514e"
BINS = [0, 2, 4, 8, 16, 64]
BIN_LABELS = ["1–2", "3–4", "5–8", "9–16", "17+"]
NOTE = "подбор α, один seed (107); не финальный результат"


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


def fits(run: Path, need_model: bool = True):
    """Tuning fits. need_model=False: every fit with saved histories (the models themselves were not
    transferred to the new server; their evaluations are cached in analysis_tuning/)."""
    for d in sorted((run / "tuning").iterdir()):
        if not (d / ("completed.pt" if need_model else "dmt_history.csv")).exists():
            continue
        if d.name == "baseline":
            yield d, "baseline", 0.0
        else:
            m = re.fullmatch(r"(.+)_a([0-9.]+)", d.name)
            yield d, m.group(1), float(m.group(2))


class SavedRun:
    """The saved CUDA run without S.Experiment: Experiment derives the run directory from the CURRENT device,
    so a CPU-only analysis through it would create (and look into) a new, empty directory."""

    def __init__(self, cfg, device):
        S.sources(ROOT)
        ident = S.run_identity(cfg, S.torch.device("cuda"))          # the saved runs were trained on CUDA
        key = hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]
        self.dir = ROOT / "runs" / f"stack02_{key}"
        assert json.loads((self.dir / "identity.json").read_text()) == ident, "environment does not match the saved run"
        self.cfg, self.device = cfg, device
        self.tuning = S.sample_stack(cfg, cfg.tuning_examples, S.derive_seed(502, 0))


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
    for d, variant, alpha in fits(run, need_model=False):
        if (variant, alpha) in done:
            continue
        assert (d / "completed.pt").exists(), f"{d.name}: no cached evaluation and no model"
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
    for d, variant, alpha in fits(run, need_model=False):
        for stage in ("smt", "dmt"):
            h = pd.read_csv(d / f"{stage}_history.csv").dropna(subset=["val_teacher_long_acc"])
            h = h[["step", "val_teacher_long_acc", "val_rnn_long_acc", "val_teacher_long_ce", "val_rnn_long_ce"]].copy()
            h["variant"], h["alpha"], h["stage"] = variant, alpha, stage
            rows.append(h)
    return pd.concat(rows, ignore_index=True)


GLOSSARY = """## Как читать
* **Задача** — Stack Operations без шума, генератор авторов SMT (`StackDataset`): последовательность из 64 операций
  push(значение 1–4) и pop; после каждого pop модель должна назвать извлечённое значение. Глубина стека ≤ 8.
  **MQAR в этой версии ещё не обучался** (был только в архивном пилоте 01).
* **Ответ** — значение, которое нужно назвать на pop. **Давность** — сколько шагов элемент пролежал в стеке
  (шаг pop − шаг push). **Дальний ответ** — давность ≥ 8 (в Stack давность всегда нечётная, т.е. ≥ 9).
  Случайное угадывание — 25%.
* **Teacher** — энкодер-трансформер, сжимающий прошлое в память m_t, + декодер, предсказывающий будущие ответы.
  **RNN** — рекуррентная модель, обновляющая m_t шаг за шагом. **SMT** учит teacher и RNN (RNN — одношаговым
  переходам), **DMT** доучивает RNN на собственных траекториях. Итоговая модель — RNN после DMT.
* **Дополнительный loss** (MemUP-идея): в каждый момент t выбираем K=4 будущих ответа и даём им вес α поверх
  исходного SMT loss. Варианты отличаются правилом выбора:
  baseline — без добавки; uniform — добавка на все ответы; random-K — K случайных;
  **surprise-K** — K самых неожиданных для маленькой модели, видящей только 8 последних токенов (основной вариант);
  surprise-K по входам декодера — неожиданность при виде только тех входов, что видит декодер (доп.);
  oracle — K случайных среди ответов, которым по разметке генератора нужна память m_t (доп., подсмотр разметки).
* «Ответу нужна память m_t» — элемент положен в стек не позже момента t, поэтому декодер может взять его только
  из памяти. Иначе элемент появится во входах декодера после t, и память не нужна.
"""
STAGE_MODEL = {("smt", "teacher"): "teacher после SMT", ("smt", "rnn"): "RNN до DMT", ("dmt", "rnn"): "RNN после DMT"}


def chance_line(ax, horizontal=True):
    ax.axhline(.25, color=MUTED, ls=":", lw=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-memory-fraction", type=float, default=0.12)
    a = ap.parse_args()
    dev = S.setup_runtime(a.threads, a.gpu_memory_fraction)
    style()
    OUT.mkdir(exist_ok=True)
    cfg = S.Config()
    ex = SavedRun(cfg, dev)
    run = ex.dir
    long = cfg.long_delay
    readme = [f"# Stack Operations: данные, отбор ответов, baseline и подбор α (версия 2 графиков)\n",
              "Это **не финальное сравнение**: все обучения ниже — на одном seed (107) при подборе α. Финал — три "
              "seeds (7, 17, 27) на отдельном test по правилу из `preregistration.md`. Checkpoint выбирался по "
              "validation, α — по отдельному tuning-набору (512 последовательностей). Источник: `" + run.name + "`.\n",
              GLOSSARY]

    # ---------------- fig1: data ----------------
    tun = ex.tuning
    audit = pd.read_csv(run / "data_audit.csv")
    audit_ru = audit.rename(columns={"split": "набор", "sequences": "последовательностей", "pops": "ответов (pop)",
                                     "long_pops": "дальних ответов", "long_fraction": "доля дальних",
                                     "ge16_fraction": "доля с давностью ≥ 16", "delay_median": "медиана давности",
                                     "delay_p95": "95-й перцентиль давности", "unique_sequences": "уникальных"})
    audit_ru.to_csv(OUT / "table_data_audit.csv", index=False)
    w = S.windows(S.slice_batch(tun, 0, 128))
    v = w["valid"]
    off = (w["target_t"] - w["anchor"][:, None])[v].numpy()
    pref = w["prefix"][v].numpy()
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 3.9))
    dl = tun["delay"][tun["target_mask"]].numpy()
    ax[0].hist(dl, bins=np.arange(0, 65) + .5, color=COLOR["uniform"], rwidth=.85)
    ax[0].axvline(long, color=MUTED, ls="--", lw=1)
    ax[0].text(long + 1, 3000, f"дальние (давность ≥ {long}):\n{(dl >= long).mean():.0%} всех ответов", color=MUTED, fontsize=9)
    ax[0].set(title="Сколько шагов элемент лежит в стеке до pop", xlabel="давность = шаг pop − шаг push",
              ylabel="число ответов (лог. шкала)", yscale="log", xlim=(0, 64))
    share = pd.Series(pref).groupby(off).mean()
    ax[1].plot(share.index, share.values, color=COLOR["uniform"])
    ax[1].set(title="Каким будущим ответам нужна память m_t",
              xlabel="на сколько шагов после момента t стоит ответ", ylabel="доля ответов, взятых из памяти",
              xlim=(0, 63), ylim=(0, 1.02))
    ax[1].annotate("ответ в сам момент t\nвсегда берётся из памяти", xy=(0, 1), xytext=(8, .8), fontsize=9,
                   color=MUTED, arrowprops=dict(arrowstyle="->", color=MUTED))
    fig.suptitle("Данные: Stack Operations (генератор авторов SMT), 64 операции, глубина ≤ 8, значения 1–4",
                 x=.01, ha="left", fontsize=11)
    save(fig, "fig1_data")
    readme += ["## 1. Данные\n", "![](fig1_data.png)\n",
               "Слева: большинство элементов извлекаются почти сразу; дальних ответов около 15%. Справа: чем дальше "
               "в будущем ответ, тем чаще его элемент положен уже после t — тогда декодер видит его во входах и "
               "память не нужна. Поэтому «неожиданный» ответ не обязательно учит память.\n",
               md_table(audit_ru) + "\n"]

    # ---------------- fig2: selection audit ----------------
    sel = pd.read_csv(run / "selection_audit.csv")
    q = json.loads((run / "detector_quality.json").read_text())
    pd.DataFrame(q).T.to_csv(OUT / "table_detector_quality.csv")
    s = sel[sel.anchors == "all anchors"].set_index("rule")
    rules = ["random_k (exact expectation)", "surprise_k", "surprise_ctx_k", "oracle_prefix_k"]
    names = ["random-K\n(случайные)", "surprise-K\n(окно 8 токенов)", "surprise-K\n(входы декодера), доп.",
             "oracle\n(разметка), доп."]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    x = np.arange(len(rules))
    ax.bar(x - .2, s.loc[rules, "memory_dependent"], .38, color=COLOR["uniform"],
           label="ответу нужна память m_t (элемент положен до t)")
    ax.bar(x + .2, s.loc[rules, "long_memory_dependent"], .38, color=COLOR["random_k"],
           label=f"… и это дальний ответ (давность ≥ {long})")
    for i, r in enumerate(rules):
        for dx, col in ((-.2, "memory_dependent"), (.2, "long_memory_dependent")):
            ax.text(i + dx, s.loc[r, col] + .02, f"{s.loc[r, col]:.0%}", ha="center", fontsize=9, color=INK)
    ceiling = s.loc["oracle_prefix_k", "pairs"] / s.loc["random_k", "pairs"]
    ax.axhline(ceiling, color=MUTED, ls="--", lw=1)
    ax.text(.5, ceiling + .03, f"максимум для любого правила с K=4: {ceiling:.0%}\n(часто таких ответов меньше 4)",
            color=MUTED, fontsize=8.5, ha="center")
    ax.set_xticks(x, names); ax.set_ylim(0, 1.15)
    ax.set_ylabel("доля среди выбранных ответов")
    ax.set_title("Каким ответам правило отбора даёт дополнительный вес (до обучения, tuning-набор)")
    ax.legend(loc="upper left", fontsize=9)
    save(fig, "fig2_selection")
    sel_ru = sel[sel.anchors == "all anchors"][["rule", "pairs", "memory_dependent", "long_memory_dependent",
                                                "future_source", "mean_delay"]].rename(columns={
        "rule": "правило", "pairs": "выбрано пар", "memory_dependent": "нужна память m_t",
        "long_memory_dependent": "нужна память и дальний", "future_source": "элемент положен после t",
        "mean_delay": "средняя давность"})
    sel_ru.to_csv(OUT / "table_selection_audit.csv", index=False)
    readme += ["## 2. Каким ответам дополнительный loss даёт вес (до обучения)\n", "![](fig2_selection.png)\n",
               f"Модель с окном 8 токенов идеально находит ответы, чей элемент старше окна (AUC "
               f"{q['window']['auc_outside_window']:.3f}), но {s.loc['surprise_k', 'future_source']:.0%} выбранных ею "
               "ответов не требуют памяти m_t: их элемент положен после t. Правило по входам декодера почти достигает "
               "максимума доли ответов из памяти.\n", md_table(sel_ru) + "\n"]

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
            rows.append({"вариант": SHORT[vv], "α": aa, "выбран": "✓" if locked.get(vv) == aa else "",
                         "teacher после SMT: точность": get(vv, aa, "smt", "teacher", "acc"),
                         "teacher после SMT: CE": get(vv, aa, "smt", "teacher", "ce"),
                         "RNN до DMT: точность": get(vv, aa, "smt", "rnn", "acc"),
                         "RNN после DMT: точность": get(vv, aa, "dmt", "rnn", "acc"),
                         "RNN после DMT: CE": get(vv, aa, "dmt", "rnn", "ce"),
                         "RNN после DMT через голову teacher: точность": get(vv, aa, "dmt", "rnn", "shared_acc"),
                         "RNN после DMT: точность на всех ответах": get(vv, aa, "dmt", "rnn", "acc_all")})
    tune_tab = pd.DataFrame(rows)
    tune_tab.to_csv(OUT / "table_tuning_long_answers.csv", index=False)

    # ---------------- fig3: alpha sweep ----------------
    base_rnn, base_t = get("baseline", 0.0, "dmt", "rnn", "acc"), get("baseline", 0.0, "smt", "teacher", "acc")
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for i, (stage, model, title, ref) in enumerate((("smt", "teacher", "Teacher (энкодер + декодер) после SMT", base_t),
                                                     ("dmt", "rnn", "RNN после DMT — итоговая модель", base_rnn))):
        for vv in ORDER[1:]:
            sub = piv[(piv.variant == vv) & (piv.stage == stage) & (piv.model == model)].sort_values("alpha")
            if sub.empty:
                continue
            ax[i].plot(sub.alpha, sub.acc, marker="o", ms=7, color=COLOR[vv], label=LABEL[vv])
            if vv in locked and locked[vv] in sub.alpha.values:
                y = float(sub[sub.alpha == locked[vv]].acc.iloc[0])
                ax[i].plot([locked[vv]], [y], marker="o", ms=14, mfc="none", mec=COLOR[vv], mew=2)
        ax[i].axhline(ref, color=COLOR["baseline"], ls="--", lw=1.5, label=LABEL["baseline"])
        chance_line(ax[i])
        ax[i].text(.105, .265, "случайное угадывание, 25%", color=MUTED, fontsize=8.5)
        ax[i].set(title=title, xlabel="α — вес дополнительного loss (лог. шкала)", xscale="log", ylim=(0, 1))
        ax[i].set_xticks([.1, .3, 1.0], ["0,1", "0,3", "1,0"])
    ax[0].set_ylabel(f"точность на дальних ответах (давность ≥ {long})")
    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, -.12))
    fig.suptitle(f"Подбор α на tuning-наборе ({NOTE}). Кружком обведён выбранный α.", x=.01, ha="left", fontsize=11)
    save(fig, "fig3_alpha_tuning")

    # ---------------- fig4: learning curves ----------------
    hist = histories(run)
    hist.to_csv(OUT / "table_validation_histories.csv", index=False)
    chosen = dict(locked) if locked else {
        vv: float(piv[(piv.variant == vv) & (piv.stage == "dmt") & (piv.model == "rnn")].sort_values("ce").alpha.iloc[0])
        for vv in piv.variant.unique()}
    chosen["baseline"] = 0.0
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for vv in ORDER:
        if vv not in chosen:
            continue
        for i, (stage, col) in enumerate((("smt", "val_teacher_long_acc"), ("dmt", "val_rnn_long_acc"))):
            h = hist[(hist.variant == vv) & (hist.alpha == chosen[vv]) & (hist.stage == stage)]
            ax[i].plot(h.step, h[col], color=COLOR[vv], ls="--" if vv == "baseline" else "-",
                       label=LABEL[vv] if vv == "baseline" else f"{LABEL[vv]}, α = {chosen[vv]:g}")
    ax[0].set(title="Этап SMT: teacher", xlabel="шаг обучения SMT", ylabel=f"точность на дальних ответах (давность ≥ {long})",
              ylim=(0, 1), xlim=(0, cfg.smt_steps))
    ax[1].set(title="Этап DMT: RNN (стартует с весов после SMT)", xlabel="шаг обучения DMT", ylim=(0, 1), xlim=(0, cfg.dmt_steps))
    for a_ in ax:
        chance_line(a_)
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9, bbox_to_anchor=(.5, -.12))
    fig.suptitle(f"Ход обучения при выбранном α: точность на validation ({NOTE}); пунктир снизу — случайное угадывание",
                 x=.01, ha="left", fontsize=11)
    save(fig, "fig4_learning_curves")

    # ---------------- fig5: teacher vs RNN ----------------
    fig, ax = plt.subplots(figsize=(7, 6.4))
    for vv in ORDER:
        alphas = sorted(piv[piv.variant == vv].alpha.unique())
        for aa in alphas:
            x_, y_ = get(vv, aa, "smt", "teacher", "acc"), get(vv, aa, "dmt", "rnn", "acc")
            ax.scatter(x_, y_, s=40 + 120 * aa, color=COLOR[vv], edgecolor="white", lw=1.5, zorder=3,
                       label=SHORT[vv] if aa == alphas[-1] else None)
    ax.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--")
    ax.text(.12, .16, "RNN = teacher", rotation=42, color=MUTED, fontsize=9)
    ax.text(.62, .38, "ниже диагонали: RNN хуже teacher,\nчасть памяти не перенеслась", color=MUTED, fontsize=8.5)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel=f"точность teacher после SMT (дальние ответы)",
           ylabel=f"точность RNN после DMT (дальние ответы)",
           title="Переносит ли DMT память teacher в RNN\n(точка — одно обучение; чем больше точка, тем больше α)")
    ax.legend(fontsize=8.5, loc="upper left")
    save(fig, "fig5_teacher_vs_rnn")

    # ---------------- fig6: retention probe ----------------
    pr.to_csv(OUT / "table_h2_probe.csv", index=False)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for vv in ORDER:
        if vv not in chosen:
            continue
        p = pr[(pr.variant == vv) & (pr.alpha == chosen[vv])].pivot(index="horizon", columns="model", values="acc")
        if p.empty:
            continue
        ax.plot(p.index, 100 * (p["rnn_from_teacher"] - p["teacher"]), marker="o", color=COLOR[vv],
                ls="--" if vv == "baseline" else "-",
                label=LABEL[vv] if vv == "baseline" else f"{LABEL[vv]}, α = {chosen[vv]:g}")
    ax.axhline(0, color=INK, lw=1)
    ax.text(5, .5, "0 = RNN ничего не теряет", color=MUTED, fontsize=8.5)
    ax.set(xlim=(0, 33), xlabel="L — сколько обновлений RNN сделала сама, стартовав с памяти teacher",
           ylabel="точность RNN − точность teacher, п.п.",
           title="Теряет ли RNN информацию при собственных обновлениях (tuning-набор, дальние ответы)")
    ax.set_xticks([0, 1, 2, 4, 8, 16, 32])
    ax.legend(fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.17), ncol=2)
    save(fig, "fig6_h2_probe")

    # ---------------- table: accuracy by age ----------------
    dt = ev[ev.apply(lambda r: chosen.get(r.variant) == r.alpha, axis=1)]
    dt = dt[~((dt.stage == "dmt") & (dt.model == "teacher"))]          # teacher is frozen in DMT: same as after SMT
    bydelay = dt.groupby(["variant", "stage", "model", "delay_bin"], observed=True).correct.mean().unstack("delay_bin")
    bydelay = bydelay.reset_index()
    bydelay["модель"] = [STAGE_MODEL[(st, m)] for st, m in zip(bydelay.stage, bydelay.model)]
    bydelay["order"] = bydelay.variant.map({v: i for i, v in enumerate(ORDER)})
    bydelay = bydelay.sort_values(["order", "stage", "model"])
    bydelay["вариант"] = bydelay.variant.map(SHORT) + bydelay.variant.map(lambda v: f", α = {chosen[v]:g}")
    bydelay = bydelay[["вариант", "модель", *BIN_LABELS]].rename(columns={b: f"давность {b}" for b in BIN_LABELS})
    bydelay.to_csv(OUT / "table_accuracy_by_age.csv", index=False)

    gate = json.loads((run / "baseline_gate.json").read_text())
    pd.DataFrame([gate]).to_csv(OUT / "table_baseline_gate.csv", index=False)

    obs = "\n".join(f"* {SHORT[vv]}, α = {chosen[vv]:g}: teacher {get(vv, chosen[vv], 'smt', 'teacher', 'acc'):.1%}, "
                    f"RNN после DMT {get(vv, chosen[vv], 'dmt', 'rnn', 'acc'):.1%} (CE {get(vv, chosen[vv], 'dmt', 'rnn', 'ce'):.3f})"
                    for vv in ORDER if vv in chosen)
    readme += ["## 3. Baseline\n",
               f"Проверка работоспособности пройдена: на дальних ответах teacher {gate['teacher_long_accuracy']:.1%}, "
               f"RNN после DMT {gate['rnn_dmt_long_accuracy']:.1%} (пороги {gate['required_teacher']:.0%} / "
               f"{gate['required_rnn']:.0%}, случайное — 25%). Ответы с давностью < 8 решены почти полностью.\n",
               "## 4. Подбор α (одинаковые сетка и бюджет у всех вариантов)\n", "![](fig3_alpha_tuning.png)\n",
               md_table(tune_tab) + "\n", "![](fig4_learning_curves.png)\n",
               "### Что видно (предварительно: один seed, не test)\n", obs + "\n",
               "* Основной surprise-K (окно 8 токенов) лучше random-K, но хуже uniform. Правила, у которых больше "
               "выбранных ответов действительно берутся из памяти m_t (по входам декодера, oracle), дают больший "
               "прирост. Проверка — три финальных seeds на test.\n"
               "* **Ограничение:** у всех вариантов выбран α = 1,0 — край заранее заданной сетки.\n"
               "* uniform — сильный контроль: при α = 1 он меняет соотношение слагаемых SMT loss "
               "(decoder : dynamics : uniformity); Adam почти не чувствителен к общему масштабу loss.\n",
               "## 5. Переносится ли память teacher в RNN (вопрос научника про последовательные обновления)\n",
               "![](fig5_teacher_vs_rnn.png)\n", "![](fig6_h2_probe.png)\n",
               "Слева: при слабой памяти teacher точки на диагонали или выше — RNN ничего не теряет (выше — за счёт "
               "дообученной головы RNN). При сильной памяти (oracle, входы декодера) точки ниже диагонали. Справа то же "
               "напрямую: RNN, стартовав с памяти teacher, за 8–16 своих шагов теряет до 15–17 п.п. у этих вариантов.\n",
               "## Точность по давности (выбранные α, tuning-набор)\n", md_table(bydelay) + "\n",
               "## Файлы\n",
               "* `table_*.csv` — все таблицы; `table_validation_histories.csv` — validation по шагам для всех обучений.\n"
               "* `locked_alphas.json` — выбранные α. `preregistration.md` — правило вывода, записанное до подбора α.\n"
               "* Окно 8 токенов, K = 4, размеры моделей, число шагов, seeds и пороги — наши настройки, научник их не фиксировал.\n"]
    shutil.copy(run / "preregistration.md", OUT / "preregistration.md")
    (OUT / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print("outputs written:", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
