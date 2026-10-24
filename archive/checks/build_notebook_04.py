"""Builds 04_smt_memup_v3.ipynb (version 3: the version-1 comparison in a converged regime).
Run: python checks/build_notebook_04.py [--execute]   (execution runs only the CPU-safe cells; GPU cells are off by flags)
"""
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
md = lambda s: nb.cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: nb.cells.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# SMT × MemUP, Stack без шума — версия 3: сравнение в сходящемся режиме

**Что было до этого (14.10.2026):**
* **Версия 1** (`02_smt_memup_stack.ipynb`, отчёт `outputs_test/README.md`). К исходному SMT loss добавлено слагаемое,
  усиливающее K будущих целей; полное predictive-окно.
  * На test H1 в согласованной форме не поддержана: surprise-K по окну 8 токенов лучше random-K, но неустойчиво
    против uniform; все варианты перевзвешивания немного хуже baseline.
  * Отбор по неожиданности **при входах декодера** (ctx) дал +11 п.п. на всех seeds.
  * H2: RNN теряет часть памяти энкодера.
* **Версия 2** (`outputs_v2_window/README.md`, только validation). Короткое окно SMT лучше полного. Дальние цели
  усиливают удержание в памяти teacher, но surprise против random меняет знак по seeds.
* **Проверка сходимости** (`outputs_convergence/README.md`). При 30 000 шагах SMT teacher решает задачу на 100%, RNN — 93%
  после DMT 6000. Версии 1 и 2 шли в **недообученном** режиме (3000 шагов, косинус LR до 0,1×; DMT 1500).

**Версия 3** повторяет сравнение версии 1 в сходящемся режиме:
* SMT 10 000 шагов с постоянным LR после разогрева (как у авторов), DMT 6000;
* новые seeds 37 / 47 / 57 и новый test;
* ctx — заранее заявленная подтверждающая гипотеза.

Протокол и правило вывода — `proposals/preregistration_v3.md`; ниже его краткая версия.
""")

md(r"""
## 0. Протокол (кратко)
| | версия 1 | версия 3 |
|---|---|---|
| SMT | 3000 шагов, косинус до 0,1× | **10 000, разогрев 200 → постоянный 3e-4** |
| DMT | 1500 | **6000** (косинус) |
| α | подбор, выбрано 1,0 | **1,0 без подбора** |
| seeds | 7 / 17 / 27 | **37 / 47 / 57** |
| test | поток 503 | **поток 603** (новые 2048 последовательностей) |

Остальное как в версии 1: авторский SMT loss, все anchors, модель d = 64, Stack T = 64, детекторы, K = 4, r = 8,
выбор checkpoint по validation.

**Гипотезы:**
* **H1-ctx (подтверждающая):** surprise_ctx_k лучше baseline, random_k и uniform.
* **H1 (согласованная, повтор):** surprise_k лучше random_k и uniform.
* **H2:** перенос памяти в RNN.

**Правило** для каждого сравнения:
1. средняя ΔCE < 0;
2. ΔCE < 0 у всех трёх seeds;
3. 95% ДИ по последовательностям не содержит 0;
4. Δточность ≥ 0.

**Вторичная метрика:** скорость — шаги до 80 / 90 / 95% точности.
""")

code(r"""
from pathlib import Path
import sys, os, json, subprocess
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "smt_memup_v3.py").exists())
sys.path.insert(0, str(ROOT)); os.chdir(ROOT)
import pandas as pd, matplotlib.pyplot as plt, torch
import smt_memup_stack as S, smt_memup_v3 as V
pd.set_option("display.width", 200)
S.sources(ROOT)
CFG = V.V3Config()
RUN = V.saved_run_dir(CFG, ROOT)          # computed from the CUDA identity; creates nothing (safe on CPU)
print("repository:", ROOT)
print("torch", torch.__version__, "| CUDA visible:", torch.cuda.is_available())
print("version-3 run directory:", RUN.name, "| exists:", RUN.exists())
print("config:", {k: getattr(CFG, k) for k in ("smt_steps", "dmt_steps", "smt_lr_schedule", "dmt_lr_schedule", "alpha", "seeds", "K", "local_window")})
""")

md("## 1. Окружение и технические проверки (CPU)")
code(r"""
for script in ("checks/check_resume_identity.py", "checks/test_smt_memup_v3.py"):
    r = subprocess.run([sys.executable, script], capture_output=True, text=True, cwd=ROOT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    print(f"$ python {script}\n" + "\n".join(l for l in r.stdout.splitlines() if not l.startswith("  ")))
""")

md(r"""
## 2. Стоимость
Время шага (с оценками на validation каждые 250 шагов) измерено на этой карте (RTX 5090) в обучениях версии 2.
Они шли по 3 параллельно на свободном GPU — ровно так, как будет запущена версия 3.
""")
code(r"""
import glob
rows = []
for f in glob.glob("runs/stackwin_*/final/seed*/*/*_history.csv"):
    h = pd.read_csv(f).dropna(subset=["seconds_per_step"])
    rows.append(("smt" if "smt_" in f else "dmt", h.seconds_per_step.iloc[-1]))
t = pd.DataFrame(rows, columns=["stage", "s"]).groupby("stage").s.describe()[["min", "50%", "max"]]
print("seconds per step with 3 parallel processes (version-2 fits):"); print(t.round(3))
fit_h = (CFG.smt_steps * t.loc["smt"] + CFG.dmt_steps * t.loc["dmt"]) / 3600
print("\none version-3 fit (SMT 10k + DMT 6k), hours: median %.2f (range %.2f-%.2f)" % (fit_h["50%"], fit_h["min"], fit_h["max"]))
for n in (15, 18):
    print(f"{n} fits, 3 at a time: ~{n / 3 * fit_h['50%']:.1f} h of wall time (+ test report ~0.5 h)")
""")

md(r"""
## 3. Запуск обучения (GPU — только когда он свободен)
Из корня репозитория, одна команда: обучает 15 вариантов × seeds, или 18 с oracle; test не запускает.
```bash
checks/run_v3.sh                 # или: checks/run_v3.sh --with-oracle
```
Скрипт:
1. отказывается стартовать, если GPU занят чужими процессами (> 4 ГБ);
2. проверяет диск;
3. прогоняет проверки на CPU;
4. запускает планировщик `checks/gpu_jobs.py`: 3 процесса параллельно, одна задача на обучение, порядок по seeds.

Логи: `runs/gpu_jobs.log`, `runs/jobs/v3_*.log`. Ячейка ниже делает то же из ноутбука, но выключена флагом.
""")
code(r"""
RUN_V3_TRAINING = False          # True only when the GPU is free and the user has agreed
WITH_ORACLE = False
if RUN_V3_TRAINING:
    print(subprocess.run(["checks/run_v3.sh"] + (["--with-oracle"] if WITH_ORACLE else []), capture_output=True, text=True, cwd=ROOT).stdout)
else:
    print("training cell is off (RUN_V3_TRAINING = False)")
""")

md(r"""
## 4. Мониторинг (CPU, можно запускать в любой момент)
Ячейка показывает:
* прогресс и скорость обучения (шаги до 80 / 90 / 95% точности teacher; до 80 / 90% у RNN в DMT);
* кривые на validation.

Validation — набор выбора checkpoint, а не test.
""")
code(r"""
variants = list(V.VARIANTS_MAIN) + list(V.VARIANTS_OPTIONAL)
sp = V.speed_table(RUN, CFG.seeds, variants)
print("no version-3 fits yet" if sp.empty else sp.round(3).to_string())
if not sp.empty:
    COLOR = {"baseline": "#8a8984", "uniform": "#2a78d6", "random_k": "#eb6834", "surprise_k": "#1baf7a",
             "surprise_ctx_k": "#eda100", "oracle_prefix_k": "#e87ba4"}
    fig, ax = plt.subplots(2, len(CFG.seeds), figsize=(15, 7), sharey="row", squeeze=False)
    for j, seed in enumerate(CFG.seeds):
        for v in variants:
            d = RUN / "final" / f"seed{seed}" / v
            for i, (st, col) in enumerate((("smt", "val_teacher_long_acc"), ("dmt", "val_rnn_long_acc"))):
                f = d / f"{st}_history.csv"
                if f.exists():
                    h = pd.read_csv(f).dropna(subset=[col])
                    ax[i, j].plot(h.step, h[col], color=COLOR[v], label=v if (i, j) == (0, 0) else None)
        ax[0, j].set_title(f"seed {seed}"); ax[1, j].set_xlabel("step")
    ax[0, 0].set_ylabel("teacher (SMT), long-pop acc"); ax[1, 0].set_ylabel("RNN (DMT), long-pop acc")
    fig.legend(loc="lower center", ncol=6); plt.show()
""")

md(r"""
## 5. Test — ОДИН раз, только с разрешения пользователя
Запускать после того, как все обучения готовы: таблица мониторинга, столбец `complete`. По test ничего не выбирается.
```bash
python smt_memup_v3.py --phase report --confirm-test --variants baseline,uniform,random_k,surprise_k,surprise_ctx_k --gpu-memory-fraction 0.3
python make_test_outputs.py --preset v3 --run runs/<каталог версии 3> --out outputs_v3
```
Если обучался oracle, в `--variants` добавить `oracle_prefix_k`.
""")
code(r"""
RUN_V3_TEST = False              # True only with the user's explicit permission, after all fits are complete
if RUN_V3_TEST:
    vs = ",".join(v for v in variants if (RUN / "final" / "seed37" / v / "completed.pt").exists())
    r = subprocess.run([sys.executable, "smt_memup_v3.py", "--phase", "report", "--confirm-test", "--variants", vs,
                        "--gpu-memory-fraction", "0.3"], capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-3000:], r.stderr[-2000:])
    r = subprocess.run([sys.executable, "make_test_outputs.py", "--preset", "v3", "--run", str(RUN), "--out", "outputs_v3"],
                       capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-1000:], r.stderr[-1000:])
else:
    print("test cell is off (RUN_V3_TEST = False)")
""")

md("## 6. Результаты по test (появятся после раздела 5)")
code(r"""
out = ROOT / "outputs_v3"
if (out / "table_test_decision_rule.csv").exists():
    from IPython.display import Image, display
    display(pd.read_csv(out / "table_test_decision_rule.csv"))
    for f in ("figT1_primary_per_seed.png", "figT2_paired_deltas.png", "figT3_stages.png", "figT5_h2_retention.png"):
        display(Image(filename=str(out / f)))
    print((RUN / "speed.csv").exists() and pd.read_csv(RUN / "speed.csv").round(3).to_string() or "")
else:
    print("no version-3 test results yet")
""")

path = ROOT / "04_smt_memup_v3.ipynb"
nb.metadata["kernelspec"] = {"name": "smt-memup", "display_name": "Python (smt-memup)", "language": "python"}
nbf.write(nb, path)
print("written", path)
if "--execute" in sys.argv:
    from nbclient import NotebookClient
    nb = nbf.read(path, as_version=4)
    NotebookClient(nb, timeout=1800, kernel_name="smt-memup", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, path)
    print("executed (CPU cells)")
