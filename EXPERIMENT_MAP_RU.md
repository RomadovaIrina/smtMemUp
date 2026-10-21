# Карта доказательств: утверждение → конфиг → команда → данные → метрика

## Как читать таблицы

**Столбец «воспроизведение»:**
* «из файлов» — число пересчитывается из CSV/JSON в git;
* «нужен GPU-перезапуск» — весов нет в git, а число получено при обучении;
* «только лог» — нет таблицы, есть только текстовый лог.

**Общие правила для всех строк, если не сказано иное:**
* метрика — точность ответа на длинных ответах (`delay ≥ long_delay`), checkpoint выбран по наименьшему validation CE
  на длинных ответах (bestCE);
* Stack: `long_delay = 8`, validation — поток 501, 512 последовательностей;
* MQAR: validation — поток 511, tuning — 512.

**Обозначения окон:**
* **Tf = T = 64** — полное окно, Stack версии 1;
* **Tf = 8** — короткое, целевая постановка.

## A. Stack

| утверждение / число | конфиг | команда (архив) | исходные данные | метрика | сиды | выборка | воспроизведение |
|---|---|---|---|---|---|---|---|
| v1: полное окно, перевзвешивание целей не улучшает RNN (H1 не поддержана) | `smt_memup_stack.Config`, **Tf = T = 64**, SMT 3000 / DMT 1500 | `python smt_memup_stack.py --phase final` → `--phase report --confirm-test` | `runs/stack02_4c69a2819ab8/test_*.csv`, `outputs_test/table_test_*.csv` | CE и точность RNN после DMT, парные разности, бутстреп | 7, 17, 27 | **test (503)** | сводки из файлов; события (`test_events.csv.gz`) и 15 из 18 весов не в git |
| v2: короткое окно Tf = 8 + дальние цели; surprise vs random меняет знак по сидам | `smt_memup_window.WindowConfig`, Tf = 8, 3000 / 1500 | `python smt_memup_window.py --phase final` | `runs/stackwin_26f6a039d62c/final/seed*/*/dmt_history.csv` | RNN после DMT, bestCE | 7, 17, 27 | validation | из файлов (`make_v2_outputs.py` требует `completed.pt`, которых в git нет) |
| v4 D: Tf = 8 base_window — RNN **97,9%**, ctx_far **98,6%**, far-probe 48% → 99% | `smt_memup_v4.V4Config`, Tf = 8, SMT 20k / DMT 12k | `python smt_memup_v4.py --phase diag --job base_window` (и `ctx_far`) | `runs/stackv4_d459c6921638/diag/{base_window,ctx_far}/{smt,dmt}_history.csv` | RNN после DMT, bestCE; probe — на bestCE SMT | **1 (107)** | validation | из файлов |
| v4 D: full BPTT **98,6%** (LR 3e-4) | та же RNN, BPTT 20k | `--phase diag --job bptt_full_lr0.0003` | `…/diag/bptt_full_lr0.0003/bptt_history.csv` | RNN, bestCE | 1 | validation (LR по tuning: `diag_lr_selection.csv`) | из файлов |
| v4 D: BPTT-1 не учится | то же, BPTT-1 | `--job bptt_1_lr0.0003` | `…/diag/bptt_1_lr0.0003/bptt_history.csv` | bestCE **33,8%** (в status.md 36,7% — последняя точка) | 1 | validation | из файлов |
| v4 доп.: random_far 93,2%, surprise_far 92,5% (10k / 6k) | Tf = 8, **другой бюджет** | `checks/smt_memup_v4_extra.py` | `runs/stackv4_490492c183d5/diag/*` | bestCE | 1 | validation | из файлов; с 20k / 12k напрямую не сравнивать |
| v4 доп.: полное окно Tf = 64 — 88,9% (10k / 6k) | Tf = 64 | там же | `runs/stackv4_5638fbb66ae7/diag/full_window` | bestCE | 1 | validation | из файлов; **та же траектория, что абляция `tf64`** |
| линейные пробы: BPTT хранит стек списком, SMT — вершину и глубину | v4 D | `python analysis_stack_probes.py` | `outputs_v4/probes/*.csv` | точность линейной пробы | 1 | validation | таблица в git; пересчёт требует весов (их нет) |
| абляции (10 строк: full 94,5%, no_dyn 66,0%, detached 91,7%, no_unif 93,9%, tf1 37,9%, tf4 68,1%, tf16 93,5%, tf64 89,0%, mem4 95,3%, mem16 91,7%) | V4Config, Tf = 8 (кроме tf*), SMT 11k / DMT 5k | `python checks/smt_memup_ablation.py --name <имя>` | `runs/stackv4_*/ablation/<имя>/{smt,dmt}_history.csv`; `outputs_ablation/table_ablation.csv` | teacher и RNN до DMT — bestCE SMT; RNN после — bestCE DMT; far-probe — max | **1** | validation | из файлов. **`detached` не отключает градиент L_dyn в энкодер** (`AUDIT_RU.md` §3); `full` = та же траектория, что v4 D base_window |
| `detached` пропускает градиент L_dyn в энкодер | — | `python checks/check_detached_gradients.py` (CPU, секунды) | вывод команды | норма градиента | — | — | **проверено 21.10** |

## B. MQAR

| утверждение / число | конфиг | команда (архив) | исходные данные | метрика | сиды | выборка | воспроизведение |
|---|---|---|---|---|---|---|---|
| pilot01 (T = 64, N = 8, V = 128): SMT teacher 11,7% среди записанных (шанс 12,5%), gate не пройден | `smt_memup_mqar.candidate("pilot01")`, Tf = T, SMT 3000 / DMT 1500 | `python smt_memup_mqar.py --phase baseline` | `runs/mqar03_pilot01_cb4a1ef021d2/baseline_gate.json` | teacher j = 0, среди записанных, перемешанный порядок запросов | 1 | tuning | из файла |
| то же, длинный бюджет 20k / 10k: 12,3% | pilot01 | `checks/smt_memup_mqar_long.py` | `runs/mqar03_pilot01long_ea929281632e/long/*/tuning_summary.csv` | teacher / RNN, среди записанных | 1 | tuning | из файлов |
| Transformer 99,8–100% на pilot01; RNN full BPTT ≈ шанс | `mqar_baselines.JOBS` (`tf_lr*`, `rnn_lr*`) | `python mqar_baselines.py --job <имя>` | `runs/mqar_baselines_249ffb5a0def/*/summary.csv` | точность по словарю и среди записанных | 1 | validation / tuning | из файлов |
| свип по N (2 / 4 / 8): Transformer 98,9–99,9%; SMT-teacher и RNN ≈ 1/N | pilot01, N переопределено | `mqar_baselines.py --job tf_kv2 …`; `checks/smt_memup_mqar_probe_arch.py --name kv2 …` | `outputs_mqar/table_pairs_sweep.csv` | среди записанных; **teacher — j = 0** | 1 | tuning | из файлов |
| **короткий MQAR** (T = 16, N = 2; T = 32, N = 2 / 4): SMT-teacher 50,8 / 49,3 / 24,8%; Transformer 98,9–99,8%; RNN BPTT ≈ 1/N | Tf = T, SMT 6000 / DMT 50; `long_delay = T/4` | `checks/smt_memup_mqar_probe_arch.py --name T16kv2` и т.д.; `mqar_baselines.py --job tf_T16kv2` и т.д. | `outputs_mqar/table_length_sweep.csv` ← `runs/mqar03_pilot01arch_T*/arch/*/teacher_summary.csv`, `runs/mqar_baselines_849a1231e25f/*/summary.csv` | **teacher: только j = 0 на позиции запроса**; RNN после DMT из 50 шагов не оценивалась | 1 | tuning | из файлов. Режим «память до запроса» **не измерялся** |
| узкое место: 8 токенов → 48% (T = 16) / 12,8% (T = 64); все выходы энкодера → 100% | teacher, только L_dec, Tf = T, 4000 шагов, LR 3e-4 без разогрева | `python checks/mqar_bottleneck_test.py --mode bottleneck|full --T 16 --pairs 2` | **только лог** `runs/jobs/bneck_{bottleneck,full}_T{16,64}.log` | j = 0, среди записанных; validation 128 последовательностей | 1 | validation | **только лог**; причина не изолирована: меняются и число позиций, и способ чтения |
| две реализации SMT-loss совпадают | — | `python checks/mqar_loss_crosscheck.py` | только лог `runs/jobs/xcheck_*.log` | loss в 16 точках | 1 | — | только лог; совпадение ~1e-3. **Новое:** `test_mqar_losses` сверяет loss и градиенты на CPU |
| Zoology attention (N = 256, D = 16, d = 64): 99,96% / 99,97% | `zoology_repro/configs.py` | `python zoology_repro/run_repro.py` | `runs/zoology_repro/*/*/summary.json`, `outputs_mqar/table_zooB.csv` | точность Zoology (по словарю, все запросы) | авторский | **test Zoology** + независимый набор 3000 | из файлов |

## C. Новые повторы (план зафиксирован 21.10, ещё не запускались)

* **Что запускается:** `reruns/plan.json`, 48 запусков, создаётся командой `python reruns/make_plan.py`.
* **Один запуск:** `python smt_minimal.py train --spec reruns/plan.json --index <i>`.
* **Очередь на GPU:** `python checks/gpu_jobs.py reruns/jobs.json`.
* **Протокол:**
  * сиды 201, 202, 203;
  * checkpoint выбирается по validation;
  * все числа считаются на **независимом** потоке 901 (1024 последовательности), который ни для какого выбора не
    используется;
  * сохраняются полные кривые, `identity.json` (конфиг, хэш кода, версии), время и шаг выбранного checkpoint.

| приоритет | что | семейства × сиды | что проверяет |
|---|---|---|---|
| 1 | MQAR T = 16, N = 2 и T = 32, N = 4 | SMT (6000 + DMT 2000): teacher j = 0, teacher «после пар», teacher «до запроса», RNN до и после DMT; teacher только L_dec с 8 токенами; teacher с полным доступом; Transformer; RNN full BPTT — 5 × 3 × 2 = 30 | вопрос Айдара: оба режима чтения отдельно, teacher отдельно от RNN, несколько сидов |
| 2 | Stack, Tf = 8 | SMT без дальних целей; SMT + ctx_far; та же RNN full BPTT; BPTT-1 — 4 × 3 = 12 | основные контроли на 3 сидах вместо одного |
| 3 | Stack, абляции | без L_dyn; авторский detached (`dyn_detach="both"`) — 2 × 3 = 6 | ключевые абляции, в том числе правильно определённый detached |

**Чего план не повторяет:**
* абляции Tf и памяти;
* random_far / surprise_far;
* полное окно;
* MQAR pilot01 (T = 64, N = 8);
* Zoology;
* линейные пробы.

Для них остаются архивные числа: один сид, validation.

Исторические числа выше не меняются; результаты повторов будут показаны рядом, как новые опыты.
