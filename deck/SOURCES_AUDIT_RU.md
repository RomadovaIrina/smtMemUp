# Аудит источников для презентации 26.10.2026

Каждый показанный результат — одна строка. Протоколы разведены по разделам, в одном сравнении они не смешиваются.

**Сокращения:**
* **ind. eval** — независимый набор: поток 901, 1024 последовательности; используется только для новых запусков;
* **val** — validation: поток 501 (Stack) или 511 (MQAR), 512 последовательностей; по нему выбирается чекпойнт;
* **best** — чекпойнт с минимальным validation CE на дальних ответах;
* **last** — последние веса.

Состояние новых запусков снято с RTX 3090 в 07:30 26.10 (`bash sync_results.sh`). Машинная версия таблицы со всеми
полями: [`tables/new_runs_status.csv`](tables/new_runs_status.csv).

## А. Новые запуски, RTX 3090: код ноутбука v1, seeds 201–203

Общее для раздела:
* LR: линейный разогрев 200 шагов, затем константа;
* batch 32, AdamW (β 0,9 / 0,95, wd 0,01), обрезка нормы градиента до 1,0;
* SMT LR 3e-4, DMT LR 1e-4.

| задача, конфигурация | метод (имя в коде) | seeds | оценка | бюджет | чекпойнт | поле метрики | источник | завершён |
|---|---|---|---|---|---|---|---|---|
| Stack T = 64, V = 5, Tf = 8, K_mem = 8 | SMT → DMT (`stack_smt_tf8`) | 201, 202, 203 | ind. eval | SMT 11 000 / DMT 5 000 | best | `eval_best_rnn_dmt_long_acc` (delay ≥ 9) | `results_3090/notebook_runs/stack_smt_tf8_s20*/summary.json` | да; у seed 203 `smt_history.csv` обрезан на шаге 6500 |
| то же | + дальние цели по ошибке без памяти (`stack_smt_tf8_ctx_far`), K_far = 2 | 201, 202 (203 ждёт) | ind. eval | 11 000 / 5 000 | best | то же | `…/stack_smt_tf8_ctx_far_s20*/summary.json` | 201 и 202 да (у 201 история SMT обрезана); 203 нет |
| то же | RNN, полный BPTT (`stack_bptt_full`), LR 3e-4 | 201–203 | ind. eval | BPTT 20 000 | best | `eval_best_rnn_bptt_long_acc` | `…/stack_bptt_full_s20*/` | да (первые попытки 202 и 203 упали по OOM, перезапущены) |
| то же | RNN, BPTT-1 (`stack_bptt_1`) | 201 (202 идёт, 203 в очереди) | ind. eval | 20 000 | best | `eval_best_rnn_bptt_long_acc` | `…/stack_bptt_1_s20*/` | 201 да; первые попытки — OOM, их **не считаем нулём** |
| то же | seed 201: `no_dyn`, `target_detached`, `random_far` | 201 | ind. eval | 11 000 / 5 000 | best | `eval_best_rnn_dmt_long_acc` | `…/stack_smt_tf8_{no_dyn,target_detached,random_far}_s201_*/` | да |
| то же | seed 201: `fully_detached`, `surprise_far` | 201 | — | 11 000 / 5 000 | — | — | — | нет: `fully_detached` идёт; `surprise_far` упал по OOM в 07:03 и перезапущен |
| MQAR T = 16, N = 2, V = 128, Tf = T | SMT → DMT (`mqar_T16N2_smt`) | 201–203 | ind. eval | SMT 6 000 / DMT 2 000 | best | teacher: `eval_best_teacher_smt_long_acc` (delay ≥ 4); до запросов: `eval_best_teacher_pre_after_pairs_smt_long_acc`; RNN до DMT: `eval_best_rnn_smt_long_acc`; после DMT: `eval_best_rnn_dmt_long_acc`; среди записанных значений — те же поля с `_cand_acc` | `…/mqar_T16N2_smt_s20*/` | да |
| то же | teacher 8 токенов, только L_dec (`mqar_T16N2_teacher_bottleneck`: c_dyn = c_unif = 0, без DMT) | 201–203 | ind. eval | 6 000 | best | `eval_best_teacher_smt_long_acc` | `…/mqar_T16N2_teacher_bottleneck_s20*/` | да |
| то же | диагностика: декодер читает все выходы энкодера (`mqar_T16N2_teacher_full`) | 201–203 | ind. eval | 6 000 | best | `eval_best_teacher_full_smt_long_acc` | `…/mqar_T16N2_teacher_full_s20*/` | да |
| то же | causal Transformer (`mqar_T16N2_transformer`), LR 1e-3 | 201–203 | ind. eval | ≤ 3 000, остановка при val ≥ 0,99 | best | `eval_best_lm_long_acc` | `…/mqar_T16N2_transformer_s20*/` | да |
| то же | RNN, полный BPTT (`mqar_T16N2_rnn_bptt_full`), LR 1e-3 | 201–203 | ind. eval | 6 000 | best | `eval_best_rnn_bptt_long_acc` | `…/mqar_T16N2_rnn_bptt_full_s20*/` | да, но **не сошёлся**: плато ~50% у 202 и 203, рост у 201 |
| то же | RNN, полный BPTT, длинный бюджет (`mqar_T16N2_rnn_bptt_full_long`) | 201, 202 (203 идёт) | ind. eval | до 60 000, стоп при val ≥ 0,99 | best | то же | `…/mqar_T16N2_rnn_bptt_full_long_s20*/` | 201: 60 000 шагов, 55,7%; 202: стоп на 51 000, 99,3% |
| MQAR T = 32, N = 4 | SMT → DMT (`mqar_T32N4_smt`) | 201–203 | ind. eval | 6 000 / 2 000 | best | как у T16 (delay ≥ 8) | `…/mqar_T32N4_smt_s20*/` (у 202 и 203 первые попытки — OOM) | да |
| то же | causal Transformer | 201–203 | ind. eval | ≤ 3 000 | best | `eval_best_lm_long_acc` | `…/mqar_T32N4_transformer_s20*/` | да |
| то же | RNN, полный BPTT | — | — | 60 000 | — | — | — | нет, в очереди |

## Б. Старый код, архив: Stack

| что | метод | seeds | оценка | бюджет | чекпойнт | поле | источник | завершён |
|---|---|---|---|---|---|---|---|---|
| v2: короткое окно и дальние цели, Tf = 8, K_far = 2 | без дальних / случайные / локальная неожиданность / ошибка без памяти / все дальние / диагностика «источник в памяти»; полное окно Tf = 64 | 7, 17, 27 | val | SMT 3 000 / DMT 1 500 (косинус) — **недообучено** | best | `smt_acc` (teacher), `far_memory_acc` (probe), `dmt_acc` (RNN) | `archive/outputs_v2_window/table_validation_by_seed.csv` | да |
| v4-диагностика | без дальних целей; + ошибка без памяти | 107 | val | SMT 20 000 / DMT 12 000 | best и кривые | `val_far_memory_acc`, `val_rnn_long_acc` | `archive/runs/stackv4_d459c6921638/diag/*/` | да |
| v4-диагностика | RNN, полный BPTT / BPTT-1, LR 3e-4 | 107 | val | 20 000 | best | `val_rnn_long_acc` | `archive/outputs_v4/README.md`, `table_diag_progress.csv` | да: 98,6% / 36,7%; BPTT-1 ещё рос |
| v4, другой бюджет | случайные / локальная неожиданность / полное окно | 107 | val | 10 000 / 6 000 | best | — | `archive/outputs_v4/README.md` | да; с 20k / 12k **не сравнивать** |
| абляции | все компоненты, без L_dyn, цель L_dyn отцеплена (старое имя `detached`), без L_unif, Tf = 1 / 4 / 16 / 64, K_mem = 4 / 16 | 107 | val | 11 000 / 5 000 | best | `teacher`, `rnn_before_dmt`, `far_probe`, `rnn_after_dmt` | `archive/outputs_ablation/table_ablation.csv` | да |

**Far-memory probe.** Что это:
* замороженный декодер teacher предсказывает дальние цели (j ≥ Tf) из m_t;
* берутся только те цели, чей push произошёл не позже t;
* оценка на 128 validation-последовательностях.

Код: `archive/smt_memup_window.py:144`. В новых запусках probe не считался.

## В. Старый код, архив: MQAR

| что | метод | seed | оценка | бюджет | поле | источник |
|---|---|---|---|---|---|---|
| короткий MQAR T = 16, N = 2 | SMT teacher, j = 0 | 107 | tuning | 6 000, **косинус до 0,1×** | `teacher_long_full_vocab` = 50,8% | `archive/runs/mqar03_pilot01arch_T16kv2_*/arch/T16kv2/teacher_summary.csv` |
| pilot01 T = 64, N = 8 | teacher 8 токенов / все выходы энкодера (только L_dec) | 107 | val, 128 посл. | 4 000 | 12,8% / 100% | только логи `archive/runs/jobs/bneck_{bottleneck,full}_T64.log` |
| pilot01 T = 64, N = 8 | causal Transformer / RNN, полный BPTT | 107 | val | Transformer до 1 000; RNN 20 000 | delay ≥ 24: Transformer 100%, RNN 1,7% (13,2% среди записанных) | `archive/outputs_mqar/table_mqarA_summary.csv` |
| pilot01, длинный бюджет | SMT teacher и RNN | 107 | tuning | 20 000 / 10 000, косинус | 12,3% среди записанных | `archive/status.md:52` |

## Что исправлено против презентации 20.10 (v3)

1. **«Короткий MQAR teacher не выучивает связывание»** (слайды 19 и 22 v3) — неверно для текущего протокола:
   * на T = 16, N = 2 teacher даёт 99,1% (j = 0) и 99,9% (память только после пар, до запросов) на трёх seeds;
   * обучение 6 000 шагов при постоянном LR. Кривая стоит на плато ~50% и прыгает к 100% между шагами 5 000 и 6 000;
   * старый запуск (50,8%) имел тот же бюджет, но LR по косинусу падал до 3e-5. Плюс другой seed и tuning-набор.

   Вывод «8 токенов памяти принципиально не хранят пары» снят.
2. **«Доступ ко всем выходам энкодера — 12,8% → 100%»** на T = 16 остаётся верным только как разница в **скорости**:
   * все выходы энкодера: скачок около шага 1 500;
   * 8 токенов: скачок около 5 000–5 750.

   На T = 64 это по-прежнему один seed и только лог.
3. **«RNN с полным BPTT на MQAR на уровне угадывания»**:
   * при 6 000 шагах он не сошёлся, поэтому 45,2% — не потолок;
   * при бюджете до 60 000 шагов: seed 202 — 99,3% (остановка на 51 000), seed 201 — 55,7%, seed 203 идёт;
   * то есть та же RNN с полным BPTT задачу решает, но не на каждом seed.
4. **Итоговая RNN на коротком MQAR** — 57,6% против 99% у teacher. Это наблюдение. Причина не установлена:
   * DMT всего 2 000 шагов;
   * полный BPTT той же RNN решает задачу только на части seeds и при бюджете в 10 раз больше.
5. **Названия моделей.** Base / Context / Oracle / detached / full заменены описательными названиями (таблица в
   приложении презентации).
