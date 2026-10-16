# Диагностическая очередь 16.10.2026 — графики и таблицы (обновляется автоматически)

**Обновлено 16.10.2026 22:58.** Только validation, промежуточные данные: пока задачи не завершены, это не результаты.
Собирает `python make_v4_outputs.py` из `runs/`. Протокол — `proposals/preregistration_v4.md`, статус — `status.md`.
Каталог запуска Stack: `runs/stackv4_d459c6921638`.

## Stack v4, часть D (seed 107)
![](figD1_validation_curves.png)

![](figD2_rnn_vs_sequences.png)

Как читать:
* (а) teacher после SMT на дальних ответах (давность ≥ 8);
* (б) far-memory probe — хранит ли промежуточная память teacher m_t элемент. **Это не итоговая RNN**;
* (в) итоговая RNN в DMT;
* (г) BPTT-ориентиры, та же RNN с нуля: стиль линии = LR (точки 1e-4, сплошная 3e-4, штрихи 1e-3).

Прогресс:
* «шаги до плато» и «прирост за последнюю четверть» считаются по уже пройденным шагам;
* правило достаточности применяется только к завершённым обучениям.

| run | stage | steps_done | steps_max | best_acc | best_step | last_acc | gain_last_quarter_of_done | far_memory_acc_last | sec_per_step | train_hours | peak_allocated_mib |
|---|---|---|---|---|---|---|---|---|---|---|---|
| base_window | smt | 20000 | 20000 | 100.0% | 14500 | 99.9% | +0.0 п.п. | 47.1% | 0.34 | 1.90 | 7732 |
| base_window | dmt | 1000 | 12000 | 96.7% | 1000 | 96.7% | +0.4 п.п. | — | 0.65 | 0.18 | 6444 |
| bptt_1_lr0.0001 | bptt | 10500 | 20000 | 30.7% | 5500 | 29.7% | +0.0 п.п. | — | 0.62 | 1.82 | 666 |
| bptt_1_lr0.0003 | bptt | 12000 | 20000 | 33.4% | 10500 | 32.5% | +1.2 п.п. | — | 0.63 | 2.08 | 666 |
| bptt_full_lr0.0001 | bptt | 12500 | 20000 | 87.1% | 12500 | 87.1% | +4.2 п.п. | — | 0.62 | 2.16 | 666 |
| bptt_full_lr0.0003 | bptt | 12500 | 20000 | 97.5% | 12000 | 96.8% | +1.6 п.п. | — | 0.61 | 2.13 | 666 |
| bptt_full_lr0.001 | bptt | 12500 | 20000 | 89.9% | 12000 | 89.5% | +1.1 п.п. | — | 0.62 | 2.16 | 666 |
| ctx_far | smt | 20000 | 20000 | 100.0% | 18500 | 99.9% | +0.1 п.п. | 99.2% | 0.34 | 1.91 | 7732 |
| ctx_far | dmt | 1000 | 12000 | 97.3% | 1000 | 97.3% | +0.8 п.п. | — | 0.65 | 0.18 | 6444 |

**Время шага** измерено под нагрузкой: до 7 процессов на одной GPU. Чистая стоимость одного процесса — в
`benchmark_rtx5090_2026-10-16.json`.

Решение по бюджетам финала: ещё нет (после всех 8 обучений: `python smt_memup_v4.py --phase diag_report`).

## MQAR, проверка A (pilot01) и Zoology, проверка Б
Графики и таблицы: `outputs_mqar/figA1_mqar_baselines.png`, `table_mqarA_progress.csv`, `table_mqarA_summary.csv`,
`table_zooB.csv`.

_нет данных_

_нет данных_

## Файлы
* `table_diag_histories.csv` — все точки validation: run, stage, step, метрики, время, память.
* `table_diag_progress.csv` — сводка выше.
* Логи задач — `runs/jobs/*.log`, планировщик — `runs/gpu_jobs.log`.
* Истории и стоимость — `runs/stackv4_*/diag/*/`, `runs/mqar_baselines_*/*/`, `runs/zoology_repro/`. Checkpoints
  (`*.pt`) в git не входят.
