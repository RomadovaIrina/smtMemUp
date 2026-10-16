# SMT × MemUP: Stack Operations и MQAR без шума

Тема — обучение рекуррентных моделей длинным зависимостям:
* SMT (Supervised Memory Training) с MemUP-подобным отбором будущих целей;
* задачи Stack Operations и MQAR (Zoology), без шума.

**Текущий статус — в [`status.md`](status.md)**: что выполнено, где подтверждения, что только подготовлено, что
заблокировано, какой следующий запуск. История до 16.10.2026 — [`history/README_2026-10-14.md`](history/README_2026-10-14.md)
(прежний README без изменений).

## Целевая постановка (одобрена научником)
SMT с коротким будущим окном Tf = 8 плюс MemUP-подобное предсказание дальних целей вне окна по памяти m_t и
промежуточным входам декодера. Сравнение:
* SMT-варианты: base_window, uniform_far, random_far, surprise_far (окно r = 8), ctx_far; oracle_far — диагностика;
* ориентиры — та же RNN с нуля: full BPTT и BPTT-1.

Протокол — [`proposals/preregistration_v4.md`](proposals/preregistration_v4.md), код — `smt_memup_v4.py`.

## Версии экспериментов
| версия | постановка | код | состояние |
|---|---|---|---|
| 1 | Stack, полное окно Tf = T, перевзвешивание целей | `smt_memup_stack.py` | 18/18 обучений, test выполнен: `outputs_test/` |
| C1/C2 | проверка сходимости (полное окно) | `smt_memup_stack.py --override` | выполнена: `outputs_convergence/` |
| 2 | Stack, Tf = 8 + дальние цели | `smt_memup_window.py` | 18/18 обучений, только validation: `outputs_v2_window/`; test не запускался |
| 3 | версия 1 в сходящемся режиме (полное окно) | `smt_memup_v3.py` | подготовлена, **не запускается**: не целевая постановка |
| **4** | **целевая: Tf = 8 + дальние цели + BPTT-ориентиры** | `smt_memup_v4.py` | код и CPU-проверки готовы; диагностика ждёт GPU |
| MQAR pilot01 | SMT на MQAR | `smt_memup_mqar.py` | gate не пройден: `outputs_mqar/protocol_mqar_v1.md` |
| MQAR A / Б | обычные бейзлайны: pilot01 и воспроизведение Zoology | `mqar_baselines.py`, `zoology_repro/` | готово к запуску: `outputs_mqar/zoology_reproduction.md` |

**Код, определяющий identity существующих запусков, не меняется:**
* `smt_memup_stack.py`, `smt_memup_window.py`, `smt_memup_v3.py`, `smt_memup_mqar.py`;
* хэш файла входит в имя каталога запуска.

Новые эксперименты — в новых файлах и каталогах `runs/stackv4_*`, `runs/mqar_baselines_*`, `runs/zoology_repro/`.

## Файлы
* Ноутбуки:
  * `02_smt_memup_stack.ipynb` — Stack, версия 1;
  * `03_smt_memup_mqar.ipynb` — MQAR, pilot01;
  * `04_smt_memup_v3.ipynb` — версия 3;
  * `01_…_archive.ipynb` — архив.
* Код экспериментов:
  * `smt_memup_stack.py` (v1), `smt_memup_window.py` (v2), `smt_memup_v3.py` (v3), `smt_memup_v4.py` (v4);
  * `smt_memup_mqar.py` (MQAR pilot01), `mqar_baselines.py` (MQAR A), `zoology_repro/` (MQAR Б);
  * `h2_retention.py` — анализ H2.
* Отчёты и протоколы:
  * `outputs_test/` — v1, test;
  * `outputs_convergence/` — C1/C2;
  * `outputs_v2_window/` — v2, validation;
  * `outputs/`, `outputs_v2/` — подбор α версии 1;
  * `outputs_mqar/` — MQAR;
  * `proposals/` — протоколы и предрегистрации;
  * `interim_results_2026-10-14/` — промежуточные таблицы.
* Проверки и запуск (`checks/`):
  * `test_*.py` — CPU-проверки;
  * `benchmark_train_step.py` — замер полного train step;
  * `check_v2_test_ready.py` — готовность test v2;
  * `check_resume_identity.py` — соответствие окружения запуску v1;
  * `gpu_jobs.py` + `make_jobs_v4.py` + `run_v4_diag.sh` — очередь GPU;
  * `fit_one.py`, `run_v3.sh` и др. — запускатели прежних версий.
* `runs/` — состояние запусков. В git только истории, identity, таблицы и детекторы; checkpoints — нет (см. `status.md`).

## Окружение (точные версии: от них зависит identity запусков)
```bash
uv venv --python 3.12.3 envs/smt-memup
uv pip install --python envs/smt-memup/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/smt-memup/bin/python numpy==2.1.0 pandas==2.3.3 matplotlib==3.10.8 einops==0.8.1 x-transformers==2.18.5 tqdm==4.67.1 pydantic==2.14.1 ipykernel jupyter nbformat nbclient nbconvert
uv pip install --python envs/smt-memup/bin/python wandb==0.22.2 pyyaml   # Zoology импортирует wandb
```
На арендованном сервере окружение лежит в `/workspace/envs/smt-memup`, репозиторий — в `/workspace/smtMemUp`.
Образ сервера (`/venv/main`) не трогаем.

## Авторские исходники (не входят в репозиторий)
```bash
git clone https://github.com/akarshkumar0101/smt.git upstream/smt && git -C upstream/smt checkout cf0d0f04be606ab9d97b1a449a6846450d656ad0
git clone https://github.com/griver/memup.git upstream/memup && git -C upstream/memup checkout d27657137f6e427e30fcd4ae885b20b071b2917d
git clone https://github.com/HazyResearch/zoology.git upstream/zoology && git -C upstream/zoology checkout 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb
```

## Команды
```bash
# CPU-проверки (без GPU)
CUDA_VISIBLE_DEVICES="" envs/smt-memup/bin/python checks/test_smt_memup_v4.py
CUDA_VISIBLE_DEVICES="" envs/smt-memup/bin/python checks/test_mqar_baselines.py
CUDA_VISIBLE_DEVICES="" envs/smt-memup/bin/python checks/test_smt_memup_window.py
# замер train step (GPU, ~минуты)
envs/smt-memup/bin/python checks/benchmark_train_step.py
# диагностическая очередь (GPU; только по согласованию): Stack v4 часть D, MQAR A, Zoology Б
checks/run_v4_diag.sh
# итоги диагностики Stack (CPU-безопасно после обучений)
envs/smt-memup/bin/python smt_memup_v4.py --phase diag_report
# готовность test версии 2 (CPU)
envs/smt-memup/bin/python checks/check_v2_test_ready.py --sha
```

## Правила работы
* Test каждой версии создаётся один раз и только с разрешения пользователя. По test ничего не выбирается.
* GPU-запуски, аренда и платные действия — только по согласованию.
* Push — только по просьбе пользователя. Перед коммитом проверять, что в него не попадают checkpoints
  (`*.pt` в `.gitignore`).
* Чужие процессы, файлы и окружения не трогаем.
