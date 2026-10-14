# SMT × MemUP: Stack Operations и MQAR без шума

* `02_smt_memup_stack.ipynb` — Stack Operations: аудит, протокол, фазы эксперимента, предрегистрация (раздел 8.0).
* `smt_memup_stack.py` — весь код Stack-эксперимента; импорт ничего не обучает. Хэш файла входит в идентификатор запуска.
* `03_smt_memup_mqar.ipynb` + `smt_memup_mqar.py` — MQAR (Zoology): данные, аудит отбора, выбор окна, оценка стоимости; обучение ещё не запускалось. Общие части импортируются из `smt_memup_stack.py` без изменений.
* `checks/` — технические проверки (`test_smt_memup_stack.py` — 16, `test_smt_memup_mqar.py` — 12), запуск на общем GPU (`wait_and_run.sh`, `run_final_parallel.sh`), проверка переноса (`check_resume_identity.py`), сборка ноутбука 03 (`build_notebook_03.py`).
* `make_outputs.py` — графики и таблицы Stack по сохранённым запускам (без test).
* `outputs/`, `outputs_v2/` — графики и таблицы Stack (v2 — читаемые подписи); `outputs_mqar/` — подготовка MQAR.
* `runs/stack02_4c69a2819ab8/` — **состояние Stack-эксперимента**, нужное для продолжения (см. ниже). Остальное в `runs/` в git не входит.
* `01_smt_memup_preparation_archive.ipynb` — архив версии 01 с её сохранёнными результатами (не результаты кода 02).

## Окружение (точные версии: от них зависит идентификатор запуска)

```bash
uv venv --python 3.12.3 envs/smt-memup
uv pip install --python envs/smt-memup/bin/python torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python envs/smt-memup/bin/python numpy==2.1.0 pandas==2.3.3 matplotlib==3.10.8 einops==0.8.1 x-transformers==2.18.5 tqdm==4.67.1 pydantic==2.14.1 ipykernel jupyter nbformat nbclient nbconvert
uv pip install --python envs/smt-memup/bin/python wandb==0.22.2 pyyaml   # Zoology импортирует wandb (на identity Stack не влияет)
envs/smt-memup/bin/python -m ipykernel install --user --name smt-memup --display-name "Python (smt-memup)"
```

## Авторские исходники (не входят в репозиторий)

```bash
git clone https://github.com/akarshkumar0101/smt.git upstream/smt && git -C upstream/smt checkout cf0d0f04be606ab9d97b1a449a6846450d656ad0
git clone https://github.com/griver/memup.git upstream/memup && git -C upstream/memup checkout d27657137f6e427e30fcd4ae885b20b071b2917d
git clone https://github.com/HazyResearch/zoology.git upstream/zoology && git -C upstream/zoology checkout 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb
```

## Фазы

```bash
python smt_memup_stack.py --phase audit|baseline|tune|final --variants baseline,uniform,random_k,surprise_k,surprise_ctx_k,oracle_prefix_k
python smt_memup_stack.py --phase report --confirm-test --variants baseline,uniform,random_k,surprise_k,surprise_ctx_k,oracle_prefix_k
python smt_memup_mqar.py --phase benchmark|audit|baseline|tune|final|report --config <имя из CANDIDATES> [--override '{...}']
```

---

## Продолжение на другом сервере (инструкция для агента)

### 0. Состояние на 14.10.2026, ~20:50 (сервер с RTX 5090, репозиторий в `/mem`) — читать первым
**Сделано:**
* **Stack, версия 1 (полное окно, перевзвешивание целей).** Финал 18/18, test выполнен один раз.
  Отчёт — `outputs_test/README.md`, в нём есть текст для научника. Кратко:
  * H1 в согласованной форме при этом бюджете не поддержана: surprise-K лучше random-K на всех seeds, но против uniform
    неустойчиво, и все варианты перевзвешивания немного хуже baseline;
  * контекстный surprise (ctx) даёт +11 п.п. на всех seeds;
  * H2: RNN теряет часть информации энкодера, сильнее при сильной памяти.
* **Проверка сходимости** (`outputs_convergence/README.md`): при 30 000 шагах SMT teacher решает задачу на 100%,
  итоговая RNN — 93% после DMT 6000. Версии 1 и 2 шли в недообученном режиме; DMT 1500 мало.
* **Stack, версия 2 (короткое окно Tf = 8 + дальние цели).** 18/18 обучений, только validation: `outputs_v2_window/README.md`.
  Короткое окно само по себе лучше полного. Дальняя supervision усиливает удержание в промежуточной памяти teacher
  (oracle > ctx > surprise ≈ random), но surprise_far против random_far меняет знак по seeds.
* **MQAR** (`outputs_mqar/protocol_mqar_v1.md`): baseline `pilot01` не прошёл gate по связыванию ключ–значение
  (teacher 11,7% среди записанных значений при уровне наугад 12,5%). По решению пользователя MQAR отложен до проверенной
  на Stack интеграции.

**Завтра: версия 3 — сравнение версии 1 в сходящемся режиме** (всё подготовлено, на GPU ещё не запускалось):
* что это: SMT 10 000 шагов (постоянный LR после разогрева, как у авторов), DMT 6000, новые seeds 37 / 47 / 57,
  новый test. Surprise по входам декодера (ctx) — подтверждающая гипотеза. Протокол — `proposals/preregistration_v3.md`;
  ноутбук — `04_smt_memup_v3.ipynb`; код — `smt_memup_v3.py`; проверки — `checks/test_smt_memup_v3.py` (7/7);
* запуск при свободном GPU: `checks/run_v3.sh` (15 обучений, ~6,3 ч) или `checks/run_v3.sh --with-oracle` (18, ~7,5 ч);
* прогресс (безопасно на CPU): `python smt_memup_v3.py --phase speed` или раздел 4 ноутбука;
* test — один раз, только с разрешения: раздел 5 ноутбука или команды в предрегистрации;
* **не править `smt_memup_v3.py` во время обучений**: его хэш входит в identity, и путь каталога запуска изменится.

**Не сделано / ждёт решения:**
* **Test версии 2 не запускался — только с разрешения пользователя.** Команды:
  `python smt_memup_window.py --phase report --confirm-test`, затем `python make_v2_outputs.py --test`.
* **Следующий шаг (предложение):** отдельная предрегистрация сравнения отбора (random / surprise / контекстный) в сходящемся
  режиме. Метрики: шаги до порога, итоговая RNN после длинного DMT, H2.

**Инфраструктура:**
* `checks/gpu_jobs.py` — планировщик задач с зависимостями и порогом памяти на задачу. `checks/fit_one.py` — одно
  обучение (version 1 / 2) с сохранённой identity. «Вежливые» запускатели (`run_final_polite.py`, `polite_queue.py`)
  уступают GPU соседям.
* **GPU и диск делятся с `/dub`.** Не запускать без разрешения; при нехватке диска не трогать чужое.
* **Тип устройства входит в identity**, поэтому `--phase report` запускать только с видимым GPU. `make_outputs.py`
  работает и на CPU.
* **Git по SSH.** Соседний проект добавил `~/.ssh/config` со своим ключом для github.com. Для этого репозитория в
  `.git/config` задан `core.sshCommand` с нашим ключом `~/.ssh/id_ed25519`.

**Что не в git:**
* checkpoints всех обучений и сырые события test (`runs/stack02_4c69a2819ab8/test_events.csv.gz`, 141 МБ) лежат только
  на сервере;
* шесть `.pt` seed 7 в git — устаревшие версии от 13.10.

### 1. Где остановились (13.10.2026, ~19:00; актуальное — раздел 0)

**Stack Operations** (ноутбук 02, запуск `runs/stack02_4c69a2819ab8`):
* baseline (gate пройден), аудит отбора и подбор α закончены. У всех вариантов зафиксирован α = 1,0 — это край сетки, ограничение записано в `outputs_v2/README.md`.
* Финальные обучения (6 вариантов × seeds 7/17/27 = 18) **остановлены пользователем** — GPU понадобился соседней задаче. Состояние:

  | обучение | что сделано |
  |---|---|
  | seed7/baseline | готово (`completed.pt`) |
  | seed7/uniform | SMT готов, DMT 750/1500 |
  | seed7/random_k | SMT готов, DMT 250/1500 |
  | seed7/surprise_ctx_k | SMT 1250/3000 |
  | остальные 14 | не начаты |

* Test **не создавался**. Отчёт по test — только после всех 18 обучений и с разрешения пользователя.
* Предрегистрация (правило вывода): `runs/stack02_4c69a2819ab8/preregistration.md`, она же — раздел 8.0 ноутбука 02.
* Предварительные графики по подбору α (один seed, не test): `outputs_v2/README.md`.

**MQAR** (ноутбук 03): ничего не обучалось. Готовы:
* код и 12 проверок;
* аудит данных и отбора с идеальными детекторами;
* выбор окна r по правилу по данным;
* оценки стоимости — **не замер**.

Главный вывод: на MQAR почти всем ответам нужна память, поэтому отбор различает в основном давние и недавние ответы. Даже идеальный детектор обогащает дальние ответы в 1,1–1,3 раза (в Stack — ×2,9). Это нужно обсудить с научником до дорогих запусков. Итоги: `outputs_mqar/README.md`.

### 2. Правила работы с пользователем (важно)
* Общение по-русски, коротко и по делу; о прогрессе долгих обучений писать в чат (завершения SMT/DMT, ошибки).
* **GPU общий, соседние задачи в приоритете.** Ничего не запускать на GPU — ни обучение, ни замер — без явного разрешения пользователя. Работа на CPU (ноутбуки, анализ, графики, тесты) — можно, с `CUDA_VISIBLE_DEVICES=""`.
* Свои процессы: 4 потока, `nice -n 10`, лимит доли памяти GPU (`--gpu-memory-fraction`), перед запуском смотреть `nvidia-smi --query-compute-apps`. Останавливать по PID: `pkill -f` может убить собственную оболочку.
* Отдельное окружение `envs/smt-memup`; окружения других проектов не трогать.
* Перед пушем проверять, что в коммит не попадают большие checkpoints; push — только когда пользователь попросит.

### 3. Научные договорённости (не нарушать)
* Согласовано с научником: к **исходному** SMT loss добавляется MemUP-подобное слагаемое; задачи — Stack Operations **и** MQAR, без шума; MNIST позже; отдельно проверить, сохраняется ли информация энкодера при последовательных обновлениях RNN (H2).
* Наши настройки (научник их не фиксировал): окно r (Stack — 8, пришло из ноутбука 01; MQAR — по правилу), K, размеры моделей, шаги, seeds, пороги gate, первичная метрика.
* Основное сравнение — baseline / uniform / random-K / surprise-K. `surprise_ctx_k` и `oracle_prefix_k` — вторичная диагностика (oracle читает metadata генератора).
* Test — один раз, в отчёт входят **все шесть** вариантов. По test ничего не выбирать. Отсутствие эффекта при этом бюджете не является общим опровержением гипотезы.
* **Не менять `smt_memup_stack.py`**, пока не сделан отчёт по Stack. Его хэш — часть идентификатора запуска: после правки код создаст новый каталог и не найдёт α и checkpoints.

### 4. Перенос и проверка (CPU)
```bash
git clone git@github.com:RomadovaIrina/smtMemUp.git && cd smtMemUp
# окружение и upstream — команды выше
envs/smt-memup/bin/python checks/check_resume_identity.py        # должно напечатать "OK"
CUDA_VISIBLE_DEVICES="" envs/smt-memup/bin/python checks/test_smt_memup_stack.py
CUDA_VISIBLE_DEVICES="" envs/smt-memup/bin/python checks/test_smt_memup_mqar.py
```
* Если `check_resume_identity.py` пишет MISMATCH — поставить точные версии из раздела «Окружение»: Python 3.12.3, torch 2.9.1+cu128, x-transformers 2.18.5. Код не трогать.
* Пути в ноутбуках заданы как `ROOT = Path("/mem/work")`. На новом сервере положить репозиторий туда же или поменять `ROOT` в первой кодовой ячейке.
* Начатые финальные обучения делались на RTX 5090. Продолжение на другой GPU отличается только недетерминизмом CUDA; это стоит упомянуть в отчёте.

### 5. Следующие шаги (каждый запуск GPU — только с разрешения)
1. **Финал Stack** — продолжится с сохранённых checkpoints. Три параллельных процесса ждут свободную память; каждый занимает ≤ 30% GPU.
   ```bash
   setsid nohup checks/run_final_parallel.sh > /dev/null 2>&1 < /dev/null &
   # логи: runs/final_parallel.log, runs/final_w1.log … final_w3.log
   ```
   Без параллели: `checks/wait_and_run.sh "final" "baseline,uniform,random_k,surprise_k,surprise_ctx_k,oracle_prefix_k" runs/tune_final.log`. Скрипты сами находят корень репозитория; Python берут из `PY` (по умолчанию `/mem/envs/smt-memup/bin/python`, иначе `envs/smt-memup/bin/python` в репозитории). Время: ~25 мин на обучение, ~3 ч на три параллельных процесса, ~7,5 ч последовательно. Порог старта первого процесса (`FIRST_MIB=28000`) рассчитан на пустую карту 32 ГБ.
2. **Отчёт Stack** (после 18 обучений, с разрешения): команда `report` выше, затем ячейка раздела 9 ноутбука 02. Графики по test в `make_outputs.py` ещё не сделаны.
3. **MQAR:**
   * обсудить с научником конфигурацию и то, что отбор на MQAR различает слабее;
   * с разрешения — замер на GPU: раздел 6.1 ноутбука 03 или `python smt_memup_mqar.py --phase benchmark --config zoology_128 --override '{"anchors_per_sequence": 32, "dmt_rollout": 64}' --gpu-memory-fraction 0.25`;
   * далее audit → baseline → предрегистрация → tune → final → report.
4. Отдельно потом: влияние окна r на Stack (там r = 8 — наш выбор).

### 6. Что лежит в `runs/stack02_4c69a2819ab8/` (в git)
* `identity.json` — идентификатор запуска;
* `preregistration*.md` — правило вывода;
* `locked_alphas.json`, `alpha_tuning.csv` — подобранные α;
* `baseline_*` — разметка и проверки baseline;
* `detector_*.pt` — детекторы;
* `*audit*`, `detector_quality.json` — аудит данных и отбора;
* `tuning/*/*_history.csv` — validation по шагам; `analysis_tuning/` — кеш оценок для `make_outputs.py`;
* `final/seed7/*` — checkpoints финальных обучений.

Модели подбора α (`tuning/*/completed.pt`) не перенесены: для продолжения и графиков они не нужны.
