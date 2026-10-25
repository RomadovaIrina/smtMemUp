#!/usr/bin/env python3
"""Builds SMT_MemUP_FROM_ZERO_v2.ipynb.
Version 2 = version 1 with (a) every table / figure rendering moved to viz_utils.py, (b) new sections: definitions,
mathematical formulation of Stack and MQAR, configurations and how every run was launched, and a models section with
code, schema, dimensions, parameters and losses for each model. The computational cells are taken unchanged from
build_notebook.py (version 1), so the equivalence with the archived code (§14) checks exactly the same functions.
    python build_notebook_v2.py --execute        # build and run top to bottom (SMOKE, CPU)"""
import argparse
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import build_notebook as V1  # noqa: E402  (fills V1.CELLS; nothing is executed)

SRC = {c.metadata["tags"][0]: c.source for c in V1.CELLS if c.cell_type == "code"}
MD1 = {c.metadata["tags"][0]: c.source for c in V1.CELLS if c.cell_type == "markdown"}
V1_TO_V2 = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 12, 6: 7, 7: 8, 8: 9, 9: 10, 10: 11, 11: 13, 12: 14, 13: 15, 14: 16, 15: 17, 16: 18}


def remap(text):
    """Renumber §N references of a version-1 markdown cell to the version-2 numbering."""
    return re.sub(r"§(\d+)", lambda m: f"§{V1_TO_V2[int(m.group(1))]}", text)


MD1 = {k: remap(v) for k, v in MD1.items()}
CELLS = []


def md(tag, text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n"), metadata={"tags": [tag]}))


def code(tag, text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n"), metadata={"tags": [tag]}))


def v1(tag, *replacements):
    """Reuse a version-1 code cell; replacements only swap rendering calls (display -> vz.show) - no computation."""
    s = SRC[tag]
    for a, b in replacements:
        assert a in s, (tag, a[:50])
        s = s.replace(a, b)
    assert "display(" not in s and "plt." not in s, f"rendering left in {tag}"
    code(tag, s)


# =====================================================================================================================
md("title", r"""
# SMT × MemUP с нуля — версия 2

* **Что изменилось по сравнению с версией 1:**
  * весь код таблиц и графиков вынесен в `viz_utils.py`; в ноутбуке только вычисления;
  * добавлены разделы: определения (§1), математическая постановка задач (§2), конфигурации и как запускали (§3),
    модели — код, схема, размерности, параметры и лоссы для каждой (§6–§7).
* **Вычислительный код** (данные, модели, лоссы, обучение, оценка) тот же, что в версии 1. В §14 он сверяется со
  старой реализацией.
* **Режимы:**
  * `SMOKE` (по умолчанию) — CPU, ~2 минуты: проверки, трассы, сверка;
  * `FULL` — обучает один выбранный эксперимент (§15).
* **Ячейки** найти по заголовку `§N` и тегу (`metadata.tags`).
""")

# ---------------------------------------------------------------------------------------------------------------------
md("s0-setup", "## §0. Зависимости, режим, seed")
v1("s0-setup", ("import matplotlib.pyplot as plt\n", "import viz_utils as vz          # tables and figures only\n"),
   ('pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)\n', ""))
md("s0-start", MD1["s0-start"])
v1("s0-seeds")

# =====================================================================================================================
md("s1-definitions", r"""
## §1. Определения

Каждый термин ниже значит ровно одно во всём ноутбуке. Если в старых файлах слово употреблялось иначе, это отмечено.

### Данные
| термин | определение |
|---|---|
| **ID (токен)** | целое число из `{0, …, V−1}`, индекс строки таблицы эмбеддингов. Текстовой токенизации нет |
| **последовательность** | `x = (x_0, …, x_{T−1})` из T ID; одна строка батча |
| **ответ (метка)** `y_t` | правильный ID на позиции t или `−1` («ответа нет») |
| **маска ответов** `target_mask` | `y_t ≠ −1`. Только эти позиции входят в loss и в метрики |
| **источник** `source_pos(t)` | позиция, где ответ `y_t` был записан во входе: push элемента (Stack) или значение пары (MQAR) |
| **задержка** `delay(t)` | `t − source_pos(t)` |
| **длинный ответ** | ответ с `delay ≥ long_delay` (Stack: 8; MQAR при T = 16 / 32 / 64: 4 / 8 / 24) |
| **угадывание (chance)** | точность случайного ответа: Stack — 1/4; MQAR по словарю — 1/(V/2); MQAR среди записанных — 1/N |
| **поток данных** | детерминированный генератор батчей: батч шага s — `derive_seed(поток, seed, s)` |
| **validation** | 512 последовательностей потока 501 (Stack) / 511 (MQAR); по ним **выбирается** чекпойнт |
| **tuning** | поток 502 / 512; в архиве — выбор α и LR, gate |
| **test** | поток 503 (только Stack v1); в версиях 2–4 test не создавался |
| **независимая оценка** | поток 901, 1024 последовательности; только для новых запусков, ни для какого выбора не используется |

### SMT
| термин | определение |
|---|---|
| **якорь (anchor)** `t` | позиция, для которой teacher строит память `m_t` по `x_0..x_t`. В SMT якорем является **каждая** позиция |
| **окно энкодера (контекст)** `x_ctx[t]` | `x_{t−T+1..t}`, слева паддинг `−1`. Так как длина окна = T, энкодер всегда видит весь префикс `x_0..x_t` |
| **будущие входы** `x_fut[t]` | `x_{t+1..t+T}`, справа паддинг `−1` |
| **цели декодера** `y_fut[t]` | `y_{t..t+T−1}` |
| **выход j** | j-й выход декодера для якоря t. Видит `m_t` и `x_{t+1..t+j}`, предсказывает `y_{t+j}` |
| **окно предсказания** `Tf` | выходы `j < Tf` входят в основной loss `L_dec`. **Tf = 8** — целевая постановка (версии 2, 4, абляции); **Tf = T = 64** — Stack версии 1 и абляция `tf64`; в MQAR всегда Tf = T |
| **ближняя цель** | валидная пара (t, j) с `j < Tf` |
| **дальняя цель** | валидная пара (t, j) с `j ≥ Tf` |
| **prefix-цель** | ответ записан не позже якоря (`source ≤ t`), значит, обязан быть в `m_t`. Метаданные |
| **teacher** | модель «энкодер + декодер» (§6.1). Строит память по окну и предсказывает будущие ответы. На инференсе итоговой модели не используется |
| **энкодер** | двунаправленный Transformer teacher-а; превращает окно и K регистров в память |
| **регистры памяти** | K обучаемых векторов, приписанных к окну справа; их выходы и есть память |
| **память** `m_t` | K векторов по d = 64 (после RMS-нормы), развёрнутые в вектор длины Z = K·d = 512 |
| **декодер** | каузальный Transformer teacher-а над `[m_t, embed(x_fut)]` |
| **teacher readout** | выход j = 0 декодера: предсказание `y_t` по одной `m_t` |
| **RNN** | итоговая рекуррентная модель (§6.2): обновитель f + readout |
| **обновитель** f | `m_t = f(m_{t−1}, x_t)` — Transformer-энкодер над K + 1 токенами. Не GRU |
| **readout RNN** | свой декодер и голова, инициализированные **копией** декодера и головы teacher |
| `m̂_t` | предсказание обновителя на памяти teacher: `f(m_{t−1}, x_t)` в SMT |
| `m^rnn_t` | состояние RNN при собственном откате с `m_{−1} = 0` (DMT, BPTT, инференс) |
| **SMT** | стадия совместного обучения teacher и f: `L_dec (+ L_far) + 0,1·L_dyn + 0,001·L_unif` |
| **DMT** | стадия обучения только RNN на своих состояниях: MSE к памяти teacher + CE readout. Teacher заморожен |
| **full BPTT / BPTT-1** | та же RNN, обученная прямо на ответах; градиент через все шаги / только через текущий шаг |

### Отбор дальних целей и абляции
| термин | определение |
|---|---|
| **base** / `base_window` | SMT с Tf = 8 без дальних целей (`far = "none"`). Архивная абляция `full` — это он же |
| **детектор** | отдельная маленькая GRU, оценивающая «неожиданность» цели. Заморожен, с SMT-моделями весов не делит |
| `surprise_far` | дальние цели с наибольшим CE оконного детектора (8 последних входов) |
| `ctx_far` | дальние цели с наибольшим CE детектора по входам декодера без памяти |
| `random_far`, `uniform_far` | случайные K / все дальние цели |
| **oracle** (`oracle_far`) | пул дальних целей ограничен prefix-целями по разметке генератора. **Диагностика**, не метод |
| **target_detached** (старое имя `detached`) | L_dyn с отцеплённой целью `m_t`; градиент в энкодер через `m_{t−1}` остаётся |
| **fully detached** | L_dyn с отцеплёнными `m_{t−1}` и `m_t` (= авторский `detach_rnn = True`); в энкодер градиента нет |
| `no_dyn`, `no_unif` | `c_dyn = 0` / `c_unif = 0` |
| `tf1 / tf4 / tf16 / tf64`, `mem4 / mem16` | Tf = 1 / 4 / 16 / 64; K = 4 / 16 |

### Оценка
| термин | определение |
|---|---|
| **accuracy** | доля ответов, где argmax логитов по всему словарю = `y_t` |
| **accuracy среди записанных** (MQAR, `cand`) | argmax только по N значениям, записанным в этой последовательности; диагностика, угадывание 1/N |
| **выбранный чекпойнт** | веса с наименьшим validation CE на длинных ответах |
| **последняя точка** | веса после последнего шага обучения |
| **far-memory probe** | точность декодера teacher на дальних prefix-целях по `m_t` и входам декодера (архивная диагностика) |
| режимы оценки | `rnn`, `teacher` (j = 0), `teacher_pre` (память до запроса), `teacher_full`, `lm` — §10 |
""")

# =====================================================================================================================
md("s2-math", r"""
## §2. Математическая постановка задач

### 2.1 Stack

**Алфавит.** `Σ = {0, 1, …, n}`, n = 4, V = n + 1 = 5. `0` — операция pop; `e ∈ {1..n}` — операция «push e».

**Порождение одной последовательности** (генератор авторов SMT). Стек `S` — список, `S_0 = ∅`, ёмкость M = 8. Для
`t = 0..T−1`:
1. `push_t = 1` с вероятностью 1/2, если `0 < |S_t| < M`; `push_t = 1`, если `|S_t| = 0`; `push_t = 0`, если
   `|S_t| = M`.
2. `e_t ~ U{1..n}` (вытягивается на каждом шаге, используется только при push).
3. Если `push_t`: `x_t = e_t`, `S_{t+1} = S_t ++ [e_t]`, `y_t = −1`.
4. Иначе: `x_t = 0`, `y_t = top(S_t)`, `S_{t+1} = S_t` без вершины.

**Задача.** По префиксу `x_0..x_t` выдать `y_t` для каждого t, где `x_t = 0`. Ответ — детерминированная функция
префикса (симуляция стека). Значит, задача решается без ошибок, если модель хранит содержимое стека.

**Источник и задержка.** Для pop на t источник `s(t)` — момент push снятого элемента; `delay = t − s(t)`. Он всегда
нечётный: между push и его pop лежит сбалансированная подпоследовательность чётной длины. «Длинные» ответы —
`delay ≥ 8`, около 15% ответов.

**Цель обучения итоговой модели:** `min E[ −log p(y_t | x_0..x_t) ]` по позициям с `y_t ≠ −1`.

**Метрика:** accuracy на длинных ответах. Угадывание — 1/4.

### 2.2 MQAR (multi-query associative recall, генератор Zoology, без шума)

**Параметры:** V (словарь, чётный), T (длина, чётная), N (число пар), `4N ≤ T`, a = 0,01.

**Множества ID:** заполнитель `0`; ключи `K = {1..V/2−1}`; значения `W = {V/2..V−1}`.

**Порождение:**
1. Ключи `k_1..k_N` — выборка из K **без повторов**; значения `v_1..v_N` — из W без повторов. Пара i связывает
   `k_i ↦ v_i`.
2. Запись пар: `x_{2(i−1)} = k_i`, `x_{2(i−1)+1} = v_i`. Одна пара = 2 ID, все пары занимают позиции `0..2N−1`.
3. Промежутки: `g_1..g_N` из `{0..(T−2N)/2−1}` без повторов, `P(g) ∝ (g+1)^{a−1}` (последовательный выбор без
   возвращения). Позиция запроса ключа i — `q_i = 2N + 2·g_i` (чётная).
4. `x_{q_i} = k_i`; остальные позиции `≥ 2N` — `0`.
5. Метки: `y_{q_i} = v_i`; все остальные `y_t = −1`.

**Задача.** На каждой позиции запроса t выдать `y_t = v_i`, где `k_i = x_t`. Значение ищется **по содержимому** ключа
среди пар, записанных в начале.

**Свойства, проверенные кодом (§14):**
* после позиции `2N−1` во входе нет ни одного ID значения, поэтому ответ не утекает;
* `y` однозначно выводится из `x` по правилу «значение, записанное сразу после запрошенного ключа».

**Цель обучения:** `min E[ −log p(y_t | x_0..x_t) ]` по позициям запросов.

**Метрики:**
* accuracy по всему словарю: угадывание среди значений — 1/(V/2) = 1/64 при V = 128;
* accuracy среди N записанных значений: угадывание 1/N.

**Задержка** `delay = t − (2i−1)`. Длинные ответы — `delay ≥ T/4` (pilot01: ≥ 24).

### 2.3 Что значит «решить» задачу для каждой модели
* **Итоговая RNN** получает `x_0..x_t` по одному ID и обязана выдать `y_t` из своего состояния (§10, режим `rnn`).
* **Teacher** — вспомогательная модель обучения. Его точность (режимы `teacher`, `teacher_pre`) — диагностика того,
  что память **может** содержать нужную информацию, а не результат метода.
""")

# =====================================================================================================================
md("s3-config", r"""
## §3. Конфигурации и как запускали

### 3.1 Поля конфигурации
Все поля — в ячейке ниже, с комментариями. Здесь — смысл групп:
* **Данные:**
  * `task`, `seq_len` (T);
  * Stack: `n_elements`, `max_stack_size`;
  * MQAR: `vocab` (V), `num_kv_pairs` (N), `power_a`;
  * `long_delay`.
* **Модель:** `d_model` (d), `mem_tokens` (K), `encoder_depth` / `decoder_depth` / `rnn_depth`, `n_heads`.
* **SMT-loss:** `future_window` (Tf), `far` (правило дальних целей), `far_K`, `alpha_far`,
  `coef_dec` / `coef_dyn` / `coef_unif`, `dyn_detach`.
* **Обучение:**
  * `batch_size`;
  * шаги и LR по стадиям: `smt_*`, `dmt_*`, `bptt_*`, `lm_*`;
  * `dmt_coef_dyn`, `warmup_steps`, `weight_decay`, `clip_grad`, `schedule`.
* **Выбор и оценка:** `eval_every`, `validation_examples`, `eval_examples`, `stop_accuracy`.

### 3.2 Архивные запуски (старый код, GPU, 14–20.10.2026)

| имя | код и команда | данные | Tf | бюджет | расписание LR | сиды | выборка |
|---|---|---|---|---|---|---|---|
| Stack v1 | `archive/smt_memup_stack.py --phase final` / `--phase report --confirm-test` | Stack T = 64 | **64 = T** | SMT 3000 / DMT 1500 | разогрев 200 + косинус до 0,1× | 7, 17, 27 | **test 503** |
| Stack v2 | `archive/smt_memup_window.py --phase final` | Stack T = 64 | 8 | 3000 / 1500 | то же | 7, 17, 27 | validation |
| Stack v4, диагностика | `archive/smt_memup_v4.py --phase diag --job base_window / ctx_far` | Stack T = 64 | 8 | 20 000 / 12 000 | разогрев + константа | 107 | validation |
| то же, BPTT | `--job bptt_full_lr{1e-4, 3e-4, 1e-3}`, `bptt_1_lr…` | — | — | 20 000 | то же | 107 | validation; LR по tuning |
| Stack v4, доп. | `archive/checks/smt_memup_v4_extra.py` | — | 8 / 64 | 10 000 / 6000 | то же | 107 | validation |
| абляции | `archive/checks/smt_memup_ablation.py --name <имя>` | — | 8 (tf*: 1 / 4 / 16 / 64) | 11 000 / 5000 | то же | 107 | validation |
| MQAR pilot01 | `archive/smt_memup_mqar.py --phase baseline` | V = 128, T = 64, N = 8 | 64 | 3000 / 1500 | косинус | 107 | tuning |
| MQAR, длина и число пар | `archive/checks/smt_memup_mqar_probe_arch.py --name T16kv2 …` | T = 16 / 32, N = 2 / 4 | T | 6000 / 50 | косинус | 107 | tuning |
| MQAR-бейзлайны | `archive/mqar_baselines.py --job tf_T16kv2 …` | то же | — | Transformer ≤ 3000, RNN 6000, остановка при 0,99 | разогрев + константа | 107 | validation / tuning |
| тест узкого места | `archive/checks/mqar_bottleneck_test.py --mode bottleneck / full --T 16 / 64` | — | T | 4000, LR 3e-4 без разогрева | константа | 107 | validation 128 (только лог) |
| Zoology | `archive/zoology_repro/run_repro.py` | V = 8192, T = 256, N = 16 | — | 64 эпохи, early stop 0,99 | косинус | авторский | test Zoology |

Во всех SMT-запусках: batch 32, AdamW (0,9 / 0,95), wd 0,01, обрезка нормы 1,0, разогрев 200 шагов, LR SMT 3e-4,
LR DMT 1e-4, оценка каждые 250 (v1, v2, MQAR) или 500 шагов (v4, абляции).

### 3.3 Новые запуски (этот ноутбук, §15, очередь `run_experiments.py`)
* **14 экспериментов × сиды 201, 202, 203.** Конфигурации — словарь `EXPERIMENTS` в §15. Бюджеты как в архиве:
  * Stack: SMT 11 000 / DMT 5000, BPTT 20 000, LR 3e-4;
  * MQAR: SMT 6000 / DMT 2000; Transformer 3000 при LR 1e-3 с остановкой 0,99; RNN BPTT 6000 при LR 1e-3 с
    остановкой 0,99.
* **Выбор чекпойнта** — по validation. Числа сообщаются на независимом потоке 901, для выбранного чекпойнта и
  последней точки.
* **Запуск:** на RTX 3090 (`~/smt_memup_project`) с 25.10.2026, от **версии 1** ноутбука: вычислительный код тот же.
""")
v1("s1-config")

# =====================================================================================================================
md("s4-data", r"""
## §4. Данные: код генераторов

**Определения:**
* `sample_stack` / `sample_mqar` возвращают словарь `[B, T]`-тензоров `x`, `y`, `target_mask`, `source_pos`, `delay`
  (§1).
* Генераторы — копии авторских (источник и commit в комментариях). Модели получают только `x`.
""")
v1("s2-stack")
v1("s2-mqar")
code("s4-tables", r'''
stack_val = validation_set(STACK)
assert (stack_val["delay"][stack_val["target_mask"]] % 2 == 1).all(), "Stack delays are odd"
print("Stack: answers per sequence", float(stack_val["target_mask"].sum(1).float().mean()),
      "| share of long answers", float((stack_val["delay"][stack_val["target_mask"]] >= 8).float().mean()))
vz.show(vz.stack_table(stack_val).head(34), title="Stack, validation sequence 0")
pilot_val = validation_set(MQAR_PILOT01)
vz.show(vz.mqar_table(pilot_val, MQAR_PILOT01.num_kv_pairs).head(34), title="MQAR pilot01 (V=128, T=64, N=8), validation sequence 0")
''')

# =====================================================================================================================
md("s5-windows", MD1["s3-windows"])
v1("s3-windows",
   ('''def show_window(batch, t, b=0, js=range(0, 6), Tf=None):
    T = batch["x"].shape[1]; w = windows(batch); r = b * T + t
    print(f"anchor t={t}: x_ctx = {w['x_ctx'][r].tolist()}")
    rows = [dict(j=j, decoder_inputs=str(w["x_fut"][r, :j].tolist()), target=f"y_{t+j} = {int(w['y_fut'][r, j])}",
                 valid=bool(w["valid"][r, j]), in_L_dec=(None if Tf is None else bool(w["valid"][r, j] and j < Tf)),
                 written_at_or_before_t=bool(w["prefix"][r, j])) for j in js]
    display(pd.DataFrame(rows))''',
    '''def show_window(batch, t, b=0, js=range(0, 6), Tf=None):   # rendering in viz_utils.window_table
    return vz.window_table(windows(batch), b * batch["x"].shape[1] + t, t, js, Tf)'''))
md("s5-stack-far", MD1["s3-stack-far"])
v1("s3-stack-far")
md("s5-mqar-views", MD1["s3-mqar-views"])
v1("s3-mqar-views")

# =====================================================================================================================
md("s6-models", r"""
## §6. Модели: код, схема, размерности, параметры

**Общие определения:**
* `N` — число окон или последовательностей в вызове; `d = 64`; `K = 8`; `Z = K·d = 512`; `V` — словарь.
* **Transformer-блок** (библиотека `x-transformers` 2.18.5, одинаковый для всех моделей SMT):
  ```
  h ← h + Attn(RMSNorm(h))        Attn: 4 головы × 64 = 256 на q / k / v, выход 256 → 64, rotary на 32 из 64 размерностей
  h ← h + FF(RMSNorm(h))          FF: 64 → 256 → 64, GELU
  после последнего блока: RMSNorm
  ```
* **`Encoder`** — внимание без маски по времени (двунаправленное); `mask` — только маска ключей для паддинга.
* **`Decoder`** — каузальная маска: позиция i видит позиции ≤ i.
* Числа параметров ниже посчитаны кодом в ячейке `s6-params`.

### 6.1 Teacher (энкодер + память + декодер) — `Teacher`
```
x_ctx [N,T] ─embed─▶ [N,T,64] ─┐
регистры [K,64] ─repeat─▶ [N,K,64] ─┴─cat─▶ [N,T+K,64] ─Encoder(3 блока, маска паддинга)─▶ [N,T+K,64]
   ─берём последние K─▶ [N,K,64] ─RMSNorm каждого токена─▶ ─flatten─▶ m_t [N,512]
m_t ─▶ [N,K,64] ─┐
x_fut [N,T] ─embed (тот же)─▶ [N,T,64] ─┴─cat─▶ [N,K+T,64] ─Decoder(2 блока, каузальный)─▶ ─Linear 64→V─▶
   [N,K+T,V] ─срез [K−1 : −1]─▶ logits [N,T,V]: выход j ↔ позиция K−1+j ↔ цель y_{t+j}
```

| часть | форма весов | параметров (Stack / MQAR V = 128) |
|---|---|---|
| `embed` | V × 64, общий для `x_ctx` и `x_fut` | 320 / 8 192 |
| `encoder` | 3 блока | 296 320 |
| `encoder_mem_register` | 8 × 64 | 512 |
| `decoder` | 2 блока | 197 568 |
| `to_logits` | 64 → V | 325 / 8 320 |
| **итого** | — | **495 045 / 510 912** |

**Обучается:** только в SMT. В DMT заморожен.

**Лоссы:** `L_dec^{Tf}`, `L_far` (декодер и энкодер), `L_unif` (энкодер), `L_dyn` (энкодер — если не fully
detached).

### 6.2 RNN (обновитель f + readout) — `RNN`
```
m_{t−1} [N,512] ─▶ [N,K,64] ─┐
x_t [N,1] ─СВОЙ embed─▶ [N,1,64] ─┴─cat─▶ [N,K+1,64] ─Encoder(2 блока)─▶ ─первые K─▶ RMSNorm ─▶ m_t [N,512]
(без residual: m_t не содержит m_{t−1} напрямую)
readout: m_t ─▶ [N,K,64] ─Decoder(2 блока)─▶ последняя позиция ─Linear 64→V─▶ logits [N,V]
```

| часть | параметров (Stack / MQAR) |
|---|---|
| `embed` (свой) | 320 / 8 192 |
| `rnn` = обновитель f | 197 568 |
| `decoder` + `to_logits` = readout (**копия** teacher перед DMT и на каждой оценке SMT) | 197 893 / 205 888 |
| **итого** | **395 781 / 411 648** |

**Обучается:**
* SMT: f и `embed` — только от `L_dyn`, один шаг на памяти teacher; readout не обучается;
* DMT: f и `embed` — от MSE к памяти teacher на своих состояниях; readout — от CE на отцеплённых состояниях;
* BPTT: всё — от CE.

**Инференс итоговой модели:** `m_{−1} = 0`, `m_t = f(m_{t−1}, x_t)`, ответ — `argmax readout(m_t)`.

### 6.3 Детекторы отбора дальних целей — отдельные GRU
```
WindowDetector  (surprise_far): x_{k−7..k} [M,8] ─embed (V+1)→16─▶ GRU 16→64 ─последнее h─▶ Linear 64→V ─▶ CE(y_k)
ContextDetector (ctx_far):      [BOS, x_{t+1..t+j−1}] ─embed (V+2)→16─▶ GRU 16→64 ─▶ Linear 64→V ─▶ CE(y_{t+j})  (без памяти)
```
* Около 16 тыс. параметров каждый.
* Обучены один раз в архивной версии 1 (2000 шагов) и заморожены; файлы `archive/runs/stack02_4c69a2819ab8/detector_*.pt`.
* Скоры считаются под `no_grad`, на инференсе детекторы не используются.

### 6.4 Обычный каузальный Transformer (MQAR-бейзлайн) — `CausalTransformer`
```
x [B,T] ─embed V→64 + обучаемые позиции T×64─▶ Decoder(2 блока, 4 головы, FF×4, каузальный) ─▶ Linear 64→V ─▶ logits [B,T,V]
logits[:, t] ↔ y_t
```
* 214 976 параметров при T = 16, V = 128.
* Ничего не делит с SMT-моделями.

### 6.5 Teacher с полным доступом (диагностика MQAR) — `encode_full` / `decode_full`
* Те же веса teacher-а, но декодер читает **все T + K выходов** энкодера (`[N, T+K, 64]`, паддинг замаскирован) вместо
  K токенов памяти.
* Меняются сразу три вещи:
  * число векторов;
  * прямой доступ к выходам по позициям;
  * нормировка (нет RMS-нормы памяти).

### 6.6 Zoology attention (только архив)
* d 64, 2 блока, 1 голова MHA с dropout 0,1, без MLP, обучаемые позиции 256, V = 8192.
* 574 592 параметра (из лога `archive/runs/zoology_repro`). Код — авторский, `archive/zoology_repro/`.

### 6.7 Почему такие размеры
* **Авторское:** классы, порядок слоёв, RMS-норма памяти, коэффициенты лоссов.
* **Наш выбор, без поиска гиперпараметров:**
  * d = 64, K = 8;
  * глубины 3 / 2 / 2, 4 головы.

  У авторов по умолчанию d 256, K 16, глубины 8 / 4 / 8.
* **Абляцией на одном seed проверены:** K = 4 / 16 и Tf = 1 / 4 / 16 / 64.
""")
v1("s4-model")
v1("s4-make")
code("s6-params", r'''
cnt = lambda m: sum(p.numel() for p in m.parameters())
teacher, rnn = make_models(STACK, SEED); tm, rm = make_models(MQAR_T16, SEED); dets = load_detectors()
params = [("teacher.embed", cnt(teacher.embed), cnt(tm.embed)), ("teacher.encoder", cnt(teacher.encoder), cnt(tm.encoder)),
          ("teacher.encoder_mem_register", cnt(teacher.encoder_mem_register), cnt(tm.encoder_mem_register)),
          ("teacher.decoder", cnt(teacher.decoder), cnt(tm.decoder)), ("teacher.to_logits", cnt(teacher.to_logits), cnt(tm.to_logits)),
          ("TEACHER", cnt(teacher), cnt(tm)), ("rnn.embed", cnt(rnn.embed), cnt(rm.embed)), ("rnn.rnn (updater f)", cnt(rnn.rnn), cnt(rm.rnn)),
          ("rnn readout (decoder + to_logits)", cnt(rnn.decoder) + cnt(rnn.to_logits), cnt(rm.decoder) + cnt(rm.to_logits)),
          ("RNN", cnt(rnn), cnt(rm)), ("WindowDetector", cnt(dets["window"]), None), ("ContextDetector", cnt(dets["context"]), None),
          ("CausalTransformer (T=16)", None, cnt(make_lm(MQAR_T16, 0)))]
assert params[5][1] == 495045 and params[9][1] == 395781 and params[12][2] == 214976
vz.show(params, ["part", "Stack (V=5)", "MQAR (V=128)"])
''')

# =====================================================================================================================
md("s7-losses", MD1["s6-losses"] + r"""

**Сводка «лосс → какие параметры получают градиент»** (проверено нормами градиентов в §8):

| лосс | стадия | teacher: embed / encoder / регистры | teacher: decoder / голова | RNN: f / embed | RNN: readout |
|---|---|---|---|---|---|
| L_dec^{Tf}, L_far | SMT | да | да | — | — |
| L_dyn (joint) | SMT | да (через `m_{t−1}` и `m_t`) | — | да | — |
| L_dyn target_detached | SMT | да (через `m_{t−1}`) | — | да | — |
| L_dyn fully detached | SMT | — | — | да | — |
| L_unif | SMT | да | — | — | — |
| MSE DMT | DMT | заморожен | заморожен | да, один шаг | — |
| CE readout DMT | DMT | заморожен | заморожен | — | да |
| CE BPTT | BPTT | — | — | да (через все шаги / только текущий) | да |
""")
v1("s6-losses")
v1("s6-smoke", ('display(pd.DataFrame(out, columns=["loss", "value", "L_far"]))', 'vz.show(out, ["loss", "value", "L_far"])'))

# =====================================================================================================================
md("s8-subtle", MD1["s7-subtle"])
for t in ("s7-encoder-prefix", "s7-j0", "s7-tf-compute"):
    v1(t)
v1("s7-ldyn", ("gd = pd.DataFrame([enc_grad(m) for m in (\"none\", \"target\", \"both\")]); display(gd)",
               "gd = vz.show([enc_grad(m) for m in (\"none\", \"target\", \"both\")])"))
for t in ("s7-dmt", "s7-bptt", "s7-labels"):
    v1(t)

md("s9-train", MD1["s8-train"])
v1("s8-train")
md("s10-eval", MD1["s9-eval"])
v1("s9-eval")
md("s11-fit", MD1["s10-fit"])
v1("s10-fit")

# =====================================================================================================================
md("s12-trace", r"""
## §12. Трасса тензоров: B = 2, T = 8 и B = 32, T = 64

**Определение.** Трасса — список промежуточных тензоров прямого прохода SMT с формой и смыслом осей. На каждой
границе стоит `assert`, что пошаговое вычисление совпадает с `encode` / `decode`.
""")
v1("s5-trace", ("display(trace(replace(STACK, seq_len=8, max_stack_size=4), B=2))\ndisplay(trace(STACK, B=32))",
                "vz.show(trace(replace(STACK, seq_len=8, max_stack_size=4), B=2))\nvz.show(trace(STACK, B=32))"))
md("s13-examples", MD1["s11-trace-examples"])
v1("s11-trace-examples", ("sb = sample(tiny_cfg, 1, derive_seed(901, 0)); display(stack_table(sb)); follow_answer(tiny_cfg, t_s, r_s, sb)",
                          "sb = sample(tiny_cfg, 1, derive_seed(901, 0)); vz.show(vz.stack_table(sb)); follow_answer(tiny_cfg, t_s, r_s, sb)"),
   ("mq1 = sample(MQAR_T16, 1, derive_seed(901, 0)); display(mqar_table(mq1, MQAR_T16))",
    "mq1 = sample(MQAR_T16, 1, derive_seed(901, 0)); vz.show(vz.mqar_table(mq1, MQAR_T16.num_kv_pairs))"))

md("s14-equivalence", MD1["s12-equivalence"])
v1("s12-equivalence", ("display(pd.DataFrame(REPORT))", "vz.show(REPORT)"))
v1("s12-fit-equivalence", ("eq_table = pd.DataFrame(REPORT); display(eq_table.tail(3));", "eq_table = pd.DataFrame(REPORT); vz.show(eq_table.tail(3));"))
v1("s12-interventions")

md("s15-full", MD1["s13-full"])
v1("s13-full")

# =====================================================================================================================
md("s16-archive", MD1["s14-archive-results"])
code("s16-archive", r'''
R = ARCHIVE / "runs"
def best_last(path, acc, ce):
    """selected checkpoint (min validation CE on long answers) vs last point vs maximum over the curve"""
    h = pd.read_csv(path).dropna(subset=[ce]); i = h[ce].idxmin()
    return dict(best_step=int(h.loc[i, "step"]), at_best=float(h.loc[i, acc]), last_step=int(h.step.iloc[-1]),
                last=float(h[acc].iloc[-1]), max=float(h[acc].max()))
rows = []
for name, d in [("v4 base_window Tf=8 (20k/12k)", "stackv4_d459c6921638/diag/base_window"), ("v4 ctx_far (20k/12k)", "stackv4_d459c6921638/diag/ctx_far")]:
    rows.append(dict(run=name, stage="DMT: final RNN", **best_last(R / d / "dmt_history.csv", "val_rnn_long_acc", "val_rnn_long_ce")))
    rows.append(dict(run=name, stage="SMT: far-memory probe", **best_last(R / d / "smt_history.csv", "val_far_memory_acc", "val_teacher_long_ce")))
for v in ("bptt_full", "bptt_1"):
    rows.append(dict(run=f"v4 {v} LR 3e-4", stage="BPTT: RNN", **best_last(R / f"stackv4_d459c6921638/diag/{v}_lr0.0003/bptt_history.csv", "val_rnn_long_acc", "val_rnn_long_ce")))
abl = {"full": "8318239d86e6", "target_detached (old name 'detached')": "8318239d86e6", "no_dyn": "93364b393f72", "no_unif": "7af76cfa569f",
       "tf1": "7c073f59ff92", "tf4": "03af753e0b10", "tf16": "b93af0011028", "tf64": "8067828308c1", "mem4": "7ccc3a06ed73", "mem16": "4cd5be1d3493"}
for name, key in abl.items():
    folder = "detached" if name.startswith("target_detached") else name
    rows.append(dict(run=f"ablation {name} (11k/5k)", stage="DMT: final RNN", **best_last(R / f"stackv4_{key}/ablation/{folder}/dmt_history.csv", "val_rnn_long_acc", "val_rnn_long_ce")))
res = vz.show(pd.DataFrame(rows).round(4), title="archived results (old code, GPU, seed 107, validation)")
b1 = res[res.run.str.contains("bptt_1")].iloc[0]
assert round(b1["at_best"], 3) == 0.338 and round(b1["last"], 3) == 0.367
same = (pd.read_csv(R / "stackv4_8318239d86e6/ablation/full/smt_history.csv")["loss"].values ==
        pd.read_csv(R / "stackv4_d459c6921638/diag/base_window/smt_history.csv")["loss"].values[:11000]).all()
print(f"BPTT-1: selected {b1['at_best']:.1%} (step {b1['best_step']}) vs last {b1['last']:.1%} (step {b1['last_step']});",
      "ablation 'full' = v4 base_window trajectory:", bool(same))
vz.archived_curves(R, [
    ("Stack v4: final RNN during DMT", {"base_window Tf=8": "stackv4_d459c6921638/diag/base_window/dmt_history.csv",
                                        "ctx_far": "stackv4_d459c6921638/diag/ctx_far/dmt_history.csv"}, "val_rnn_long_acc", "val_rnn_long_ce"),
    ("Stack v4: same RNN trained directly", {"full BPTT": "stackv4_d459c6921638/diag/bptt_full_lr0.0003/bptt_history.csv",
                                            "BPTT-1": "stackv4_d459c6921638/diag/bptt_1_lr0.0003/bptt_history.csv"}, "val_rnn_long_acc", "val_rnn_long_ce"),
    ("Stack v4: SMT teacher, j=0", {"base_window Tf=8": "stackv4_d459c6921638/diag/base_window/smt_history.csv",
                                    "ctx_far": "stackv4_d459c6921638/diag/ctx_far/smt_history.csv"}, "val_teacher_long_acc", "val_teacher_long_ce")])
vz.show(pd.read_csv(ARCHIVE / "outputs_mqar/table_length_sweep.csv"), title="MQAR length sweep (archive; teacher = j=0 at the query; tuning; seed 107)")
''')

md("s17-changes", MD1["s15-changes"])
md("s18-summary", "## §18. Итог прогона")
v1("s16-summary", ("subtle-point asserts (§7), interventions (§12), SMOKE fit (§10): passed", "subtle-point asserts (§8), interventions (§14), SMOKE fit (§11): passed"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    nb = nbf.v4.new_notebook(cells=CELLS, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    path = ROOT / "SMT_MemUP_FROM_ZERO_v2.ipynb"
    if a.execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, path)
    print(f"wrote {path.name}: {len(CELLS)} cells")


if __name__ == "__main__":
    main()
