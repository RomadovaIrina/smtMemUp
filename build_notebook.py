#!/usr/bin/env python3
"""Builds SMT_MemUP_FROM_ZERO.ipynb from the cell list below (stable headings + cell tags).
    python build_notebook.py                     # write the notebook (no outputs)
    python build_notebook.py --execute           # write and run top to bottom in SMOKE mode (CPU), save outputs
The notebook is the deliverable; this file only keeps its cells in one reviewable place."""
import argparse
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
CELLS = []


def md(tag, text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n"), metadata={"tags": [tag]}))


def code(tag, text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n"), metadata={"tags": [tag]}))


# =====================================================================================================================
md("title", r"""
# SMT × MemUP с нуля: Stack и MQAR в одном ноутбуке

* **Как читать.** Ноутбук исполняется сверху вниз. Каждая секция начинается с заголовка вида `§N` и имеет тег ячейки
  (`metadata.tags`). Номера ячеек меняются при правках, заголовки и теги — нет. Навигация «вопрос → заголовок →
  функция» и разбор по ячейкам — в `NOTEBOOK_WALKTHROUGH_RU.md`.
* **Что внутри:** весь код, нужный для обсуждаемых постановок:
  * данные — копия генераторов авторов;
  * модели — копия классов `Teacher` / `RNN` авторов SMT, слои Transformer из библиотеки `x-transformers`;
  * окна, все лоссы, SMT, DMT, full BPTT, BPTT-1, абляции, MQAR-бейзлайны, диагностика, оценка и графики.
* **Архивный код не импортируется.** Модули `archive/smt_memup_*.py` подключаются только в §12, чтобы сверить
  результаты.
* **Режимы:**
  * `SMOKE` (по умолчанию) — CPU, несколько минут: формы, трассы, один forward/backward каждого лосса, все проверки,
    сверка со старым кодом, графики архивных результатов;
  * `FULL` — дополнительно обучает **один явно выбранный** эксперимент (§13) в новый каталог `results/notebook_runs/`.
    При обычном `Run All` в режиме `SMOKE` дорогое обучение не запускается.
* **Пометки:**
  * **[история]** — так было в архивных GPU-запусках;
  * **[новое]** — диагностика, добавленная 21–24.10.2026, на GPU ещё не запускалась.
""")

# =====================================================================================================================
md("s0-setup", r"""
## §0. Зависимости, режим, seed, пути

* **Версии исторических GPU-запусков:** Python 3.12.3, torch 2.9.1+cu128, x-transformers 2.18.5, einops 0.8.1, numpy
  2.1.0, pandas 2.3.3.
* **Авторский код:** нужен только для сверки в §12. Его коммиты закреплены, клонирование — в `START` ячейке ниже
  (выполняется вручную).
""")
code("s0-setup", r'''
MODE = "SMOKE"            # "SMOKE": CPU checks only.  "FULL": also trains FULL_EXPERIMENT (§13)
FULL_EXPERIMENT = None    # e.g. "stack_smt_tf8"; FULL does nothing while this is None
FULL_SEED = 201           # seed of the FULL run (fixed in advance: 201, 202, 203)
SEED = 0                  # seed of the demonstration cells (the experiments use their own fixed seeds, §13)

import copy, hashlib, json, math, os, platform, random, subprocess, sys, time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import importlib.metadata as _md
from x_transformers import Decoder, Encoder, TransformerWrapper
import matplotlib.pyplot as plt

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "archive").is_dir() and (p / "build_notebook.py").exists())
ARCHIVE = ROOT / "archive"                    # old code, runs, logs - read only
UPSTREAM = ROOT / "upstream"                  # author repositories (only for the comparison in §12)
PINNED = {"smt": "cf0d0f04be606ab9d97b1a449a6846450d656ad0", "zoology": "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb",
          "memup": "d27657137f6e427e30fcd4ae885b20b071b2917d"}
IGNORE = -1                                   # "no answer here" in labels, "no token" in padded windows
DEVICE = torch.device("cpu")                  # all checks (§2-§12) run on CPU; §13 switches FULL training to the GPU
torch.set_num_threads(4)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)
print("python", platform.python_version(), "| torch", torch.__version__, "| x-transformers", _md.version("x-transformers"),
      "| device", DEVICE, "| MODE", MODE)
''')
md("s0-start", r"""
**START (вручную, один раз): авторские репозитории для сверки (§12).**
```bash
git clone https://github.com/akarshkumar0101/smt.git upstream/smt && git -C upstream/smt checkout cf0d0f04be606ab9d97b1a449a6846450d656ad0
git clone https://github.com/HazyResearch/zoology.git upstream/zoology && git -C upstream/zoology checkout 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb
# архивный код ищет их в archive/upstream:  ln -s ../upstream archive/upstream   (Windows: mklink /J archive\upstream upstream)
```
""")
code("s0-seeds", r'''
def derive_seed(*parts):
    """One integer per (stream, run seed, step): a batch never depends on what was sampled before it (as archived)."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])

@contextmanager
def seeded(seed):
    """The author generators and calc_uniformity_loss use the GLOBAL numpy / torch RNGs: seed them inside the block,
    restore the previous state afterwards (identical to archive/smt_memup_stack.py:isolated_seed)."""
    st = np.random.get_state(), random.getstate(), torch.get_rng_state()
    cu = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(int(seed) % 2**32); random.seed(int(seed)); torch.manual_seed(int(seed)); yield
    finally:
        np.random.set_state(st[0]); random.setstate(st[1]); torch.set_rng_state(st[2])
        if cu is not None: torch.cuda.set_rng_state_all(cu)
''')

# =====================================================================================================================
md("s1-config", r"""
## §1. Конфигурация и смысл осей

| ось | смысл | основное значение |
|---|---|---|
| `B` | последовательностей в батче | 32 |
| `T` | длина последовательности = окно энкодера | Stack 64; MQAR 16 / 32 / 64 |
| `R = B·T` | число окон (якорей); строка `r = b·T + t` | 2048 |
| `t` | позиция якоря: память строится по `x_0..x_t` | 0..T−1 |
| `j` | номер выхода декодера: выход `j` якоря `t` предсказывает `y_{t+j}` | 0..T−1 |
| `K` | число memory-токенов | 8 |
| `d` (`d_model`) | ширина токена / вектора | 64 |
| `Z = K·d` | длина плоского состояния памяти `m_t` | 512 |
| `V` | число классов = размер словаря ID | Stack 5; MQAR 128 (pilot01) |
| `N` | число пар «ключ–значение» в MQAR | 2 / 4 / 8 |
| `Tf` | сколько выходов `j < Tf` входят в основной loss | **8 в целевой постановке**; 64 = T в Stack v1 и абляции `tf64`; T во всех MQAR |

**Откуда размеры** (подробно — `NOTEBOOK_WALKTHROUGH_RU.md`, §1):
* классы и форма loss — авторские;
* `d = 64`, `K = 8`, глубины 3 / 2 / 2, 4 головы — **наш выбор** для задач с V = 5 и T = 64, без отдельной оптимизации.
  У авторов SMT по умолчанию d = 256, K = 16, глубины 8 / 4 / 8;
* абляцией проверены только K = 4 / 8 / 16 и Tf = 1 / 4 / 8 / 16 / 64, на одном seed;
* бюджеты шагов — по ресурсам и правилу из диагностики v4 (11k / 5k / 20k).
""")
code("s1-config", r'''
@dataclass(frozen=True)
class Config:
    task: str = "stack"          # stack | mqar
    seq_len: int = 64            # T
    n_elements: int = 4          # Stack: push IDs 1..4, pop ID 0 -> V = 5
    max_stack_size: int = 8
    vocab: int = 128             # MQAR: V; keys 1..V/2-1, values V/2..V-1, filler 0
    num_kv_pairs: int = 8        # MQAR: N
    power_a: float = 0.01        # MQAR: Zoology default (query gaps ~ power law)
    long_delay: int = 8          # "long" answers: written >= long_delay positions earlier
    d_model: int = 64            # d
    mem_tokens: int = 8          # K
    encoder_depth: int = 3
    decoder_depth: int = 2
    rnn_depth: int = 2           # depth of the updater f
    n_heads: int = 4
    future_window: int = 8       # Tf
    far: str = "none"            # none | uniform_far | random_far | surprise_far | ctx_far | oracle_far
    far_K: int = 2
    alpha_far: float = 1.0
    coef_dec: float = 1.0        # authors' default
    coef_dyn: float = 0.1        # authors' default (0 = ablation no_dyn)
    coef_unif: float = 0.001     # authors' default (0 = ablation no_unif)
    dyn_detach: str = "none"     # none | target (OLD ablation "detached" = target_detached) | both (authors' detach_rnn)
    batch_size: int = 32
    smt_steps: int = 11000; smt_lr: float = 3e-4
    dmt_steps: int = 5000;  dmt_lr: float = 1e-4; dmt_coef_dyn: float = 10.0   # authors' DMT coef_dyn = 10
    bptt_steps: int = 20000; bptt_lr: float = 3e-4
    lm_steps: int = 3000;   lm_lr: float = 1e-3
    warmup_steps: int = 200; weight_decay: float = 0.01; clip_grad: float = 1.0
    schedule: str = "warmup_constant"   # v4 / ablations; v1 / v2 used "cosine" (to 0.1 x)
    min_lr_ratio: float = 0.1
    eval_every: int = 500; eval_batch_size: int = 128
    validation_examples: int = 512      # stream 501 (Stack) / 511 (MQAR): checkpoint selection
    eval_examples: int = 1024           # stream 901: independent evaluation of NEW runs
    stop_accuracy: float = 0.0          # > 0: early stop at this validation long accuracy

    @property
    def V(self): return self.n_elements + 1 if self.task == "stack" else self.vocab
    @property
    def Z(self): return self.d_model * self.mem_tokens

STACK = Config()                                                                   # Stack, Tf = 8 (target setting)
STACK_V1 = replace(STACK, future_window=64, smt_steps=3000, dmt_steps=1500, schedule="cosine", eval_every=250)
MQAR_T16 = Config(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4, future_window=16)
MQAR_T32 = Config(task="mqar", seq_len=32, num_kv_pairs=4, long_delay=8, future_window=32)
MQAR_PILOT01 = Config(task="mqar", seq_len=64, num_kv_pairs=8, long_delay=24, future_window=64)
print(STACK.V, STACK.Z, MQAR_PILOT01.V)
''')

# =====================================================================================================================
md("s2-data", r"""
## §2. Данные: целые ID, ответы, маска

«Токен» в этом проекте — **целочисленный ID**, индекс строки таблицы `nn.Embedding(V, 64)`. Текстовой токенизации
нет: генераторы сразу выдают ID.

| поле | форма | смысл |
|---|---|---|
| `x` | `[B, T]` | вход на каждой позиции |
| `y` | `[B, T]` | правильный ответ **на этой же позиции**; −1, если отвечать не нужно |
| `target_mask` | `[B, T]` | `y ≠ −1` — единственные позиции, которые входят в loss и метрики |
| `source_pos` | `[B, T]` | где ответ был записан (Stack: push, MQAR: значение пары) — только метаданные |
| `delay` | `[B, T]` | `t − source_pos` — только метаданные |

**Модели получают только `x`.** `y` используется как цель loss, а `source_pos` / `delay` — только для метрик,
аудита и oracle-диагностики. Это проверяется в §12.

### §2.1 Stack — копия авторского генератора
Источник: `upstream/smt/src/my_datasets/synthetic.py:40-64` @ cf0d0f0.
""")
code("s2-stack", r'''
def author_stack_sample_batch(batch_size, ctx_len, n_elements=4, max_stack_size=8):
    """Copy of StackDataset.sample_batch (authors). Global numpy RNG. x: 0 = pop, 1..4 = push e; y: popped e at a pop,
    -1 at a push. push / pop are 50/50 when both are possible; push is forced on an empty stack, pop on a full one."""
    x = torch.zeros(batch_size, ctx_len, dtype=torch.long)
    y = torch.zeros(batch_size, ctx_len, dtype=torch.long)
    for b in range(batch_size):
        stack = []
        for t in range(ctx_len):
            can_push, can_pop = len(stack) < max_stack_size, len(stack) > 0
            if can_push and can_pop: should_push = np.random.rand() < 0.5
            elif can_push: should_push = True
            elif can_pop: should_push = False
            rand_element = np.random.randint(n_elements) + 1          # drawn every step, used only on push
            if should_push:
                stack.append(rand_element); act, obs = rand_element, -1
            else:
                act, obs = 0, stack.pop()
            x[b, t], y[b, t] = act, obs
    return dict(x=x, y=y)

def annotate(x, y, source):
    mask = y.ne(IGNORE)                                                   # [B,T] bool
    pos = torch.arange(x.shape[1]).expand_as(x)                           # [B,T]
    return dict(x=x, y=y, target_mask=mask, source_pos=source,
                delay=torch.where(mask, pos - source, torch.full_like(x, IGNORE)))

def sample_stack(cfg, batch_size, seed):
    with seeded(seed):
        raw = author_stack_sample_batch(batch_size, cfg.seq_len, cfg.n_elements, cfg.max_stack_size)
    x, y = raw["x"], raw["y"]                                             # [B,T], [B,T]
    source = torch.full_like(x, IGNORE)                                   # replay: where each popped element was pushed
    for b in range(len(x)):
        st = []
        for t, tok in enumerate(x[b].tolist()):
            if tok > 0: st.append(t)
            else: source[b, t] = st.pop()
    return annotate(x, y, source)
''')
md("s2-mqar", r"""
### §2.2 MQAR — копия генератора Zoology
Источник: `upstream/zoology/zoology/data/multiquery_ar.py:88-141` @ 1ad20d1, `random_non_queries=False` (без шума).

**Раскладка последовательности:**
* одна пара = **2 ID**: ключ на позиции `2i`, значение на `2i+1`, `i < N`;
* ключи берутся из `1..V/2−1`, значения — из `V/2..V−1`; внутри последовательности без повторов;
* затем каждый ключ запрашивается ровно один раз, на позиции `2N + 2·gap`; `gap` берётся из степенного распределения без
  повторов;
* остальные позиции после пар — заполнитель 0;
* **метка (значение) стоит на позиции запроса**;
* паддинга внутри последовательности нет: −1 появляется только при построении окон (§3).
""")
code("s2-mqar", r'''
def zoology_multiquery_ar(vocab_size, num_examples, input_seq_len, seed, power_a=0.01, num_kv_pairs=8):
    """Copy of zoology multiquery_ar with num_passes=1, random_non_queries=False. Seeds the global numpy RNG itself."""
    assert input_seq_len % 2 == 0 and vocab_size > input_seq_len and 4 * num_kv_pairs <= input_seq_len
    np.random.seed(seed)
    context_size = num_kv_pairs * 2
    key_vocab_size = vocab_size // 2
    key_choices, value_choices = np.arange(1, key_vocab_size), np.arange(key_vocab_size, vocab_size)
    keys = np.apply_along_axis(np.random.choice, 1, np.tile(key_choices, (num_examples, 1)), replace=False, size=num_kv_pairs)
    values = np.apply_along_axis(np.random.choice, 1, np.tile(value_choices, (num_examples, 1)), replace=False, size=num_kv_pairs)
    kvs = np.zeros((num_examples, context_size), dtype=np.int64)
    kvs[:, 0::2], kvs[:, 1::2] = keys, values                             # 2 IDs per pair
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1); p = p / p.sum()
    x = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=x, replace=False, p=p, size=num_kv_pairs)
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)            # query key i at 2N + 2*gap_i
    examples = np.concatenate([kvs, queries], axis=1)                     # [n, T+1]
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)   # value right AFTER its query ...
    return torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])   # ... and labels shifted by one: label AT query

def sample_mqar(cfg, batch_size, seed):
    with seeded(seed):
        x, y = zoology_multiquery_ar(cfg.vocab, batch_size, cfg.seq_len, int(seed) % 2**32, cfg.power_a, cfg.num_kv_pairs)
    x, y = x.long().clone(), y.long().clone()                            # [B,T], [B,T]
    y[y == -100] = IGNORE
    N = cfg.num_kv_pairs
    keys = x[:, 0:2 * N:2]                                               # [B,N]
    slot = (x[:, :, None] == keys[:, None, :]).float().argmax(-1)        # [B,T,N] -> [B,T]: which pair a query asks
    source = torch.where(y.ne(IGNORE), 2 * slot + 1, torch.full_like(x, IGNORE))
    return annotate(x, y, source)

def sample(cfg, batch_size, seed):
    return (sample_stack if cfg.task == "stack" else sample_mqar)(cfg, batch_size, seed)

# data streams (as archived): train batch of step s = derive_seed(301, run_seed, s) (SMT/BPTT/LM), 302 (DMT);
# validation (checkpoint selection) = 501 Stack / 511 MQAR; independent evaluation of NEW runs = 901.
validation_set = lambda cfg: sample(cfg, cfg.validation_examples, derive_seed(501 if cfg.task == "stack" else 511, 0))
evaluation_set = lambda cfg: sample(cfg, cfg.eval_examples, derive_seed(901, 0))
''')
md("s2-tables", r"""
### §2.3 Две разобранные последовательности

**Stack** — валидационная последовательность 0. Столбцы:
* ID — вход;
* действие;
* стек до и после шага (снизу вверх);
* ответ;
* маска;
* `source_pos`, `delay`.

У Stack задержка всегда **нечётная**: между push и его pop лежит сбалансированный кусок. Поэтому `delay ≥ 8`
фактически означает `delay ≥ 9`.

**MQAR** — pilot01: T = 64, N = 8, V = 128. Первые 16 позиций — это 8 пар; дальше идут запросы и нули.
""")
code("s2-tables", r'''
def stack_table(batch, b=0):
    x, y = batch["x"][b].tolist(), batch["y"][b].tolist()
    st, rows = [], []
    for t, tok in enumerate(x):
        before = " ".join(map(str, st)) or "-"
        st.append(tok) if tok > 0 else st.pop()
        rows.append(dict(t=t, input_id=tok, action=f"push {tok}" if tok else "pop", stack_before=before,
                         stack_after=" ".join(map(str, st)) or "-", answer_y=y[t], in_loss=int(y[t] != IGNORE),
                         source_pos=int(batch["source_pos"][b, t]), delay=int(batch["delay"][b, t])))
    return pd.DataFrame(rows)

def mqar_table(batch, cfg, b=0):
    x, y, N = batch["x"][b].tolist(), batch["y"][b].tolist(), cfg.num_kv_pairs
    meaning = lambda t: (f"key {t//2}" if t < 2*N and t % 2 == 0 else f"value {t//2}" if t < 2*N
                         else "QUERY key" if y[t] != IGNORE else "filler 0")
    return pd.DataFrame([dict(t=t, input_id=x[t], meaning=meaning(t), answer_y=y[t], in_loss=int(y[t] != IGNORE),
                              source_pos=int(batch["source_pos"][b, t]), delay=int(batch["delay"][b, t]))
                         for t in range(len(x))])

stack_val = validation_set(STACK)
print("Stack x, y:", stack_val["x"].shape, stack_val["y"].shape, "| answers per sequence:",
      float(stack_val["target_mask"].sum(1).float().mean()), "| share delay>=8:",
      float((stack_val["delay"][stack_val["target_mask"]] >= 8).float().mean()))
assert (stack_val["delay"][stack_val["target_mask"]] % 2 == 1).all(), "Stack delays are odd"
display(stack_table(stack_val).head(34))
pilot_val = validation_set(MQAR_PILOT01)
print("MQAR pilot01: V =", MQAR_PILOT01.V, "| pairs =", MQAR_PILOT01.num_kv_pairs, "-> 2 IDs per pair, positions 0..15;",
      "queries at even positions >= 16")
display(mqar_table(pilot_val, MQAR_PILOT01).head(34))
''')

# =====================================================================================================================
md("s3-windows", r"""
## §3. Окна SMT: что видит энкодер, что видит декодер, что предсказывается

Для **каждой** позиции `t` (якоря) строятся:
```
x_ctx[t] = x[t−T+1 .. t]   (слева −1)   вход энкодера   → память m_t видит x_0..x_t
x_fut[t] = x[t+1 .. t+T]   (справа −1)  входы декодера  → выход j видит x_fut[:j] = x_{t+1..t+j}
y_fut[t] = y[t .. t+T−1]   (справа −1)  цели            → выход j учится предсказывать y_{t+j}
valid    = y_fut ≠ −1                    маска пар (якорь, цель)
```

**Почему `x_fut` начинается с `t+1`, а `y_fut` с `t`, и почему нет сдвига на один токен.**
* Ответ `y_t` — это ответ на сам вход `x_t` (pop или ключ запроса) и стоит на его позиции.
* Память `m_t` уже прочитала `x_t`, поэтому выход `j = 0` предсказывает `y_t` из одной памяти.
* Выход `j ≥ 1` сначала читает `j` новых входов `x_{t+1..t+j}`, после которых стоит ответ `y_{t+j}`.
* В языковой модели вход и цель сдвинуты, потому что ответ — следующий токен. Здесь ответ относится к текущему
  токену, поэтому сдвига нет.
""")
code("s3-windows", r'''
def windows(batch):
    x, y = batch["x"], batch["y"]                                         # [B,T]
    B, T = x.shape
    unfold = lambda z, left, right: F.pad(z, (left, right), value=IGNORE).unfold(1, T, 1)   # [B,T+pad] -> [B,T,T]
    w = dict(x_ctx=unfold(x, T - 1, 0),                                   # [B,T,T]: [b,t,:] = x[b, t-T+1..t]
             x_fut=unfold(x, 0, T)[:, 1:T + 1],                           # [B,T+1,T] -> [B,T,T]: x[b, t+1..t+T]
             y_fut=unfold(y, 0, T - 1),                                   # [B,T,T]: y[b, t..t+T-1]
             source=unfold(batch["source_pos"], 0, T - 1))
    w["valid"] = w["y_fut"].ne(IGNORE)                                    # [B,T,T] bool
    w["prefix"] = w["valid"] & (w["source"] <= torch.arange(T).view(1, T, 1))   # answer written at or before t
    out = {k: v.reshape(B * T, T) for k, v in w.items()}                  # [B,T,T] -> [R,T], row r = b*T + t
    assert out["x_ctx"].shape == (B * T, T)
    return out

def show_window(batch, t, b=0, js=range(0, 6), Tf=None):
    T = batch["x"].shape[1]; w = windows(batch); r = b * T + t
    print(f"anchor t={t}: x_ctx = {w['x_ctx'][r].tolist()}")
    rows = [dict(j=j, decoder_inputs=str(w["x_fut"][r, :j].tolist()), target=f"y_{t+j} = {int(w['y_fut'][r, j])}",
                 valid=bool(w["valid"][r, j]), in_L_dec=(None if Tf is None else bool(w["valid"][r, j] and j < Tf)),
                 written_at_or_before_t=bool(w["prefix"][r, j])) for j in js]
    display(pd.DataFrame(rows))

tiny = annotate(torch.tensor([[1, 2, 0, 3, 0, 0]]), torch.tensor([[-1, -1, 2, -1, 3, 1]]), torch.tensor([[-1, -1, 1, -1, 3, 0]]))
print("hand example: push1 push2 pop(->2) push3 pop(->3) pop(->1)")
show_window(tiny, 2, js=range(6), Tf=2)
show_window(tiny, 5, js=range(3))            # right boundary: everything after T-1 is padding -1 / invalid
''')
md("s3-stack-far", r"""
### §3.1 Stack: одно окно и дальняя цель — зависит ли она от памяти?

* Берём длинный pop `t*` (`delay ≥ 8`) валидационной последовательности 0 и якорь `a < t*` с `t* − a ≥ Tf = 8`. Для
  якоря `a` это **дальняя** цель: выход `j = t* − a ≥ Tf`.
* Цель **зависит от памяти** `m_a`, если элемент был положен на стек не позже `a` (`source ≤ a`): тогда он внутри
  `x_0..x_a` и больше нигде.
* Если `source > a`, push лежит внутри входов декодера `x_{a+1..t*}`, и декодер может «списать» ответ без памяти.
* Это ручное вычисление сверяется с колонкой `prefix` кода.
""")
code("s3-stack-far", r'''
b0 = stack_val; d0 = b0["delay"][0]
t_star = int((d0 >= 8).nonzero()[0]); src = int(b0["source_pos"][0, t_star])
print(f"long pop t*={t_star}: answer y={int(b0['y'][0, t_star])}, pushed at {src}, delay {int(d0[t_star])}")
for a in (src - 1, src, t_star - 8, t_star - 3):
    j = t_star - a
    by_hand = src <= a                                   # element already on the stack at the anchor
    by_code = bool(windows(b0)["prefix"][a, j])
    print(f"  anchor a={a:2d}: j={j:2d} ({'far' if j >= 8 else 'near'}), memory-dependent by hand={by_hand}, prefix={by_code}")
    assert by_hand == by_code
show_window(b0, t_star - 12, js=[0, 1, 11, 12, 13], Tf=8)
''')
md("s3-mqar-views", r"""
### §3.2 MQAR: три момента, которые нужно различать

На pilot01 (N = 8), последовательность 0, первый запрос `q`:

1. **Сразу после записи всех пар, до первого запроса** — якорь `a = 2N − 1 = 15`. Энкодер видел все 8 пар и не видел
   ни одного запроса. Декодер, чтобы ответить на запрос `q`, получает входы `x_16..x_q`; последний из них — ключ
   запроса. **[новое]** — так ещё не оценивали.
2. **На позиции первого запроса** — якорь `t = q`, выход `j = 0`. Энкодер видел все пары **и сам ключ запроса**.
   **[история]** — все архивные числа teacher на MQAR получены так.
3. **В декодере** — выход `j = q − a` при `a < q`. Он видит память `m_a` и входы `x_{a+1..q}`.

Ни в одном из случаев ответ (ID значения ≥ V/2) не встречается во входах после позиции `2N − 1`. Это проверено в §12.
""")
code("s3-mqar-views", r'''
qs = pilot_val["target_mask"][0].nonzero().flatten().tolist(); q, q_last = qs[0], qs[-1]; N = MQAR_PILOT01.num_kv_pairs
print(f"queries at {qs}; first query t={q}: key {int(pilot_val['x'][0, q])} -> answer {int(pilot_val['y'][0, q])}")
w = windows({k: v[:1] for k, v in pilot_val.items()})
for name, a, tq in (("1) after all pairs, before the first query", 2 * N - 1, q),
                    ("2) at the position of the first query (j = 0)", q, q),
                    ("3) in the decoder, last query read from the memory after the pairs", 2 * N - 1, q_last)):
    j = tq - a
    ctx = [v for v in w["x_ctx"][a].tolist() if v != IGNORE]
    print(f"{name}: anchor a={a}, encoder sees x_0..x_{a} ({len(ctx)} IDs; query key inside: {a >= tq}); "
          f"decoder inputs x_{a+1}..x_{tq} = {w['x_fut'][a, :j].tolist()}; output j={j} -> y_fut = {int(w['y_fut'][a, j])}")
    assert int(w["y_fut"][a, j]) == int(pilot_val["y"][0, tq])
    assert not (w["x_fut"][a, :j] >= MQAR_PILOT01.vocab // 2).any(), "no value ID among decoder inputs"
# full window of anchor 15 around the last query and at the right boundary (j = 48 -> y_63; j = 49 -> beyond T: padding)
show_window({k: v[:1] for k, v in pilot_val.items()}, 2 * N - 1, js=[0, 1, j - 1, j, j + 1, 48, 49], Tf=MQAR_PILOT01.future_window)
''')

# =====================================================================================================================
md("s4-model", r"""
## §4. Модели: копия авторских классов, видно всё содержимое

Источник: `upstream/smt/src/model.py:21-226` @ cf0d0f0. Код скопирован без изменения поведения; добавлены только
комментарии с формами и `assert`. В §12 копия сверяется с оригиналом на одинаковых весах.

**Блоки Transformer** — `x_transformers.Encoder` / `Decoder` 2.18.5 (обычная зависимость):
* **устройство слоя:** pre-norm RMSNorm → multi-head attention (4 головы × 64 = 256 на q / k / v, выход 256 → 64) →
  residual → RMSNorm → FF 64 → 256 → 64 (GELU) → residual; после последнего слоя — финальный RMSNorm;
* **позиции:** rotary на первых 32 из 64 размерностей головы, отдельных позиционных эмбеддингов нет;
* **`Encoder`** — двунаправленное внимание, маска ключей для паддинга;
* **`Decoder`** — каузальное внимание (позиция `i` видит `≤ i`).
""")
code("s4-model", r'''
# ---- authors' loss helpers (model.py:21-87) ----
def calc_cross_entropy_loss(logits, y):          # logits [...,V], y [...] -> CE [...]
    shape = y.shape
    logits, y = rearrange(logits, "... V -> (...) V"), rearrange(y, "... -> (...)")
    return F.cross_entropy(logits, y, reduction="none").reshape(shape)

def calc_mse_loss(y_pred, y):                    # elementwise, [...] -> [...]
    return F.mse_loss(y_pred, y, reduction="none")

def calc_uniformity_loss(z, t=2, sample=512):    # z [...,D] -> scalar; uses the GLOBAL torch RNG (randperm)
    z = z / (z.norm(dim=-1, keepdim=True) + torch.finfo(z.dtype).eps)
    z = rearrange(z, "... D -> (...) D")
    idx = torch.randperm(len(z))[:sample]
    z = z[idx]
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

def masked_mean(loss, mask):                     # authors' version has no guard for an empty mask
    return (loss * mask).sum() / mask.sum()

def normalize_mem(z, rms_norm=True, fsq=None):   # RMS-norm over the last axis (each memory token separately)
    if rms_norm: z = F.rms_norm(z, z.shape[-1:])
    assert fsq is None
    return z

@dataclass
class TeacherConfig:
    vocab_size: int = 256; mem_size: int = 256; mem_tokens: int = 1; mem_rms_norm: bool = True; mem_fsq: float = None
    d_model: int = 256; encoder_depth: int = 8; decoder_depth: int = 4; n_heads: int = 8
    rotary_pos_emb: bool = True; use_rmsnorm: bool = True

class Teacher(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert cfg.mem_size % cfg.mem_tokens == 0 and cfg.d_model * cfg.mem_tokens == cfg.mem_size
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)                               # V x d, shared by enc & dec
        self.encoder = Encoder(dim=cfg.d_model, depth=cfg.encoder_depth, heads=cfg.n_heads,
                               rotary_pos_emb=cfg.rotary_pos_emb, use_rmsnorm=cfg.use_rmsnorm)  # bidirectional
        self.decoder = Decoder(dim=cfg.d_model, depth=cfg.decoder_depth, heads=cfg.n_heads,
                               rotary_pos_emb=cfg.rotary_pos_emb, use_rmsnorm=cfg.use_rmsnorm)  # causal
        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size)                              # d -> V
        self.encoder_mem_register = nn.Embedding(cfg.mem_tokens, cfg.d_model)               # K learned registers

    def encode(self, x_ctx):                                       # x_ctx [N,Tc] IDs, -1 = padding
        B, Tc = x_ctx.shape
        mask_x = x_ctx != -1                                       # [N,Tc] True = real token
        x_ctx = x_ctx.clamp(min=0)                                 # -1 -> 0 (masked anyway)
        mask_mem = torch.ones((B, self.cfg.mem_tokens), dtype=torch.bool, device=mask_x.device)
        mask = torch.cat([mask_x, mask_mem], dim=1)                # [N,Tc+K]
        x_emb = self.embed(x_ctx)                                  # [N,Tc] -> [N,Tc,d]
        mem_emb = self.encoder_mem_register(torch.arange(self.cfg.mem_tokens, device=x_ctx.device))   # [K,d]
        mem_emb = repeat(mem_emb, "K D -> B K D", B=B)             # [N,K,d]
        z = self.encoder(torch.cat([x_emb, mem_emb], dim=1), mask=mask)   # [N,Tc+K,d] -> [N,Tc+K,d], key mask
        z = z[:, -self.cfg.mem_tokens:, :]                         # last K outputs = registers -> [N,K,d]
        z = normalize_mem(z, rms_norm=self.cfg.mem_rms_norm, fsq=self.cfg.mem_fsq)   # each token RMS-normed
        z = rearrange(z, "B K d -> B (K d)")                       # [N,K,d] -> [N,Z]
        return z

    def decode(self, z, x_fut):                                    # z [N,Z], x_fut [N,Tf]
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)              # [N,K,d]
        x_fut = x_fut.clamp(min=0)                                 # padding -1 -> 0: only feeds outputs past the end
        h = self.decoder(torch.cat([z, self.embed(x_fut)], dim=1))               # [N,K+Tf,d], causal
        logits = self.to_logits(h)[:, self.cfg.mem_tokens - 1:-1, :]             # positions K-1..K+Tf-2 -> [N,Tf,V]
        return logits                                              # output j at position K-1+j: sees z, x_fut[:j]

    def teacher_readout(self, z):                                  # z [N,Z] -> [N,V]
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)
        return self.to_logits(self.decoder(z)[:, -1, :])           # = decode output j = 0

@dataclass
class RNNConfig:
    vocab_size: int = 256; mem_size: int = 256; mem_tokens: int = 1; mem_rms_norm: bool = True; mem_fsq: float = None
    d_model: int = 256; rnn_depth: int = 8; decoder_depth: int = 4; n_heads: int = 8
    rotary_pos_emb: bool = True; use_rmsnorm: bool = True

class RNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        assert cfg.mem_size % cfg.mem_tokens == 0 and cfg.d_model * cfg.mem_tokens == cfg.mem_size
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)                               # its OWN embedding
        self.rnn = Encoder(dim=cfg.d_model, depth=cfg.rnn_depth, heads=cfg.n_heads,
                           rotary_pos_emb=cfg.rotary_pos_emb, use_rmsnorm=cfg.use_rmsnorm)    # updater f (NOT a GRU)
        self.decoder = Decoder(dim=cfg.d_model, depth=cfg.decoder_depth, heads=cfg.n_heads,
                               rotary_pos_emb=cfg.rotary_pos_emb, use_rmsnorm=cfg.use_rmsnorm)  # readout decoder
        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size)

    def rnn_update(self, z, x):                                    # z [N,Z] = m_{t-1}, x [N,1] = x_t
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)              # [N,K,d]
        inputs = torch.cat([z, self.embed(x)], dim=1)              # [N,K,d] ++ [N,1,d] -> [N,K+1,d]
        z = self.rnn(inputs)                                       # [N,K+1,d], bidirectional over the K+1 tokens
        z_next = z[:, :self.cfg.mem_tokens, :]                     # first K outputs -> [N,K,d]
        z_next = normalize_mem(z_next, rms_norm=self.cfg.mem_rms_norm, fsq=self.cfg.mem_fsq)
        return rearrange(z_next, "B K d -> B (K d)")               # [N,Z] = m_t (no residual connection)

    def rnn_readout(self, z):                                      # z [N,Z] -> [N,V]
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)
        return self.to_logits(self.decoder(z)[:, -1, :])
''')
code("s4-make", r'''
def make_models(cfg, seed, device=None):
    """Same seed -> same initial weights (teacher created first, then the RNN) as every archived run."""
    common = dict(vocab_size=cfg.V, mem_size=cfg.Z, mem_tokens=cfg.mem_tokens, d_model=cfg.d_model,
                  decoder_depth=cfg.decoder_depth, n_heads=cfg.n_heads)
    device = device or DEVICE
    with seeded(seed):
        teacher = Teacher(TeacherConfig(**common, encoder_depth=cfg.encoder_depth)).to(device)
        rnn = RNN(RNNConfig(**common, rnn_depth=cfg.rnn_depth)).to(device)
    copy_readout(teacher, rnn)
    return teacher, rnn

def copy_readout(teacher, rnn):
    """RNN readout (decoder + head) := COPY of the teacher's (separate tensors). At init, at every SMT evaluation and
    before DMT (authors' train_smt.py)."""
    rnn.decoder.load_state_dict(teacher.decoder.state_dict())
    rnn.to_logits.load_state_dict(teacher.to_logits.state_dict())

def encode_all(teacher, x_ctx, chunk=4096):                      # [R,T] -> [R,Z] (chunks only save memory)
    return torch.cat([teacher.encode(p) for p in x_ctx.split(chunk)])

def rnn_rollout(rnn, x, init=None, detach=False):
    """m_t = f(m_{t-1}, x_t), m_{-1} = 0. x [B,T] -> [B,T,Z]. detach=True cuts the gradient between steps."""
    state = torch.zeros(x.shape[0], rnn.cfg.mem_size, device=x.device) if init is None else init   # [B,Z]
    out = []
    for t in range(x.shape[1]):
        state = rnn.rnn_update(state.detach() if detach else state, x[:, t:t + 1])               # [B,Z]
        out.append(state)
    return torch.stack(out, 1)                                       # [B,T,Z]

# ---- diagnostic "full access" (archive/checks/mqar_bottleneck_test.py:27-42) ----
def encode_full(teacher, x_ctx):
    """ALL T+K encoder outputs (not only the K registers), WITHOUT the memory RMS-norm. Compared with encode it changes
    three things at once: number of readable vectors (T+K vs K), direct access to per-position outputs, normalisation."""
    B = x_ctx.shape[0]
    mask = torch.cat([x_ctx.ne(IGNORE), torch.ones(B, teacher.cfg.mem_tokens, dtype=torch.bool, device=x_ctx.device)], 1)
    regs = repeat(teacher.encoder_mem_register(torch.arange(teacher.cfg.mem_tokens, device=x_ctx.device)), "K D -> B K D", B=B)
    return teacher.encoder(torch.cat([teacher.embed(x_ctx.clamp(min=0)), regs], 1), mask=mask), mask   # [N,T+K,d], [N,T+K]

def decode_full(teacher, h, mask, x_fut):                        # h [N,n,d] -> logits [N,T,V]; output j at n-1+j
    n = h.shape[1]
    inp = torch.cat([h, teacher.embed(x_fut.clamp(min=0))], 1)
    m = torch.cat([mask, torch.ones(x_fut.shape, dtype=torch.bool, device=x_fut.device)], 1)
    return teacher.to_logits(teacher.decoder(inp, mask=m))[:, n - 1:-1]

# ---- far-target selection detectors: SEPARATE small GRUs (archive/smt_memup_stack.py:434-481) ----
class WindowDetector(nn.Module):
    """surprise_far [история]: q(y_k | x_{k-r+1..k}), r = 8, GRU from the zero state."""
    def __init__(self, V, embed=16, hidden=64, r=8):
        super().__init__(); self.r = r
        self.embed = nn.Embedding(V + 1, embed, padding_idx=0); self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)
    def forward(self, x_windows):                                  # [M,r] -> [M,V]
        _, h = self.gru(self.embed(x_windows + 1)); return self.head(h[-1])
    def scores(self, batch):                                       # surprise = CE of the TRUE label y_k: [B,T]
        dev = self.head.weight.device
        x, y = batch["x"].to(dev), batch["y"].to(dev)
        win = F.pad(x, (self.r - 1, 0), value=IGNORE).unfold(1, self.r, 1)          # [B,T,r]
        logits = self(win.reshape(-1, self.r)).reshape(*x.shape, -1)                 # [B,T,V]
        ce = F.cross_entropy(logits.flatten(0, 1), y.clamp(min=0).flatten(), reduction="none")
        return (ce.reshape(x.shape) * batch["target_mask"].to(dev)).detach()

class ContextDetector(nn.Module):
    """ctx_far [история]: q(y_{t+j} | x_{t+1..t+j}) WITHOUT memory (what the decoder can infer from its own inputs)."""
    def __init__(self, V, embed=16, hidden=64):
        super().__init__(); self.bos = V + 1
        self.embed = nn.Embedding(V + 2, embed, padding_idx=0); self.gru = nn.GRU(embed, hidden, batch_first=True)
        self.head = nn.Linear(hidden, V)
    def forward(self, x_fut):                                      # [R,T] -> [R,T,V]
        inp = torch.cat([torch.full_like(x_fut[:, :1], self.bos - 1), x_fut[:, :-1]], 1)
        h, _ = self.gru(self.embed(inp + 1)); return self.head(h)

def load_detectors(V=5):
    """Trained once in version 1 (2000 steps), frozen; files in archive/runs/stack02_4c69a2819ab8/."""
    dets = {}
    for kind, cls in (("window", WindowDetector), ("context", ContextDetector)):
        m = cls(V); m.load_state_dict(torch.load(ARCHIVE / f"runs/stack02_4c69a2819ab8/detector_{kind}.pt",
                                                map_location="cpu", weights_only=True)["best_model"])
        dets[kind] = m.to(DEVICE).eval().requires_grad_(False)
    return dets

# ---- MQAR baseline: ordinary causal Transformer (archive/mqar_baselines.py:97-105) ----
class CausalTransformer(nn.Module):
    def __init__(self, cfg, d_model=64, depth=2, heads=4):
        super().__init__()
        self.net = TransformerWrapper(num_tokens=cfg.V, max_seq_len=cfg.seq_len,
                                      attn_layers=Decoder(dim=d_model, depth=depth, heads=heads))   # learned abs. pos.
    def forward(self, x): return self.net(x)                       # [B,T] -> [B,T,V]; logits[:,t] -> y_t

def make_lm(cfg, seed, device=None):
    with seeded(seed): return CausalTransformer(cfg).to(device or DEVICE)
''')
md("s4-params", r"""
### §4.1 Параметры каждой части, посчитанные кодом, и кто что обучает
""")
code("s4-params", r'''
cnt = lambda m: sum(p.numel() for p in m.parameters())
teacher, rnn = make_models(STACK, SEED)
tm, rm = make_models(MQAR_T16, SEED)
dets = load_detectors()
rows = [("teacher.embed (V x d)", cnt(teacher.embed), cnt(tm.embed)),
        ("teacher.encoder (3 layers)", cnt(teacher.encoder), cnt(tm.encoder)),
        ("teacher.encoder_mem_register (K x d)", cnt(teacher.encoder_mem_register), cnt(tm.encoder_mem_register)),
        ("teacher.decoder (2 layers)", cnt(teacher.decoder), cnt(tm.decoder)),
        ("teacher.to_logits (d -> V)", cnt(teacher.to_logits), cnt(tm.to_logits)),
        ("rnn.embed (own)", cnt(rnn.embed), cnt(rm.embed)),
        ("rnn.rnn = updater f (2 layers)", cnt(rnn.rnn), cnt(rm.rnn)),
        ("rnn.decoder + to_logits = readout (copy of teacher's)", cnt(rnn.decoder) + cnt(rnn.to_logits), cnt(rm.decoder) + cnt(rm.to_logits)),
        ("TEACHER total", cnt(teacher), cnt(tm)), ("RNN total", cnt(rnn), cnt(rm)),
        ("WindowDetector (GRU)", cnt(dets["window"]), None), ("ContextDetector (GRU)", cnt(dets["context"]), None),
        ("CausalTransformer (MQAR, T=16)", None, cnt(make_lm(MQAR_T16, 0)))]
display(pd.DataFrame(rows, columns=["part", "Stack (V=5)", "MQAR (V=128)"]))
display(pd.DataFrame([
    ("teacher embed / encoder / registers", "L_dec, L_far, L_unif, L_dyn (unless dyn_detach='both')", "frozen", "-"),
    ("teacher decoder / to_logits", "L_dec, L_far", "frozen", "-"),
    ("rnn.embed, updater f", "L_dyn only (one step, input = teacher memory)", "MSE to teacher states", "CE through the steps"),
    ("rnn readout (decoder + head)", "no gradient; overwritten by copy_readout", "readout CE on detached states", "CE"),
    ("detectors", "frozen (scores under no_grad)", "-", "-")],
    columns=["parameters", "SMT", "DMT", "BPTT"]))
''')

# =====================================================================================================================
md("s5-trace", r"""
## §5. Трасса тензоров: маленький пример и основная конфигурация B = 32, T = 64

Каждая строка таблицы — промежуточный тензор прямого прохода SMT, его форма и что значит каждая ось. Для
memory-регистров дополнительно печатается RMS каждого токена после нормировки: он равен 1, потому что RMS-норма
применяется к каждому из K токенов отдельно.
""")
code("s5-trace", r'''
@torch.no_grad()
def trace(cfg, B, seed=SEED):
    batch = sample(cfg, B, derive_seed(301, seed, 0)); T = cfg.seq_len
    teacher, rnn = make_models(cfg, seed)
    rows = []; add = lambda name, t, axes: rows.append((name, tuple(t.shape), axes))
    add("x (input IDs)", batch["x"], "B, T"); add("y (answers, -1 = none)", batch["y"], "B, T")
    add("target_mask", batch["target_mask"], "B, T")
    w = windows(batch)
    add("x_ctx", w["x_ctx"], "R=B*T windows, T context IDs (left pad -1)")
    add("x_fut", w["x_fut"], "R, T future IDs (right pad -1)"); add("y_fut", w["y_fut"], "R, T targets (j)")
    add("valid", w["valid"], "R, T")
    xc = w["x_ctx"][: min(256, B * T)]                             # a slice (memory), same per-row computation
    emb = teacher.embed(xc.clamp(min=0)); add("embed(x_ctx)", emb, "rows, T, d")
    regs = repeat(teacher.encoder_mem_register.weight, "K D -> N K D", N=len(xc)); add("registers", regs, "rows, K, d")
    enc_in = torch.cat([emb, regs], 1); add("encoder input", enc_in, "rows, T+K, d")
    mask = torch.cat([xc.ne(IGNORE), torch.ones(len(xc), cfg.mem_tokens, dtype=torch.bool)], 1); add("encoder key mask", mask, "rows, T+K")
    enc_out = teacher.encoder(enc_in, mask=mask); add("encoder output", enc_out, "rows, T+K, d")
    mem_tok = F.rms_norm(enc_out[:, -cfg.mem_tokens:], (cfg.d_model,)); add("memory tokens (RMS-normed)", mem_tok, "rows, K, d")
    rms = mem_tok.pow(2).mean(-1).sqrt()
    m = rearrange(mem_tok, "N K d -> N (K d)"); add("m_t (flat)", m, "rows, Z=K*d")
    assert torch.allclose(m, teacher.encode(xc), atol=1e-5)
    xf = w["x_fut"][: len(xc)]
    dec_in = torch.cat([rearrange(m, "N (K d) -> N K d", K=cfg.mem_tokens), teacher.embed(xf.clamp(min=0))], 1)
    add("decoder input [m_t tokens, embed(x_fut)]", dec_in, "rows, K+T, d")
    dec_out = teacher.decoder(dec_in); add("decoder output", dec_out, "rows, K+T, d")
    logits = teacher.to_logits(dec_out)[:, cfg.mem_tokens - 1:-1]; add("logits (positions K-1..K+T-2)", logits, "rows, T outputs j, V")
    assert torch.allclose(logits, teacher.decode(m, xf), atol=1e-5)
    prev = torch.cat([torch.zeros(1, cfg.Z), m[:-1]]); xt = batch["x"].reshape(-1, 1)[: len(xc)]
    upd_in = torch.cat([rearrange(prev, "N (K d) -> N K d", K=cfg.mem_tokens), rnn.embed(xt)], 1); add("updater input [m_{t-1}, x_t]", upd_in, "rows, K+1, d")
    upd_out = rnn.rnn(upd_in); add("updater output", upd_out, "rows, K+1, d")
    m_hat = rnn.rnn_update(prev, xt); add("m^_t = f(m_{t-1}, x_t)", m_hat, "rows, Z")
    add("rnn_readout(m) logits", rnn.rnn_readout(m), "rows, V")
    ce = calc_cross_entropy_loss(logits, w["y_fut"][: len(xc)].clamp(min=0)); add("CE per (anchor, j)", ce, "rows, T")
    add("MSE per number (L_dyn)", calc_mse_loss(m_hat, m), "rows, Z")
    near = w["valid"][: len(xc)] & (torch.arange(T) < cfg.future_window); add("near mask (j < Tf)", near, "rows, T")
    df = pd.DataFrame(rows, columns=["tensor", "shape", "axes"])
    print(f"{cfg.task}: B={B}, T={T} -> R={B*T} windows (rows shown: first {len(xc)}); RMS of each memory token: "
          f"min {float(rms.min()):.4f}, max {float(rms.max()):.4f}")
    return df

display(trace(replace(STACK, seq_len=8, max_stack_size=4), B=2))
display(trace(STACK, B=32))
''')

# =====================================================================================================================
md("s6-losses", r"""
## §6. Функции потерь

**SMT [история]:**
$$L = c_{dec}\big(L^{T_f}_{dec} + \alpha L_{far}\big) + c_{dyn}L_{dyn} + c_{unif}L_{unif},\quad c_{dec}=1,\ c_{dyn}=0{,}1,\ c_{unif}=0{,}001$$

* $L^{T_f}_{dec} = \dfrac{\sum_{t}\sum_{j<T_f}\mathrm{valid}_{t,j}\,\mathrm{CE}(y_{t+j}\mid m_t, x_{t+1..t+j})}{\#\{(t,j):\ \mathrm{valid},\ j<T_f\}}$
  — среднее по всем валидным парам батча.
* $L_{far} = \dfrac{\sum_t N_t\cdot\mathrm{mean}_{k\in S_t}\mathrm{CE}_{t,k}}{\sum_t N_t}$, где пул
  $P_t=\{j\ge T_f,\ \mathrm{valid}\}$, $N_t=|P_t|$, $|S_t|=\min(K_{far},N_t)$.
  При случайном $S_t$ матожидание равно среднему CE по дальним парам.
* $L_{dyn} = \mathrm{mean}\big(f(m_{t-1},x_t) - m_t\big)^2$ по $B\cdot T\cdot Z$ числам, $m_{-1}=0$.
* $L_{unif} = \log\mathrm{mean}_{i\ne j} e^{-2\lVert\tilde z_i-\tilde z_j\rVert^2}$ по 512 случайным нормированным
  memory-токенам.

**Как выбираются дальние цели (`far`):**

| `far` | правило | статус |
|---|---|---|
| `uniform_far` | все цели пула | [история] |
| `random_far` | случайные K | [история] |
| `surprise_far` | K с наибольшим CE оконного детектора | [история] |
| `ctx_far` | K с наибольшим CE детектора по входам декодера | [история] |
| `oracle_far` | пул ограничен разметкой генератора (`source ≤ t`) — **диагностика**: использует метаданные, которых нет у метода | [история] |

**Селекторы используют правильную метку.** Скор = CE детектора на **правильном** ответе. Это допустимо: отбор решает
только, какие цели из уже известных обучающих меток взвесить сильнее (метки и так входят в L_dec). На инференсе ни
детектор, ни отбор не выполняются (§9).

**DMT [история]:** $L = 10\cdot\mathrm{mean}(m^{rnn}_t - m_t)^2 + \mathrm{CE}(y_t\mid\mathrm{readout}(\mathrm{sg}[m^{rnn}_t]))$.

**BPTT [история]:** $L=\mathrm{CE}(y_t\mid\mathrm{readout}(m^{rnn}_t))$ — full BPTT или BPTT-1.

**LM [история, MQAR]:** $L=\mathrm{CE}(y_t\mid\mathrm{logits}_t)$.

Все CE усредняются только по позициям или парам, где есть ответ.
""")
code("s6-losses", r'''
def topk_mask(scores, pool, K, generator):                        # [R,T] -> [R,T] bool, ties broken by seeded order
    s = torch.where(pool, scores.float(), torch.full_like(scores, -torch.inf, dtype=torch.float32))
    perm = torch.rand(s.shape, generator=generator, device="cpu").to(s.device).argsort(1)
    order = torch.sort(s.gather(1, perm), dim=1, descending=True, stable=True).indices
    return torch.zeros_like(pool).scatter(1, perm.gather(1, order)[:, :K], True) & pool

def far_scores(cfg, w, batch, dets):                               # [R,T] surprise of each (anchor, j) target
    if cfg.far == "surprise_far":                                  # window detector score of target position t+j
        local = dets["window"].scores(batch)                       # [B,T]
        B, T = batch["x"].shape
        seq = torch.arange(B).repeat_interleave(T)[:, None]        # [R,1]
        tgt = (torch.arange(T)[None, :] + torch.arange(T).repeat(B)[:, None]).clamp(max=T - 1)   # [R,T] = t+j
        return local[seq, tgt]
    if cfg.far == "ctx_far":                                       # CE of y_{t+j} given x_{t+1..t+j}, no memory
        return torch.cat([F.cross_entropy(dets["context"](xf).flatten(0, 1), yf.clamp(min=0).flatten(),
                                          reduction="none").reshape(yf.shape) * yf.ne(IGNORE)
                          for xf, yf in zip(w["x_fut"].split(4096), w["y_fut"].split(4096))])
    return None

def smt_loss(teacher, rnn, batch, cfg, generator=None, dets=None):
    B, T = batch["x"].shape
    w = windows({k: v.to(DEVICE) for k, v in batch.items()})       # each [R,T]
    memory = encode_all(teacher, w["x_ctx"]).reshape(B, T, -1)      # m_t: [R,T] -> [R,Z] -> [B,T,Z]
    logits = teacher.decode(memory.reshape(B * T, -1), w["x_fut"])  # [R,T,V]  (ALL T outputs are computed)
    ce = calc_cross_entropy_loss(logits, w["y_fut"].clamp(min=0))   # [R,T]; invalid pairs masked below
    j = torch.arange(T, device=DEVICE)
    near, far = w["valid"] & (j < cfg.future_window), w["valid"] & (j >= cfg.future_window)    # [R,T] each
    dec = (ce * near).sum() / near.sum().clamp(min=1)               # L_dec^Tf
    if cfg.far == "none":
        aux = dec * 0
    else:
        pool = far & w["prefix"] if cfg.far == "oracle_far" else far
        if cfg.far == "uniform_far":
            chosen = pool
        else:
            with torch.no_grad():
                scores = far_scores(cfg, w, batch, dets)
            scores = torch.zeros_like(ce) if scores is None else scores
            chosen = topk_mask(scores, pool, cfg.far_K, generator)  # [R,T] bool, no gradient through the choice
        n, k = pool.sum(1).to(ce.dtype), chosen.sum(1).to(ce.dtype) # [R], [R]
        row = (ce * chosen).sum(1) / k.clamp(min=1)                 # [R]: mean CE of the chosen far targets
        aux = (n * row).sum() / n.sum().clamp(min=1)                # L_far (same operation order as the archive)
    prev = torch.cat([torch.zeros_like(memory[:, :1]), memory[:, :-1]], 1)   # m_{t-1}: [B,T,Z], m_{-1} = 0
    if cfg.dyn_detach == "both": prev = prev.detach()               # authors' detach_rnn: input detached too
    pred = rnn.rnn_update(prev.reshape(B * T, -1), batch["x"].to(DEVICE).reshape(B * T, 1))   # [R,Z]
    target = memory.reshape(B * T, -1)                              # [R,Z]
    if cfg.dyn_detach in ("target", "both"): target = target.detach()   # 'target' = OLD ablation "detached"
    dyn = calc_mse_loss(pred, target).mean()                        # scalar over R*Z numbers
    unif = calc_uniformity_loss(memory.reshape(B * T * cfg.mem_tokens, cfg.d_model))   # [R*K, d] -> scalar
    loss = cfg.coef_dec * (dec + cfg.alpha_far * aux) + cfg.coef_dyn * dyn + cfg.coef_unif * unif
    return loss, dict(loss=loss, decoder=dec, auxiliary=aux, dynamics=dyn, uniformity=unif)

def teacher_full_access_loss(teacher, batch, cfg):
    """[история, диагностика] L_dec over all valid pairs (Tf = T), decoder reads ALL encoder outputs."""
    w = windows({k: v.to(DEVICE) for k, v in batch.items()})
    h, mask = encode_full(teacher, w["x_ctx"]); logits = decode_full(teacher, h, mask, w["x_fut"])   # [R,T,V]
    ce = F.cross_entropy(logits.flatten(0, 1).float(), w["y_fut"].clamp(min=0).flatten(), reduction="none").reshape(w["y_fut"].shape)
    loss = (ce * w["valid"]).sum() / w["valid"].sum().clamp(min=1)
    return loss, dict(loss=loss, decoder=loss)

def dmt_loss(teacher, rnn, batch, cfg):
    B, T = batch["x"].shape
    with torch.no_grad():
        target = encode_all(teacher, windows(batch)["x_ctx"].to(DEVICE)).reshape(B, T, -1)   # [B,T,Z], frozen
    x = batch["x"].to(DEVICE)
    states = rnn_rollout(rnn, x, detach=True)                       # [B,T,Z]: own states, gradient cut between steps
    dyn = (states - target).square().mean()
    logits = rnn.rnn_readout(states.detach().flatten(0, 1)).reshape(B, T, -1)   # sg[states] -> readout only
    ce = calc_cross_entropy_loss(logits, batch["y"].to(DEVICE).clamp(min=0))    # [B,T]
    mask = batch["target_mask"].to(DEVICE)
    readout = (ce * mask).sum() / mask.sum().clamp(min=1)
    loss = cfg.dmt_coef_dyn * dyn + readout
    return loss, dict(loss=loss, dynamics=dyn, readout=readout)

def bptt_loss(rnn, batch, cfg, truncate):
    """truncate=False: full BPTT (= authors' forward_bptt). truncate=True: BPTT-1 - same forward values, state NOT
    reset, incoming state of every update detached."""
    x = batch["x"].to(DEVICE)
    states = rnn_rollout(rnn, x, detach=truncate)                   # [B,T,Z]
    logits = rnn.rnn_readout(states.flatten(0, 1)).reshape(*x.shape, -1)   # [B,T,V]
    ce = calc_cross_entropy_loss(logits, batch["y"].to(DEVICE).clamp(min=0))
    mask = batch["target_mask"].to(DEVICE)
    loss = (ce * mask).sum() / mask.sum().clamp(min=1)
    return loss, dict(loss=loss, readout=loss)

def lm_loss(model, batch):
    logits = model(batch["x"].to(DEVICE))                           # [B,T,V]
    ce = F.cross_entropy(logits.flatten(0, 1).float(), batch["y"].clamp(min=0).flatten().to(DEVICE),
                         reduction="none").reshape(batch["y"].shape)
    mask = batch["target_mask"].to(DEVICE)
    loss = (ce * mask).sum() / mask.sum().clamp(min=1)
    return loss, dict(loss=loss)
''')
code("s6-smoke", r'''
# one forward/backward of every loss (SMOKE): finite scalars, shapes of the selected far targets
small = replace(STACK, seq_len=16, max_stack_size=4)
b = sample(small, 4, 7)
out = []
for name, cfg_ in [("SMT Tf=8", small), ("SMT ctx_far", replace(small, far="ctx_far")),
                   ("SMT surprise_far", replace(small, far="surprise_far")), ("SMT random_far", replace(small, far="random_far")),
                   ("SMT oracle_far (diagnostic)", replace(small, far="oracle_far")), ("SMT uniform_far", replace(small, far="uniform_far"))]:
    t_, r_ = make_models(cfg_, 1)
    with seeded(5): loss, parts = smt_loss(t_, r_, b, cfg_, torch.Generator().manual_seed(3), dets)
    loss.backward(); out.append((name, float(loss), float(parts["auxiliary"])))
t_, r_ = make_models(small, 1); t_.requires_grad_(False)
out.append(("DMT", float(dmt_loss(t_, r_, b, small)[0]), None))
for tr in (False, True):
    _, r_ = make_models(small, 1); l, _ = bptt_loss(r_, b, small, tr); l.backward(); out.append((f"BPTT truncate={tr}", float(l), None))
mq = sample(MQAR_T16, 4, 7); t_, r_ = make_models(MQAR_T16, 1)
l, _ = teacher_full_access_loss(t_, mq, MQAR_T16); l.backward(); out.append(("MQAR teacher full access", float(l), None))
l, _ = lm_loss(make_lm(MQAR_T16, 0), mq); l.backward(); out.append(("MQAR causal Transformer", float(l), None))
display(pd.DataFrame(out, columns=["loss", "value", "L_far"]))
''')

# =====================================================================================================================
md("s7-subtle", r"""
## §7. Тонкие места, проверенные кодом

Каждая ячейка этой секции содержит `assert`. Если утверждение неверно, ноутбук останавливается.
""")
code("s7-encoder-prefix", r'''
# 7.1 Teacher encoder is BIDIRECTIONAL, but its input is only the prefix x_0..x_t: changing x after t cannot change m_t;
#     changing an EARLIER input does change it (inside the prefix every token sees every other token).
cfg_ = replace(STACK, seq_len=16, max_stack_size=4); bt = sample(cfg_, 4, 11); t_, r_ = make_models(cfg_, 0)
t0 = 9; x2 = bt["x"].clone(); x2[:, t0 + 1:] = torch.randint(0, 5, x2[:, t0 + 1:].shape)
with torch.no_grad():
    rows = torch.arange(4) * 16 + t0
    m1 = t_.encode(windows(bt)["x_ctx"][rows]); m2 = t_.encode(windows(dict(bt, x=x2))["x_ctx"][rows])
    x3 = bt["x"].clone(); x3[:, 0] = (x3[:, 0] % 4) + 1
    m3 = t_.encode(windows(dict(bt, x=x3))["x_ctx"][rows])
print("max |m_t change| after editing x_{>t}:", float((m1 - m2).abs().max()), "| after editing x_0:", float((m1 - m3).abs().max()))
assert torch.allclose(m1, m2, atol=1e-6) and not torch.allclose(m1, m3)
''')
code("s7-j0", r'''
# 7.2 Decoder is CAUSAL; output j = 0 is exactly teacher_readout(m_t); output j does not change if x_fut[j:] changes.
w_ = windows(bt)
with torch.no_grad():
    m = t_.encode(w_["x_ctx"]); full = t_.decode(m, w_["x_fut"])
    print("max |decode[:,0] - teacher_readout|:", float((full[:, 0] - t_.teacher_readout(m)).abs().max()))
    assert torch.allclose(full[:, 0], t_.teacher_readout(m), atol=1e-5)
    xf2 = w_["x_fut"].clone(); xf2[:, 4:] = torch.randint(0, 5, xf2[:, 4:].shape)
    assert torch.allclose(full[:, :5], t_.decode(m, xf2)[:, :5], atol=1e-5), "outputs j <= 4 must not see x_fut[4:]"
print("causal: outputs j<=4 unchanged after editing x_fut[4:]")
''')
code("s7-tf-compute", r'''
# 7.3 Tf = 8 restricts which (anchor, j) pairs enter L_dec, NOT the computation: decode still runs all T outputs.
w_ = windows(sample(STACK, 2, 3)); t_, r_ = make_models(STACK, 0)
with torch.no_grad(): lg = t_.decode(t_.encode(w_["x_ctx"][:8]), w_["x_fut"][:8])
print("decoder logits per window:", tuple(lg.shape), f"-> T = {STACK.seq_len} outputs computed, {STACK.future_window} used by L_dec")
assert lg.shape[1] == STACK.seq_len
''')
code("s7-ldyn", r'''
# 7.4 L_dyn gradient into the encoder: joint / OLD 'detached' (= target_detached) / authors' detach_rnn (both detached)
def enc_grad(dyn_detach):
    c = replace(small, coef_dec=0.0, coef_unif=0.0, coef_dyn=1.0, dyn_detach=dyn_detach)
    t_, r_ = make_models(c, 3)
    with seeded(1): smt_loss(t_, r_, sample(c, 4, 77), c)[0].backward()
    g = lambda m, pref: math.sqrt(float(sum(float((p.grad ** 2).sum()) for n, p in m.named_parameters()
                                            if n.startswith(pref) and p.grad is not None)))
    return dict(dyn_detach=dyn_detach, encoder=g(t_, ("embed", "encoder")), updater=g(r_, "rnn."), decoder=g(t_, "decoder"))
gd = pd.DataFrame([enc_grad(m) for m in ("none", "target", "both")]); display(gd)
assert gd.encoder[0] > 0 and gd.encoder[1] > 0 and gd.encoder[2] == 0 and gd.decoder.eq(0).all()
assert np.allclose(gd.updater, gd.updater[0])
print("OLD 'detached' detaches only the target m_t: the encoder still gets L_dyn gradient through the updater input m_{t-1}."
      " Correct name: target_detached. Fully detached ('both') = authors' detach_rnn=True: zero gradient to the encoder.")
''')
code("s7-dmt", r'''
# 7.5 DMT: the readout CE sees detached states -> trains only the readout; the MSE trains the updater, one step deep.
def dmt_grads(coef_dyn):
    t_, r_ = make_models(small, 3); t_.requires_grad_(False)
    dmt_loss(t_, r_, sample(small, 4, 77), replace(small, dmt_coef_dyn=coef_dyn))[0].backward()
    return {k: float(sum(p.grad.abs().sum() for n, p in r_.named_parameters() if n.startswith(k) and p.grad is not None))
            for k in ("embed", "rnn.", "decoder", "to_logits")}
g_ce, g_all = dmt_grads(0.0), dmt_grads(10.0)
print("readout CE only:", g_ce); print("full DMT loss  :", g_all)
assert g_ce["rnn."] == 0 and g_ce["embed"] == 0 and g_ce["decoder"] > 0 and g_all["rnn."] > 0
''')
code("s7-bptt", r'''
# 7.6 Full BPTT vs BPTT-1: identical forward values with identical weights, different backward graphs; BPTT-1 does not
#     reset the state. The loss at the last step reaches m_{-1} only in full BPTT.
_, r_ = make_models(small, 3); xb = sample(small, 4, 77)["x"]
with torch.no_grad():
    s_full, s_one = rnn_rollout(r_, xb), rnn_rollout(r_, xb, detach=True)
    s_reset = torch.stack([r_.rnn_update(torch.zeros(4, small.Z), xb[:, t:t+1]) for t in range(xb.shape[1])], 1)
print("forward full vs BPTT-1:", float((s_full - s_one).abs().max()), "| vs reset-to-zero each step:", float((s_full - s_reset).abs().max()))
assert torch.equal(s_full, s_one) and not torch.allclose(s_full, s_reset)
for detach in (False, True):
    init = torch.zeros(4, small.Z, requires_grad=True)
    F.cross_entropy(r_.rnn_readout(rnn_rollout(r_, xb, init=init, detach=detach)[:, -1]), torch.ones(4, dtype=torch.long)).backward()
    g0 = 0.0 if init.grad is None else float(init.grad.abs().sum())
    print(f"{'BPTT-1   ' if detach else 'full BPTT'}: |d loss(t=T-1) / d m_(-1)| = {g0:.3e}")
    assert (g0 == 0) == detach
''')
code("s7-labels", r'''
# 7.7 The models receive only x. Annotations (y, source_pos, delay) never enter a forward pass: feeding garbage there
#     leaves every model output unchanged (they only enter losses / metrics).
bt2 = dict(bt, y=torch.full_like(bt["y"], 3), source_pos=torch.zeros_like(bt["x"]), delay=torch.zeros_like(bt["x"]))
t_, r_ = make_models(cfg_, 0)
with torch.no_grad():
    assert torch.equal(encode_all(t_, windows(bt)["x_ctx"]), encode_all(t_, windows(bt2)["x_ctx"]))
    assert torch.equal(rnn_rollout(r_, bt["x"]), rnn_rollout(r_, bt2["x"]))
print("outputs unchanged when y / source_pos / delay are replaced")
''')

# =====================================================================================================================
md("s8-train", r"""
## §8. Обучение: шаг оптимизатора, стадии, выбор чекпойнта

* **Один шаг:** `zero_grad` → forward → loss → `backward` → обрезка нормы градиента до 1,0 → `AdamW.step`
  (β 0,9 / 0,95, wd 0,01) → `scheduler.step`.
* **Расписание LR:** линейный разогрев на `min(200, steps/10)` шагов, затем константа (v4 и абляции). В v1 / v2 после
  разогрева шёл косинус до 0,1×.
* **Оценка** — каждые `eval_every` шагов на validation. Сохраняются:
  * **выбранный чекпойнт** — с наименьшим CE на длинных ответах (для SMT это teacher при `j = 0`);
  * **последние веса**.
* **Стадии:**
  * SMT: teacher и RNN обучаются вместе;
  * перед DMT: `copy_readout`, teacher замораживается;
  * DMT: обучается только RNN; начало — с выбранного SMT-чекпойнта.
""")
code("s8-train", r'''
def make_optimizer(params, lr, cfg, steps):
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    warm = max(1, min(cfg.warmup_steps, steps // 10))
    if cfg.schedule == "warmup_constant":
        lam = lambda s: (s + 1) / warm if s < warm else 1.0
    else:                                                           # cosine to min_lr_ratio (v1 / v2)
        lam = lambda s: (s + 1) / warm if s < warm else cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * 0.5 * (
            1 + math.cos(math.pi * min(1.0, (s - warm) / max(1, steps - warm))))
    return opt, torch.optim.lr_scheduler.LambdaLR(opt, lam)

def run_stage(name, steps, lr, params, step_fn, monitor_fn, models, out, cfg, log=print, stop_key=None):
    opt, sched = make_optimizer(params, lr, cfg, steps)
    best, best_step, history, tic = float("inf"), None, [], time.perf_counter()
    for step in range(1, steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, parts = step_fn(step - 1)
        assert torch.isfinite(loss), (name, step, parts)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
        opt.step(); sched.step()
        row = dict(step=step, lr=opt.param_groups[0]["lr"], grad_norm=float(norm), **{f"train_{k}": float(v.detach()) for k, v in parts.items()})
        if step % cfg.eval_every == 0 or step == steps:
            score, metrics = monitor_fn(); row.update(metrics, seconds=time.perf_counter() - tic)
            if score < best:
                best, best_step = score, step
                torch.save(dict(step=step, models={k: m.state_dict() for k, m in models.items()}), out / f"{name}_best.pt")
            log(f"{name} {step}/{steps} " + " ".join(f"{k}={v:.3f}" for k, v in metrics.items() if k.endswith("acc")))
            pd.DataFrame(history + [row]).to_csv(out / f"{name}_history.csv", index=False)
            if stop_key and cfg.stop_accuracy > 0 and metrics[stop_key] >= cfg.stop_accuracy: history.append(row); break
        history.append(row)
    last = copy.deepcopy({k: m.state_dict() for k, m in models.items()})
    for k, m in models.items():
        m.load_state_dict(torch.load(out / f"{name}_best.pt", weights_only=True)["models"][k])
    return dict(best_step=best_step, last_step=history[-1]["step"], last_models=last, history=pd.DataFrame(history))
''')

# =====================================================================================================================
md("s9-eval", r"""
## §9. Инференс и метрики

| режим | что исполняется | статус |
|---|---|---|
| `rnn` | RNN с `m_{−1} = 0` читает `x_0..x_t` по одному, затем `readout(m^rnn_t)`. **Это инференс итоговой модели SMT → DMT и BPTT-моделей.** Нет teacher, энкодера, детектора, отбора. | [история] |
| `teacher` | `decode(encode(x_0..x_t))[:, 0]`: энкодер **видел** pop / ключ запроса | [история] |
| `teacher_pre` | память `m_a` после всех пар и до запроса (`a = 2N−1` или `a = t−1`), выход `j = t−a` | **[новое]** |
| `teacher_full` | как `teacher`, но декодер читает все выходы энкодера | [история, диагностика] |
| `lm` | `logits[:, t]` каузальной модели | [история] |

**Метрики:**
* accuracy = `argmax(logits) == y_t` по **всему словарю**;
* для MQAR отдельно — accuracy **среди N записанных значений** (`cand`, угадывание 1/N). Это диагностика: она
  использует список значений, записанных в последовательности, которого у модели как ограничения нет;
* «длинные» ответы — `delay ≥ long_delay`.
""")
code("s9-eval", r'''
@torch.no_grad()
def answer_logits(kind, model, batch, cfg, anchor="query"):
    rows = []
    for lo in range(0, len(batch["x"]), cfg.eval_batch_size):
        part = {k: v[lo:lo + cfg.eval_batch_size] for k, v in batch.items()}
        x, y = part["x"], part["y"]; B, T = x.shape
        b, t = torch.where(part["target_mask"])                     # A answer positions
        if kind == "rnn":
            lg = model.rnn_readout(rnn_rollout(model, x.to(DEVICE))[b, t])           # [B,T,Z] -> [A,Z] -> [A,V]
        elif kind == "lm":
            lg = model(x.to(DEVICE))[b, t]                                           # [B,T,V] -> [A,V]
        else:
            w = windows(part)
            if kind == "teacher_pre":
                a = torch.full_like(t, 2 * cfg.num_kv_pairs - 1) if anchor == "after_pairs" else t - 1
                assert (a < t).all() and (a >= 2 * cfg.num_kv_pairs - 1).all()
                rows_, j = b * T + a, t - a
            else:
                rows_, j = b * T + t, torch.zeros_like(t)
            xc, xf = w["x_ctx"][rows_].to(DEVICE), w["x_fut"][rows_].to(DEVICE)       # [A,T] each
            full = decode_full(model, *encode_full(model, xc), xf) if kind == "teacher_full" else model.decode(model.encode(xc), xf)
            lg = full[torch.arange(len(j)), j]                                       # gather output j: [A,T,V] -> [A,V]
            assert torch.equal(w["y_fut"][rows_, j], y[b, t]), "output j of anchor a is aligned with y_t"
        lg, target = lg.float().cpu(), y[b, t]
        r = dict(sequence=(b + lo).numpy(), t=t.numpy(), delay=part["delay"][b, t].numpy(), target=target.numpy(),
                 pred=lg.argmax(-1).numpy(), correct=(lg.argmax(-1) == target).numpy(),
                 ce=F.cross_entropy(lg, target, reduction="none").numpy())
        if cfg.task == "mqar":                                      # diagnostic: choice among the N written values
            written = x[:, 1:2 * cfg.num_kv_pairs:2][b]             # [A,N]
            r["cand_correct"] = (lg.gather(1, written).argmax(1) == (written == target[:, None]).float().argmax(1)).numpy()
        rows.append(pd.DataFrame(r))
    return pd.concat(rows, ignore_index=True)

def long_metrics(ev, cfg, prefix):
    lg = ev[ev.delay >= cfg.long_delay]
    out = {f"{prefix}_long_acc": float(lg.correct.mean()), f"{prefix}_long_ce": float(lg.ce.mean()), f"{prefix}_all_acc": float(ev.correct.mean())}
    if "cand_correct" in ev: out[f"{prefix}_long_cand_acc"] = float(lg.cand_correct.mean())
    return out

t_, r_ = make_models(MQAR_T16, 0); vq = validation_set(MQAR_T16)
for kind, model, anchor in (("teacher", t_, "query"), ("teacher_pre", t_, "after_pairs"), ("teacher_pre", t_, "before_query"),
                            ("teacher_full", t_, "query"), ("rnn", r_, "query")):
    print(kind, anchor, long_metrics(answer_logits(kind, model, vq, MQAR_T16, anchor), MQAR_T16, "untrained"))
''')

# =====================================================================================================================
md("s10-fit", r"""
## §10. Полные процедуры: SMT → DMT, BPTT, LM, teacher с полным доступом
""")
code("s10-fit", r'''
def fit(spec, out_root=ROOT / "results" / "notebook_runs", log=print):
    """spec = dict(name, family in {smt, teacher_full, bptt_full, bptt_1, transformer}, seed, config: Config).
    Writes config.json, <stage>_history.csv, <stage>_best.pt and summary.json (validation-selected AND last weights,
    evaluated on the independent stream 901) into a NEW folder <name>_<timestamp>."""
    cfg, seed, family = spec["config"], int(spec["seed"]), spec["family"]
    out = out_root / f"{spec['name']}_s{seed}_{time.strftime('%Y%m%d-%H%M%S')}_{os.getpid()}"; out.mkdir(parents=True)
    (out / "config.json").write_text(json.dumps(dict(family=family, seed=seed, config=asdict(cfg), torch=torch.__version__,
                                                    device=str(DEVICE), notebook="SMT_MemUP_FROM_ZERO.ipynb"), indent=2))
    val, ev_set = validation_set(cfg), evaluation_set(cfg)
    teacher, rnn = make_models(cfg, seed)
    d = load_detectors(cfg.V) if cfg.far in ("ctx_far", "surprise_far") else None
    summary = dict(name=spec["name"], family=family, seed=seed)
    def both(prefix, r, models, ev_fn):                             # selected AND last weights on stream 901
        keep = {k: copy.deepcopy(m.state_dict()) for k, m in models.items()}
        summary.update({f"{prefix}_best_step": r["best_step"], f"{prefix}_last_step": r["last_step"], **ev_fn("best")})
        for k, m in models.items(): m.load_state_dict(r["last_models"][k])
        summary.update(ev_fn("last"))
        for k, m in models.items(): m.load_state_dict(keep[k])
    rnn_val = lambda: (lambda ev: (float(ev[ev.delay >= cfg.long_delay].ce.mean()), long_metrics(ev, cfg, "val_rnn")))(answer_logits("rnn", rnn, val, cfg))
    if family in ("smt", "teacher_full"):
        kind = "teacher_full" if family == "teacher_full" else "teacher"
        params = [*teacher.parameters(), *rnn.parameters()] if family == "smt" else list(teacher.parameters())
        def step_fn(s):
            teacher.train(); rnn.train(); batch = sample(cfg, cfg.batch_size, derive_seed(301, seed, s))
            gen = torch.Generator().manual_seed(derive_seed(303, seed, s))       # top-K tie breaking
            with seeded(derive_seed(304, seed, s)):                              # L_unif random subset
                return teacher_full_access_loss(teacher, batch, cfg) if family == "teacher_full" else smt_loss(teacher, rnn, batch, cfg, gen, d)
        def monitor():
            copy_readout(teacher, rnn); ev = answer_logits(kind, teacher, val, cfg); m = long_metrics(ev, cfg, "val_teacher")
            if family == "smt":
                m.update(long_metrics(answer_logits("rnn", rnn, val, cfg), cfg, "val_rnn"))
                if cfg.task == "mqar": m.update(long_metrics(answer_logits("teacher_pre", teacher, val, cfg, "after_pairs"), cfg, "val_teacher_pre"))
            return float(ev[ev.delay >= cfg.long_delay].ce.mean()), m
        def ev_smt(tag):
            copy_readout(teacher, rnn); o = {}
            modes = [(kind, teacher, "query")] + ([("teacher_pre", teacher, "after_pairs"), ("teacher_pre", teacher, "before_query")]
                                                  if (cfg.task == "mqar" and family == "smt") else []) + ([("rnn", rnn, "query")] if family == "smt" else [])
            for k_, mdl, anc in modes:
                o.update(long_metrics(answer_logits(k_, mdl, ev_set, cfg, anc), cfg, f"eval_{tag}_{k_}{'' if anc == 'query' else '_' + anc}_smt"))
            return o
        r = run_stage("smt", cfg.smt_steps, cfg.smt_lr, params, step_fn, monitor, dict(teacher=teacher, rnn=rnn), out, cfg, log, "val_teacher_long_acc")
        both("smt", r, dict(teacher=teacher, rnn=rnn), ev_smt); copy_readout(teacher, rnn)
        if family == "smt" and cfg.dmt_steps > 0:
            teacher.requires_grad_(False); teacher.eval(); frozen = copy.deepcopy(teacher.state_dict())
            def dmt_step(s):
                rnn.train(); batch = sample(cfg, cfg.batch_size, derive_seed(302, seed, s))
                with seeded(derive_seed(305, seed, s)): return dmt_loss(teacher, rnn, batch, cfg)
            r = run_stage("dmt", cfg.dmt_steps, cfg.dmt_lr, list(rnn.parameters()), dmt_step, rnn_val, dict(rnn=rnn), out, cfg, log, "val_rnn_long_acc")
            assert all(torch.equal(v, teacher.state_dict()[k]) for k, v in frozen.items()), "teacher changed in DMT"
            both("dmt", r, dict(rnn=rnn), lambda tag: long_metrics(answer_logits("rnn", rnn, ev_set, cfg), cfg, f"eval_{tag}_rnn_dmt"))
    elif family in ("bptt_full", "bptt_1"):
        def step_fn(s):
            rnn.train(); batch = sample(cfg, cfg.batch_size, derive_seed(301, seed, s))
            with seeded(derive_seed(306, seed, s)): return bptt_loss(rnn, batch, cfg, family == "bptt_1")
        r = run_stage("bptt", cfg.bptt_steps, cfg.bptt_lr, list(rnn.parameters()), step_fn, rnn_val, dict(rnn=rnn), out, cfg, log, "val_rnn_long_acc")
        both("bptt", r, dict(rnn=rnn), lambda tag: long_metrics(answer_logits("rnn", rnn, ev_set, cfg), cfg, f"eval_{tag}_rnn_bptt"))
    elif family == "transformer":
        model = make_lm(cfg, seed)
        def step_fn(s): model.train(); return lm_loss(model, sample(cfg, cfg.batch_size, derive_seed(301, seed, s)))
        def monitor():
            model.eval(); ev = answer_logits("lm", model, val, cfg); return float(ev[ev.delay >= cfg.long_delay].ce.mean()), long_metrics(ev, cfg, "val_lm")
        r = run_stage("lm", cfg.lm_steps, cfg.lm_lr, list(model.parameters()), step_fn, monitor, dict(model=model), out, cfg, log, "val_lm_long_acc")
        both("lm", r, dict(model=model), lambda tag: (model.eval(), long_metrics(answer_logits("lm", model, ev_set, cfg), cfg, f"eval_{tag}_lm"))[1])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return out, summary

# SMOKE: a 6-step SMT -> 4-step DMT fit on a tiny Stack to exercise stages, selection and saving (numbers meaningless)
tiny_cfg = replace(STACK, seq_len=16, max_stack_size=4, d_model=32, n_heads=2, smt_steps=6, dmt_steps=4, eval_every=2,
                   validation_examples=16, eval_examples=16, batch_size=8, warmup_steps=2, long_delay=4)
smoke_dir, smoke_summary = fit(dict(name="smoke_stack_smt", family="smt", seed=1, config=tiny_cfg), out_root=ROOT / "results" / "smoke", log=lambda s: None)
print(smoke_dir.relative_to(ROOT), {k: v for k, v in smoke_summary.items() if "step" in k})
''')

# =====================================================================================================================
md("s11-trace-examples", r"""
## §11. Два разбора от ID входа до ответа и loss

Используются крошечные модели, обученные в §10 (smoke). Это **не результаты**: проверяется только путь данных.
* **Stack:** последний pop последовательности 0 независимого потока.
* **MQAR:** последний запрос, T = 16, N = 2, необученная модель.

Для каждого ответа показаны:
* строка окна;
* выход `j = 0`;
* CE этой пары в `L_dec`;
* та же пара как выход `j > 0` более ранних якорей;
* строка оценки.
""")
code("s11-trace-examples", r'''
@torch.no_grad()
def follow_answer(cfg, teacher, rnn, batch):
    T = cfg.seq_len; tq = int(batch["target_mask"][0].nonzero().max()); w = windows(batch)
    print(f"answer position t={tq}: input ID x_t={int(batch['x'][0, tq])}, answer y_t={int(batch['y'][0, tq])}, delay={int(batch['delay'][0, tq])}")
    print(f"  encoder input x_ctx[t] = {w['x_ctx'][tq].tolist()}")
    m_t = teacher.encode(w["x_ctx"][tq:tq + 1]); logits = teacher.decode(m_t, w["x_fut"][tq:tq + 1])   # [1,Z], [1,T,V]
    ce0 = float(calc_cross_entropy_loss(logits, w["y_fut"][tq:tq + 1].clamp(min=0))[0, 0])
    print(f"  m_t {tuple(m_t.shape)} -> decode {tuple(logits.shape)}; output j=0 predicts y_fut[t,0]={int(w['y_fut'][tq, 0])}: "
          f"argmax={int(logits[0, 0].argmax())}, CE={ce0:.3f}  -> enters L_dec (j=0 < Tf={cfg.future_window}, valid)")
    for a in (tq - 1, tq - 3):
        j = tq - a
        lg = teacher.decode(teacher.encode(w["x_ctx"][a:a + 1]), w["x_fut"][a:a + 1])[0, j]
        print(f"  same answer as output j={j} of anchor a={a}: decoder inputs {w['x_fut'][a, :j].tolist()}, "
              f"CE={float(F.cross_entropy(lg[None], batch['y'][0, tq][None])):.3f} ({'near' if j < cfg.future_window else 'far'})")
    for kind, mdl in (("teacher", teacher), ("rnn", rnn)):
        r = answer_logits(kind, mdl, batch, cfg); r = r[r.t == tq].iloc[0]
        print(f"  evaluation row [{kind}]: pred={int(r.pred)} target={int(r.target)} correct={bool(r.correct)} CE={r.ce:.3f}")
    if cfg.task == "mqar":
        for anc in ("after_pairs", "before_query"):
            r = answer_logits("teacher_pre", teacher, batch, cfg, anc); r = r[r.t == tq].iloc[0]
            print(f"  evaluation row [teacher_pre, {anc}] [новое]: pred={int(r.pred)} correct={bool(r.correct)} among written={bool(r.cand_correct)}")

t_s, r_s = make_models(tiny_cfg, 1)
t_s.load_state_dict(torch.load(smoke_dir / "smt_best.pt", weights_only=True)["models"]["teacher"])
r_s.load_state_dict(torch.load(smoke_dir / "dmt_best.pt", weights_only=True)["models"]["rnn"])
sb = sample(tiny_cfg, 1, derive_seed(901, 0)); display(stack_table(sb)); follow_answer(tiny_cfg, t_s, r_s, sb)
mq1 = sample(MQAR_T16, 1, derive_seed(901, 0)); display(mqar_table(mq1, MQAR_T16))
t_m, r_m = make_models(MQAR_T16, 1); follow_answer(MQAR_T16, t_m, r_m, mq1)
''')

# =====================================================================================================================
md("s12-equivalence", r"""
## §12. Сверка со старой реализацией и с авторским кодом

**Только в этой секции** импортируются архивные модули (`archive/smt_memup_*.py`, `archive/mqar_baselines.py`,
`archive/checks/mqar_bottleneck_test.py`) и оригиналы авторов (`upstream/`). Они служат эталоном.

**Что сравнивается:** на фиксированных батчах и одинаковых весах.
* Случайность зафиксирована одинаковыми seed:
  * `seeded(5)` вокруг лосса — для подмножества `L_unif`;
  * `Generator(3)` — для разрешения ничьих при отборе.
* В таблице для каждой проверки — максимальное расхождение и допуск.
""")
code("s12-equivalence", r'''
sys.path[:0] = [str(ARCHIVE), str(ARCHIVE / "checks")]
for repo in ("smt", "zoology"):
    head = subprocess.run(["git", "-C", str(UPSTREAM / repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert head == PINNED[repo], f"upstream/{repo} at {head}"
import smt_memup_stack as S, smt_memup_window as W, smt_memup_v4 as V, smt_memup_mqar as M, mqar_baselines as BL, mqar_bottleneck_test as BN
S.sources(ARCHIVE); AM = S.sources().model                     # authors' model.py (original)
CPU_ = torch.device("cpu"); assert DEVICE.type == "cpu", "run §12 on CPU"
REPORT = []
def diff(a, b):
    if isinstance(a, dict): return max(diff(a[k], b[k]) for k in b)
    a, b = torch.as_tensor(a).double(), torch.as_tensor(b).double()
    return float((a - b).abs().max()) if a.numel() else 0.0
def check(what, old_code, d, tol):
    REPORT.append(dict(check=what, old_code=old_code, max_abs_diff=d, tolerance=tol, ok=d <= tol)); assert d <= tol, (what, d)
def grads(*ms): return {f"{i}.{n}": (p.grad.clone() if p.grad is not None else torch.zeros_like(p)) for i, m in enumerate(ms) for n, p in m.named_parameters()}

SV, SC = V.V4Config(), STACK
MM = M.candidate("pilot01", seq_len=16, num_kv_pairs=2, K=1, long_delay=4, local_window=4)
# data, annotation, windows, masks
a, b_ = sample_stack(SC, 8, 12345), S.sample_stack(SV, 8, 12345)
check("Stack batch x, y, mask, source, delay", "S.sample_stack", diff({k: a[k] for k in ("x","y","source_pos","delay")}, {k: b_[k] for k in ("x","y","source_pos","delay")}), 0)
wa, wb = windows(a), S.windows(b_)
check("windows x_ctx, x_fut, y_fut, valid, prefix", "S.windows", diff({k: wa[k] for k in ("x_ctx","x_fut","y_fut","valid","prefix")}, {k: wb[k] for k in ("x_ctx","x_fut","y_fut","valid","prefix")}), 0)
check("validation stream 501", "S.sample_stack(501)", diff(validation_set(SC)["x"], S.sample_stack(SV, 512, S.derive_seed(501, 0))["x"]), 0)
a, b_ = sample_mqar(MQAR_T16, 16, 777), M.sample_mqar(MM, 16, 777)
check("MQAR batch (copied Zoology generator)", "M.sample_mqar", diff({k: a[k] for k in ("x","y","source_pos","delay")}, {k: b_[k] for k in ("x","y","source_pos","delay")}), 0)
ap, bp = sample_mqar(MQAR_PILOT01, 64, 31), M.sample_mqar(M.candidate("pilot01"), 64, 31)
check("MQAR pilot01 batch", "M.sample_mqar", diff(ap["x"], bp["x"]) + diff(ap["y"], bp["y"]), 0)
# weights and model functions vs the authors' original classes
t1, r1 = make_models(SC, 107, CPU_); t2, r2 = S.make_models(SV, 107, CPU_)
check("initial weights (teacher + RNN)", "S.make_models", diff({k: v for k, v in t1.state_dict().items()}, t2.state_dict()) + diff(r1.state_dict(), r2.state_dict()), 0)
assert type(t2).__module__ == "smt_author_model"
xs = windows(sample_stack(SC, 2, 9))
with torch.no_grad():
    m1, m2 = t1.encode(xs["x_ctx"]), t2.encode(xs["x_ctx"]); check("Teacher.encode", "authors' Teacher.encode", diff(m1, m2), 0)
    check("Teacher.decode", "authors' Teacher.decode", diff(t1.decode(m1, xs["x_fut"]), t2.decode(m2, xs["x_fut"])), 0)
    check("Teacher.teacher_readout", "authors'", diff(t1.teacher_readout(m1), t2.teacher_readout(m2)), 0)
    check("RNN.rnn_update", "authors' RNN.rnn_update", diff(r1.rnn_update(m1, xs["x_ctx"][:, -1:].clamp(min=0)), r2.rnn_update(m2, xs["x_ctx"][:, -1:].clamp(min=0))), 0)
    check("RNN.rnn_readout", "authors' RNN.rnn_readout", diff(r1.rnn_readout(m1), r2.rnn_readout(m2)), 0)
    zz = torch.randn(300, 64)
    with seeded(3): u1 = calc_uniformity_loss(zz)
    with seeded(3): u2 = AM.calc_uniformity_loss(zz)
    check("calc_uniformity_loss", "authors'", diff(u1, u2), 0)
# losses: every component + gradients
small_c, small_v = replace(SC, seq_len=24, max_stack_size=6), replace(SV, seq_len=24, max_stack_size=6)
bs = sample_stack(small_c, 4, 99)
det_old = W.load_v1_detectors(SV, ARCHIVE, CPU_)
for far, variant in (("none", "base_window"), ("ctx_far", "ctx_far"), ("surprise_far", "surprise_far"),
                     ("random_far", "random_far"), ("oracle_far", "oracle_far"), ("uniform_far", "uniform_far")):
    t1, r1 = make_models(small_c, 7, CPU_)
    with seeded(5): l1, p1 = smt_loss(t1, r1, bs, replace(small_c, far=far), torch.Generator().manual_seed(3), dets)
    l1.backward(); g1 = grads(t1, r1)
    t2, r2 = make_models(small_c, 7, CPU_)
    with seeded(5): l2, p2 = W.smt_window_loss(t2, r2, bs, small_v, variant, det_old, torch.Generator().manual_seed(3), CPU_)
    l2.backward(); g2 = grads(t2, r2)
    check(f"SMT loss components [{variant}]: L_dec, L_far, L_dyn, L_unif, total", "W.smt_window_loss",
          diff({k: p1[k].detach() for k in ("decoder","auxiliary","dynamics","uniformity","loss")}, {k: p2[k].detach() for k in ("decoder","auxiliary","dynamics","uniformity","loss")}), 0)
    check(f"SMT gradients [{variant}] (all parameters)", "W.smt_window_loss", diff(g1, g2), 0)
# old 'detached' patch vs dyn_detach='target'
real = AM.calc_mse_loss; c1 = replace(small_c, dyn_detach="target")
t1, r1 = make_models(small_c, 7, CPU_)
with seeded(5): smt_loss(t1, r1, bs, c1, torch.Generator().manual_seed(3))[0].backward()
t2, r2 = make_models(small_c, 7, CPU_); AM.calc_mse_loss = lambda p, t: real(p, t.detach())
try:
    with seeded(5): W.smt_window_loss(t2, r2, bs, small_v, "base_window", {}, torch.Generator().manual_seed(3), CPU_)[0].backward()
finally:
    AM.calc_mse_loss = real
check("dyn_detach='target' == OLD ablation 'detached' (gradients)", "checks/smt_memup_ablation.py patch", diff(grads(t1, r1), grads(t2, r2)), 0)
for name, f1, f2 in (("DMT loss + gradients", lambda t, r: dmt_loss(t, r, bs, small_c), lambda t, r: S.dmt_loss(t, r, bs, small_v, CPU_)),
                     ("full BPTT loss + gradients", lambda t, r: bptt_loss(r, bs, small_c, False), lambda t, r: V.bptt_loss(r, bs, small_v, CPU_, False)),
                     ("BPTT-1 loss + gradients", lambda t, r: bptt_loss(r, bs, small_c, True), lambda t, r: V.bptt_loss(r, bs, small_v, CPU_, True))):
    res = []
    for f in (f1, f2):
        t, r = make_models(small_c, 11, CPU_); t.requires_grad_(name.startswith("DMT") is False)
        l, _ = f(t, r); l.backward(); res.append((l.detach(), grads(r)))
    check(name, "S.dmt_loss / V.bptt_loss", diff(res[0][0], res[1][0]) + diff(res[0][1], res[1][1]), 0)
t1, r1 = make_models(small_c, 3, CPU_); bt3 = sample_stack(small_c, 4, 1)
check("authors' forward_bptt == full BPTT", "AM.forward_bptt", diff(bptt_loss(r1, bt3, small_c, False)[0].detach(),
      AM.forward_bptt(r1, bt3, AM.BPTTConfig(T=24))["loss"].detach()), 1e-6)
# MQAR losses, full access, baselines
bq = sample_mqar(MQAR_T16, 4, 31)
t1, r1 = make_models(MQAR_T16, 7, CPU_)
with seeded(5): l1 = smt_loss(t1, r1, bq, MQAR_T16, torch.Generator().manual_seed(3))[0]
l1.backward(); g1 = grads(t1, r1)
t2, r2 = make_models(MQAR_T16, 7, CPU_)
with seeded(5): l2 = M.smt_loss(t2, r2, bq, MM, "baseline", 0.0, {}, torch.Generator().manual_seed(3), CPU_)[0]
l2.backward()
check("MQAR SMT loss (archive trims masked padding)", "M.smt_loss", diff(l1.detach(), l2.detach()), 1e-5)
check("MQAR SMT gradients", "M.smt_loss", diff(g1, grads(t2, r2)), 5e-4)
t1, _ = make_models(MQAR_T16, 7, CPU_); t2, _ = make_models(MQAR_T16, 7, CPU_)
w_ = S.windows(bq); lgo = BN.logits_for(t2, w_, "full", CPU_)
with torch.no_grad(): lgn = decode_full(t1, *encode_full(t1, w_["x_ctx"]), w_["x_fut"])
check("full-access logits", "mqar_bottleneck_test.logits_for('full')", diff(lgn, lgo), 0)
bc = replace(BL.BaselineConfig(), seq_len=16, kv_pairs=2)
check("causal Transformer init", "BL.make_model", diff(make_lm(MQAR_T16, 107, CPU_).state_dict(), BL.make_model(bc, CPU_).state_dict()), 0)
check("causal Transformer loss", "BL.target_loss", diff(lm_loss(make_lm(MQAR_T16, 107, CPU_), bq)[0].detach(), BL.target_loss(BL.make_model(bc, CPU_), bq, CPU_).detach()), 1e-6)
_, r1 = make_models(MQAR_T16, 107, CPU_)
check("MQAR RNN full BPTT loss", "BL.target_loss(rnn_bptt)", diff(bptt_loss(r1, bq, MQAR_T16, False)[0].detach(), BL.target_loss(BL.make_model(replace(bc, model="rnn_bptt"), CPU_), bq, CPU_).detach()), 1e-6)
# one optimizer step (same optimizer, same schedule)
for who in ("new", "old"):
    t, r = make_models(small_c, 21, CPU_); params = [*t.parameters(), *r.parameters()]
    opt = torch.optim.AdamW(params, lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)
    sch = torch.optim.lr_scheduler.LambdaLR(opt, V.lr_lambda(small_v, 100, "warmup_constant") if who == "old" else make_optimizer(params, 3e-4, small_c, 100)[1].lr_lambdas[0])
    with seeded(5):
        l = smt_loss(t, r, bs, small_c, torch.Generator().manual_seed(3))[0] if who == "new" else W.smt_window_loss(t, r, bs, small_v, "base_window", {}, torch.Generator().manual_seed(3), CPU_)[0]
    l.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sch.step()
    if who == "new": after_new = copy.deepcopy(t.state_dict())
check("one AdamW step (weights after the step)", "V._run_stage step", diff(after_new, t.state_dict()), 0)
# evaluation rows
vs = S.sample_stack(SV, 16, S.derive_seed(501, 0)); t1, r1 = make_models(SC, 3, CPU_)
old = S.evaluate(t1, r1, vs, SV, CPU_)
for kind, mdl in (("teacher", t1), ("rnn", r1)):
    new = answer_logits(kind, mdl, vs, SC); o = old[old.model == kind].reset_index(drop=True)
    check(f"evaluation rows Stack [{kind}]: correct / CE", "S.evaluate", diff(new.correct.values, o.correct.values) + diff(new.ce.values, o.ce.values), 1e-5)
vm = M.sample_mqar(MM, 16, S.derive_seed(511, 0)); t1, r1 = make_models(MQAR_T16, 3, CPU_)
old = M.evaluate(t1, r1, vm, MM, CPU_)
for kind, mdl in (("teacher", t1), ("rnn", r1)):
    new = answer_logits(kind, mdl, vm, MQAR_T16); o = old[old.model == kind].reset_index(drop=True)
    check(f"evaluation rows MQAR [{kind}]: cand_correct / CE", "M.evaluate", diff(new.cand_correct.values, o.cand_correct.values) + diff(new.ce.values, o.ce.values), 1e-4)
display(pd.DataFrame(REPORT))
''')
code("s12-fit-equivalence", r'''
# complete short SMT -> DMT fit: selected steps and selected weights vs the archived V.fit_smt (data streams, seeds,
# optimiser, schedule, evaluation, checkpoint rule)
import tempfile
small_fit = dict(seq_len=24, max_stack_size=6, smt_steps=6, dmt_steps=4, eval_every=2, validation_examples=16, batch_size=8, warmup_steps=2)
cfg_new = replace(STACK, **small_fit, eval_examples=16)
cfg_old = replace(V.V4Config(), **small_fit, dmt_batch_size=8)
with tempfile.TemporaryDirectory() as tmp:
    old = V.fit_smt(cfg_old, "base_window", 107, Path(tmp) / "old", {}, S.sample_stack(cfg_old, 16, S.derive_seed(501, 0)), CPU_, log=lambda s: None)
    new_dir, new_sum = fit(dict(name="eq", family="smt", seed=107, config=cfg_new), out_root=Path(tmp), log=lambda s: None)
    ns, nd = (torch.load(new_dir / f, weights_only=True) for f in ("smt_best.pt", "dmt_best.pt"))
    check("selected SMT step / DMT step", "V.fit_smt", abs(ns["step"] - old["smt_best_step"]) + abs(nd["step"] - old["dmt_best_step"]), 0)
    check("teacher at the selected SMT checkpoint", "V.fit_smt", diff(ns["models"]["teacher"], old["teacher"]), 0)
    check("RNN at the selected DMT checkpoint", "V.fit_smt", diff(nd["models"]["rnn"], old["rnn_dmt"]), 0)
eq_table = pd.DataFrame(REPORT); display(eq_table.tail(3)); print(f"{int(eq_table.ok.sum())}/{len(eq_table)} checks within tolerance")
''')
code("s12-interventions", r'''
# interventions: (1) future inputs do not change earlier memory (7.1); (2) a late input does not change an earlier causal
# output (7.2); (3) MQAR: permuting keys / values changes the answers ONLY by the rule "answer = value written after the
# queried key"; (4) the answer ID never appears among the inputs after the pairs.
def mqar_answers_by_rule(x, N):                     # independent re-derivation of y from x alone
    keys, vals = x[:2 * N:2].tolist(), x[1:2 * N:2].tolist()
    return [vals[keys.index(k)] if (t >= 2 * N and k != 0) else -1 for t, k in enumerate(x.tolist())]
cfgp = MQAR_PILOT01; bp = sample_mqar(cfgp, 200, 2024); N = cfgp.num_kv_pairs
for i in range(200):
    x, y = bp["x"][i], bp["y"][i]
    assert y.tolist() == mqar_answers_by_rule(x, N)
    perm = torch.randperm(N); x2 = x.clone(); x2[1:2 * N:2] = x[1:2 * N:2][perm]           # shuffle VALUES among pairs
    y2 = torch.tensor(mqar_answers_by_rule(x2, N))
    changed = (y2 != y) & y.ne(-1)
    assert all(int(y2[t]) == int(x2[2 * keys_idx + 1]) for t in changed.nonzero().flatten().tolist()
               for keys_idx in [x[:2 * N:2].tolist().index(int(x[t]))])
    assert not (x[2 * N:] >= cfgp.vocab // 2).any(), "a value ID after the pairs would leak the answer"
print("MQAR rule: 200 sequences re-derived from x alone; value permutation changes answers exactly as the rule says; "
      "no value ID after the pairs")
''')

# =====================================================================================================================
md("s13-full", r"""
## §13. FULL: один явно выбранный эксперимент

* Запускается только если `MODE = "FULL"` и задан `FULL_EXPERIMENT`. Результат пишется в **новый** каталог
  `results/notebook_runs/<имя>_<время>/`: `config.json`, истории, чекпойнты, `summary.json`.
* Архивные GPU-результаты этим не перезаписываются.
* Критерии остановки заданы заранее: бюджет шагов; `stop_accuracy` — только где он указан (MQAR-бейзлайны, как в
  архиве).
* **Новым ноутбуком на GPU пока ничего не обучено.**
""")
code("s13-full", r'''
EXPERIMENTS = {   # seeds 201/202/203 (new); configs fixed before running
    "stack_smt_tf8": dict(family="smt", config=STACK),
    "stack_smt_tf8_ctx_far": dict(family="smt", config=replace(STACK, far="ctx_far")),
    "stack_bptt_full": dict(family="bptt_full", config=STACK),
    "stack_bptt_1": dict(family="bptt_1", config=STACK),
    "stack_smt_tf8_no_dyn": dict(family="smt", config=replace(STACK, coef_dyn=0.0)),
    "stack_smt_tf8_target_detached": dict(family="smt", config=replace(STACK, dyn_detach="target")),   # = OLD 'detached'
    "stack_smt_tf8_fully_detached": dict(family="smt", config=replace(STACK, dyn_detach="both")),      # [новое]
    "mqar_T16N2_smt": dict(family="smt", config=replace(MQAR_T16, smt_steps=6000, dmt_steps=2000, eval_every=250)),
    "mqar_T16N2_teacher_bottleneck": dict(family="smt", config=replace(MQAR_T16, smt_steps=6000, dmt_steps=0, coef_dyn=0.0, coef_unif=0.0, eval_every=250)),
    "mqar_T16N2_teacher_full": dict(family="teacher_full", config=replace(MQAR_T16, smt_steps=6000, eval_every=250)),
    "mqar_T16N2_transformer": dict(family="transformer", config=replace(MQAR_T16, eval_every=250, stop_accuracy=0.99)),
    "mqar_T16N2_rnn_bptt_full": dict(family="bptt_full", config=replace(MQAR_T16, bptt_steps=6000, bptt_lr=1e-3, eval_every=250, stop_accuracy=0.99)),
    "mqar_T32N4_smt": dict(family="smt", config=replace(MQAR_T32, smt_steps=6000, dmt_steps=2000, eval_every=250)),
    "mqar_T32N4_transformer": dict(family="transformer", config=replace(MQAR_T32, eval_every=250, stop_accuracy=0.99)),
}
if MODE == "FULL" and FULL_EXPERIMENT:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")   # read by every function at call time
    e = EXPERIMENTS[FULL_EXPERIMENT]
    out_dir, summary = fit(dict(name=FULL_EXPERIMENT, seed=FULL_SEED, **e))
    print(out_dir); print(json.dumps(summary, indent=2))
else:
    print("FULL not requested: nothing trained. Experiments available:", ", ".join(EXPERIMENTS))
''')

# =====================================================================================================================
md("s14-archive-results", r"""
## §14. Архивные GPU-результаты: графики из сохранённых историй

* **Происхождение:** числа получены **старым кодом** на GPU (16–20.10.2026) и прочитаны из
  `archive/runs/*/…_history.csv`. Новый ноутбук их не повторял.
* **Сиды и выборка:** везде один seed 107, validation.
* **Сглаживание:** кривые не сглажены, точки — оценки каждые 500 шагов.
* **Пометки на кривых:**
  * **●** — чекпойнт, выбранный по минимальному validation CE на длинных ответах;
  * **×** — последняя точка.
""")
code("s14-archive-results", r'''
R = ARCHIVE / "runs"
def hist(path, col):
    h = pd.read_csv(path); return h.dropna(subset=[col])
def best_last(path, acc, ce):
    h = hist(path, ce); i = h[ce].idxmin()
    return dict(best_step=int(h.loc[i, "step"]), at_best=float(h.loc[i, acc]), last_step=int(h.step.iloc[-1]), last=float(h[acc].iloc[-1]), max=float(h[acc].max()))
rows = []
for name, d in [("v4 base_window Tf=8 (20k/12k)", "stackv4_d459c6921638/diag/base_window"), ("v4 ctx_far (20k/12k)", "stackv4_d459c6921638/diag/ctx_far")]:
    rows.append(dict(run=name, stage="DMT: final RNN", **best_last(R / d / "dmt_history.csv", "val_rnn_long_acc", "val_rnn_long_ce")))
    h = hist(R / d / "smt_history.csv", "val_teacher_long_ce"); i = h.val_teacher_long_ce.idxmin()
    rows.append(dict(run=name, stage="SMT: far-memory probe", best_step=int(h.loc[i, "step"]), at_best=float(h.loc[i, "val_far_memory_acc"]),
                     last_step=int(h.step.iloc[-1]), last=float(h.val_far_memory_acc.iloc[-1]), max=float(h.val_far_memory_acc.max())))
for lr in ("0.0003",):
    for v in ("bptt_full", "bptt_1"):
        rows.append(dict(run=f"v4 {v} LR {lr}", stage="BPTT: RNN", **best_last(R / f"stackv4_d459c6921638/diag/{v}_lr{lr}/bptt_history.csv", "val_rnn_long_acc", "val_rnn_long_ce")))
abl = {"full": "8318239d86e6", "target_detached (old name 'detached')": "8318239d86e6", "no_dyn": "93364b393f72", "no_unif": "7af76cfa569f",
       "tf1": "7c073f59ff92", "tf4": "03af753e0b10", "tf16": "b93af0011028", "tf64": "8067828308c1", "mem4": "7ccc3a06ed73", "mem16": "4cd5be1d3493"}
for name, key in abl.items():
    folder = "detached" if name.startswith("target_detached") else name
    rows.append(dict(run=f"ablation {name} (11k/5k)", stage="DMT: final RNN", **best_last(R / f"stackv4_{key}/ablation/{folder}/dmt_history.csv", "val_rnn_long_acc", "val_rnn_long_ce")))
res = pd.DataFrame(rows); display(res.round(4))
b1 = res[res.run.str.contains("bptt_1")].iloc[0]
print(f"BPTT-1: selected checkpoint (min val CE, step {b1.best_step}) = {b1["at_best"]:.1%}; last point (step {b1["last_step"]}) = {b1["last"]:.1%}")
print("far-memory probe: the archived tables show the MAXIMUM over the SMT curve (column 'max'); it is not taken at the "
      "checkpoint used for the RNN numbers (column 'at_best').")
same = (pd.read_csv(R / "stackv4_8318239d86e6/ablation/full/smt_history.csv")["loss"].values ==
        pd.read_csv(R / "stackv4_d459c6921638/diag/base_window/smt_history.csv")["loss"].values[:11000]).all()
print("ablation 'full' SMT history identical to the v4 diagnostic base_window up to step 11000:", bool(same), "-> not an independent run")

fig, axes = plt.subplots(1, 3, figsize=(17, 4.2))
for ax, (title, runs, col, ce) in zip(axes, [
        ("Stack v4: final RNN during DMT (seed 107, validation)", {"base_window Tf=8": "stackv4_d459c6921638/diag/base_window/dmt_history.csv",
                                                                   "ctx_far": "stackv4_d459c6921638/diag/ctx_far/dmt_history.csv"}, "val_rnn_long_acc", "val_rnn_long_ce"),
        ("Stack v4: same RNN trained directly (seed 107, validation)", {"full BPTT": "stackv4_d459c6921638/diag/bptt_full_lr0.0003/bptt_history.csv",
                                                                       "BPTT-1": "stackv4_d459c6921638/diag/bptt_1_lr0.0003/bptt_history.csv"}, "val_rnn_long_acc", "val_rnn_long_ce"),
        ("Stack v4: SMT teacher, j=0 (seed 107, validation)", {"base_window Tf=8": "stackv4_d459c6921638/diag/base_window/smt_history.csv",
                                                             "ctx_far": "stackv4_d459c6921638/diag/ctx_far/smt_history.csv"}, "val_teacher_long_acc", "val_teacher_long_ce")]):
    for label, path in runs.items():
        h = hist(R / path, ce); ax.plot(h.step, h[col], label=label)
        i = h[ce].idxmin(); ax.scatter(h.loc[i, "step"], h.loc[i, col], marker="o", s=40); ax.scatter(h.step.iloc[-1], h[col].iloc[-1], marker="x", s=50)
    ax.set_title(title, fontsize=9); ax.set_xlabel("optimizer step"); ax.set_ylabel(f"{col} (long answers, delay >= 8)")
    ax.axhline(0.25, ls=":", c="gray"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
fig.text(0.5, -0.03, "o = checkpoint selected by min validation CE on long answers; x = last point; no smoothing; chance 25%", ha="center", fontsize=9)
plt.tight_layout(); plt.show()
ls = pd.read_csv(ARCHIVE / "outputs_mqar/table_length_sweep.csv"); display(ls)
print("MQAR length sweep [история]: 'SMT teacher' = j = 0 at the query (encoder saw the query key); tuning split; seed 107.")
''')

# =====================================================================================================================
md("s15-changes", r"""
## §15. Изменения поведения по сравнению со старым кодом и найденные несоответствия

**Изменения поведения в ноутбуке.** При конфигурации архива всё остальное воспроизводит старый код: §12 сверяет это
побитно, кроме MQAR, где совпадение ~1e-6.

| что | старый код | ноутбук | влияние |
|---|---|---|---|
| имя абляции | `detached` | `dyn_detach="target"` = **target_detached**; новая `dyn_detach="both"` = fully detached | старые числа не меняются, меняется их название и смысл |
| MQAR SMT-окна | обрезка левого замаскированного паддинга | окна полной длины T с маской | ~1e-6 в loss (§12) |
| оценка teacher на MQAR | только `j = 0` на позиции запроса | + `teacher_pre` (после пар, до запроса) | **новая** диагностика, на обученных моделях не запускалась |
| итоговая оценка новых запусков | validation / tuning | + независимый поток 901 для выбранного и последнего чекпойнта | только новые запуски |
| `masked_mean` | у авторов без защиты от пустой маски | `.clamp(min=1)` | на непустых масках совпадает |
| ранняя остановка LM / BPTT на MQAR | `stop_accuracy = 0.99` в `mqar_baselines.py` | то же, в конфиге эксперимента | — |
| тест узкого места | постоянный LR 3e-4 без разогрева, 4000 шагов, 128 последовательностей | разогрев 200 шагов, бюджет и валидация — из конфига | **новые** запуски не равны архивному тесту |

**Несоответствия, найденные при разборе.** Записаны до исправлений; исправлены только описания, прежние числа не
трогались.
1. Архивная абляция `detached` не отключает градиент L_dyn в энкодер (§7.4); правильное имя — target_detached. Её
   identity совпадает с `full`, потому что патч в identity не записан.
2. Абляция `full` — та же траектория, что диагностика `base_window`, только короче; `tf64` = «полное окно» v4. Это не
   независимые повторы.
3. BPTT-1: 36,7% — последняя точка, на выбранном по CE чекпойнте 33,8% (§14).
4. Far-probe в таблицах — максимум по кривой SMT, а не значение на выбранном чекпойнте (§14).
5. На MQAR teacher измерялся только при `j = 0`, когда энкодер уже видел ключ запроса (§3.2).
6. Диагностика «все выходы энкодера» меняет сразу три вещи: число векторов, прямой доступ к позициям и нормировку (нет
   RMS-нормы памяти). Это не чистая проверка вместимости 8×64.
7. `Tf = 8` не сокращает вычисления: декодер считает все T выходов (§7.3).
8. Первый переход `f(0, x_0) → m_0` в L_dyn и старт DMT с нулевого состояния по всей длине — наш выбор: у авторов DMT
   стартует из состояния teacher.
""")

# =====================================================================================================================
md("s16-summary", r"""
## §16. Итог прогона

Ниже — сводка выполненных в этом прогоне проверок. Если ячейка дошла до конца, все `assert` прошли.
""")
code("s16-summary", r'''
print(f"equivalence checks within tolerance: {int(eq_table.ok.sum())}/{len(eq_table)}")
print("subtle-point asserts (§7), interventions (§12), SMOKE fit (§10): passed")
print("trained with the new notebook on GPU: nothing (MODE =", MODE, ", FULL_EXPERIMENT =", FULL_EXPERIMENT, ")")
''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    nb = nbf.v4.new_notebook(cells=CELLS, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    path = ROOT / "SMT_MemUP_FROM_ZERO.ipynb"
    if a.execute:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=a.timeout, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
    nbf.write(nb, path)
    print(f"wrote {path.name}: {len(CELLS)} cells")


if __name__ == "__main__":
    sys.exit(main())
