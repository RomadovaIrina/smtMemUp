#!/usr/bin/env python3
"""Builds SMT_MemUP_EXPLAINABLE.ipynb: a short executable notebook to explain the task, shapes, training and inference.
    python build_notebook_explainable.py             # write the notebook (no outputs)
    python build_notebook_explainable.py --execute   # write and run it top to bottom on CPU (a few minutes), save outputs
Optional long checks (equivalence with the code that produced results_3090/) live in explainable_extras.py."""
import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
CELLS = []


def md(tag, text):
    CELLS.append(nbf.v4.new_markdown_cell(text.strip("\n"), metadata={"tags": [tag]}))


def code(tag, text):
    CELLS.append(nbf.v4.new_code_cell(text.strip("\n"), metadata={"tags": [tag]}))


md("title", r"""
# SMT × MemUP: задача, размерности, обучение и инференс — коротко и исполняемо

* Ноутбук идёт сверху вниз на **CPU** за несколько минут. Каждая ячейка с `assert` проверяет своё утверждение.
* Полные обучения здесь **не запускаются**. Их запускает `run_experiments.py` кодом `SMT_MemUP_FROM_ZERO.ipynb`.
  Сохранённые результаты читаются в §9.
* **Пометки происхождения:**
  * **[авторы SMT]** — копия кода авторов SMT, поведение не менялось;
  * **[Zoology]** — копия генератора Zoology;
  * **[наше]** — наш код;
  * **[адаптация]** — наш код по мотивам авторского.
* Сверка с кодом, который дал результаты `results_3090/`: `explainable_extras.check_against_v1` (необязательный
  раздел в конце).
""")

md("s0", "## §0. Окружение")
code("s0-setup", r'''
import json, math, random, time
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F
from einops import rearrange, repeat
from x_transformers import Decoder, Encoder          # Transformer blocks (ordinary dependency, version 2.18.5)

ROOT = next(p for p in (Path.cwd(), *Path.cwd().parents) if (p / "archive").is_dir() and (p / "results_3090").is_dir())
DEVICE = torch.device("cpu"); torch.set_num_threads(4)
IGNORE = -1                                          # label "no answer here" and padding ID in windows; NOT a class
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)

def derive_seed(*parts):                             # one seed per (data stream, run seed, step) [наше]
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])

@contextmanager
def seeded(seed):                                    # the author generators use the GLOBAL numpy / torch RNGs [наше]
    st = np.random.get_state(), random.getstate(), torch.get_rng_state()
    try:
        np.random.seed(int(seed) % 2**32); random.seed(int(seed)); torch.manual_seed(int(seed)); yield
    finally:
        np.random.set_state(st[0]); random.setstate(st[1]); torch.set_rng_state(st[2])
print("torch", torch.__version__, "| root", ROOT.name)
''')

md("s1", r"""
## §1. Обозначения и основная конфигурация

`K_mem` — число токенов памяти. `K_far` — сколько дальних целей выбирается на якорь. Это разные величины.
""")
code("s1-config", r'''
@dataclass(frozen=True)
class Config:                                        # [наше]; same values as the GPU runs of results_3090/
    task: str = "stack"          # stack | mqar
    T: int = 64                  # example length; positions 0..T-1; also the encoder window
    n_elements: int = 4          # Stack: push IDs 1..4, pop ID 0
    max_stack_size: int = 8
    vocab: int = 128             # MQAR vocabulary V
    N: int = 2                   # MQAR: number of key-value pairs
    power_a: float = 0.01        # MQAR: Zoology query-gap power law
    long_delay: int = 8          # an answer is "long" if delay >= long_delay (Stack delays are odd -> >= 9)
    d: int = 64                  # width of every token / vector
    K_mem: int = 8               # number of memory tokens
    enc_depth: int = 3; dec_depth: int = 2; rnn_depth: int = 2; heads: int = 4
    T_f: int = 8                 # near offsets j = 0..T_f-1 enter L_near
    far: str = "none"            # far-target rule: none | random_far | surprise_far | ctx_far | uniform_far | oracle_far
    K_far: int = 2               # far targets chosen per anchor
    alpha_far: float = 1.0
    c_dec: float = 1.0; c_dyn: float = 0.1; c_unif: float = 0.001
    dyn_detach: str = "none"     # none | target | both (ablations)
    c_dmt_mse: float = 10.0      # DMT: weight of the state MSE
    B: int = 32                  # sequences per batch
    @property
    def V(self): return self.n_elements + 1 if self.task == "stack" else self.vocab   # number of classes = IDs
    @property
    def Z(self): return self.K_mem * self.d                                           # flat memory length

STACK = Config()
MQAR_T16 = Config(task="mqar", T=16, N=2, long_delay=4, T_f=16)
MQAR_T32 = Config(task="mqar", T=32, N=4, long_delay=8, T_f=32)
c = STACK
display(pd.DataFrame([
    ("B", "последовательностей в батче", c.B), ("T", "длина примера, позиции 0..T-1", c.T),
    ("V", "число ID = число классов (Stack: 0 = pop, 1..4 = push)", c.V), ("K_mem", "токенов памяти", c.K_mem),
    ("d", "ширина токена", c.d), ("Z = K_mem·d", "длина плоской памяти m_t", c.Z),
    ("R = B·T", "окон (якорей) в батче", c.B * c.T), ("T_f", "ближних смещений j в L_near", c.T_f),
    ("K_far", "дальних целей на якорь (если far != none)", c.K_far)], columns=["символ", "смысл", "Stack"]))
assert (c.B, c.T, c.V, c.K_mem, c.d, c.Z, c.T_f, c.K_far) == (32, 64, 5, 8, 64, 512, 8, 2)
''')

md("s2", r"""
## §2. Данные: целые ID, ответы на тех же позициях

Текстового токенизатора нет: генератор сразу выдаёт целые ID.
* **Stack:** `0` = pop, `1..4` = push элемента. На pop `y` равен снятому элементу; на push `y = −1`.
* `−1` означает «ответа нет»: такие позиции маскируются в loss и метриках. Это **не шестой класс**.
""")
code("s2-stack", r'''
def author_stack_sample_batch(batch_size, ctx_len, n_elements=4, max_stack_size=8):
    """[авторы SMT] StackDataset.sample_batch, upstream/smt/src/my_datasets/synthetic.py:40-64 @ cf0d0f0."""
    x = torch.zeros(batch_size, ctx_len, dtype=torch.long); y = torch.zeros(batch_size, ctx_len, dtype=torch.long)
    for b in range(batch_size):
        stack = []
        for t in range(ctx_len):
            can_push, can_pop = len(stack) < max_stack_size, len(stack) > 0
            if can_push and can_pop: should_push = np.random.rand() < 0.5
            elif can_push: should_push = True
            elif can_pop: should_push = False
            rand_element = np.random.randint(n_elements) + 1
            if should_push: stack.append(rand_element); act, obs = rand_element, -1
            else: act, obs = 0, stack.pop()
            x[b, t], y[b, t] = act, obs
    return dict(x=x, y=y)

def annotate(x, y, source):                          # [наше] metadata for metrics only; models never see it
    mask = y.ne(IGNORE); pos = torch.arange(x.shape[1]).expand_as(x)
    return dict(x=x, y=y, target_mask=mask, source_pos=source, delay=torch.where(mask, pos - source, torch.full_like(x, IGNORE)))

def stack_sources(x):                                # [наше] replay: position of the push that each pop removes
    source = torch.full_like(x, IGNORE)
    for b in range(len(x)):
        st = []
        for t, tok in enumerate(x[b].tolist()):
            if tok > 0: st.append(t)
            else: source[b, t] = st.pop()
    return source

def sample_stack(cfg, batch_size, seed):
    with seeded(seed):
        raw = author_stack_sample_batch(batch_size, cfg.T, cfg.n_elements, cfg.max_stack_size)
    return annotate(raw["x"], raw["y"], stack_sources(raw["x"]))       # x, y: [batch, T] int64

b = sample_stack(STACK, 2, 7)
print("x[0, :16] =", b["x"][0, :16].tolist()); print("y[0, :16] =", b["y"][0, :16].tolist())
assert b["x"].dtype == torch.long and set(b["x"].unique().tolist()) <= set(range(STACK.V))
assert ((b["x"] == 0) == b["y"].ne(IGNORE)).all(), "an answer exactly at every pop, -1 at every push"
''')

md("s2-example", r"""
### §2.1 Сквозной пример Stack

`x = [1, 2, 0, 3, 0, 0]`, `y = [−1, −1, 2, −1, 3, 1]`. `delay` = позиция pop − позиция push. Это не то же
самое, что смещение `j` в декодере (§3).
""")
code("s2-example", r'''
x_ex = torch.tensor([[1, 2, 0, 3, 0, 0]]); y_ex = torch.tensor([[-1, -1, 2, -1, 3, 1]])
ex = annotate(x_ex, y_ex, stack_sources(x_ex))
rows, st = [], []
for t, tok in enumerate(x_ex[0].tolist()):
    st.append(tok) if tok > 0 else st.pop()
    rows.append(dict(t=t, x_t=tok, operation="pop" if tok == 0 else f"push {tok}", stack_after=str(st),
                     y_t=int(y_ex[0, t]), source_push=int(ex["source_pos"][0, t]), delay=int(ex["delay"][0, t])))
display(pd.DataFrame(rows))
for t in range(6):                                   # the label of a pop is exactly the element its source push wrote
    if y_ex[0, t] != IGNORE: assert y_ex[0, t] == x_ex[0, ex["source_pos"][0, t]]
assert ex["source_pos"][0].tolist() == [-1, -1, 1, -1, 3, 0] and ex["delay"][0].tolist() == [-1, -1, 1, -1, 1, 5]
''')

md("s2-mqar", r"""
### §2.2 MQAR — генератор Zoology

* Одна пара = **2 ID**: ключ на позиции `2i`, значение на `2i+1`, `i < N`.
* Ключи берутся из `1..V/2−1`, значения — из `V/2..V−1`.
* Потом каждый ключ запрашивается ровно один раз: запрос — это сам ID ключа. Остальные позиции — заполнитель `0`.
* **Метка стоит на позиции запроса.**
""")
code("s2-mqar", r'''
def zoology_multiquery_ar(vocab_size, num_examples, input_seq_len, seed, power_a=0.01, num_kv_pairs=8):
    """[Zoology] zoology/data/multiquery_ar.py:88-141 @ 1ad20d1, num_passes = 1, random_non_queries = False."""
    assert input_seq_len % 2 == 0 and vocab_size > input_seq_len and 4 * num_kv_pairs <= input_seq_len
    np.random.seed(seed)
    context_size, key_vocab_size = num_kv_pairs * 2, vocab_size // 2
    key_choices, value_choices = np.arange(1, key_vocab_size), np.arange(key_vocab_size, vocab_size)
    keys = np.apply_along_axis(np.random.choice, 1, np.tile(key_choices, (num_examples, 1)), replace=False, size=num_kv_pairs)
    values = np.apply_along_axis(np.random.choice, 1, np.tile(value_choices, (num_examples, 1)), replace=False, size=num_kv_pairs)
    kvs = np.zeros((num_examples, context_size), dtype=np.int64); kvs[:, 0::2], kvs[:, 1::2] = keys, values
    space = (input_seq_len - context_size) // 2
    p = power_a * np.arange(1, space + 1) ** (power_a - 1); p = p / p.sum()
    xx = np.stack([np.arange(space, dtype=int)] * num_examples)
    gaps = np.apply_along_axis(np.random.choice, axis=1, arr=xx, replace=False, p=p, size=num_kv_pairs)
    queries = np.zeros((num_examples, input_seq_len - context_size + 1), dtype=np.int64)
    np.put_along_axis(queries, gaps * 2, values=keys, axis=1)
    examples = np.concatenate([kvs, queries], axis=1)
    labels = np.full((num_examples, input_seq_len + 1), -100, dtype=np.int64)
    np.put_along_axis(labels, gaps * 2 + context_size + 1, values=values, axis=1)
    return torch.tensor(examples[:, :-1]), torch.tensor(labels[:, 1:])          # label shifted onto the query position

def sample_mqar(cfg, batch_size, seed):              # [адаптация] -100 -> -1, plus metadata
    with seeded(seed):
        x, y = zoology_multiquery_ar(cfg.vocab, batch_size, cfg.T, int(seed) % 2**32, cfg.power_a, cfg.N)
    x, y = x.long().clone(), y.long().clone(); y[y == -100] = IGNORE
    keys = x[:, 0:2 * cfg.N:2]                                                       # [batch, N]
    slot = (x[:, :, None] == keys[:, None, :]).float().argmax(-1)                    # which pair a query asks
    return annotate(x, y, torch.where(y.ne(IGNORE), 2 * slot + 1, torch.full_like(x, IGNORE)))

sample = lambda cfg, n, seed: (sample_stack if cfg.task == "stack" else sample_mqar)(cfg, n, seed)
mq = sample_mqar(MQAR_T16, 1, derive_seed(901, 0)); N, V = MQAR_T16.N, MQAR_T16.V
xm, ym = mq["x"][0].tolist(), mq["y"][0].tolist()
role = lambda t: f"key {t // 2}" if t < 2 * N and t % 2 == 0 else f"value {t // 2}" if t < 2 * N else ("QUERY (key ID)" if ym[t] != IGNORE else "filler 0")
display(pd.DataFrame([dict(t=t, x_t=xm[t], role=role(t), y_t=ym[t], value_written_at=int(mq["source_pos"][0, t]),
                           delay=int(mq["delay"][0, t])) for t in range(MQAR_T16.T)]).T)
k = mq["x"][0, 0:2 * N:2]; v = mq["x"][0, 1:2 * N:2]
assert ((1 <= k) & (k < V // 2)).all() and ((V // 2 <= v) & (v < V)).all(), "keys 1..V/2-1, values V/2..V-1"
for t in mq["target_mask"][0].nonzero().flatten().tolist():     # at a query: x_t = key i, y_t = value i
    i = (k == xm[t]).nonzero().item(); assert ym[t] == int(v[i]) and int(mq["source_pos"][0, t]) == 2 * i + 1
''')

md("s3", r"""
## §3. Окна SMT: что видит энкодер, что видит декодер, что предсказывается

Для каждого якоря `t`:

| тензор | что содержит | паддинг |
|---|---|---|
| `x_ctx[t]` | `x_{t−T+1..t}` | слева −1 |
| `x_fut[t]` | `x_{t+1..t+T}` | справа −1 |
| `y_fut[t]` | `y_{t..t+T−1}` | справа −1 |

* **Выход `j`** предсказывает `y_{t+j}` по памяти `m_t` и входам `x_{t+1..t+j}`.
* **`prefix`:** источник ответа (push или значение) записан не позже `t`. Тогда ответ можно взять только из памяти.
""")
code("s3-windows", r'''
def windows(batch):                                  # [наше], identical to SMT_MemUP_FROM_ZERO.windows
    x, y = batch["x"], batch["y"]; B, T = x.shape                                   # x, y: [B, T]
    unfold = lambda z, left, right: F.pad(z, (left, right), value=IGNORE).unfold(1, T, 1)   # -> [B, T(+1), T]
    w = dict(x_ctx=unfold(x, T - 1, 0),                  # [B,T,T]: row t = x[t-T+1 .. t]
             x_fut=unfold(x, 0, T)[:, 1:T + 1],          # [B,T,T]: row t = x[t+1 .. t+T]
             y_fut=unfold(y, 0, T - 1),                  # [B,T,T]: row t = y[t .. t+T-1]
             source=unfold(batch["source_pos"], 0, T - 1))
    w["valid"] = w["y_fut"].ne(IGNORE)
    w["prefix"] = w["valid"] & (w["source"] <= torch.arange(T).view(1, T, 1))      # source written at or before t
    return {k: v.reshape(B * T, T) for k, v in w.items()}                           # [B,T,T] -> [R = B*T, T], row r = b*T + t

def show_anchor(batch, t, T_f, js=range(4), b=0):
    T = batch["x"].shape[1]; w = windows(batch); r = b * T + t
    print(f"anchor t={t}: x_ctx = {w['x_ctx'][r].tolist()} | x_fut = {w['x_fut'][r].tolist()} | "
          f"y_fut = {w['y_fut'][r].tolist()} | valid = {w['valid'][r].int().tolist()}")
    display(pd.DataFrame([dict(j=j, decoder_reads=f"m_{t} + x_fut[:{j}] = {w['x_fut'][r, :j].tolist()}",
                               predicts=f"y_{t + j} = {int(w['y_fut'][r, j])}", valid=bool(w["valid"][r, j]),
                               near_or_far=("near (j < T_f)" if j < T_f else "far (j ≥ T_f)") if w["valid"][r, j] else "-",
                               source_push=(int(w["source"][r, j]) if w["valid"][r, j] else "-"),
                               source_is=("prefix: only in m_t" if w["prefix"][r, j] else "future: inside x_fut[:j]") if w["valid"][r, j] else "-")
                          for j in js]))

print("TOY T_f = 2 only to make the near / far border visible; the experiments use T_f = 8.")
show_anchor(ex, 2, T_f=2); show_anchor(ex, 3, T_f=2)
w = windows(ex)
assert w["x_ctx"][2].tolist() == [-1, -1, -1, 1, 2, 0] and w["x_fut"][2].tolist() == [3, 0, 0, -1, -1, -1]
assert w["y_fut"][2].tolist() == [2, -1, 3, 1, -1, -1] and w["prefix"][2].tolist() == [True, False, False, True, False, False]
assert w["x_ctx"][3].tolist() == [-1, -1, 1, 2, 0, 3] and w["y_fut"][3].tolist() == [-1, 3, 1, -1, -1, -1]
assert w["prefix"][3].tolist() == [False, True, True, False, False, False]       # push of y_4 (t=3) and of y_5 (t=0) are <= 3
for r in range(6):                                   # alignment: output j of anchor t targets y_{t+j}
    for j in range(6):
        if r + j < 6: assert w["y_fut"][r, j] == y_ex[0, r + j]
''')
code("s3-t10", r'''
bs = sample_stack(STACK, STACK.B, derive_seed(301, 201, 0)); ws = windows(bs); t = 10
real_ctx = int(ws["x_ctx"][t].ne(IGNORE).sum()); real_fut = int(ws["x_fut"][t].ne(IGNORE).sum())
print(f"T = 64, anchor t = {t}: x_ctx has {real_ctx} real inputs (x_0..x_10) and {64 - real_ctx} padding; "
      f"x_fut has {real_fut} real inputs (x_11..x_63) and {64 - real_fut} padding")
assert (real_ctx, 64 - real_ctx, real_fut) == (11, 53, 53)
assert torch.equal(ws["x_ctx"][t, -11:], bs["x"][0, :11]) and (ws["x_ctx"][t, :53] == IGNORE).all()
assert ws["x_ctx"].shape == ws["x_fut"].shape == ws["y_fut"].shape == (STACK.B * STACK.T, STACK.T)    # [R=2048, T=64]
''')

md("s4-math", r"""
## §4. Математика и loss

**Обозначения:**
* `T` — длина примера, индексы `0..T−1`;
* `t` — якорь: память `m_t` строится по `x_0..x_t`;
* `j` — смещение цели: выход декодера `G_j` предсказывает `y_{t+j}`;
* `T_f` — сколько ближайших смещений `j = 0..T_f−1` входят в основной декодерный loss;
* `delay` — расстояние от push до pop (§2.1). Это **не** `j`.

$$L_{SMT}=c_{dec}\,(L_{near}+\alpha_{far}L_{far})+c_{dyn}L_{dyn}+c_{unif}L_{unif},\qquad L_{DMT}=10\cdot\mathrm{MSE}+\mathrm{CE}$$

| член | функция | вход → цель | маска / вес | градиент получают |
|---|---|---|---|---|
| `L_near` | `smt_loss` → `dec` | `decode(m_t, x_fut)[:, j]` → `y_{t+j}`, CE | `valid & j < T_f`; среднее по парам | teacher: embed, encoder, регистры, decoder, head |
| `L_far` | `smt_loss` → `aux` | те же логиты, `j ≥ T_f` | `K_far` выбранных из пула `valid & j ≥ T_f`; `far="none"` → 0 | то же |
| `L_dyn` | `smt_loss` → `dyn` | `f(m_{t−1}, x_t)` → `m_t`, MSE по `Z` числам; `m_{−1} = 0` | все `B·T` якорей | updater RNN и (через `m_{t−1}` и `m_t`) энкодер |
| `L_unif` | `calc_uniformity_loss` | токены памяти `m_t` | 512 случайных токенов | энкодер |
| `10·MSE` (DMT) | `dmt_loss` → `dyn` | `f(m̂_{t−1}, x_t)` → `m_t` замороженного энкодера | вход `m̂_{t−1}` detach | updater RNN и `rnn.embed` |
| `CE` (DMT) | `dmt_loss` → `readout` | `readout(sg[m̂_t])` → `y_t` | позиции с ответом | только readout RNN |

Базовая конфигурация: `c_dec = 1`, `c_dyn = 0,1`, `c_unif = 0,001`, `α_far = 1`, `T_f = 8`, `K_far = 2`.
""")

md("s5", r"""
## §5. Модели — **[авторы SMT]** `upstream/smt/src/model.py:21-226` @ cf0d0f0

Код скопирован, добавлены только комментарии с формами. В авторском коде `mem_tokens` — это наш `K_mem`.
""")
code("s5-model", r'''
def calc_cross_entropy_loss(logits, y):              # [..., V], [...] -> [...]
    shape = y.shape
    return F.cross_entropy(rearrange(logits, "... V -> (...) V"), rearrange(y, "... -> (...)"), reduction="none").reshape(shape)

def calc_mse_loss(y_pred, y): return F.mse_loss(y_pred, y, reduction="none")

def calc_uniformity_loss(z, t=2, sample=512):        # z [..., d] -> scalar (uses the global torch RNG)
    z = z / (z.norm(dim=-1, keepdim=True) + torch.finfo(z.dtype).eps)
    z = rearrange(z, "... D -> (...) D"); z = z[torch.randperm(len(z))[:sample]]
    return torch.pdist(z, p=2).pow(2).mul(-t).exp().mean().log()

@dataclass
class TeacherConfig:
    vocab_size: int = 256; mem_size: int = 256; mem_tokens: int = 1; d_model: int = 256
    encoder_depth: int = 8; decoder_depth: int = 4; n_heads: int = 8

class Teacher(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)                  # trainable table [V, d]
        self.encoder = Encoder(dim=cfg.d_model, depth=cfg.encoder_depth, heads=cfg.n_heads, rotary_pos_emb=True, use_rmsnorm=True)
        self.decoder = Decoder(dim=cfg.d_model, depth=cfg.decoder_depth, heads=cfg.n_heads, rotary_pos_emb=True, use_rmsnorm=True)
        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size)                 # d -> V
        self.encoder_mem_register = nn.Embedding(cfg.mem_tokens, cfg.d_model)   # K_mem learned INITIAL registers [K_mem, d]

    def encode(self, x_ctx):                         # x_ctx [N, T] IDs (-1 = pad) -> m [N, Z]
        B, Tc = x_ctx.shape
        mask = torch.cat([x_ctx != -1, torch.ones(B, self.cfg.mem_tokens, dtype=torch.bool)], 1)   # [N, T+K_mem]
        x_emb = self.embed(x_ctx.clamp(min=0))                                  # [N, T, d]
        mem_emb = repeat(self.encoder_mem_register(torch.arange(self.cfg.mem_tokens)), "K D -> B K D", B=B)  # [N, K_mem, d]
        z = self.encoder(torch.cat([x_emb, mem_emb], 1), mask=mask)             # [N, T+K_mem, d], bidirectional
        z = F.rms_norm(z[:, -self.cfg.mem_tokens:], (self.cfg.d_model,))       # register OUTPUTS [N, K_mem, d], RMS-normed
        return rearrange(z, "B K d -> B (K d)")                                 # m_t [N, Z = K_mem*d]

    def decode(self, z, x_fut):                      # z [N, Z], x_fut [N, T] -> logits [N, T, V]
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)             # [N, K_mem, d]
        h = self.decoder(torch.cat([z, self.embed(x_fut.clamp(min=0))], 1))    # [N, K_mem+T, d], causal
        return self.to_logits(h)[:, self.cfg.mem_tokens - 1:-1]                # output j at position K_mem-1+j: sees m, x_fut[:j]

    def teacher_readout(self, z):                    # = decode output j = 0: [N, Z] -> [N, V]
        return self.to_logits(self.decoder(rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens))[:, -1])

@dataclass
class RNNConfig:
    vocab_size: int = 256; mem_size: int = 256; mem_tokens: int = 1; d_model: int = 256
    rnn_depth: int = 8; decoder_depth: int = 4; n_heads: int = 8

class RNN(nn.Module):
    def __init__(self, cfg):
        super().__init__(); self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)                  # its OWN trainable table [V, d]
        self.rnn = Encoder(dim=cfg.d_model, depth=cfg.rnn_depth, heads=cfg.n_heads, rotary_pos_emb=True, use_rmsnorm=True)  # updater f
        self.decoder = Decoder(dim=cfg.d_model, depth=cfg.decoder_depth, heads=cfg.n_heads, rotary_pos_emb=True, use_rmsnorm=True)
        self.to_logits = nn.Linear(cfg.d_model, cfg.vocab_size)

    def rnn_update(self, z, x):                      # m_{t-1} [N, Z], x_t [N, 1] -> m_t [N, Z]
        z = rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens)             # [N, K_mem, d]
        z = self.rnn(torch.cat([z, self.embed(x)], 1))                          # [N, K_mem+1, d], attention over K_mem+1 tokens
        z = F.rms_norm(z[:, :self.cfg.mem_tokens], (self.cfg.d_model,))         # first K_mem outputs; no residual to m_{t-1}
        return rearrange(z, "B K d -> B (K d)")

    def rnn_readout(self, z):                        # [N, Z] -> [N, V]
        return self.to_logits(self.decoder(rearrange(z, "B (K d) -> B K d", K=self.cfg.mem_tokens))[:, -1])
''')
code("s5-make", r'''
def make_models(cfg, seed):                          # [наше] same seed -> same weights as the GPU runs
    common = dict(vocab_size=cfg.V, mem_size=cfg.Z, mem_tokens=cfg.K_mem, d_model=cfg.d, decoder_depth=cfg.dec_depth, n_heads=cfg.heads)
    with seeded(seed):
        teacher = Teacher(TeacherConfig(**common, encoder_depth=cfg.enc_depth)); rnn = RNN(RNNConfig(**common, rnn_depth=cfg.rnn_depth))
    copy_readout(teacher, rnn); return teacher, rnn

def copy_readout(teacher, rnn):                      # [авторы SMT, train_smt.py] RNN readout := copy of the teacher's
    rnn.decoder.load_state_dict(teacher.decoder.state_dict()); rnn.to_logits.load_state_dict(teacher.to_logits.state_dict())

def rnn_rollout(rnn, x, init=None, detach=False):    # m_t = f(m_{t-1}, x_t), m_{-1} = 0: x [B, T] -> [B, T, Z]
    state = torch.zeros(x.shape[0], rnn.cfg.mem_size) if init is None else init
    out = []
    for t in range(x.shape[1]):
        state = rnn.rnn_update(state.detach() if detach else state, x[:, t:t + 1]); out.append(state)
    return torch.stack(out, 1)

teacher, rnn = make_models(STACK, 201)
cnt = lambda m: sum(p.numel() for p in m.parameters())
display(pd.DataFrame([("teacher.embed [V, d]", tuple(teacher.embed.weight.shape), cnt(teacher.embed)),
                      ("teacher.encoder_mem_register [K_mem, d]", tuple(teacher.encoder_mem_register.weight.shape), cnt(teacher.encoder_mem_register)),
                      ("teacher.encoder (3 layers)", "-", cnt(teacher.encoder)), ("teacher.decoder (2 layers)", "-", cnt(teacher.decoder)),
                      ("rnn.embed [V, d] (separate)", tuple(rnn.embed.weight.shape), cnt(rnn.embed)),
                      ("rnn.rnn = updater f (2 layers)", "-", cnt(rnn.rnn)), ("teacher total", "-", cnt(teacher)), ("rnn total", "-", cnt(rnn))],
                     columns=["part", "shape", "parameters"]))
assert teacher.embed.weight.data_ptr() != rnn.embed.weight.data_ptr() and not torch.equal(teacher.embed.weight, rnn.embed.weight)
with torch.no_grad():
    regs = teacher.encoder_mem_register.weight                                   # [8, 64]: same for every sequence
    m0 = teacher.encode(windows(ex)["x_ctx"][:2]).reshape(2, STACK.K_mem, STACK.d)     # memory AFTER the encoder: [2, 8, 64]
assert regs.shape == m0.shape[1:] == (8, 64) and not torch.allclose(regs, m0[0]) and not torch.allclose(m0[0], m0[1])
print("registers [8,64] are parameters (identical for every input); memory m_t [8,64] is the encoder OUTPUT at them (depends on x_0..x_t)")
''')
code("s5-trace", r'''
# one Stack batch in the main configuration, forward pass of SMT with the shape after every step
with torch.no_grad():
    rows = []; add = lambda name, z, axes: rows.append((name, tuple(z.shape), axes))
    add("x", bs["x"], "[B=32, T=64] IDs"); add("x_ctx / x_fut / y_fut", ws["x_ctx"], "[R=B·T=2048, T=64]")
    xc, xf = ws["x_ctx"][:256], ws["x_fut"][:256]                                 # first 256 anchors (memory); same per row
    add("embed(x_ctx) ++ registers", torch.cat([teacher.embed(xc.clamp(min=0)), repeat(teacher.encoder_mem_register.weight, "K D -> N K D", N=256)], 1), "[256, T+K_mem=72, d=64]")
    m = teacher.encode(xc); add("m_t = encode(x_ctx)", m, "[256, Z=512] = [256, K_mem=8 × d=64]")
    logits = teacher.decode(m, xf); add("decode(m_t, x_fut)", logits, "[256, T=64 outputs j, V=5]")
    prev = torch.cat([torch.zeros(1, STACK.Z), m[:-1]]); add("[m_{t-1}, embed(x_t)] → f", torch.cat([rearrange(prev, "N (K d) -> N K d", K=8), rnn.embed(bs["x"].reshape(-1, 1)[:256])], 1), "[256, K_mem+1=9, d]")
    add("f(m_{t-1}, x_t)", rnn.rnn_update(prev, bs["x"].reshape(-1, 1)[:256]), "[256, Z=512]")
    add("rnn_readout(m)", rnn.rnn_readout(m), "[256, V=5]")
    display(pd.DataFrame(rows, columns=["tensor", "shape", "axes"]))
    assert torch.allclose(logits[:, 0], teacher.teacher_readout(m), atol=1e-5)   # output j = 0 == teacher_readout
    xf2 = xf.clone(); xf2[:, 4:] = torch.randint(0, 5, xf2[:, 4:].shape)
    assert torch.allclose(logits[:, :5], teacher.decode(m, xf2)[:, :5], atol=1e-5), "output j must not see x_fut[j:]"
    x2 = bs["x"].clone(); x2[:, 11:] = torch.randint(0, 5, x2[:, 11:].shape)     # editing the future cannot change m_10
    assert torch.allclose(teacher.encode(ws["x_ctx"][10:11]), teacher.encode(windows(dict(bs, x=x2))["x_ctx"][10:11]), atol=1e-6)
print("asserts: j = 0 equals teacher_readout; outputs j <= 4 ignore x_fut[4:]; m_10 ignores x_11..x_63")
''')

md("s6", r"""
## §6. Выбор дальних целей — **[адаптация]** `archive/smt_memup_stack.py:434-481`, `smt_memup_window.py`

Пул якоря `t` — валидные пары `j ≥ T_f`; из пула выбираются `K_far` целей.
* **`ctx_far`** — CE детектора, который видит только входы декодера `x_{t+1..t+j}` без памяти;
* **`surprise_far`** — CE оконного детектора по 8 последним входам;
* **`random_far`** — случайный выбор;
* **`uniform_far`** — весь пул;
* **`oracle_far`** — диагностика: пул сужен до целей, чей источник уже в памяти.

Детекторы — отдельные маленькие GRU. Они заморожены и при инференсе не используются.
""")
code("s6-far", r'''
class WindowDetector(nn.Module):                     # q(y_k | x_{k-7..k})
    def __init__(self, V, embed=16, hidden=64, r=8):
        super().__init__(); self.r = r
        self.embed = nn.Embedding(V + 1, embed, padding_idx=0); self.gru = nn.GRU(embed, hidden, batch_first=True); self.head = nn.Linear(hidden, V)
    def forward(self, xw): _, h = self.gru(self.embed(xw + 1)); return self.head(h[-1])
    def scores(self, batch):                         # CE of the TRUE label at every position: [B, T]
        x, y = batch["x"], batch["y"]
        win = F.pad(x, (self.r - 1, 0), value=IGNORE).unfold(1, self.r, 1)
        ce = F.cross_entropy(self(win.reshape(-1, self.r)), y.clamp(min=0).flatten(), reduction="none")
        return (ce.reshape(x.shape) * batch["target_mask"]).detach()

class ContextDetector(nn.Module):                    # q(y_{t+j} | x_{t+1..t+j}) WITHOUT memory
    def __init__(self, V, embed=16, hidden=64):
        super().__init__(); self.bos = V + 1
        self.embed = nn.Embedding(V + 2, embed, padding_idx=0); self.gru = nn.GRU(embed, hidden, batch_first=True); self.head = nn.Linear(hidden, V)
    def forward(self, x_fut):
        inp = torch.cat([torch.full_like(x_fut[:, :1], self.bos - 1), x_fut[:, :-1]], 1); h, _ = self.gru(self.embed(inp + 1)); return self.head(h)

def load_detectors(V=5):                             # trained once (2000 steps), frozen
    dets = {}
    for kind, cls in (("window", WindowDetector), ("context", ContextDetector)):
        m = cls(V); m.load_state_dict(torch.load(ROOT / f"archive/runs/stack02_4c69a2819ab8/detector_{kind}.pt", map_location="cpu", weights_only=True)["best_model"])
        dets[kind] = m.eval().requires_grad_(False)
    return dets

def far_scores(cfg, w, batch, dets):                 # [R, T] score of each (anchor, j) target
    if cfg.far == "surprise_far":
        local = dets["window"].scores(batch); B, T = batch["x"].shape
        seq = torch.arange(B).repeat_interleave(T)[:, None]
        tgt = (torch.arange(T)[None, :] + torch.arange(T).repeat(B)[:, None]).clamp(max=T - 1)
        return local[seq, tgt]
    if cfg.far == "ctx_far":
        return torch.cat([F.cross_entropy(dets["context"](xf).flatten(0, 1), yf.clamp(min=0).flatten(), reduction="none").reshape(yf.shape) * yf.ne(IGNORE)
                          for xf, yf in zip(w["x_fut"].split(4096), w["y_fut"].split(4096))])
    return None                                      # random_far: all scores 0 -> random tie breaking

def topk_mask(scores, pool, K, generator):           # K highest scores in the pool; ties broken by a seeded permutation
    s = torch.where(pool, scores.float(), torch.full_like(scores, -torch.inf, dtype=torch.float32))
    perm = torch.rand(s.shape, generator=generator).argsort(1)
    order = torch.sort(s.gather(1, perm), dim=1, descending=True, stable=True).indices
    return torch.zeros_like(pool).scatter(1, perm.gather(1, order)[:, :K], True) & pool

dets = load_detectors()
far_pool = ws["valid"] & (torch.arange(64) >= STACK.T_f)                               # [R, T]
with torch.no_grad(): sc = far_scores(replace(STACK, far="ctx_far"), ws, bs, dets)
chosen = topk_mask(sc, far_pool, STACK.K_far, torch.Generator().manual_seed(3))
r = int(far_pool.sum(1).argmax())
print(f"anchor row {r}: far pool j = {far_pool[r].nonzero().flatten().tolist()[:12]}...; chosen (ctx_far, K_far=2): {chosen[r].nonzero().flatten().tolist()}")
assert (chosen.sum(1) <= STACK.K_far).all() and (chosen <= far_pool).all()
''')

md("s7", "## §7. Функции потерь — **[адаптация]** авторской SMT/DMT; идентичны коду GPU-запусков (проверка в конце)")
code("s7-smt", r'''
def smt_loss(teacher, rnn, batch, cfg, generator=None, dets=None):
    B, T = batch["x"].shape
    w = windows(batch)                                                   # each [R, T], R = B*T
    memory = torch.cat([teacher.encode(p) for p in w["x_ctx"].split(4096)]).reshape(B, T, -1)   # m_t: [B, T, Z]
    logits = teacher.decode(memory.reshape(B * T, -1), w["x_fut"])       # [R, T, V]: all T outputs are computed
    ce = calc_cross_entropy_loss(logits, w["y_fut"].clamp(min=0))       # [R, T]; -1 targets are masked below
    j = torch.arange(T)
    near, far = w["valid"] & (j < cfg.T_f), w["valid"] & (j >= cfg.T_f)  # [R, T] each
    dec = (ce * near).sum() / near.sum().clamp(min=1)                    # L_near
    if cfg.far == "none":
        aux = dec * 0                                                    # L_far = 0
    else:
        pool = far & w["prefix"] if cfg.far == "oracle_far" else far
        if cfg.far == "uniform_far":
            chosen = pool
        else:
            with torch.no_grad(): scores = far_scores(cfg, w, batch, dets)
            chosen = topk_mask(torch.zeros_like(ce) if scores is None else scores, pool, cfg.K_far, generator)   # no gradient through the choice
        n, k = pool.sum(1).to(ce.dtype), chosen.sum(1).to(ce.dtype)      # [R], [R]
        aux = (n * ((ce * chosen).sum(1) / k.clamp(min=1))).sum() / n.sum().clamp(min=1)   # L_far: pool-size-weighted mean
    prev = torch.cat([torch.zeros_like(memory[:, :1]), memory[:, :-1]], 1)   # m_{t-1}: [B, T, Z], m_{-1} = 0
    if cfg.dyn_detach == "both": prev = prev.detach()
    pred = rnn.rnn_update(prev.reshape(B * T, -1), batch["x"].reshape(B * T, 1))   # f(m_{t-1}, x_t): [R, Z]
    target = memory.reshape(B * T, -1)
    if cfg.dyn_detach in ("target", "both"): target = target.detach()
    dyn = calc_mse_loss(pred, target).mean()                             # L_dyn over R*Z numbers
    unif = calc_uniformity_loss(memory.reshape(B * T * cfg.K_mem, cfg.d))    # L_unif over [R*K_mem, d] tokens
    loss = cfg.c_dec * (dec + cfg.alpha_far * aux) + cfg.c_dyn * dyn + cfg.c_unif * unif
    return loss, dict(loss=loss, L_near=dec, L_far=aux, L_dyn=dyn, L_unif=unif)

small = replace(STACK, T=16, max_stack_size=4)
sb = sample_stack(small, 4, 7); out = []
for far in ("none", "ctx_far", "surprise_far", "random_far"):
    t_, r_ = make_models(replace(small, far=far), 1)
    with seeded(5): loss, parts = smt_loss(t_, r_, sb, replace(small, far=far), torch.Generator().manual_seed(3), dets)
    loss.backward()
    g = lambda m, pref: sum(float(p.grad.abs().sum()) for n_, p in m.named_parameters() if n_.startswith(pref) and p.grad is not None)
    out.append(dict(far=far, **{k: round(float(v.detach()), 4) for k, v in parts.items()}, grad_encoder=g(t_, "encoder"), grad_updater=g(r_, "rnn."),
                    grad_rnn_readout=g(r_, "decoder") + g(r_, "to_logits")))
    if far == "none": assert float(parts["L_far"]) == 0.0
display(pd.DataFrame(out))
assert all(o["grad_rnn_readout"] == 0 and o["grad_updater"] > 0 and o["grad_encoder"] > 0 for o in out)
print("SMT: the RNN readout gets no gradient (it is a copy of the teacher's readout); the updater learns only from L_dyn")
''')
code("s7-dmt-bptt", r'''
def dmt_loss(teacher, rnn, batch, cfg):
    B, T = batch["x"].shape
    with torch.no_grad():
        target = teacher.encode(windows(batch)["x_ctx"]).reshape(B, T, -1)      # m_t of the FROZEN encoder: [B, T, Z]
    states = rnn_rollout(rnn, batch["x"], detach=True)                        # own states m^_t: [B, T, Z]; input detached
    dyn = (states - target).square().mean()                                   # MSE -> updater (one step deep)
    logits = rnn.rnn_readout(states.detach().flatten(0, 1)).reshape(B, T, -1)   # sg[m^_t] -> readout only
    ce = calc_cross_entropy_loss(logits, batch["y"].clamp(min=0)); mask = batch["target_mask"]
    readout = (ce * mask).sum() / mask.sum().clamp(min=1)
    return cfg.c_dmt_mse * dyn + readout, dict(MSE=dyn, CE=readout)

def bptt_loss(rnn, batch, truncate):                 # truncate=False: full BPTT; True: BPTT-1 (state kept, gradient cut)
    states = rnn_rollout(rnn, batch["x"], detach=truncate)
    ce = calc_cross_entropy_loss(rnn.rnn_readout(states.flatten(0, 1)).reshape(*batch["x"].shape, -1), batch["y"].clamp(min=0))
    mask = batch["target_mask"]; return (ce * mask).sum() / mask.sum().clamp(min=1), {}

t_, r_ = make_models(small, 3); t_.requires_grad_(False)
with torch.no_grad():
    assert torch.equal(rnn_rollout(r_, sb["x"]), rnn_rollout(r_, sb["x"], detach=True)), "detach does not change the state values"
def grads_of(loss_fn):
    _, r2 = make_models(small, 3); loss_fn(r2).backward()
    return {k: sum(float(p.grad.abs().sum()) for n_, p in r2.named_parameters() if n_.startswith(k) and p.grad is not None) for k in ("embed", "rnn.", "decoder", "to_logits")}
g_ce = grads_of(lambda r2: dmt_loss(t_, r2, sb, replace(small, c_dmt_mse=0.0))[0])
g_mse = grads_of(lambda r2: dmt_loss(t_, r2, sb, small)[1]["MSE"])
display(pd.DataFrame([dict(loss="DMT: CE only", **g_ce), dict(loss="DMT: MSE only", **g_mse)]))
assert g_ce["rnn."] == 0 and g_ce["decoder"] > 0 and g_mse["rnn."] > 0 and g_mse["decoder"] == 0
for trunc in (False, True):                          # does the loss at the last step reach m_{-1}?
    init = torch.zeros(4, small.Z, requires_grad=True)
    F.cross_entropy(r_.rnn_readout(rnn_rollout(r_, sb["x"], init=init, detach=trunc)[:, -1]), torch.ones(4, dtype=torch.long)).backward()
    reaches = init.grad is not None and bool(init.grad.abs().sum() > 0)
    print(f"{'BPTT-1   ' if trunc else 'full BPTT'}: gradient of the loss at t = T-1 reaches m_(-1): {reaches}"); assert reaches == (not trunc)
''')

md("s8", "## §8. Три режима на одном примере: SMT, DMT, инференс")
code("s8-trace", r'''
t_, r_ = make_models(STACK, 201); x6 = ex["x"]                     # the hand example x = [1, 2, 0, 3, 0, 0]
with torch.no_grad():
    m_enc = t_.encode(windows(ex)["x_ctx"])                          # [6, Z]: m_0..m_5 of the encoder
    t = 3
    smt_pred = r_.rnn_update(m_enc[t - 1:t], x6[:, t:t + 1])         # SMT:  f(m_{t-1} of the ENCODER, x_t)
    m_hat = rnn_rollout(r_, x6)[0]                                   # DMT / inference: own states m^_0..m^_5
    dmt_pred = r_.rnn_update(m_hat[t - 1:t], x6[:, t:t + 1])         # DMT:  f(m^_{t-1} of the RNN, x_t)
    print(f"t={t}: SMT compares f(m_{t-1}[encoder], x_{t}) with m_{t}[encoder]: MSE {float((smt_pred - m_enc[t]).pow(2).mean()):.3f}")
    print(f"t={t}: DMT compares f(m^_{t-1}[RNN],     x_{t}) with m_{t}[frozen encoder]: MSE {float((dmt_pred - m_enc[t]).pow(2).mean()):.3f}")
    assert torch.allclose(dmt_pred[0], m_hat[t], atol=1e-6)
    t_.encode = lambda *a: (_ for _ in ()).throw(RuntimeError("inference must not call the encoder"))   # prove it below
    pops = ex["target_mask"][0].nonzero().flatten()
    pred = r_.rnn_readout(rnn_rollout(r_, x6)[0, pops]).argmax(-1)                                    # only f and readout
print("inference: answers at pops", pops.tolist(), "->", pred.tolist(), "(untrained weights; uses only x, f and the RNN readout)")
del t_
''')

md("s9", "## §9. Метрики и мини-обучение (SMOKE, CPU)")
code("s9-metrics", r'''
@torch.no_grad()
def answers(kind, model, batch, cfg, anchor="query"):
    """kind: rnn (final model), teacher (j = 0 at the answer), teacher_pre (memory of anchor 2N-1, after all pairs)."""
    b, t = torch.where(batch["target_mask"]); y = batch["y"][b, t]; T = batch["x"].shape[1]
    if kind == "rnn":
        lg = model.rnn_readout(rnn_rollout(model, batch["x"])[b, t])
    else:
        w = windows(batch); a = torch.full_like(t, 2 * cfg.N - 1) if kind == "teacher_pre" else t
        rows, j = b * T + a, t - a
        lg = model.decode(model.encode(w["x_ctx"][rows]), w["x_fut"][rows])[torch.arange(len(j)), j]
        assert torch.equal(w["y_fut"][rows, j], y)
    res = dict(correct=(lg.argmax(-1) == y), long=batch["delay"][b, t] >= cfg.long_delay)
    if cfg.task == "mqar":                           # diagnostic: choice among the N values written in this sequence
        written = batch["x"][:, 1:2 * cfg.N:2][b]
        res["cand_correct"] = lg.gather(1, written).argmax(1) == (written == y[:, None]).float().argmax(1)
    return {k: v.float() for k, v in res.items()}

def metrics(r):
    out = dict(acc_long_full_vocab=float(r["correct"][r["long"].bool()].mean()))
    if "cand_correct" in r: out["acc_long_among_written"] = float(r["cand_correct"][r["long"].bool()].mean())
    return out

def train(params, steps, lr, step_fn):               # [наше] AdamW, warm-up then constant LR, clip 1.0 (as on the GPU)
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.01); warm = max(1, steps // 10); log = []
    for s in range(steps):
        for g_ in opt.param_groups: g_["lr"] = lr * min(1.0, (s + 1) / warm)
        opt.zero_grad(); loss = step_fn(s); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); log.append(float(loss))
    return log

mock = replace(MQAR_T16, d=32, heads=2, B=16)        # tiny MQAR: numbers are meaningless, the pipeline is real
tm, rm = make_models(mock, 0); val = sample_mqar(mock, 64, derive_seed(511, 0))
tic = time.time()
smt_log = train([*tm.parameters(), *rm.parameters()], 30, 1e-3, lambda s: smt_loss(tm, rm, sample_mqar(mock, mock.B, derive_seed(301, 0, s)), mock)[0])
copy_readout(tm, rm); tm.requires_grad_(False)
dmt_log = train(list(rm.parameters()), 20, 1e-3, lambda s: dmt_loss(tm, rm, sample_mqar(mock, mock.B, derive_seed(302, 0, s)), mock)[0])
_, rb = make_models(mock, 0)
bptt_log = train(list(rb.parameters()), 20, 1e-3, lambda s: bptt_loss(rb, sample_mqar(mock, mock.B, derive_seed(301, 0, s)), False)[0])
display(pd.DataFrame([dict(stage="SMT (30 steps)", first=smt_log[0], last=smt_log[-1], **metrics(answers("teacher", tm, val, mock))),
                      dict(stage="teacher_pre: memory after pairs", first=None, last=None, **metrics(answers("teacher_pre", tm, val, mock))),
                      dict(stage="DMT (20 steps) -> final RNN", first=dmt_log[0], last=dmt_log[-1], **metrics(answers("rnn", rm, val, mock))),
                      dict(stage="full BPTT (20 steps)", first=bptt_log[0], last=bptt_log[-1], **metrics(answers("rnn", rb, val, mock)))]))
assert all(np.isfinite(smt_log + dmt_log + bptt_log)); print(f"SMOKE training: {time.time() - tic:.0f} s on CPU")
''')

md("s10", r"""
## §10. Сохранённые результаты (только чтение)

**Новые запуски:** RTX 3090, seeds 201–203. Числа — независимый набор (поток 901), чекпойнт выбран по validation.
Запуск без `summary.json` показан как «не завершён»: это не ноль. Старые архивные результаты получены по другому
протоколу и в одну таблицу с новыми не сводятся.
""")
code("s10-results", r'''
NAMES = {"stack_smt_tf8": "SMT: короткое окно, без дальних целей", "stack_smt_tf8_ctx_far": "SMT + дальние цели по ошибке без памяти",
         "stack_smt_tf8_surprise_far": "SMT + дальние цели по локальной неожиданности", "stack_smt_tf8_random_far": "SMT + случайные дальние цели",
         "stack_smt_tf8_no_dyn": "SMT без L_dyn", "stack_smt_tf8_target_detached": "SMT, цель L_dyn отцеплена",
         "stack_smt_tf8_fully_detached": "SMT, L_dyn полностью отцеплен", "stack_bptt_full": "RNN, полный BPTT",
         "stack_bptt_1": "RNN, градиент на один переход (BPTT-1)", "mqar_T16N2_smt": "MQAR 16/2: SMT → DMT",
         "mqar_T16N2_teacher_bottleneck": "MQAR 16/2: teacher 8 токенов, только L_dec",
         "mqar_T16N2_teacher_full": "MQAR 16/2: диагностика, декодер читает все выходы энкодера",
         "mqar_T16N2_transformer": "MQAR 16/2: causal Transformer", "mqar_T16N2_rnn_bptt_full": "MQAR 16/2: RNN, полный BPTT (6000 шагов)",
         "mqar_T16N2_rnn_bptt_full_long": "MQAR 16/2: RNN, полный BPTT (до 60000 шагов)", "mqar_T32N4_smt": "MQAR 32/4: SMT → DMT",
         "mqar_T32N4_transformer": "MQAR 32/4: causal Transformer", "mqar_T32N4_rnn_bptt_full_long": "MQAR 32/4: RNN, полный BPTT (до 60000 шагов)"}
KEYS = [("итоговая RNN", ("eval_best_rnn_dmt_long_acc", "eval_best_rnn_bptt_long_acc")), ("teacher j=0", ("eval_best_teacher_smt_long_acc", "eval_best_teacher_full_smt_long_acc")),
        ("teacher до запросов", ("eval_best_teacher_pre_after_pairs_smt_long_acc",)), ("Transformer", ("eval_best_lm_long_acc",))]
rows = {}
for d in sorted((ROOT / "results_3090" / "notebook_runs").glob("*_s20*_*")):
    name, seed = d.name.split("_s20")[0], 200 + int(d.name.split("_s20")[1][0])
    s = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else None
    if s or (name, seed) not in rows:
        rows[(name, seed)] = dict(model=NAMES.get(name, name), seed=seed, finished=s is not None,
                                  **{lab: (round(100 * next(s[k] for k in ks if k in s), 1) if s and any(k in s for k in ks) else None) for lab, ks in KEYS})
display(pd.DataFrame(rows.values()).sort_values(["model", "seed"]).reset_index(drop=True))
display(pd.read_csv(ROOT / "archive/outputs_ablation/table_ablation.csv")[["name", "what", "teacher", "far_probe", "rnn_after_dmt"]]
        .rename(columns={"what": "СТАРЫЙ протокол: абляции, seed 107, validation, SMT 11k / DMT 5k"}))
''')

md("s11", r"""
## §11. Необязательно: сверка с кодом GPU-запусков

Ячейка исполняет определения `SMT_MemUP_FROM_ZERO.ipynb` (этим кодом обучены все запуски `results_3090/`). Затем
сравнивает с функциями этого ноутбука на одинаковых весах:
* данные и окна;
* все компоненты SMT-loss и градиенты для шести правил дальних целей;
* DMT и оба BPTT.

Длится около минуты.
""")
code("s11-optional", r'''
RUN_OPTIONAL_CHECK = True
if RUN_OPTIONAL_CHECK:
    import explainable_extras
    display(explainable_extras.check_against_v1(globals()))
''')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--execute", action="store_true"); a = ap.parse_args()
    nb = nbf.v4.new_notebook(cells=CELLS, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    out = ROOT / "SMT_MemUP_EXPLAINABLE.ipynb"
    nbf.write(nb, out)
    n_code = sum(c.cell_type == "code" for c in CELLS)
    print(f"wrote {out.name}: {len(CELLS)} cells ({n_code} code)")
    if a.execute:
        from nbconvert.preprocessors import ExecutePreprocessor
        ExecutePreprocessor(timeout=1800, kernel_name="python3").preprocess(nb, {"metadata": {"path": str(ROOT)}})
        nbf.write(nb, out); print("executed and saved with outputs")


if __name__ == "__main__":
    main()
