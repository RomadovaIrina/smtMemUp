#!/usr/bin/env python3
"""Builds two task notebooks from the same computational cells as SMT_MemUP_FROM_ZERO(_v2):
    smtMemUp_stack.ipynb   - Stack only
    smtMemUpMQAR.ipynb     - MQAR only
Each has: definitions, mathematical formulation, configs, worked examples (hand-made tables checked by asserts),
windows, models (code, schema, dims, params), losses, checks, training, evaluation, traces, equivalence with the old
code, FULL runs, archived results. Rendering lives in viz_utils.py.
    python build_notebooks_split.py --execute"""
import argparse
import re
import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import build_notebook as V1  # noqa: E402
import build_notebook_v2 as V2  # noqa: E402

SRC = {c.metadata["tags"][0]: c.source for c in V1.CELLS if c.cell_type == "code"}
MD2 = {c.metadata["tags"][0]: c.source for c in V2.CELLS if c.cell_type == "markdown"}
MD1 = V2.MD1


def lines(tag, start=None, stop=None):
    """Lines [start, stop) of a version-1 code cell (0-based, as printed by the inspection in the commit log)."""
    return "\n".join(SRC[tag].splitlines()[start:stop])


def sections(text, keep_prefixes):
    """Keep only '### ' blocks whose heading starts with one of keep_prefixes (text before the first '### ' is kept)."""
    parts = re.split(r"(?m)^(?=### )", text)
    return "".join(p for p in parts if not p.startswith("### ") or p[4:].startswith(tuple(keep_prefixes)))


def rows_without(text, words):
    return "\n".join(l for l in text.splitlines() if not (l.startswith("|") and any(w in l.split("|")[1] for w in words)))


class NB:
    def __init__(self):
        self.cells = []

    def md(self, tag, text):
        self.cells.append(nbf.v4.new_markdown_cell(text.strip("\n"), metadata={"tags": [tag]}))

    def code(self, tag, text):
        assert "display(" not in text and "plt." not in text, f"rendering code in {tag}"
        self.cells.append(nbf.v4.new_code_cell(text.strip("\n"), metadata={"tags": [tag]}))


SETUP = SRC["s0-setup"].replace("import matplotlib.pyplot as plt\n", "import viz_utils as vz          # tables and figures only\n") \
                       .replace('pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.max_rows", 200)\n', "")
EQ = SRC["s12-equivalence"].splitlines()
EQ_HEADER = "\n".join(EQ[:18])
cut = lambda a, b: "\n".join(EQ[a:b])
ANNOTATE = SRC["s2-stack"][SRC["s2-stack"].index("def annotate"):SRC["s2-stack"].index("def sample_stack")].rstrip()


def common_head(nb, task_title, task_word):
    nb.md("title", f"""
# SMT × MemUP: {task_title}

* **Один ноутбук — одна задача.** Код данных, моделей, лоссов, обучения и оценки — внутри ноутбука; таблицы и графики
  рисует `viz_utils.py`.
* **Происхождение кода.** Тот же вычислительный код, что в `SMT_MemUP_FROM_ZERO.ipynb` (версия 1). Со старой реализацией
  он сверяется в §15.
* **Режимы:**
  * `SMOKE` (по умолчанию) — CPU, проверки и примеры;
  * `FULL` — обучает один выбранный эксперимент (§16) в новый каталог.
* **Пометки:** [история] — так было в архивных GPU-запусках; [новое] — добавлено 21–25.10.2026.
""")
    nb.md("s0-setup", "## §0. Зависимости, режим, seed")
    nb.code("s0-setup", SETUP)
    nb.md("s0-start", MD1["s0-start"])
    nb.code("s0-seeds", SRC["s0-seeds"])


# =====================================================================================================================
# Worked examples
# =====================================================================================================================
STACK_EXAMPLE_MD = r"""
## §5. Разобранные примеры

### 5.1 Одна короткая последовательность Stack, вручную

**Определение.** Строка таблицы — один шаг t: вход-команда `x_t`, стек **после** её выполнения (снизу вверх) и
правильный ответ `y_t`. Ответ есть только у pop: это снятый элемент. У push ответа нет, `y_t = −1`.

Последовательность команд: `push 1, push 2, pop, push 3, pop, pop`.

| позиция t | вход x_t | действие | стек после команды | метка y_t | маска (в loss?) | источник | delay |
|---|---|---|---|---|---|---|---|
| 0 | 1 | push 1 | `[1]` | −1 | 0 | — | — |
| 1 | 2 | push 2 | `[1, 2]` | −1 | 0 | — | — |
| 2 | 0 | pop | `[1]` | 2 | 1 | 1 | 1 |
| 3 | 3 | push 3 | `[1, 3]` | −1 | 0 | — | — |
| 4 | 0 | pop | `[1]` | 3 | 1 | 3 | 1 |
| 5 | 0 | pop | `[]` | 1 | 1 | 0 | 5 |

**Тензоры:**
* `x = [1, 2, 0, 3, 0, 0]`, `y = [−1, −1, 2, −1, 3, 1]`;
* форма обоих для **одной** последовательности — `[1, 6]`; для архивного батча из 32 последовательностей длины 64 —
  `[32, 64]`;
* маска `target_mask = y ≠ −1 = [0, 0, 1, 0, 1, 1]`: в loss и метрики входят только позиции 2, 4 и 5.

**Столбец «стек» и разметка.** Стек нужен только для проверки разметки, в сеть отдельным входом он не поступает. Сеть
получает **только `x`**: на позиции 2 она видит `x_0..x_2 = 1, 2, 0` и должна сама «знать», что на вершине 2.
`source` и `delay` — тоже метаданные: они нужны только метрике «длинные ответы».

**Ячейка ниже проверяет:**
* `y` выводится из `x` независимой симуляцией стека;
* такая же раскладка встречается у настоящего генератора (стек, push / pop, ответ на позиции pop);
* формы на каждом шаге модели для этого примера.
"""

STACK_EXAMPLE_CODE = r'''
x_ex = torch.tensor([[1, 2, 0, 3, 0, 0]])                          # [1,6]  one sequence, 6 positions
y_ex = torch.tensor([[-1, -1, 2, -1, 3, 1]])                       # [1,6]  answers (-1 = no answer)
st, y_sim, src_sim = [], [], []                                    # independent stack simulation from x alone
for t, tok in enumerate(x_ex[0].tolist()):
    if tok > 0: st.append((tok, t)); y_sim.append(-1); src_sim.append(-1)
    else: v, s = st.pop(); y_sim.append(v); src_sim.append(s)
assert y_sim == y_ex[0].tolist()
ex = annotate(x_ex, y_ex, torch.tensor([src_sim]))
assert ex["target_mask"][0].int().tolist() == [0, 0, 1, 0, 1, 1] and ex["delay"][0].tolist() == [-1, -1, 1, -1, 1, 5]
vz.show(vz.stack_table(ex), title="the hand example, rebuilt by code")
print("x", tuple(ex["x"].shape), "| y", tuple(ex["y"].shape), "| archived batch:", tuple(sample(STACK, 32, derive_seed(301, 107, 0))["x"].shape))

# the same example through windows, the model and every loss term (Tf = 2 so that both near and far targets exist)
cfg_ex = replace(STACK, seq_len=6, future_window=2)
w = windows(ex)                                                    # every tensor [R=1*6, 6]
print("x_ctx", tuple(w["x_ctx"].shape), "| anchor 2:", w["x_ctx"][2].tolist(), "| x_fut[2]:", w["x_fut"][2].tolist(),
      "| y_fut[2]:", w["y_fut"][2].tolist())
assert w["x_ctx"][2].tolist() == [-1, -1, -1, 1, 2, 0] and w["y_fut"][2].tolist() == [2, -1, 3, 1, -1, -1]
teacher_ex, rnn_ex = make_models(cfg_ex, 0)
with torch.no_grad(), seeded(1):
    m = encode_all(teacher_ex, w["x_ctx"])                         # [6,6] -> [6,512]
    logits = teacher_ex.decode(m, w["x_fut"])                      # [6,512],[6,6] -> [6,6,5]
    ce = calc_cross_entropy_loss(logits, w["y_fut"].clamp(min=0))  # [6,6]
    j = torch.arange(6); near = w["valid"] & (j < 2); far = w["valid"] & (j >= 2)
    loss, parts = smt_loss(teacher_ex, rnn_ex, ex, cfg_ex)
    states = rnn_rollout(rnn_ex, ex["x"])                          # [1,6] -> [1,6,512]
    rl = rnn_ex.rnn_readout(states.flatten(0, 1))                  # [6,512] -> [6,5]
print("m", tuple(m.shape), "| logits", tuple(logits.shape), "| CE", tuple(ce.shape),
      f"| near pairs {int(near.sum())}, far pairs {int(far.sum())} (Tf=2)")
assert abs(float((ce * near).sum() / near.sum()) - float(parts["decoder"])) < 1e-6, "L_dec = mean CE over near pairs"
print(f"L_dec={float(parts['decoder']):.4f}  L_dyn={float(parts['dynamics']):.4f}  L_unif={float(parts['uniformity']):.4f}  "
      f"L={float(loss):.4f} (scalar)")
print("RNN states", tuple(states.shape), "| readout logits", tuple(rl.shape), "| answers read at positions",
      ex["target_mask"][0].nonzero().flatten().tolist())
'''

MQAR_EXAMPLE_MD = r"""
## §5. Разобранные примеры

### 5.1 Одна короткая последовательность MQAR, вручную (T = 16, N = 2, V = 128)

**Определения:**
* **пара** — два соседних ID: ключ (`1..63`), затем значение (`64..127`);
* **запрос** — повтор ключа на позиции `≥ 2N`;
* **ответ** — значение, записанное в паре с этим ключом;
* остальные позиции после пар — заполнитель `0`.

Столбец «записанные пары после шага» показывает, какую связку «ключ → значение» уже **можно** знать после чтения
`x_0..x_t`. Это только проверка разметки, в сеть он не подаётся.

| позиция t | вход x_t | роль | записанные пары после шага | метка y_t | маска | источник | delay |
|---|---|---|---|---|---|---|---|
| 0 | 16 | ключ пары 0 | `{}` (ключ без значения) | −1 | 0 | — | — |
| 1 | 97 | значение пары 0 | `{16→97}` | −1 | 0 | — | — |
| 2 | 5 | ключ пары 1 | `{16→97}` | −1 | 0 | — | — |
| 3 | 121 | значение пары 1 | `{16→97, 5→121}` | −1 | 0 | — | — |
| 4 | 16 | **запрос** ключа 16 | без изменений | **97** | 1 | 1 | 3 |
| 5–7 | 0 | заполнитель | без изменений | −1 | 0 | — | — |
| 8 | 5 | **запрос** ключа 5 | без изменений | **121** | 1 | 3 | 5 |
| 9–15 | 0 | заполнитель | без изменений | −1 | 0 | — | — |

**Тензоры:**
* `x = [16, 97, 5, 121, 16, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]`;
* `y = [−1, −1, −1, −1, 97, −1, −1, −1, 121, −1, −1, −1, −1, −1, −1, −1]`;
* форма обоих для **одной** последовательности — `[1, 16]`; батч архивных запусков T = 16 — `[32, 16]`; pilot01
  (T = 64, N = 8) — `[32, 64]`.

**Маска и метрики:**
* маска: только позиции запросов 4 и 8;
* «длинный» ответ при T = 16 — `delay ≥ 4`: первый запрос (delay 3) в метрику длинных ответов не входит, второй (5)
  входит.

**Эта последовательность — настоящая:** validation-последовательность 0 (поток 511) генератора Zoology. Ячейка ниже это
проверяет.

### 5.2 Что видит модель при ответе на запрос t = 8

| режим | память построена по | вход декодера | выход |
|---|---|---|---|
| j = 0 на позиции запроса [история] | `x_0..x_8` — пары **и ключ запроса** | — | j = 0 → 121 |
| память после всех пар [новое] | `x_0..x_3` — только пары | `x_4..x_8 = 16, 0, 0, 0, 5` | j = 5 → 121 |
| память за шаг до запроса [новое] | `x_0..x_7` | `x_8 = 5` | j = 1 → 121 |
| итоговая RNN | читает `x_0..x_8` по одному ID; `m_8 = f(m_7, x_8)` | — | readout(m_8) → 121 |
"""

MQAR_EXAMPLE_CODE = r'''
x_ex = torch.tensor([[16, 97, 5, 121, 16, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0]])                  # [1,16]
y_ex = torch.tensor([[-1, -1, -1, -1, 97, -1, -1, -1, 121, -1, -1, -1, -1, -1, -1, -1]])      # [1,16]
val16 = validation_set(MQAR_T16)
assert torch.equal(val16["x"][:1], x_ex) and torch.equal(val16["y"][:1], y_ex), "the hand example is validation sequence 0"
pairs = dict(zip(x_ex[0, 0:4:2].tolist(), x_ex[0, 1:4:2].tolist()))                          # rule: answer = pairs[key]
assert [pairs.get(k, -1) if t >= 4 and k else -1 for t, k in enumerate(x_ex[0].tolist())] == y_ex[0].tolist()
ex = {k: v[:1] for k, v in val16.items()}
vz.show(vz.mqar_table(ex, 2), title="the hand example, rebuilt by code")
print("x", tuple(ex["x"].shape), "| y", tuple(ex["y"].shape), "| archived T=16 batch:", tuple(sample(MQAR_T16, 32, derive_seed(301, 107, 0))["x"].shape),
      "| pilot01 batch:", tuple(sample(MQAR_PILOT01, 32, derive_seed(301, 107, 0))["x"].shape))

w = windows(ex)                                                                               # [R=16, 16] each
for name, a, j in (("j=0 at the query", 8, 0), ("memory after the pairs", 3, 5), ("one step before", 7, 1)):
    print(f"{name}: x_ctx[{a}] = {[v for v in w['x_ctx'][a].tolist() if v != IGNORE]}, decoder inputs = "
          f"{w['x_fut'][a, :j].tolist()}, target y_fut[{a},{j}] = {int(w['y_fut'][a, j])}")
    assert int(w["y_fut"][a, j]) == 121
teacher_ex, rnn_ex = make_models(MQAR_T16, 0)
with torch.no_grad(), seeded(1):
    m = encode_all(teacher_ex, w["x_ctx"])                       # [16,16] -> [16,512]
    logits = teacher_ex.decode(m, w["x_fut"])                    # -> [16,16,128]
    ce = calc_cross_entropy_loss(logits, w["y_fut"].clamp(min=0))  # [16,16]
    loss, parts = smt_loss(teacher_ex, rnn_ex, ex, MQAR_T16)
    states = rnn_rollout(rnn_ex, ex["x"])                        # [1,16,512]
print("m", tuple(m.shape), "| logits", tuple(logits.shape), "| CE", tuple(ce.shape), "| valid pairs", int(w["valid"].sum()),
      "(each query is the target of every anchor a <= t: 5 + 9 = 14)")
assert int(w["valid"].sum()) == 14
print(f"L_dec={float(parts['decoder']):.4f} (untrained; a uniform guess over 128 IDs would give ln 128 = {math.log(128):.2f})  L={float(loss):.4f}; RNN states", tuple(states.shape))
'''


# =====================================================================================================================
def build_stack():
    nb = NB()
    common_head(nb, "Stack", "Stack")
    nb.md("s1-definitions", rows_without(MD2["s1-definitions"], ["MQAR"]))
    nb.md("s2-math", sections(MD2["s2-math"], ["2.1", "2.3"]))
    nb.md("s3-config", rows_without(MD2["s3-config"], ["MQAR", "тест узкого", "Zoology"]))
    nb.code("s1-config", SRC["s1-config"])
    nb.md("s4-data", "## §4. Данные: генератор Stack\n\nКопия авторского генератора. Определения полей — §1; модели получают только `x`.")
    nb.code("s2-stack", SRC["s2-stack"])
    nb.code("s4-streams", '''
sample = sample_stack
# data streams (as archived): train batch of step s = derive_seed(301, run_seed, s) (SMT/BPTT), 302 (DMT);
# validation (checkpoint selection) = stream 501; independent evaluation of NEW runs = stream 901.
validation_set = lambda cfg: sample(cfg, cfg.validation_examples, derive_seed(501, 0))
evaluation_set = lambda cfg: sample(cfg, cfg.eval_examples, derive_seed(901, 0))
stack_val = validation_set(STACK)
assert (stack_val["delay"][stack_val["target_mask"]] % 2 == 1).all(), "Stack delays are odd"
print("answers per sequence", float(stack_val["target_mask"].sum(1).float().mean()), "| share of long answers (delay >= 8)",
      float((stack_val["delay"][stack_val["target_mask"]] >= 8).float().mean()))
vz.show(vz.stack_table(stack_val).head(34), title="real generator: validation sequence 0")
''')
    nb.md("s5-examples", STACK_EXAMPLE_MD)
    return nb, "stack"


def build_mqar():
    nb = NB()
    common_head(nb, "MQAR", "MQAR")
    nb.md("s1-definitions", sections(MD2["s1-definitions"], ["Данные", "SMT", "Оценка"]))
    nb.md("s2-math", sections(MD2["s2-math"], ["2.2", "2.3"]))
    nb.md("s3-config", rows_without(MD2["s3-config"], ["Stack", "абляции"]))
    nb.code("s1-config", SRC["s1-config"])
    nb.md("s4-data", "## §4. Данные: генератор MQAR\n\nКопия генератора Zoology. Определения полей — §1; модели получают только `x`.")
    nb.code("s2-mqar", ANNOTATE + "\n\n" + lines("s2-mqar", 0, 32) + '''

sample = sample_mqar
# data streams (as archived): train batch of step s = derive_seed(301, run_seed, s) (SMT/BPTT/LM), 302 (DMT);
# validation (checkpoint selection) = stream 511; independent evaluation of NEW runs = stream 901.
validation_set = lambda cfg: sample(cfg, cfg.validation_examples, derive_seed(511, 0))
evaluation_set = lambda cfg: sample(cfg, cfg.eval_examples, derive_seed(901, 0))
pilot_val = validation_set(MQAR_PILOT01)
vz.show(vz.mqar_table(pilot_val, MQAR_PILOT01.num_kv_pairs).head(34), title="real generator: pilot01 (V=128, T=64, N=8), validation sequence 0")
''')
    nb.md("s5-examples", MQAR_EXAMPLE_MD)
    return nb, "mqar"


def body(nb, task):
    is_stack = task == "stack"
    example = STACK_EXAMPLE_CODE if is_stack else MQAR_EXAMPLE_CODE
    split_at = "# the same example through" if is_stack else "w = windows(ex)"
    example_data, example_model = example[:example.index(split_at)], example[example.index(split_at):]
    nb.code("s5-example", example_data)
    # windows
    nb.md("s6-windows", MD1["s3-windows"].replace("## §5.", "## §6."))
    nb.code("s3-windows", SRC["s3-windows"].replace(
        SRC["s3-windows"][SRC["s3-windows"].index("def show_window"):SRC["s3-windows"].index("tiny = annotate")],
        "def show_window(batch, t, b=0, js=range(0, 6), Tf=None):   # rendering in viz_utils.window_table\n"
        "    return vz.window_table(windows(batch), b * batch[\"x\"].shape[1] + t, t, js, Tf)\n\n"))
    if is_stack:
        nb.md("s6-stack-far", MD1["s3-stack-far"].replace("### §5.1", "### §6.1"))
        nb.code("s3-stack-far", SRC["s3-stack-far"])
    else:
        nb.md("s6-mqar-views", MD1["s3-mqar-views"].replace("### §5.2", "### §6.1"))
        nb.code("s3-mqar-views", SRC["s3-mqar-views"])
    # models
    models_md = MD2["s6-models"].replace("## §6.", "## §7.").replace("### 6.", "### 7.")
    models_md = sections(models_md, ["7.1", "7.2", "7.3", "7.7"] if is_stack else ["7.1", "7.2", "7.4", "7.5", "7.6", "7.7"])
    nb.md("s7-models", models_md)
    nb.code("s4-model", SRC["s4-model"])
    nb.code("s4-make", SRC["s4-make"])
    nb.code("s7-params", '''
cnt = lambda m: sum(p.numel() for p in m.parameters())
CFG = STACK if TASK == "stack" else MQAR_T16
teacher, rnn = make_models(CFG, SEED); dets = load_detectors()
rows = [("teacher.embed (V x d)", cnt(teacher.embed)), ("teacher.encoder (3 blocks)", cnt(teacher.encoder)),
        ("teacher.encoder_mem_register (K x d)", cnt(teacher.encoder_mem_register)), ("teacher.decoder (2 blocks)", cnt(teacher.decoder)),
        ("teacher.to_logits (d -> V)", cnt(teacher.to_logits)), ("TEACHER total", cnt(teacher)), ("rnn.embed (own)", cnt(rnn.embed)),
        ("rnn.rnn = updater f (2 blocks)", cnt(rnn.rnn)), ("rnn readout (copy of teacher decoder + head)", cnt(rnn.decoder) + cnt(rnn.to_logits)),
        ("RNN total", cnt(rnn))]
rows += ([("WindowDetector (GRU)", cnt(dets["window"])), ("ContextDetector (GRU)", cnt(dets["context"]))] if TASK == "stack"
         else [("CausalTransformer (T=16)", cnt(make_lm(MQAR_T16, 0)))])
assert rows[5][1] == (495045 if TASK == "stack" else 510912)
vz.show(rows, ["part", f"parameters ({TASK}, V={CFG.V})"])
''')
    # losses
    nb.md("s8-losses", MD2["s7-losses"].replace("§8", "§9").replace("## §7.", "## §8."))
    nb.code("s6-losses", SRC["s6-losses"])
    if is_stack:
        nb.code("s6-smoke", SRC["s6-smoke"].replace('display(pd.DataFrame(out, columns=["loss", "value", "L_far"]))',
                                                    'vz.show(out, ["loss", "value", "L_far"])')
                .replace('mq = sample(MQAR_T16, 4, 7); t_, r_ = make_models(MQAR_T16, 1)\n', "")
                .replace('l, _ = teacher_full_access_loss(t_, mq, MQAR_T16); l.backward(); out.append(("MQAR teacher full access", float(l), None))\n', "")
                .replace('l, _ = lm_loss(make_lm(MQAR_T16, 0), mq); l.backward(); out.append(("MQAR causal Transformer", float(l), None))\n', ""))
    else:
        nb.code("s6-smoke", '''
small = MQAR_T16
b = sample(small, 4, 7); out = []
t_, r_ = make_models(small, 1)
with seeded(5): loss, parts = smt_loss(t_, r_, b, small, torch.Generator().manual_seed(3))
loss.backward(); out.append(("SMT (Tf = T = 16)", float(loss)))
t_, r_ = make_models(small, 1); t_.requires_grad_(False); out.append(("DMT", float(dmt_loss(t_, r_, b, small)[0])))
_, r_ = make_models(small, 1); l, _ = bptt_loss(r_, b, small, False); l.backward(); out.append(("RNN full BPTT", float(l)))
t_, _ = make_models(small, 1); l, _ = teacher_full_access_loss(t_, b, small); l.backward(); out.append(("teacher full access", float(l)))
l, _ = lm_loss(make_lm(small, 0), b); l.backward(); out.append(("causal Transformer", float(l)))
vz.show(out, ["loss", "value"])
''')
    nb.md("s8-example", "### 8.1 Пример из §5 через окна, модель и все лоссы\n\nТа же последовательность, что в таблице §5.1: "
                        "формы на каждом шаге и значение каждого члена лосса (модель не обучена — числа служат только проверкой путей и форм).")
    nb.code("s8-example", example_model)
    # subtle points
    nb.md("s9-subtle", MD1["s7-subtle"].replace("## §8.", "## §9."))
    if is_stack:
        for t in ("s7-encoder-prefix", "s7-j0", "s7-tf-compute"):
            nb.code(t, SRC[t])
    else:
        nb.code("s7-encoder-prefix", '''
# 9.1 the bidirectional encoder sees only the prefix: editing x after t leaves m_t unchanged, editing x_0 changes it
cfg_ = MQAR_T16; bt = sample(cfg_, 4, 11); t_, r_ = make_models(cfg_, 0)
t0 = 9; x2 = bt["x"].clone(); x2[:, t0 + 1:] = torch.randint(0, cfg_.vocab, x2[:, t0 + 1:].shape)
with torch.no_grad():
    rows = torch.arange(4) * 16 + t0
    m1 = t_.encode(windows(bt)["x_ctx"][rows]); m2 = t_.encode(windows(dict(bt, x=x2))["x_ctx"][rows])
    x3 = bt["x"].clone(); x3[:, 0] = (x3[:, 0] % 62) + 1
    m3 = t_.encode(windows(dict(bt, x=x3))["x_ctx"][rows])
print("max |m_t change| after editing x_{>t}:", float((m1 - m2).abs().max()), "| after editing x_0:", float((m1 - m3).abs().max()))
assert torch.allclose(m1, m2, atol=1e-6) and not torch.allclose(m1, m3)
''')
        nb.code("s7-j0", SRC["s7-j0"].replace("xf2[:, 4:] = torch.randint(0, 5, xf2[:, 4:].shape)",
                                               "xf2[:, 4:] = torch.randint(0, cfg_.vocab, xf2[:, 4:].shape)"))
    nb.code("s7-ldyn", SRC["s7-ldyn"].replace('gd = pd.DataFrame([enc_grad(m) for m in ("none", "target", "both")]); display(gd)',
                                              'gd = vz.show([enc_grad(m) for m in ("none", "target", "both")])'))
    for t in ("s7-dmt", "s7-bptt", "s7-labels"):
        nb.code(t, SRC[t])
    # training, evaluation, fit
    nb.md("s10-train", MD1["s8-train"].replace("## §9.", "## §10."))
    nb.code("s8-train", SRC["s8-train"])
    nb.md("s11-eval", MD1["s9-eval"].replace("## §10.", "## §11."))
    nb.code("s9-eval", lines("s9-eval", 0, 39) + ("" if is_stack else '''

t_, r_ = make_models(MQAR_T16, 0); vq = validation_set(MQAR_T16)
for kind, model, anchor in (("teacher", t_, "query"), ("teacher_pre", t_, "after_pairs"), ("teacher_pre", t_, "before_query"),
                            ("teacher_full", t_, "query"), ("rnn", r_, "query")):
    print(kind, anchor, long_metrics(answer_logits(kind, model, vq, MQAR_T16, anchor), MQAR_T16, "untrained"))
'''))
    nb.md("s12-fit", MD1["s10-fit"].replace("## §11.", "## §12."))
    tiny = ('tiny_cfg = replace(STACK, seq_len=16, max_stack_size=4, d_model=32, n_heads=2, smt_steps=6, dmt_steps=4, eval_every=2,\n'
            '                   validation_examples=16, eval_examples=16, batch_size=8, warmup_steps=2, long_delay=4)') if is_stack else \
           ('tiny_cfg = replace(MQAR_T16, d_model=32, n_heads=2, smt_steps=6, dmt_steps=4, eval_every=2,\n'
            '                   validation_examples=16, eval_examples=16, batch_size=8, warmup_steps=2)')
    nb.code("s10-fit", lines("s10-fit", 0, 66) + f'''

# SMOKE: a 6-step SMT -> 4-step DMT fit on a tiny {task} to exercise stages, selection and saving (numbers meaningless)
{tiny}
smoke_dir, smoke_summary = fit(dict(name="smoke_{task}_smt", family="smt", seed=1, config=tiny_cfg), out_root=ROOT / "results" / "smoke", log=lambda s: None)
print(smoke_dir.relative_to(ROOT), {{k: v for k, v in smoke_summary.items() if "step" in k}})
''')
    # trace and end-to-end example
    nb.md("s13-trace", MD2["s12-trace"].replace("## §12.", "## §13.").replace("B = 2, T = 8 и B = 32, T = 64", "B = 2, T = 8 и B = 32, T = 64" if is_stack else "B = 2, T = 16 и B = 32, T = 64 (pilot01)"))
    nb.code("s5-trace", lines("s5-trace", 0, 40) + ("\n\nvz.show(trace(replace(STACK, seq_len=8, max_stack_size=4), B=2))\nvz.show(trace(STACK, B=32))"
                                                     if is_stack else "\n\nvz.show(trace(MQAR_T16, B=2))\nvz.show(trace(MQAR_PILOT01, B=32))"))
    nb.md("s14-follow", MD1["s11-trace-examples"].replace("## §13.", "## §14."))
    follow_def = lines("s11-trace-examples", 0, 22)
    if is_stack:
        nb.code("s11-trace-examples", follow_def + '''

t_s, r_s = make_models(tiny_cfg, 1)
t_s.load_state_dict(torch.load(smoke_dir / "smt_best.pt", weights_only=True)["models"]["teacher"])
r_s.load_state_dict(torch.load(smoke_dir / "dmt_best.pt", weights_only=True)["models"]["rnn"])
sb = sample(tiny_cfg, 1, derive_seed(901, 0)); vz.show(vz.stack_table(sb)); follow_answer(tiny_cfg, t_s, r_s, sb)
''')
    else:
        nb.code("s11-trace-examples", follow_def + '''

t_m, r_m = make_models(tiny_cfg, 1)
t_m.load_state_dict(torch.load(smoke_dir / "smt_best.pt", weights_only=True)["models"]["teacher"])
r_m.load_state_dict(torch.load(smoke_dir / "dmt_best.pt", weights_only=True)["models"]["rnn"])
mq1 = sample(tiny_cfg, 1, derive_seed(901, 0)); vz.show(vz.mqar_table(mq1, tiny_cfg.num_kv_pairs)); follow_answer(tiny_cfg, t_m, r_m, mq1)
''')
    # equivalence
    nb.md("s15-equivalence", MD1["s12-equivalence"].replace("## §14.", "## §15."))
    if is_stack:
        nb.code("s12-equivalence", "\n".join([EQ_HEADER, cut(18, 24), cut(28, 79), cut(98, 114), "vz.show(REPORT)"]))
        nb.code("s12-fit-equivalence", SRC["s12-fit-equivalence"].replace("eq_table = pd.DataFrame(REPORT); display(eq_table.tail(3));",
                                                                          "eq_table = pd.DataFrame(REPORT); vz.show(eq_table.tail(3));"))
    else:
        nb.code("s12-equivalence", "\n".join([EQ_HEADER, cut(24, 28), '''
# weights and model functions vs the authors' original classes (MQAR sizes)
t1, r1 = make_models(MQAR_T16, 107, CPU_); t2, r2 = S.make_models(MM, 107, CPU_)
check("initial weights (teacher + RNN)", "S.make_models", diff(t1.state_dict(), t2.state_dict()) + diff(r1.state_dict(), r2.state_dict()), 0)
xs = windows(sample_mqar(MQAR_T16, 2, 9))
with torch.no_grad():
    m1, m2 = t1.encode(xs["x_ctx"]), t2.encode(xs["x_ctx"]); check("Teacher.encode", "authors' Teacher.encode", diff(m1, m2), 0)
    check("Teacher.decode", "authors' Teacher.decode", diff(t1.decode(m1, xs["x_fut"]), t2.decode(m2, xs["x_fut"])), 0)
    check("RNN.rnn_update", "authors' RNN.rnn_update", diff(r1.rnn_update(m1, xs["x_ctx"][:, -1:].clamp(min=0)), r2.rnn_update(m2, xs["x_ctx"][:, -1:].clamp(min=0))), 0)
    check("RNN.rnn_readout", "authors' RNN.rnn_readout", diff(r1.rnn_readout(m1), r2.rnn_readout(m2)), 0)
''', cut(79, 98), '''
# DMT on MQAR
bq2 = sample_mqar(MQAR_T16, 4, 31)
t1, r1 = make_models(MQAR_T16, 7, CPU_); t1.requires_grad_(False); l1 = dmt_loss(t1, r1, bq2, MQAR_T16)[0]
t2, r2 = make_models(MQAR_T16, 7, CPU_); t2.requires_grad_(False); l2 = M.dmt_loss(t2, r2, bq2, MM, CPU_)[0]
check("MQAR DMT loss (archive trims masked padding)", "M.dmt_loss", diff(l1.detach(), l2.detach()), 1e-5)
''', cut(114, 119),
                                              "vz.show(REPORT); eq_table = pd.DataFrame(REPORT)"]))
        nb.code("s12-interventions", SRC["s12-interventions"])
    # FULL
    nb.md("s16-full", MD1["s13-full"].replace("## §15.", "## §16."))
    nb.code("s13-full", SRC["s13-full"].replace('if MODE == "FULL" and FULL_EXPERIMENT:',
                                                f'EXPERIMENTS = {{k: v for k, v in EXPERIMENTS.items() if k.startswith("{task}")}}\n'
                                                'if MODE == "FULL" and FULL_EXPERIMENT:'))
    # archived results
    if is_stack:
        nb.md("s17-archive", MD1["s14-archive-results"].replace("## §16.", "## §17."))
        nb.code("s16-archive", V2_ARCHIVE_STACK)
    else:
        nb.md("s17-archive", "## §17. Архивные GPU-результаты MQAR (старый код, seed 107)\n\nЧисла прочитаны из сохранённых таблиц "
                             "`archive/outputs_mqar/` и логов; новым кодом не повторялись. «SMT teacher» везде — режим j = 0 на позиции запроса.")
        nb.code("s16-archive", '''
A = ARCHIVE / "outputs_mqar"
vz.show(pd.read_csv(A / "table_length_sweep.csv"), title="length sweep (tuning): T = 16 / 32, N = 2 / 4")
vz.show(pd.read_csv(A / "table_pairs_sweep.csv"), title="pairs sweep (tuning): pilot01 T = 64, N = 2 / 4 / 8")
vz.show(pd.read_csv(A / "table_zooB.csv"), title="Zoology attention reproduction (authors' code)")
for f in sorted((ARCHIVE / "runs/jobs").glob("bneck_*_T*.log")):
    last = [l for l in f.read_text().splitlines() if " step " in l][-1]
    print(f.name, "->", last.split(" ", 1)[1])
print("bottleneck test: one seed, log only, 128 validation sequences; full access changes 3 things at once (§7.5)")
''')
    nb.md("s18-changes", MD1["s15-changes"].replace("## §17.", "## §18."))
    nb.md("s19-summary", "## §19. Итог прогона")
    nb.code("s19-summary", f'''
print(f"equivalence checks within tolerance: {{int(eq_table.ok.sum())}}/{{len(eq_table)}}")
print("worked examples (§5), subtle-point asserts (§9), SMOKE fit (§12): passed")
print("trained with this notebook on GPU: nothing (MODE =", MODE, ", FULL_EXPERIMENT =", FULL_EXPERIMENT, ")")
''')
    return nb


V2_ARCHIVE_STACK = next(c.source for c in V2.CELLS if c.cell_type == "code" and c.metadata["tags"] == ["s16-archive"]).replace(
    '\nvz.show(pd.read_csv(ARCHIVE / "outputs_mqar/table_length_sweep.csv"), title="MQAR length sweep (archive; teacher = j=0 at the query; tuning; seed 107)")', "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true")
    a = ap.parse_args()
    for builder, fname in ((build_stack, "smtMemUp_stack.ipynb"), (build_mqar, "smtMemUpMQAR.ipynb")):
        nb, task = builder()
        nb.cells.insert(2, nbf.v4.new_code_cell(f'TASK = "{task}"', metadata={"tags": ["s0-task"]}))
        body(nb, task)
        book = nbf.v4.new_notebook(cells=nb.cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
        if a.execute:
            from nbclient import NotebookClient
            NotebookClient(book, timeout=3600, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}).execute()
        nbf.write(book, ROOT / fname)
        print(f"wrote {fname}: {len(nb.cells)} cells", flush=True)


if __name__ == "__main__":
    main()
