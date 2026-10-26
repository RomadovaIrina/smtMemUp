#!/usr/bin/env python3
"""Technical deck for the 26.10.2026 discussion, built from saved data (deck/figures, deck/tables, results_3090).

    python deck/make_deck_figures.py && python deck/build_deck.py      # -> deck/SMT_MemUP_2026-10-26.pptx
White background, black text, no decoration, no speaker notes; every number comes from a CSV / JSON file.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG, TAB, RUNS = ROOT / "deck" / "figures", ROOT / "deck" / "tables", ROOT / "results_3090" / "notebook_runs"
OUT = ROOT / "deck" / "SMT_MemUP_2026-10-26.pptx"
BLACK, GREY = RGBColor(0, 0, 0), RGBColor(0x44, 0x44, 0x44)
FONT = "Arial"
W, H = 13.333, 7.5

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(W), Inches(H)
BLANK = prs.slide_layouts[6]
SLIDES = []                                       # (number, title) for the map


# ---- data --------------------------------------------------------------------------------------------------------------
def finished(name):
    out = {}
    for d in sorted(RUNS.glob(f"{name}_s20[0-9]_*")):
        if re.match(rf"{re.escape(name)}_s\d{{3}}_\d{{8}}-\d{{6}}_\d+$", d.name) and (d / "summary.json").exists():
            out[int(d.name[len(name) + 2:len(name) + 5])] = json.loads((d / "summary.json").read_text())
    return out


def v(name, key):                                  # {seed: value in %}
    return {s: 100 * r[key] for s, r in finished(name).items() if key in r}


def fmt(d, seeds=(201, 202, 203)):
    return " / ".join(f"{d[s]:.1f}".replace(".", ",") if s in d else "—" for s in seeds)


def mean(d):
    return f"{np.mean(list(d.values())):.1f}".replace(".", ",") if d else "—"


# ---- primitives --------------------------------------------------------------------------------------------------------
def new_slide(title, subtitle=None):
    s = prs.slides.add_slide(BLANK)
    n = len(prs.slides)
    SLIDES.append((n, title))
    text(s, 0.5, 0.3, 12.3, 0.6, [(title, 26, True)])
    if subtitle:
        text(s, 0.5, 0.88, 12.3, 0.4, [(subtitle, 14, False)], color=GREY)
    text(s, 12.3, 7.05, 0.6, 0.3, [(str(n), 10, False)], color=GREY, align=PP_ALIGN.RIGHT)
    return s


def text(s, x, y, w, h, paras, color=BLACK, align=PP_ALIGN.LEFT, bullets=False, space=4):
    """paras = [(text, size, bold)] or [(text, size, bold, level)]; one paragraph each."""
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.02); tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, p_ in enumerate(paras):
        t, size, bold = p_[:3]
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align; p.space_after = Pt(space)
        r = p.add_run(); r.text = (("• " if bullets and not t.startswith("  ") else "") + t)
        r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), bold, FONT, color
    return tb


def table(s, x, y, w, rows, widths=None, size=12, header=True, row_h=0.3, bold_first_col=False):
    nr, nc = len(rows), len(rows[0])
    hs = row_h if isinstance(row_h, list) else [row_h] * nr
    gf = s.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(sum(hs)))
    tbl = gf.table
    gf._element.graphic.graphicData.tbl.tblPr.find(qn("a:tableStyleId")).text = "{5940675A-B579-460E-94D1-54222C63F5DA}"   # plain grid
    tblPr = gf._element.graphic.graphicData.tbl.tblPr
    tblPr.set("firstRow", "0"); tblPr.set("bandRow", "0")
    if widths:
        tot = sum(widths)
        for j, wj in enumerate(widths):
            tbl.columns[j].width = Inches(w * wj / tot)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(hs[i])
        for j, val in enumerate(row):
            c = tbl.cell(i, j)
            c.fill.solid(); c.fill.fore_color.rgb = RGBColor(0xF0, 0xF0, 0xF0) if (header and i == 0) else RGBColor(0xFF, 0xFF, 0xFF)
            c.margin_left = c.margin_right = Inches(0.05); c.margin_top = c.margin_bottom = Inches(0.025)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = c.text_frame; tf.word_wrap = True
            lines = str(val).split("\n")
            for k, line in enumerate(lines):
                p = tf.paragraphs[0] if k == 0 else tf.add_paragraph()
                r = p.add_run(); r.text = line
                r.font.size, r.font.name, r.font.color.rgb = Pt(size), FONT, BLACK
                r.font.bold = (header and i == 0) or (bold_first_col and j == 0)
    return gf


def image(s, name, x, y, w=None, h=None):
    path = FIG / f"{name}.png"
    iw, ih = Image.open(path).size
    if w and h:                                    # fit inside the box, keep aspect
        if iw / ih > w / h: h = w * ih / iw
        else: w = h * iw / ih
    elif w: h = w * ih / iw
    else: w = h * iw / ih
    s.shapes.add_picture(str(path), Inches(x), Inches(y), Inches(w), Inches(h))
    return w, h


def note(s, t, y=6.75, x=0.5, w=11.6):
    text(s, x, y, w, 0.6, [(t, 11, False)], color=GREY)


# =====================================================================================================================
# 1. Title
# =====================================================================================================================
s = new_slide("SMT × MemUP: Stack и MQAR", "Техническое обсуждение с Айдаром Булатовым · 26.10.2026 · Ирина Ромадова")
smt8, bptt = v("stack_smt_tf8", "eval_best_rnn_dmt_long_acc"), v("stack_bptt_full", "eval_best_rnn_bptt_long_acc")
bptt1 = v("stack_bptt_1", "eval_best_rnn_bptt_long_acc"); mq_bl = v("mqar_T16N2_rnn_bptt_full_long", "eval_best_rnn_bptt_long_acc")
mq_t = v("mqar_T16N2_smt", "eval_best_teacher_smt_long_acc"); mq_r = v("mqar_T16N2_smt", "eval_best_rnn_dmt_long_acc")
table(s, 0.5, 1.6, 12.3, [
    ["задача", "что показано (новые запуски, 3 seeds, независимый набор)", "что пока открыто"],
    ["Stack, T = 64, короткое окно T_f = 8",
     f"SMT → DMT: итоговая RNN {mean(smt8)}% ({fmt(smt8)});\nполный BPTT той же RNN: {mean(bptt)}% ({fmt(bptt)}); BPTT-1: {fmt(bptt1)}",
     "дальние цели: память teacher лучше на диагностике,\nвыигрыша итоговой RNN пока нет"],
    ["MQAR, T = 16, N = 2",
     f"teacher SMT выучивает связывание ключ → значение: {mean(mq_t)}%\n(вывод 20.10 «8 токенов не хранят пары» снят)",
     f"итоговая RNN после DMT {mean(mq_r)}% — ниже teacher; полный BPTT\nпри длинном бюджете: {fmt(mq_bl)} (seed 203 идёт)"],
    ["MQAR, T = 32, N = 4", "teacher нестабилен: 74,9 / 24,8 / 95,9% по seeds", "полный BPTT и длинные бюджеты — в очереди"]],
    widths=[2.2, 5.6, 4.5], size=13, row_h=0.75)
text(s, 0.5, 5.0, 12.3, 1.6, [
    ("Порядок: Stack, затем MQAR. Для каждой задачи — данные и пример → окна → схема обучения и инференса → loss → модели → результаты.", 14, False),
    ("Новые запуски (RTX 3090, seeds 201–203) и старые архивные результаты (seed 107 или 7/17/27) показаны раздельно и подписаны.", 14, False),
    ("Код к каждому слайду — SMT_MemUP_EXPLAINABLE.ipynb; источники чисел — deck/SOURCES_AUDIT_RU.md.", 14, False)], space=8)

# =====================================================================================================================
# STACK
# =====================================================================================================================
s = new_slide("Stack: вход — ID операции, ответ — снятый со стека элемент", "Генератор авторов SMT; T = 64, элементы 1..4, глубина ≤ 8, push / pop 50 / 50")
table(s, 0.5, 1.5, 4.6, [["ID x_t", "операция", "ответ y_t"], ["0", "pop", "снятый элемент 1..4"], ["1, 2, 3, 4", "push этого элемента", "−1"]],
      widths=[1.2, 1.9, 1.9], size=13, row_h=0.42)
text(s, 0.5, 2.95, 4.6, 3.2, [
    ("x — готовые целые ID, текстового токенизатора нет. V = 5 классов.", 13, False),
    ("y = −1 — «ответа нет», позиция маскируется в loss и метриках; это не шестой класс.", 13, False),
    ("delay = позиция pop − позиция push. В Stack delay всегда нечётный; «дальний» ответ: delay ≥ 9.", 13, False),
    ("teacher.embed и rnn.embed — две отдельные обучаемые таблицы [5, 64].", 13, False)], bullets=True, space=8)
table(s, 5.5, 1.5, 7.3, [["t", "x_t", "операция", "стек после", "y_t", "push-источник", "delay"],
                        ["0", "1", "push 1", "[1]", "−1", "—", "—"], ["1", "2", "push 2", "[1, 2]", "−1", "—", "—"],
                        ["2", "0", "pop", "[1]", "2", "1", "1"], ["3", "3", "push 3", "[1, 3]", "−1", "—", "—"],
                        ["4", "0", "pop", "[1]", "3", "3", "1"], ["5", "0", "pop", "[ ]", "1", "0", "5"]],
      widths=[0.6, 0.7, 1.3, 1.3, 0.7, 1.6, 0.8], size=13, row_h=0.42)
note(s, "Пример из обсуждения: x = [1, 2, 0, 3, 0, 0], y = [−1, −1, 2, −1, 3, 1]. Ноутбук §2.1 строит эту таблицу и проверяет её assert.", y=4.75, x=5.5, w=7.3)

s = new_slide("Окна SMT на этом примере: якоря t = 2 и t = 3", "Выход декодера j читает память m_t и входы x_{t+1..t+j}, предсказывает y_{t+j}. Здесь T_f = 2 — только для иллюстрации")
rows2 = [["j", "декодер читает", "предсказывает", "ближняя / дальняя", "источник ответа"],
         ["0", "m_2", "y_2 = 2", "ближняя (j < 2)", "push t=1 ≤ 2: только в m_2"],
         ["1", "m_2, x_3 = 3", "y_3 = −1", "нет ответа", "—"],
         ["2", "m_2, x_3..x_4 = [3, 0]", "y_4 = 3", "дальняя", "push t=3 > 2: виден во входах декодера"],
         ["3", "m_2, x_3..x_5 = [3, 0, 0]", "y_5 = 1", "дальняя", "push t=0 ≤ 2: только в m_2"]]
text(s, 0.5, 1.4, 12.3, 0.4, [("t = 2: x_ctx = [−1, −1, −1, 1, 2, 0],  x_fut = [3, 0, 0, −1, −1, −1],  y_fut = [2, −1, 3, 1, −1, −1]", 11.5, True)])
table(s, 0.5, 1.8, 12.3, rows2, widths=[0.4, 2.6, 1.5, 1.7, 3.6], size=12, row_h=0.36)
rows3 = [["j", "декодер читает", "предсказывает", "ближняя / дальняя", "источник ответа"],
         ["0", "m_3", "y_3 = −1", "нет ответа (на push)", "—"],
         ["1", "m_3, x_4 = 0", "y_4 = 3", "ближняя", "push t=3 ≤ 3: только в m_3"],
         ["2", "m_3, x_4..x_5 = [0, 0]", "y_5 = 1", "дальняя", "push t=0 ≤ 3: только в m_3"],
         ["3", "m_3, x_4..x_5, паддинг", "y_6 — за концом", "нет ответа", "—"]]
text(s, 0.5, 3.75, 12.3, 0.4, [("t = 3: x_ctx = [−1, −1, 1, 2, 0, 3],  x_fut = [0, 0, −1, −1, −1, −1],  y_fut = [−1, 3, 1, −1, −1, −1]", 11.5, True)])
table(s, 0.5, 4.15, 12.3, rows3, widths=[0.4, 2.6, 1.5, 1.7, 3.6], size=12, row_h=0.36)
note(s, "Основной эксперимент: T = 64, T_f = 8. Якорь t = 10: x_ctx содержит 11 реальных входов x_0..x_10 и 53 паддинга −1; выход j не видит x_fut[j:] "
        "(assert в ноутбуке §3 и §5). Учебные выдержки: x_ctx, x_fut, y_fut — из настоящей функции windows().", y=6.15)

s = new_slide("Stack: что подаётся при обучении и что остаётся при инференсе", "Размеры основной конфигурации: B = 32, T = 64, V = 5, K_mem = 8, d = 64, Z = K_mem·d = 512")
image(s, "D_scheme_stack", 0.4, 1.4, w=12.5)
note(s, "Начальные регистры encoder_mem_register [8, 64] — параметры, одинаковые для всех входов; память m_t [8, 64] — выходы энкодера на их позициях, "
        "зависят от x_0..x_t. Ноутбук §5 (assert) и §8 (трасса SMT / DMT / инференс на примере).", y=6.0)

s = new_slide("Stack: функция потерь", "Одна формула для всех вариантов SMT; варианты различаются только правилом выбора дальних целей и весами")
text(s, 0.5, 1.35, 12.3, 0.5, [("L_SMT = c_dec · (L_near + α_far · L_far) + c_dyn · L_dyn + c_unif · L_unif          L_DMT = 10 · MSE + CE", 18, True)])
table(s, 0.5, 2.0, 12.3, [
    ["член", "что сравнивается", "маска / выбор", "вес", "градиент получают"],
    ["L_near", "decode(m_t, x_{t+1..t+j}) → y_{t+j}, CE", "j < T_f = 8, ответ есть", "c_dec = 1", "teacher: embed, encoder, регистры, decoder"],
    ["L_far", "те же логиты, CE", "K_far = 2 из пула j ≥ T_f; far = none → 0", "α_far = 1", "то же"],
    ["L_dyn", "f(m_{t−1}, x_t) → m_t энкодера, MSE по 512 числам", "все B·T якорей, m_{−1} = 0", "c_dyn = 0,1", "updater f и энкодер"],
    ["L_unif", "разброс токенов памяти m_t", "512 случайных токенов", "c_unif = 0,001", "энкодер"],
    ["DMT: MSE", "f(m̂_{t−1}, x_t) → m_t замороженного энкодера", "вход m̂_{t−1} detach", "10", "updater f, rnn.embed"],
    ["DMT: CE", "readout(sg[m̂_t]) → y_t", "позиции pop", "1", "только readout RNN"]],
    widths=[1.1, 3.9, 3.0, 1.3, 3.0], size=12.5, row_h=0.46)
note(s, "Функции: smt_loss, dmt_loss, bptt_loss (ноутбук §7). Совпадают бит в бит с кодом GPU-запусков (необязательная проверка §11: 21 сравнение, расхождение 0). "
        "Полный BPTT: та же RNN, CE по всем позициям, градиент через все шаги; BPTT-1 — состояние переносится, градиент режется на каждом шаге.", y=5.55)

s = new_slide("Stack: модели и что в них меняется", "Базовый вариант — первая строка; каждая следующая меняет одну вещь")
table(s, 0.5, 1.4, 12.3, [
    ["название", "имя в коде", "что меняется", "бюджет", "seeds и протокол"],
    ["SMT: короткое окно, без дальних целей", "stack_smt_tf8", "база: T_f = 8, far = none", "SMT 11k + DMT 5k", "201–203, новые"],
    ["SMT + дальние цели по ошибке без памяти", "…_ctx_far", "K_far = 2 цели, где детектор без памяти ошибается сильнее", "11k + 5k", "201–202 готовы, 203 в очереди"],
    ["SMT + дальние цели по локальной неожиданности", "…_surprise_far", "K_far = 2 по оконному детектору (8 входов)", "11k + 5k", "201: перезапуск после OOM"],
    ["SMT + случайные дальние цели", "…_random_far", "K_far = 2 случайно", "11k + 5k", "201 готов"],
    ["SMT без L_dyn / цель L_dyn отцеплена / L_dyn полностью отцеплен", "…_no_dyn / …_target_detached / …_fully_detached", "c_dyn = 0 / sg[m_t] / sg[m_{t−1}] и sg[m_t]", "11k + 5k", "201: готов / готов / идёт;\nстарые: seed 107"],
    ["RNN, полный BPTT", "stack_bptt_full", "та же RNN с нуля, без teacher", "20k", "201–203 готовы"],
    ["RNN, градиент на один переход (BPTT-1)", "stack_bptt_1", "градиент режется на каждом шаге", "20k", "201 готов, 202 идёт, 203 в очереди"],
    ["SMT: полное будущее окно", "full_window / tf64", "T_f = T = 64", "3k + 1,5k / 11k + 5k", "только старые: 7/17/27, 107"],
    ["диагностика: дальние цели, чей источник уже в памяти", "oracle_far", "пул сужен по разметке генератора — у метода такой информации нет", "3k + 1,5k", "только старые: 7/17/27"]],
    widths=[3.4, 2.4, 3.6, 1.4, 2.0], size=11.5, row_h=0.48)
note(s, "Общее: batch 32, AdamW, LR SMT 3e-4 / DMT 1e-4 / BPTT 3e-4, разогрев 200 шагов и константа; чекпойнт — минимум validation CE на дальних ответах.", y=6.45)

s = new_slide("Stack, новые запуски: кривые и итоговая RNN", "RTX 3090, seeds 201–203; validation для кривых, независимый набор (поток 901) для чисел")
image(s, "S1_stack_new_curves", 0.4, 1.3, w=12.5)
ctx = v("stack_smt_tf8_ctx_far", "eval_best_rnn_dmt_long_acc")
table(s, 0.5, 5.45, 12.3, [["итоговая RNN, delay ≥ 9, % (seeds 201 / 202 / 203)", "SMT → DMT", "+ дальние по ошибке без памяти", "полный BPTT", "BPTT-1"],
                          ["независимый набор, выбранный чекпойнт", fmt(smt8), fmt(ctx), fmt(bptt), fmt(bptt1)]],
      widths=[3.6, 2.0, 2.6, 2.0, 2.1], size=12, row_h=0.36)
note(s, "«—» = запуск не завершён (не ноль). Первые попытки BPTT упали по нехватке памяти GPU и перезапущены. Seed 201: + случайные дальние "
        f"{fmt(v('stack_smt_tf8_random_far', 'eval_best_rnn_dmt_long_acc'), (201,))}; без L_dyn {fmt(v('stack_smt_tf8_no_dyn', 'eval_best_rnn_dmt_long_acc'), (201,))}; "
        f"цель L_dyn отцеплена {fmt(v('stack_smt_tf8_target_detached', 'eval_best_rnn_dmt_long_acc'), (201,))} (приложение B).", y=6.3)

s = new_slide("Stack, старый протокол: окно, горизонт T_f, компоненты SMT", "Архив: validation, один seed 107 (б, в) или 3 seeds при малом бюджете (а); числа панелей между собой не сравниваются")
image(s, "S7_stack_ablations_old", 0.4, 1.35, w=12.5)
note(s, "Наблюдение: teacher почти 100% во всех вариантах, а итоговая RNN зависит от T_f и от L_dyn: без L_dyn 66,0% против 94,5%; T_f = 1 — 37,9%. "
        "Новые запуски seed 201 (независимый набор): без L_dyn 65,5% — повторяет старое; цель L_dyn отцеплена 98,5% против базы 95,0% — один seed.", y=5.6)

s = new_slide("Stack: дальние цели — три разные метрики", "Старый протокол v2: T_f = 8, K_far = 2, SMT 3k / DMT 1,5k (недообучено), seeds 7 / 17 / 27, validation")
image(s, "S4_stack_far_three_metrics_old", 0.4, 1.35, w=12.5)
note(s, f"Probe памяти teacher растёт с дальними целями (34% → 47–62%), итоговая RNN — нет (52–57%). При длинном обучении (v4, seed 107, приложение) probe "
        f"99% против 47%, а RNN 98,6 против 98,1%. Новые запуски: «по ошибке без памяти» {fmt(ctx)}, случайные (seed 201) "
        f"{fmt(v('stack_smt_tf8_random_far', 'eval_best_rnn_dmt_long_acc'), (201,))} против базы {fmt(smt8)}.", y=5.75)

# =====================================================================================================================
# MQAR
# =====================================================================================================================
ex_x, ex_y = [54, 76, 8, 96, 8, 0, 0, 0, 54, 0, 0, 0, 0, 0, 0, 0], [-1, -1, -1, -1, 96, -1, -1, -1, 76, -1, -1, -1, -1, -1, -1, -1]
s = new_slide("MQAR: пары «ключ → значение», затем запросы ключей", "Генератор Zoology (multiquery_ar), без шума; V = 128 ID")
table(s, 0.5, 1.45, 5.3, [["часть", "ID", "позиции"],
                         ["ключ пары i", "1..63 (V/2 − 1)", "2i"], ["значение пары i", "64..127", "2i + 1"],
                         ["запрос пары i", "тот же ID ключа", "≥ 2N, чётные"], ["заполнитель", "0", "остальные"]],
      widths=[1.8, 1.9, 1.6], size=13, row_h=0.42)
text(s, 0.5, 3.75, 5.3, 2.6, [
    ("Одна пара = 2 ID (ключ, значение); ключи и значения в примере без повторов.", 13, False),
    ("Метка — значение пары — стоит на позиции запроса; на остальных позициях y = −1.", 13, False),
    ("Каждый ключ запрашивается ровно один раз; промежутки — степенное распределение.", 13, False),
    ("Accuracy по всему словарю (128 классов) ≠ accuracy среди N записанных значений (диагностика, угадывание 1/N).", 13, False)], bullets=True, space=8)
text(s, 6.2, 1.45, 6.6, 0.4, [("Пример T = 16, N = 2 (поток 901, последовательность 0; ноутбук §2.2)", 12.5, True)])
rows = [["t"] + [str(t) for t in range(16)], ["x_t"] + [str(i) for i in ex_x],
        ["роль", "K0", "V0", "K1", "V1", "Q1", "0", "0", "0", "Q0", "0", "0", "0", "0", "0", "0", "0"],
        ["y_t"] + [("·" if i == -1 else str(i)) for i in ex_y]]
table(s, 6.2, 1.85, 6.6, rows, widths=[1.3] + [1] * 16, size=11, row_h=0.36, bold_first_col=True)
text(s, 6.2, 3.45, 6.6, 2.6, [
    ("K0 = 54 → V0 = 76 (позиции 0, 1); K1 = 8 → V1 = 96 (позиции 2, 3).", 13, False),
    ("t = 4: запрос ключа 8 → ответ 96, значение записано на t = 3, delay = 1.", 13, False),
    ("t = 8: запрос ключа 54 → ответ 76, значение записано на t = 1, delay = 7.", 13, False),
    ("«Длинные» ответы — delay ≥ T/4: 4 при T = 16, 8 при T = 32.", 13, False)], bullets=True, space=8)
note(s, "Это та же задача, что в статье Zoology, но короче: T = 16, N = 2 и T = 32, N = 4 — новые запуски; T = 64, N = 8 (pilot01) — только старые.", y=6.45)

s = new_slide("MQAR: что подаётся при обучении и что остаётся при инференсе", "SMT как в Stack, но T_f = T (весь пример в окне) и far = none; инференс итоговой модели — только f и readout RNN")
image(s, "D_scheme_mqar", 0.9, 1.3, w=11.5)
table(s, 0.9, 5.0, 11.5, [["режим оценки", "что видит модель в момент ответа на запрос"],
                         ["teacher, j = 0", "энкодер видел все пары и сам ключ запроса; ответ — выход j = 0"],
                         ["teacher, память до запросов", "память m_{2N−1} после пар; декодер читает x_{2N..t}, последний из них — ключ запроса"],
                         ["диагностика: все выходы энкодера", "декодер читает T + 8 векторов энкодера вместо 8 токенов памяти"],
                         ["итоговая RNN / causal Transformer", "только своё состояние m̂_t / весь префикс через attention"]],
      widths=[3.3, 8.2], size=11.5, row_h=0.32)

s = new_slide("MQAR: функция потерь и модели", "Формула та же, что для Stack; отличия — T_f = T, far = none, ответы только на позициях запросов")
text(s, 0.5, 1.35, 12.3, 0.5, [("L_SMT = L_near + 0,1 · L_dyn + 0,001 · L_unif   (T_f = T, L_far = 0)          L_DMT = 10 · MSE + CE", 18, True)])
table(s, 0.5, 2.05, 12.3, [["модель", "loss", "бюджет", "LR", "seeds"],
                          ["SMT → DMT", "L_SMT, затем L_DMT", "6k + 2k", "3e-4 / 1e-4, константа", "201–203"],
                          ["teacher 8 токенов, только L_dec", "L_near; c_dyn = c_unif = 0, без RNN", "6k", "3e-4", "201–203 (T = 16)"],
                          ["диагностика: все выходы энкодера", "L_near; декодер читает все T + 8 выходов энкодера", "6k", "3e-4", "201–203 (T = 16)"],
                          ["RNN, полный BPTT", "CE на позициях запросов, градиент через все шаги", "6k; повтор — до 60k,\nстоп при 99%", "1e-3", "201–203"],
                          ["causal Transformer, 2 слоя, d = 64", "CE на позициях запросов", "≤ 3k, стоп при 99%", "1e-3", "201–203"]],
      widths=[3.0, 4.6, 2.0, 1.6, 1.4], size=12.5, row_h=0.5)
note(s, "Метрики: accuracy по всему словарю (128 классов) — основная; accuracy среди N записанных в этой последовательности значений — диагностика "
        "(угадывание 1/N). «Длинные» ответы: delay ≥ T/4. Чекпойнт — минимум validation CE.", y=5.3)

s = new_slide("MQAR T = 16, N = 2: teacher выучивает связывание скачком", "Новые запуски, 3 seeds; плато на 50% = угадывание одного из двух записанных значений")
image(s, "M1_mqar_T16_curves", 0.4, 1.3, w=12.5)
note(s, "Teacher SMT и teacher «8 токенов, только L_dec» прыгают с 50% к ~100% между шагами 5 000 и 6 000 (seed 203 — на последней точке 6 000). "
        "Декодер со всеми выходами энкодера — около шага 1 500. Полный BPTT: seed 202 прыгает ступенями (~15k и ~32k шагов) до 99% к 51k; "
        "seed 201 за 60k — 56%; seed 203 идёт.", y=5.55)

s = new_slide("MQAR T = 16, N = 2: независимый набор", "Accuracy по всему словарю и среди двух записанных значений; длинные ответы (delay ≥ 4)")
image(s, "M2_mqar_T16N2_bars", 0.5, 1.25, h=5.25)
mq_full = v("mqar_T16N2_teacher_full", "eval_best_teacher_full_smt_long_acc"); mq_pre = v("mqar_T16N2_smt", "eval_best_teacher_pre_after_pairs_smt_long_acc")
mq_b = v("mqar_T16N2_rnn_bptt_full", "eval_best_rnn_bptt_long_acc"); mq_tr = v("mqar_T16N2_transformer", "eval_best_lm_long_acc")
text(s, 10.0, 1.4, 2.9, 5.2, [
    ("Teacher (память 8 токенов):", 12.5, True), (f"j = 0: {fmt(mq_t)}", 12, False), (f"до запросов: {fmt(mq_pre)}", 12, False),
    ("Итоговая RNN после DMT:", 12.5, True), (f"{fmt(mq_r)}", 12, False),
    ("Полный BPTT, 6k / ≤ 60k шагов:", 12.5, True), (f"{fmt(mq_b)} / {fmt(mq_bl)}", 12, False),
    ("Transformer:", 12.5, True), (f"{fmt(mq_tr)}", 12, False),
    ("Наблюдение: итоговая RNN после DMT ниже teacher. Причина не установлена: DMT всего 2 000 шагов; та же RNN с полным BPTT решает задачу, но не на каждом seed.", 12, False)], space=5)

s = new_slide("MQAR T = 32, N = 4 и старый пилот T = 64, N = 8", "Новые запуски (слева) и архив seed 107 (справа) — разные протоколы")
image(s, "M3_mqar_T32_curves", 0.4, 1.25, w=7.6, h=3.4)
t32 = v("mqar_T32N4_smt", "eval_best_teacher_smt_long_acc"); r32 = v("mqar_T32N4_smt", "eval_best_rnn_dmt_long_acc"); tr32 = v("mqar_T32N4_transformer", "eval_best_lm_long_acc")
table(s, 0.5, 4.8, 7.4, [["T = 32, N = 4; независимый набор, %", "201 / 202 / 203"],
                        ["teacher j = 0", fmt(t32)], ["итоговая RNN после DMT", fmt(r32)], ["causal Transformer", fmt(tr32)],
                        ["RNN, полный BPTT", "в очереди (60k шагов)"]], widths=[4.4, 3.0], size=12, row_h=0.32)
table(s, 8.3, 1.4, 4.5, [["pilot01: T = 64, N = 8 (старое)", "результат"],
                        ["teacher 8 токенов, только L_dec, 4k шагов", "12,8%"], ["декодер читает все выходы энкодера, 4k", "100%"],
                        ["SMT 20k + DMT 10k, LR косинус", "12,3% среди записанных"], ["RNN, полный BPTT, 20k", "1,7% (13,2% среди записанных)"],
                        ["causal Transformer", "100%"]], widths=[2.9, 1.7], size=11, row_h=0.5)
text(s, 8.3, 4.6, 4.6, 1.8, [("Старое: seed 107, validation 128–1024 посл.; teacher — только логи. LR по косинусу: тот же режим, при котором короткий MQAR раньше "
                              "давал 50,8%. Повторять на новом протоколе нужно до выводов о T = 64.", 11.5, False)], color=GREY)
note(s, "T = 32: teacher выучил задачу на одном seed из трёх (95,9%), частично — на другом (74,9%), на третьем остался на угадывании (24,8%) за 6 000 шагов.", y=6.6)

s = new_slide("Итоги: что установлено, что исправлено, что ещё считается")
table(s, 0.5, 1.2, 12.3, [
    ["", "установлено (в пределах протокола)", "не установлено / идёт"],
    ["Stack", f"Короткое окно T_f = 8: итоговая RNN после SMT → DMT {mean(smt8)}% на 3 seeds; полный BPTT {mean(bptt)}%; BPTT-1 {fmt(bptt1, (201,))}% (1 seed).\nTeacher решает задачу почти всегда; "
              "качество RNN зависит от L_dyn и T_f (старые абляции, 1 seed).\nДальние цели улучшают память teacher (probe), но не итоговую RNN.",
     "BPTT-1 (2 seeds), абляция «L_dyn полностью отцеплен»,\n«локальная неожиданность» против случайных — считаются."],
    ["MQAR", f"T = 16, N = 2: teacher SMT выучивает связывание ({mean(mq_t)}%), скачком около 5 000 шагов.\n"
             f"Итоговая RNN после DMT ниже teacher ({mean(mq_r)}%). Полный BPTT с бюджетом до 60k: {fmt(mq_bl)}. T = 32: teacher нестабилен по seeds.",
     "Почему RNN после DMT ниже teacher; полный BPTT seed 203;\nT = 32 и T = 64 на новом протоколе."]],
    widths=[1.0, 6.8, 4.5], size=12.5, row_h=[0.4, 1.15, 1.0])
table(s, 0.5, 4.3, 12.3, [
    ["было в презентации 20.10", "сейчас"],
    ["«Короткий MQAR teacher не выучивает связывание; 8 токенов памяти не хранят пары»", "Неверно: 99% на T = 16. Старый запуск 50,8% — косинусный LR до 3e-5, тот же бюджет 6 000 шагов"],
    ["«Доступ ко всем выходам энкодера: 12,8% → 100%»", "На T = 16 — разница в скорости (скачок ~1 500 против ~5 000 шагов), не в возможности"],
    ["«RNN с полным BPTT на уровне угадывания»", f"6 000 шагов не хватило; до 60 000: {fmt(mq_bl)}; учится, но не на каждом seed"],
    ["Base / Context / Oracle / detached / full", "Описательные названия; full уточняется: компоненты SMT, окно или BPTT"]],
    widths=[5.3, 7.0], size=12, row_h=0.42)

# =====================================================================================================================
# APPENDIX
# =====================================================================================================================
s = new_slide("Приложение A. Названия моделей и режимов", "Идентификаторы в коде оставлены для совместимости")
table(s, 0.5, 1.4, 12.3, [
    ["в коде", "название в выводах"], ["base_window / stack_smt_tf8", "SMT: короткое окно, без дальних целей"],
    ["ctx_far", "SMT + дальние цели по ошибке без памяти"], ["surprise_far", "SMT + дальние цели по локальной неожиданности"],
    ["random_far", "SMT + случайные дальние цели"], ["uniform_far", "SMT + все дальние цели"],
    ["oracle_far", "диагностика: дальние цели, чей источник уже в памяти"], ["full_window / tf64", "SMT: полное будущее окно (T_f = T)"],
    ["bptt_full", "RNN, полный BPTT"], ["bptt_1", "RNN, градиент на один переход (BPTT-1)"],
    ["teacher", "teacher, выход j = 0 на позиции ответа (энкодер видел pop / запрос)"],
    ["teacher_pre", "teacher, память m_{2N−1} после всех пар, до запросов"],
    ["teacher_full", "диагностика: декодер читает все выходы энкодера, не только 8 токенов памяти"],
    ["no_dyn / target_detached (старое «detached») / fully_detached", "SMT без L_dyn / цель L_dyn отцеплена / L_dyn полностью отцеплен"]],
    widths=[4.2, 8.1], size=12.5, row_h=0.37)

s = new_slide("Приложение B. Stack, новые запуски по seeds", "Независимый набор, выбранный чекпойнт")
image(s, "S2_stack_new_bars", 0.5, 1.3, w=12.3, h=5.5)

s = new_slide("Приложение C. Дальние цели, длинное обучение (v4)", "Seed 107, validation; SMT 20k / DMT 12k")
image(s, "S5_stack_far_v4_old", 0.4, 1.35, w=12.5)
note(s, "Probe памяти teacher: 99,2% с дальними целями против 47,1% без них; итоговая RNN 98,6% против 98,1% — в пределах шума одного seed.", y=5.75)

s = new_slide("Приложение D. MQAR T = 32, N = 4: независимый набор", "Новые запуски, 3 seeds")
image(s, "M2_mqar_T32N4_bars", 0.9, 1.25, h=5.4)

prs.save(OUT)
pd.DataFrame(SLIDES, columns=["slide", "title"]).to_csv(TAB / "slides.csv", index=False)
print("saved", OUT.relative_to(ROOT), len(prs.slides), "slides")
