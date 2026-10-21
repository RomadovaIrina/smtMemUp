#!/usr/bin/env python3
"""Prints the markdown tables of EXAMPLES_RU.md from the real generators (validation sequence 0 of each setting).
Run:  python checks/make_examples_ru.py > /tmp/examples.md   (the tables in EXAMPLES_RU.md were pasted from it)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smt_minimal as C  # noqa: E402


def table(cfg, batch, rows=None):
    t = C.trace_table(cfg, batch, 0)
    if rows is not None:
        t = t[t.t.isin(rows)]
    head = "| t | смысл | токен x_t (= вход) | метка y_t | маска лосса | источник | delay |\n|---|---|---|---|---|---|---|"
    ru = lambda m: (m.replace("push", "push").replace("pop", "pop").replace("key of pair", "ключ пары")
                    .replace("value of pair", "значение пары").replace("query key", "запрос (ключ)")
                    .replace("filler 0", "заполнитель 0"))
    body = "\n".join(f"| {r.t} | {ru(r.meaning)} | {r.token} | {r.label if r.label != -1 else '−1'} | {r.loss_mask} | "
                     f"{r.source if r.source != -1 else '—'} | {r.delay if r.delay != -1 else '—'} |" for r in t.itertuples())
    return head + "\n" + body


def window_view(cfg, batch, a, t):
    w = C.windows(batch)
    T = cfg.seq_len
    j = t - a
    ctx = [v for v in w["x_ctx"][a].tolist() if v != C.IGNORE]
    return (f"* якорь a = {a}: энкодер видит x_0..x_{a} = `{ctx}` → память m_{a};\n"
            f"* выход декодера j = {j} видит m_{a} и x_fut[a, :{j}] = x_{a + 1}..x_{t} = `{w['x_fut'][a, :j].tolist()}`;\n"
            f"* его метка y_fut[a, {j}] = y_{t} = {int(w['y_fut'][a, j])}; valid = {bool(w['valid'][a, j])}; "
            f"{'ближняя (j < Tf)' if j < cfg.future_window else 'дальняя (j ≥ Tf)'} цель при Tf = {cfg.future_window}.")


if __name__ == "__main__":
    s = C.Config()
    sb = C.validation_set(s)
    print("## Stack, T = 64 (валидация, последовательность 0)\n")
    print(table(s, sb))
    d = sb["delay"][0]
    t_long = int((d >= 8).nonzero()[0])
    src = int(sb["source_pos"][0, t_long])
    print(f"\nДлинный pop: t = {t_long}, источник {src}, delay {int(d[t_long])}.\n")
    for a in (t_long, t_long - 3, t_long - 12, src):
        if 0 <= a <= t_long:
            print(window_view(s, sb, a, t_long) + "\n")
    for name, cfg in (("T = 16, N = 2", C.Config(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4, future_window=16)),
                      ("T = 32, N = 4", C.Config(task="mqar", seq_len=32, num_kv_pairs=4, long_delay=8, future_window=32))):
        b = C.validation_set(cfg)
        print(f"\n## MQAR, {name} (валидация, последовательность 0)\n")
        print(table(cfg, b))
        q = int(b["target_mask"][0].nonzero()[-1])
        N = cfg.num_kv_pairs
        print(f"\nПоследний запрос: t = {q}.\n")
        for a in (q, 2 * N - 1, q - 1):
            print(window_view(cfg, b, a, q) + "\n")
