"""Commands:  python -m smtcore trace --task stack|mqar     (position table of one real sequence)
            python -m smtcore demo  --task stack|mqar     (~2 min CPU: tiny SMT -> DMT, one answer traced)
            python -m smtcore train --plan configs/plan.json --index I   (one run; GPU if available)
            python -m smtcore train --spec my_run.json
            python -m smtcore list  --plan configs/plan.json"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from .config import Config, diff_from_default
from .data import sample, stack_states, validation_set
from .evaluate import answer_logits
from .models import make_models
from .train import run
from .upstream import IGNORE, ROOT, author, derive_seed
from .windows import windows

TRACE = {"stack": Config(),
         "mqar": Config(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4, future_window=16)}
DEMO = {"stack": Config(seq_len=16, max_stack_size=4, long_delay=4, d_model=32, n_heads=2, smt_steps=150,
                        dmt_steps=100, eval_every=50, validation_examples=64, eval_examples=64, batch_size=16,
                        warmup_steps=20),
        "mqar": Config(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4, future_window=16, d_model=32, n_heads=2,
                       smt_steps=150, dmt_steps=50, eval_every=50, validation_examples=64, eval_examples=64,
                       batch_size=16, warmup_steps=20)}


def trace_table(cfg: Config, batch: dict, b: int = 0) -> pd.DataFrame:
    """One sequence: position, meaning, input ID, task state before/after, answer ID, in loss?, source, delay."""
    x, y = batch["x"][b], batch["y"][b]
    states = stack_states(x) if cfg.task == "stack" else None
    N, rows = cfg.num_kv_pairs, []
    for t in range(len(x)):
        tok, lab = int(x[t]), int(y[t])
        if cfg.task == "stack":
            meaning = f"push {tok}" if tok > 0 else "pop"
            before, after = (" ".join(map(str, s)) or "-" for s in states[t])
        else:
            meaning = (f"key of pair {t // 2}" if t < 2 * N and t % 2 == 0 else f"value of pair {t // 2}" if t < 2 * N
                       else "query key" if lab != IGNORE else "filler 0")
            before = after = ""
        rows.append(dict(t=t, meaning=meaning, input_id=tok, state_before=before, state_after=after,
                         answer_id=lab, in_loss=int(lab != IGNORE), source=int(batch["source_pos"][b, t]),
                         delay=int(batch["delay"][b, t])))
    out = pd.DataFrame(rows)
    return out if cfg.task == "stack" else out.drop(columns=["state_before", "state_after"])


@torch.no_grad()
def demo(task: str, log=print):
    cfg = DEMO[task]
    out = ROOT / "results" / "demo"
    spec = dict(name=f"demo_{task}_smt", family="smt", seed=1, config=diff_from_default(cfg))
    with torch.enable_grad():
        print(json.dumps(run(spec, out, torch.device("cpu"), log), indent=2))
    teacher, rnn = make_models(cfg, 1, torch.device("cpu"))
    teacher.load_state_dict(torch.load(out / spec["name"] / "smt_best.pt", weights_only=True)["models"]["teacher"])
    rnn.load_state_dict(torch.load(out / spec["name"] / "dmt_best.pt", weights_only=True)["models"]["rnn"])
    teacher.eval(); rnn.eval()
    batch = sample(cfg, 1, derive_seed(901, 0))            # sequence 0 of the independent evaluation stream
    table = trace_table(cfg, batch)
    print(table.to_string(index=False))
    t = int(table[table.answer_id != IGNORE].t.iloc[-1])
    w = windows(batch)
    print(f"\nAnswer at t = {t}: input x_t = {int(batch['x'][0, t])}, answer y_t = {int(batch['y'][0, t])}, "
          f"delay {int(batch['delay'][0, t])}.")
    print(f"windows row t: x_ctx = {w['x_ctx'][t].tolist()}  (encoder sees x_0..x_t; -1 = padding, masked)")
    m_t = teacher.encode(w["x_ctx"][t:t + 1])
    ce = author("model").calc_cross_entropy_loss(teacher.decode(m_t, w["x_fut"][t:t + 1]), w["y_fut"][t:t + 1].clamp(min=0))
    print(f"SMT loss, anchor t, output j = 0: y_fut[t, 0] = {int(w['y_fut'][t, 0])}, CE = {float(ce[0, 0]):.3f}; it enters "
          f"L_dec^Tf because j = 0 < Tf = {cfg.future_window} and valid[t, 0] = {bool(w['valid'][t, 0])}")
    for a in range(max(0, t - 3), t):
        print(f"  the same y_t is also output j = {t - a} of anchor a = {a}: decoder sees m_{a} and "
              f"x_fut[a, :{t - a}] = {w['x_fut'][a, :t - a].tolist()}")
    for kind, model in (("teacher", teacher), ("rnn", rnn)):
        r = answer_logits(kind, model, batch, cfg, torch.device("cpu"))
        r = r[r.t == t].iloc[0]
        print(f"evaluation row ({kind}): pred = {int(r.pred)}, answer = {int(r.target)}, correct = {bool(r.correct)}, "
              f"CE = {r.ce:.3f}")
    if task == "mqar":
        for anchor in ("after_pairs", "before_query"):
            r = answer_logits("teacher_pre", teacher, batch, cfg, torch.device("cpu"), anchor)
            r = r[r.t == t].iloc[0]
            a = 2 * cfg.num_kv_pairs - 1 if anchor == "after_pairs" else t - 1
            print(f"evaluation row (teacher_pre, {anchor}: memory m_{a}, decoder inputs x_{a + 1}..x_{t}): "
                  f"pred = {int(r.pred)}, correct = {bool(r.correct)}, among written = {bool(r.cand_correct)}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["trace", "demo", "train", "list"])
    p.add_argument("--task", default="stack", choices=["stack", "mqar"])
    p.add_argument("--plan", default=str(ROOT / "configs" / "plan.json"))
    p.add_argument("--index", type=int)
    p.add_argument("--spec")
    p.add_argument("--out", default=str(ROOT / "results" / "runs"))
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args(argv)
    torch.set_num_threads(a.threads)
    pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.command == "trace":
        cfg = TRACE[a.task]
        print(trace_table(cfg, validation_set(cfg), 0).to_string(index=False))
    elif a.command == "demo":
        demo(a.task, log)
    elif a.command == "list":
        for i, r in enumerate(json.loads(Path(a.plan).read_text())["runs"]):
            print(i, r["tier"], r["name"], r["family"], json.dumps(r["config"]))
    else:
        spec = json.loads(Path(a.spec).read_text()) if a.spec else json.loads(Path(a.plan).read_text())["runs"][a.index]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(json.dumps(run(spec, Path(a.out), device, log), indent=2))


if __name__ == "__main__":
    main()
