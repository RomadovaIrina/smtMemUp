#!/usr/bin/env python3
"""MQAR on ONE Zoology cell, all model classes on the same data: T = 64, N = 4, V = 8192 (smallest cell of Zoology fig. 2).

Data (identical for every model): the authors' prepare_data with the figure-2 settings - 100 000 training and 3 000
test sequences (DataConfig seed 123, power_a 0.01, random_non_queries False). Metric = Zoology's valid/accuracy:
argmax == label over ALL query positions, full vocabulary. Every run also reports accuracy on an independent set of
3 000 sequences (seed 9123), and on long answers (delay >= 16).

Families (results/zoology_cell/<run_id>/: config.json, history.jsonl, summary.json):
  zoology  --mixer {attention, base_conv, h3, hyena, based} --d {64,128} --lr L
           authors' trainer and models (upstream/zoology @ 1ad20d1), figure-2 configuration: 2 layers, no MLP, batch 512,
           AdamW wd 0.1, cosine over 64 epochs (no warm-up), early stop at valid/accuracy > 0.99.
  ours_rnn --lr L      our RNN (authors' SMT RNN class, d 64, K 8) trained directly with full BPTT, EXACTLY the Zoology
           protocol above (same batch 512, epochs, optimiser, schedule, early stop; no gradient clipping, as Zoology).
  ours_smt             SMT -> DMT (our teacher + RNN, d 64, K 8, Tf = T) on the same training set. The Zoology budget is
           NOT feasible for SMT on a 24 GB GPU (every sequence = 64 windows): SMT 20 000 steps x batch 32 (micro-batch 8,
           accumulation 4) = 6.4 epochs, then DMT 10 000 steps x 32 = 3.2 epochs; LR 3e-4 / 1e-4 warm-up + constant,
           clip 1.0 (our protocol). Evaluated: teacher j=0, teacher with memory before the query, RNN before/after DMT.
The code of our models, losses, rollout and evaluation is executed from the definition cells of
SMT_MemUP_FROM_ZERO.ipynb (no second implementation).

  python zoology_cell/run_cell.py --queue                 # every job below, GPU-memory gated (MAX_PAR=2)
  python zoology_cell/run_cell.py --family zoology --mixer h3 --d 64 --lr 0.0022
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "upstream" / "zoology"
SMOKE = os.environ.get("SMOKE") == "1"            # tiny subsets and steps: path check only
OUT = ROOT / "results" / ("zoology_cell_smoke" if os.environ.get("SMOKE") == "1" else "zoology_cell")
PINNED = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
sys.path.insert(0, str(ZOO))
T, N, V = 64, 4, 8192
LRS = [float(x) for x in np.logspace(-4, -2, 4)]               # 1e-4, 4.64e-4, 2.15e-3, 1e-2 (figure 2)
MIXERS = {
    "attention": dict(name="zoology.mixers.attention.MHA", kwargs={"dropout": 0.1, "num_heads": 1}),
    "base_conv": dict(name="zoology.mixers.base_conv.BaseConv", kwargs={"l_max": T, "kernel_size": [3, -1, 3, -1]}),
    "h3": dict(name="zoology.mixers.h3.H3", kwargs={"l_max": T, "d_state": T, "head_dim": 2}),
    "hyena": dict(name="zoology.mixers.hyena.Hyena", kwargs={"l_max": T}),
    "based": dict(name="zoology.mixers.hybrid.Hybrid", kwargs={"configs": [
        dict(name="zoology.mixers.base_conv.BaseConv", kwargs={"l_max": T, "kernel_size": 3, "implicit_long_conv": True}),
        dict(name="zoology.mixers.based.Based", kwargs={"l_max": T, "feature_dim": 8, "num_key_value_heads": 1, "num_heads": 1,
                                                        "feature_name": "taylor_exp", "train_view": "quadratic"})]}),
}


def data_config():
    from zoology.config import DataConfig
    from zoology.data.multiquery_ar import MQARConfig
    kw = {"num_kv_pairs": N, "train_power_a": 0.01, "test_power_a": 0.01, "random_non_queries": False}
    return DataConfig(train_configs=[MQARConfig(num_examples=100_000, vocab_size=V, input_seq_len=T, **kw)],
                      test_configs=[MQARConfig(num_examples=3_000, vocab_size=V, input_seq_len=T, **kw)],
                      batch_size=512, cache_dir=str(ROOT / "results" / "zoology_cell" / "cache"))


def independent_segment():
    return data_config().test_configs[0].model_copy(update=dict(num_examples=3000)).build(seed=9123)


# =====================================================================================================================
# Zoology models with the authors' trainer
# =====================================================================================================================
def run_zoology(mixer, d, lr):
    import torch
    import zoology.train as ZT
    from zoology.config import LoggerConfig, ModelConfig, TrainConfig
    from zoology.logger import WandbLogger
    run_id = f"zoology_{mixer}_d{d}_lr{lr:.2e}"
    out = OUT / run_id
    if (out / "summary.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)

    class FileLogger(WandbLogger):
        def __init__(self, config):
            self.no_logger = False; self.file = open(out / "history.jsonl", "a")
        def log_config(self, config):
            (out / "config.json").write_text(json.dumps(config.model_dump(), indent=2, default=str))
        def log_model(self, model, config):
            self.log({"num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})
        def log(self, metrics):
            self.file.write(json.dumps({"time": time.time(), **metrics}) + "\n"); self.file.flush()
        def finish(self):
            self.file.close()

    model = ModelConfig(d_model=d, n_layers=2, block_type="TransformerBlock", max_position_embeddings=T if mixer == "attention" else 0,
                        vocab_size=V, sequence_mixer=MIXERS[mixer], state_mixer=dict(name="torch.nn.Identity", kwargs={}))
    config = TrainConfig(model=model, data=data_config(), learning_rate=lr, max_epochs=64, run_id=run_id, logger=LoggerConfig())
    captured = {}
    real_fit = ZT.Trainer.fit

    def fit_and_keep(self):
        captured["model"] = self.model
        return real_fit(self)
    ZT.Trainer.fit = fit_and_keep
    ZT.WandbLogger = FileLogger
    tic = time.perf_counter()
    ZT.train(config)
    rows = [json.loads(l) for l in (out / "history.jsonl").read_text().splitlines()]
    ep = [r for r in rows if "valid/accuracy" in r]
    seg = independent_segment(); m = captured["model"].eval(); correct = total = 0
    with torch.no_grad():
        for i in range(0, 3000, 100):
            x, y = seg.inputs[i:i + 100].cuda(), seg.labels[i:i + 100].cuda()
            p = m(x).argmax(-1); k = y != -100; correct += int((p[k] == y[k]).sum()); total += int(k.sum())
    summary = dict(run_id=run_id, family="zoology", mixer=mixer, d_model=d, lr=lr, epochs_run=len(ep),
                   final_valid_accuracy=ep[-1]["valid/accuracy"], max_valid_accuracy=max(r["valid/accuracy"] for r in ep),
                   early_stopped=ep[-1]["valid/accuracy"] > 0.99, independent_accuracy=correct / total,
                   num_parameters=next((r["num_parameters"] for r in rows if "num_parameters" in r), None),
                   wall_seconds=time.perf_counter() - tic, zoology=PINNED)
    (out / "summary.json").write_text(json.dumps(summary, indent=2)); print(json.dumps(summary), flush=True)


# =====================================================================================================================
# our models: definitions executed from the notebook
# =====================================================================================================================
DEFS = {"s0-setup", "s0-seeds", "s1-config", "s2-stack", "s2-mqar", "s3-windows", "s4-model", "s4-make", "s6-losses", "s8-train", "s9-eval"}


def notebook_namespace():
    import nbformat
    nb = nbformat.read(ROOT / "SMT_MemUP_FROM_ZERO.ipynb", as_version=4)
    g = {"display": print, "__name__": "__notebook__"}
    for c in nb.cells:
        if c.cell_type == "code" and set(c.metadata.get("tags", [])) & DEFS:
            exec(compile(c.source, c.metadata["tags"][0], "exec"), g)
    import torch
    g["DEVICE"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return g


def zoology_tensors():
    """(train, test, independent) as our batch dicts: x, y (-1 = no answer), target_mask, source_pos, delay."""
    import torch
    from zoology.data.utils import prepare_data
    train_dl, test_dl = prepare_data(data_config())
    seg = lambda dl: (dl.dataset.inputs if hasattr(dl.dataset, "inputs") else torch.cat([b[0] for b in dl]),
                      dl.dataset.labels if hasattr(dl.dataset, "labels") else torch.cat([b[1] for b in dl]))
    ind = independent_segment()
    out = []
    for x, y in (seg(train_dl), seg(test_dl), (ind.inputs, ind.labels)):
        x, y = x.long().clone(), y.long().clone(); y[y == -100] = -1
        keys = x[:, 0:2 * N:2]
        slot = (x[:, :, None] == keys[:, None, :]).float().argmax(-1)
        source = torch.where(y.ne(-1), 2 * slot + 1, torch.full_like(x, -1))
        mask = y.ne(-1); pos = torch.arange(T).expand_as(x)
        out.append(dict(x=x, y=y, target_mask=mask, source_pos=source, delay=torch.where(mask, pos - source, torch.full_like(x, -1))))
    if SMOKE:
        out = [{k: v[:n] for k, v in d.items()} for d, n in zip(out, (256, 64, 64))]
    return out


def evaluate_ours(g, kind, model, data, cfg, anchor="query"):
    ev = g["answer_logits"](kind, model, data, cfg, anchor)
    lg = ev[ev.delay >= cfg.long_delay]
    return dict(acc=float(ev.correct.mean()), long_acc=float(lg.correct.mean()), cand_acc=float(ev.cand_correct.mean()))


def run_ours_rnn(lr, seed=201):
    import torch
    run_id = f"ours_rnn_bptt_d64_lr{lr:.2e}"
    out = OUT / run_id
    if (out / "summary.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    g = notebook_namespace(); dev = g["DEVICE"]
    cfg = g["Config"](task="mqar", vocab=V, seq_len=T, num_kv_pairs=N, long_delay=T // 4, future_window=T, eval_batch_size=256)
    train, test, ind = zoology_tensors()
    _, rnn = g["make_models"](cfg, seed)
    opt = torch.optim.AdamW(rnn.parameters(), lr=lr, weight_decay=0.1)                     # Zoology optimiser
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=64, eta_min=0.0)         # per epoch, no warm-up
    (out / "config.json").write_text(json.dumps(dict(family="ours_rnn", lr=lr, seed=seed, protocol="zoology figure-2", config=dict(vars(cfg)) if hasattr(cfg, "__dict__") else str(cfg)), indent=2, default=str))
    gen = torch.Generator().manual_seed(seed); hist = open(out / "history.jsonl", "a"); tic = time.perf_counter()
    for epoch in range(1 if SMOKE else 64):
        rnn.train(); perm = torch.randperm(len(train["x"]), generator=gen); losses = []
        for i in range(0, len(perm), 512):
            idx = perm[i:i + 512]; batch = {k: v[idx] for k, v in train.items()}
            opt.zero_grad(set_to_none=True)
            loss, _ = g["bptt_loss"](rnn, batch, cfg, False)                              # full BPTT, masked CE
            loss.backward(); opt.step(); losses.append(float(loss.detach()))
        sched.step(); rnn.eval()
        m = evaluate_ours(g, "rnn", rnn, test, cfg)
        hist.write(json.dumps(dict(epoch=epoch, train_loss=float(np.mean(losses)), **{f"valid/{k}": v for k, v in m.items()},
                                   seconds=time.perf_counter() - tic)) + "\n"); hist.flush()
        print(run_id, epoch, m, flush=True)
        if m["acc"] > 0.99:
            break
    s = dict(run_id=run_id, family="ours_rnn", lr=lr, epochs_run=epoch + 1, final_valid_accuracy=m["acc"],
             independent=evaluate_ours(g, "rnn", rnn, ind, cfg), wall_seconds=time.perf_counter() - tic,
             num_parameters=sum(p.numel() for p in rnn.parameters()))
    (out / "summary.json").write_text(json.dumps(s, indent=2)); print(json.dumps(s), flush=True)


def run_ours_smt(seed=201, smt_steps=20000, dmt_steps=10000, micro=8, accum=4, eval_every=1000):
    import torch
    if SMOKE:
        smt_steps, dmt_steps, micro, accum, eval_every = 2, 2, 2, 2, 1
    run_id = "ours_smt_d64"
    out = OUT / run_id
    if (out / "summary.json").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    g = notebook_namespace(); dev = g["DEVICE"]
    cfg = g["Config"](task="mqar", vocab=V, seq_len=T, num_kv_pairs=N, long_delay=T // 4, future_window=T, eval_batch_size=64)
    train, test, ind = zoology_tensors()
    teacher, rnn = g["make_models"](cfg, seed)
    (out / "config.json").write_text(json.dumps(dict(family="ours_smt", seed=seed, smt_steps=smt_steps, dmt_steps=dmt_steps, batch=micro * accum,
                                                     micro_batch=micro, protocol="ours (warm-up 200 + constant, clip 1.0)"), indent=2))
    hist = open(out / "history.jsonl", "a"); gen = torch.Generator().manual_seed(seed); tic = time.perf_counter()
    order = torch.randperm(len(train["x"]), generator=gen); cursor = [0]

    def next_batch(n):
        nonlocal order
        if cursor[0] + n > len(order):
            order = torch.randperm(len(train["x"]), generator=gen); cursor[0] = 0
        idx = order[cursor[0]:cursor[0] + n]; cursor[0] += n
        return {k: v[idx] for k, v in train.items()}

    def stage(name, steps, lr, params, loss_fn, monitor):
        opt, sched = g["make_optimizer"](params, lr, cfg, steps)
        best, best_state = float("inf"), None
        for step in range(1, steps + 1):
            opt.zero_grad(set_to_none=True); total = 0.0
            for _ in range(accum):
                loss = loss_fn(next_batch(micro)) / accum; loss.backward(); total += float(loss.detach())
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
            if step % eval_every == 0 or step == steps:
                score, m = monitor()
                hist.write(json.dumps(dict(stage=name, step=step, train_loss=total, **m, seconds=time.perf_counter() - tic)) + "\n"); hist.flush()
                print(run_id, name, step, m, flush=True)
                if score < best:
                    best = score; best_state = {k: v.detach().clone() for k, v in (teacher.state_dict() if name == "smt" else rnn.state_dict()).items()}
        return best_state

    def smt_monitor():
        g["copy_readout"](teacher, rnn); teacher.eval()
        ev = g["answer_logits"]("teacher", teacher, test, cfg)
        m = {"valid/teacher_acc": float(ev.correct.mean()),
             "valid/teacher_pre_acc": evaluate_ours(g, "teacher_pre", teacher, test, cfg, "after_pairs")["acc"],
             "valid/rnn_acc": evaluate_ours(g, "rnn", rnn, test, cfg)["acc"]}
        return float(ev.ce.mean()), m

    def smt_loss(b):
        teacher.train(); rnn.train()
        with g["seeded"](int(torch.randint(0, 2**31, (1,), generator=gen))):
            return g["smt_loss"](teacher, rnn, b, cfg, torch.Generator().manual_seed(0))[0]

    best = stage("smt", smt_steps, 3e-4, [*teacher.parameters(), *rnn.parameters()], smt_loss, smt_monitor)
    teacher.load_state_dict(best); g["copy_readout"](teacher, rnn); teacher.requires_grad_(False); teacher.eval()
    res = dict(smt_teacher=evaluate_ours(g, "teacher", teacher, ind, cfg),
               smt_teacher_pre_after_pairs=evaluate_ours(g, "teacher_pre", teacher, ind, cfg, "after_pairs"),
               rnn_before_dmt=evaluate_ours(g, "rnn", rnn, ind, cfg))

    def rnn_monitor():
        ev = g["answer_logits"]("rnn", rnn, test, cfg)
        return float(ev.ce.mean()), {"valid/rnn_acc": float(ev.correct.mean())}

    best = stage("dmt", dmt_steps, 1e-4, list(rnn.parameters()), lambda b: (rnn.train(), g["dmt_loss"](teacher, rnn, b, cfg)[0])[1], rnn_monitor)
    rnn.load_state_dict(best)
    res["rnn_after_dmt"] = evaluate_ours(g, "rnn", rnn, ind, cfg)
    s = dict(run_id=run_id, family="ours_smt", independent=res, wall_seconds=time.perf_counter() - tic,
             note="selected checkpoints by Zoology-test CE; Zoology protocol infeasible for SMT, budget: 6.4 + 3.2 epochs")
    (out / "summary.json").write_text(json.dumps(s, indent=2)); print(json.dumps(s), flush=True)


# =====================================================================================================================
def jobs():
    j = [["--family", "zoology", "--mixer", m, "--d", str(d), "--lr", str(lr)] for m in ("attention", "base_conv", "h3", "hyena", "based")
         for d in (64, 128) for lr in LRS]
    j += [["--family", "ours_rnn", "--lr", str(lr)] for lr in LRS]
    j.insert(0, ["--family", "ours_smt"])                       # longest job first
    return j


NEED_MIB = {"zoology": 4000, "ours_rnn": 10000, "ours_smt": 9000}     # start a job only with this much free GPU memory


def queue(max_par=2):
    """Starts are serialised across ALL queues on this GPU by /tmp/smt_gpu_start.lock (also used by run_experiments.py):
    under the lock, wait until the job's memory need is free, start it, wait 120 s for its allocation, release."""
    import fcntl
    procs, todo = [], jobs()
    (OUT / "logs").mkdir(parents=True, exist_ok=True)
    free_mib = lambda: int(subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                                          capture_output=True, text=True).stdout.split()[0])
    while todo or procs:
        procs = [p for p in procs if p.poll() is None]
        if todo and len(procs) < max_par:
            with open("/tmp/smt_gpu_start.lock", "w") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                while free_mib() < NEED_MIB[todo[0][1]]:
                    time.sleep(30)
                a = todo.pop(0); name = "_".join(x.lstrip("-") for x in a[1:])
                procs.append(subprocess.Popen([sys.executable, __file__, *a], cwd=ROOT, stdout=open(OUT / "logs" / f"{name}.log", "a"),
                                              stderr=subprocess.STDOUT))
                print(time.strftime("%F %T"), "started", name, f"free {free_mib()} MiB", flush=True)
                time.sleep(120)
        time.sleep(15)
    print(time.strftime("%F %T"), "queue finished", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["zoology", "ours_rnn", "ours_smt"])
    ap.add_argument("--mixer"); ap.add_argument("--d", type=int); ap.add_argument("--lr", type=float)
    ap.add_argument("--queue", action="store_true")
    a = ap.parse_args()
    assert subprocess.run(["git", "-C", str(ZOO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == PINNED
    OUT.mkdir(parents=True, exist_ok=True)
    if a.queue:
        queue(int(os.environ.get("MAX_PAR", 2)))
    elif a.family == "zoology":
        run_zoology(a.mixer, a.d, a.lr)
    elif a.family == "ours_rnn":
        run_ours_rnn(a.lr)
    else:
        run_ours_smt()
