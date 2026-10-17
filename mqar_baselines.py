"""MQAR, check A: is OUR configuration pilot01 learnable by ordinary models trained directly on the answers?

  * transformer: an ordinary causal Transformer (x-transformers Decoder, pinned 2.18.5), next-token style: the logits
    at the query key x_t predict y_t (Zoology's label alignment: labels are already shifted by the generator);
  * rnn_bptt: OUR recurrent model (author SMT RNN class, the same sizes as pilot01: state 8 x 64, updater depth 2,
    readout decoder depth 2), trained from scratch with full BPTT through the 64 updates (authors' forward_bptt loss).
    This is NOT a reproduction of any Zoology architecture.

Data, splits and metrics are those of pilot01 (smt_memup_mqar.py, imported, not modified): the Zoology generator at
the pinned commit (random_non_queries=False, power_a=0.01), validation stream 511, tuning stream 512.
Training data: either a fresh stream (pilot01's SMT stream 301, batch 32) or a FIXED small training set (overfitting
check). Metrics at target positions only:
  * full-vocabulary accuracy and CE (comparable with Zoology's accuracy, which is over the vocabulary);
  * accuracy / CE among the N values written in the sequence (chance 1/N = 12.5% for pilot01; NOT comparable with
    full-vocabulary numbers);
  * pred_in_written (the model knows the value set), first query (query_rank 0), every query rank;
  * the same on a shuffled query order (the order carries no information about the answer).
Checkpoint: best validation CE on long answers (delay >= long_delay, as pilot01). LR: lowest tuning CE on long answers.
Importing this file never trains anything.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import platform
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import smt_memup_mqar as M
import smt_memup_stack as S

MODULE_PATH = Path(__file__).resolve()
DEFAULT_ROOT = MODULE_PATH.parent
IGNORE = S.IGNORE


@dataclass(frozen=True)
class BaselineConfig:
    data: str = "pilot01"                # name in smt_memup_mqar.CANDIDATES
    kv_pairs: int = 0                    # > 0: override num_kv_pairs of the data config (pairs-count sweep, 17.10)
    model: str = "transformer"           # transformer | rnn_bptt
    # transformer (ordinary causal LM; learned absolute positions, pre-norm, GELU MLP x4)
    tf_d_model: int = 64
    tf_depth: int = 2
    tf_heads: int = 4
    # optimisation
    lr: float = 1e-3
    steps: int = 10000
    batch_size: int = 32
    train_set: int = 0                   # 0 = fresh sequences every step; > 0 = fixed training set of this size
    schedule: str = "warmup_constant"
    warmup_steps: int = 200
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    eval_every: int = 500
    eval_examples: int = 512             # validation / tuning size (pilot01: 512)
    seed: int = 107                      # pilot01 tuning seed
    stop_accuracy: float = 0.0           # > 0: stop once validation long-answer accuracy reaches it (saves GPU time)


def data_config(cfg: BaselineConfig) -> M.MQARConfig:
    return M.candidate(cfg.data, num_kv_pairs=cfg.kv_pairs) if cfg.kv_pairs else M.candidate(cfg.data)


def code_sha256() -> str:
    return hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest()


def run_identity(cfg: BaselineConfig, device: torch.device) -> dict:
    src = S.sources()
    return dict(task="mqar_baselines", config=asdict(cfg), data=asdict(data_config(cfg)), code_sha256=code_sha256(),
                mqar_code_sha256=M.code_sha256(), stack_code_sha256=S.code_sha256(), zoology=M.ZOOLOGY_REVISION,
                repositories=src.revisions, torch=str(torch.__version__),
                x_transformers=importlib.metadata.version("x-transformers"), device=device.type,
                python=platform.python_version())


# --------------------------------------------------------------------------------------
# Models: both map x (B, T) to logits (B, T, V), logits[:, t] predicts y[:, t]
# --------------------------------------------------------------------------------------

class CausalTransformer(torch.nn.Module):
    def __init__(self, cfg: BaselineConfig, dcfg: M.MQARConfig):
        super().__init__()
        from x_transformers import Decoder, TransformerWrapper
        self.net = TransformerWrapper(num_tokens=dcfg.vocab, max_seq_len=dcfg.seq_len,
                                      attn_layers=Decoder(dim=cfg.tf_d_model, depth=cfg.tf_depth, heads=cfg.tf_heads))

    def forward(self, x):
        return self.net(x)


class RecurrentLM(torch.nn.Module):
    """Author RNN rolled out from the zero state; readout of every state (as forward_bptt)."""

    def __init__(self, rnn):
        super().__init__()
        self.rnn = rnn

    def forward(self, x):
        states = S.rnn_rollout(self.rnn, x)                                   # full BPTT: no detach
        return self.rnn.rnn_readout(states.flatten(0, 1)).reshape(*x.shape, -1)


def make_model(cfg: BaselineConfig, device) -> torch.nn.Module:
    dcfg = data_config(cfg)
    with S.isolated_seed(cfg.seed):
        if cfg.model == "transformer":
            model = CausalTransformer(cfg, dcfg)
        elif cfg.model == "rnn_bptt":
            model = None
        else:
            raise ValueError(cfg.model)
    if cfg.model == "rnn_bptt":
        model = RecurrentLM(S.make_models(dcfg, cfg.seed, device)[1])          # same init as the pilot01 RNN
    return model.to(device)


def target_loss(model, batch: dict, device):
    logits = model(batch["x"].to(device))
    ce = F.cross_entropy(logits.flatten(0, 1).float(), batch["y"].clamp(min=0).flatten().to(device),
                         reduction="none").reshape(batch["y"].shape)
    return S.pair_mean(ce, batch["target_mask"].to(device))


# --------------------------------------------------------------------------------------
# Evaluation at target positions
# --------------------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, batch: dict, dcfg: M.MQARConfig, device, chunk: int = 128, **labels) -> pd.DataFrame:
    model.eval()
    N = dcfg.num_kv_pairs
    frames = []
    for lo in range(0, len(batch["x"]), chunk):
        part = S.slice_batch(batch, lo, lo + chunk)
        x = part["x"]
        logits = model(x.to(device)).float().cpu()
        b, t = torch.where(part["target_mask"])
        y = part["y"][b, t]
        lg = logits[b, t]
        written = x[:, 1:2 * N:2][b]
        cand = lg.gather(1, written)
        cand_y = (written == y[:, None]).float().argmax(1)
        pred = lg.argmax(-1)
        frames.append(pd.DataFrame(dict(
            **labels, sequence=(b + lo).numpy(), t=t.numpy(), delay=part["delay"][b, t].numpy(), target=y.numpy(),
            query_rank=(part["target_mask"].long().cumsum(1)[b, t] - 1).numpy(),
            pair_slot=((part["source_pos"][b, t] - 1) // 2).numpy(),
            correct=(pred == y).numpy(), ce=F.cross_entropy(lg, y, reduction="none").numpy(),
            pred_in_written=(pred[:, None] == written).any(1).numpy(),
            cand_correct=(cand.argmax(1) == cand_y).numpy(), cand_ce=F.cross_entropy(cand, cand_y, reduction="none").numpy())))
    return pd.concat(frames, ignore_index=True)


def summary(ev: pd.DataFrame, dcfg: M.MQARConfig) -> pd.DataFrame:
    groups = {"all": ev, f"delay>={dcfg.long_delay}": ev[ev.delay >= dcfg.long_delay],
              f"delay<{dcfg.long_delay}": ev[ev.delay < dcfg.long_delay], "first query": ev[ev.query_rank == 0]}
    keys = [k for k in ("name", "split", "checkpoint") if k in ev.columns]
    rows = []
    for g, d in groups.items():
        t = d.groupby(keys).agg(answers=("correct", "size"), accuracy=("correct", "mean"), ce=("ce", "mean"),
                                cand_accuracy=("cand_correct", "mean"), cand_ce=("cand_ce", "mean"),
                                pred_in_written=("pred_in_written", "mean")).reset_index() if keys else \
            pd.DataFrame([dict(answers=len(d), accuracy=d.correct.mean(), ce=d.ce.mean(), cand_accuracy=d.cand_correct.mean(),
                               cand_ce=d.cand_ce.mean(), pred_in_written=d.pred_in_written.mean())])
        t.insert(0, "group", g)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def _long(ev, dcfg):
    lg = ev[ev.delay >= dcfg.long_delay]
    return dict(val_long_ce=float(lg.ce.mean()), val_long_acc=float(lg.correct.mean()),
                val_long_cand_acc=float(lg.cand_correct.mean()), val_all_acc=float(ev.correct.mean()),
                val_first_query_acc=float(ev[ev.query_rank == 0].correct.mean()))


# --------------------------------------------------------------------------------------
# Training (resumable at every evaluation)
# --------------------------------------------------------------------------------------

def splits(dcfg: M.MQARConfig, n: int) -> tuple[dict, dict]:
    validation = M.sample_mqar(dcfg, n, S.derive_seed(511, 0))                # pilot01 validation / tuning streams
    tuning = M.sample_mqar(dcfg, n, S.derive_seed(512, 0))
    assert not set(S.fingerprints(validation)) & set(S.fingerprints(tuning))
    return validation, tuning


def fixed_train_set(cfg: BaselineConfig, dcfg: M.MQARConfig, avoid: set) -> dict:
    data = M.sample_mqar(dcfg, cfg.train_set, S.derive_seed(515, cfg.seed))
    assert not set(S.fingerprints(data)) & avoid, "fixed training set overlaps validation/tuning"
    return data


def fit(cfg: BaselineConfig, out: Path, validation: dict, tuning: dict, device, log=print) -> dict:
    dcfg = data_config(cfg)
    out.mkdir(parents=True, exist_ok=True)
    identity = dict(run=run_identity(cfg, device), validation=S.split_hash(validation))
    done = out / "completed.pt"
    if done.exists():
        saved = torch.load(done, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity, f"{done}: identity mismatch"
        return saved
    model = make_model(cfg, device)
    initial_hash = S.state_hash(model)
    params = list(model.parameters())
    opt = torch.optim.AdamW(params, lr=cfg.lr, betas=(0.9, 0.95), weight_decay=cfg.weight_decay)
    warm = max(1, min(cfg.warmup_steps, cfg.steps // 10))
    if cfg.schedule == "warmup_constant":
        lam = lambda s: (s + 1) / warm if s < warm else 1.0
    else:
        lam = S.lr_lambda(replace(M.MQARConfig(), warmup_steps=cfg.warmup_steps), cfg.steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    train = fixed_train_set(cfg, dcfg, set(S.fingerprints(validation)) | set(S.fingerprints(tuning))) if cfg.train_set else None
    last, best_path = out / "last.pt", out / "best.pt"
    start, history, best, cost = 0, [], float("inf"), dict(train_seconds=0.0, eval_seconds=0.0)
    if last.exists():
        saved = torch.load(last, map_location="cpu", weights_only=True)
        assert saved["identity"] == identity and saved["initial_hash"] == initial_hash
        model.load_state_dict(saved["model"]); opt.load_state_dict(saved["optimizer"]); sched.load_state_dict(saved["scheduler"])
        start, history, best, cost = saved["step"], saved["history"], saved["best"], saved["cost"]
        log(f"resume {out.name} at {start}/{cfg.steps}")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device); torch.cuda.synchronize(device)
    tic, stopped = time.perf_counter(), False
    for step in range(start, cfg.steps):
        model.train()
        if train is None:
            batch = M.sample_mqar(dcfg, cfg.batch_size, S.derive_seed(301, cfg.seed, step))    # pilot01 SMT stream
        else:
            idx = torch.randint(0, cfg.train_set, (cfg.batch_size,), generator=torch.Generator().manual_seed(S.derive_seed(316, cfg.seed, step)))
            batch = {k: v[idx] for k, v in train.items()}
        opt.zero_grad(set_to_none=True)
        loss = target_loss(model, batch, device)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{out.name} step {step}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(params, cfg.clip_grad)
        opt.step(); sched.step()
        row = dict(step=step + 1, lr=opt.param_groups[0]["lr"], loss=float(loss.detach()), grad_norm=float(norm))
        monitor = (step + 1) % cfg.eval_every == 0 or step + 1 == cfg.steps
        if monitor:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t_eval = time.perf_counter(); cost["train_seconds"] += t_eval - tic
            ev = evaluate(model, validation, dcfg, device)
            row.update(_long(ev, dcfg))
            if train is not None:
                tr = evaluate(model, S.slice_batch(train, 0, min(cfg.train_set, 512)), dcfg, device)
                row.update(train_set_acc=float(tr.correct.mean()), train_set_long_acc=float(tr[tr.delay >= dcfg.long_delay].correct.mean()))
            if row["val_long_ce"] < best:
                best = row["val_long_ce"]
                torch.save(dict(identity=identity, step=step + 1, model=model.state_dict()), best_path)
            tic = time.perf_counter(); cost["eval_seconds"] += tic - t_eval
            if device.type == "cuda":
                cost["peak_allocated_mib"] = max(cost.get("peak_allocated_mib", 0), torch.cuda.max_memory_allocated(device) / 2**20)
                cost["peak_reserved_mib"] = max(cost.get("peak_reserved_mib", 0), torch.cuda.max_memory_reserved(device) / 2**20)
            row.update(train_seconds=cost["train_seconds"], eval_seconds=cost["eval_seconds"])
            log(f"{out.name} {step + 1}/{cfg.steps} val long acc={row['val_long_acc']:.3f} cand={row['val_long_cand_acc']:.3f} "
                f"first={row['val_first_query_acc']:.3f}" + (f" train_set acc={row['train_set_acc']:.3f}" if train is not None else ""))
            stopped = cfg.stop_accuracy > 0 and row["val_long_acc"] >= cfg.stop_accuracy
        history.append(row)
        if monitor:
            torch.save(dict(identity=identity, initial_hash=initial_hash, step=step + 1, model=model.state_dict(),
                            optimizer=opt.state_dict(), scheduler=sched.state_dict(), history=history, best=best,
                            cost=cost), last)
            pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        if stopped:
            log(f"{out.name}: validation long accuracy >= {cfg.stop_accuracy} at step {step + 1}, stopping (rule set in advance)")
            break
    last_state = copy.deepcopy(model.state_dict())
    best_saved = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(best_saved["model"])
    steps_done = history[-1]["step"]
    result = dict(identity=identity, initial_hash=initial_hash, model=model.state_dict(), last_model=last_state,
                  best_step=best_saved["step"], cost=dict(cost, optimizer_steps=steps_done,
                                                          sequences=steps_done * cfg.batch_size,
                                                          distinct_training_sequences=cfg.train_set or steps_done * cfg.batch_size))
    frames = []
    shuffled = M.shuffle_query_order(tuning, dcfg, S.derive_seed(514, 0))
    for which, state in (("best", result["model"]), ("last", last_state)):
        model.load_state_dict(state)
        for split, data in (("validation", validation), ("tuning", tuning), ("tuning_shuffled_order", shuffled)):
            frames.append(evaluate(model, data, dcfg, device, split=split, checkpoint=which))
        if train is not None:
            frames.append(evaluate(model, S.slice_batch(train, 0, min(cfg.train_set, 512)), dcfg, device,
                                   split="fixed_train_set", checkpoint=which))
    events = pd.concat(frames, ignore_index=True)
    summary(events, dcfg).to_csv(out / "summary.csv", index=False)
    (out / "cost.json").write_text(json.dumps(result["cost"], indent=2))
    torch.save(result, done)
    return result


# --------------------------------------------------------------------------------------
# The check-A queue (small; not a sweep of all models)
# --------------------------------------------------------------------------------------

JOBS = {
    # overfitting on a fixed small training set (capacity / optimisation): 256 sequences, ~2000 answers
    "tf_overfit256": dict(model="transformer", train_set=256, steps=3000, lr=1e-3),
    "rnn_overfit256": dict(model="rnn_bptt", train_set=256, steps=3000, lr=1e-3),
    # generalisation to independent sequences: pilot01 fresh stream, small LR grid, selection on tuning
    "tf_lr3e-4": dict(model="transformer", lr=3e-4, steps=10000, stop_accuracy=0.99),
    "tf_lr1e-3": dict(model="transformer", lr=1e-3, steps=10000, stop_accuracy=0.99),
    "rnn_lr1e-4": dict(model="rnn_bptt", lr=1e-4, steps=20000, stop_accuracy=0.99),
    "rnn_lr3e-4": dict(model="rnn_bptt", lr=3e-4, steps=20000, stop_accuracy=0.99),
    "rnn_lr1e-3": dict(model="rnn_bptt", lr=1e-3, steps=20000, stop_accuracy=0.99),
    # pairs-count sweep (17.10): where do the recurrent models break? (8 pairs = the runs above)
    "tf_kv2": dict(model="transformer", kv_pairs=2, lr=1e-3, steps=3000, stop_accuracy=0.99),
    "tf_kv4": dict(model="transformer", kv_pairs=4, lr=1e-3, steps=3000, stop_accuracy=0.99),
    "rnn_kv2": dict(model="rnn_bptt", kv_pairs=2, lr=1e-3, steps=6000, stop_accuracy=0.99),
    "rnn_kv4": dict(model="rnn_bptt", kv_pairs=4, lr=1e-3, steps=6000, stop_accuracy=0.99),
}


def run_dir(root: Path, device) -> Path:
    S.sources(root)
    ident = run_identity(BaselineConfig(), device)
    return Path(root) / "runs" / f"mqar_baselines_{hashlib.sha256(json.dumps(ident, sort_keys=True).encode()).hexdigest()[:12]}"


def report(root: Path, device) -> pd.DataFrame:
    """CPU-safe: collects summary.csv of finished jobs; LR choice per model = lowest tuning long CE (best checkpoint)."""
    d = run_dir(root, device)
    rows = []
    for name in JOBS:
        f = d / name / "summary.csv"
        if f.exists():
            s = pd.read_csv(f)
            s.insert(0, "job", name)
            rows.append(s)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def main(argv=None):
    p = argparse.ArgumentParser(description="MQAR check A: causal Transformer and RNN full BPTT on pilot01")
    p.add_argument("--job", choices=sorted(JOBS))
    p.add_argument("--report", action="store_true")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--gpu-memory-fraction", type=float, default=None)
    a = p.parse_args(argv)
    device = S.setup_runtime(a.threads, a.gpu_memory_fraction)
    root = DEFAULT_ROOT
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.report:
        pd.set_option("display.width", 250)
        print(report(root, torch.device("cuda")).to_string()); return
    d = run_dir(root, device)
    d.mkdir(parents=True, exist_ok=True)
    cfg = replace(BaselineConfig(), **JOBS[a.job])
    dcfg = data_config(cfg)
    validation, tuning = splits(dcfg, cfg.eval_examples)
    log(f"run directory: {d}")
    fit(cfg, d / a.job, validation, tuning, device, log)


if __name__ == "__main__":
    main()
