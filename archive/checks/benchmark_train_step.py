#!/usr/bin/env python3
"""Benchmark of a FULL training step (zero_grad, forward, backward, gradient clipping, optimizer.step) for every model
of the planned queue. Throw-away models with the experiment sizes; no experiment state is touched.

Method per component: pre-generated batches (data generation is timed separately, on CPU); WARMUP steps not measured;
then STEPS measured steps, each bracketed by CUDA events (GPU time) and synchronize + perf_counter (wall time);
peak allocated / reserved memory after torch.cuda.reset_peak_memory_stats() at the start of the measured steps.
Evaluation passes (validation monitoring) are timed separately.

  python checks/benchmark_train_step.py [--steps 20 --warmup 5] [--only smt_ctx_far,bptt_full] [--out runs/benchmark]

The projection printed at the end is an ESTIMATE from these measurements (single process, no other GPU jobs); parallel
processes on one GPU do not scale linearly.
"""
import argparse
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smt_memup_stack as S  # noqa: E402
import smt_memup_v4 as V  # noqa: E402
import smt_memup_window as W  # noqa: E402


def timed_steps(step_fn, params, batches, warmup, steps, device, clip=1.0):
    opt = torch.optim.AdamW(params, lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)
    cuda = device.type == "cuda"
    gpu_ms, wall_ms = [], []
    for i in range(warmup + steps):
        if i == warmup and cuda:
            torch.cuda.synchronize(device); torch.cuda.reset_peak_memory_stats(device)
        if cuda:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize(device)
        tic = time.perf_counter()
        if cuda:
            start.record()
        opt.zero_grad(set_to_none=True)
        loss = step_fn(batches[i % len(batches)], i)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, clip)
        opt.step()
        if cuda:
            end.record(); torch.cuda.synchronize(device)
        if i >= warmup:
            wall_ms.append(1e3 * (time.perf_counter() - tic))
            if cuda:
                gpu_ms.append(start.elapsed_time(end))
    out = dict(wall_ms_median=float(np.median(wall_ms)), wall_ms_p90=float(np.quantile(wall_ms, .9)),
               measured_steps=steps, warmup_steps=warmup)
    if cuda:
        out.update(gpu_ms_median=float(np.median(gpu_ms)),
                   peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
                   peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20)
    return out


def timed(fn, device, repeats=2):
    times = []
    for _ in range(repeats):
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        tic = time.perf_counter()
        fn()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(time.perf_counter() - tic)
    return float(min(times))


def stack_components(cfg, device, dets):
    comps = {}
    gen_batch = lambda bs, s: S.sample_stack(cfg, bs, S.derive_seed(9001, s))
    for variant in ("base_window", "ctx_far", "surprise_far"):
        def make(variant=variant):
            teacher, rnn = S.make_models(cfg, 0, device)
            params = [*teacher.parameters(), *rnn.parameters()]
            def step(batch, i):
                teacher.train(); rnn.train()
                return W.smt_window_loss(teacher, rnn, batch, cfg, variant, dets, torch.Generator().manual_seed(i), device)[0]
            return step, params, cfg.batch_size
        comps[f"smt_{variant}"] = make

    def make_dmt():
        teacher, rnn = S.make_models(cfg, 0, device)
        teacher.requires_grad_(False)
        def step(batch, i):
            teacher.eval(); rnn.train()
            return S.dmt_loss(teacher, rnn, batch, cfg, device)[0]
        return step, list(rnn.parameters()), cfg.dmt_batch_size
    comps["dmt"] = make_dmt
    for variant in V.BPTT_VARIANTS:
        def make(variant=variant):
            _, rnn = S.make_models(cfg, 0, device)
            def step(batch, i):
                rnn.train()
                return V.bptt_loss(rnn, batch, cfg, device, truncate=variant == "bptt_1")[0]
            return step, list(rnn.parameters()), cfg.bptt_batch_size
        comps[variant] = make
    for kind in ("window", "context"):
        def make(kind=kind):
            det = (S.WindowDetector if kind == "window" else S.ContextDetector)(cfg).to(device)
            def step(batch, i):
                det.train()
                return S.detector_train_loss(det, kind, batch, cfg, device)
            return step, list(det.parameters()), cfg.detector_batch_size
        comps[f"detector_{kind}"] = make
    return comps, gen_batch


def mqar_components(device):
    import mqar_baselines as B
    import smt_memup_mqar as M
    comps = {}
    for model in ("transformer", "rnn_bptt"):
        def make(model=model):
            cfg = replace(B.BaselineConfig(), model=model)
            net = B.make_model(cfg, device)
            def step(batch, i):
                net.train()
                return B.target_loss(net, batch, device)
            return step, list(net.parameters()), cfg.batch_size
        comps[f"mqar_{model}"] = make
    dcfg = B.data_config(B.BaselineConfig())
    return comps, (lambda bs, s: M.sample_mqar(dcfg, bs, S.derive_seed(9002, s)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--out", default=str(ROOT / "runs" / "benchmark"))
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    device = S.setup_runtime(a.threads)
    S.sources(ROOT)
    cfg = V.V4Config()
    dets = W.load_v1_detectors(cfg, ROOT, device)
    stack, gen_stack = stack_components(cfg, device, dets)
    mqar, gen_mqar = mqar_components(device)
    only = set(a.only.split(",")) - {""}
    res = dict(device=torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
               torch=torch.__version__, measured=time.strftime("%F %T"), components={})
    for group, comps, gen in (("stack", stack, gen_stack), ("mqar", mqar, gen_mqar)):
        for name, make in comps.items():
            if only and name not in only:
                continue
            step, params, bs = make()
            tic = time.perf_counter()
            batches = [gen(bs, s) for s in range(4)]
            data_ms = 1e3 * (time.perf_counter() - tic) / 4
            r = timed_steps(step, params, batches, a.warmup, a.steps, device)
            r.update(batch_size=bs, data_generation_ms_per_batch=data_ms, parameters=sum(p.numel() for p in params))
            res["components"][name] = r
            print(name, json.dumps(r), flush=True)
            del step, params, batches
            if device.type == "cuda":
                torch.cuda.empty_cache()
    # evaluation passes used for monitoring (every eval_every steps)
    val = S.sample_stack(cfg, cfg.validation_examples, S.derive_seed(501, 0))
    teacher, rnn = S.make_models(cfg, 0, device)
    res["eval_smt_s"] = timed(lambda: (S.evaluate(teacher, rnn, val, cfg, device),
                                       W.far_probe(teacher, S.slice_batch(val, 0, 128), cfg, device)), device)
    res["eval_dmt_s"] = timed(lambda: S.evaluate(teacher, rnn, val, cfg, device), device)
    res["eval_bptt_s"] = timed(lambda: V.evaluate_rnn(rnn, val, cfg, device), device)
    print(json.dumps({k: v for k, v in res.items() if k.startswith("eval")}))
    c = res["components"]
    ms = lambda k: (c[k]["wall_ms_median"] + c[k]["data_generation_ms_per_batch"]) / 1e3 if k in c else float("nan")
    evals = lambda steps: steps / cfg.eval_every
    proj = {
        "diag_smt_one_fit_h": (cfg.smt_steps * ms("smt_ctx_far") + evals(cfg.smt_steps) * res["eval_smt_s"]
                               + cfg.dmt_steps * ms("dmt") + evals(cfg.dmt_steps) * res["eval_dmt_s"]) / 3600,
        "diag_bptt_full_one_fit_h": (cfg.bptt_steps * ms("bptt_full") + evals(cfg.bptt_steps) * res["eval_bptt_s"]) / 3600,
        "diag_bptt_1_one_fit_h": (cfg.bptt_steps * ms("bptt_1") + evals(cfg.bptt_steps) * res["eval_bptt_s"]) / 3600,
    }
    proj["diag_total_sequential_h"] = (2 * proj["diag_smt_one_fit_h"] + 3 * proj["diag_bptt_full_one_fit_h"]
                                       + 3 * proj["diag_bptt_1_one_fit_h"])
    res["projection_estimate"] = proj
    print("ESTIMATE (from the measured medians, single process):", json.dumps(proj, indent=2))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    f = out / f"bench_{device.type}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    f.write_text(json.dumps(res, indent=2))
    print("written", f)


if __name__ == "__main__":
    main()
