#!/usr/bin/env python3
"""Job list of the minimal diagnostic queue (16.10.2026) for checks/gpu_jobs.py. List order = priority.
  1. Stack version 4, part D (proposals/preregistration_v4.md): base_window, ctx_far, bptt_full x 3 LR, bptt_1 x 3 LR;
  2. MQAR check A on pilot01 (mqar_baselines.py): overfitting + generalisation, Transformer and RNN full BPTT;
  3. Zoology check B (zoology_repro/): smoke cell N=64 first, the main cell N=256 only after the smoke succeeded.
Nothing here creates a test split. The final comparison of version 4 is NOT in this queue: it needs the protocol
fixed from part D.

  python checks/make_jobs_v4.py [--groups stack,mqar,zoology] [--python PY] > runs/jobs_v4.json
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ap = argparse.ArgumentParser()
ap.add_argument("--groups", default="stack,mqar,zoology")
ap.add_argument("--python", default="/workspace/envs/smt-memup/bin/python")
a = ap.parse_args()
groups = a.groups.split(",")
py = a.python
jobs = []
if "stack" in groups:
    names = ["base_window", "ctx_far"] + [f"{v}_lr{lr:g}" for v in ("bptt_full", "bptt_1") for lr in (1e-4, 3e-4, 1e-3)]
    for n in names:
        jobs.append(dict(name=f"v4d_{n}", min_free=9000 if n in ("base_window", "ctx_far") else 3000, after=[],
                         cmd=shlex.join([py, "smt_memup_v4.py", "--phase", "diag", "--job", n, "--threads", "4"])))
    jobs.append(dict(name="v4d_report", min_free=2000, after=[j["name"] for j in jobs],
                     cmd=shlex.join([py, "smt_memup_v4.py", "--phase", "diag_report", "--threads", "4"])))
if "mqar" in groups:
    for n in ("tf_overfit256", "rnn_overfit256", "tf_lr3e-4", "tf_lr1e-3", "rnn_lr1e-4", "rnn_lr3e-4", "rnn_lr1e-3"):
        jobs.append(dict(name=f"mqarA_{n}", min_free=3000, after=[],
                         cmd=shlex.join([py, "mqar_baselines.py", "--job", n, "--threads", "4"])))
if "zoology" in groups:
    env = "PYTHONPATH=upstream/zoology"
    jobs.append(dict(name="zooB_smoke_n64", min_free=8000, after=[],
                     cmd=f"env ZOO_CELL=n64_d64 {env} {shlex.quote(py)} zoology_repro/run_repro.py"))
    jobs.append(dict(name="zooB_main_n256", min_free=12000, after=["zooB_smoke_n64"],
                     cmd=f"env ZOO_CELL=n256_d64 {env} {shlex.quote(py)} zoology_repro/run_repro.py"))
print(json.dumps(jobs, indent=1))
