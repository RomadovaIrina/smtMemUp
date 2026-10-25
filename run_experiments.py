#!/usr/bin/env python3
"""Runs the FULL experiments of SMT_MemUP_FROM_ZERO.ipynb (§13) as a GPU queue.

Every run = the notebook executed top to bottom with MODE="FULL", FULL_EXPERIMENT=<name>, FULL_SEED=<seed>:
the CPU checks (§2-§12, incl. the equivalence with the old code) run first in every job, then §13 trains on the GPU
into results/notebook_runs/<name>_<time>/ (config.json, *_history.csv, *_best.pt, summary.json). The executed notebook
of each job is kept in results/notebook_runs/executed/<name>_s<seed>.ipynb, its log in results/notebook_runs/logs/.
Jobs whose executed notebook already exists are skipped. Order = priority (MQAR first, then Stack).

    MAX_PAR=3 python run_experiments.py                     # all experiments x seeds 201 202 203
    python run_experiments.py --only mqar_T16N2_smt --seeds 201
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
NB = ROOT / "SMT_MemUP_FROM_ZERO.ipynb"
OUT = ROOT / "results" / "notebook_runs"
ORDER = ["mqar_T16N2_smt", "mqar_T16N2_teacher_bottleneck", "mqar_T16N2_teacher_full", "mqar_T16N2_transformer",
         "mqar_T16N2_rnn_bptt_full", "mqar_T32N4_smt", "mqar_T32N4_transformer",
         "stack_smt_tf8", "stack_bptt_full", "stack_bptt_1", "stack_smt_tf8_ctx_far", "stack_smt_tf8_no_dyn",
         "stack_smt_tf8_target_detached", "stack_smt_tf8_fully_detached"]
HEAVY = {"stack_smt_tf8", "stack_smt_tf8_ctx_far", "stack_smt_tf8_no_dyn", "stack_smt_tf8_target_detached",
         "stack_smt_tf8_fully_detached"}                  # ~8 GB each in the SMT stage


def job(name: str, seed: int):
    done = OUT / "executed" / f"{name}_s{seed}.ipynb"
    if done.exists():
        return name, seed, "skipped (done)"
    nb = nbformat.read(NB, as_version=4)
    cell = next(c for c in nb.cells if c.cell_type == "code" and "s0-setup" in c.metadata.get("tags", []))
    src = cell.source.replace('MODE = "SMOKE"', 'MODE = "FULL"').replace("FULL_EXPERIMENT = None", f'FULL_EXPERIMENT = "{name}"')
    cell.source = src.replace("FULL_SEED = 201", f"FULL_SEED = {seed}")
    assert f'FULL_EXPERIMENT = "{name}"' in cell.source and f"FULL_SEED = {seed}" in cell.source
    tmp = OUT / "executed" / f"_{name}_s{seed}.ipynb"
    nbformat.write(nb, tmp)
    log = OUT / "logs" / f"{name}_s{seed}.log"
    tic = time.time()
    with open(log, "w") as f:
        rc = subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
                             "--ExecutePreprocessor.timeout=-1", "--output", done.name, str(tmp)],
                            cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    tmp.unlink(missing_ok=True)
    return name, seed, f"exit {rc} after {(time.time() - tic) / 60:.1f} min"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=ORDER)
    ap.add_argument("--seeds", nargs="*", type=int, default=[201, 202, 203])
    a = ap.parse_args()
    for d in ("executed", "logs"):
        (OUT / d).mkdir(parents=True, exist_ok=True)
    jobs = [(n, s) for n in ORDER if n in a.only for s in a.seeds]
    heavy = [j for j in jobs if j[0] in HEAVY]
    light = [j for j in jobs if j[0] not in HEAVY]
    status = OUT / "queue_status.jsonl"

    def run(j):
        r = job(*j)
        with open(status, "a") as f:
            f.write(json.dumps(dict(time=time.strftime("%F %T"), name=r[0], seed=r[1], result=r[2])) + "\n")
        print(time.strftime("%F %T"), *r, flush=True)

    # light jobs (MQAR, BPTT: < 2 GB) and heavy SMT jobs (~8 GB) in two pools so that at most MAX_HEAVY SMT stages share
    # the 24 GB card
    with ThreadPoolExecutor(int(os.environ.get("MAX_PAR", 3))) as light_pool, \
            ThreadPoolExecutor(int(os.environ.get("MAX_HEAVY", 2))) as heavy_pool:
        futures = [light_pool.submit(run, j) for j in light] + [heavy_pool.submit(run, j) for j in heavy]
        for f in futures:
            f.result()
