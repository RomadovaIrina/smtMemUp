#!/usr/bin/env python3
"""Job list for version 3 (converged regime) for checks/gpu_jobs.py. Seed-major order, so complete seed blocks
(all variants of one seed, paired by initial weights and data) finish first. The optional oracle diagnostic runs last.
The test report is NOT included: it needs the user's permission (see proposals/preregistration_v3.md).

  python checks/make_jobs_v3.py [--with-oracle] > runs/jobs_v3.json
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smt_memup_v3 as V  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--with-oracle", action="store_true")
ap.add_argument("--python", default="/mem/envs/smt-memup/bin/python")
a = ap.parse_args()
cfg = V.V3Config()
jobs = []
for seed in cfg.seeds:
    for v in V.VARIANTS_MAIN:
        jobs.append(dict(name=f"v3_s{seed}_{v}", min_free=10500, after=[],
                         cmd=shlex.join([a.python, "smt_memup_v3.py", "--phase", "fit", "--seed", str(seed), "--variant", v,
                                         "--threads", "4", "--gpu-memory-fraction", "0.30"])))
if a.with_oracle:
    for seed in cfg.seeds:
        jobs.append(dict(name=f"v3_s{seed}_oracle_prefix_k", min_free=10500, after=[],
                         cmd=shlex.join([a.python, "smt_memup_v3.py", "--phase", "fit", "--seed", str(seed),
                                         "--variant", "oracle_prefix_k", "--threads", "4", "--gpu-memory-fraction", "0.30"])))
print(json.dumps(jobs, indent=1))
