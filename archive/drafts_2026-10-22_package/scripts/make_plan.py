#!/usr/bin/env python3
"""configs/experiments.json x seeds -> configs/plan.json (run specs) and configs/gpu_jobs.json (queue for
scripts/gpu_jobs.py, ordered by tier). Run:  python scripts/make_plan.py [--python /path/to/python]"""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--python", default="python")
    a = ap.parse_args()
    exp = json.loads((ROOT / "configs" / "experiments.json").read_text())
    runs = [dict(name=f"{e['name']}_s{s}", family=e["family"], seed=s, tier=e["tier"], min_free_mib=e["min_free_mib"],
                 config=e["config"]) for e in sorted(exp["experiments"], key=lambda e: e["tier"]) for s in exp["seeds"]]
    (ROOT / "configs" / "plan.json").write_text(json.dumps(dict(seeds=exp["seeds"], runs=runs), indent=1))
    jobs = [dict(name=r["name"], min_free=r["min_free_mib"], after=[],
                 cmd=f"env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True {a.python} -m smtcore train --index {i}")
            for i, r in enumerate(runs)]
    (ROOT / "configs" / "gpu_jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"{len(runs)} runs: " + ", ".join(f"tier {t}: {sum(r['tier'] == t for r in runs)}" for t in (1, 2, 3)))
