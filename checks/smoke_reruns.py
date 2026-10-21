#!/usr/bin/env python3
"""CPU smoke test of every run family in reruns/plan.json: each spec with tiny sizes and 4 steps, in a temp folder.
It only checks that the code path runs end to end (training, selection, evaluation, summary.json); no numbers."""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

import smt_minimal as C  # noqa: E402

TINY = dict(d_model=32, n_heads=2, smt_steps=4, dmt_steps=2, bptt_steps=4, lm_steps=4, eval_every=2,
            validation_examples=16, eval_examples=16, batch_size=8, warmup_steps=1)

if __name__ == "__main__":
    torch.set_num_threads(4)
    plan = json.loads((ROOT / "reruns" / "plan.json").read_text())["runs"]
    seen = {}
    for spec in plan:                                     # one spec per (task, name without seed)
        seen.setdefault(spec["name"].rsplit("_s", 1)[0], spec)
    with tempfile.TemporaryDirectory() as d:
        for key, spec in seen.items():
            spec = dict(spec, config={**spec["config"], **TINY, **({"seq_len": 16, "max_stack_size": 4, "long_delay": 4}
                                                                  if spec["config"].get("task", "stack") == "stack" else {})})
            s = C.run(spec, Path(d), torch.device("cpu"), log=lambda m: None)
            print(f"OK {key}: " + ", ".join(f"{k}={v:.2f}" for k, v in s.items() if k.endswith("long_acc")), flush=True)
    print(f"{len(seen)} families ran")
