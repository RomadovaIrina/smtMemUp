#!/usr/bin/env python3
"""Every experiment of configs/experiments.json runs end to end on CPU with tiny sizes (4 steps, temp folder):
training, early-stop rule, checkpoint selection, best/last evaluation, summary.json. Numbers are meaningless.
    python tests/smoke_runs.py"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

from smtcore.train import run  # noqa: E402

TINY = dict(d_model=32, n_heads=2, smt_steps=4, dmt_steps=2, bptt_steps=4, lm_steps=4, eval_every=2,
            validation_examples=16, eval_examples=16, batch_size=8, warmup_steps=1)

if __name__ == "__main__":
    torch.set_num_threads(4)
    exps = json.loads((ROOT / "configs" / "experiments.json").read_text())["experiments"]
    with tempfile.TemporaryDirectory() as d:
        for e in exps:
            cfg = {**e["config"], **TINY}
            if cfg.get("task", "stack") == "stack":
                cfg.update(seq_len=16, max_stack_size=4, long_delay=4)
            if e["config"].get("dmt_steps") == 0:
                cfg["dmt_steps"] = 0
            s = run(dict(name=e["name"], family=e["family"], seed=1, config=cfg), Path(d), torch.device("cpu"),
                    log=lambda m: None)
            keys = sorted(k for k in s if k.startswith("eval_best") and k.endswith("long_acc"))
            assert keys and all(k.replace("best", "last") in s for k in keys), e["name"]
            print(f"OK {e['name']}: " + ", ".join(k.replace("eval_best_", "") for k in keys), flush=True)
    print(f"{len(exps)} experiments ran")
