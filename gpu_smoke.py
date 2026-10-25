#!/usr/bin/env python3
"""GPU smoke test of every FULL experiment of the notebook: executes the notebook's code cells (definitions only,
SMOKE mode) in this process, then runs fit() for every entry of EXPERIMENTS on CUDA with tiny sizes and 4 steps.
Catches device mismatches and shape errors that the CPU run cannot see. Numbers are meaningless.
    python gpu_smoke.py"""
import json
from dataclasses import replace
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent
DEFS = {"s0-setup", "s0-seeds", "s1-config", "s2-stack", "s2-mqar", "s3-windows", "s4-model", "s4-make", "s6-losses",
        "s8-train", "s9-eval", "s10-fit", "s13-full"}     # definition cells (the checks run in the SMOKE notebook)

if __name__ == "__main__":
    nb = nbformat.read(ROOT / "SMT_MemUP_FROM_ZERO.ipynb", as_version=4)
    g = {"display": print, "__name__": "__notebook__"}
    for c in nb.cells:
        if c.cell_type == "code" and set(c.metadata.get("tags", [])) & DEFS:
            exec(compile(c.source, c.metadata.get("tags", ["?"])[0], "exec"), g)
    import torch
    assert torch.cuda.is_available()
    g["DEVICE"] = torch.device("cuda")
    tiny = dict(smt_steps=4, dmt_steps=2, bptt_steps=4, lm_steps=4, eval_every=2, validation_examples=16,
                eval_examples=16, batch_size=8, warmup_steps=1)
    out = ROOT / "results" / "gpu_smoke"
    for name, e in g["EXPERIMENTS"].items():
        cfg = replace(e["config"], **tiny)
        d, s = g["fit"](dict(name=name, seed=1, family=e["family"], config=cfg), out_root=out, log=lambda m: None)
        keys = [k for k in s if k.startswith("eval_best") and k.endswith("long_acc")]
        assert keys, name
        print("OK", name, json.dumps({k: round(s[k], 3) for k in keys}), flush=True)
    print(f"GPU smoke: {len(g['EXPERIMENTS'])} experiments ran on {torch.cuda.get_device_name(0)}")
