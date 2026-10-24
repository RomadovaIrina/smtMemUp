#!/usr/bin/env python3
"""Train ONE (seed, variant) fit of the saved experiment, exactly as its final phase would (same run identity,
directory, locked alpha / config, detectors, data streams, resume). Lets independent final fits run as parallel
processes. The experiment modules are imported, not modified.

  python checks/fit_one.py --exp stack --seed 17 --variant oracle_prefix_k      # version 1 (smt_memup_stack.py)
  python checks/fit_one.py --exp window --seed 7 --variant surprise_far          # version 2 (smt_memup_window.py)
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smt_memup_stack as S  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", choices=["stack", "window"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--gpu-memory-fraction", type=float, default=None)
    a = ap.parse_args()
    device = S.setup_runtime(a.threads, a.gpu_memory_fraction)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.exp == "stack":
        ex = S.Experiment(S.Config(), root=ROOT, device=device, log=log)
        assert a.seed in ex.cfg.seeds and a.variant in S.ALL_VARIANTS
        alphas = json.loads((ex.dir / "locked_alphas.json").read_text())["alphas"]
        ex._need_detectors([a.variant])                 # loads the saved detectors (no training)
        log(f"run directory: {ex.dir}")
        S.fit(ex.cfg, a.variant, alphas[a.variant], a.seed, ex.dir / "final" / f"seed{a.seed}" / a.variant,
              ex.detectors, ex.validation, device, log)
    else:
        import smt_memup_window as W
        ex = W.Experiment(W.WindowConfig(), root=ROOT, device=device, log=log)
        assert a.seed in ex.cfg.seeds and a.variant in W.ALL_VARIANTS
        gate = ex.dir / "baseline_gate.json"
        assert gate.exists() and json.loads(gate.read_text())["passed"], "version-2 gate not passed yet"
        log(f"run directory: {ex.dir}")
        W.fit(ex.cfg, a.variant, a.seed, ex.dir / "final" / f"seed{a.seed}" / a.variant, ex.detectors,
              ex.validation, device, log)
    log(f"done {a.exp} seed{a.seed}/{a.variant}")


if __name__ == "__main__":
    main()
