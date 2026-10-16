#!/usr/bin/env python3
"""Extra DIAGNOSTIC SMT runs of version 4 on the tuning seed (107), exactly as part D (same config, schedules, budgets,
data streams, initial weights, detectors, evaluation, checkpoint rule) - they only add variants that part D does not
train. Not the final comparison; no test split. smt_memup_v4.py is imported, not modified (its hash is the identity of
the running part-D fits), so the runs land in the SAME run directory runs/stackv4_<key>/diag/<variant>.

  python checks/smt_memup_v4_extra.py --variant random_far        # also surprise_far, uniform_far, oracle_far
  python checks/smt_memup_v4_extra.py --variant full_window       # Tf = T = 64 (the SMT paper's synthetic setting);
                                                                   # its own run directory (the config differs)
The file name contains "smt_memup_" so that checks/gpu_polite.py counts it as ours.
"""
import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smt_memup_stack as S  # noqa: E402
import smt_memup_v4 as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["random_far", "surprise_far", "uniform_far", "oracle_far", "full_window"])
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    device = S.setup_runtime(a.threads)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    if a.variant == "full_window":
        cfg = replace(V.V4Config(), future_window=V.V4Config().seq_len)     # base loss over all future targets
        variant = "base_window"                                            # no far term (the far pool is empty)
    else:
        cfg, variant = V.V4Config(), a.variant
    ex = V.Experiment(cfg, root=ROOT, device=device, log=log)
    log(f"run directory: {ex.dir}")
    V.fit_smt(cfg, variant, cfg.tuning_seed, ex.dir / "diag" / a.variant, ex.detectors, ex.validation, device, log)
    log(f"done {a.variant}")


if __name__ == "__main__":
    main()
