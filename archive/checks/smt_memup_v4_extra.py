#!/usr/bin/env python3
"""Extra DIAGNOSTIC SMT runs of version 4 on the tuning seed (107), as part D (same schedules, data streams, initial weights, detectors, evaluation, checkpoint rule) - they only add variants that part D does not
train, with a SHORTER budget (default SMT 12k, DMT 8k) so that they finish overnight; this budget is part of the
config, so the runs get their own run directory. Not the final comparison; no test split. smt_memup_v4.py is imported,
not modified (its hash is the identity of the running part-D fits).

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
import torch  # noqa: E402

import smt_memup_stack as S  # noqa: E402
import smt_memup_v4 as V  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["random_far", "surprise_far", "uniform_far", "oracle_far", "full_window"])
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smt-steps", type=int, default=12000)
    ap.add_argument("--dmt-steps", type=int, default=8000)
    a = ap.parse_args()
    device = S.setup_runtime(a.threads)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    # short diagnostic budget (decided 17.10 so that the runs finish overnight): with the constant LR after warm-up an
    # SMT curve up to 12k steps is directly comparable with the part-D curves at the same steps
    base = replace(V.V4Config(), smt_steps=a.smt_steps, dmt_steps=a.dmt_steps)
    if a.variant == "full_window":
        cfg = replace(base, future_window=base.seq_len)     # base loss over all future targets
        variant = "base_window"                            # no far term (the far pool is empty)
    else:
        cfg, variant = base, a.variant
    # memory hygiene only (no effect on numerics): free cached CUDA blocks whenever the readout is copied (every SMT
    # evaluation and the SMT -> DMT switch); on 17.10 a part-D process grew to 14 GB at that switch and another job OOMed
    real_copy = S.copy_readout

    def copy_and_free(teacher, rnn):
        real_copy(teacher, rnn)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    S.copy_readout = copy_and_free
    ex = V.Experiment(cfg, root=ROOT, device=device, log=log)
    log(f"run directory: {ex.dir}")
    V.fit_smt(cfg, variant, cfg.tuning_seed, ex.dir / "diag" / a.variant, ex.detectors, ex.validation, device, log)
    log(f"done {a.variant}")


if __name__ == "__main__":
    main()
