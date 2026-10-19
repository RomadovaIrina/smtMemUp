#!/usr/bin/env python3
"""Stack ablations of SMT where it works (advisor, 19.10.2026): switch off one component at a time and see what the
result depends on. Short-window SMT without the far term (base_window, Tf = 8) is the reference; budget = the one fixed
by the version-4 diagnostic rule (SMT 11k, DMT 5k), tuning seed 107, validation only. smt_memup_v4.py / _window.py /
_stack.py are imported, not modified.

  --name full       reference: author loss L_dec^{Tf} + 0.1 L_dyn + 0.001 L_unif, joint training, DMT
  --name no_dyn     coef_dyn = 0: the updater is not trained during SMT and the encoder is not pushed to be Markovian
  --name detached   L_dyn trains the updater but its gradient does NOT reach the encoder (memory target detached) -
                    the "detached SMT" ablation of the SMT paper (sec. 3.6)
  --name no_unif    coef_unif = 0 (no uniformity term)
  --name tf1/tf4/tf16/tf64   predictive window Tf (tf64 = Tf = T, the paper's synthetic setting)
  --name mem4/mem16 memory tokens 4 / 16 (state 4x64 / 16x64)
"RNN without DMT" is read from every run's SMT stage (RNN rolled out with the copied head), so it needs no run.
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

RUNS = {"full": {}, "no_dyn": dict(coef_dyn=0.0), "detached": {}, "no_unif": dict(coef_unif=0.0),
        "tf1": dict(future_window=1), "tf4": dict(future_window=4), "tf16": dict(future_window=16),
        "tf64": dict(future_window=64), "mem4": dict(mem_tokens=4), "mem16": dict(mem_tokens=16)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, choices=sorted(RUNS))
    ap.add_argument("--smt-steps", type=int, default=11000)
    ap.add_argument("--dmt-steps", type=int, default=5000)
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    device = S.setup_runtime(a.threads)
    log = lambda s: print(time.strftime("%H:%M:%S"), s, flush=True)
    cfg = replace(V.V4Config(), smt_steps=a.smt_steps, dmt_steps=a.dmt_steps, **RUNS[a.name])
    real_copy = S.copy_readout                          # memory hygiene only (no effect on numerics)

    def copy_and_free(teacher, rnn):
        real_copy(teacher, rnn)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    S.copy_readout = copy_and_free
    if a.name == "detached":
        m = S.sources(ROOT).model
        real_mse = m.calc_mse_loss                      # in the SMT window loss it is used only for L_dyn

        def mse_detached_target(pred, target):
            return real_mse(pred, target.detach())
        m.calc_mse_loss = mse_detached_target
    ex = V.Experiment(cfg, root=ROOT, device=device, log=log)
    log(f"run directory: {ex.dir}")
    V.fit_smt(cfg, "base_window", cfg.tuning_seed, ex.dir / "ablation" / a.name, ex.detectors, ex.validation, device, log)
    log(f"done {a.name}")


if __name__ == "__main__":
    main()
