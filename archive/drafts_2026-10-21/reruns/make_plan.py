#!/usr/bin/env python3
"""Writes reruns/plan.json (run specs for smt_minimal.py) and reruns/jobs.json (queue for checks/gpu_jobs.py).

Fixed BEFORE any rerun (21.10.2026). Seeds 201, 202, 203 are new (archived runs used 7/17/27, 67/77/87, 107).
Checkpoint selection: validation set (stream 501 Stack / 511 MQAR), lowest CE on long answers.
Reported numbers: the independent evaluation set (stream 901, 1024 sequences), never used for any choice.
Budgets: Stack = the archived version-4 rule (SMT 11k, DMT 5k, BPTT 20k, LR 3e-4); MQAR = the archived length sweep
(SMT/teacher 6k, Transformer 3k at LR 1e-3, RNN BPTT 6k at LR 1e-3), DMT 2k (archived sweep: 50, token stage only).
Equal optimizer steps are NOT equal compute: cost.json-like seconds are saved per run (summary.json).
Priority (the queue runs in this order): 1 MQAR short, 2 Stack main controls, 3 Stack ablations.
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEEDS = (201, 202, 203)
PY = "env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /workspace/envs/smt-memup/bin/python"

MQAR = {"T16N2": dict(task="mqar", seq_len=16, num_kv_pairs=2, long_delay=4, future_window=16),
        "T32N4": dict(task="mqar", seq_len=32, num_kv_pairs=4, long_delay=8, future_window=32)}
MQAR_RUNS = {   # name -> (family, config, GPU MiB needed)
    "smt": ("smt", dict(smt_steps=6000, dmt_steps=2000), 3000),
    "teacher_bottleneck": ("smt", dict(smt_steps=6000, dmt_steps=0, coef_dyn=0.0, coef_unif=0.0), 3000),
    "teacher_full": ("teacher_full", dict(smt_steps=6000, dmt_steps=0), 3000),
    "transformer": ("transformer", dict(lm_steps=3000, lm_lr=1e-3), 1000),
    "rnn_bptt_full": ("bptt_full", dict(bptt_steps=6000, bptt_lr=1e-3), 1000),
}
STACK_MAIN = {
    "smt_tf8": ("smt", dict(), 9000),                                  # short window, no far targets (base_window)
    "smt_tf8_ctx_far": ("smt", dict(far="ctx_far"), 9000),
    "rnn_bptt_full": ("bptt_full", dict(), 1000),
    "rnn_bptt_1": ("bptt_1", dict(), 1000),
}
STACK_ABL = {
    "smt_tf8_no_dyn": ("smt", dict(coef_dyn=0.0), 9000),
    "smt_tf8_dyn_detach_both": ("smt", dict(dyn_detach="both"), 9000),   # the authors' detached SMT (new)
}


def main():
    runs = []
    for tier, groups in ((1, [(f"mqar_{k}", MQAR_RUNS, v) for k, v in MQAR.items()]),
                         (2, [("stack", STACK_MAIN, {})]), (3, [("stack", STACK_ABL, {})])):
        for prefix, table, base in groups:
            for name, (family, cfg, mib) in table.items():
                for seed in SEEDS:
                    runs.append(dict(name=f"{prefix}_{name}_s{seed}", family=family, seed=seed, tier=tier,
                                     min_free_mib=mib, config={**base, **cfg}))
    (HERE / "plan.json").write_text(json.dumps(dict(seeds=SEEDS, runs=runs), indent=1))
    jobs = [dict(name=r["name"], min_free=r["min_free_mib"], after=[],
                 cmd=f"{PY} smt_minimal.py train --spec reruns/plan.json --index {i}") for i, r in enumerate(runs)]
    (HERE / "jobs.json").write_text(json.dumps(jobs, indent=1))
    print(f"{len(runs)} runs: " + ", ".join(f"tier {t}: {sum(r['tier'] == t for r in runs)}" for t in (1, 2, 3)))


if __name__ == "__main__":
    main()
