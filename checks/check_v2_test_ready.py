#!/usr/bin/env python3
"""Is everything in place to run the version-2 TEST report exactly by its original protocol?
CPU only, creates nothing, never generates the test split.  python checks/check_v2_test_ready.py [--sha]

The report (python smt_memup_window.py --phase report --confirm-test, then python make_v2_outputs.py --test) needs:
  * this environment mapping to the saved run directory runs/stackwin_26f6a039d62c (identity: config, stack + window
    code hashes, upstream commits, torch, x-transformers, python, device type 'cuda');
  * the 18 completed version-2 fits (6 variants x seeds 7/17/27) - final_fits() re-checks each identity;
  * the version-1 full-window baseline fits of the same seeds (reference 'full_window');
  * the version-1 detectors (in git).
Missing checkpoints must be COPIED from the server where they were trained; retraining would be a different run.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

import smt_memup_stack as S  # noqa: E402
import smt_memup_window as W  # noqa: E402

V2 = ROOT / "runs" / "stackwin_26f6a039d62c"
V1 = ROOT / "runs" / "stack02_4c69a2819ab8"
SEEDS = (7, 17, 27)


def required() -> list[Path]:
    files = [V2 / "final" / f"seed{s}" / v / "completed.pt" for s in SEEDS for v in W.ALL_VARIANTS]
    files += [V1 / "final" / f"seed{s}" / "baseline" / "completed.pt" for s in SEEDS]
    files += [V1 / "detector_window.pt", V1 / "detector_context.pt", V1 / "identity.json", V2 / "identity.json",
              V2 / "baseline_gate.json"]
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sha", action="store_true", help="print sha256 of present checkpoints (to compare with the source)")
    a = ap.parse_args()
    S.sources(ROOT)
    current = W.run_identity(W.WindowConfig(), torch.device("cuda"))
    key = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()[:12]
    saved = json.loads((V2 / "identity.json").read_text())
    diff = sorted(k for k in set(saved) | set(current) if saved.get(k) != current.get(k))
    print("saved run:", V2.name, "| this environment ->", f"stackwin_{key}")
    for k in diff:
        print(f"  MISMATCH {k}: saved={saved.get(k)!r} current={current.get(k)!r}" if k != "config" else "  MISMATCH config")
    missing, present = [], []
    for f in required():
        (present if f.exists() else missing).append(f)
    print(f"present {len(present)} / {len(present) + len(missing)} required files")
    for f in missing:
        print("  MISSING", f.relative_to(ROOT))
    v2_test_done = (V2 / "test_report.json").exists()
    print("version-2 test already run:", v2_test_done)
    bad = []
    for f in present:
        if f.name == "completed.pt":
            obj = torch.load(f, map_location="cpu", weights_only=True)
            run = obj["identity"]["run"] if "run" in obj["identity"] else None
            ok = run == (saved if V2 in f.parents else json.loads((V1 / "identity.json").read_text()))
            if not ok:
                bad.append(f)
            if a.sha:
                print(f"  {hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(ROOT)}  identity_ok={ok}")
    for f in bad:
        print("  IDENTITY MISMATCH", f.relative_to(ROOT))
    if diff or missing or bad:
        sys.exit("NOT READY: copy the missing files from the training server and/or fix the environment (do not retrain).")
    print("READY: the version-2 test report can be run (only with the user's permission).")


if __name__ == "__main__":
    main()
