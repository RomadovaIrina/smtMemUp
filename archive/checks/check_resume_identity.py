"""Before resuming on another machine: does this environment map to the saved Stack run directory?
Run: python checks/check_resume_identity.py   (CPU only, does not touch the GPU)

The run directory name is a hash of the run identity (config, code hash, upstream commits, torch,
x-transformers, python version, device type). Any difference -> a NEW directory, and the saved locked
alphas, detectors and partial final fits are not picked up. Fix the environment instead of the code."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

import smt_memup_stack as S  # noqa: E402

SAVED = ROOT / "runs" / "stack02_4c69a2819ab8"
S.sources(ROOT)
current = S.run_identity(S.Config(), torch.device("cuda"))
key = hashlib.sha256(json.dumps(current, sort_keys=True).encode()).hexdigest()[:12]
saved = json.loads((SAVED / "identity.json").read_text())
diff = {k: (saved.get(k), current.get(k)) for k in set(saved) | set(current) if saved.get(k) != current.get(k)}
print("expected directory:", SAVED.name, "| this environment ->", f"stack02_{key}")
for k, (a, b) in diff.items():
    print(f"  MISMATCH {k}: saved={a!r} current={b!r}" if k != "config" else "  MISMATCH config")
need = ["locked_alphas.json", "baseline_gate.json", "detector_window.pt", "detector_context.pt", "preregistration.md"]
missing = [n for n in need if not (SAVED / n).exists()]
print("missing state files:", missing or "none")
print("torch.cuda.is_available():", torch.cuda.is_available())
if diff or missing or f"stack02_{key}" != SAVED.name:
    sys.exit("NOT OK: do not resume until the environment matches (see README, 'Продолжение на другом сервере').")
print("OK: this environment resumes the saved Stack run.")
