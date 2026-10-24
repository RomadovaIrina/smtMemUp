"""Before the test report: which of the 18 Stack final fits are complete, and does every checkpoint belong to the
saved run (identity, initial weights, detectors, validation split)? CPU only. Exit code 0 only if all 18 are complete.
Run: CUDA_VISIBLE_DEVICES="" python checks/check_final_fits.py"""
import sys, json, hashlib, torch
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
import smt_memup_stack as S
S.sources(ROOT)
cfg = S.Config(); dev = torch.device("cuda")
D = ROOT/"runs/stack02_4c69a2819ab8"
ident_run = S.run_identity(cfg, dev)
alphas = json.loads((D/"locked_alphas.json").read_text())["alphas"]
val = S.sample_stack(cfg, cfg.validation_examples, S.derive_seed(501, 0))
dets = {}
for k in ("window","context"):
    sd = torch.load(D/f"detector_{k}.pt", map_location="cpu", weights_only=True)
    assert sd["identity"]["run"] == ident_run, k
    m = (S.WindowDetector if k=="window" else S.ContextDetector)(cfg); m.load_state_dict(sd["best_model"]); dets[k]=m
    print("detector", k, "step", sd["step"], "best", round(sd["best"],4))
init, complete = {}, []
for seed in cfg.seeds:
    t, r = S.make_models(cfg, seed, torch.device("cpu")); init[seed] = S.state_hash(t, r)
for seed in cfg.seeds:
    for v in S.ALL_VARIANTS:
        out = D/"final"/f"seed{seed}"/v
        dh = {k: S.state_hash(m) for k, m in dets.items() if (k=="window" and v=="surprise_k") or (k=="context" and v=="surprise_ctx_k")}
        ident = dict(run=ident_run, variant=v, alpha=float(alphas[v]), seed=int(seed), detectors=dh, validation=S.split_hash(val))
        st = []
        if (out/"completed.pt").exists():
            s = torch.load(out/"completed.pt", map_location="cpu", weights_only=True)
            st.append(f"completed ok={s['identity']==ident and s['initial_hash']==init[seed]} smt_best={s['smt_best_step']} dmt_best={s['dmt_best_step']}")
        for stage in ("smt","dmt"):
            p = out/f"{stage}_last.pt"
            if p.exists():
                s = torch.load(p, map_location="cpu", weights_only=True)
                ok = s["identity"]==ident and s["initial_hash"]==init[seed]
                b = torch.load(out/f"{stage}_best.pt", map_location="cpu", weights_only=True) if (out/f"{stage}_best.pt").exists() else None
                st.append(f"{stage}_last step={s['step']} ok={ok} sched_last_epoch={s['scheduler']['last_epoch']} hist={len(s['history'])} best_step={b['step'] if b else None}")
        print(f"seed{seed}/{v}: " + ("; ".join(st) if st else "not started"))
        complete.append(bool(st) and st[0].startswith("completed ok=True"))
print(f"complete: {sum(complete)}/18")
sys.exit(0 if all(complete) and len(complete) == 18 else 1)
