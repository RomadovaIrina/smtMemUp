#!/usr/bin/env bash
# Rebuild the interim figures/tables of the diagnostic queue and push them with the small run files (histories,
# cost, identities, job logs) to GitHub. Checkpoints (*.pt) are never added.
#   checks/publish_progress.sh            # once
#   checks/publish_progress.sh --loop     # every PUBLISH_EVERY seconds (default 1800) while the scheduler runs, then once more
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/envs/smt-memup/bin/python}; [ -x "$PY" ] || PY=envs/smt-memup/bin/python

publish() {
  CUDA_VISIBLE_DEVICES="" "$PY" make_v4_outputs.py || return 1
  shopt -s nullglob
  local files=(outputs_v4 outputs_mqar make_v4_outputs.py checks/publish_progress.sh
               runs/gpu_jobs.log runs/jobs_v4.json runs/run_v4_diag.log runs/jobs/v4d_*.log runs/jobs/v4x_*.log runs/jobs_v4_extra.json checks/smt_memup_v4_extra.py runs/jobs/mqarA_*.log
               runs/jobs/zooB_*.log runs/jobs/cpu_*.log runs/cpu_checks/*.log runs/benchmark/*.json
               runs/benchmark/parallel_*/*.log
               runs/stackv4_*/identity.json runs/stackv4_*/diag/*/*.csv runs/stackv4_*/diag/*/cost.json
               runs/stackv4_*/diag_*.csv runs/stackv4_*/diag_decision.json
               runs/mqar_baselines_*/*/*.csv runs/mqar_baselines_*/*/cost.json
               runs/zoology_repro/*/*/*.json runs/zoology_repro/*/*/*.jsonl)
  git add -f -- "${files[@]}"
  if git diff --cached --name-only | grep -q '\.pt$'; then echo "refusing: a checkpoint is staged"; git reset -q; return 1; fi
  git diff --cached --quiet && { echo "nothing new"; return 0; }
  git -c user.name=RomadovaIrina -c user.email=gordovmedia@gmail.com commit -q -F - <<EOF
Progress $(date '+%d.%m %H:%M'): diagnostic queue (Stack v4 D, MQAR A, Zoology B) - interim validation figures, tables, logs

Interim, validation only; see outputs_v4/README.md. Checkpoints are not included.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
  git push -q origin HEAD:main HEAD:v4-short-window-bptt && echo "pushed $(git rev-parse --short HEAD)"
}

if [ "${1:-}" = "--loop" ]; then
  while pgrep -f "checks/gpu_jobs.py" > /dev/null; do
    publish; sleep "${PUBLISH_EVERY:-1800}"
  done
  publish
else
  publish
fi
