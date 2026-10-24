#!/usr/bin/env bash
# Version 3 (converged regime): all final fits at full speed on a GPU that is ours.
#   checks/run_v3.sh                  # 5 variants x 3 seeds = 15 fits (~6.5 h with 3 parallel processes)
#   checks/run_v3.sh --with-oracle    # + oracle diagnostic, 18 fits (~7.5 h)
# Refuses to start while another process holds GPU memory (> 4 GB), unless FORCE=1.
# Logs: runs/gpu_jobs.log (scheduler), runs/jobs/v3_*.log (each fit). Progress: python smt_memup_v3.py --phase speed
# The TEST report is never started here (needs the user's permission; see proposals/preregistration_v3.md).
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/mem/envs/smt-memup/bin/python}; [ -x "$PY" ] || PY=envs/smt-memup/bin/python
used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "${used}" -gt 4000 ] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU is in use by other processes (${used} MiB):"; nvidia-smi --query-compute-apps=pid,used_memory --format=csv
  echo "Start anyway with FORCE=1 checks/run_v3.sh (the scheduler still waits for free memory)."; exit 1
fi
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "${free_gb}" -ge 3 ] || { echo "only ${free_gb} GB free on disk; version 3 needs ~2 GB (checkpoints + test events)"; exit 1; }
CUDA_VISIBLE_DEVICES="" "$PY" checks/test_smt_memup_v3.py > runs/v3_checks.log 2>&1 || { echo "checks failed: runs/v3_checks.log"; exit 1; }
"$PY" checks/make_jobs_v3.py "$@" --python "$PY" > runs/jobs_v3.json
MAX_JOBS=${MAX_JOBS:-3} setsid nohup "$PY" checks/gpu_jobs.py runs/jobs_v3.json > /dev/null 2>&1 < /dev/null &
sleep 3
echo "version 3 started: $(grep -c '"name"' runs/jobs_v3.json) fits; scheduler log runs/gpu_jobs.log"
tail -n 3 runs/gpu_jobs.log
