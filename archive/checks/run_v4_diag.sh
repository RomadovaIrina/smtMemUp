#!/usr/bin/env bash
# Minimal diagnostic queue (Stack v4 part D, MQAR check A, Zoology check B) on a GPU that is ours.
#   checks/run_v4_diag.sh                         # all three groups
#   QUEUE_GROUPS=stack checks/run_v4_diag.sh            # only Stack part D
# Runs the CPU checks first and refuses to start if any fails or if another process holds GPU memory (FORCE=1 to
# override). Logs: runs/gpu_jobs.log (scheduler), runs/jobs/<job>.log. No test split is created by this queue.
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/envs/smt-memup/bin/python}; [ -x "$PY" ] || PY=envs/smt-memup/bin/python
used=$(nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "${used}" -gt 2000 ] && [ "${FORCE:-0}" != 1 ]; then
  echo "GPU is in use by other processes (${used} MiB); start anyway with FORCE=1"; exit 1
fi
free_gb=$(df -BG --output=avail . | tail -1 | tr -dc 0-9)
[ "${free_gb}" -ge 5 ] || { echo "only ${free_gb} GB free on disk"; exit 1; }
mkdir -p runs/jobs
for t in checks/test_smt_memup_v4.py checks/test_mqar_baselines.py; do
  CUDA_VISIBLE_DEVICES="" "$PY" "$t" > "runs/jobs/cpu_$(basename "$t" .py).log" 2>&1 || { echo "checks failed: $t"; exit 1; }
done
"$PY" checks/make_jobs_v4.py --groups "${QUEUE_GROUPS:-stack,mqar,zoology}" --python "$PY" > runs/jobs_v4.json
MAX_JOBS=${MAX_JOBS:-7} START_FREE_MIB=${START_FREE_MIB:-3000} STAGGER=${STAGGER:-20} \
  setsid nohup "$PY" checks/gpu_jobs.py runs/jobs_v4.json > /dev/null 2>&1 < /dev/null &
sleep 3
echo "started: $(grep -c '"name"' runs/jobs_v4.json) jobs; scheduler log runs/gpu_jobs.log"
tail -n 3 runs/gpu_jobs.log
