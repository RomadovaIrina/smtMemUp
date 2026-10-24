#!/usr/bin/env bash
# Resume the ablations that were OOM-killed (full, detached) only once the main ablation queue has finished,
# so that they do not compete for GPU memory; retry up to 3 times (each attempt resumes from the last checkpoint).
cd "$(dirname "$0")/.."
P=/workspace/envs/smt-memup/bin/python
while pgrep -f 'gpu_jobs.py runs/jobs_0419.json' >/dev/null; do sleep 60; done
attempt() {
  local name=$1
  for i in 1 2 3; do
    echo "$(date '+%F %T') started: abl_${name} (resume, attempt $i)" >> runs/gpu_jobs.log
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P checks/smt_memup_ablation.py --name "$name" >> "runs/jobs/abl_${name}.log" 2>&1
    rc=$?
    echo "$(date '+%F %T') exit $rc: abl_${name} (resume, attempt $i)" >> runs/gpu_jobs.log
    [ $rc -eq 0 ] && return 0
    sleep 60
  done
  return 1
}
attempt full & sleep 60; attempt detached & wait
