#!/usr/bin/env bash
# mem16 needs more GPU memory than the other ablations: run it alone after resume_final.sh (retry up to 3 times).
cd "$(dirname "$0")/.."
P=/workspace/envs/smt-memup/bin/python
while pgrep -f 'gpu_jobs.py runs/jobs_0419.json' >/dev/null || pgrep -f 'runs/resume_final' >/dev/null; do sleep 60; done
for i in 1 2 3; do
  echo "$(date "+%F %T") started: abl_mem16 (rerun, attempt $i)" >> runs/gpu_jobs.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P checks/smt_memup_ablation.py --name mem16 >> runs/jobs/abl_mem16.log 2>&1
  rc=$?; echo "$(date "+%F %T") exit $rc: abl_mem16 (rerun, attempt $i)" >> runs/gpu_jobs.log
  [ $rc -eq 0 ] && break; sleep 60
done
