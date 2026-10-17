#!/usr/bin/env bash
# Zoology check B, rerun after the cache_dir fix: smoke cell first, the main cell only if the smoke succeeded.
cd /workspace/smtMemUp
PY=/workspace/envs/smt-memup/bin/python
echo "$(date '+%F %T') started: zooB_smoke_n64 (rerun after cache_dir fix)" >> runs/gpu_jobs.log
if env ZOO_CELL=n64_d64 PYTHONPATH=upstream/zoology nice -n 10 $PY zoology_repro/run_repro.py >> runs/jobs/zooB_smoke_n64.log 2>&1; then
  echo "$(date '+%F %T') done: zooB_smoke_n64 (exit 0)" >> runs/gpu_jobs.log
  echo "$(date '+%F %T') started: zooB_main_n256" >> runs/gpu_jobs.log
  env ZOO_CELL=n256_d64 PYTHONPATH=upstream/zoology nice -n 10 $PY zoology_repro/run_repro.py >> runs/jobs/zooB_main_n256.log 2>&1
  echo "$(date '+%F %T') done: zooB_main_n256 (exit $?)" >> runs/gpu_jobs.log
else
  echo "$(date '+%F %T') failed: zooB_smoke_n64 rerun" >> runs/gpu_jobs.log
fi
