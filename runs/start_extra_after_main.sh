#!/usr/bin/env bash
# Starts the extra diagnostic runs only AFTER the main queue scheduler (pid 8371) has finished.
cd /workspace/smtMemUp
while kill -0 8371 2>/dev/null; do sleep 60; done
echo "$(date '+%F %T') main queue finished; starting extra runs" >> runs/gpu_jobs.log
MAX_JOBS=4 START_FREE_MIB=10000 STAGGER=60 exec /workspace/envs/smt-memup/bin/python checks/gpu_jobs.py runs/jobs_v4_extra.json
