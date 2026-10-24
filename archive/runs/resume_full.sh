#!/usr/bin/env bash
cd /workspace/smtMemUp
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 10000 ]; do sleep 30; done
echo "$(date '+%F %T') started: abl_full (resume after OOM)" >> runs/gpu_jobs.log
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /workspace/envs/smt-memup/bin/python checks/smt_memup_ablation.py --name full >> runs/jobs/abl_full.log 2>&1
echo "$(date '+%F %T') done: abl_full resume (exit $?)" >> runs/gpu_jobs.log
