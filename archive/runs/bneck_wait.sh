#!/usr/bin/env bash
# Starts the bottleneck tests one by one when enough GPU memory is free (the ablation queue keeps priority).
cd /workspace/smtMemUp
P=/workspace/envs/smt-memup/bin/python
run() { local need=$1 name=$2; shift 2
  until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$need" ]; do sleep 30; done
  echo "$(date '+%F %T') started: $name" >> runs/gpu_jobs.log
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True setsid nohup $P checks/mqar_bottleneck_test.py "$@" > runs/jobs/$name.log 2>&1 < /dev/null &
  sleep 90; }
run 3000 bneck_bottleneck_T16 --mode bottleneck
run 3000 bneck_full_T16 --mode full
run 8000 bneck_bottleneck_T64 --mode bottleneck --T 64 --pairs 8
run 10000 bneck_full_T64 --mode full --T 64 --pairs 8
