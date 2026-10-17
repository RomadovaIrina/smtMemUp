#!/usr/bin/env bash
# Runs the stack-memory probes (CPU, low priority) once both part-D SMT runs have finished DMT.
cd /workspace/smtMemUp
until ls runs/stackv4_*/diag/base_window/completed.pt runs/stackv4_*/diag/ctx_far/completed.pt > /dev/null 2>&1; do sleep 120; done
CUDA_VISIBLE_DEVICES= nice -n 19 /workspace/envs/smt-memup/bin/python analysis_stack_probes.py --threads 2 > runs/probes.log 2>&1
