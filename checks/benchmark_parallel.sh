#!/usr/bin/env bash
# How do the latency-bound steps (DMT, BPTT rollouts of 64 sequential updates) scale when several processes share the
# GPU? Runs N copies of benchmark_train_step.py restricted to the given components at the same time.
#   checks/benchmark_parallel.sh 4 bptt_full,dmt,smt_ctx_far
set -eu
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/envs/smt-memup/bin/python}
N=${1:-4}; ONLY=${2:-bptt_full,dmt}
out=runs/benchmark/parallel_${N}_$(date +%H%M%S); mkdir -p "$out"
for i in $(seq 1 "$N"); do
  "$PY" checks/benchmark_train_step.py --steps 20 --warmup 5 --only "$ONLY" --out "$out" > "$out/p$i.log" 2>&1 &
done
wait
grep -h -E "^(smt_|dmt|bptt)" "$out"/p*.log | awk '{print $1, $3, $4}' | sort
