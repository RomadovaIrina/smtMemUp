#!/usr/bin/env bash
# Final fits in 3 parallel workers. Each worker owns a disjoint group of variants and trains it
# for all seeds, so no two processes touch the same run directory. The experiment code is not
# changed (same code hash, same locked alphas). Worker 1 starts only when the GPU is almost empty
# (FIRST_MIB free), the next ones when NEXT_MIB is free; starts are staggered so each worker sees
# the memory already taken by the previous one. Each worker resumes after CUDA OOM (wait_and_run.sh).
# The test report is never started here.
set -u
cd "$(dirname "$0")/.."                       # repository root, wherever it is cloned
PY=${PY:-/mem/envs/smt-memup/bin/python}; [ -x "$PY" ] || PY=envs/smt-memup/bin/python
export PY
FIRST_MIB=${FIRST_MIB:-28000}; NEXT_MIB=${NEXT_MIB:-11000}; STAGGER=${STAGGER:-150}
WORKER_SETS=("baseline,uniform" "random_k,surprise_k" "surprise_ctx_k,oracle_prefix_k")
LOG=runs/final_parallel.log
wait_free() {
  local s=0 f
  while [ "$s" -lt 3 ]; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    if [ "${f:-0}" -ge "$1" ]; then s=$((s+1)); else s=0; fi
    [ "$s" -lt 3 ] && sleep 30
  done
}
pids=()
for i in "${!WORKER_SETS[@]}"; do
  need=$NEXT_MIB; [ "$i" -eq 0 ] && need=$FIRST_MIB
  echo "$(date '+%F %T') worker $((i+1)) [${WORKER_SETS[$i]}] waiting for >= ${need} MiB free" | tee -a "$LOG"
  wait_free "$need"
  NEED_MIB=$NEXT_MIB GPU_FRACTION=0.30 checks/wait_and_run.sh "final" "${WORKER_SETS[$i]}" "runs/final_w$((i+1)).log" &
  pids+=($!)
  echo "$(date '+%F %T') worker $((i+1)) started, pid $!" | tee -a "$LOG"
  [ "$i" -lt $(( ${#WORKER_SETS[@]} - 1 )) ] && sleep "$STAGGER"
done
status=0
for p in "${pids[@]}"; do wait "$p" || status=1; done
echo "$(date '+%F %T') all workers finished, status ${status}" | tee -a "$LOG"
exit "$status"
