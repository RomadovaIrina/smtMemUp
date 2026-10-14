#!/usr/bin/env bash
# Version-1 Stack test report, started only after the final launcher exits AND all 18 fits pass the check.
# User decision (14.10.2026): test first, before any further GPU work. Usage:
#   setsid nohup checks/run_test_after_final.sh <final-launcher-pid> > /dev/null 2>&1 < /dev/null &
# Log: runs/test_report.log. Waits for >= NEED_MIB free GPU memory (the GPU is shared) before each GPU step.
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/mem/envs/smt-memup/bin/python}
LOG=runs/test_report.log
NEED_MIB=${NEED_MIB:-8000}
log() { echo "$(date '+%F %T') $*" >> "$LOG"; }
wait_gpu() {
  while :; do
    f=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
    [ "${f:-0}" -ge "$NEED_MIB" ] && return
    sleep 10
  done
}
log "waiting for the final launcher (pid $1) to exit"
while [ -e "/proc/$1" ]; do sleep 20; done
log "final launcher exited; checking all 18 fits"
if ! CUDA_VISIBLE_DEVICES="" "$PY" checks/check_final_fits.py >> "$LOG" 2>&1; then
  log "NOT all 18 fits complete: the test is NOT started"; exit 1
fi
wait_gpu
log "starting the test report (single use of the test split)"
nice -n 10 "$PY" smt_memup_stack.py --phase report --confirm-test \
  --variants baseline,uniform,random_k,surprise_k,surprise_ctx_k,oracle_prefix_k \
  --threads 4 --gpu-memory-fraction 0.25 >> "$LOG" 2>&1 || { log "report FAILED"; exit 1; }
log "report done; building test tables, figures and the H2 analysis"
wait_gpu
nice -n 10 "$PY" make_test_outputs.py --h2 --gpu-memory-fraction 0.25 >> "$LOG" 2>&1 || { log "make_test_outputs FAILED"; exit 1; }
log "ALL DONE"
