#!/usr/bin/env bash
# Usage: checks/wait_and_run.sh "tune final" "baseline,uniform,..."  [log]
# Before EACH phase waits until the shared GPU has >= NEED_MIB free for STREAK checks in a row,
# then runs that phase (all phases are resumable). Stops at the first failing phase.
# Never runs the test report.
set -u
cd "$(dirname "$0")/.."                       # repository root, wherever it is cloned
PY=${PY:-/mem/envs/smt-memup/bin/python}; [ -x "$PY" ] || PY=envs/smt-memup/bin/python
export PY
PHASES=$1; VARIANTS=$2; LOG=${3:-runs/tune_final.log}
NEED_MIB=${NEED_MIB:-12000}; STREAK=${STREAK:-3}; INTERVAL=${INTERVAL:-30}
for phase in $PHASES; do
  [ "$phase" = report ] && { echo "report is never started automatically" | tee -a "$LOG"; exit 2; }
  attempt=1
  while :; do
    echo "$(date '+%F %T') [$phase] attempt $attempt: waiting for >= ${NEED_MIB} MiB free GPU memory" | tee -a "$LOG"
    s=0
    while [ "$s" -lt "$STREAK" ]; do
      free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
      if [ "${free:-0}" -ge "$NEED_MIB" ]; then s=$((s+1)); else s=0; fi
      [ "$s" -lt "$STREAK" ] && sleep "$INTERVAL"
    done
    echo "$(date '+%F %T') [$phase] start, free=${free} MiB" | tee -a "$LOG"
    nice -n 10 "$PY" smt_memup_stack.py --phase "$phase" --variants "$VARIANTS" \
      --threads 4 --gpu-memory-fraction "${GPU_FRACTION:-0.35}" >> "$LOG" 2>&1
    status=$?
    echo "$(date '+%F %T') [$phase] finished, exit code ${status}" | tee -a "$LOG"
    [ "$status" -eq 0 ] && break
    # resume from the last checkpoint only for CUDA out-of-memory; any other error stops the chain
    tail -n 40 "$LOG" | grep -q "out of memory" || exit "$status"
    [ "$attempt" -ge 5 ] && exit "$status"
    attempt=$((attempt+1)); sleep 300
  done
done
exit 0
