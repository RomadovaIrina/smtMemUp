#!/usr/bin/env bash
# Waits until the shared GPU has enough free memory for several consecutive checks,
# then runs ONLY the baseline phase (resumable). Exits after training or after MAX_WAIT.
set -u
cd /mem/work
NEED_MIB=${NEED_MIB:-12000}
STREAK_NEEDED=${STREAK_NEEDED:-3}
INTERVAL=${INTERVAL:-30}
MAX_WAIT=${MAX_WAIT:-43200}
LOG=runs/baseline.log
streak=0; waited=0
echo "$(date '+%F %T') waiting for >= ${NEED_MIB} MiB free GPU memory" | tee -a "$LOG"
while [ "$streak" -lt "$STREAK_NEEDED" ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
  if [ "${free:-0}" -ge "$NEED_MIB" ]; then streak=$((streak+1)); else streak=0; fi
  if [ "$streak" -lt "$STREAK_NEEDED" ]; then
    sleep "$INTERVAL"; waited=$((waited+INTERVAL))
    if [ "$waited" -ge "$MAX_WAIT" ]; then echo "$(date '+%F %T') gave up waiting" | tee -a "$LOG"; exit 3; fi
  fi
done
echo "$(date '+%F %T') free=${free} MiB, starting baseline" | tee -a "$LOG"
nice -n 10 /mem/envs/smt-memup/bin/python smt_memup_stack.py --phase baseline --threads 4 \
  --gpu-memory-fraction 0.35 >> "$LOG" 2>&1
status=$?
echo "$(date '+%F %T') baseline finished with exit code ${status}" | tee -a "$LOG"
exit $status
