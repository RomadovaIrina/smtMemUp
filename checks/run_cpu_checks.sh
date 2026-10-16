#!/usr/bin/env bash
# All CPU checks in parallel (GPU hidden). Logs: runs/cpu_checks/<name>.log, summary at the end.
#   checks/run_cpu_checks.sh
set -u
cd "$(dirname "$0")/.."
PY=${PY:-/workspace/envs/smt-memup/bin/python}; [ -x "$PY" ] || PY=envs/smt-memup/bin/python
export CUDA_VISIBLE_DEVICES=""
mkdir -p runs/cpu_checks
run() { local name=$1; shift; "$PY" "$@" > "runs/cpu_checks/$name.log" 2>&1; echo "exit=$?" >> "runs/cpu_checks/$name.log"; }
run resume_identity_v1 checks/check_resume_identity.py &
run v2_test_ready checks/check_v2_test_ready.py --sha &
run test_v4 checks/test_smt_memup_v4.py &
run test_mqar_baselines checks/test_mqar_baselines.py &
run test_window checks/test_smt_memup_window.py &
run test_stack checks/test_smt_memup_stack.py &
run test_mqar checks/test_smt_memup_mqar.py &
run test_v3 checks/test_smt_memup_v3.py &
wait
for f in runs/cpu_checks/*.log; do echo "=== $f"; tail -n 30 "$f"; done
