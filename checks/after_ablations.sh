#!/usr/bin/env bash
# After the ablation queue (and the resumed runs) finish: MQAR bottleneck tests at T = 64 (pilot01), tables, push.
#   setsid nohup checks/after_ablations.sh > runs/after_ablations.log 2>&1 &
cd "$(dirname "$0")/.."
P=/workspace/envs/smt-memup/bin/python
while pgrep -f 'gpu_jobs.py runs/jobs_0419.json' >/dev/null || pgrep -f 'runs/resume_' >/dev/null; do sleep 60; done
echo "$(date '+%F %T') ablations finished; starting T64 bottleneck tests" >> runs/gpu_jobs.log
for m in bottleneck full; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $P checks/mqar_bottleneck_test.py --mode $m --T 64 --pairs 8 --steps 4000 \
    > runs/jobs/bneck_${m}_T64.log 2>&1 &
  sleep 30
done
wait
echo "$(date '+%F %T') done: T64 bottleneck tests" >> runs/gpu_jobs.log
CUDA_VISIBLE_DEVICES= $P make_ablation_outputs.py > runs/make_ablation_outputs.log 2>&1
shopt -s nullglob
git add -f outputs_ablation outputs_mqar runs/gpu_jobs.log runs/*.sh runs/*.log runs/jobs/*.log checks/after_ablations.sh \
  runs/stackv4_*/ablation/*/*.csv runs/stackv4_*/ablation/*/cost.json runs/stackv4_*/identity.json proposals status.md
if git diff --cached --name-only | grep -q '\.pt$'; then git reset -q; echo "refusing: checkpoint staged"; exit 1; fi
git -c user.name=RomadovaIrina -c user.email=gordovmedia@gmail.com commit -q \
  -m 'Stack SMT ablations finished; MQAR bottleneck test at T = 64 (automatic push at the end of the queue)' \
  -m 'Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>'
git push -q origin HEAD:main && echo "$(date '+%F %T') pushed $(git rev-parse --short HEAD) - ALL DONE" >> runs/gpu_jobs.log
