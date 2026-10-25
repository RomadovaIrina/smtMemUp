#!/usr/bin/env bash
# Copy results of the 3090 queue into this repo: histories, summaries, configs, logs, executed notebooks, queue status.
# Checkpoints (*.pt) stay on the server.   bash sync_results.sh [user@host]
set -e
HOST=${1:-irina@100.69.122.37}
cd "$(dirname "$0")"
ssh "$HOST" 'cd ~/smt_memup_project/results && tar czf /tmp/smt_results.tgz --exclude="*.pt" --exclude="executed/_*" notebook_runs && cp ~/smt_queue.log /tmp/smt_queue.log'
scp -q "$HOST":/tmp/smt_results.tgz /tmp/smt_results.tgz
scp -q "$HOST":/tmp/smt_queue.log results/queue_3090.log
mkdir -p results && tar xzf /tmp/smt_results.tgz -C results
echo "synced: $(ls results/notebook_runs | grep -c _s20) run folders, $(ls results/notebook_runs/executed | wc -l) executed notebooks"
