#!/usr/bin/env bash
# Copy results of the 3090 queues into this repo (results_3090/; the server never pulls this folder over live files): histories, summaries, configs, logs, executed notebooks, queue status.
# Checkpoints (*.pt) stay on the server.   bash sync_results.sh [user@host]
set -e
HOST=${1:-irina@100.69.122.37}
cd "$(dirname "$0")"
ssh "$HOST" 'cd ~/smt_memup_project/results && tar czf /tmp/smt_results.tgz --exclude="*.pt" --exclude="executed/_*" --exclude="cache" notebook_runs $(ls -d zoology_cell 2>/dev/null) && cp ~/smt_queue.log /tmp/smt_queue.log'
scp -q "$HOST":/tmp/smt_results.tgz /tmp/smt_results.tgz
mkdir -p results_3090 && scp -q "$HOST":/tmp/smt_queue.log results_3090/queue_3090.log
tar xzf /tmp/smt_results.tgz -C results_3090
echo "synced: $(ls results_3090/notebook_runs | grep -c _s20) run folders, $(ls results_3090/notebook_runs/executed | wc -l) executed notebooks, $(ls results_3090/zoology_cell 2>/dev/null | wc -l) zoology-cell entries"
