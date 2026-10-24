#!/usr/bin/env python3
"""Minimal GPU queue for an exclusively rented GPU. Jobs (configs/gpu_jobs.json) start in list order (= priority) while
fewer than MAX_JOBS run and at least the job's min_free MiB are free (nvidia-smi), STAGGER seconds apart.
Output of each job: results/logs/<name>.log; queue log: results/logs/queue.log. Resumable: finished runs are skipped by
smtcore.train.run (summary.json exists), interrupted ones resume from <stage>_last.pt.
    MAX_JOBS=3 python scripts/gpu_queue.py configs/gpu_jobs.json"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT / "results" / "logs"
MAX_JOBS, STAGGER = int(os.environ.get("MAX_JOBS", 3)), int(os.environ.get("STAGGER", 60))


def free_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.split()[0])


def log(msg):
    with open(LOGS / "queue.log", "a") as f:
        f.write(f"{time.strftime('%F %T')} {msg}\n")


if __name__ == "__main__":
    LOGS.mkdir(parents=True, exist_ok=True)
    jobs = json.loads(Path(sys.argv[1]).read_text())
    todo, procs, last = list(jobs), {}, 0.0
    log(f"queue: {len(jobs)} jobs, max {MAX_JOBS}")
    while todo or procs:
        for name, p in list(procs.items()):
            if p.poll() is not None:
                log(f"{'done' if p.returncode == 0 else 'FAILED'}: {name} (exit {p.returncode})")
                del procs[name]
        if todo and len(procs) < MAX_JOBS and time.time() - last >= STAGGER and free_mib() >= todo[0]["min_free"]:
            j = todo.pop(0)
            procs[j["name"]] = subprocess.Popen(shlex.split(j["cmd"]), cwd=ROOT, start_new_session=True,
                                                stdout=open(LOGS / f"{j['name']}.log", "a"), stderr=subprocess.STDOUT)
            last = time.time()
            log(f"started: {j['name']} (free {free_mib()} MiB)")
        time.sleep(10)
    log("queue finished")
