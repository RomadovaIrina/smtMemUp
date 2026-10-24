#!/usr/bin/env python3
"""Full-speed job scheduler for a GPU that is ours (user, 14.10.2026: the neighbours stopped their GPU work).

Jobs run in list order (= priority) as soon as their dependencies are done, up to MAX_JOBS at once, and only while
>= START_FREE_MIB GPU memory is free (STAGGER seconds between starts so the previous process has allocated).
A failed job is logged; jobs that depend on it never start. New jobs are not started while a foreign process holds
more than FOREIGN_MIB (nothing of ours is killed).
  python checks/gpu_jobs.py jobs.json      # [{"name": ..., "cmd": "...", "after": [...]}, ...]
Each job's output goes to runs/jobs/<name>.log; scheduler log: runs/gpu_jobs.log.
"""
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_polite import gpu_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MAX_JOBS = int(os.environ.get("MAX_JOBS", 4))
START_FREE_MIB = int(os.environ.get("START_FREE_MIB", 10500))
STAGGER = int(os.environ.get("STAGGER", 40))
FOREIGN_MIB = int(os.environ.get("FOREIGN_MIB", 4000))
LOG = ROOT / "runs" / "gpu_jobs.log"


def log(msg):
    try:
        with open(LOG, "a") as f:
            f.write(f"{time.strftime('%F %T')} {msg}\n")
    except OSError:
        pass


def main():
    jobs = json.loads(Path(sys.argv[1]).read_text())
    state = {j["name"]: "todo" for j in jobs}
    procs = {}
    (ROOT / "runs" / "jobs").mkdir(exist_ok=True)
    log(f"scheduler: {len(jobs)} jobs, max {MAX_JOBS} at once, start at >= {START_FREE_MIB} MiB free")
    last_start = 0.0
    while any(s in ("todo", "running") for s in state.values()):
        for name, p in list(procs.items()):
            rc = p.poll()
            if rc is not None:
                state[name] = "done" if rc == 0 else "failed"
                log(f"{state[name]}: {name} (exit {rc})")
                del procs[name]
        for j in jobs:                                   # a job whose dependency failed can never run
            if state[j["name"]] == "todo" and any(state.get(d) == "failed" for d in j.get("after", [])):
                state[j["name"]] = "failed"
                log(f"skipped (dependency failed): {j['name']}")
        free, foreign, _ = gpu_state()
        ready = [j for j in jobs if state[j["name"]] == "todo" and all(state.get(d) == "done" for d in j.get("after", []))]
        ready = [j for j in ready if free >= j.get("min_free", START_FREE_MIB)]     # per-job memory need (CPU jobs: 0)
        if (ready and len(procs) < MAX_JOBS and time.time() - last_start >= STAGGER
                and sum(foreign.values()) <= FOREIGN_MIB):
            j = ready[0]
            out = open(ROOT / "runs" / "jobs" / f"{j['name']}.log", "a")
            procs[j["name"]] = subprocess.Popen(["nice", "-n", "10", *shlex.split(j["cmd"])], cwd=ROOT,
                                                env={k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"},
                                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            state[j["name"]] = "running"
            last_start = time.time()
            log(f"started: {j['name']} (pid {procs[j['name']].pid}, free {free} MiB)")
        time.sleep(5)
    log("all jobs: " + ", ".join(f"{k}={v}" for k, v in state.items()))


if __name__ == "__main__":
    main()
