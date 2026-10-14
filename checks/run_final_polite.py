#!/usr/bin/env python3
"""Final Stack fits that yield the shared GPU to other users (launcher only; experiment code unchanged).

Same disjoint worker groups as run_final_parallel.sh, same command (`--phase final`, 4 threads, nice 10,
--gpu-memory-fraction 0.30). Policy, checked every POLL seconds:
  * a foreign process (a GPU process not running one of our scripts, checks/gpu_polite.py) is on the GPU -> keep at most ONE of our workers and
    pause it (SIGSTOP: no compute, ~8 GB stays allocated); extra workers are killed; if free memory falls
    below KILL_FREE_MIB the paused one is killed too. Killed fits resume from their last checkpoint
    (saved every 250 steps; the data stream and random draws are a function of (seed, step)).
  * no foreign process for QUIET_S seconds -> resume paused workers; start new ones (up to MAX_WORKERS)
    when >= START_FREE_MIB MiB are free, one at a time, STAGGER seconds apart.
The disk is shared too: below DISK_PAUSE_GB free the workers are paused (no checkpoint writes) and resume at
DISK_OK_GB. Failures with OOM / no space / right at start-up are retried. A killed process may leave a half-written .pt; every finished write is copied to <name>.pt.bak and a file that
does not load is restored from it before the group restarts. The test split is never created here.
Run: setsid nohup envs/smt-memup/bin/python checks/run_final_polite.py > /dev/null 2>&1 < /dev/null &
Log: runs/final_polite.log, worker output: runs/final_w{1,2,3}.log
"""
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # this supervisor never touches the GPU itself
import torch  # noqa: E402  (only torch.load on CPU for checkpoint integrity)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_polite import gpu_state  # noqa: E402  (foreign = not one of our scripts, see gpu_polite.py)

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("PY", "/mem/envs/smt-memup/bin/python")
if not Path(PY).exists():
    PY = str(ROOT / "envs/smt-memup/bin/python")
GROUPS = ["baseline,uniform", "random_k,surprise_k", "surprise_ctx_k,oracle_prefix_k"]
MAX_WORKERS = int(os.environ.get("MAX_WORKERS", 2))
START_FREE_MIB = int(os.environ.get("START_FREE_MIB", 10000))
KILL_FREE_MIB = int(os.environ.get("KILL_FREE_MIB", 1500))
QUIET_S = int(os.environ.get("QUIET_S", 120))
STAGGER = int(os.environ.get("STAGGER", 90))
POLL = int(os.environ.get("POLL", 15))
FRACTION = os.environ.get("GPU_FRACTION", "0.30")
DISK_PAUSE_GB = float(os.environ.get("DISK_PAUSE_GB", 1.5))   # disk shared with /dub: below this, pause (no checkpoint writes)
DISK_OK_GB = float(os.environ.get("DISK_OK_GB", 3.0))
FINAL = ROOT / "runs" / "stack02_4c69a2819ab8" / "final"
LOG = ROOT / "runs" / "final_polite.log"


def log(msg):
    line = f"{time.strftime('%F %T')} {msg}"
    try:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    except OSError:
        pass                                           # disk full: keep running, the state is in memory
    print(line, flush=True)


def disk_free_gb():
    return shutil.disk_usage(ROOT).free / 2**30


def group_dirs(group):
    return [d for v in group.split(",") for d in FINAL.glob(f"seed*/{v}") if d.is_dir()]


_seen = {}


def backup_checkpoints():
    """Copy every completed .pt write (stable for 5 s and loadable) to .pt.bak."""
    for p in FINAL.glob("seed*/*/*.pt"):
        m = p.stat().st_mtime
        if _seen.get(p) == m or time.time() - m < 5:
            continue
        try:
            torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue                                   # still being written, or broken: keep the old backup
        tmp = p.with_suffix(".pt.bak.tmp")
        try:                                           # atomic: a half-written copy never replaces a good backup
            shutil.copy2(p, tmp)
            os.replace(tmp, p.with_suffix(".pt.bak"))
            _seen[p] = m
        except OSError:
            tmp.unlink(missing_ok=True)


def repair(group):
    for d in group_dirs(group):
        for p in d.glob("*.pt"):
            try:
                torch.load(p, map_location="cpu", weights_only=True)
            except Exception as e:
                bak = p.with_suffix(".pt.bak")
                if bak.exists():
                    shutil.copy2(bak, p)
                    log(f"restored {p.relative_to(ROOT)} from backup ({type(e).__name__})")
                else:
                    p.unlink()
                    log(f"removed unreadable {p.relative_to(ROOT)} (no backup)")


class Worker:
    def __init__(self, i, group):
        self.i, self.group = i, group
        self.proc, self.state, self.attempts, self.killed = None, "todo", 0, False
        self.logfile = ROOT / "runs" / f"final_w{i}.log"

    def start(self):
        repair(self.group)
        env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
        env["OMP_NUM_THREADS"] = "4"
        cmd = ["nice", "-n", "10", PY, "smt_memup_stack.py", "--phase", "final", "--variants", self.group,
               "--threads", "4", "--gpu-memory-fraction", FRACTION]
        with open(self.logfile, "a") as f:
            f.write(f"{time.strftime('%F %T')} [polite] start {self.group}\n")
            self.proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        self.state, self.killed = "running", False
        self.started = time.time()
        self.attempts += 1
        log(f"worker {self.i} [{self.group}] started, pid {self.proc.pid}")

    def pause(self, why="foreign GPU process"):
        os.kill(self.proc.pid, signal.SIGSTOP)
        self.state = "paused"
        log(f"worker {self.i} paused ({why})")

    def resume(self):
        os.kill(self.proc.pid, signal.SIGCONT)
        self.state = "running"
        log(f"worker {self.i} resumed")

    def kill(self, why):
        self.killed = True
        os.kill(self.proc.pid, signal.SIGKILL)       # works on a stopped process too
        self.proc.wait()
        self.proc, self.state = None, "todo"
        log(f"worker {self.i} killed ({why}); will resume from its last checkpoint")

    def reap(self):
        if self.proc is None or self.proc.poll() is None:
            return
        rc = self.proc.returncode
        self.proc = None
        if rc == 0:
            self.state = "done"
            log(f"worker {self.i} [{self.group}] finished")
            return
        tail = self.logfile.read_text(errors="ignore")[-6000:]
        transient = ("out of memory" in tail or "No space left" in tail or "ENOSPC" in tail
                     or time.time() - self.started < 120)   # a failure right at start-up (e.g. import on a full disk)
        if transient and self.attempts < 8:
            self.state = "todo"
            log(f"worker {self.i} failed with exit {rc} (transient: OOM / disk / start-up); will retry")
        else:
            self.state = "failed"
            log(f"worker {self.i} FAILED with exit {rc}; see {self.logfile.relative_to(ROOT)}")


def main():
    workers = [Worker(i + 1, g) for i, g in enumerate(GROUPS)]
    log(f"polite final launcher: max {MAX_WORKERS} workers, start at >= {START_FREE_MIB} MiB free, "
        f"quiet {QUIET_S}s, fraction {FRACTION}")
    last_foreign, last_start = time.time() - QUIET_S, 0.0
    foreign_seen = None
    while any(w.state not in ("done", "failed") for w in workers):
        for w in workers:
            w.reap()
        free, foreign, _ = gpu_state()
        disk = disk_free_gb()
        if disk < DISK_PAUSE_GB:                      # shared disk almost full: stop writing until it recovers
            for w in workers:
                if w.state == "running":
                    w.pause(f"disk free {disk:.1f} GB < {DISK_PAUSE_GB}")
            time.sleep(POLL)
            continue
        if foreign:
            last_foreign = time.time()
            if foreign_seen != set(foreign):
                log(f"foreign GPU processes {foreign} MiB; free {free} MiB")
                foreign_seen = set(foreign)
            active = [w for w in workers if w.state in ("running", "paused")]
            for w in active[1:]:
                w.kill("yield GPU memory to a foreign process")
            for w in active[:1]:
                if w.state == "running":
                    w.pause()
                if free < KILL_FREE_MIB:
                    w.kill(f"free memory {free} MiB < {KILL_FREE_MIB}")
        else:
            if foreign_seen:
                log(f"no foreign GPU processes; free {free} MiB")
                foreign_seen = None
            if time.time() - last_foreign >= QUIET_S:
                for w in workers:
                    if w.state == "paused" and disk >= DISK_OK_GB:
                        w.resume()
                n_running = sum(w.state == "running" for w in workers)
                todo = [w for w in workers if w.state == "todo"]
                if (todo and n_running < MAX_WORKERS and free >= START_FREE_MIB and disk >= DISK_OK_GB
                        and time.time() - last_start >= STAGGER):
                    todo[0].start()
                    last_start = time.time()
        backup_checkpoints()
        time.sleep(POLL)
    log("all groups: " + ", ".join(f"{w.group}={w.state}" for w in workers))
    sys.exit(0 if all(w.state == "done" for w in workers) else 1)


if __name__ == "__main__":
    main()
