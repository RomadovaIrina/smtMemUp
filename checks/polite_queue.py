#!/usr/bin/env python3
"""Run a queue of our GPU commands one after another, yielding the shared GPU to foreign processes.

Usage: setsid nohup envs/smt-memup/bin/python checks/polite_queue.py --log runs/mqar_queue.log \
           "smt_memup_mqar.py --phase benchmark --config pilot01" "smt_memup_mqar.py --phase audit ..." &
Each item is run as `nice -n 10 <PY> <item> --threads 4 --gpu-memory-fraction <F>` from the repository root.
  * a command starts only after QUIET_S seconds without foreign GPU processes (checks/gpu_polite.py) and with
    >= START_FREE_MIB MiB free;
  * next to a foreign process a command may run while >= START_FREE_FOREIGN_MIB is free at start; it is KILLED
    (not paused: nothing of ours keeps GPU memory) as soon as free memory < KILL_FREE_FOREIGN_MIB, and restarted later; all our phases resume from their last checkpoint (every eval_every steps);
  * a .pt file left half-written by the kill is restored from its .pt.bak copy (made after every finished write)
    under runs/mqar03_*;
  * a command that fails by itself stops the queue. The test report is never started from here.
"""
import argparse
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # the supervisor itself never touches the GPU
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gpu_polite import KILL_FREE_FOREIGN_MIB, START_FREE_FOREIGN_MIB, gpu_state  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PY = os.environ.get("PY", "/mem/envs/smt-memup/bin/python")
if not Path(PY).exists():
    PY = str(ROOT / "envs/smt-memup/bin/python")
QUIET_S = int(os.environ.get("QUIET_S", 20))
START_FREE_MIB = int(os.environ.get("START_FREE_MIB", 8000))
POLL = int(os.environ.get("POLL", 2))
ALLOW_NEXT_TO_FOREIGN = os.environ.get("ALLOW_NEXT_TO_FOREIGN", "1") == "1"   # 0: start only when no neighbour is on the GPU
_seen = {}


CKPT_GLOBS = ["runs/mqar03_*/**/*.pt"]


def checkpoints():
    return [p for g in CKPT_GLOBS for p in ROOT.glob(g)]


def backup():
    for p in checkpoints():
        m = p.stat().st_mtime
        if _seen.get(p) == m or time.time() - m < 5:
            continue
        try:
            torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        shutil.copy2(p, p.with_suffix(".pt.bak"))
        _seen[p] = m


def repair(log):
    for p in checkpoints():
        try:
            torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            bak = p.with_suffix(".pt.bak")
            if bak.exists():
                shutil.copy2(bak, p); log(f"restored {p.relative_to(ROOT)} from backup")
            else:
                p.unlink(); log(f"removed unreadable {p.relative_to(ROOT)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("items", nargs="+")
    ap.add_argument("--log", default="runs/polite_queue.log")
    ap.add_argument("--fraction", default="0.25")
    ap.add_argument("--after-pid", type=int, default=None,
                    help="start nothing while this process (e.g. the Stack final launcher) is alive: it has priority")
    ap.add_argument("--ckpt-glob", action="append", default=None,
                    help="checkpoints to back up / repair (default runs/mqar03_*/**/*.pt)")
    a = ap.parse_args()
    if a.ckpt_glob:
        CKPT_GLOBS[:] = a.ckpt_glob
    logpath = ROOT / a.log

    def log(msg):
        line = f"{time.strftime('%F %T')} {msg}"
        try:
            with open(logpath, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    for item in a.items:
        assert "report" not in item, "the test report is never started automatically"
    log(f"queue: {a.items}")
    last_foreign = time.time() - QUIET_S
    for item in a.items:
        cmd = ["nice", "-n", "10", PY, *shlex.split(item), "--threads", "4", "--gpu-memory-fraction", a.fraction]
        proc = None
        while True:
            free, foreign, ours = gpu_state()
            others_ours = set(ours) - ({proc.pid} if proc is not None else set())
            if foreign:
                last_foreign = time.time()
            if proc is not None:
                rc = proc.poll()
                if rc is not None:
                    if rc == 0:
                        log(f"done: {item}")
                        break
                    log(f"FAILED (exit {rc}): {item}")
                    sys.exit(rc)
                if foreign and free < KILL_FREE_FOREIGN_MIB:
                    os.kill(proc.pid, signal.SIGKILL); proc.wait(); proc = None
                    log(f"killed (neighbour needs memory, free {free} MiB); will restart from the last checkpoint: {item}")
            elif a.after_pid and Path(f"/proc/{a.after_pid}").exists():
                pass                                   # a higher-priority launcher is still running
            elif ((not foreign and time.time() - last_foreign >= QUIET_S and free >= START_FREE_MIB)
                  or (foreign and ALLOW_NEXT_TO_FOREIGN and not others_ours
                      and free >= max(START_FREE_MIB, START_FREE_FOREIGN_MIB))):
                repair(log)
                env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
                env["OMP_NUM_THREADS"] = "4"
                out = open(logpath, "a")
                out.write(f"{time.strftime('%F %T')} start: {' '.join(cmd)}\n"); out.flush()
                proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
            backup()
            time.sleep(POLL)
    log("queue finished")


if __name__ == "__main__":
    main()
