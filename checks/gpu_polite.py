"""Shared-GPU etiquette for our launchers (no experiment code here).

A GPU process is OURS if its command line runs one of our scripts; everything else (e.g. the /dub pipelines) is
FOREIGN and has priority. Both launchers (run_final_polite.py for Stack, polite_queue.py for MQAR) use the same
definition, so they never mistake each other for a neighbour.
"""
import os
import subprocess
from pathlib import Path

# Next to a neighbour (user, 14.10: "if anything fits, fit in"): keep / start ONE of our processes only while plenty of
# memory is free; the moment the neighbour grows and free memory drops below KILL_FREE_FOREIGN_MIB, our process is
# killed (SIGKILL frees its memory at once) and later resumes from its last checkpoint.
KILL_FREE_FOREIGN_MIB = int(os.environ.get("KILL_FREE_FOREIGN_MIB", 12000))
START_FREE_FOREIGN_MIB = int(os.environ.get("START_FREE_FOREIGN_MIB", 22000))

OUR_SCRIPTS = ("smt_memup_", "fit_one.py", "make_outputs.py", "make_test_outputs.py", "make_interim_figures.py",
               "h2_retention", "make_mqar_outputs.py", "mqar_baselines.py", "zoology_repro/run_repro.py",
               "benchmark_train_step.py")


def smi(query, apps=False):
    kind = "--query-compute-apps" if apps else "--query-gpu"
    out = subprocess.run(["nvidia-smi", f"{kind}={query}", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, timeout=60).stdout
    return [[c.strip() for c in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def cmdline(pid):
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
    except OSError:
        return ""


def is_ours(pid):
    c = cmdline(pid)
    return any(s in c for s in OUR_SCRIPTS)


def gpu_state():
    """(free MiB, {pid: MiB} of foreign processes, {pid: MiB} of ours)."""
    free = int(smi("memory.free")[0][0])
    apps = {int(r[0]): int(r[1]) if r[1].isdigit() else 0 for r in smi("pid,used_memory", apps=True)}
    ours = {p: m for p, m in apps.items() if is_ours(p)}
    return free, {p: m for p, m in apps.items() if p not in ours}, ours
