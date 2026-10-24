"""Pinned author code and seeding. Everything that comes from outside this repository is loaded here."""
from __future__ import annotations

import importlib.util
import random
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PINNED = {"smt": "cf0d0f04be606ab9d97b1a449a6846450d656ad0",          # SMT: Teacher, RNN, losses, Stack generator
          "memup": "d27657137f6e427e30fcd4ae885b20b071b2917d",        # MemUP: idea only (top-K surprising targets)
          "zoology": "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"}      # Zoology: MQAR generator, attention baseline
IGNORE = -1                     # label / padding value: "no answer here" (Stack push, MQAR non-query) or "no token"


def _git_head(path: Path) -> str:
    return subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True,
                          text=True).stdout.strip()


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_CACHE: dict = {}


def author(name: str):
    """'model' -> upstream/smt/src/model.py; 'stack' -> upstream/smt/src/my_datasets/synthetic.py;
    'mqar' -> zoology.data.multiquery_ar.multiquery_ar. Refuses to run if a checkout is not at the pinned commit."""
    if name not in _CACHE:
        repo = "zoology" if name == "mqar" else "smt"
        head = _git_head(ROOT / "upstream" / repo)
        if head != PINNED[repo]:
            raise RuntimeError(f"upstream/{repo} is at {head}, expected {PINNED[repo]} (scripts/setup_upstream.sh)")
        if name == "model":       # same module name as the archived code, so both share one loaded module
            _CACHE[name] = sys.modules.get("smt_author_model") or _load("smt_author_model", ROOT / "upstream/smt/src/model.py")
        elif name == "stack":
            _CACHE[name] = (sys.modules.get("smt_author_synthetic")
                            or _load("smt_author_synthetic", ROOT / "upstream/smt/src/my_datasets/synthetic.py"))
        else:
            sys.path.insert(0, str(ROOT / "upstream" / "zoology"))
            from zoology.data.multiquery_ar import multiquery_ar
            _CACHE[name] = multiquery_ar
    return _CACHE[name]


def derive_seed(*parts: int) -> int:
    """One integer per (stream, run seed, step): a batch never depends on what was sampled before it."""
    return int(np.random.SeedSequence([int(p) for p in parts]).generate_state(1)[0])


@contextmanager
def seeded(seed: int):
    """Seed numpy / python / torch (the author generators and calc_uniformity_loss use the global RNGs);
    the previous global state is restored afterwards."""
    states = np.random.get_state(), random.getstate(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        np.random.seed(int(seed) % 2**32); random.seed(int(seed)); torch.manual_seed(int(seed))
        yield
    finally:
        np.random.set_state(states[0]); random.setstate(states[1]); torch.set_rng_state(states[2])
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)
