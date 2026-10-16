#!/usr/bin/env python3
"""Run zoology_repro/configs.py with the authors' trainer (upstream/zoology at the pinned commit, NOT modified).

What this wrapper adds (nothing that changes training):
  * per-epoch metrics written to runs/zoology_repro/<cell>/<run_id>/metrics.jsonl (the authors log to W&B only);
  * wall time, epochs, early-stop epoch, peak CUDA memory;
  * the final model is also evaluated on an INDEPENDENT 3000-sequence set (seed 9123, same generator settings),
    because Zoology selects the LR and reports accuracy on the same test set (no validation split).

  ZOO_CELL=n64_d64  PYTHONPATH=upstream/zoology python zoology_repro/run_repro.py      # smoke
  ZOO_CELL=n256_d64 PYTHONPATH=upstream/zoology python zoology_repro/run_repro.py      # main (4 LRs, sequential)
  ZOO_LRS=0.000464 ...                                                                 # a subset of the LR grid
Pre-stated success criterion (outputs_mqar/zoology_reproduction.md): best over the 4 LRs of the final
valid/accuracy >= 0.99 -> reproduced; < 0.97 for every LR -> not reproduced; otherwise inconclusive.
"""
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ZOO = ROOT / "upstream" / "zoology"
PINNED = "1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb"
sys.path.insert(0, str(ZOO))

import torch  # noqa: E402
import zoology.train as ZT  # noqa: E402
from zoology.logger import WandbLogger  # noqa: E402

assert subprocess.run(["git", "-C", str(ZOO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() == PINNED


class FileLogger(WandbLogger):
    """Same interface as the authors' logger; writes locally instead of W&B."""
    path: Path = None

    def __init__(self, config):
        self.no_logger = False
        self.file = open(self.path / "metrics.jsonl", "a")

    def log_config(self, config):
        (self.path / "config.json").write_text(json.dumps(config.model_dump(), indent=2, default=str))

    def log_model(self, model, config):
        self.log({"num_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)})

    def log(self, metrics):
        self.file.write(json.dumps({"time": time.time(), **metrics}) + "\n"); self.file.flush()

    def finish(self):
        self.file.close()


@torch.no_grad()
def independent_accuracy(model, config, seed=9123):
    seg = config.data.test_configs[0].model_copy(update=dict(num_examples=3000)).build(seed=seed)
    model.eval()
    correct = total = 0
    for i in range(0, len(seg.inputs), 100):
        x, y = seg.inputs[i:i + 100].cuda(), seg.labels[i:i + 100].cuda()
        pred = model(x).argmax(-1)
        m = y != -100
        correct += int((pred[m] == y[m]).sum()); total += int(m.sum())
    return correct / total


def main():
    spec = importlib.util.spec_from_file_location("zoo_cfg", ROOT / "zoology_repro" / "configs.py")
    cfgmod = importlib.util.module_from_spec(spec); spec.loader.exec_module(cfgmod)
    out_root = ROOT / "runs" / "zoology_repro" / cfgmod.CELL
    captured = {}
    real_fit = ZT.Trainer.fit

    def fit_and_keep(self):
        captured["model"] = self.model
        return real_fit(self)
    ZT.Trainer.fit = fit_and_keep
    for config in cfgmod.configs:
        d = out_root / config.run_id
        if (d / "summary.json").exists():
            print("done already:", config.run_id); continue
        d.mkdir(parents=True, exist_ok=True)
        FileLogger.path = d
        ZT.WandbLogger = FileLogger
        torch.cuda.reset_peak_memory_stats()
        tic = time.perf_counter()
        ZT.train(config)
        wall = time.perf_counter() - tic
        rows = [json.loads(l) for l in (d / "metrics.jsonl").read_text().splitlines()]
        epochs = [r for r in rows if "valid/accuracy" in r]
        summary = dict(run_id=config.run_id, lr=config.learning_rate, epochs_run=len(epochs),
                       final_valid_accuracy=epochs[-1]["valid/accuracy"], max_valid_accuracy=max(r["valid/accuracy"] for r in epochs),
                       early_stopped=epochs[-1]["valid/accuracy"] > config.early_stopping_threshold,
                       independent_test_accuracy=independent_accuracy(captured["model"], config),
                       wall_seconds=wall, peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20, zoology=PINNED)
        (d / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
