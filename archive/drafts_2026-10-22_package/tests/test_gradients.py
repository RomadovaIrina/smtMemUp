#!/usr/bin/env python3
"""Which parameters receive a gradient from which loss (nonzero / exactly zero), on a small Stack batch. CPU.
Prints a table (also used in docs/LOSSES_RU.md) and asserts the expected pattern.
    python tests/test_gradients.py"""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from smtcore.config import Config  # noqa: E402
from smtcore.data import sample_stack  # noqa: E402
from smtcore.losses import bptt_loss, dmt_loss, smt_loss  # noqa: E402
from smtcore.models import make_models, rnn_rollout  # noqa: E402
from smtcore.upstream import seeded  # noqa: E402

CPU = torch.device("cpu")
GROUPS = {"teacher.embed": ("t", "embed."), "teacher.encoder": ("t", "encoder."),
          "teacher.registers": ("t", "encoder_mem_register."), "teacher.decoder+head": ("t", ("decoder.", "to_logits.")),
          "rnn.embed": ("r", "embed."), "rnn.updater_f": ("r", "rnn."), "rnn.readout": ("r", ("decoder.", "to_logits."))}


def group_norms(teacher, rnn):
    out = {}
    for g, (who, pref) in GROUPS.items():
        model = teacher if who == "t" else rnn
        gs = [p.grad for n, p in model.named_parameters() if n.startswith(pref) and p.grad is not None]
        out[g] = float(torch.sqrt(sum((x.double() ** 2).sum() for x in gs))) if gs else 0.0
    return out


def run(name, cfg, fn):
    teacher, rnn = make_models(cfg, 3, CPU)
    batch = sample_stack(cfg, 4, 77)
    with seeded(1):
        loss = fn(teacher, rnn, batch, cfg)
    loss.backward()
    return name, group_norms(teacher, rnn)


def main():
    base = Config(seq_len=16, max_stack_size=4, d_model=32, n_heads=2)
    only = lambda **kw: replace(base, **{**dict(coef_dec=0.0, coef_dyn=0.0, coef_unif=0.0), **kw})
    smt = lambda t, r, b, c: smt_loss(t, r, b, c, CPU)[0]
    rows = [run("L_dec^Tf", only(coef_dec=1.0), smt),
            run("L_dyn (joint)", only(coef_dyn=1.0), smt),
            run("L_dyn target_detached (old 'detached')", only(coef_dyn=1.0, dyn_detach="target"), smt),
            run("L_dyn fully detached (authors' detach_rnn)", only(coef_dyn=1.0, dyn_detach="both"), smt),
            run("L_unif", only(coef_unif=1.0), smt)]

    def dmt(t, r, b, c):
        t.requires_grad_(False)
        return dmt_loss(t, r, b, c, CPU)[0]

    def dmt_readout_only(t, r, b, c):
        t.requires_grad_(False)
        return dmt_loss(t, r, b, replace(c, dmt_coef_dyn=0.0), CPU)[0]

    rows += [run("DMT total", base, dmt), run("DMT readout CE only", base, dmt_readout_only),
             run("BPTT full", base, lambda t, r, b, c: bptt_loss(r, b, c, CPU, False)[0]),
             run("BPTT-1", base, lambda t, r, b, c: bptt_loss(r, b, c, CPU, True)[0])]
    names = list(GROUPS)
    print("| loss | " + " | ".join(names) + " |\n|" + "---|" * (len(names) + 1))
    for name, g in rows:
        print(f"| {name} | " + " | ".join(f"{g[k]:.2e}" if g[k] else "0" for k in names) + " |")
    G = dict(rows)
    nz = lambda r, k: G[r][k] > 0
    assert all(nz("L_dec^Tf", k) for k in ("teacher.embed", "teacher.encoder", "teacher.registers", "teacher.decoder+head"))
    assert not any(nz("L_dec^Tf", k) for k in ("rnn.embed", "rnn.updater_f", "rnn.readout"))
    for r in ("L_dyn (joint)", "L_dyn target_detached (old 'detached')"):
        assert nz(r, "teacher.encoder") and nz(r, "rnn.updater_f") and not nz(r, "teacher.decoder+head")
    r = "L_dyn fully detached (authors' detach_rnn)"
    assert not nz(r, "teacher.encoder") and not nz(r, "teacher.embed") and nz(r, "rnn.updater_f")
    assert not any(nz("L_unif", k) for k in ("teacher.decoder+head", "rnn.updater_f")) and nz("L_unif", "teacher.encoder")
    assert not any(nz("DMT total", k) for k in ("teacher.encoder", "teacher.decoder+head"))
    assert not nz("DMT readout CE only", "rnn.updater_f") and nz("DMT readout CE only", "rnn.readout")
    assert all(nz(b, "rnn.updater_f") and nz(b, "rnn.readout") for b in ("BPTT full", "BPTT-1"))
    # BPTT-1 vs full: the loss at the LAST position reaches the first state only with full BPTT
    cfg = base
    teacher, rnn = make_models(cfg, 3, CPU)
    x = sample_stack(cfg, 4, 77)["x"]
    for detach, expect_zero in ((False, False), (True, True)):
        init = torch.zeros(4, cfg.mem_size, requires_grad=True)          # m_{-1}
        states = rnn_rollout(rnn, x, init=init, detach=detach)
        F.cross_entropy(rnn.rnn_readout(states[:, -1]), torch.ones(4, dtype=torch.long)).backward()
        g0 = float(init.grad.abs().sum()) if init.grad is not None else 0.0
        assert (g0 == 0) == expect_zero, (detach, g0)
        print(f"loss at t = T-1 -> gradient on m_(-1): {'BPTT-1' if detach else 'full BPTT'} {g0:.3e}")
        rnn.zero_grad()
    print("\nall gradient-pattern assertions passed")


if __name__ == "__main__":
    main()
