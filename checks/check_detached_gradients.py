#!/usr/bin/env python3
"""Does the `detached` ablation (checks/smt_memup_ablation.py) cut the gradient of L_dyn into the encoder?

CPU check on one small Stack batch. It calls the REAL version-2/4 loss smt_memup_window.smt_window_loss with
coef_dec = coef_unif = 0 and coef_dyn = 1, so the total loss equals L_dyn, and measures the gradient of L_dyn on the
teacher encoder (embedding, Transformer encoder, memory registers) and on the RNN updater in three modes:

  joint        : the code as used by the reference run `full` (no patch);
  ablation     : the monkey-patch of the `detached` run: calc_mse_loss(pred, target.detach());
  author_detach: the authors' SMTConfig.detach_rnn=True semantics (upstream src/model.py, forward_smt):
                 BOTH the updater input m_{t-1} and the MSE target m_t are detached.

In smt_window_loss L_dyn = MSE(f(m_{t-1}, x_t), m_t) where BOTH m_{t-1} (input of the updater) and m_t (target) are
encoder outputs. The ablation patch detaches only m_t, so the gradient still reaches the encoder through m_{t-1}.
Run:  python checks/check_detached_gradients.py
"""
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402

import smt_memup_stack as S  # noqa: E402
import smt_memup_v4 as V  # noqa: E402
import smt_memup_window as W  # noqa: E402

ENCODER_PREFIXES = ("embed.", "encoder.", "encoder_mem_register.")


def grad_norms(mode: str, cfg, batch, seed: int = 107):
    m = S.sources(ROOT).model
    real_mse = m.calc_mse_loss
    teacher, rnn = S.make_models(cfg, seed, torch.device("cpu"))
    if mode == "ablation":                               # exactly the patch of checks/smt_memup_ablation.py
        m.calc_mse_loss = lambda pred, target: real_mse(pred, target.detach())
    elif mode == "author_detach":                        # detach_rnn=True: input AND target detached
        real_encode = teacher.encode
        state = {"calls": 0}

        def encode_detached(x_ctx):                      # every encoder output used by L_dyn is detached; with
            state["calls"] += 1                          # coef_dec = coef_unif = 0 nothing else uses it
            return real_encode(x_ctx).detach()
        teacher.encode = encode_detached
    try:
        loss, parts = W.smt_window_loss(teacher, rnn, batch, cfg, "base_window", {}, torch.Generator().manual_seed(1),
                                        torch.device("cpu"))
        loss.backward()
    finally:
        m.calc_mse_loss = real_mse
    enc = [p.grad for n, p in teacher.named_parameters() if n.startswith(ENCODER_PREFIXES) and p.grad is not None]
    upd = [p.grad for n, p in rnn.named_parameters() if n.startswith(("rnn.", "embed.")) and p.grad is not None]
    norm = lambda gs: float(torch.sqrt(sum((g.double() ** 2).sum() for g in gs))) if gs else 0.0
    return dict(mode=mode, loss=float(loss.detach()), dyn=float(parts["dynamics"].detach()), encoder_grad_norm=norm(enc),
                encoder_tensors_with_grad=sum(int(g.abs().sum() > 0) for g in enc), updater_grad_norm=norm(upd))


def main():
    torch.manual_seed(0)
    cfg = replace(V.V4Config(), coef_dec=0.0, coef_unif=0.0, coef_dyn=1.0, seq_len=16, max_stack_size=4)
    batch = S.sample_stack(cfg, 4, S.derive_seed(301, 107, 0))
    rows = [grad_norms(m, cfg, batch) for m in ("joint", "ablation", "author_detach")]
    for r in rows:
        print(f"{r['mode']:>14}: L_dyn={r['dyn']:.5f}  |grad encoder|={r['encoder_grad_norm']:.3e} "
              f"(tensors with grad: {r['encoder_tensors_with_grad']})  |grad updater|={r['updater_grad_norm']:.3e}")
    j, a, d = rows
    assert abs(j["dyn"] - a["dyn"]) < 1e-6 and abs(j["dyn"] - d["dyn"]) < 1e-6, "forward values must be identical"
    assert a["encoder_grad_norm"] > 0, "expected: the ablation patch still sends L_dyn gradient into the encoder"
    assert d["encoder_grad_norm"] == 0, "author detach_rnn: no L_dyn gradient into the encoder"
    assert abs(a["updater_grad_norm"] - d["updater_grad_norm"]) / d["updater_grad_norm"] < 1e-6, \
        "the updater gradient must be the same in both detach variants"
    print("RESULT: the `detached` ablation does NOT fully cut the L_dyn gradient into the encoder: the target m_t is "
          "detached, but the updater input m_{t-1} is not. Ratio |grad enc| ablation/joint = "
          f"{a['encoder_grad_norm'] / j['encoder_grad_norm']:.3f}")


if __name__ == "__main__":
    main()
