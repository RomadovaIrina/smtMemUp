#!/usr/bin/env python3
"""MQAR (19.10.2026): is the memory bottleneck of the SMT teacher what prevents key->value binding?

The author Teacher (bidirectional encoder over x_ctx + K memory registers appended after the inputs; causal decoder over
[memory, x_fut]) is trained with the decoder loss only (L_dec over all anchors and all valid future targets; no L_dyn,
no L_unif, no RNN), in two variants that differ ONLY in what the decoder sees:
  --mode bottleneck : the K = 8 memory tokens (exactly the SMT teacher's decode)
  --mode full       : ALL encoder outputs (every context position + the registers; padding masked) - no bottleneck
Reported: accuracy of the j = 0 output at query positions (= the teacher readout of SMT) on validation, full vocabulary
and among the N written values (chance 1/N).
"""
import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from einops import repeat  # noqa: E402

import smt_memup_mqar as M  # noqa: E402
import smt_memup_stack as S  # noqa: E402


def encode_full(teacher, x_ctx):
    """Author Teacher.encode up to the encoder output, but returning every position (+ the key-padding mask)."""
    B, Tc = x_ctx.shape
    mask_x = x_ctx != -1
    mask = torch.cat([mask_x, torch.ones((B, teacher.cfg.mem_tokens), dtype=torch.bool, device=x_ctx.device)], 1)
    mem = repeat(teacher.encoder_mem_register(torch.arange(teacher.cfg.mem_tokens, device=x_ctx.device)), "K D -> B K D", B=B)
    h = teacher.encoder(torch.cat([teacher.embed(x_ctx.clamp(min=0)), mem], 1), mask=mask)
    return h, mask


def decode_full(teacher, h, mask, x_fut):
    """Causal decoder over [all encoder outputs, x_fut]; output at the last encoder position predicts y_t (j = 0)."""
    n = h.shape[1]
    inp = torch.cat([h, teacher.embed(x_fut.clamp(min=0))], 1)
    m = torch.cat([mask, torch.ones(x_fut.shape, dtype=torch.bool, device=x_fut.device)], 1)
    return teacher.to_logits(teacher.decoder(inp, mask=m))[:, n - 1:-1]


def logits_for(teacher, w, mode, device):
    x_ctx, x_fut = w["x_ctx"].to(device), w["x_fut"].to(device)
    if mode == "bottleneck":
        return teacher.decode(teacher.encode(x_ctx), x_fut)
    h, mask = encode_full(teacher, x_ctx)
    return decode_full(teacher, h, mask, x_fut)


@torch.no_grad()
def evaluate(teacher, batch, cfg, mode, device):
    teacher.eval()
    w = S.windows(batch)
    lg = torch.cat([logits_for(teacher, {k: v[i:i + 256] for k, v in w.items()}, mode, device)[:, 0].float().cpu()
                    for i in range(0, len(w["x_ctx"]), 256)])                    # j = 0 (teacher readout)
    y = w["y_fut"][:, 0]
    q = y.ne(S.IGNORE)
    N = cfg.num_kv_pairs
    written = batch["x"][:, 1:2 * N:2].repeat_interleave(batch["x"].shape[1], 0)
    cand = lg.gather(1, written)
    cand_y = (written == y[:, None]).float().argmax(1)
    return float((lg.argmax(-1) == y)[q].float().mean()), float((cand.argmax(1) == cand_y)[q].float().mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["bottleneck", "full"])
    ap.add_argument("--T", type=int, default=16)
    ap.add_argument("--pairs", type=int, default=2)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=3e-4)
    a = ap.parse_args()
    device = S.setup_runtime(4)
    S.sources(ROOT)
    cfg = M.candidate("pilot01", seq_len=a.T, num_kv_pairs=a.pairs, K=1, long_delay=a.T // 4, local_window=a.T // 4)
    val = M.sample_mqar(cfg, 128, S.derive_seed(511, 0))
    teacher, _ = S.make_models(cfg, 107, device)
    opt = torch.optim.AdamW(teacher.parameters(), lr=a.lr, betas=(0.9, 0.95), weight_decay=0.01)
    for step in range(a.steps):
        teacher.train()
        batch = M.sample_mqar(cfg, 32, S.derive_seed(301, 107, step))
        w = S.windows(batch)
        lg = logits_for(teacher, w, a.mode, device)
        y = w["y_fut"].to(device)
        ce = F.cross_entropy(lg.flatten(0, 1).float(), y.clamp(min=0).flatten(), reduction="none").reshape(y.shape)
        loss = S.pair_mean(ce, w["valid"].to(device))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(teacher.parameters(), 1.0); opt.step()
        if (step + 1) % 250 == 0:
            acc, cand = evaluate(teacher, val, cfg, a.mode, device)
            print(time.strftime("%H:%M:%S"), f"{a.mode} T={a.T} N={a.pairs} step {step + 1} train_ce={float(loss):.3f} "
                  f"readout acc={acc:.3f} among written={cand:.3f} (chance 1/{a.pairs})", flush=True)


if __name__ == "__main__":
    main()
