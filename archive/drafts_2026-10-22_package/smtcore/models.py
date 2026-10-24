"""3. Models and forward helpers. Shapes: N windows or sequences, T positions, K memory tokens, d = d_model,
Z = K*d, V vocabulary.

Teacher (authors' class, upstream/smt/src/model.py:105-167):
    encode(x_ctx (N,T)) -> m (N,Z):  embed(x) (N,T,d) ++ K learned registers (N,K,d) -> bidirectional Transformer
        encoder over N x (T+K) tokens (padding -1 masked) -> last K outputs -> RMS-norm -> flatten.
    decode(m (N,Z), x_fut (N,T)) -> logits (N,T,V): causal Transformer decoder over [m as K tokens, embed(x_fut)];
        outputs at positions K-1 .. K+T-2; output j sees m and x_fut[:, :j].
    teacher_readout(m) = decode output j = 0.
RNN (authors' class, model.py:184-226):
    rnn_update(m_{t-1} (N,Z), x_t (N,1)) -> m_t (N,Z):  Transformer ENCODER (bidirectional, rnn_depth layers) over
        [m_{t-1} as K tokens, embed(x_t)] (N, K+1, d) -> first K outputs -> RMS-norm.   (This is the updater f.
        It is NOT a GRU; the GRUs below are separate selection detectors.)
    rnn_readout(m (N,Z)) -> logits (N,V): its own causal decoder + head, initialised as a COPY of the teacher's.
"""
from __future__ import annotations

import torch
from einops import repeat

from .config import Config
from .upstream import IGNORE, ROOT, author, seeded


def make_models(cfg: Config, seed: int, device):
    """Same seed -> same initial weights (teacher created first, then the RNN), as in every archived run."""
    m = author("model")
    common = dict(vocab_size=cfg.vocab_size, mem_size=cfg.mem_size, mem_tokens=cfg.mem_tokens, d_model=cfg.d_model,
                  decoder_depth=cfg.decoder_depth, n_heads=cfg.n_heads)
    with seeded(seed):
        teacher = m.Teacher(m.TeacherConfig(**common, encoder_depth=cfg.encoder_depth)).to(device)
        rnn = m.RNN(m.RNNConfig(**common, rnn_depth=cfg.rnn_depth)).to(device)
    copy_readout(teacher, rnn)
    return teacher, rnn


def copy_readout(teacher, rnn) -> None:
    """RNN readout := copy of teacher decoder + head (separate tensors). Done at init, at every SMT evaluation and
    before DMT, as in the authors' train_smt.py. During SMT the RNN readout gets no gradient."""
    rnn.decoder.load_state_dict(teacher.decoder.state_dict())
    rnn.to_logits.load_state_dict(teacher.to_logits.state_dict())


def encode_all(teacher, x_ctx: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """(B*T, T) context windows -> (B*T, Z) memories m_t, in chunks (memory only; the result is identical)."""
    return torch.cat([teacher.encode(p) for p in x_ctx.split(chunk)])


def rnn_rollout(rnn, x: torch.Tensor, init=None, detach: bool = False) -> torch.Tensor:
    """m_t = f(m_{t-1}, x_t), m_{-1} = 0 (or init). x (B, T) -> states (B, T, Z).
    detach=True: the incoming state of every update is cut from the graph (DMT, BPTT-1)."""
    state = torch.zeros(x.shape[0], rnn.cfg.mem_size, device=x.device) if init is None else init     # (B, Z)
    out = []
    for t in range(x.shape[1]):
        state = rnn.rnn_update(state.detach() if detach else state, x[:, t:t + 1])                   # (B, Z)
        out.append(state)
    return torch.stack(out, 1)


def encode_full(teacher, x_ctx):
    """Diagnostic 'full access': the authors' encoder, but ALL its T + K outputs are returned instead of the last K.
    x_ctx (N, T) -> h (N, T+K, d), key mask (N, T+K). Changes TWO things vs SMT at once: how many vectors the decoder
    can read (T + K instead of K) and how it reads them (attention to per-position outputs)."""
    B = x_ctx.shape[0]
    mask = torch.cat([x_ctx.ne(IGNORE), torch.ones(B, teacher.cfg.mem_tokens, dtype=torch.bool, device=x_ctx.device)], 1)
    regs = repeat(teacher.encoder_mem_register(torch.arange(teacher.cfg.mem_tokens, device=x_ctx.device)), "K D -> B K D", B=B)
    return teacher.encoder(torch.cat([teacher.embed(x_ctx.clamp(min=0)), regs], 1), mask=mask), mask


def decode_full(teacher, h, mask, x_fut):
    """Causal decoder over [h (N, n, d), embed(x_fut) (N, T, d)] with padding masked -> logits (N, T, V);
    output j is position n-1+j."""
    n = h.shape[1]
    inp = torch.cat([h, teacher.embed(x_fut.clamp(min=0))], 1)
    m = torch.cat([mask, torch.ones(x_fut.shape, dtype=torch.bool, device=x_fut.device)], 1)
    return teacher.to_logits(teacher.decoder(inp, mask=m))[:, n - 1:-1]


class ContextDetector(torch.nn.Module):
    """ctx_far only. A SEPARATE small GRU (embed V+2 -> 16, GRU 16 -> 64, linear 64 -> V; 16k parameters):
    q(y_{t+j} | x_{t+1..t+j}) WITHOUT memory. High CE = the target cannot be inferred from the decoder's own inputs.
    Trained once in the archived version 1 and frozen (archive/runs/stack02_4c69a2819ab8/detector_context.pt)."""

    def __init__(self, vocab_size: int, embed: int = 16, hidden: int = 64):
        super().__init__()
        self.bos = vocab_size + 1
        self.embed = torch.nn.Embedding(vocab_size + 2, embed, padding_idx=0)
        self.gru = torch.nn.GRU(embed, hidden, batch_first=True)
        self.head = torch.nn.Linear(hidden, vocab_size)

    def forward(self, x_fut):                          # (N, T) -> (N, T, V); output j has seen x_fut[:, :j] only
        inp = torch.cat([torch.full_like(x_fut[:, :1], self.bos - 1), x_fut[:, :-1]], 1)
        h, _ = self.gru(self.embed(inp + 1))           # +1: -1 padding -> index 0 (padding_idx)
        return self.head(h)


def load_context_detector(cfg: Config, device) -> ContextDetector:
    saved = torch.load(ROOT / "archive/runs/stack02_4c69a2819ab8/detector_context.pt", map_location="cpu", weights_only=True)
    det = ContextDetector(cfg.vocab_size)
    det.load_state_dict(saved["best_model"])
    return det.to(device).eval().requires_grad_(False)


class CausalTransformer(torch.nn.Module):
    """MQAR reference LM: x-transformers TransformerWrapper + Decoder (d 64, 2 layers, 4 heads, FF x4, learned absolute
    positions), as archive/mqar_baselines.py. x (B, T) -> logits (B, T, V); logits[:, t] predicts y_t."""

    def __init__(self, cfg: Config, d_model: int = 64, depth: int = 2, heads: int = 4):
        super().__init__()
        from x_transformers import Decoder, TransformerWrapper
        self.net = TransformerWrapper(num_tokens=cfg.vocab_size, max_seq_len=cfg.seq_len,
                                      attn_layers=Decoder(dim=d_model, depth=depth, heads=heads))

    def forward(self, x):
        return self.net(x)


def make_lm(cfg: Config, seed: int, device) -> CausalTransformer:
    with seeded(seed):
        return CausalTransformer(cfg).to(device)
