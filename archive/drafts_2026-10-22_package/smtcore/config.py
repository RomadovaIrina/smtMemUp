"""One Config for both tasks. Defaults = Stack, short window Tf = 8 (the archived version-4 / ablation reference).
Where a number comes from is listed in docs/MODELS_RU.md ("why these sizes"): most sizes are our research choice."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    task: str = "stack"              # stack | mqar
    seq_len: int = 64                # T: tokens per sequence (= encoder window, so m_t always sees x_0..x_t)
    # --- Stack (authors' StackDataset) ---
    n_elements: int = 4              # push IDs 1..4, pop ID 0 -> vocabulary 5
    max_stack_size: int = 8
    # --- MQAR (Zoology multiquery_ar, random_non_queries=False) ---
    vocab: int = 128                 # IDs: 0 filler, keys 1..63, values 64..127
    num_kv_pairs: int = 2            # N pairs = 2N tokens at positions 0..2N-1, then N queries
    power_a: float = 0.01            # Zoology default: query gaps ~ power law
    long_delay: int = 8              # "long" answers: written >= long_delay positions earlier (Stack 8; MQAR T/4)
    # --- model sizes (authors' classes; sizes are ours) ---
    d_model: int = 64
    mem_tokens: int = 8              # K: memory m_t = K tokens x d_model = 512 numbers
    encoder_depth: int = 3
    decoder_depth: int = 2
    rnn_depth: int = 2               # depth of the updater f
    n_heads: int = 4
    # --- SMT loss ---
    future_window: int = 8           # Tf: decoder outputs j < Tf enter L_dec (Stack v1 / tf64 / MQAR: Tf = T)
    far: str = "none"                # none | ctx_far: + far_K far targets (j >= Tf) chosen by the frozen detector
    far_K: int = 2
    alpha_far: float = 1.0
    coef_dec: float = 1.0
    coef_dyn: float = 0.1            # 0 = ablation no_dyn
    coef_unif: float = 0.001         # 0 = ablation no_unif
    dyn_detach: str = "none"         # none (joint; every archived run) | target (archived ablation "detached",
                                     # renamed target_detached) | both (authors' detach_rnn=True: no L_dyn grad to encoder)
    # --- optimisation ---
    batch_size: int = 32
    smt_steps: int = 11000
    smt_lr: float = 3e-4
    dmt_steps: int = 5000
    dmt_lr: float = 1e-4
    dmt_coef_dyn: float = 10.0
    bptt_steps: int = 20000
    bptt_lr: float = 3e-4
    lm_steps: int = 3000             # MQAR causal Transformer
    lm_lr: float = 1e-3
    warmup_steps: int = 200          # linear warm-up, then constant LR
    weight_decay: float = 0.01
    clip_grad: float = 1.0
    # --- evaluation / selection ---
    eval_every: int = 500
    eval_batch_size: int = 128
    validation_examples: int = 512   # checkpoint selection (stream 501 Stack / 511 MQAR)
    eval_examples: int = 1024        # independent evaluation (stream 901): reported numbers, never used for choices
    stop_accuracy: float = 0.0       # > 0: stop a stage when validation long accuracy reaches it (rule fixed in config)

    @property
    def vocab_size(self) -> int:
        return self.n_elements + 1 if self.task == "stack" else self.vocab

    @property
    def mem_size(self) -> int:       # Z = K * d
        return self.d_model * self.mem_tokens


def load_config(path_or_dict) -> Config:
    """configs/*.json hold only the fields that differ from the defaults above."""
    data = path_or_dict if isinstance(path_or_dict, dict) else json.loads(Path(path_or_dict).read_text())
    return replace(Config(), **data)


def diff_from_default(cfg: Config) -> dict:
    default = asdict(Config())
    return {k: v for k, v in asdict(cfg).items() if default[k] != v}
