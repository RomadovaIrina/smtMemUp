"""Check B: reproduction of ONE published Zoology MQAR attention baseline with the authors' own code.

Source: Zoology (arXiv 2312.04927, v1, 8 Dec 2023 - the only arXiv version), Figure 2 top row (attention); code at the
pinned commit 1ad20d193b6113cae1e8f3c655c300d7b4b3f4bb, file
zoology/experiments/paper_configs/iclr24_zoology_figure2/configs.py (the repository's figure-2 reproduction config).
This file copies that config for ONE cell and changes only what does not affect training:
  * sequence mixer list -> attention only; (input_seq_len, num_kv_pairs) -> (256, 16); d_model -> 64;
  * cache_dir and the W&B logger (LoggerConfig() = no W&B; metrics are written locally by run_repro.py).
Everything else is the authors' value: vocab 8192, power_a 0.01 (MQARConfig default; the config's train_power_a /
test_power_a keys are ignored by pydantic), random_non_queries False, 100k train / 3k test, batch 256, 2 layers,
1-head MHA with attention dropout 0.1, no MLP (Identity state mixer), learned absolute positions (max 256),
AdamW wd 0.1, cosine over 64 epochs without warm-up, early stop at valid accuracy > 0.99, LR in logspace(-4, -2, 4).
A smoke configuration (N = 64, D = 4, batch 512, the same figure-2 cell family) is provided for a cheap first run.
"""
import os

import numpy as np
from zoology.config import DataConfig, LoggerConfig, ModelConfig, TrainConfig
from zoology.data.multiquery_ar import MQARConfig

VOCAB_SIZE = 8_192
CELL = os.environ.get("ZOO_CELL", "n256_d64")          # n256_d64 (main) | n64_d64 (smoke)
CACHE = os.environ.get("ZOO_CACHE")                    # None = no data cache
LRS = [float(x) for x in os.environ["ZOO_LRS"].split(",")] if os.environ.get("ZOO_LRS") else list(np.logspace(-4, -2, 4))

input_seq_len, num_kv_pairs, batch_size, d_model = {"n256_d64": (256, 16, 256, 64), "n64_d64": (64, 4, 512, 64)}[CELL]

factory_kwargs = {"num_kv_pairs": num_kv_pairs, "train_power_a": 0.01, "test_power_a": 0.01, "random_non_queries": False}
data = DataConfig(
    train_configs=[MQARConfig(num_examples=100_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
    test_configs=[MQARConfig(num_examples=3_000, vocab_size=VOCAB_SIZE, input_seq_len=input_seq_len, **factory_kwargs)],
    batch_size=batch_size,
    cache_dir=CACHE,
)

configs = []
for lr in LRS:
    model = ModelConfig(
        d_model=d_model,
        n_layers=2,
        block_type="TransformerBlock",
        max_position_embeddings=input_seq_len,
        vocab_size=VOCAB_SIZE,
        sequence_mixer=dict(name="zoology.mixers.attention.MHA", kwargs={"dropout": 0.1, "num_heads": 1}),
        state_mixer=dict(name="torch.nn.Identity", kwargs={}),
    )
    configs.append(TrainConfig(
        model=model, data=data, learning_rate=lr, max_epochs=64,
        run_id=f"attention-seqlen{input_seq_len}-dmodel{d_model}-lr{lr}-kv{num_kv_pairs}",
        logger=LoggerConfig(),
    ))
