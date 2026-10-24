"""SMT x MemUP on Stack and MQAR: the compact, executable core.

Read in this order (each file is short and depends only on the files above it):
    upstream.py   pinned author code (SMT Teacher/RNN and losses, Stack generator, Zoology MQAR) + seeds
    config.py     one Config dataclass; JSON configs in configs/
    data.py       1. Stack and MQAR batches: token IDs, answers, loss mask, metadata
    windows.py    2. SMT windows: for every anchor t the encoder context, decoder inputs and targets
    models.py     3. models: author Teacher (encoder, memory, decoder) and RNN (updater f + readout), detector,
                     causal Transformer; forward helpers (encode, rollout, full-access read)
    losses.py     4. every loss: L_dec^Tf, L_far, L_dyn, L_unif (SMT), DMT, direct CE (BPTT, LM)
    train.py      5. SMT -> DMT stages, BPTT and LM baselines; optimiser step; checkpoint selection; run identity
    evaluate.py   6. inference and metrics (teacher j = 0, teacher before the query, RNN, LM)
    cli.py        trace / demo / train commands  (python -m smtcore ...)

Nothing here re-implements SMT: Teacher and RNN are the authors' classes (upstream/smt/src/model.py at cf0d0f0).
tests/test_equivalence.py checks every function against the archived experiment code (archive/).
"""
