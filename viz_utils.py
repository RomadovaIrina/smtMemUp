"""Visualisation helpers for SMT_MemUP_FROM_ZERO_v2.ipynb: tables and figures only.
Nothing here computes a model output, a loss or a metric that the notebook relies on; every function receives
already-computed data (tensors, dicts, DataFrames) and only formats or draws it."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

IGNORE = -1


def show(df_or_rows, columns=None, title=None):
    """Display a table (DataFrame or list of dicts / tuples) in Jupyter, or print it outside Jupyter."""
    df = df_or_rows if isinstance(df_or_rows, pd.DataFrame) else pd.DataFrame(df_or_rows, columns=columns)
    if title:
        print(title)
    try:
        from IPython.display import display
        display(df)
    except ImportError:
        print(df.to_string())
    return df


def stack_table(batch, b=0):
    """Position table of one Stack sequence: input ID, action, stack before/after, answer, mask, source, delay."""
    x, y = batch["x"][b].tolist(), batch["y"][b].tolist()
    st, rows = [], []
    for t, tok in enumerate(x):
        before = " ".join(map(str, st)) or "-"
        st.append(tok) if tok > 0 else st.pop()
        rows.append(dict(t=t, input_id=tok, action=f"push {tok}" if tok else "pop", stack_before=before,
                         stack_after=" ".join(map(str, st)) or "-", answer_y=y[t], in_loss=int(y[t] != IGNORE),
                         source_pos=int(batch["source_pos"][b, t]), delay=int(batch["delay"][b, t])))
    return pd.DataFrame(rows)


def mqar_table(batch, num_kv_pairs, b=0):
    """Position table of one MQAR sequence: input ID, role (key / value / query / filler), answer, mask, source."""
    x, y, N = batch["x"][b].tolist(), batch["y"][b].tolist(), num_kv_pairs
    role = lambda t: (f"key {t // 2}" if t < 2 * N and t % 2 == 0 else f"value {t // 2}" if t < 2 * N
                      else "QUERY key" if y[t] != IGNORE else "filler 0")
    return pd.DataFrame([dict(t=t, input_id=x[t], role=role(t), answer_y=y[t], in_loss=int(y[t] != IGNORE),
                              source_pos=int(batch["source_pos"][b, t]), delay=int(batch["delay"][b, t]))
                         for t in range(len(x))])


def window_table(w, row, t, js, Tf=None):
    """One anchor of precomputed windows w (dict of [R,T] tensors): decoder inputs / target / masks per output j."""
    print(f"anchor t={t}: x_ctx = {w['x_ctx'][row].tolist()}")
    return show([dict(j=j, decoder_inputs=str(w["x_fut"][row, :j].tolist()), target=f"y_{t + j} = {int(w['y_fut'][row, j])}",
                      valid=bool(w["valid"][row, j]),
                      in_L_dec=(None if Tf is None else bool(w["valid"][row, j] and j < Tf)),
                      written_at_or_before_t=bool(w["prefix"][row, j])) for j in js])


def archived_curves(R: Path, panels, chance=0.25):
    """panels = [(title, {label: history.csv path relative to R}, accuracy column, CE column)].
    Plots accuracy vs optimizer step; o = checkpoint with min CE (selection rule), x = last point. No smoothing."""
    fig, axes = plt.subplots(1, len(panels), figsize=(5.7 * len(panels), 4.2))
    for ax, (title, runs, col, ce) in zip(axes, panels):
        for label, path in runs.items():
            h = pd.read_csv(R / path).dropna(subset=[ce])
            ax.plot(h.step, h[col], label=label)
            i = h[ce].idxmin()
            ax.scatter(h.loc[i, "step"], h.loc[i, col], marker="o", s=40)
            ax.scatter(h.step.iloc[-1], h[col].iloc[-1], marker="x", s=50)
        ax.set_title(title, fontsize=9); ax.set_xlabel("optimizer step"); ax.set_ylabel(f"{col} (long answers)")
        ax.axhline(chance, ls=":", c="gray"); ax.set_ylim(0, 1.02); ax.legend(fontsize=8)
    fig.text(0.5, -0.03, "o = checkpoint selected by min validation CE on long answers; x = last point; no smoothing",
             ha="center", fontsize=9)
    plt.tight_layout(); plt.show()
