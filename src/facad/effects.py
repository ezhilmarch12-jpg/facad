"""Festival effect tables: shift E and spread V per (context[, hour block], signal)."""
from __future__ import annotations

from typing import Dict, Hashable, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

MAD_K = 1.4826


def _mad(x) -> float:
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan
    return float(MAD_K * np.median(np.abs(x - np.median(x))))


def coherence(z: Iterable[float], region: Iterable[Hashable], day: Optional[Iterable] = None) -> float:
    """Festival coherence index C = |sum_r n_r sign(m_r)| / sum_r n_r.

    m_r is region r's median residual in the context and n_r its number of days (distinct ``day``
    values, or rows when ``day`` is None). C = 1 when all regions move the same way; near 0 when they cancel.
    """
    df = pd.DataFrame({"z": np.asarray(z, float), "r": list(region),
                       "d": list(day) if day is not None else range(len(list(region)))}).dropna(subset=["z"])
    if df.empty:
        return 0.0
    med = df.groupby("r").z.median()
    if len(med) < 2:
        return 1.0
    n = df.groupby("r").d.nunique().reindex(med.index)
    return float(abs((np.sign(med) * n).sum()) / n.sum())


class EffectTable:
    """Shrunken festival effect table learned from training residuals.

    For each context c (optionally split by hour block) and signal s:

    * ``w = n / (n + lam)`` with n the number of region-days in c;
    * ``E = w * median(z)`` (set to 0 when ``gate`` is on and coherence C < ``tau``);
    * ``V = max(1, 1 + w * (MAD(z) / MAD_ref - 1))`` with MAD_ref the spread on days without context.

    Lookups fall back from (context, block) to (context,) to the festival group, then to no correction.
    """

    def __init__(self, lam: float = 5.0, min_support: int = 1, gate: bool = False, tau: float = 0.5,
                 group_fallback: bool = True):
        self.lam, self.min_support, self.gate, self.tau = lam, min_support, gate, tau
        self.group_fallback = group_fallback
        self.table: Dict[Tuple, Dict[str, Tuple[float, float, int, float]]] = {}
        self.signals: list = []

    # ------------------------------------------------------------------ learning
    def fit(self, z: pd.DataFrame, ctx: pd.Series, region: Optional[pd.Series] = None,
            day: Optional[pd.Series] = None, block: Optional[pd.Series] = None) -> "EffectTable":
        """z: standardised residuals (rows x signals); ctx: context label per row (None/"" = none)."""
        z = pd.DataFrame(z).reset_index(drop=True)
        self.signals = list(z.columns)
        n_rows = len(z)
        ctx = pd.Series(np.asarray(ctx, dtype=object)).replace("", None)
        region = pd.Series(np.asarray(region, dtype=object)) if region is not None else pd.Series(["all"] * n_rows)
        day = pd.Series(np.asarray(day, dtype=object)) if day is not None else pd.Series(np.arange(n_rows))
        D = z.assign(_ctx=ctx, _reg=region, _day=day)
        D["_grp"] = D._ctx.map(lambda c: c.split("@")[0] if isinstance(c, str) else None)
        if block is not None:
            D["_blk"] = np.asarray(block, dtype=object)
        none = D._ctx.isna()
        ref = {s: (_mad(D.loc[none, s]) or 1.0) for s in self.signals}
        keysets = [("_ctx",)]
        if block is not None:
            keysets.insert(0, ("_ctx", "_blk"))
        if self.group_fallback:
            keysets.append(("_grp",))
        self.table = {}
        for keys in keysets:
            sub = D[~none]
            for k, g in sub.groupby(list(keys)):
                k = k if isinstance(k, tuple) else (k,)
                tag = "grp" if keys == ("_grp",) else "ctx"
                n = g[["_reg", "_day"]].drop_duplicates().shape[0]
                if n < self.min_support:
                    continue
                w = n / (n + self.lam) if self.lam > 0 else 1.0
                entry = {}
                for s in self.signals:
                    C = coherence(g[s], g["_reg"], g["_day"])
                    E = w * float(np.nanmedian(g[s]))
                    if self.gate and C < self.tau:
                        E = 0.0
                    m = _mad(g[s])
                    V = max(1.0, 1.0 + w * ((m if np.isfinite(m) else ref[s]) / ref[s] - 1.0))
                    entry[s] = (E, V, n, C)
                self.table[(tag,) + k] = entry
        return self

    # ------------------------------------------------------------------ lookup
    def lookup(self, ctx, block=None) -> Dict[str, Tuple[float, float, int, float]]:
        if not isinstance(ctx, str) or ctx == "":
            return {}
        for key in ([("ctx", ctx, block)] if block is not None else []) + [("ctx", ctx)] + \
                   ([("grp", ctx.split("@")[0])] if self.group_fallback else []):
            if key in self.table:
                return self.table[key]
        return {}

    def frames(self, ctx: Iterable, block: Optional[Iterable] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """E, V and n as row-aligned frames for a sequence of contexts."""
        ctx = list(ctx)
        blocks = list(block) if block is not None else [None] * len(ctx)
        E = np.zeros((len(ctx), len(self.signals))); V = np.ones_like(E); N = np.zeros_like(E)
        for i, (c, b) in enumerate(zip(ctx, blocks)):
            ent = self.lookup(c, b)
            for j, s in enumerate(self.signals):
                if s in ent:
                    E[i, j], V[i, j], N[i, j] = ent[s][0], ent[s][1], ent[s][2]
        cols = self.signals
        return pd.DataFrame(E, columns=cols), pd.DataFrame(V, columns=cols), pd.DataFrame(N, columns=cols)

    def to_frame(self) -> pd.DataFrame:
        """Long table: level, key, signal, E, V, n, C (for inspection or export)."""
        rows = []
        for key, ent in self.table.items():
            for s, (E, V, n, C) in ent.items():
                rows.append({"level": key[0], "key": "|".join(map(str, key[1:])), "signal": s,
                             "E": E, "V": V, "n": n, "C": C})
        return pd.DataFrame(rows)
