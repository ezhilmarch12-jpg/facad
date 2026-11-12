"""FACAD: calendar adjustment of standardised residuals from any detector."""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import pandas as pd

from .effects import EffectTable


class FACAD:
    """Festival-Aware Calendar-Adjusted Detection.

    Adjusted residual  z* = (z - E_c) / V_c ;  score = max over signals of |z*|.

    Parameters
    ----------
    lam : shrinkage strength (w = n / (n + lam)); 5 in the paper.
    gate : coherence gate (FACAD-CG): keep the shift only where coherence C >= tau.
    tau : coherence threshold, 0.5 in the paper.
    hierarchical : also learn one table per region group (FACAD-H) and blend it with the national
        table with weight n_region / (n_region + lam).
    mode : "full" (E and V), "shift" (E only), "spread" (V only) or "none" (calendar-blind).
    spread_prior : optional {context: V} from an external source (e.g. mobility data); the applied
        spread becomes max(V, V_prior) (FACAD-HP).
    """

    def __init__(self, lam: float = 5.0, gate: bool = False, tau: float = 0.5, hierarchical: bool = False,
                 mode: str = "full", min_support: int = 1, spread_prior: Optional[Dict[str, float]] = None):
        if mode not in ("full", "shift", "spread", "none"):
            raise ValueError("mode must be full, shift, spread or none")
        self.lam, self.gate, self.tau, self.hierarchical = lam, gate, tau, hierarchical
        self.mode, self.min_support, self.spread_prior = mode, min_support, spread_prior or {}
        self.national: Optional[EffectTable] = None
        self.regional: Dict[str, EffectTable] = {}
        self.region_group: Dict[str, str] = {}

    def _table(self):
        return EffectTable(lam=self.lam, min_support=self.min_support, gate=self.gate, tau=self.tau)

    def fit(self, z: pd.DataFrame, ctx, region=None, day=None, block=None,
            region_group: Optional[Dict[str, str]] = None) -> "FACAD":
        """Learn effect tables from training residuals ``z`` (rows x signals) and their contexts."""
        z = pd.DataFrame(z).reset_index(drop=True)
        self.signals = list(z.columns)
        self.national = self._table().fit(z, ctx, region, day, block)
        self.regional = {}
        if self.hierarchical:
            if region is None or region_group is None:
                raise ValueError("hierarchical=True needs region and region_group")
            self.region_group = dict(region_group)
            rg = pd.Series(np.asarray(region, dtype=object)).map(self.region_group)
            for grp in rg.dropna().unique():
                m = (rg == grp).values
                self.regional[grp] = self._table().fit(
                    z[m], np.asarray(ctx, dtype=object)[m], np.asarray(region, dtype=object)[m],
                    None if day is None else np.asarray(day, dtype=object)[m],
                    None if block is None else np.asarray(block, dtype=object)[m])
        return self

    def effects(self, ctx, region=None, block=None):
        """Row-aligned E and V frames for the given contexts (and regions, for FACAD-H)."""
        if self.national is None:
            raise RuntimeError("call fit() first")
        ctx = list(np.asarray(ctx, dtype=object))
        E, V, _ = self.national.frames(ctx, block)
        if self.hierarchical and region is not None:
            regs = list(np.asarray(region, dtype=object))
            for grp, tab in self.regional.items():
                rows = [i for i, r in enumerate(regs) if self.region_group.get(r) == grp]
                if not rows:
                    continue
                Er, Vr, Nr = tab.frames([ctx[i] for i in rows],
                                        None if block is None else [list(block)[i] for i in rows])
                w = Nr.values / (Nr.values + self.lam)
                E.iloc[rows] = w * Er.values + (1 - w) * E.iloc[rows].values
                V.iloc[rows] = w * Vr.values + (1 - w) * V.iloc[rows].values
        if self.spread_prior:
            vp = np.array([self.spread_prior.get(c, 1.0) if isinstance(c, str) else 1.0 for c in ctx])
            V = pd.DataFrame(np.maximum(V.values, vp[:, None]), columns=V.columns)
        if self.mode in ("spread", "none"):
            E[:] = 0.0
        if self.mode in ("shift", "none"):
            V[:] = 1.0
        return E, V

    def adjust(self, z: pd.DataFrame, ctx, region=None, block=None) -> pd.DataFrame:
        """Calendar-adjusted residuals z* = (z - E) / V, same index as ``z``."""
        z = pd.DataFrame(z)
        E, V = self.effects(ctx, region, block)
        return pd.DataFrame((z[self.signals].values - E.values) / V.values, index=z.index, columns=self.signals)

    def score(self, z: pd.DataFrame, ctx, region=None, block=None) -> pd.Series:
        """Anomaly score: max over signals of |z*|."""
        return self.adjust(z, ctx, region, block).abs().max(axis=1)
