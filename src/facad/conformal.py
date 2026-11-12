"""Mondrian (context-conditional) split-conformal thresholds."""
from __future__ import annotations

from typing import Dict, Iterable

import numpy as np
import pandas as pd


class MondrianConformal:
    """One threshold per calendar category (e.g. holiday / window / normal).

    Each threshold is the ceil((n + 1)(1 - alpha))-th smallest calibration score of clean (non-anomalous)
    rows in that category, so the false-alarm rate is at most alpha within every category under
    exchangeability. Categories with fewer than ``min_n`` calibration rows use the pooled threshold.
    """

    def __init__(self, alpha: float = 0.05, min_n: int = 20):
        self.alpha, self.min_n = alpha, min_n
        self.thresholds: Dict[str, float] = {}

    @staticmethod
    def _q(s: np.ndarray, alpha: float) -> float:
        s = np.sort(s[~np.isnan(s)])
        n = len(s)
        k = int(np.ceil((n + 1) * (1 - alpha))) - 1
        return float(s[min(k, n - 1)]) if n else np.inf

    def fit(self, scores: Iterable[float], category: Iterable[str]) -> "MondrianConformal":
        df = pd.DataFrame({"s": np.asarray(scores, float), "c": list(category)})
        self.pooled = self._q(df.s.values, self.alpha)
        self.thresholds = {c: (self._q(g.s.values, self.alpha) if len(g) >= self.min_n else self.pooled)
                           for c, g in df.groupby("c")}
        return self

    def threshold(self, category: Iterable[str]) -> np.ndarray:
        return np.array([self.thresholds.get(c, self.pooled) for c in category])

    def predict(self, scores: Iterable[float], category: Iterable[str]) -> np.ndarray:
        """Boolean alarms."""
        return np.asarray(scores, float) >= self.threshold(list(category))
