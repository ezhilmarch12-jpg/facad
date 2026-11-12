"""Residual producers: robust z-scores against a seasonal (same-weekday) baseline."""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_K = 1.4826


def robust_z(x: pd.Series, baseline: pd.Series, mask_scale=None) -> pd.Series:
    """(x - baseline) / (1.4826 * MAD of x - baseline); MAD computed on ``mask_scale`` rows if given."""
    r = x - baseline
    ref = r[mask_scale] if mask_scale is not None else r
    ref = ref.dropna()
    mad = MAD_K * np.median(np.abs(ref - np.median(ref))) if len(ref) else np.nan
    return r / max(mad, 1e-6)


def seasonal_z(y: pd.Series, period: int = 7, k: int = 4, mask_scale=None) -> pd.Series:
    """Calendar-blind SeasonalZ: deviation from the median of the same slot in the previous ``k`` periods.

    ``period`` = 7 for daily data with a weekly cycle, 168 for hourly data. Uses past values only.
    """
    base = pd.concat([y.shift(period * j) for j in range(1, k + 1)], axis=1).median(axis=1)
    return robust_z(y, base, mask_scale)
