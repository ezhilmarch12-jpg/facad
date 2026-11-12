"""Evaluation helpers: matched-budget thresholds, metrics and point-anomaly injection."""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def threshold_at_fpr(score, label, fpr: float = 0.05) -> float:
    """Threshold giving the requested false-alarm rate on normal rows (the 'matched budget')."""
    score, label = np.asarray(score, float), np.asarray(label)
    return float(np.quantile(score[label == 0], 1 - fpr))


def evaluate(score, label, in_window, thr: float, suppress=None) -> dict:
    """Overall and festival-window false-alarm rates, precision, recall, F1, PR-AUC and in-window recall.

    ``suppress`` (boolean array) removes alarms, e.g. for the whitelisting baseline.
    """
    score, label = np.asarray(score, float), np.asarray(label).astype(int)
    win = np.asarray(in_window).astype(bool)
    pred = score >= thr
    if suppress is not None:
        pred &= ~np.asarray(suppress).astype(bool)
    tp = int((pred & (label == 1)).sum()); fp = int((pred & (label == 0)).sum())
    fn = int((~pred & (label == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    neg = label == 0
    return {
        "FPR_window": float(pred[neg & win].mean()) if (neg & win).any() else np.nan,
        "FPR_outside": float(pred[neg & ~win].mean()) if (neg & ~win).any() else np.nan,
        "FPR_overall": float(pred[neg].mean()),
        "precision": prec, "recall": rec, "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "PR_AUC": float(average_precision_score(label, score)) if label.any() else np.nan,
        "recall_in_window": float(pred[(label == 1) & win].mean()) if ((label == 1) & win).any() else np.nan,
    }


def inject_point_anomalies(z: pd.DataFrame, n: int, rng: np.random.Generator, eligible=None,
                           magnitude=(3.0, 6.0), max_signals: int = 3):
    """Add +/- magnitude (in residual-scale units) to 1..max_signals random signals on ``n`` random rows.

    Returns (z_injected, label). ``eligible`` (boolean array) limits which rows may be chosen.
    """
    z = pd.DataFrame(z).copy()
    label = np.zeros(len(z), dtype=int)
    cand = np.flatnonzero(np.ones(len(z), bool) if eligible is None else np.asarray(eligible, bool))
    for i in rng.choice(cand, size=min(n, len(cand)), replace=False):
        k = rng.integers(1, min(max_signals, z.shape[1]) + 1)
        cols = rng.choice(z.columns, size=k, replace=False)
        z.iloc[i, [z.columns.get_loc(c) for c in cols]] += rng.choice([-1, 1]) * rng.uniform(*magnitude)
        label[i] = 1
    return z, label
