"""FACAD: Festival-Aware Calendar-Adjusted anomaly Detection.

Learns how each festival context shifts (E) and widens (V) the standardised residuals of any
anomaly detector, pools these effects across regions with shrinkage, optionally gates the shift
on cross-region coherence, removes them before scoring, and sets Mondrian conformal thresholds
per calendar context.
"""
from .calendar import FESTIVAL_GROUPS, PRIORITY, festival_group, festival_contexts, state_festivals
from .effects import EffectTable, coherence
from .detector import FACAD
from .baselines import robust_z, seasonal_z
from .conformal import MondrianConformal
from .metrics import evaluate, inject_point_anomalies, threshold_at_fpr

__version__ = "1.0.0"
__all__ = ["FACAD", "EffectTable", "MondrianConformal", "coherence", "festival_contexts",
           "festival_group", "state_festivals", "FESTIVAL_GROUPS", "PRIORITY", "robust_z",
           "seasonal_z", "evaluate", "inject_point_anomalies", "threshold_at_fpr"]
