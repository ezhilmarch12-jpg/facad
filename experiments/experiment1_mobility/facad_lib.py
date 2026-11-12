"""FACAD: Festival-Aware Calendar-Adjusted anomaly Detection — core library.

Kept as a plain module so the notebook stays readable; the notebook imports
and documents every function used.
"""
import math
import numpy as np
import pandas as pd
import ephem
import holidays
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import average_precision_score

# ---------------------------------------------------------------- calendar
FESTIVAL_GROUPS = {
    "diwali": ["diwali", "deepavali", "govardhan", "bhai"],
    "harvest": ["pongal", "sankranti", "makar", "thiruvalluvar", "uzhavar", "lohri", "bihu", "magh"],
    "holi": ["holi"],
    "dussehra": ["dussehra", "vijaya", "ayudha", "durga", "navaratri", "maha navami", "mahanavami"],
    "onam": ["onam", "thiruvonam"],
    "ganesh": ["ganesh", "vinayaka"],
    "eid": ["id-ul", "eid", "bakrid", "muharram", "milad"],
    "christmas": ["christmas"],
    "newyear_regional": ["ugadi", "puthandu", "gudi", "vishu", "baisakhi", "vaisakhi", "rongali", "bengali new year", "cheti chand"],
    "national": ["republic", "independence", "gandhi", "ambedkar", "may day", "labour"],
}

STATE_CODES = {
    "Andaman and Nicobar Islands": "AN", "Andhra Pradesh": "AP", "Arunachal Pradesh": "AR",
    "Assam": "AS", "Bihar": "BR", "Chandigarh": "CH", "Chhattisgarh": "CG",
    "Dadra and Nagar Haveli": "DH", "Daman and Diu": "DH", "Delhi": "DL", "Goa": "GA",
    "Gujarat": "GJ", "Haryana": "HR", "Himachal Pradesh": "HP", "Jammu and Kashmir": "JK",
    "Jharkhand": "JH", "Karnataka": "KA", "Kerala": "KL", "Lakshadweep": "LD",
    "Madhya Pradesh": "MP", "Maharashtra": "MH", "Manipur": "MN", "Meghalaya": "ML",
    "Mizoram": "MZ", "Nagaland": "NL", "Odisha": "OD", "Puducherry": "PY", "Punjab": "PB",
    "Rajasthan": "RJ", "Sikkim": "SK", "Tamil Nadu": "TN", "Telangana": "TS",
    "Tripura": "TR", "Uttar Pradesh": "UP", "Uttarakhand": "UK", "West Bengal": "WB",
}


def festival_group(name: str) -> str:
    n = name.lower()
    for g, keys in FESTIVAL_GROUPS.items():
        if any(k in n for k in keys):
            return g
    return "other"


# Curated supplement: major Telugu-state festivals absent from the `holidays` package's AP/TS lists
# (checked for 2020-2025). Sankranti is a 3-day festival (Bhogi, Sankranti, Kanuma).
_SANKRANTI = ["2020-01-15", "2021-01-14", "2022-01-14", "2023-01-15", "2024-01-15", "2025-01-14"]
_UGADI = ["2020-03-25", "2021-04-13", "2022-04-02", "2023-03-22", "2024-04-09", "2025-03-30"]
_VINAYAKA = ["2020-08-22", "2021-09-10", "2022-08-31", "2023-09-18", "2024-09-07", "2025-08-27"]
SUPPLEMENT = {}
for _st in ("AP", "TS"):
    sup = {}
    for d in _SANKRANTI:
        for k in (-1, 0, 1):
            sup[pd.Timestamp(d) + pd.Timedelta(days=k)] = "harvest"
    sup.update({pd.Timestamp(d): "newyear_regional" for d in _UGADI})
    sup.update({pd.Timestamp(d): "ganesh" for d in _VINAYAKA})
    SUPPLEMENT[_st] = sup


def state_festivals(subdiv, years, supplement=True):
    """Return {date: set(groups)} for one state from the public `holidays` package (+ curated supplement)."""
    h = holidays.India(years=years, subdiv=subdiv) if subdiv else holidays.India(years=years)
    out = {}
    for d, name in h.items():
        for part in str(name).split(";"):
            out.setdefault(pd.Timestamp(d), set()).add(festival_group(part))
    if supplement and subdiv in SUPPLEMENT:
        for d, g in SUPPLEMENT[subdiv].items():
            if d.year in years:
                out.setdefault(d, set()).add(g)
    return out


def lunar_features(dates):
    """Moon-sun elongation at 06:00 IST -> tithi (1..30), sin/cos phase, new/full-moon flags."""
    rows = []
    for d in dates:
        t = ephem.Date(pd.Timestamp(d).to_pydatetime().replace(hour=0, minute=30))  # 06:00 IST = 00:30 UTC
        m, s = ephem.Moon(t), ephem.Sun(t)
        elong = (math.degrees(ephem.Ecliptic(m).lon) - math.degrees(ephem.Ecliptic(s).lon)) % 360
        tithi = int(elong // 12) + 1
        rows.append({
            "lunar_sin": math.sin(math.radians(elong)), "lunar_cos": math.cos(math.radians(elong)),
            "tithi": tithi, "amavasya": int(tithi in (30, 1)), "purnima": int(tithi in (15, 16)),
        })
    return pd.DataFrame(rows, index=pd.DatetimeIndex(dates))


GROUPS = list(FESTIVAL_GROUPS) + ["other"]
PRE, POST = 3, 2  # proximity kernel widths (days)


def calendar_frame(dates, fest_map, lunar=None):
    """Calendar features for one region: festival flags by group, proximity kernel, lunar."""
    idx = pd.DatetimeIndex(dates)
    f = pd.DataFrame(0.0, index=idx, columns=[f"fest_{g}" for g in GROUPS])
    any_f = pd.Series(0, index=idx)
    for d, gs in fest_map.items():
        if d in f.index:
            for g in gs:
                f.loc[d, f"fest_{g}"] = 1
            any_f.loc[d] = 1
    f["fest_any"] = any_f
    fest_days = sorted(d for d in fest_map if idx.min() - pd.Timedelta(days=10) <= d <= idx.max() + pd.Timedelta(days=10))
    fd = np.array(fest_days, dtype="datetime64[ns]")
    dd = idx.values
    if len(fd):
        diff = (fd[None, :] - dd[:, None]).astype("timedelta64[D]").astype(int)  # + => festival ahead
        ahead = np.where(diff > 0, diff, 999).min(1)
        behind = np.where(diff < 0, -diff, 999).min(1)
    else:
        ahead = behind = np.full(len(idx), 999)
    for k in range(1, PRE + 1):
        f[f"pre_{k}"] = (ahead == k).astype(int)
    for k in range(1, POST + 1):
        f[f"post_{k}"] = (behind == k).astype(int)
    f["days_to_fest"] = np.minimum(ahead, 30) / 30.0
    if lunar is not None:
        f = f.join(lunar.loc[idx, ["lunar_sin", "lunar_cos", "amavasya", "purnima"]])
    return f


def festival_window(cal):
    """Days a calendar-blind detector is likely to confuse: festival day +/- kernel."""
    cols = ["fest_any"] + [c for c in cal.columns if c.startswith("pre_") or c.startswith("post_")]
    return cal[cols].max(1).astype(bool)


# ---------------------------------------------------------------- history features (no look-ahead)
def history_features(y: pd.Series):
    """Past-only level features for one signal."""
    h = pd.DataFrame(index=y.index)
    past = y.shift(1)
    h["med7"] = past.rolling(7, min_periods=3).median()
    h["med28"] = past.rolling(28, min_periods=7).median()
    h["lag7"] = y.shift(7)
    h["lag14"] = y.shift(14)
    h["sameDOW_med4"] = pd.concat([y.shift(7 * k) for k in range(1, 5)], axis=1).median(1)
    return h


def dow_frame(idx):
    return pd.get_dummies(pd.DatetimeIndex(idx).dayofweek, prefix="dow").set_index(pd.DatetimeIndex(idx)).astype(int)


# ---------------------------------------------------------------- detectors
class ResidualDetector:
    """Normal-behaviour regressor per signal (pooled over regions) + robust residual z-score.

    use_calendar=True  -> FACAD (festival-aware)
    use_calendar=False -> calendar-blind ablation (identical otherwise)
    """

    def __init__(self, signals, cal_cols=None, use_calendar=True, seed=0, combine="max"):
        self.signals, self.cal_cols, self.use_calendar = signals, cal_cols or [], use_calendar
        self.seed, self.combine = seed, combine
        self.models, self.scale = {}, {}

    def _X(self, df, s):
        cols = [f"{s}__{c}" for c in ["med7", "med28", "lag7", "lag14", "sameDOW_med4"]]
        cols += [c for c in df.columns if c.startswith("dow_")]
        if self.use_calendar:
            cols += self.cal_cols
        return df[cols]

    def fit(self, df):
        for s in self.signals:
            m = df[s].notna() & df[f"{s}__med28"].notna()
            X, y = self._X(df[m], s), df.loc[m, s]
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                l2_regularization=1.0, random_state=self.seed)
            mdl.fit(X, y)
            r = y - mdl.predict(X)
            sc = r.groupby(df.loc[m, "region"]).apply(lambda v: 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6)
            self.models[s], self.scale[s] = mdl, sc
        return self

    def calibrate(self, df):
        """Re-estimate the robust residual scale on held-out (validation) data."""
        for s in self.signals:
            X = self._X(df, s)
            m = df[s].notna() & X.notna().all(1)
            r = df.loc[m, s] - self.models[s].predict(X[m])
            self.scale[s] = r.groupby(df.loc[m, "region"]).apply(
                lambda v: 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-6)
        return self

    def zscores(self, df):
        Z = pd.DataFrame(index=df.index)
        for s in self.signals:
            X = self._X(df, s)
            pred = self.models[s].predict(X.fillna(X.median()))
            sc = df["region"].map(self.scale[s]).fillna(self.scale[s].median()).values
            Z[s] = (df[s].values - pred) / sc
        return Z

    def score(self, df):
        Z = self.zscores(df).abs()
        return Z.max(1) if self.combine == "max" else np.sqrt((Z ** 2).mean(1))


class SeasonalZ:
    """Classical baseline: deviation from same-weekday median of past 4 weeks, robust z."""

    def __init__(self, signals):
        self.signals = signals

    def calibrate(self, df):
        return self.fit(df)

    def fit(self, df):
        self.scale = {}
        for s in self.signals:
            r = df[s] - df[f"{s}__sameDOW_med4"]
            self.scale[s] = r.groupby(df["region"]).apply(lambda v: 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))) + 1e-6)
        return self

    def score(self, df):
        Z = pd.DataFrame({s: ((df[s] - df[f"{s}__sameDOW_med4"]) / df["region"].map(self.scale[s])).abs()
                          for s in self.signals})
        return Z.max(1)


class IFDetector:
    """Isolation Forest on signals + level features (+ optional calendar features)."""

    def __init__(self, signals, cal_cols=None, use_calendar=False, seed=0):
        self.signals, self.cal_cols, self.use_calendar, self.seed = signals, cal_cols or [], use_calendar, seed

    def _X(self, df):
        cols = []
        for s in self.signals:
            df = df.assign(**{f"{s}__dev": df[s] - df[f"{s}__sameDOW_med4"]})
            cols.append(f"{s}__dev")
        cols += [c for c in df.columns if c.startswith("dow_")]
        if self.use_calendar:
            cols += self.cal_cols
        return df[cols].fillna(0)

    def fit(self, df):
        self.m = IsolationForest(n_estimators=300, random_state=self.seed).fit(self._X(df))
        return self

    def calibrate(self, df):
        return self

    def score(self, df):
        return pd.Series(-self.m.score_samples(self._X(df)), index=df.index)


# ---------------------------------------------------------------- injection + metrics
def inject(df, signals, n_per_region, rng, protect, mag=(3.0, 6.0), scale=None, n_fest=0):
    """Inject labelled point anomalies into 1-3 random signals on non-protected days.

    Magnitude is m x (robust scale of the signal in that region), sign random.
    Returns a copy with column `label` (1 = injected anomaly).
    """
    out = df.copy()
    out["label"] = 0
    out["fest_attack"] = 0
    for reg, g in out.groupby("region"):
        cand = g.index[~protect.loc[g.index] & g[signals].notna().all(1)]
        if len(cand) < n_per_region:
            continue
        pick = list(rng.choice(cand, n_per_region, replace=False))
        fcand = g.index[protect.loc[g.index] & g[signals].notna().all(1)]
        if n_fest and len(fcand) > 2 * n_fest:
            fpick = rng.choice(fcand, n_fest, replace=False)
            pick += list(fpick)
            out.loc[fpick, "fest_attack"] = 1
        for ix in pick:
            k = rng.integers(1, min(3, len(signals)) + 1)
            sigs = rng.choice(signals, k, replace=False)
            sign = rng.choice([-1, 1])
            for s in sigs:
                sc = scale[s].get(reg, scale[s].median())
                out.loc[ix, s] += sign * rng.uniform(*mag) * sc
            out.loc[ix, "label"] = 1
    return out


def threshold_at_fpr(score, label, fpr):
    neg = score[label == 0]
    return np.quantile(neg, 1 - fpr)


def evaluate(score, label, fest_win, thr, fest_attack=None, suppress=None):
    pred = score >= thr
    if suppress is not None:            # e.g. festival whitelisting baseline
        pred = pred & ~suppress
    tp = int((pred & (label == 1)).sum()); fp = int((pred & (label == 0)).sum())
    fn = int((~pred & (label == 1)).sum())
    neg_f = (label == 0) & fest_win
    neg_n = (label == 0) & ~fest_win
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "FPR_festival": float(pred[neg_f].mean()),
        "FPR_nonfestival": float(pred[neg_n].mean()),
        "FPR_overall": float(pred[label == 0].mean()),
        "precision": prec, "recall": rec,
        "F1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "PR_AUC": float(average_precision_score(label, score)),
        "festival_share_of_FP": float((pred & neg_f).sum() / max(fp, 1)),
        "recall_festival_attacks": float(pred[fest_attack == 1].mean()) if fest_attack is not None and (fest_attack == 1).any() else np.nan,
        "recall_other_attacks": float(pred[(label == 1) & (fest_attack != 1)].mean()) if fest_attack is not None else rec,
    }


# ---------------------------------------------------------------- FACAD (calendar-adjusted residual)
PRIORITY = ["diwali", "harvest", "holi", "dussehra", "onam", "ganesh", "eid", "christmas",
            "newyear_regional", "national", "other"]


def festival_context(cal):
    """For every day pick the dominant festival context (group, offset in -PRE..+POST) or None.

    Offsets: 0 = festival day, -k = k days before, +k = k days after.
    The dominant context is the nearest festival; ties are broken by PRIORITY.
    """
    idx = cal.index
    ctx = pd.Series([None] * len(cal), index=idx, dtype=object)
    best = pd.Series(99, index=idx)
    for off in [0] + [-k for k in range(1, PRE + 1)] + list(range(1, POST + 1)):
        shifted = cal[[f"fest_{g}" for g in GROUPS]].shift(off)  # festival `off` days away (off<0: day is before it)
        for g in PRIORITY[::-1]:  # later assignment = higher priority
            hit = (shifted[f"fest_{g}"] == 1) & (best >= abs(off))
            take = hit & ((best > abs(off)) | ctx.isna() | (best == abs(off)))
            ctx[take] = f"{g}@{off:+d}"
            best[take] = abs(off)
    return ctx


class FACAD:
    """Festival-Aware Calendar-Adjusted anomaly Detector.

    1. Seasonal baseline: same-weekday median of the previous 4 weeks (past only).
    2. Robust scale per (region, signal) from MAD of baseline deviations.
    3. Calendar effect table E[ctx, signal] = median standardised deviation observed on
       training days with that festival context (pooled over all regions that celebrate it),
       and a dispersion factor V[ctx, signal] >= 1.
    4. Adjusted score  z* = (z - E[ctx]) / V[ctx];  anomaly score = max_signal |z*|.
    Contexts seen fewer than `min_support` times in training fall back to their festival-group
    average (offset-agnostic), then to no adjustment.
    """

    def __init__(self, signals, min_support=3, shrink=5.0):
        self.signals, self.min_support, self.shrink = signals, min_support, shrink

    def _z(self, df):
        return pd.DataFrame({s: (df[s] - df[f"{s}__sameDOW_med4"]) / df["region"].map(self.scale[s])
                             for s in self.signals}, index=df.index)

    def fit(self, df, ctx):
        nonf = ctx.isna()
        self.scale = {}
        for s in self.signals:
            r = (df[s] - df[f"{s}__sameDOW_med4"])[nonf]
            self.scale[s] = r.groupby(df.loc[nonf, "region"]).apply(
                lambda v: 1.4826 * np.nanmedian(np.abs(v - np.nanmedian(v))) + 1e-6)
        Z = self._z(df)
        Z["ctx"] = ctx.values
        Z["grp"] = Z["ctx"].str.split("@").str[0]
        self.E, self.V, self.Eg, self.Vg = {}, {}, {}, {}
        for key, tab, store_e, store_v in [("ctx", None, self.E, self.V), ("grp", None, self.Eg, self.Vg)]:
            for k, g in Z.dropna(subset=[key]).groupby(key):
                n = len(g)
                if n < self.min_support:
                    continue
                w = n / (n + self.shrink)  # shrink towards 0 for rare contexts
                med = g[self.signals].median()
                mad = 1.4826 * (g[self.signals] - med).abs().median()
                store_e[k] = w * med
                store_v[k] = np.maximum(1.0, 1 + w * (mad - 1))
        return self

    def effect_frames(self, df, ctx):
        """Vectorised lookup of E and V for every row (context -> group fallback -> none)."""
        ctx = pd.Series(np.asarray(ctx, dtype=object), index=df.index)
        key = ctx.where(ctx.isin(list(self.E)), None)
        grp = ctx.where(key.isna() & ctx.notna(), None).map(lambda c: c.split("@")[0] if isinstance(c, str) else None)
        Et = pd.DataFrame(self.E).T.reindex(columns=self.signals); Vt = pd.DataFrame(self.V).T.reindex(columns=self.signals)
        Egt = pd.DataFrame(self.Eg).T.reindex(columns=self.signals); Vgt = pd.DataFrame(self.Vg).T.reindex(columns=self.signals)
        E = Et.reindex(key.values).values
        V = Vt.reindex(key.values).values
        Eg = Egt.reindex(grp.values).values if len(Egt) else np.full_like(E, np.nan)
        Vg = Vgt.reindex(grp.values).values if len(Vgt) else np.full_like(V, np.nan)
        E = np.where(np.isnan(E), Eg, E); V = np.where(np.isnan(V), Vg, V)
        E = np.nan_to_num(E, nan=0.0); V = np.nan_to_num(V, nan=1.0)
        return (pd.DataFrame(E, index=df.index, columns=self.signals),
                pd.DataFrame(V, index=df.index, columns=self.signals))

    def adjusted_z(self, df, ctx):
        E, V = self.effect_frames(df, ctx)
        return (self._z(df) - E) / V

    def score(self, df, ctx):
        return self.adjusted_z(df, ctx).abs().max(1)


class FACADWrap:
    """FACAD calendar adjustment applied on top of ANY residual detector (here the calendar-blind GBR).

    The base detector's per-signal z-scores replace the seasonal z; E/V tables are learned from the base
    detector's z-scores on the training period, exactly as in FACAD.
    """

    def __init__(self, base, signals, min_support=3, shrink=5.0):
        self.base, self.signals = base, signals
        self.adj = FACAD(signals, min_support, shrink)

    def fit(self, df, ctx):
        Z = self.base.zscores(df)
        Z["ctx"] = np.asarray(ctx, dtype=object)
        Z["grp"] = Z["ctx"].map(lambda c: c.split("@")[0] if isinstance(c, str) else None)
        self.adj.E, self.adj.V, self.adj.Eg, self.adj.Vg = {}, {}, {}, {}
        for key, store_e, store_v in [("ctx", self.adj.E, self.adj.V), ("grp", self.adj.Eg, self.adj.Vg)]:
            for k, g in Z.dropna(subset=[key]).groupby(key):
                n = len(g)
                if n < self.adj.min_support:
                    continue
                w = n / (n + self.adj.shrink)
                med = g[self.signals].median()
                mad = 1.4826 * (g[self.signals] - med).abs().median()
                # base z is already standardised by non-calendar scale; centre its non-festival spread at 1
                store_e[k] = w * med
                store_v[k] = np.maximum(1.0, 1 + w * (mad / self._ref - 1))
        return self

    def set_reference(self, df, ctx):
        """Typical |z| spread on non-calendar days, so V is relative to normal-day dispersion."""
        Z = self.base.zscores(df)
        nonf = pd.isna(pd.Series(np.asarray(ctx, dtype=object)))
        Zn = Z[nonf.values]
        self._ref = pd.Series(1.4826 * (Zn - Zn.median()).abs().median(), index=self.signals).clip(lower=1e-6)
        return self

    def score(self, df, ctx):
        Z = self.base.zscores(df)
        E, V = self.adj.effect_frames(df, ctx)
        return ((Z - E) / V).abs().max(1)
