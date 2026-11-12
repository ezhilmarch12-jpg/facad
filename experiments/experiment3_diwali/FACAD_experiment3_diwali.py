# %% [markdown]
# # FACAD on real Indian traffic — Cloudflare Radar NetFlows by state, 17–24 Oct 2026
#
# This notebook applies **FACAD (Festival-Aware Calendar-Adjusted anomaly Detection)** to hourly Cloudflare traffic shares
# for 19 Indian states during the Dussehra / Durga Puja week. It runs leave-one-state-out experiments with injected anomalies,
# ablations, a cold-start transfer from mobility data, and a check on real (unlabelled) hours.
#
# **Assumptions (stated before use)**
# 1. `timestamp_utc` is the master clock; Excel-formatted IST columns are only verified against it.
# 2. The target is `clr` = ln(share) − mean of ln(share) over all 20 regions in the hour. `OTHER` enters only that mean.
# 3. Festival effects for a held-out state are learned from the **other 18 states in the same week** (cross-state transfer).
#    This uses other states' data from the same days, including hours after the hour being scored, so E-A measures
#    *transfer across states*, not prediction from past years. E-B (mobility prior from 2020–22) is the strictly earlier source.
# 4. Robust scales for the causal variants are expanding MADs of **past** residuals only (≥ 24 past residuals; non-holiday hours
#    if ≥ 24 exist, else all past hours). This sets a common scoring window (printed in Section 2: 19 Oct 07:30 to 24 Oct 05:30 IST, 119 h per state).
# 5. Effect-table sample sizes are counted in **state-days**, so the shrinkage λ = 5 acts on days, not hours.
# 6. Injected anomalies shift the held-out state's log-share by δ; after re-normalising shares to 100, its clr moves by 0.95·δ
#    (other states move by −δ/20, which does not affect the held-out state's score and is ignored for them).
# 7. Isolation Forest calibration scores are in-sample on training states; Prophet is fitted on each state's own week
#    (transductive). Both are labelled as such.

# %%
import os, sys, json, time, math, warnings, tracemalloc, logging
warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR); logging.getLogger("prophet").setLevel(logging.ERROR)
import numpy as np, pandas as pd, scipy, sklearn, matplotlib, lightgbm as lgb, holidays, prophet
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import wilcoxon
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

SEEDS = list(range(int(os.environ.get("FACAD_SEEDS", "10")))); ALPHA = 0.05; LAMBDA = 5.0; MATCH_FPR = 0.05; N_EVENTS = 8; N_IN_WINDOW = 3
os.makedirs("results", exist_ok=True); os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6, "font.size": 9,
                     "axes.titlesize": 9.5, "axes.titlelocation": "left", "axes.edgecolor": "#8a8984"})
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
GREY = "#b9b8b2"; HOL = "#eda100"
RUNLOG = []
def log(m):
    RUNLOG.append(m); print(m)
for p in (np, pd, scipy, sklearn, matplotlib, lgb, holidays, prophet):
    print(f"{p.__name__:12s} {p.__version__}")

# %% [markdown]
# ## 1. Data, CLR and calendars

# %%
SRC = "data/cloudflare_radar_india_state_netflows_hourly_20261024_20261111_long.csv"
raw = pd.read_csv(SRC, keep_default_na=False, dtype=str)
df = raw.copy()
df["ts"] = pd.to_datetime(df.timestamp_utc, utc=True)
ist = df.ts.dt.tz_convert("Asia/Kolkata").dt.tz_localize(None)
assert (pd.to_datetime(raw.timestamp_ist, format="%Y-%m-%d %H:%M") == ist).all()
df["ist"], df["date"] = ist, ist.dt.strftime("%Y-%m-%d")
df["hour"] = ist.dt.hour
for c in ["share_pct"]:
    df[c] = pd.to_numeric(df[c])
for c in ["is_holiday", "in_festival_window", "is_weekend"]:
    df[c] = pd.to_numeric(df[c]).astype(int)
assert df.groupby("ts").share_pct.sum().sub(100).abs().max() < 1e-3
df["lnshare"] = np.log(df.share_pct)
df["clr"] = df.lnshare - df.groupby("ts").lnshare.transform("mean")
STATES = sorted(s for s in df.state_code.unique() if s != "OTHER")
TS = np.sort(df.ts.unique()); T = len(TS)
log(f"Loaded {len(df)} rows; {len(STATES)} states + OTHER; {T} hours; CLR computed over 20 regions")

REGION = {"AP": "South", "TS": "South", "KA": "South", "KL": "South", "TN": "South",
          "MH": "West", "GJ": "West", "MP": "West",
          "DL": "North", "HR": "North", "PB": "North", "JK": "North", "UP": "North", "RJ": "North",
          "WB": "East", "OD": "East", "BR": "East", "JH": "East", "AS": "East"}
print(pd.Series(REGION).groupby(lambda s: REGION[s]).apply(lambda v: ", ".join(v.index)).to_string())

# national calendar (same for every state), with the corrected offset convention (k < 0 = before the festival)
def group_of(name):
    n = name.lower()
    if any(k in n for k in ["diwali", "deepavali", "govardhan", "bhai duj", "bhai dooj"]):
        return "diwali"
    if any(k in n for k in ["dussehra", "vijaya", "durga", "navami", "ashtami", "saptami", "ayudha", "mahalaya"]):
        return "dussehra"
    return "national" if any(k in n for k in ["gandhi", "independence", "republic"]) else "other"
PRI = ["diwali", "dussehra", "national", "other"]
def context_for(dates, fmap, pre=3, post=2):
    out = {}
    for d in dates:
        best = None
        for fd, g in fmap.items():
            k = (d - fd).days
            if -pre <= k <= post:
                key = (abs(k), PRI.index(g))
                if best is None or key < best[0]:
                    best = (key, f"{g}@{k:+d}")
        out[d] = best[1] if best else ""
    return out
nat = {pd.Timestamp(d): group_of(n) for d, n in holidays.India(years=2026).items()}
days = pd.to_datetime(sorted(df.date.unique()))
ctx_nat = {d.strftime("%Y-%m-%d"): c for d, c in context_for(days, nat).items()}
print("national contexts:", ctx_nat)

# per-state frames (167 rows each, time-ordered)
S = {}
for s in STATES:
    g = df[df.state_code == s].sort_values("ts").reset_index(drop=True)
    assert len(g) == T
    g["ctx"] = g.festival_context
    g["ctx_day"] = np.where(g.ctx.str.endswith("@+0"), g.ctx, "")
    g["ctx_nat"] = g.date.map(ctx_nat)
    g["is_holiday_nat"] = g.date.isin([d for d, c in ctx_nat.items() if c.endswith("@+0")]).astype(int)
    g["block"] = g.hour // 6
    g["offset"] = pd.to_numeric(g.ctx.str.extract(r"@([+-]\d+)")[0], errors="coerce")
    g["bucket"] = np.where(g.is_holiday == 1, "holiday", np.where(g.in_festival_window == 1, "window", "normal"))
    S[s] = g[["ts", "ist", "date", "hour", "block", "is_weekend", "is_holiday", "in_festival_window", "ctx", "ctx_day",
              "ctx_nat", "is_holiday_nat", "offset", "bucket", "clr", "share_pct", "holiday_name"]]
print(S["MH"].groupby("date")[["ctx", "bucket", "holiday_name"]].first().to_string())

# %% [markdown]
# ## 2. Baselines, scales and features

# %%
MIN_PAST = 24
def resid_causal(x, hour):
    """clr minus the median of the same IST hour on all earlier dates (NaN if none)."""
    return x - x.groupby(hour).transform(lambda v: v.shift(1).expanding().median())

def resid_lodo(x, hour):
    """clr minus the median of the same IST hour on all OTHER dates (transductive)."""
    out = pd.Series(np.nan, index=x.index)
    for h, v in x.groupby(hour):
        vals = v.values
        for j, ix in enumerate(v.index):
            out[ix] = vals[j] - np.median(np.delete(vals, j))
    return out

def scale_causal(r, is_hol):
    """Expanding MAD of past residuals: non-holiday if >= 24 exist, else all past; NaN before 24 past residuals."""
    r, h = r.values, is_hol.values
    out = np.full(len(r), np.nan)
    for t in range(len(r)):
        past = r[:t]; ok = ~np.isnan(past)
        if ok.sum() < MIN_PAST:
            continue
        nh = past[ok & (h[:t] == 0)]
        v = nh if len(nh) >= MIN_PAST else past[ok]
        out[t] = 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9
    return pd.Series(out, index=range(len(r)))

def scale_static(r, is_hol):
    v = r[(is_hol == 0) & r.notna()]
    v = v if len(v) >= MIN_PAST else r.dropna()
    return 1.4826 * np.median(np.abs(v - np.median(v))) + 1e-9

def gbm_features(g, x, calendar=False):
    F = pd.DataFrame(index=g.index)
    for k in (1, 2, 3, 24):
        F[f"lag{k}"] = x.shift(k)
    p = x.shift(1)
    for w in (3, 6, 24):
        F[f"rm{w}"] = p.rolling(w, min_periods=w).mean(); F[f"rs{w}"] = p.rolling(w, min_periods=w).std()
    F["hsin"], F["hcos"] = np.sin(2 * np.pi * g.hour / 24), np.cos(2 * np.pi * g.hour / 24)
    F["wkend"] = g.is_weekend
    if calendar:
        F["is_holiday"] = g.is_holiday; F["in_window"] = g.in_festival_window; F["offset"] = g.offset
        F["fg_diwali"] = g.ctx.str.startswith("diwali").astype(int)
    return F

def zscores(g, x, model_blind=None, model_cal=None):
    """All per-state residual z-scores for a (possibly injected) clr series x."""
    out = {}
    ra = resid_causal(x, g.hour); sa = scale_causal(ra, g.is_holiday)
    out["ra"], out["za"] = ra, ra / sa
    rb = resid_lodo(x, g.hour); out["rb"], out["zb"] = rb, rb / scale_static(rb, g.is_holiday)
    for key, m, cal in (("c", model_blind, False), ("k", model_cal, True)):
        if m is None:
            continue
        F = gbm_features(g, x, cal); ok = F.drop(columns=[c for c in ["offset"] if c in F]).notna().all(axis=1)
        pred = pd.Series(np.nan, index=g.index); pred[ok] = m.predict(F[ok])
        r = x - pred; out[f"r{key}"], out[f"z{key}"] = r, r / scale_causal(r, g.is_holiday)
    return out

# common scoring window: hours where the causal z (a) and the GBM z (c) are both defined, identical for all states
probe = zscores(S["AP"], S["AP"].clr, *(2 * [None]))
SCORE_MASK = probe["za"].notna().values.copy()
F0 = gbm_features(S["AP"], S["AP"].clr); ok0 = F0.notna().all(axis=1).values
r_ok = np.zeros(T, bool); r_ok[ok0] = True
cnt = np.cumsum(np.r_[0, r_ok[:-1]])           # past GBM residuals available before t
SCORE_MASK &= (cnt >= MIN_PAST)
SIDX = np.where(SCORE_MASK)[0]
log(f"Scoring window: {S['AP'].ist[SIDX[0]]} to {S['AP'].ist[SIDX[-1]]} IST ({len(SIDX)} h per state); earlier hours are warm-up")

# %% [markdown]
# ## 3. GBM forecasters (cross-state, cached by excluded set)
# LightGBM predicts clr from past-only features. A model is trained on the states **not** in an excluded set: {i} for the
# held-out fold, and {i, j} for the nested calibration of training state j (171 pairs + 19 singles, per feature set).

# %%
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=20, subsample=1.0,
                  colsample_bytree=1.0, random_state=42, deterministic=True, force_row_wise=True, verbose=-1)
TRAIN_ROWS = {}
for cal in (False, True):
    for s in STATES:
        F = gbm_features(S[s], S[s].clr, cal); ok = F.drop(columns=[c for c in ["offset"] if c in F]).notna().all(axis=1)
        TRAIN_ROWS[(s, cal)] = (F[ok], S[s].clr[ok])
GBM_CACHE, GBM_TIMES = {}, []
def gbm(excluded, cal):
    key = (frozenset(excluded), cal)
    if key not in GBM_CACHE:
        Xs = [TRAIN_ROWS[(s, cal)][0] for s in STATES if s not in excluded]
        ys = [TRAIN_ROWS[(s, cal)][1] for s in STATES if s not in excluded]
        t0 = time.perf_counter()
        GBM_CACHE[key] = lgb.LGBMRegressor(**LGB_PARAMS).fit(pd.concat(Xs), pd.concat(ys))
        GBM_TIMES.append(time.perf_counter() - t0)
    return GBM_CACHE[key]

# clean z for every state under every exclusion pattern we need
t0 = time.perf_counter()
CLEAN = {}            # (state, excluded frozenset) -> z dict (a/b do not depend on exclusion)
for s in STATES:
    base = zscores(S[s], S[s].clr)
    CLEAN[(s, None)] = base
for i in STATES:
    for j in STATES:
        excl = frozenset([i, j]) if i != j else frozenset([i])
        if (j, excl) in CLEAN:
            continue
        z = zscores(S[j], S[j].clr, gbm(excl, False), gbm(excl, True))
        CLEAN[(j, excl)] = {k: z[k] for k in ("rc", "zc", "rk", "zk")}
log(f"Trained {len(GBM_CACHE)} LightGBM forecasters ({np.mean(GBM_TIMES):.2f} s mean fit) in {time.perf_counter() - t0:.0f} s")

def clean_z(s, variant, excl):
    if variant in ("za", "zb", "ra", "rb"):
        return CLEAN[(s, None)][variant]
    return CLEAN[(s, frozenset(excl))][variant]

# %% [markdown]
# ## 4. FACAD effect tables
# ```
# Algorithm FACAD (per held-out state i)
#   Input: z-scores of training states (baseline a, b or c), their calendars, held-out calendar, λ = 5, kernel −3…+2,
#          hour blocks 00–06 / 06–12 / 12–18 / 18–24, α = 0.05
#   1. For every context key c = (festival@k, hour block) seen in training states:
#        n_c = number of state-days with c;  w_c = n_c / (n_c + λ)
#        E_c = w_c · median(z | c)
#        V_c = max(1, 1 + w_c · (MAD(z | c) / MAD(z | normal hours) − 1))        # spread relative to normal hours
#      Fall back to (festival@k) without block, then to E = 0, V = 1.
#   2. FACAD-H: same tables within the held-out state's REGION (South/West/North/East); combine
#        E = w_r · E_region + (1 − w_r) · E_national,  w_r = n_r / (n_r + λ)  (V likewise)
#   3. Score each hour: z* = (z − E_c(t)) / V_c(t);  score = |z*|
#   4. Thresholds: matched budget (FPR = 5 % on the held-out state), global calibrated (95th pct of nested-calibration
#      scores from training states) and Mondrian conformal (per bucket: holiday / window / normal, level ⌈(n+1)(1−α)⌉/n).
# ```

# %%
def mad(v):
    v = np.asarray(v); v = v[~np.isnan(v)]
    return 1.4826 * np.median(np.abs(v - np.median(v))) if len(v) else np.nan

TAU = 0.5
COH_LOG = []
def learn_table(states, variant, excl, ctx_col="ctx", block=True, lam=LAMBDA, gate=False):
    rows = []
    for s in states:
        # GBM residuals of a training state come from a model that excluded that state too (out-of-sample)
        ex = [excl[0], s] if (variant in ("zc", "zk") and len(excl)) else excl
        z = clean_z(s, variant, ex).values; g = S[s]
        ok = ~np.isnan(z)
        rows.append(pd.DataFrame({"state": s, "date": g.date.values[ok], "ctx": g[ctx_col].values[ok],
                                  "block": g.block.values[ok], "z": z[ok]}))
    R = pd.concat(rows, ignore_index=True)
    ref = mad(R.z[R.ctx == ""]) or 1.0
    tab = {}
    for keys, by in ((("ctx", "block"), ["ctx", "block"]), (("ctx",), ["ctx"])):
        if keys == ("ctx", "block") and not block:
            continue
        for k, grp in R[R.ctx != ""].groupby(by):
            n = grp[["state", "date"]].drop_duplicates().shape[0]
            w = n / (n + lam) if lam > 0 else 1.0
            per = grp.groupby("state").z.median(); ns = grp.groupby("state").date.nunique()
            C = 1.0 if len(per) < 2 else float(abs((np.sign(per) * ns).sum()) / ns.sum())
            kk = k if isinstance(k, tuple) else (k,)
            if gate and len(states) >= 17:
                COH_LOG.append({"excl": "|".join(sorted(excl)), "key": "|".join(map(str, kk)), "C": C, "n_states": len(per)})
            E = w * np.median(grp.z) if (not gate or C >= TAU) else 0.0
            tab[kk] = (E, max(1.0, 1 + w * (mad(grp.z) / ref - 1)), n)
    return tab

def lookup(tab, ctx, block):
    if ctx == "":
        return 0.0, 1.0, 0
    return tab.get((ctx, block)) or tab.get((ctx,)) or (0.0, 1.0, 0)

def apply_table(z, g, tab, ctx_col="ctx", mode="full", tab_reg=None, lam=LAMBDA):
    E = np.zeros(len(z)); V = np.ones(len(z))
    for t in range(len(z)):
        e, v, n = lookup(tab, g[ctx_col].iat[t], g.block.iat[t])
        if tab_reg is not None:
            er, vr, nr = lookup(tab_reg, g[ctx_col].iat[t], g.block.iat[t])
            if nr:
                wr = nr / (nr + lam)
                e, v = wr * er + (1 - wr) * e, wr * vr + (1 - wr) * v
        E[t], V[t] = e, v
    if mode == "shift":
        V[:] = 1
    elif mode == "spread":
        E[:] = 0
    elif mode == "none":
        E[:] = 0; V[:] = 1
    return (np.asarray(z) - E) / V

# mobility prior (E-B): national E cancels in CLR, so only the spread V transfers; mean V over the 6 mobility signals
MOB = pd.read_csv("data/mobility_prior_diwali.csv")
VPRIOR = MOB.groupby("offset").V.mean().to_dict()
print("mobility V prior by offset:", {k: round(v, 3) for k, v in VPRIOR.items()})
def apply_prior(z, g):
    V = np.array([VPRIOR.get(int(o), 1.0) if g.ctx.iat[t].startswith("diwali") else 1.0
                  for t, o in enumerate(g.offset.fillna(99))])
    return np.asarray(z) / V

# %% [markdown]
# ## 5. Anomaly injection

# %%
TYPES = ["spike", "level_shift", "flatten"]
def inject(g, seed, state):
    """8 non-overlapping events in the scoring window: 3 inside festival windows, 5 outside. Returns new clr, labels, meta."""
    rng = np.random.default_rng(1000 * seed + STATES.index(state))
    x = g.clr.values.copy()
    sig = mad(CLEAN[(state, None)]["rb"].values)
    win = g.in_festival_window.values.astype(bool)
    lab = np.zeros(T, int); typ = np.array([""] * T, dtype=object); used = np.zeros(T, bool)
    events = []
    plan = [True] * N_IN_WINDOW + [False] * (N_EVENTS - N_IN_WINDOW)
    daymean = g.groupby("date").clr.transform("mean").values
    for want in plan:
      placed = False
      for inside in (want, not want):          # fall back to the other zone if the requested one is full
        if placed:
            break
        for _ in range(500):
            kind = TYPES[rng.integers(3)]
            L = 1 if kind == "spike" else (int(rng.integers(3, 7)) if kind == "level_shift" else int(rng.integers(4, 7)))
            st = int(rng.choice(SIDX))
            hrs = np.arange(st, st + L)
            if hrs[-1] >= T or not SCORE_MASK[hrs].all() or (win[hrs] != inside).any():
                continue
            lo, hi = max(0, st - 2), min(T, st + L + 2)
            if used[lo:hi].any():
                continue
            if kind == "spike":
                delta = np.full(L, rng.choice([-1, 1]) * rng.uniform(3, 6) * sig / 0.95)
            elif kind == "level_shift":
                delta = np.full(L, rng.choice([-1, 1]) * rng.uniform(2, 4) * sig / 0.95)
            else:
                delta = (daymean[hrs] - x[hrs]) / 0.95
                if np.mean(np.abs(delta * 0.95)) < 1.5 * sig:
                    continue
            x[hrs] += 0.95 * delta
            lab[hrs] = 1; typ[hrs] = kind; used[hrs] = True
            events.append({"start": st, "len": L, "type": kind, "inside_window": inside, "requested_inside": want})
            placed = True
            break
    return pd.Series(x), lab, typ, events
log("Injection: clr += 0.95·δ (log-share shift δ re-normalised to sum 100); σ = MAD of the state's transductive residual; "
    "8 events per state per seed (3 inside festival windows), non-overlapping with a 2-h gap")

# %% [markdown]
# ## 6. Methods, thresholds and metrics

# %%
def conformal_q(scores, alpha=ALPHA):
    s = np.sort(np.asarray(scores)[~np.isnan(scores)]); n = len(s)
    if n == 0:
        return np.inf
    k = math.ceil((n + 1) * (1 - alpha))
    return s[min(k, n) - 1] if k <= n else np.inf

def metrics(score, lab, typ, g, thr, suppress=None):
    m = SCORE_MASK
    sc, lb, ty = score[m], lab[m], typ[m]
    b = g.bucket.values[m]; win = g.in_festival_window.values[m] == 1
    thr = thr if np.ndim(thr) else np.full(len(sc), thr)
    pred = sc >= (thr[m] if len(thr) == T else thr)
    if suppress is not None:
        pred &= ~suppress[m]
    neg = lb == 0
    tp, fp, fn = (pred & ~neg).sum(), (pred & neg).sum(), (~pred & ~neg).sum()
    P = tp / (tp + fp) if tp + fp else 0.0; Rr = tp / (tp + fn) if tp + fn else 0.0
    out = {"FPR_holiday": pred[neg & (b == "holiday")].mean() if (neg & (b == "holiday")).any() else np.nan,
           "FPR_window": pred[neg & (b == "window")].mean() if (neg & (b == "window")).any() else np.nan,
           "FPR_normal": pred[neg & (b == "normal")].mean() if (neg & (b == "normal")).any() else np.nan,
           "FPR_overall": pred[neg].mean(), "precision": P, "recall": Rr, "F1": 2 * P * Rr / (P + Rr) if P + Rr else 0.0,
           "PR_AUC": average_precision_score(lb, np.nan_to_num(sc, nan=-1e9)) if lb.any() else np.nan,
           "recall_in_window": pred[(lb == 1) & win].mean() if ((lb == 1) & win).any() else np.nan,
           "recall_outside": pred[(lb == 1) & ~win].mean() if ((lb == 1) & ~win).any() else np.nan}
    for t_ in TYPES:
        sel = (lb == 1) & (ty == t_)
        out[f"recall_{t_}"] = pred[sel].mean() if sel.any() else np.nan
    return out

def matched_thr(score, lab, suppress=None):
    sc = score.copy()
    if suppress is not None:
        sc = np.where(suppress, -np.inf, sc)
    neg = SCORE_MASK & (lab == 0)
    return np.quantile(sc[neg], 1 - MATCH_FPR, method="higher")

def mondrian_thr(cal_scores, cal_buckets, g):
    q = {b: conformal_q(cal_scores[cal_buckets == b]) for b in ("holiday", "window", "normal")}
    return np.array([q[b] for b in g.bucket.values]), q

# configs of table-based methods: name -> (z variant, ctx column, use blocks, λ, mode, hierarchical)
TABLE_METHODS = {
    "FACAD-a": ("za", "ctx", True, LAMBDA, "full", False),
    "FACAD-H-a": ("za", "ctx", True, LAMBDA, "full", True),
    "FACAD-b": ("zb", "ctx", True, LAMBDA, "full", False),
    "FACAD-c": ("zc", "ctx", True, LAMBDA, "full", False),
    "FACAD-CG": ("za", "ctx", True, LAMBDA, "full", False, True),
    "FACAD-H-CG": ("za", "ctx", True, LAMBDA, "full", True, True),
    # ablations on baseline a
    "abl: shift only": ("za", "ctx", True, LAMBDA, "shift", False),
    "abl: spread only": ("za", "ctx", True, LAMBDA, "spread", False),
    "abl: no kernel (festival day only)": ("za", "ctx_day", True, LAMBDA, "full", False),
    "abl: national calendar": ("za", "ctx_nat", True, LAMBDA, "full", False),
    "abl: no shrinkage (λ=0)": ("za", "ctx", True, 0.0, "full", False),
    "abl: no hour blocks": ("za", "ctx", False, LAMBDA, "full", False),
}
MAIN = ["FACAD-CG", "FACAD-H-CG", "FACAD-a", "FACAD-H-a", "SeasonalZ-a", "FACAD-b", "SeasonalZ-b", "FACAD-c", "GBM-c (blind)",
        "GBM-cal (calendar inputs)", "IF", "IF+cal", "Prophet-hol", "Whitelist-a", "FACAD-prior (mobility)"]
ABL = ["FACAD-a", "SeasonalZ-a"] + [k for k in TABLE_METHODS if k.startswith("abl:")] + ["FACAD-H-a"]

def table_score(name, z, g, i, excl_states):
    """i = the state being scored (its region is used by FACAD-H); excl_states = states left out of learning."""
    v, ctxc, blk, lam, mode, hier = TABLE_METHODS[name][:6]
    gate = len(TABLE_METHODS[name]) > 6 and TABLE_METHODS[name][6]
    train = [s for s in STATES if s not in excl_states]
    key = (name, frozenset(excl_states), REGION[i] if hier else None)
    if key not in TAB_CACHE:
        tab = learn_table(train, v, excl_states, ctxc, blk, lam, gate)
        treg = learn_table([s for s in train if REGION[s] == REGION[i]], v, excl_states, ctxc, blk, lam, gate) if hier else None
        TAB_CACHE[key] = (tab, treg)
    tab, treg = TAB_CACHE[key]
    return np.abs(apply_table(z, g, tab, ctxc, mode, treg, lam))
TAB_CACHE = {}

IF_FEATS = ["ra", "za", "hsin", "hcos"]
def if_frame(g, z, cal):
    F = pd.DataFrame({"ra": z["ra"].values, "za": z["za"].values,
                      "hsin": np.sin(2 * np.pi * g.hour / 24), "hcos": np.cos(2 * np.pi * g.hour / 24)})
    if cal:
        F["is_holiday"] = g.is_holiday.values; F["in_window"] = g.in_festival_window.values; F["offset"] = g.offset.fillna(9).values
    return F

PROPHET_CACHE = {}
def prophet_score(g, x, key):
    if key in PROPHET_CACHE:
        return PROPHET_CACHE[key]
    hol = [{"holiday": n, "ds": d, "lower_window": -3, "upper_window": 2}
           for d, n in g[g.is_holiday == 1].groupby("date").holiday_name.first().items()]
    H = pd.DataFrame(hol) if hol else None
    if H is not None:
        H["ds"] = pd.to_datetime(H.ds)
    m = prophet.Prophet(daily_seasonality=True, weekly_seasonality=False, yearly_seasonality=False, holidays=H,
                        uncertainty_samples=0)
    m.fit(pd.DataFrame({"ds": g.ist, "y": x.values}))
    r = x.values - m.predict(pd.DataFrame({"ds": g.ist})).yhat.values
    sc = np.abs(r / scale_static(pd.Series(r), g.is_holiday))
    PROPHET_CACHE[key] = sc
    return sc

# %% [markdown]
# ## 7. E-A / E-B / E-C — leave-one-state-out with nested calibration

# %%
def all_scores(i, g, z, excl, if_models):
    """Score dict for every method on one (possibly injected) series of state i; excl = training exclusion set."""
    sc = {}
    for name in TABLE_METHODS:
        v = TABLE_METHODS[name][0]
        sc[name] = table_score(name, z[v].values, g, i, excl)
    sc["SeasonalZ-a"] = np.abs(z["za"].values); sc["SeasonalZ-b"] = np.abs(z["zb"].values)
    sc["GBM-c (blind)"] = np.abs(z["zc"].values); sc["GBM-cal (calendar inputs)"] = np.abs(z["zk"].values)
    sc["Whitelist-a"] = np.abs(z["za"].values)
    sc["FACAD-prior (mobility)"] = np.abs(apply_prior(z["za"].values, g))
    for nm, cal in (("IF", False), ("IF+cal", True)):
        F = if_frame(g, z, cal); ok = F.notna().all(axis=1).values
        s_ = np.full(T, np.nan); s_[ok] = -if_models[cal].score_samples(F[ok]); sc[nm] = s_
    return sc

RESULTS, CASE, CAL_Q, RUNTIME, EVENTS = [], {}, [], [], []
t_all = time.perf_counter()
for i in STATES:
    t_fold = time.perf_counter()
    g = S[i]; train = [s for s in STATES if s != i]
    # Isolation Forest trained on training states (clean, scoring window)
    if_models = {}
    for cal in (False, True):
        Xtr = pd.concat([if_frame(S[s], {**CLEAN[(s, None)]}, cal)[SCORE_MASK] for s in train]).dropna()
        if_models[cal] = IsolationForest(n_estimators=300, random_state=42).fit(Xtr)
    # nested calibration: clean scores of each training state j with everything learned without {i, j}
    cal_sc, cal_b = {m: [] for m in MAIN}, []
    for j in train:
        zj = {**CLEAN[(j, None)], **CLEAN[(j, frozenset([i, j]))]}
        sj = all_scores(j, S[j], zj, [i, j], if_models)
        sj["Prophet-hol"] = prophet_score(S[j], S[j].clr, ("clean", j))
        for m in MAIN:
            cal_sc[m].append(sj[m][SCORE_MASK])
        cal_b.append(S[j].bucket.values[SCORE_MASK])
    cal_b = np.concatenate(cal_b)
    for m in MAIN:
        cal_sc[m] = np.concatenate(cal_sc[m])
    # whitelist never alarms in festival windows: calibrate on normal hours only
    gthr = {m: conformal_q(cal_sc[m][cal_b == "normal"] if m == "Whitelist-a" else cal_sc[m]) for m in MAIN}
    for m in MAIN:
        _, q = mondrian_thr(cal_sc[m], cal_b, g)
        CAL_Q.append({"state": i, "method": m, **{f"q_{k}": v for k, v in q.items()}, "q_global": gthr[m]})
    t_cal = time.perf_counter() - t_fold
    for seed in SEEDS:
        x, lab, typ, ev = inject(g, seed, i)
        for e in ev:
            EVENTS.append({"state": i, "seed": seed, **e})
        t_sc = time.perf_counter()
        z = zscores(g, x, gbm([i], False), gbm([i], True))
        sc = all_scores(i, g, z, [i], if_models)
        t_sc = time.perf_counter() - t_sc
        sc["Prophet-hol"] = prophet_score(g, x, ("inj", i, seed))
        supp = g.in_festival_window.values == 1
        for m in sc:
            s_ = np.nan_to_num(sc[m], nan=0.0)
            sup = supp if m == "Whitelist-a" else None
            RESULTS.append({"state": i, "seed": seed, "method": m, "mode": "matched",
                            **metrics(s_, lab, typ, g, matched_thr(s_, lab, sup), sup)})
            if m in MAIN:
                RESULTS.append({"state": i, "seed": seed, "method": m, "mode": "global",
                                **metrics(s_, lab, typ, g, gthr[m], sup)})
                mth, _ = mondrian_thr(cal_sc[m], cal_b, g)
                if m == "Whitelist-a":
                    mth = np.full(T, gthr[m])
                RESULTS.append({"state": i, "seed": seed, "method": m, "mode": "mondrian",
                                **metrics(s_, lab, typ, g, mth, sup)})
        if seed == 0:
            CASE[i] = {"x": x, "lab": lab, "typ": typ, "scores": sc, "cal_q": {m: mondrian_thr(cal_sc[m], cal_b, g)[1] for m in MAIN}}
    RUNTIME.append({"state": i, "fold_seconds": time.perf_counter() - t_fold, "calibration_seconds": t_cal,
                    "score_seconds_per_seed_all_methods": t_sc})
    print(f"{i}: fold done in {time.perf_counter() - t_fold:.1f} s")
R = pd.DataFrame(RESULTS)
R.to_csv("results/all_runs.csv", index=False)
pd.DataFrame(EVENTS).to_csv("results/injected_events.csv", index=False)
log(f"E-A/E-B/E-C finished: {len(R)} result rows, {time.perf_counter() - t_all:.0f} s")
ev = pd.DataFrame(EVENTS)
print("injected events per state-seed:", ev.groupby(["state", "seed"]).size().describe()[["min", "max"]].to_dict(),
      "| types:", ev.type.value_counts().to_dict(), "| inside windows:", int(ev.inside_window.sum()))

# %% [markdown]
# ## 8. Results
# Each metric is averaged over the 10 seeds per state, then summarised over the 19 held-out states (mean ± SD across states).
# The SD over seeds of the 19-state mean is also reported.

# %%
MET = ["FPR_holiday", "FPR_window", "FPR_normal", "FPR_overall", "precision", "recall", "F1", "PR_AUC",
       "recall_in_window", "recall_outside", "recall_spike", "recall_level_shift", "recall_flatten"]
def per_state(mode):
    return R[R["mode"] == mode].groupby(["method", "state"])[MET].mean()
def summary(mode, methods):
    ps = per_state(mode)
    m_, s_ = ps.groupby("method").mean(), ps.groupby("method").std()
    seed_sd = R[R["mode"] == mode].groupby(["method", "seed"])[MET].mean().groupby("method").std()
    out = m_.round(3).astype(str) + " ± " + s_.round(3).astype(str)
    return out.loc[[x for x in methods if x in out.index]], m_.loc[[x for x in methods if x in m_.index]], seed_sd

def md(df_):
    return df_.to_markdown()

T1, T1m, T1sd = summary("matched", MAIN)
T1.to_csv("results/T1_main_matched.csv"); open("results/T1_main_matched.md", "w").write(md(T1))
T1[["FPR_holiday", "FPR_window", "FPR_normal", "precision", "recall", "F1", "PR_AUC", "recall_in_window", "recall_outside"]]

# %%
rows = []
for mode in ("global", "mondrian"):
    _, mm, _ = summary(mode, MAIN)
    for m in MAIN:
        if m in mm.index:
            rows.append({"method": m, "threshold": mode, **mm.loc[m, ["FPR_holiday", "FPR_window", "FPR_normal", "FPR_overall",
                                                                      "recall", "F1", "recall_in_window"]].round(3).to_dict()})
T2 = pd.DataFrame(rows).set_index(["method", "threshold"])
T2.to_csv("results/T2_calibrated_thresholds.csv"); open("results/T2_calibrated_thresholds.md", "w").write(md(T2))
T2

# %% [markdown]
# ### Paired Wilcoxon tests over the 19 held-out states (seed-averaged), with bootstrap CI of the median difference

# %%
rng_b = np.random.default_rng(42)
def boot_ci(d, B=2000):
    d = np.asarray(d); meds = [np.median(rng_b.choice(d, len(d))) for _ in range(B)]
    return np.percentile(meds, [2.5, 97.5])
ps_m = per_state("matched")
pairs = [("FACAD-a", m) for m in ["SeasonalZ-a", "GBM-c (blind)", "GBM-cal (calendar inputs)", "IF", "IF+cal",
                                  "Prophet-hol", "Whitelist-a", "FACAD-prior (mobility)"]] + \
        [("FACAD-H-a", "FACAD-a"), ("FACAD-H-a", "SeasonalZ-a"), ("FACAD-prior (mobility)", "SeasonalZ-a"),
         ("FACAD-prior (mobility)", "Whitelist-a"), ("FACAD-b", "SeasonalZ-b"), ("FACAD-c", "GBM-c (blind)"), ("FACAD-c", "GBM-cal (calendar inputs)")]
rows = []
for a, b in pairs:
    for k in ("FPR_holiday", "PR_AUC", "F1", "recall_in_window"):
        d = (ps_m.loc[a][k] - ps_m.loc[b][k]).dropna()
        p = wilcoxon(d).pvalue if (d != 0).any() else 1.0
        lo, hi = boot_ci(d)
        rows.append({"method": a, "vs": b, "metric": k, "median_diff": round(np.median(d), 3), "ci95_low": round(lo, 3),
                     "ci95_high": round(hi, 3), "p_value": float(f"{p:.3g}"), "n_states": len(d)})
T3 = pd.DataFrame(rows)
T3.to_csv("results/T3_wilcoxon.csv", index=False); open("results/T3_wilcoxon.md", "w").write(T3.to_markdown(index=False))
T3.pivot_table(index=["method", "vs"], columns="metric", values=["median_diff", "p_value"], sort=False)

# %%
T4 = ps_m.loc[["SeasonalZ-a", "FACAD-a", "FACAD-H-a"]][["FPR_holiday", "PR_AUC", "F1", "recall_in_window"]].unstack(0)
T4.columns = [f"{m} | {k}" for k, m in T4.columns]
T4 = T4[[c for m in ["SeasonalZ-a", "FACAD-a", "FACAD-H-a"] for c in T4.columns if c.startswith(m + " |")]].round(3)
T4.insert(0, "region", T4.index.map(REGION))
T4.to_csv("results/T4_per_state.csv"); open("results/T4_per_state.md", "w").write(T4.to_markdown())
T4

# %%
T5, T5m, _ = summary("matched", ABL)
T5 = T5[["FPR_holiday", "FPR_window", "FPR_normal", "F1", "PR_AUC", "recall_in_window"]]
T5.to_csv("results/T5_ablation.csv"); open("results/T5_ablation.md", "w").write(md(T5))
T5

# %%
T6 = T1m[["recall_spike", "recall_level_shift", "recall_flatten", "recall_in_window", "recall_outside"]].round(3)
T6.to_csv("results/T6_recall_by_type.csv"); open("results/T6_recall_by_type.md", "w").write(md(T6))
T6

# %% [markdown]
# ### Runtime and memory

# %%
RT = pd.DataFrame(RUNTIME)
tracemalloc.start(); t0 = time.perf_counter()
_z = zscores(S["TN"], S["TN"].clr); _tab = learn_table([s for s in STATES if s != "TN"], "za", ["TN"])
_sc = apply_table(_z["za"].values, S["TN"], _tab)
t_facad = time.perf_counter() - t0; peak = tracemalloc.get_traced_memory()[1] / 2**20; tracemalloc.stop()
T7 = pd.DataFrame([
    {"item": "FACAD-a: learn table from 18 states + score one state (167 h)", "value": round(t_facad, 3), "unit": "s"},
    {"item": "FACAD-a peak memory for the step above", "value": round(peak, 1), "unit": "MiB"},
    {"item": "LightGBM forecaster fit (mean over cached models)", "value": round(float(np.mean(GBM_TIMES)), 3), "unit": "s"},
    {"item": "All non-Prophet methods: score one injected series", "value": round(RT.score_seconds_per_seed_all_methods.mean(), 3), "unit": "s"},
    {"item": "Full LOSO fold incl. nested calibration, 10 seeds, Prophet", "value": round(RT.fold_seconds.mean(), 1), "unit": "s"},
])
T7.to_csv("results/T7_runtime.csv", index=False); open("results/T7_runtime.md", "w").write(T7.to_markdown(index=False))
T7

# %% [markdown]
# ### E-B — mobility transfer: does a state's mobility response to Dussehra predict its traffic-share response?
# Descriptive check across states: the state-relative mobility deviation on Dussehra days (2020–21, Google mobility) against the
# state's own median clr z on its Dussehra festival-day hours in this week (no injection).

# %%
mob_state = pd.read_csv("mobility_state_diwali_effect.csv", index_col=0)
mob_state = mob_state[~mob_state.index.duplicated()]
own = pd.Series({s: np.nanmedian(CLEAN[(s, None)]["za"].values[(S[s].ctx.str.endswith("@+0")) & SCORE_MASK]) for s in STATES})
rows = []
for sig_ in ["workplaces", "residential", "retail and recreation", "transit stations"]:
    a_ = mob_state.loc[own.index, sig_]
    r_ = scipy.stats.spearmanr(a_, own)
    rows.append({"mobility signal": sig_, "spearman_rho": round(r_.statistic, 3), "p_value": float(f"{r_.pvalue:.3g}"), "n_states": len(own)})
EB = pd.DataFrame(rows); EB.to_csv("results/EB_mobility_vs_traffic.csv", index=False)
EB

# %% [markdown]
# ### E-D — real-event check on clean data (no injection)

# %%
top = []
for s in STATES:
    g = S[s]; z = CLEAN[(s, None)]["za"].values
    tab = learn_table([x for x in STATES if x != s], "za", [s])
    fa = np.abs(apply_table(z, g, tab))
    for t in SIDX:
        top.append({"state": s, "ist": g.ist.iat[t], "holiday": g.holiday_name.iat[t] or "", "bucket": g.bucket.iat[t],
                    "score_SeasonalZ": abs(z[t]), "score_FACAD": fa[t], "clr_resid": CLEAN[(s, None)]["ra"].iat[t]})
TOP = pd.DataFrame(top)
tz_ = TOP.sort_values("score_SeasonalZ", ascending=False).head(20)
tf_ = TOP.sort_values("score_FACAD", ascending=False).head(20)
print("SeasonalZ top 20 — holiday hours:", int((tz_.bucket == "holiday").sum()), "| window:", int((tz_.bucket == "window").sum()))
print("FACAD     top 20 — holiday hours:", int((tf_.bucket == "holiday").sum()), "| window:", int((tf_.bucket == "window").sum()))
ED = pd.concat([tz_.assign(ranked_by="SeasonalZ-a"), tf_.assign(ranked_by="FACAD-a")])
ED.round(3).to_csv("results/ED_top20_real_hours.csv", index=False)
# hours where several states are flagged together (cross-state events) by FACAD at its median matched threshold
thr_f = np.quantile(TOP.score_FACAD, 0.95)
multi = TOP[TOP.score_FACAD >= thr_f].groupby("ist").state.apply(lambda v: ", ".join(sorted(v)))
print("hours with >= 3 states above FACAD's 95th percentile:")
print(multi[multi.str.count(",") >= 2].to_string())
tf_[["state", "ist", "holiday", "bucket", "score_FACAD", "score_SeasonalZ", "clr_resid"]].round(3)

# %% [markdown]
# ## 9. Figures

# %%
# Fig 1 pipeline
fig, ax = plt.subplots(figsize=(10, 2.6)); ax.axis("off"); ax.grid(False)
boxes = [("Hourly share\n(19 states + OTHER)", 0.02), ("CLR transform\nper hour", 0.2), ("Baseline\n(a) causal / (b) LODO / (c) LightGBM", 0.38),
         ("Residual z\n(causal robust scale)", 0.58), ("Festival effect table\nE, V from other states\n(+ region: FACAD-H)", 0.76)]
for txt, x0 in boxes:
    ax.add_patch(FancyBboxPatch((x0, 0.45), 0.16, 0.42, boxstyle="round,pad=0.01,rounding_size=0.02", fc="white", ec="#8a8984"))
    ax.text(x0 + 0.08, 0.66, txt, ha="center", va="center", fontsize=8)
for x0 in (0.18, 0.36, 0.56, 0.74):
    ax.annotate("", xy=(x0 + 0.02, 0.66), xytext=(x0, 0.66), arrowprops=dict(arrowstyle="->", color="#8a8984"))
ax.add_patch(FancyBboxPatch((0.58, 0.02), 0.34, 0.3, boxstyle="round,pad=0.01,rounding_size=0.02", fc="#2a78d6", alpha=0.12, ec="#2a78d6"))
ax.text(0.75, 0.17, "score = |(z − E) / V|  →  Mondrian conformal threshold\nper bucket: holiday / festival window / normal (α = 0.05)", ha="center", va="center", fontsize=8)
ax.annotate("", xy=(0.75, 0.33), xytext=(0.84, 0.45), arrowprops=dict(arrowstyle="->", color="#8a8984"))
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("FACAD pipeline on Cloudflare traffic shares")
plt.savefig("figures/fig1_pipeline.png", bbox_inches="tight"); plt.close('all')

# %%
# Fig 2 festival-hour FPR at matched budget (SD over seeds of the 19-state mean)
FIGM = ["FACAD-a", "FACAD-H-a", "FACAD-c", "FACAD-b", "FACAD-prior (mobility)", "SeasonalZ-a", "SeasonalZ-b",
        "GBM-c (blind)", "GBM-cal (calendar inputs)", "IF", "IF+cal", "Prophet-hol", "Whitelist-a"]
col = {m: (CAT[0] if m.startswith("FACAD") else GREY) for m in FIGM}
v = T1m.loc[FIGM, "FPR_holiday"]; sd = T1sd.loc[FIGM, "FPR_holiday"]
fig, ax = plt.subplots(figsize=(7.2, 4.8))
y = np.arange(len(FIGM))
ax.barh(y, v, xerr=sd, color=[col[m] for m in FIGM], height=0.62, error_kw=dict(lw=0.8, ecolor="#52514e"))
for k, (a_, e_) in enumerate(zip(v, sd)):
    ax.text(a_ + e_ + 0.004, k, f"{a_:.3f}", va="center", fontsize=7.5)
ax.axvline(MATCH_FPR, color="#52514e", ls="--", lw=0.8); ax.text(MATCH_FPR, -0.9, " 5% overall budget", fontsize=7, color="#52514e")
ax.set_yticks(y); ax.set_yticklabels(FIGM); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.set_xlabel("false-alarm rate on clean holiday hours (matched 5% overall FPR)")
ax.set_title("Festival-hour false alarms by method (blue = FACAD variants; error bar = SD over 10 seeds)")
plt.tight_layout(); plt.savefig("figures/fig2_holiday_fpr.png", bbox_inches="tight"); plt.close('all')

# %%
# Fig 3 PR-AUC and recall inside/outside windows
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
pa = T1m.loc[FIGM, "PR_AUC"]; psd = T1sd.loc[FIGM, "PR_AUC"]
axes[0].barh(y, pa, xerr=psd, color=[col[m] for m in FIGM], height=0.62, error_kw=dict(lw=0.8, ecolor="#52514e"))
for k, a_ in enumerate(pa):
    axes[0].text(a_ + 0.01, k, f"{a_:.3f}", va="center", fontsize=7.5)
axes[0].set_xlabel("PR-AUC (threshold-free)"); axes[0].set_title("Detection quality")
axes[1].barh(y - 0.2, T1m.loc[FIGM, "recall_outside"], height=0.38, color=CAT[0], label="outside festival windows")
axes[1].barh(y + 0.2, T1m.loc[FIGM, "recall_in_window"], height=0.38, color=CAT[1], label="inside festival windows")
axes[1].set_xlabel("recall at matched 5% FPR"); axes[1].legend(frameon=False, loc="lower right"); axes[1].set_title("Recall of injected anomalies")
for ax in axes:
    ax.set_yticks(y); ax.set_yticklabels(FIGM); ax.grid(axis="y", visible=False)
axes[0].invert_yaxis()
plt.tight_layout(); plt.savefig("figures/fig3_prauc_recall.png", bbox_inches="tight"); plt.close('all')

# %%
# Fig 4 per-state paired dots
a_ = ps_m.loc["SeasonalZ-a", "FPR_holiday"]; b_ = ps_m.loc["FACAD-a", "FPR_holiday"]
order = (a_ - b_).sort_values().index
fig, ax = plt.subplots(figsize=(6.8, 5.6)); yy = np.arange(len(order))
for k, s in enumerate(order):
    ax.plot([a_[s], b_[s]], [k, k], color="#cfcec8", lw=1.5, zorder=1)
ax.scatter(a_[order], yy, color=CAT[1], s=34, label="SeasonalZ-a (calendar-blind)", zorder=2)
ax.scatter(b_[order], yy, color=CAT[0], s=34, label="FACAD-a", zorder=3)
ax.set_yticks(yy); ax.set_yticklabels([f"{s} ({REGION[s]})" for s in order]); ax.grid(axis="y", visible=False)
ax.set_xlabel("false-alarm rate on clean holiday hours (matched 5% budget)")
ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2); ax.set_title("Per held-out state: calendar-blind vs FACAD", pad=26)
plt.tight_layout(); plt.savefig("figures/fig4_per_state.png", bbox_inches="tight"); plt.close('all')

# %%
# Fig 5 case studies (seed 0)
fig, axes = plt.subplots(3, 4, figsize=(14, 7.5), sharex="col", gridspec_kw={"height_ratios": [1.2, 1, 1]})
for c_, s in enumerate(["WB", "KA", "TN", "AP"]):
    g = S[s]; cs = CASE[s]; m = SCORE_MASK
    for r_ in range(3):
        ax = axes[r_, c_]
        for d in g.loc[g.is_holiday == 1, "date"].unique():
            ax.axvspan(pd.Timestamp(d), pd.Timestamp(d) + pd.Timedelta(days=1), color=HOL, alpha=0.18, lw=0)
        lab = cs["lab"].astype(bool)
        st_ = np.where(np.diff(np.r_[0, lab.astype(int), 0]) != 0)[0].reshape(-1, 2)
        for a0, a1 in st_:
            ax.axvspan(g.ist.iat[a0] - pd.Timedelta(minutes=30), g.ist.iat[a1 - 1] + pd.Timedelta(minutes=30), color=CAT[6], alpha=0.15, lw=0)
    axes[0, c_].plot(g.ist, g.clr, color=GREY, lw=1, label="clean")
    axes[0, c_].plot(g.ist, cs["x"], color=CAT[0], lw=1, label="with injections")
    axes[0, c_].set_title(f"{s} — clr (seed 0)"); axes[0, c_].legend(frameon=False, fontsize=7)
    for r_, (mname, colr) in enumerate([("SeasonalZ-a", CAT[1]), ("FACAD-a", CAT[0])], start=1):
        ax = axes[r_, c_]; sc_ = np.nan_to_num(cs["scores"][mname])
        thr = matched_thr(sc_, cs["lab"])
        ax.plot(g.ist[m], sc_[m], color=colr, lw=1)
        ax.axhline(thr, color="#52514e", ls="--", lw=0.8)
        fp = m & (sc_ >= thr) & (cs["lab"] == 0); tp = m & (sc_ >= thr) & (cs["lab"] == 1)
        ax.scatter(g.ist[fp], sc_[fp], marker="x", color=CAT[7], s=22, label=f"false alarm ({fp.sum()})", zorder=3)
        ax.scatter(g.ist[tp], sc_[tp], facecolor="none", edgecolor="#0b0b0b", s=22, label=f"detected ({tp.sum()})", zorder=3)
        ax.set_ylabel(f"{mname} score"); ax.legend(frameon=False, fontsize=6.5, loc="upper left")
    for r_ in range(3):
        axes[r_, c_].tick_params(axis="x", rotation=40, labelsize=7)
fig.suptitle("Case studies: orange band = state holiday, violet band = injected anomaly, dashed = matched threshold", x=0.01, ha="left", fontsize=10.5)
plt.tight_layout(); plt.savefig("figures/fig5_case_studies.png", bbox_inches="tight"); plt.close('all')

# %%
# Fig 6 ablation
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
yy = np.arange(len(ABL))
for ax, k, lbl in ((axes[0], "FPR_holiday", "false-alarm rate on holiday hours"), (axes[1], "PR_AUC", "PR-AUC")):
    vals = T5m.loc[ABL, k]
    ax.barh(yy, vals, color=[CAT[0] if a == "FACAD-a" else (CAT[1] if a == "SeasonalZ-a" else GREY) for a in ABL], height=0.62)
    for kk, a_ in enumerate(vals):
        ax.text(a_ + 0.003, kk, f"{a_:.3f}", va="center", fontsize=7.5)
    ax.set_xlabel(lbl + " (matched 5% budget)"); ax.grid(axis="y", visible=False)
axes[0].set_yticks(yy); axes[0].set_yticklabels(ABL); axes[0].invert_yaxis()
axes[0].set_title("Ablations of FACAD-a (blue = full, orange = no calendar)")
plt.tight_layout(); plt.savefig("figures/fig6_ablation.png", bbox_inches="tight"); plt.close('all')

# %%
# Fig 7 conformal coverage per bucket
CM = ["FACAD-a", "FACAD-H-a", "FACAD-c", "SeasonalZ-a", "GBM-c (blind)", "GBM-cal (calendar inputs)"]
bk = ["FPR_holiday", "FPR_window", "FPR_normal"]
fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
for ax, b in zip(axes, bk):
    gl = [T2.loc[(m, "global"), b] for m in CM]; mo = [T2.loc[(m, "mondrian"), b] for m in CM]
    yy = np.arange(len(CM))
    ax.barh(yy - 0.2, gl, height=0.38, color=GREY, label="global threshold")
    ax.barh(yy + 0.2, mo, height=0.38, color=CAT[0], label="Mondrian conformal")
    ax.axvline(ALPHA, color=CAT[7], ls="--", lw=1); ax.set_title(b.replace("FPR_", "achieved FPR — "))
    ax.set_yticks(yy); ax.set_yticklabels(CM); ax.grid(axis="y", visible=False)
axes[0].invert_yaxis()
h_, l_ = axes[0].get_legend_handles_labels()
fig.legend(h_, l_, frameon=False, fontsize=8, loc="upper right", ncol=2)
fig.suptitle("Calibrated thresholds: achieved false-alarm rate per context bucket (target 5%, red dashed)", x=0.01, ha="left", fontsize=10.5)
plt.tight_layout(rect=(0, 0, 1, 0.93)); plt.savefig("figures/fig7_conformal_coverage.png", bbox_inches="tight"); plt.close('all')

# %%
# Fig 8 effect table heatmap: each state's own E (descriptive, its own clean data) + pooled national row
cols = [f"{k:+d}|{b}" for k in range(-3, 3) for b in range(4)]
HM = pd.DataFrame(np.nan, index=STATES + ["National (pooled)"], columns=cols)
tab_all = learn_table(STATES, "za", [], "ctx", True, LAMBDA)
for s in STATES + ["National (pooled)"]:
    tb = tab_all if s.startswith("National") else learn_table([s], "za", [], "ctx", True, 0.0)
    for (key), val in tb.items():
        if len(key) == 2 and key[0].startswith("diwali@"):
            HM.loc[s, f"{int(key[0].split('@')[1]):+d}|{key[1]}"] = val[0]
fig, ax = plt.subplots(figsize=(12, 6.4))
lim = np.nanpercentile(np.abs(HM.values), 98)
im = ax.imshow(HM.values, cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.split("|")[1] for c in cols], fontsize=7)
for k in range(6):
    ax.text(4 * k + 1.5, -1.1, f"day {k - 3:+d}", ha="center", fontsize=8)
    if k:
        ax.axvline(4 * k - 0.5, color="white", lw=2)
ax.set_yticks(range(len(HM))); ax.set_yticklabels(HM.index); ax.grid(False)
ax.axhline(len(STATES) - 0.5, color="#0b0b0b", lw=1)
ax.set_xlabel("hour block within the day (0: 00–06, 1: 06–12, 2: 12–18, 3: 18–24 IST)")
plt.colorbar(im, ax=ax, label="effect E (median standardised residual)", fraction=0.03)
ax.set_title("Dussehra effect by day offset and hour block (blank = before the scoring window or not observed)", pad=18)
plt.tight_layout(); plt.savefig("figures/fig8_effect_table.png", bbox_inches="tight"); plt.close('all')
HM.round(3).to_csv("results/effect_table_heatmap.csv")

# %% [markdown]
# ## 10. Headline numbers

# %%
def g_(m, k, mode="matched"):
    return round(float(summary(mode, [m])[1].loc[m, k]), 4)
summ = {"scoring_window_hours_per_state": int(len(SIDX)), "n_states": len(STATES), "seeds": len(SEEDS),
        "events_injected": int(len(EVENTS)),
        "matched": {m: {k: round(float(T1m.loc[m, k]), 4) for k in MET} for m in MAIN},
        "ablation_matched": {m: {k: round(float(T5m.loc[m, k]), 4) for k in ["FPR_holiday", "FPR_window", "F1", "PR_AUC", "recall_in_window"]} for m in ABL},
        "calibrated": {f"{m}|{t}": T2.loc[(m, t)].to_dict() for m, t in T2.index},
        "wilcoxon": T3.to_dict(orient="records"), "mobility_vs_traffic": EB.to_dict(orient="records"),
        "runtime": T7.to_dict(orient="records")}
json.dump(summ, open("results/summary.json", "w"), indent=1, default=float)
open("run_log.txt", "w").write("\n".join(RUNLOG) + "\n")
print(json.dumps({m: summ["matched"][m] for m in ["FACAD-a", "FACAD-H-a", "SeasonalZ-a", "FACAD-c", "GBM-c (blind)"]}, indent=1)[:3000])

pd.DataFrame(COH_LOG).to_csv("results/coherence_log.csv", index=False)
