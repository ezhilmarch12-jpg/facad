# %% [markdown]
# # FACAD — Experiment 1: Google mobility data, 36 Indian states (2020–22)
#
# Festival-Aware Calendar-Adjusted anomaly Detection. This notebook reproduces Experiment 1 of the article
# (Tables 3 and 6, Figs. 2–3 and the Experiment 1 rows of Table A1).
#
# **Data.** Google COVID-19 Community Mobility Reports for 36 Indian states/UTs, Feb-2020 to Aug-2022, from the public
# GitHub mirror `ActiveConclusion/COVID19_mobility` (file `raw/gmr_countries.csv`; see ../README.md for download and checksum).

# %%
import sys, time, warnings, json, os
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import wilcoxon
sys.path.insert(0, ".")
from facad_lib import *   # calendar encoder, detectors, injection, metrics (documented in the module)

pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)
os.makedirs("figures", exist_ok=True); os.makedirs("results", exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
                     "font.size": 9, "axes.titlesize": 10, "axes.edgecolor": "#8a8984"})
C = {"FACAD": "#2a78d6", "FACAD-GBR": "#6ba6ea", "FACAD-festivals-only": "#9fc4ef", "SeasonalZ": "#eb6834", "Blind-GBR": "#1baf7a", "Cal-GBR": "#6250d6",
     "IF": "#8a8984", "IF+cal": "#b9b8b2", "Whitelist": "#e87ba4"}
N_SEEDS, TARGET_FPR = 10, 0.05
print("numpy", np.__version__, "| pandas", pd.__version__)

# %% [markdown]
# ## 1. Load the public dataset (E1: Indian state-level activity, real data)

# %%
SIGNALS = ["retail and recreation", "grocery and pharmacy", "parks", "transit stations", "workplaces", "residential"]
raw = pd.read_csv("raw/gmr_countries.csv", low_memory=False)
ind = raw[(raw.country == "India") & (raw.region.isin(STATE_CODES))].copy()
ind["date"] = pd.to_datetime(ind.date)
print("rows:", len(ind), "| regions:", ind.region.nunique(), "| span:", ind.date.min().date(), "→", ind.date.max().date())
print("missing share per signal:\n", ind[SIGNALS].isna().mean().round(3).to_string())

# %% [markdown]
# ## 2. Calendar encoder — state-specific lunisolar festivals, proximity kernel, lunar phase
# Festival dates per state come from the open-source `holidays` package (India, with state subdivisions); the
# Hindu-calendar festivals in it (Diwali, Pongal/Sankranti, Holi, Dussehra, Onam, Ganesh Chaturthi, …) move
# across Gregorian years. Lunar phase (tithi) is computed astronomically with `ephem`
# (Moon–Sun ecliptic elongation at 06:00 IST, 12° per tithi).

# %%
dates = pd.date_range(ind.date.min(), ind.date.max())
LUN = lunar_features(dates)
YEARS = sorted(set(dates.year))

def build_panel(df_long, signals, subdiv_of, years, lunar, interpolate=3):
    frames = []
    for reg, g in df_long.groupby("region"):
        g = g.set_index("date").reindex(dates)
        g["region"] = reg
        for s in signals:
            g[s] = g[s].interpolate(limit=interpolate)
        cal = calendar_frame(dates, state_festivals(subdiv_of(reg), years), lunar)
        hist = pd.concat([history_features(g[s]).add_prefix(f"{s}__") for s in signals], axis=1)
        g = pd.concat([g[["region"] + signals], hist, dow_frame(dates), cal], axis=1)
        g["ctx"] = festival_context(cal).values
        frames.append(g)
    P = pd.concat(frames).rename_axis("date").reset_index()
    cal_cols = [c for c in P.columns if c.startswith(("fest_", "pre_", "post_", "days_to", "lunar_", "amavasya", "purnima"))]
    P["fest_win"] = festival_window(P.set_index("date")[cal_cols]).values
    return P, cal_cols

t0 = time.time()
D, CAL = build_panel(ind, SIGNALS, lambda r: STATE_CODES[r], YEARS, LUN)
print(f"panel {D.shape} built in {time.time()-t0:.1f}s; calendar features: {len(CAL)}")
print("share of days inside a festival window:", round(D.fest_win.mean(), 3))
tn = D[D.region == "Tamil Nadu"].set_index("date")
print("Tamil Nadu festivals 2021-22 (sample):")
print(tn.loc["2021-10-10":"2022-01-20", ["ctx"]].dropna().query("ctx.str.endswith('@+0')", engine="python").to_string())

# %% [markdown]
# ### 2.1 Is the festival effect real? (motivation check on raw data)
# Standardised deviation from the same-weekday 4-week median, on festival days vs ordinary days.

# %%
sz_all = SeasonalZ(SIGNALS).fit(D)
Zall = pd.DataFrame({s: ((D[s] - D[f"{s}__sameDOW_med4"]) / D.region.map(sz_all.scale[s])) for s in SIGNALS})
D["maxabsz_raw"] = Zall.abs().max(1)
grp = D.ctx.fillna("none@0").str.split("@").str[0]
eff = D.assign(grp=grp, festday=D.ctx.fillna("").str.endswith("@+0")).groupby(["grp"]).maxabsz_raw.median().sort_values()
print("median max|z| by festival group (incl. ±window):\n", eff.round(2).to_string())
print("\nP(max|z|>4): festival day", round((D.maxabsz_raw[D.ctx.fillna('').str.endswith('@+0')] > 4).mean(), 3),
      "| non-festival day", round((D.maxabsz_raw[~D.fest_win] > 4).mean(), 3))

# %% [markdown]
# ## 3. Chronological split (no look-ahead)
# Train: 2020-03-15 → 2021-06-30 · Validation (threshold & scale calibration): 2021-07-01 → 2021-09-30 · Test: 2021-10-01 → 2022-08-19.
# The test period contains Dussehra, Diwali, Christmas, Pongal/Sankranti, Holi, Ugadi/Puthandu, Eid, Independence Day.

# %%
TR = D[(D.date >= "2020-03-15") & (D.date <= "2021-06-30")].copy()
VA = D[(D.date >= "2021-07-01") & (D.date <= "2021-09-30")].copy()
TE = D[D.date >= "2021-10-01"].copy()
for n, x in [("train", TR), ("val", VA), ("test", TE)]:
    print(f"{n:5s} rows={len(x):6d}  festival-window share={x.fest_win.mean():.3f}")

# %% [markdown]
# ## 4. The FACAD algorithm
# ```
# Algorithm FACAD(train T, validation V, stream X)
#   for each region r, signal s, day t:
#       b(t)  = median(y(t-7), y(t-14), y(t-21), y(t-28))          # same-weekday seasonal baseline (past only)
#       σ_rs  = 1.4826 · MAD(y - b) over NON-festival training days  # robust scale
#       z(t)  = (y(t) - b(t)) / σ_rs
#   ctx(t) = nearest festival (group g, offset k ∈ [-3, +2]) from the state's lunisolar calendar, else ∅
#   for each context c observed ≥ m times in T (pooled across states):
#       E[c,s] = w·median(z | c),   V[c,s] = max(1, 1 + w·(MAD(z | c) - 1)),   w = n_c/(n_c + λ)
#   unseen c → festival-group average; else no adjustment
#   z*(t) = (z(t) - E[ctx(t)]) / V[ctx(t)]
#   score(t) = max_s |z*(t)|;   alarm if score(t) ≥ τ  (τ = 98th percentile of scores on V)
# ```
# Cost: O(N·S) per day after an O(N·S·|C|) training pass — it runs on a laptop and is fully interpretable
# (every adjustment is a row in the E/V table). Defaults: m = 3, λ = 5, kernel −3…+2 days.

# %%
facad_demo = FACAD(SIGNALS).fit(TR, TR.ctx)
Etab = pd.DataFrame(facad_demo.E).T.round(2)
Etab.index.name = "context"
show = [c for c in Etab.index if c.split("@")[0] in ("diwali", "harvest", "holi", "dussehra", "onam")]
print("Learned festival effect E[context, signal] (standardised units):")
print(Etab.loc[sorted(show)].to_string())
Etab.to_csv("results/facad_effect_table.csv")

# %% [markdown]
# ## 5. Evaluation protocol
# * Anomalies are injected into the **test** period only: per state, 12 anomalies on ordinary days and 4 hidden inside
#   festival windows (to test H3). Each anomaly perturbs 1–3 random signals by ±(3–6)·σ_rs. Clean festival-window days
#   are never perturbed, so any alarm on them is a false alarm.
# * Two operating points: **matched budget** — every method's threshold set so that overall test FPR = 5 %
#   (isolates *where* the false alarms fall); **validation-calibrated** — τ = 98th percentile of validation scores (deployment-realistic).
# * 10 seeds; paired Wilcoxon signed-rank test vs FACAD.
#
# Methods: **FACAD** (proposed); **SeasonalZ** (calendar-blind seasonal robust z — the same pipeline without step 3);
# **FACAD-GBR** (proposed plug-in: the FACAD calendar table applied to the calendar-blind GBR's residual z-scores);
# **Blind-GBR / Cal-GBR** (gradient-boosted normal-behaviour regressor without / with all calendar + lunar features);
# **IF / IF+cal** (Isolation Forest without / with calendar features); **Whitelist** (SeasonalZ with all alarms in festival windows suppressed).

# %%
def run_methods(TRd, VAd, TEd, signals, cal_cols, seed, n_inj=12, n_fest=4, include_gbr=True, extra=None):
    rng = np.random.default_rng(seed)
    sz = SeasonalZ(signals).fit(TRd)
    fa = FACAD(signals).fit(TRd, TRd.ctx)
    methods = {"FACAD": (lambda d: fa.score(d, d.ctx)), "SeasonalZ": sz.score}
    if include_gbr:
        bg = ResidualDetector(signals, cal_cols, False, seed).fit(TRd).calibrate(VAd)
        cg = ResidualDetector(signals, cal_cols, True, seed).fit(TRd).calibrate(VAd)
        ib = IFDetector(signals, cal_cols, False, seed).fit(TRd)
        ic = IFDetector(signals, cal_cols, True, seed).fit(TRd)
        fw_ = FACADWrap(bg, signals).set_reference(TRd, TRd.ctx).fit(TRd, TRd.ctx)
        methods.update({"FACAD-GBR": (lambda d: fw_.score(d, d.ctx)),
                        "Blind-GBR": bg.score, "Cal-GBR": cg.score, "IF": ib.score, "IF+cal": ic.score})
    if extra:
        methods.update(extra(TRd, signals))
    TEi = inject(TEd, signals, n_inj, rng, TEd.fest_win, scale=sz.scale, n_fest=n_fest)
    lab, fw, fat = TEi.label.values, TEi.fest_win.values, TEi.fest_attack.values
    rows, scores = [], {}
    items = list(methods.items()) + [("Whitelist", sz.score)]
    for name, f in items:
        sc = f(TEi).fillna(0).values
        sup = fw if name == "Whitelist" else None
        vsc = f(VAd).dropna()
        scores[name] = sc
        for mode in ["matched", "val"]:
            if mode == "matched":
                thr = threshold_at_fpr(np.where(sup, -np.inf, sc) if sup is not None else sc, lab, TARGET_FPR)
            else:
                thr = np.quantile(vsc, 0.98)
            r = evaluate(sc, lab, fw, thr, fat, sup)
            r.update(model=name, seed=seed, mode=mode)
            rows.append(r)
    return pd.DataFrame(rows), TEi, scores

t0 = time.time()
allres, keep = [], None
for seed in range(N_SEEDS):
    r, TEi, scores = run_methods(TR, VA, TE, SIGNALS, CAL, seed)
    allres.append(r)
    if seed == 0:
        keep = (TEi, scores)
R1 = pd.concat(allres, ignore_index=True)
R1.to_csv("results/e1_all_seeds.csv", index=False)
print(f"E1 done in {time.time()-t0:.0f}s")

# %% [markdown]
# ## 6. Results — E1 (real Indian state-level data)

# %%
METRICS = ["FPR_festival", "FPR_nonfestival", "FPR_overall", "precision", "recall", "F1", "PR_AUC",
           "festival_share_of_FP", "recall_festival_attacks", "recall_other_attacks"]
ORDER = ["FACAD", "SeasonalZ", "Whitelist", "FACAD-GBR", "Blind-GBR", "Cal-GBR", "IF", "IF+cal"]

def summary(R, mode, order=ORDER):
    g = R[R["mode"] == mode].groupby("model")[METRICS]
    m, s = g.mean(), g.std()
    out = m.round(3).astype(str) + " ± " + s.round(3).astype(str)
    return out.loc[[o for o in order if o in out.index]], m.loc[[o for o in order if o in m.index]]

def sig_tests(R, mode, ref="FACAD", metrics=("FPR_festival", "F1", "PR_AUC", "recall_festival_attacks")):
    rows = []
    a = R[(R["mode"] == mode) & (R.model == ref)].set_index("seed")
    for mdl in R.model.unique():
        if mdl == ref:
            continue
        b = R[(R["mode"] == mode) & (R.model == mdl)].set_index("seed")
        row = {"vs": mdl}
        for k in metrics:
            d = (a[k] - b[k]).dropna()
            row[f"Δ{k}"] = round(d.mean(), 3)
            row[f"p_{k}"] = round(wilcoxon(d).pvalue, 4) if (d != 0).any() else 1.0
        rows.append(row)
    return pd.DataFrame(rows)

for mode in ["matched", "val"]:
    tab, _ = summary(R1, mode)
    print(f"\n=== E1 — {mode} threshold (mean ± SD over {N_SEEDS} seeds) ===")
    print(tab.to_string())
    tab.to_csv(f"results/e1_summary_{mode}.csv")
print("\nPaired Wilcoxon (FACAD minus other), matched budget:")
T1 = sig_tests(R1, "matched"); print(T1.to_string(index=False)); T1.to_csv("results/e1_wilcoxon_matched.csv", index=False)
print("\nPaired Wilcoxon, validation-calibrated:")
T1v = sig_tests(R1, "val"); print(T1v.to_string(index=False)); T1v.to_csv("results/e1_wilcoxon_val.csv", index=False)
GB = ["FACAD-GBR", "Blind-GBR", "Cal-GBR"]
T1g = pd.concat([sig_tests(R1[R1.model.isin(GB)], m, ref="FACAD-GBR").assign(mode=m) for m in ["matched", "val"]])
print("\nPaired Wilcoxon (FACAD-GBR minus GBR baselines):"); print(T1g.to_string(index=False)); T1g.to_csv("results/e1_wilcoxon_facadgbr.csv", index=False)

# %%
_, M = summary(R1, "matched")
fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
for ax, (k, ttl) in zip(axes, [("FPR_festival", "False-alarm rate on festival days\n(matched 5% overall budget)"),
                               ("PR_AUC", "PR-AUC (threshold-free)"),
                               ("recall_festival_attacks", "Recall of attacks hidden\nin festival windows")]):
    vals = M[k]
    sd = R1[R1["mode"] == "matched"].groupby("model")[k].std().loc[vals.index]
    ax.barh(range(len(vals)), vals.values, xerr=sd.values, color=[C[m] for m in vals.index],
            height=0.6, error_kw=dict(lw=0.8, ecolor="#52514e"))
    ax.set_yticks(range(len(vals))); ax.set_yticklabels(vals.index); ax.invert_yaxis()
    ax.set_title(ttl, loc="left"); ax.grid(axis="y", visible=False)
    for i, (v, e) in enumerate(zip(vals.values, sd.values)):
        ax.text(v + e + (vals.max() * 0.03), i, f"{v:.3f}", va="center", fontsize=7.5, color="#0b0b0b")
    ax.set_xlim(0, (vals + sd).max() * 1.3 + 1e-3)
plt.tight_layout(); plt.savefig("figures/fig2_e1_main_results.png", dpi=200, bbox_inches="tight"); plt.show()

# %% [markdown]
# ### 6.1 Per-festival false-alarm rate (matched budget, seed-averaged)

# %%
def per_group_fpr(seed_list, TRd, VAd, TEd, signals):
    rows = []
    for seed in seed_list:
        rng = np.random.default_rng(seed)
        sz = SeasonalZ(signals).fit(TRd); fa = FACAD(signals).fit(TRd, TRd.ctx)
        TEi = inject(TEd, signals, 12, rng, TEd.fest_win, scale=sz.scale, n_fest=4)
        lab = TEi.label.values
        g = TEi.ctx.fillna("none@0").str.split("@").str[0].values
        for name, sc in [("FACAD", fa.score(TEi, TEi.ctx).fillna(0).values), ("SeasonalZ", sz.score(TEi).fillna(0).values)]:
            thr = threshold_at_fpr(sc, lab, TARGET_FPR)
            pred = sc >= thr
            for grp_ in np.unique(g):
                msk = (g == grp_) & (lab == 0)
                rows.append({"seed": seed, "model": name, "group": grp_, "FPR": pred[msk].mean(), "n": msk.sum()})
    return pd.DataFrame(rows)

PG = per_group_fpr(range(N_SEEDS), TR, VA, TE, SIGNALS)
pg = PG.groupby(["group", "model"]).FPR.mean().unstack().round(3)
pg["n_days"] = PG[(PG.seed == 0) & (PG.model == "FACAD")].groupby("group").n.first()
pg["reduction_%"] = (100 * (1 - pg.FACAD / pg.SeasonalZ.replace(0, np.nan))).round(1)
pg = pg.sort_values("SeasonalZ", ascending=False)
print(pg.to_string()); pg.to_csv("results/e1_per_festival_fpr.csv")

fig, ax = plt.subplots(figsize=(6.4, 3.4))
pgp = pg.drop(index=[i for i in pg.index if i == "none"])
y = np.arange(len(pgp))
ax.barh(y - 0.18, pgp.SeasonalZ, height=0.34, color=C["SeasonalZ"], label="SeasonalZ (calendar-blind)")
ax.barh(y + 0.18, pgp.FACAD, height=0.34, color=C["FACAD"], label="FACAD (festival-aware)")
ax.set_yticks(y); ax.set_yticklabels(pgp.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
ax.axvline(TARGET_FPR, color="#52514e", lw=0.8, ls="--"); ax.text(TARGET_FPR, -0.8, " 5% overall budget", fontsize=7, color="#52514e")
ax.set_xlabel("False-alarm rate within festival window"); ax.legend(frameon=False, loc="lower right", fontsize=7.5)
ax.set_title("False alarms by festival group (test period, matched budget)", loc="left")
plt.tight_layout(); plt.savefig("figures/fig3_per_festival_fpr.png", dpi=200, bbox_inches="tight"); plt.show()

# %% [markdown]
# ### 6.2 Case study — Tamil Nadu around Diwali 2021 and Pongal 2022

# %%
TEi0, sc0 = keep
TEi0 = TEi0.assign(FACAD=sc0["FACAD"], SeasonalZ=sc0["SeasonalZ"])
thrF = threshold_at_fpr(sc0["FACAD"], TEi0.label.values, TARGET_FPR)
thrS = threshold_at_fpr(sc0["SeasonalZ"], TEi0.label.values, TARGET_FPR)
tnx = TEi0[TEi0.region == "Tamil Nadu"].set_index("date")
fig, axes = plt.subplots(3, 1, figsize=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1, 1]})
win = slice("2021-10-01", "2022-02-15")
t = tnx.loc[win]
axes[0].plot(t.index, t["retail and recreation"], color="#2a78d6", lw=1.4, label="retail & recreation")
axes[0].plot(t.index, t["transit stations"], color="#1baf7a", lw=1.4, label="transit stations")
axes[0].set_ylabel("% vs baseline"); axes[0].legend(frameon=False, ncol=2, fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 0.12))
axes[0].set_title("Tamil Nadu, real mobility data with injected anomalies (seed 0); shaded = festival window", loc="left", pad=16)
for ax, name, thr in [(axes[1], "SeasonalZ", thrS), (axes[2], "FACAD", thrF)]:
    ax.plot(t.index, t[name], color=C[name], lw=1.3)
    ax.axhline(thr, color="#52514e", lw=0.8, ls="--")
    fp = t[(t[name] >= thr) & (t.label == 0)]; tp = t[(t[name] >= thr) & (t.label == 1)]
    ax.scatter(fp.index, fp[name], s=26, marker="x", color="#e34948", label=f"false alarm ({len(fp)})", zorder=3)
    ax.scatter(tp.index, tp[name], s=26, marker="o", facecolor="none", edgecolor="#0b0b0b", label=f"detected anomaly ({len(tp)})", zorder=3)
    ax.set_ylabel(f"{name}\nscore"); ax.legend(frameon=False, fontsize=7, loc="upper right")
for ax in axes:
    for d, lab_ in [("2021-10-14", "Ayudha Puja"), ("2021-11-04", "Diwali"), ("2021-12-25", "Christmas"), ("2022-01-14", "Pongal")]:
        ax.axvspan(pd.Timestamp(d) - pd.Timedelta(days=3), pd.Timestamp(d) + pd.Timedelta(days=2), color="#eda100", alpha=0.13, lw=0)
        if ax is axes[0]:
            ax.text(pd.Timestamp(d), -78, lab_, fontsize=7, ha="center", va="bottom", color="#52514e")
plt.tight_layout(); plt.savefig("figures/fig4_tamilnadu_case.png", dpi=200, bbox_inches="tight"); plt.show()
print("Tamil Nadu false alarms inside festival windows — SeasonalZ:",
      int(((tnx.SeasonalZ >= thrS) & (tnx.label == 0) & tnx.fest_win).sum()),
      "| FACAD:", int(((tnx.FACAD >= thrF) & (tnx.label == 0) & tnx.fest_win).sum()))

# %% [markdown]
# ## 7. Ablation (RQ3)
# * **No adjustment** – FACAD pipeline with E = 0, V = 1 (scale still from non-festival days)
# * **Mean shift only** – V = 1 · **Variance only** – E = 0
# * **Festival day only** – no pre/post kernel (contexts restricted to offset 0)
# * **Gregorian holidays only** – only fixed-date national holidays (Republic, Independence, Gandhi Jayanti, May Day, Christmas); all lunisolar festivals removed
# * **National calendar only** – the all-India holiday list applied to every state (no state-specific festivals)

# %%
class FACADVariant(FACAD):
    def __init__(self, signals, mode="full", **kw):
        super().__init__(signals, **kw); self.mode = mode
    def adjusted_z(self, df, ctx):
        Z = self._z(df)
        if self.mode == "none":
            return Z
        E, V = self.effect_frames(df, ctx)
        if self.mode == "mean":
            return Z - E
        if self.mode == "var":
            return Z / V
        return (Z - E) / V

GREG = {"national", "christmas"}

def ctx_variant(P, kind):
    c = P.ctx.copy()
    if kind == "dayonly":
        c[~c.fillna("").str.endswith("@+0")] = None
    if kind == "gregorian":
        c[~c.fillna("x@").str.split("@").str[0].isin(GREG)] = None
    return c

def national_ctx(P):
    nat = calendar_frame(dates, state_festivals(None, YEARS), LUN)
    cn = festival_context(nat)
    return P.date.map(cn)

D["ctx_nat"] = national_ctx(D).values
D["ctx_day"] = ctx_variant(D, "dayonly").values
D["ctx_greg"] = ctx_variant(D, "gregorian").values
TR, VA, TE = (D[(D.date >= "2020-03-15") & (D.date <= "2021-06-30")], D[(D.date >= "2021-07-01") & (D.date <= "2021-09-30")], D[D.date >= "2021-10-01"])

variants = [("FACAD (full)", "full", "ctx"), ("No adjustment", "none", "ctx"), ("Mean shift only", "mean", "ctx"),
            ("Variance only", "var", "ctx"), ("Festival day only", "full", "ctx_day"),
            ("Gregorian holidays only", "full", "ctx_greg"), ("National calendar only", "full", "ctx_nat")]
abl = []
for seed in range(N_SEEDS):
    rng = np.random.default_rng(seed)
    sz = SeasonalZ(SIGNALS).fit(TR)
    TEi = inject(TE, SIGNALS, 12, rng, TE.fest_win, scale=sz.scale, n_fest=4)
    lab, fw, fat = TEi.label.values, TEi.fest_win.values, TEi.fest_attack.values
    for name, mode, col in variants:
        m = FACADVariant(SIGNALS, mode).fit(TR, TR[col])
        sc = m.score(TEi, TEi[col]).fillna(0).values
        r = evaluate(sc, lab, fw, threshold_at_fpr(sc, lab, TARGET_FPR), fat); r.update(variant=name, seed=seed); abl.append(r)
A = pd.DataFrame(abl)
atab = A.groupby("variant")[["FPR_festival", "festival_share_of_FP", "F1", "PR_AUC", "recall_festival_attacks"]].agg(["mean", "std"]).round(3)
atab = atab.loc[[v[0] for v in variants]]
print(atab.to_string()); A.to_csv("results/e1_ablation_all_seeds.csv", index=False); atab.to_csv("results/e1_ablation_summary.csv")

# %%
am = A.groupby("variant").FPR_festival.mean().loc[[v[0] for v in variants]]
asd = A.groupby("variant").FPR_festival.std().loc[am.index]
fig, ax = plt.subplots(figsize=(6.2, 3.0))
cols = ["#2a78d6" if v == "FACAD (full)" else "#b9b8b2" for v in am.index]
ax.barh(range(len(am)), am.values, xerr=asd.values, color=cols, height=0.6, error_kw=dict(lw=0.8, ecolor="#52514e"))
ax.set_yticks(range(len(am))); ax.set_yticklabels(am.index); ax.invert_yaxis(); ax.grid(axis="y", visible=False)
for i, v in enumerate(am.values):
    ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=7.5)
ax.set_xlabel("False-alarm rate on festival-window days (matched 5% budget)")
ax.set_title("Ablation: which calendar knowledge matters", loc="left")
plt.tight_layout(); plt.savefig("figures/fig5_ablation.png", dpi=200, bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 8. Save consolidated results

# %%
def flat(R, mode, order):
    g = R[R["mode"] == mode].groupby("model")[METRICS]
    return {m: {k: [round(float(g.mean().loc[m, k]), 4), round(float(g.std().loc[m, k]), 4)] for k in METRICS} for m in order if m in g.mean().index}

summary_json = {"E1": {"matched": flat(R1, "matched", ORDER), "val": flat(R1, "val", ORDER),
                       "wilcoxon_matched": T1.to_dict(orient="records"), "wilcoxon_val": T1v.to_dict(orient="records"),
                       "wilcoxon_facadgbr": T1g.to_dict(orient="records"), "per_festival": pg.reset_index().to_dict(orient="records"),
                       "ablation": {v: {k: round(float(A[A.variant == v][k].mean()), 4) for k in ["FPR_festival", "F1", "PR_AUC", "recall_festival_attacks", "festival_share_of_FP"]} for v, _, _ in variants},
                       "n_rows": int(len(D)), "n_regions": int(D.region.nunique()), "test_rows": int(len(TE)),
                       "test_fest_share": float(TE.fest_win.mean())}}
with open("results/summary.json", "w") as f:
    json.dump(summary_json, f, indent=1, default=str)
print("saved results/summary.json")
