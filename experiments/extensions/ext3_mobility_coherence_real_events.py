# Experiment 1: (a) festival coherence index and coherence-gated FACAD (FACAD-CG); (b) real-event benchmark
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", "experiment1_mobility"))
src = open("FACAD_experiment1_mobility.py").read()
cut = src.index("# %% [markdown]\n# ## 5. Evaluation protocol")
import matplotlib; matplotlib.use("Agg")
exec(compile(src[:cut].replace("plt.show()", "plt.close('all')"), "e1_setup", "exec"))
from scipy.stats import wilcoxon, spearmanr
OUT = os.path.join(HERE, "results"); TAU = 0.5
SEEDS = range(10)

def coherence(model, df, ctx):
    """C[c][s] = |sum_r n_r * sign(median_r z)| / sum_r n_r over states observing context c (training data only)."""
    Z = model._z(df); Z["ctx"] = ctx.values; Z["region"] = df.region.values; Z["grp"] = Z.ctx.str.split("@").str[0]
    out = {}
    for key in ("ctx", "grp"):
        C = {}
        for c, g in Z.dropna(subset=[key]).groupby(key):
            per = g.groupby("region")[model.signals].median(); n = g.groupby("region").size()
            if len(per) < 2:
                C[c] = pd.Series(1.0, index=model.signals); continue
            C[c] = (np.sign(per).mul(n, axis=0).sum().abs() / n.sum())
        out[key] = C
    return out

class FACADCG(FACAD):
    """Coherence-gated FACAD: keep the shift E only where festival effects agree in sign across states (C >= TAU)."""
    def fit(self, df, ctx):
        super().fit(df, ctx)
        self.C = coherence(self, df, ctx)
        for store, key in ((self.E, "ctx"), (self.Eg, "grp")):
            for c in list(store):
                gate = (self.C[key].get(c, pd.Series(1.0, index=self.signals)) >= TAU).astype(float)
                store[c] = store[c] * gate.reindex(store[c].index).fillna(1.0)
        return self

t0 = time.time()
fa = FACAD(SIGNALS).fit(TR, TR.ctx)
cg = FACADCG(SIGNALS).fit(TR, TR.ctx)
spread = FACAD(SIGNALS).fit(TR, TR.ctx)
for c in spread.E: spread.E[c] = spread.E[c] * 0
for c in spread.Eg: spread.Eg[c] = spread.Eg[c] * 0
# coherence by festival group (n-weighted mean over offsets and signals) -> table
Cg = pd.DataFrame(cg.C["grp"]).T
Cg["coherence_mean"] = Cg[SIGNALS].mean(1)
Cg.round(3).to_csv(f"{OUT}/e1_coherence_by_group.csv")
print(Cg.round(2).to_string())
share_gated = np.mean([(cg.C["ctx"][c] < TAU).mean() for c in cg.C["ctx"]])
print("share of (context, signal) shifts removed by the gate:", round(share_gated, 3))
# ---- (a) injected-anomaly protocol, 10 seeds
rows = []
for s in (SEEDS if not os.environ.get("ONLY_B") else []):
    rng = np.random.default_rng(s)
    sz = SeasonalZ(SIGNALS).fit(TR)
    TEi = inject(TE, SIGNALS, 12, rng, TE.fest_win, scale=sz.scale, n_fest=4)
    lab, fw, fat = TEi.label.values, TEi.fest_win.values, TEi.fest_attack.values
    for name, sc in (("FACAD", fa.score(TEi, TEi.ctx)), ("FACAD-CG", cg.score(TEi, TEi.ctx)),
                     ("Spread only", spread.score(TEi, TEi.ctx)), ("SeasonalZ", sz.score(TEi))):
        sc = sc.fillna(0).values
        r = evaluate(sc, lab, fw, threshold_at_fpr(sc, lab, TARGET_FPR), fat); r.update(model=name, seed=s); rows.append(r)
R = pd.DataFrame(rows)
if not os.environ.get("ONLY_B"): R.to_csv(f"{OUT}/e1_cg_all_seeds.csv", index=False)
print(R.groupby("model")[["FPR_festival", "PR_AUC", "F1", "recall_festival_attacks"]].agg(["mean", "std"]).round(3).to_string())
P = R.pivot(index="seed", columns="model")
for m in ["FPR_festival", "PR_AUC", "recall_festival_attacks"]:
    for ref in ["FACAD", "SeasonalZ"]:
        d = P[m]["FACAD-CG"] - P[m][ref]
        print(m, "CG -", ref, round(float(np.median(d)), 4), "p=", wilcoxon(d).pvalue if (d != 0).any() else 1.0)
# per-festival FPR (seed-averaged) for coherence-vs-gain analysis
pf = []
for s in SEEDS:
    rng = np.random.default_rng(s)
    sz = SeasonalZ(SIGNALS).fit(TR)
    TEi = inject(TE, SIGNALS, 12, rng, TE.fest_win, scale=sz.scale, n_fest=4)
    lab = TEi.label.values; grp = TEi.ctx.fillna("none@0").str.split("@").str[0].values
    for name, sc in (("FACAD", fa.score(TEi, TEi.ctx)), ("FACAD-CG", cg.score(TEi, TEi.ctx)), ("SeasonalZ", sz.score(TEi))):
        sc = sc.fillna(0).values; thr = threshold_at_fpr(sc, lab, TARGET_FPR)
        for g_ in np.unique(grp):
            sel = (grp == g_) & (lab == 0) & TEi.fest_win.values
            if sel.sum() >= 20:
                pf.append({"seed": s, "model": name, "group": g_, "FPR": float((sc[sel] >= thr).mean()), "n": int(sel.sum())})
PF = pd.DataFrame(pf).groupby(["group", "model"]).FPR.mean().unstack()
PF["reduction_FACAD"] = 1 - PF["FACAD"] / PF["SeasonalZ"]
PF["reduction_CG"] = 1 - PF["FACAD-CG"] / PF["SeasonalZ"]
PF = PF.join(Cg["coherence_mean"])
PF.round(3).to_csv(f"{OUT}/e1_per_group_coherence_gain.csv"); print(PF.round(3).to_string())
ok = PF.dropna()
rho = spearmanr(ok.coherence_mean, ok.reduction_FACAD); print("Spearman coherence vs FACAD reduction:", rho)
# ---- (b) real-event benchmark (no injection): documented disruptions in the test period
EVENTS = [("Delhi", d, "Omicron weekend curfew") for d in ["2022-01-08", "2022-01-09", "2022-01-15", "2022-01-16", "2022-01-22", "2022-01-23"]] + \
         [("Karnataka", d, "Omicron weekend curfew") for d in ["2022-01-08", "2022-01-09", "2022-01-15", "2022-01-16"]] + \
         [("Tamil Nadu", d, "Omicron Sunday lockdown") for d in ["2022-01-09", "2022-01-16", "2022-01-23"]] + \
         [(st, d, "Bharat Bandh general strike") for st in ["Kerala", "West Bengal", "Tamil Nadu"] for d in ["2022-03-28", "2022-03-29"]] + \
         [(st, d, "Cyclone Jawad") for st, ds in [("Andhra Pradesh", ["2021-12-03", "2021-12-04"]), ("Odisha", ["2021-12-05", "2021-12-06"]), ("West Bengal", ["2021-12-05", "2021-12-06"])] for d in ds]  # IMD RSMC report
EV = pd.DataFrame(EVENTS, columns=["region", "date", "event"]); EV["date"] = pd.to_datetime(EV.date)
T = TE.copy(); T["label"] = 0
key = set(zip(EV.region, EV.date))
T["label"] = [int((r, d) in key) for r, d in zip(T.region, T.date)]
assert T.label.sum() == len(EV), (T.label.sum(), len(EV))
T["fest_attack"] = (T.label == 1) & T.fest_win
sz = SeasonalZ(SIGNALS).fit(TR)
bg = ResidualDetector(SIGNALS, CAL, False, 0).fit(TR).calibrate(VA)
cgm = ResidualDetector(SIGNALS, CAL, True, 0).fit(TR).calibrate(VA)
ib = IFDetector(SIGNALS, CAL, False, 0).fit(TR); ic = IFDetector(SIGNALS, CAL, True, 0).fit(TR)
fw_ = FACADWrap(bg, SIGNALS).set_reference(TR, TR.ctx).fit(TR, TR.ctx)
S = {"FACAD": fa.score(T, T.ctx), "FACAD-CG": cg.score(T, T.ctx), "SeasonalZ": sz.score(T), "FACAD-GBR": fw_.score(T, T.ctx),
     "Blind-GBR": bg.score(T), "Cal-GBR": cgm.score(T), "IF": ib.score(T), "IF+cal": ic.score(T)}
lab = T.label.values; fw = T.fest_win.values
rows, ev_rows = [], []
for name, sc in list(S.items()) + [("Whitelist", S["SeasonalZ"])]:
    sc = sc.fillna(0).values
    sup = fw if name == "Whitelist" else None
    thr = threshold_at_fpr(np.where(sup, -1e9, sc) if sup is not None else sc, lab, TARGET_FPR)
    r = evaluate(sc if sup is None else np.where(sup, -1e9, sc), lab, fw, thr, T.fest_attack.values.astype(int), sup)
    pred = (sc >= thr) & (~sup if sup is not None else True)
    # within-state percentile rank of each event day (1 = highest score of that state's test period)
    pct = pd.Series(sc if sup is None else np.where(sup, -1e9, sc), index=T.index).groupby(T.region).rank(pct=True)
    r.update(model=name, event_days=int(lab.sum()), detected=int(pred[lab == 1].sum()),
             detected_in_window=int(pred[(lab == 1) & fw].sum()), events_in_window=int(((lab == 1) & fw).sum()),
             median_event_pct=float(pct[lab == 1].median()))
    rows.append(r)
    for ix in np.where(lab == 1)[0]:
        ev_rows.append({"model": name, "region": T.region.iat[ix], "date": T.date.iat[ix].date(), "in_window": bool(fw[ix]),
                        "ctx": T.ctx.iat[ix], "score": float(sc[ix]), "pct": float(pct.iat[ix]), "detected": bool(pred[ix])})
RB = pd.DataFrame(rows); RB.to_csv(f"{OUT}/e1_real_events_summary.csv", index=False)
EVR = pd.DataFrame(ev_rows); EVR.to_csv(f"{OUT}/e1_real_events_by_day.csv", index=False)
print(RB[["model", "FPR_festival", "FPR_overall", "PR_AUC", "precision", "recall", "detected", "event_days", "detected_in_window", "events_in_window", "median_event_pct"]].round(3).to_string(index=False))
W = EVR.pivot_table(index=["region", "date"], columns="model", values="pct")
for ref in ["SeasonalZ", "Blind-GBR", "Cal-GBR", "IF+cal"]:
    for m in ["FACAD", "FACAD-CG"]:
        d = W[m] - W[ref]; print("pct rank", m, "-", ref, "median", round(float(d.median()), 3), "p=", wilcoxon(d).pvalue if (d != 0).any() else 1.0)
print(EVR[EVR.model.isin(["FACAD", "SeasonalZ", "Whitelist"])].pivot_table(index=["region", "date", "in_window", "ctx"], columns="model", values="detected", aggfunc="first").to_string())
print("done", round(time.time() - t0))
