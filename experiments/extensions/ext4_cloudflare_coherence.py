# Experiment 2: coherence index and coherence-gated FACAD (national and regional), same LOSO protocol, matched budget
import os, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", "experiment2_cloudflare"))
src = open("FACAD_cloudflare_experiments.py").read()
cut = src.index("# %% [markdown]\n# ## 7. E-A")
import matplotlib; matplotlib.use("Agg")
exec(compile(src[:cut].replace("plt.show()", "plt.close('all')"), "e2_setup", "exec"))
from scipy.stats import wilcoxon
OUT = os.path.join(HERE, "results"); TAU = 0.5
def learn_table_cg(states, excl, lam=LAMBDA):
    """FACAD table with coherence C = |sum_s n_s sign(median_s z)| / sum_s n_s per key; shift kept only where C >= TAU."""
    rows = []
    for s in states:
        z = clean_z(s, "za", excl).values; g = S[s]; ok = ~np.isnan(z)
        rows.append(pd.DataFrame({"state": s, "date": g.date.values[ok], "ctx": g.ctx.values[ok], "block": g.block.values[ok], "z": z[ok]}))
    Rr = pd.concat(rows, ignore_index=True); ref = mad(Rr.z[Rr.ctx == ""]) or 1.0
    tab, coh = {}, {}
    for by in (["ctx", "block"], ["ctx"]):
        for k, grp in Rr[Rr.ctx != ""].groupby(by):
            k = k if isinstance(k, tuple) else (k,)
            n = grp[["state", "date"]].drop_duplicates().shape[0]; w = n / (n + lam) if lam > 0 else 1.0
            per = grp.groupby("state").z.median(); ns = grp.groupby("state").date.nunique()
            C = 1.0 if len(per) < 2 else float(abs((np.sign(per) * ns).sum()) / ns.sum())
            E = w * np.median(grp.z) if C >= TAU else 0.0
            tab[k] = (E, max(1.0, 1 + w * (mad(grp.z) / ref - 1)), n); coh[k] = (C, len(per), n)
    return tab, coh
rows, COH = [], []
t0 = time.perf_counter()
for i in STATES:
    g = S[i]; train = [s for s in STATES if s != i]
    tab, coh = learn_table_cg(train, [i])
    treg, cohr = learn_table_cg([s for s in train if REGION[s] == REGION[i]], [i])
    for k, (C, ns, n) in coh.items():
        COH.append({"held_out": i, "key": "|".join(map(str, k)), "C": C, "n_states": ns, "n": n, "level": "national"})
    for seed in SEEDS:
        x, lab, typ, ev = inject(g, seed, i); za = zscores(g, x)["za"].values
        for nm, sc in (("FACAD-CG", np.abs(apply_table(za, g, tab))), ("FACAD-H-CG", np.abs(apply_table(za, g, tab, tab_reg=treg))),
                       ("FACAD-a", table_score("FACAD-a", za, g, i, [i])), ("FACAD-H-a", table_score("FACAD-H-a", za, g, i, [i])),
                       ("SeasonalZ-a", np.abs(za))):
            s_ = np.nan_to_num(sc, nan=0.0)
            rows.append({"state": i, "seed": seed, "method": nm, **metrics(s_, lab, typ, g, matched_thr(s_, lab))})
    print(i, round(time.perf_counter() - t0))
R = pd.DataFrame(rows); R.to_csv(f"{OUT}/e2_cg_all_runs.csv", index=False)
C = pd.DataFrame(COH); C.to_csv(f"{OUT}/e2_coherence.csv", index=False)
M = ["FPR_holiday", "PR_AUC", "F1", "recall_in_window"]
st = R.groupby(["method", "state"])[M].mean()
print(st.groupby("method").agg(["mean", "std"]).round(3).to_string())
print("national coherence by key (mean over folds):"); print(C.groupby("key").C.mean().round(3).sort_values().to_string())
print("share of national keys gated (C < TAU):", round(float((C.C < TAU).mean()), 3))
rng_b = np.random.default_rng(42)
def boot_ci(d, B=2000):
    d = np.asarray(d); meds = [np.median(rng_b.choice(d, len(d))) for _ in range(B)]; return np.percentile(meds, [2.5, 97.5])
out = []
for a, b in [("FACAD-CG", "FACAD-a"), ("FACAD-CG", "SeasonalZ-a"), ("FACAD-H-CG", "FACAD-H-a"), ("FACAD-H-CG", "SeasonalZ-a")]:
    for k in M:
        d = (st.loc[a][k] - st.loc[b][k]).dropna(); p = wilcoxon(d).pvalue if (d != 0).any() else 1.0; lo, hi = boot_ci(d)
        out.append({"method": a, "vs": b, "metric": k, "median_diff": round(float(np.median(d)), 3), "lo": round(lo, 3), "hi": round(hi, 3), "p": float(f"{p:.3g}")})
T = pd.DataFrame(out); T.to_csv(f"{OUT}/e2_cg_tests.csv", index=False); print(T.to_string())
