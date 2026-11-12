# Experiment 1 extensions: sensitivity (lambda, kernel), leave-one-state-out transfer, magnitude curve
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", "experiment1_mobility"))
src = open("FACAD_experiment1_mobility.py").read()
cut = src.index("# %% [markdown]\n# ## 5. Evaluation protocol")
import matplotlib; matplotlib.use("Agg")
exec(compile(src[:cut].replace("plt.show()", "plt.close('all')"), "e1_setup", "exec"))
import facad_lib
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
SEEDS = range(10)
def injected(seed, mag=(3.0, 6.0)):
    rng = np.random.default_rng(seed)
    sz = SeasonalZ(SIGNALS).fit(TR)
    return inject(TE, SIGNALS, 12, rng, TE.fest_win, mag=mag, scale=sz.scale, n_fest=4), sz
def ev(sc, TEi):
    lab, fw, fat = TEi.label.values, TEi.fest_win.values, TEi.fest_attack.values
    return evaluate(sc, lab, fw, threshold_at_fpr(sc, lab, TARGET_FPR), fat)
t0 = time.time()
# ---- (1) lambda sensitivity
rows = []
LAMS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
models = {l: FACAD(SIGNALS, shrink=l if l > 0 else 1e-9).fit(TR, TR.ctx) for l in LAMS}
INJ = {s: injected(s) for s in SEEDS}
for s in SEEDS:
    TEi, sz = INJ[s]
    for l, m in models.items():
        r = ev(m.score(TEi, TEi.ctx).fillna(0).values, TEi); r.update(seed=s, analysis="lambda", value=l); rows.append(r)
    r = ev(sz.score(TEi).fillna(0).values, TEi); r.update(seed=s, analysis="lambda", value=-1); rows.append(r)
print("lambda done", round(time.time() - t0))
# ---- (2) kernel sensitivity (evaluation windows fixed at -3..+2; only FACAD's context changes)
KERNELS = [(0, 0), (1, 1), (2, 1), (3, 2), (5, 3), (7, 3)]
regs = sorted(D.region.unique())
def ctx_for_kernel(pre, post):
    facad_lib.PRE, facad_lib.POST = pre, post
    m = {}
    for reg in regs:
        cal = calendar_frame(dates, state_festivals(STATE_CODES[reg], YEARS), LUN)
        m[reg] = festival_context(cal)
    facad_lib.PRE, facad_lib.POST = 3, 2
    return m
for pre, post in KERNELS:
    m = ctx_for_kernel(pre, post)
    col = f"ctx_k{pre}{post}"
    D[col] = [m[r].get(d) for r, d in zip(D.region, D.date)]
chk = (D["ctx_k32"].fillna("") == D.ctx.fillna("")).mean()
print("kernel (3,2) reproduces default ctx share:", round(chk, 4))
TRk = D[(D.date >= "2020-03-15") & (D.date <= "2021-06-30")]
for pre, post in KERNELS:
    col = f"ctx_k{pre}{post}"
    m = FACAD(SIGNALS).fit(TRk, TRk[col])
    for s in SEEDS:
        TEi, _ = INJ[s]
        TEi = TEi.copy(); TEi[col] = D.loc[TEi.index, col].values
        r = ev(m.score(TEi, TEi[col]).fillna(0).values, TEi); r.update(seed=s, analysis="kernel", value=f"-{pre}..+{post}"); rows.append(r)
print("kernel done", round(time.time() - t0))
# ---- (3) leave-one-state-out transfer: effect table learned on the other 35 states, own scale from own training data
fa_all = FACAD(SIGNALS).fit(TR, TR.ctx)
LOSO = {}
for reg in regs:
    m = FACAD(SIGNALS).fit(TR[TR.region != reg], TR[TR.region != reg].ctx)
    m.scale = fa_all.scale
    LOSO[reg] = m
for s in SEEDS:
    TEi, sz = INJ[s]
    sc = pd.Series(0.0, index=TEi.index)
    for reg in regs:
        ix = TEi.region == reg
        sc[ix] = LOSO[reg].score(TEi[ix], TEi[ix].ctx).fillna(0)
    r = ev(sc.values, TEi); r.update(seed=s, analysis="loso", value="FACAD-LOSO (table from other 35 states)"); rows.append(r)
    r = ev(fa_all.score(TEi, TEi.ctx).fillna(0).values, TEi); r.update(seed=s, analysis="loso", value="FACAD (pooled, all states)"); rows.append(r)
    r = ev(sz.score(TEi).fillna(0).values, TEi); r.update(seed=s, analysis="loso", value="SeasonalZ (calendar-blind)"); rows.append(r)
print("loso done", round(time.time() - t0))
# ---- (4) magnitude curve
for k in [2.0, 3.0, 4.0, 5.0, 6.0]:
    for s in SEEDS:
        TEi, sz = injected(s, mag=(k, k))
        for name, sc in (("FACAD", fa_all.score(TEi, TEi.ctx)), ("SeasonalZ", sz.score(TEi))):
            r = ev(sc.fillna(0).values, TEi); r.update(seed=s, analysis="magnitude", value=k, method=name); rows.append(r)
print("magnitude done", round(time.time() - t0))
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/e1_extensions_all.csv", index=False)
cols = ["FPR_festival", "PR_AUC", "F1", "recall_festival_attacks", "recall_other_attacks"]
for an in ["lambda", "kernel", "loso"]:
    print(an); print(A[A.analysis == an].groupby("value")[cols].agg(["mean", "std"]).round(3).to_string())
print(A[A.analysis == "magnitude"].groupby(["value", "method"])[cols].mean().round(3).to_string())
# Wilcoxon: LOSO vs SeasonalZ, LOSO vs pooled
from scipy.stats import wilcoxon
L = A[A.analysis == "loso"].pivot(index="seed", columns="value", values=["FPR_festival", "PR_AUC"])
for met in ["FPR_festival", "PR_AUC"]:
    a = L[met]["FACAD-LOSO (table from other 35 states)"]; b = L[met]["SeasonalZ (calendar-blind)"]; c = L[met]["FACAD (pooled, all states)"]
    print(met, "LOSO-vs-blind p=", wilcoxon(a, b).pvalue, "| LOSO-vs-pooled p=", wilcoxon(a, c).pvalue)
