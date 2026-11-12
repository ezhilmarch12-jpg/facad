# Experiment 2 extensions: lambda and kernel sensitivity, training-state count, magnitude curve, combined variants
import os, time
HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(os.path.join(HERE, "..", "experiment2_cloudflare"))
src = open("FACAD_cloudflare_experiments.py").read()
cut = src.index("# %% [markdown]\n# ## 7. E-A")
import matplotlib; matplotlib.use("Agg")
exec(compile(src[:cut].replace("plt.show()", "plt.close('all')"), "e2_setup", "exec"))
OUT = os.path.join(HERE, "results"); os.makedirs(OUT, exist_ok=True)
t0 = time.perf_counter()
# kernel context columns from each state's own holiday dates
for s in STATES:
    g = S[s]
    hd = g[g.is_holiday == 1].groupby("date").holiday_name.first()
    fmap = {pd.Timestamp(d): group_of(n) for d, n in hd.items()}
    days_ = pd.to_datetime(sorted(g.date.unique()))
    for pre, post in [(0, 0), (1, 1), (2, 1), (3, 2), (5, 3)]:
        c = {d.strftime("%Y-%m-%d"): v for d, v in context_for(days_, fmap, pre, post).items()}
        g[f"ctx_k{pre}{post}"] = g.date.map(c)
    S[s] = g
print("kernel(3,2) == ctx share:", np.mean([(S[s].ctx_k32 == S[s].ctx).mean() for s in STATES]).round(4))
LAMS = [0.0, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
for l in LAMS:
    TABLE_METHODS[f"lam={l:g}"] = ("za", "ctx", True, l, "full", False)
    TABLE_METHODS[f"H lam={l:g}"] = ("za", "ctx", True, l, "full", True)
for pre, post in [(0, 0), (1, 1), (2, 1), (3, 2), (5, 3)]:
    TABLE_METHODS[f"kernel -{pre}..+{post}"] = ("za", f"ctx_k{pre}{post}", True, LAMBDA, "full", False)
TABLE_METHODS["FACAD-H spread only"] = ("za", "ctx", True, LAMBDA, "spread", True)
def h_plus_prior(z, g, i):
    """regional table shift and spread, spread floored by the mobility prior"""
    tab, treg = TAB_CACHE[("FACAD-H-a", frozenset([i]), REGION[i])]
    E = np.zeros(T); V = np.ones(T)
    for t in range(T):
        e, v, n = lookup(tab, g.ctx.iat[t], g.block.iat[t]); er, vr, nr = lookup(treg, g.ctx.iat[t], g.block.iat[t])
        if nr:
            wr = nr / (nr + LAMBDA); e, v = wr * er + (1 - wr) * e, wr * vr + (1 - wr) * v
        vp = VPRIOR.get(int(g.offset.iat[t]), 1.0) if (g.ctx.iat[t].startswith("dussehra") and not np.isnan(g.offset.iat[t])) else 1.0
        E[t], V[t] = e, max(v, vp)
    return np.abs((z - E) / V)
rows = []
def rec(i, seed, analysis, value, sc, lab, typ, g):
    s_ = np.nan_to_num(sc, nan=0.0)
    rows.append({"state": i, "seed": seed, "analysis": analysis, "value": value, **metrics(s_, lab, typ, g, matched_thr(s_, lab))})
for i in STATES:
    g = S[i]
    for seed in SEEDS:
        x, lab, typ, ev = inject(g, seed, i)
        z = zscores(g, x)            # baselines a/b only (no GBM needed here)
        za = z["za"].values
        rec(i, seed, "base", "SeasonalZ-a", np.abs(za), lab, typ, g)
        rec(i, seed, "base", "FACAD-prior", np.abs(apply_prior(za, g)), lab, typ, g)
        for name in [k for k in TABLE_METHODS if k.startswith(("lam=", "H lam=", "kernel"))] + ["FACAD-a", "FACAD-H-a", "FACAD-H spread only"]:
            an = "lambda" if name.startswith("lam=") else ("lambda_H" if name.startswith("H lam") else ("kernel" if name.startswith("kernel") else "variant"))
            rec(i, seed, an, name, table_score(name, za, g, i, [i]), lab, typ, g)
        rec(i, seed, "variant", "FACAD-H + prior spread", h_plus_prior(za, g, i), lab, typ, g)
    print(i, "sens done", round(time.perf_counter() - t0))
# training-state count (transfer curve): FACAD-a and spread-only learned from k random other states
KS = [3, 6, 9, 12, 15, 18]
for i in STATES:
    g = S[i]; others = [s for s in STATES if s != i]
    for seed in SEEDS:
        x, lab, typ, ev = inject(g, seed, i); za = zscores(g, x)["za"].values
        rng = np.random.default_rng(7000 + 31 * seed + STATES.index(i))
        for k in KS:
            sub = list(rng.choice(others, k, replace=False)) if k < len(others) else others
            tab = learn_table(sub, "za", [i], "ctx", True, LAMBDA)
            for mode, nm in (("full", "FACAD-a"), ("spread", "spread only")):
                rec(i, seed, "ntrain", f"{nm}|{k}", np.abs(apply_table(za, g, tab, "ctx", mode)), lab, typ, g)
    print(i, "ntrain done", round(time.perf_counter() - t0))
# magnitude curve: 8 single-hour spikes of fixed size k*sigma (3 inside festival windows)
def inject_fixed(g, seed, state, k):
    rng = np.random.default_rng(5000 + 1000 * seed + STATES.index(state))
    x = g.clr.values.copy(); sig = mad(CLEAN[(state, None)]["rb"].values)
    win = g.in_festival_window.values.astype(bool); lab = np.zeros(T, int); typ = np.array([""] * T, dtype=object); used = np.zeros(T, bool)
    for want in [True] * N_IN_WINDOW + [False] * (N_EVENTS - N_IN_WINDOW):
        for inside in (want, not want):
            done = False
            for _ in range(500):
                st = int(rng.choice(SIDX))
                if win[st] != inside or used[max(0, st - 2):st + 3].any():
                    continue
                x[st] += rng.choice([-1, 1]) * k * sig; lab[st] = 1; typ[st] = "spike"; used[st] = True; done = True; break
            if done:
                break
    return pd.Series(x), lab, typ
for k in [2.0, 3.0, 4.0, 5.0, 6.0]:
    for i in STATES:
        g = S[i]
        for seed in SEEDS:
            x, lab, typ = inject_fixed(g, seed, i, k); za = zscores(g, x)["za"].values
            for nm, sc in (("SeasonalZ-a", np.abs(za)), ("FACAD-a", table_score("FACAD-a", za, g, i, [i])),
                           ("FACAD-H-a", table_score("FACAD-H-a", za, g, i, [i])), ("FACAD-prior", np.abs(apply_prior(za, g)))):
                rec(i, seed, "magnitude", f"{nm}|{k:g}", sc, lab, typ, g)
    print("magnitude", k, round(time.perf_counter() - t0))
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/e2_extensions_all.csv", index=False)
print("total", round(time.perf_counter() - t0), "s; rows", len(A))
