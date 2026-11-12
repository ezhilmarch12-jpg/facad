import pandas as pd, numpy as np
from scipy.stats import wilcoxon
R = pd.read_csv("results/all_runs.csv"); C = pd.read_csv("results/coherence_log.csv")
M = ["FPR_holiday", "FPR_window", "PR_AUC", "F1", "recall_in_window", "precision", "recall"]
st = R[R["mode"] == "matched"].groupby(["method", "state"])[M].mean()
print(st.groupby("method").agg(["mean", "std"])[["FPR_holiday", "PR_AUC", "F1", "recall_in_window"]].round(3).to_string())
rng = np.random.default_rng(42)
def boot(d, B=2000):
    d = np.asarray(d); return np.percentile([np.median(rng.choice(d, len(d))) for _ in range(B)], [2.5, 97.5])
def test(a, b, k):
    d = (st.loc[a][k] - st.loc[b][k]).dropna(); p = wilcoxon(d).pvalue if (d != 0).any() else 1.0; lo, hi = boot(d)
    return {"method": a, "vs": b, "metric": k, "mean_a": round(st.loc[a][k].mean(), 3), "mean_b": round(st.loc[b][k].mean(), 3),
            "median_diff": round(float(np.median(d)), 3), "lo": round(lo, 3), "hi": round(hi, 3), "p": float(f"{p:.3g}"),
            "better_states": int((d < 0).sum() if k.startswith("FPR") else (d > 0).sum()), "n": len(d)}
c = C[C.key.str.startswith("diwali")]
nat = c[c.n_states >= 17]
print("\nH1 coherence (national Diwali contexts): mean", round(nat.C.mean(), 3), "range by key", round(nat.groupby("key").C.mean().min(), 3), "-", round(nat.groupby("key").C.mean().max(), 3),
      "| share of keys with C >= 0.5:", round((nat.groupby("key").C.mean() >= 0.5).mean(), 3))
rows = [test("FACAD-CG", "SeasonalZ-a", "FPR_holiday"), test("FACAD-CG", "SeasonalZ-a", "PR_AUC"), test("FACAD-CG", "SeasonalZ-a", "recall_in_window"),
        test("FACAD-CG", "SeasonalZ-a", "F1"), test("FACAD-a", "SeasonalZ-a", "FPR_holiday"), test("FACAD-a", "SeasonalZ-a", "PR_AUC"),
        test("FACAD-H-CG", "SeasonalZ-a", "FPR_holiday"), test("FACAD-H-CG", "SeasonalZ-a", "PR_AUC"), test("FACAD-H-a", "SeasonalZ-a", "PR_AUC"),
        test("FACAD-prior (mobility)", "SeasonalZ-a", "FPR_holiday"), test("FACAD-prior (mobility)", "SeasonalZ-a", "PR_AUC"),
        test("FACAD-CG", "FACAD-a", "PR_AUC"), test("FACAD-CG", "FACAD-a", "FPR_holiday")]
T = pd.DataFrame(rows); print(T.to_string()); T.to_csv("results/hypothesis_tests.csv", index=False)
w = st.loc["Whitelist-a"]["recall_in_window"]
print("\nH4 whitelist recall inside windows:", round(w.mean(), 3), "| FACAD-CG:", round(st.loc["FACAD-CG"]["recall_in_window"].mean(), 3),
      "| p(CG - whitelist) =", wilcoxon(st.loc["FACAD-CG"]["recall_in_window"] - w).pvalue)
g = R[R["mode"].isin(["global", "mondrian"])].groupby(["method", "mode", "state"])[["FPR_holiday", "FPR_window", "FPR_normal", "FPR_overall", "recall"]].mean().groupby(["method", "mode"]).mean()
print(g.loc[["FACAD-CG", "FACAD-H-CG", "FACAD-a", "SeasonalZ-a"]].round(3).to_string())
