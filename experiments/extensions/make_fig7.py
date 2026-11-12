import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
MM = 1 / 25.4; W = 174 * MM
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
                     "axes.linewidth": 0.6, "lines.linewidth": 1.1, "axes.edgecolor": "#6b6a66", "pdf.fonttype": 42,
                     "axes.titlelocation": "left", "axes.titleweight": "bold"})
BLUE, ORANGE, GREEN, DARK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#8a8984"
E1 = pd.read_csv("results/e1_extensions_all.csv")
E2 = pd.read_csv("results/e2_extensions_all.csv")
def s2(sub):   # mean over states of seed-averaged values
    return sub.groupby(["value", "state"]).PR_AUC.mean().groupby("value").mean()
fig, ax = plt.subplots(1, 3, figsize=(W, W * 0.33))
# (a) lambda: PR-AUC change relative to lambda = 5
L = [0, 1, 2, 5, 10, 20, 50]; x = np.arange(len(L))
e1 = E1[E1.analysis == "lambda"].assign(v=lambda d: d.value.astype(float)).query("v >= 0").groupby("v").PR_AUC.mean()
a2 = s2(E2[E2.analysis == "lambda"]); h2 = s2(E2[E2.analysis == "lambda_H"])
a2 = pd.Series({l: a2[f"lam={l}"] for l in L}); h2 = pd.Series({l: h2[f"H lam={l}"] for l in L})
for ser, lab, c, m in ((e1, "Exp. 1 FACAD", BLUE, "o"), (a2, "Exp. 2 FACAD-a", ORANGE, "s"), (h2, "Exp. 2 FACAD-H-a", GREEN, "^")):
    ser = pd.Series([ser[float(l)] if float(l) in ser.index else ser[l] for l in L])
    ax[0].plot(x, ser - ser[3], marker=m, ms=3.5, color=c, label=lab)
ax[0].axhline(0, color=DARK, lw=0.6, ls="--"); ax[0].set_xticks(x); ax[0].set_xticklabels([str(l) for l in L])
ax[0].set_xlabel("Shrinkage λ"); ax[0].set_ylabel("PR-AUC change vs λ = 5"); ax[0].legend(frameon=False, loc="lower left", fontsize=6.3, handlelength=1.6)
ax[0].set_title("(a)")
# (b) magnitude
K = [2, 3, 4, 5, 6]
mg1 = E1[E1.analysis == "magnitude"].assign(v=lambda d: d.value.astype(float)).groupby(["method", "v"]).PR_AUC.mean()
mg2 = E2[E2.analysis == "magnitude"].copy(); mg2[["m", "k"]] = mg2.value.str.split("|", expand=True)
mg2 = mg2.groupby(["m", "k", "state"]).PR_AUC.mean().groupby(["m", "k"]).mean()
ax[1].plot(K, [mg1[("FACAD", float(k))] for k in K], marker="o", ms=3.5, color=BLUE, label="Exp. 1 FACAD")
ax[1].plot(K, [mg1[("SeasonalZ", float(k))] for k in K], marker="o", ms=3.5, color=BLUE, ls="--", mfc="white", label="Exp. 1 calendar-blind")
ax[1].plot(K, [mg2[("FACAD-H-a", str(k))] for k in K], marker="^", ms=3.5, color=GREEN, label="Exp. 2 FACAD-H-a")
ax[1].plot(K, [mg2[("SeasonalZ-a", str(k))] for k in K], marker="^", ms=3.5, color=GREEN, ls="--", mfc="white", label="Exp. 2 calendar-blind")
ax[1].set_xticks(K); ax[1].set_xlabel("Injected anomaly size (robust SDs)"); ax[1].set_ylabel("PR-AUC"); ax[1].set_ylim(0, 1)
ax[1].legend(frameon=False, loc="center right", bbox_to_anchor=(1.04, 0.53), fontsize=6, handlelength=1.6); ax[1].set_title("(b)")
# (c) number of training states (Exp. 2)
N = E2[E2.analysis == "ntrain"].copy(); N[["m", "k"]] = N.value.str.split("|", expand=True); N.k = N.k.astype(int)
N = N.groupby(["m", "k", "state"]).PR_AUC.mean().groupby(["m", "k"]).mean()
KS = [3, 6, 9, 12, 15, 18]
ax[2].plot(KS, [N[("FACAD-a", k)] for k in KS], marker="s", ms=3.5, color=ORANGE, label="FACAD-a (shift + spread)")
ax[2].plot(KS, [N[("spread only", k)] for k in KS], marker="D", ms=3.2, color=DARK, label="Spread only")
blind = s2(E2[(E2.analysis == "base")])["SeasonalZ-a"]
ax[2].axhline(blind, color=GREY, ls="--", lw=0.8); ax[2].text(18, blind + 0.0006, "calendar-blind", fontsize=6.5, color=DARK, va="bottom", ha="right")
ax[2].set_xticks(KS); ax[2].set_xlabel("Training states (Exp. 2)"); ax[2].set_ylabel("PR-AUC"); ax[2].set_ylim(0.736, 0.770); ax[2].legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.5)
ax[2].set_title("(c)")
fig.tight_layout(w_pad=1.2)
fig.savefig("Fig7.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig("Fig7.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
Image.open("Fig7.png").convert("RGB").save("Fig7.tif", compression="tiff_lzw", dpi=(600, 600))
print("ok", Image.open("Fig7.png").size)
