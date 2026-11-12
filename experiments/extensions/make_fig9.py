import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
MM = 1 / 25.4; W = 174 * MM
plt.rcParams.update({"font.family": "Liberation Sans", "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 6.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.5,
                     "axes.linewidth": 0.6, "axes.edgecolor": "#6b6a66", "pdf.fonttype": 42,
                     "axes.titlelocation": "left", "axes.titleweight": "bold"})
BLUE, ORANGE, DARK, GREY = "#2a78d6", "#eb6834", "#52514e", "#b9b8b2"
R = pd.read_csv("../experiment3_diwali/results/all_runs.csv")
M = R[R["mode"] == "matched"]
st = M.groupby(["method", "state"])[["FPR_holiday", "recall_in_window"]].mean()
b = st.loc["SeasonalZ-a"].FPR_holiday; g = st.loc["FACAD-CG"].FPR_holiday
order = b.sort_values().index
fig, ax = plt.subplots(1, 2, figsize=(W, W * 0.42), gridspec_kw={"width_ratios": [1.25, 1]})
y = np.arange(len(order))
for yi, s in zip(y, order):
    ax[0].plot([b[s], g[s]], [yi, yi], color=GREY, lw=0.8, zorder=1)
ax[0].scatter(b[order], y, s=16, color=GREY, edgecolor=DARK, lw=0.5, label="Calendar-blind", zorder=2)
ax[0].scatter(g[order], y, s=16, color=BLUE, label="FACAD-CG", zorder=3)
ax[0].set_yticks(y); ax[0].set_yticklabels(order, fontsize=6.5); ax[0].grid(axis="y", visible=False)
ax[0].set_xlabel("Holiday-hour false-alarm rate (matched 5% budget)"); ax[0].set_title("(a) Per held-out state")
ax[0].legend(frameon=False, loc="lower right")
meth = [("SeasonalZ-a", "Calendar-blind"), ("Whitelist-a", "Whitelist"), ("IF+cal", "IF + calendar"),
        ("FACAD-prior (mobility)", "FACAD-prior"), ("FACAD-a", "FACAD-a"), ("FACAD-CG", "FACAD-CG")]
mm = M.groupby("method")[["FPR_holiday", "recall_in_window"]].mean()
for k, lab in meth:
    x, yv = mm.loc[k, "FPR_holiday"], mm.loc[k, "recall_in_window"]
    c = BLUE if k.startswith("FACAD") else (ORANGE if k != "SeasonalZ-a" else DARK)
    ax[1].scatter(x, yv, s=22, color=c, zorder=3)
    dx, ha = (0.012, "left") if x < 0.3 else (-0.012, "right")
    dy = {"FACAD-a": -0.04, "FACAD-CG": 0.045, "SeasonalZ-a": 0.0}.get(k, 0.0)
    if k == "FACAD-CG": dx, ha = -0.005, "right"
    if k == "SeasonalZ-a": dx = 0.015
    ax[1].text(x + dx, yv + dy, lab, fontsize=6.5, va="center", ha=ha)
ax[1].set_xlim(-0.07, 0.42); ax[1].set_ylim(-0.05, 0.75)
ax[1].set_xlabel("Holiday-hour false-alarm rate"); ax[1].set_ylabel("Recall inside festival windows")
ax[1].set_title("(b) Trade-off (mean of 19 states)")
fig.tight_layout(w_pad=1.5)
fig.savefig("Fig9.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig("Fig9.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
Image.open("Fig9.png").convert("RGB").save("Fig9.tif", compression="tiff_lzw", dpi=(600, 600)); print("ok")
