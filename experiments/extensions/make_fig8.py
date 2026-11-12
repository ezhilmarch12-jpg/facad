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
R = "results/"
Cg = pd.read_csv(R + "e1_coherence_by_group.csv", index_col=0)["coherence_mean"]
names = {"newyear_regional": "Regional new year", "dussehra": "Dussehra", "diwali": "Diwali", "other": "Other",
         "christmas": "Christmas", "ganesh": "Ganesh", "harvest": "Harvest", "onam": "Onam", "eid": "Eid", "holi": "Holi", "national": "National days"}
Cg = Cg.sort_values()
C2 = pd.read_csv(R + "e2_coherence.csv"); d2 = C2[C2.key.str.startswith("dussehra")].C.mean()
fig, ax = plt.subplots(1, 2, figsize=(W, W * 0.42), gridspec_kw={"width_ratios": [1, 1]})
C3 = pd.read_csv("../experiment3_diwali/results/coherence_log.csv"); C3 = C3[C3.key.str.startswith("diwali") & (C3.n_states >= 17)]; d3 = C3.C.mean()
labels = ["Diwali (Exp. 3 traffic shares)", "Dussehra (Exp. 2 traffic shares)"] + [f"{names[k]} (Exp. 1)" for k in Cg.index]
vals = [d3, d2] + list(Cg.values)
cols = [ORANGE, ORANGE] + [BLUE] * len(Cg)
hatch = ["xx", "//"] + [""] * len(Cg)
y = np.arange(len(vals))
for yi, v, c, h in zip(y, vals, cols, hatch):
    ax[0].barh(yi, v, color=c if not h else "white", edgecolor=c, hatch=h, height=0.62, lw=0.8)
    ax[0].text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=6.5)
ax[0].axvline(0.5, color=DARK, ls="--", lw=0.8); ax[0].text(0.51, len(vals) - 0.4, "τ = 0.5", fontsize=6.5, color=DARK)
ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=6.8); ax[0].set_xlim(0, 1.05); ax[0].grid(axis="y", visible=False)
ax[0].set_xlabel("Coherence index (training data)"); ax[0].set_title("(a)")
E = pd.read_csv(R + "e1_real_events_by_day.csv")
P = E.pivot_table(index=["region", "date"], columns="model", values="pct").reset_index()
ev = {("Delhi", "2022-01"): "Weekend curfew", ("Karnataka", "2022-01"): "Weekend curfew", ("Tamil Nadu", "2022-01"): "Sunday lockdown"}
def kind(r, d):
    d = str(d)
    if d.startswith("2022-03"): return "General strike"
    if d.startswith("2021-12"): return "Cyclone"
    return ev[(r, d[:7])]
P["kind"] = [kind(r, d) for r, d in zip(P.region, P.date)]
mk = {"Weekend curfew": ("o", BLUE), "Sunday lockdown": ("s", BLUE), "General strike": ("^", ORANGE), "Cyclone": ("D", DARK)}
lo = min(P.SeasonalZ.min(), P.FACAD.min()) - 0.01
for k, (m, c) in mk.items():
    s = P[P.kind == k]
    ax[1].scatter(s.SeasonalZ, s.FACAD, marker=m, s=18, facecolor="white" if k == "Sunday lockdown" else c, edgecolor=c, lw=0.8, label=k, zorder=3)
ax[1].plot([lo, 1], [lo, 1], color=GREY, lw=0.8, ls="--")
ax[1].set_xlim(lo, 1.002); ax[1].set_ylim(lo, 1.002)
ax[1].set_xlabel("Event-day rank, calendar-blind (percentile)"); ax[1].set_ylabel("Event-day rank, FACAD (percentile)")
ax[1].legend(frameon=False, loc="lower right"); ax[1].set_title("(b)")
fig.tight_layout(w_pad=1.5)
fig.savefig("Fig8.pdf", bbox_inches="tight", pad_inches=0.02); fig.savefig("Fig8.png", dpi=600, bbox_inches="tight", pad_inches=0.02)
Image.open("Fig8.png").convert("RGB").save("Fig8.tif", compression="tiff_lzw", dpi=(600, 600)); print("ok", lo)
