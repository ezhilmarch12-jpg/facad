"""FACAD in 60 lines: synthetic daily traffic for 12 states with festival effects.

Run:  python examples/quickstart.py
"""
import numpy as np
import pandas as pd

from facad import FACAD, MondrianConformal, evaluate, festival_contexts, inject_point_anomalies, seasonal_z, threshold_at_fpr

rng = np.random.default_rng(0)
states = [f"S{i:02d}" for i in range(12)]
dates = pd.date_range("2024-01-01", "2025-12-31", freq="D")
fest = {d: "diwali" for d in ["2024-10-31", "2025-10-20"]} | {d: "harvest" for d in ["2024-01-15", "2025-01-14"]} \
     | {d: "holi" for d in ["2024-03-25", "2025-03-14"]}
ctx = festival_contexts(dates, fest)

rows = []
for s in states:
    sign = rng.choice([-1, 1], p=[0.1, 0.9])                       # most states rise, a few fall
    week = 1 + 0.15 * np.sin(2 * np.pi * dates.day_of_week / 7)
    y = np.asarray(100 * week * np.exp(rng.normal(0, 0.03, len(dates))))
    for i, c in enumerate(ctx):
        if c:                                                         # festival effect: shift + extra spread
            off = int(c.split("@")[1])
            y[i] *= np.exp(sign * 0.25 * np.exp(-abs(off)) + rng.normal(0, 0.08))
    z = seasonal_z(pd.Series(np.log(y), index=dates), period=7, mask_scale=ctx.isna().values)
    rows.append(pd.DataFrame({"state": s, "date": dates, "ctx": ctx.values, "z": z.values}))
D = pd.concat(rows, ignore_index=True).dropna(subset=["z"])

train = D[D.date < "2025-01-01"].reset_index(drop=True)
test = D[D.date >= "2025-01-01"].reset_index(drop=True)
z_test, label = inject_point_anomalies(test[["z"]], n=120, rng=rng, magnitude=(4, 7))
win = test.ctx.notna().values

blind = z_test.z.abs().values
model = FACAD(lam=5, gate=True, tau=0.5).fit(train[["z"]], train.ctx, region=train.state, day=train.date)
facad = model.score(z_test, test.ctx).values

for name, sc in [("calendar-blind", blind), ("FACAD-CG", facad)]:
    thr = threshold_at_fpr(sc, label, 0.05)                         # matched 5% false-alarm budget
    m = evaluate(sc, label, win, thr)
    print(f"{name:15s} window FPR {m['FPR_window']:.3f}  PR-AUC {m['PR_AUC']:.3f}  in-window recall {m['recall_in_window']:.3f}")

# context-conditional thresholds from a clean calibration slice
cal = train[train.date >= "2024-07-01"]
cat = np.where(cal.ctx.notna(), "window", "normal")
mc = MondrianConformal(alpha=0.05).fit(model.score(cal[["z"]], cal.ctx), cat)
print("Mondrian thresholds:", {str(k): round(v, 2) for k, v in mc.thresholds.items()})
print(model.national.to_frame().query("level == 'ctx'").head(6).round(3).to_string(index=False))
