"""Dussehra prior from Google Community Mobility Reports (India, 2020–2022), corrected festival offsets.

Outputs mobility_prior_dussehra.csv (national FACAD E/V per offset and signal) and
mobility_state_dussehra_effect.csv (state-relative mobility deviation on Dussehra days).
"""
import sys
sys.path.insert(0, "../experiment1_mobility")      # FACAD library with the corrected festival_context()
import numpy as np, pandas as pd
from facad_lib import (STATE_CODES, state_festivals, lunar_features, calendar_frame, festival_context,
                       history_features, dow_frame, festival_window, FACAD, SeasonalZ)

SIG = ["retail and recreation", "grocery and pharmacy", "parks", "transit stations", "workplaces", "residential"]
raw = pd.read_csv("../experiment1_mobility/raw/gmr_countries.csv", low_memory=False)
ind = raw[(raw.country == "India") & raw.region.isin(STATE_CODES)].copy()
ind["date"] = pd.to_datetime(ind.date)
dates = pd.date_range(ind.date.min(), ind.date.max())
lun = lunar_features(dates)
frames = []
for reg, g in ind.groupby("region"):
    g = g.set_index("date").reindex(dates); g["region"] = reg
    for s in SIG:
        g[s] = g[s].interpolate(limit=3)
    cal = calendar_frame(dates, state_festivals(STATE_CODES[reg], sorted(set(dates.year))), lun)
    hist = pd.concat([history_features(g[s]).add_prefix(f"{s}__") for s in SIG], axis=1)
    g = pd.concat([g[["region"] + SIG], hist, cal], axis=1)
    g["ctx"] = festival_context(cal).values
    frames.append(g)
D = pd.concat(frames).rename_axis("date").reset_index()
D = D[D.date >= "2020-03-15"]
fa = FACAD(SIG).fit(D, D.ctx)
rows = []
for k, e in fa.E.items():
    if k.startswith("dussehra@"):
        for s in SIG:
            rows.append({"context": k, "offset": int(k.split("@")[1]), "signal": s, "E": e[s], "V": fa.V[k][s]})
P = pd.DataFrame(rows).sort_values(["offset", "signal"])
P.to_csv("mobility_prior_dussehra.csv", index=False)
print(P.pivot(index="offset", columns="signal", values="V").round(3).to_string())

# state-relative mobility deviation on each state's Dussehra-group festival days (@+0), 2020 and 2021
sz = SeasonalZ(SIG).fit(D)
Z = pd.DataFrame({s: (D[s] - D[f"{s}__sameDOW_med4"]) / D.region.map(sz.scale[s]) for s in SIG})
Z["region"], Z["ctx"], Z["date"] = D.region.values, D.ctx.values, D.date.values
d0 = Z[Z.ctx == "dussehra@+0"]
st = d0.groupby("region")[SIG].median()
rel = st - st.mean()
rel.index = rel.index.map(STATE_CODES)
rel["n_days"] = d0.groupby("region").size().rename(index=STATE_CODES)
rel.round(4).to_csv("mobility_state_dussehra_effect.csv")
print(rel.round(2).to_string())
