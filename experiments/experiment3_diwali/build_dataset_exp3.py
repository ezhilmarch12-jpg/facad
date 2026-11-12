"""Convert a Cloudflare Radar NetFlows (group-by ADM1) JSON export into a tidy research dataset.

Usage: python build_dataset.py <input.json> <out_dir>
"""
import sys, json, os
import numpy as np
import pandas as pd

sys.path.insert(0, "../experiment1_mobility")
import facad_lib
from facad_lib import lunar_features, PRIORITY

# Durga Puja day names used in West Bengal / Odisha / Assam lists belong to the Dussehra group
facad_lib.FESTIVAL_GROUPS["dussehra"] += ["saptami", "ashtami", "mahalaya"]
# fix: the bare keyword "bhai" also matched "Sardar Vallabhbhai Patel's Jayanti"; match the festival name only
facad_lib.FESTIVAL_GROUPS["diwali"] = [k for k in facad_lib.FESTIVAL_GROUPS["diwali"] if k != "bhai"] + ["bhai duj", "bhai dooj", "bhaiya dooj"]
festival_group = facad_lib.festival_group
state_festivals = facad_lib.state_festivals

PRE, POST = 3, 2   # window: 3 days before to 2 days after a festival


def context_for(days, fmap):
    """Nearest festival within -PRE..+POST days. Label 'group@-k' = k days BEFORE, 'group@+k' = k days AFTER.
    Ties: smaller |k| first, then festival priority (Diwali first, 'other' last)."""
    fest = sorted(fmap.items())
    out = {}
    for d in days:
        best = None
        for fd, groups in fest:
            k = (d - fd).days            # negative: festival is ahead (day is before it)
            if -PRE <= k <= POST:
                g = sorted(groups, key=PRIORITY.index)[0]
                key = (abs(k), PRIORITY.index(g))
                if best is None or key < best[0]:
                    best = (key, f"{g}@{k:+d}")
        out[d] = best[1] if best else None
    return out

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)

# GeoNames ADM1 ids -> state (verified against geonames.org/IN/administrative-division-india.html)
GEO = {
    "1278647": ("Andaman and Nicobar Islands", "AN"), "1278629": ("Andhra Pradesh", "AP"),
    "1278341": ("Arunachal Pradesh", "AR"), "1278253": ("Assam", "AS"), "1275715": ("Bihar", "BR"),
    "1444364": ("Chhattisgarh", "CG"), "1274744": ("Chandigarh", "CH"),
    "12165662": ("Dadra and Nagar Haveli and Daman and Diu", "DH"), "1273293": ("Delhi", "DL"),
    "1271157": ("Goa", "GA"), "1270770": ("Gujarat", "GJ"), "1270101": ("Himachal Pradesh", "HP"),
    "1270260": ("Haryana", "HR"), "1444365": ("Jharkhand", "JH"), "1269320": ("Jammu and Kashmir", "JK"),
    "1267701": ("Karnataka", "KA"), "1267254": ("Kerala", "KL"), "12096464": ("Ladakh", "LA"),
    "1265206": ("Lakshadweep", "LD"), "1264418": ("Maharashtra", "MH"), "1263207": ("Meghalaya", "ML"),
    "1263706": ("Manipur", "MN"), "1264542": ("Madhya Pradesh", "MP"), "1262963": ("Mizoram", "MZ"),
    "1262271": ("Nagaland", "NL"), "1261029": ("Odisha", "OD"), "1259223": ("Punjab", "PB"),
    "1259424": ("Puducherry", "PY"), "1258899": ("Rajasthan", "RJ"), "1256312": ("Sikkim", "SK"),
    "1255053": ("Tamil Nadu", "TN"), "1254169": ("Tripura", "TR"), "1254788": ("Telangana", "TS"),
    "1444366": ("Uttarakhand", "UK"), "1253626": ("Uttar Pradesh", "UP"), "1252881": ("West Bengal", "WB"),
}

raw = json.load(open(src))
assert raw.get("success"), raw.get("errors")
res = raw["result"]
main, meta = res["main"], res["meta"]
ts = pd.to_datetime(main["timestamps"], utc=True)
series_keys = [k for k in main if k != "timestamps"]
unknown = [k for k in series_keys if k not in GEO and k != "other"]
assert not unknown, f"unmapped region ids: {unknown}"
for k in series_keys:
    assert len(main[k]) == len(ts), k

# ---- wide table (one row per hour, one column per state code)
wide = pd.DataFrame({"timestamp_utc": ts})
for k in series_keys:
    col = "OTHER" if k == "other" else GEO[k][1]
    wide[col] = pd.to_numeric(main[k], errors="coerce")
state_cols = sorted(c for c in wide.columns if c not in ("timestamp_utc", "OTHER"))
wide = wide[["timestamp_utc"] + state_cols + ["OTHER"]]
wide["row_total_pct"] = wide[state_cols + ["OTHER"]].sum(axis=1).round(6)

# ---- long table
long = wide.melt(id_vars="timestamp_utc", value_vars=state_cols + ["OTHER"], var_name="state_code", value_name="share_pct")
code2 = {v[1]: (k, v[0]) for k, v in GEO.items()}
long["geonameid"] = long.state_code.map(lambda c: code2[c][0] if c in code2 else "")
long["state"] = long.state_code.map(lambda c: code2[c][1] if c in code2 else "Other (all remaining states/UTs)")
ist = long.timestamp_utc.dt.tz_convert("Asia/Kolkata")
long["timestamp_ist"] = ist.dt.strftime("%Y-%m-%d %H:%M")
long["date_ist"] = ist.dt.strftime("%Y-%m-%d")
long["hour_ist"] = ist.dt.hour
long["day_of_week"] = ist.dt.day_name()
long["is_weekend"] = ist.dt.dayofweek.isin([5, 6]).astype(int)
long["timestamp_utc"] = long.timestamp_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")

# ---- calendar layer (state-specific, IST dates)
d0 = pd.Timestamp(long.date_ist.min()) - pd.Timedelta(days=10)
d1 = pd.Timestamp(long.date_ist.max()) + pd.Timedelta(days=10)
days = pd.date_range(d0, d1)
lun = lunar_features(days)
years = sorted(set(days.year))
cal_rows = []
for code in state_cols:
    fmap = state_festivals(code, years)
    import holidays
    names = holidays.India(years=years, subdiv=code)
    ctx = context_for(days, fmap)
    for d in days:
        nm = names.get(d.date())
        cal_rows.append({"state_code": code, "date_ist": d.strftime("%Y-%m-%d"),
                         "holiday_name": nm or "",
                         "festival_group": ";".join(sorted({festival_group(p) for p in nm.split(";")})) if nm else "",
                         "is_holiday": int(bool(nm)),
                         "festival_context": ctx.get(d) or "",
                         "in_festival_window": int(ctx.get(d) is not None)})
calr = pd.DataFrame(cal_rows)
lunr = lun.assign(date_ist=lun.index.strftime("%Y-%m-%d")).reset_index(drop=True)[["date_ist", "tithi", "amavasya", "purnima"]]
long = long.merge(calr, on=["state_code", "date_ist"], how="left").merge(lunr, on="date_ist", how="left")
for c in ["holiday_name", "festival_group", "festival_context"]:
    long[c] = long[c].fillna("")
for c in ["is_holiday", "in_festival_window"]:
    long[c] = long[c].fillna(0).astype(int)

# ---- per-state robust deviation from that state's own same-hour median (descriptive only; one week of data)
long["share_vs_state_hour_median"] = (long.share_pct /
    long.groupby(["state_code", "hour_ist"]).share_pct.transform("median")).round(4)

cols = ["timestamp_utc", "timestamp_ist", "date_ist", "hour_ist", "day_of_week", "is_weekend",
        "state_code", "state", "geonameid", "share_pct", "share_vs_state_hour_median",
        "is_holiday", "holiday_name", "festival_group", "festival_context", "in_festival_window",
        "tithi", "amavasya", "purnima"]
long = long[cols].sort_values(["state_code", "timestamp_utc"]).reset_index(drop=True)

# ---- states lookup
states = pd.DataFrame([{"state_code": c, "state": code2[c][1], "geonameid": code2[c][0],
                        "mean_share_pct": round(wide[c].mean(), 4),
                        "min_share_pct": round(wide[c].min(), 4), "max_share_pct": round(wide[c].max(), 4)}
                       for c in state_cols]).sort_values("mean_share_pct", ascending=False)

stem = "cloudflare_radar_india_state_netflows_hourly_" + ts.min().strftime("%Y%m%d") + "_" + ts.max().strftime("%Y%m%d")
wide_out = wide.copy(); wide_out["timestamp_utc"] = wide_out.timestamp_utc.dt.strftime("%Y-%m-%dT%H:%M:%SZ")
wide_out.insert(1, "timestamp_ist", wide.timestamp_utc.dt.tz_convert("Asia/Kolkata").dt.strftime("%Y-%m-%d %H:%M"))
long.to_csv(f"{out}/{stem}_long.csv", index=False)
wide_out.to_csv(f"{out}/{stem}_wide.csv", index=False)
states.to_csv(f"{out}/states_lookup.csv", index=False)
calr[calr.date_ist.between(long.date_ist.min(), long.date_ist.max())].to_csv(f"{out}/state_calendar_{ts.min():%Y%m%d}_{ts.max():%Y%m%d}.csv", index=False)
json.dump({"source": "Cloudflare Radar API, netflows timeseries_groups, groupBy=ADM1, location=IN",
           "meta": meta, "n_hours": len(ts), "n_regions": len(series_keys),
           "first_utc": str(ts.min()), "last_utc": str(ts.max())},
          open(f"{out}/source_metadata.json", "w"), indent=1, default=str)
print(stem, long.shape, wide.shape)
print("row total range", wide.row_total_pct.min(), wide.row_total_pct.max())
print("hours expected", int((ts.max() - ts.min()) / pd.Timedelta(hours=1)) + 1, "got", len(ts))
print(states.to_string(index=False))
