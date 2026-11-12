# FACAD — reproducibility package

Code, derived data and results for the article

> Ezhilarasan E. *Festival-Aware Anomaly Detection for Indian Network Traffic: Calendar-Adjusted Residuals,
> Regional Pooling and Context-Conditional Thresholds.* Submitted to *Applied Intelligence* (Springer).

FACAD (Festival-Aware Calendar-Adjusted Detection) learns how each Indian festival shifts (E) and widens (V) the
standardised residuals of any anomaly detector, pools these effects across states or regions with shrinkage, removes
them before scoring, and sets Mondrian conformal thresholds per calendar context.

## Contents

| Folder | What it reproduces |
| --- | --- |
| `experiment1_mobility/` | Experiment 1 (Google mobility, 36 states, 2020–22): Tables 3 and 6, Figs. 2–3, Table A1 rows for Experiment 1. `facad_lib.py` holds the calendar encoder, detectors, injection and metrics. |
| `experiment2_cloudflare/` | Experiment 2 (Cloudflare Radar hourly state traffic shares, 17–24 Oct 2026): Tables 4–6, Figs. 4–6, Table A1 rows for Experiment 2. Includes the derived dataset and the mobility spread prior. |
| `experiment3_diwali/` | Experiment 3, the confirmatory Diwali test (Cloudflare Radar hourly state traffic shares, 24 Oct – 11 Nov 2026): Tables 10–11, Fig. 9, Table A1 rows for Experiment 3. Contains the analysis plan written before the data were obtained (`EXPERIMENT3_ANALYSIS_PLAN.md`, SHA-256 `77c9642d8523ab0bf1bbde6c3a1df73fac2562536dd6ae022dc33d14497bb8dd`), the raw Radar export (SHA-256 `6cdb7a31d62d066c8c5afce8146c45b6b88c5492af757ac77638a60180810b1f`), the dataset builder, the mobility spread prior, the experiment script and the hypothesis tests. |
| `extensions/` | Leave-one-state-out transfer (Experiment 1), sensitivity to λ and kernel width, anomaly-size curves, number of training states and the exploratory FACAD-HP variant (Table 8, Fig. 7); the festival coherence index, coherence-gated FACAD (FACAD-CG, FACAD-H-CG) and the real-event benchmark of 25 documented disruption days (Tables 4 and 9, Fig. 8); new rows of Table A1. |

Each folder has a `results/` subfolder with the outputs used in the article.

## Data

* **Real-event labels** (`extensions/ext3_mobility_coherence_real_events.py`, list `EVENTS`): Omicron weekend curfews in Delhi and Karnataka, Sunday lockdowns in Tamil Nadu (January 2022), the Bharat Bandh general strike in Kerala, West Bengal and Tamil Nadu (28–29 March 2022) and Cyclone Jawad (dates from the IMD report), with sources cited in the article.

* **Google COVID-19 Community Mobility Reports** (public). Download `gmr_countries.csv` from the GitHub mirror
  `ActiveConclusion/COVID19_mobility` and place it in `experiment1_mobility/raw/`. File used in the article:
  SHA-256 `9f896fc84055d85c4948660535d3d682f6282ca2c1ac798a554a46c4bc401658`. The raw file is not redistributed here.
* **Cloudflare Radar** NetFlows for India grouped by state (ADM1), hourly, 17–24 October 2026, exported from
  radar.cloudflare.com on 24 October 2026. Licensed **CC BY-NC 4.0** by Cloudflare. The derived long-format table
  `experiment2_cloudflare/cloudflare_radar_india_state_netflows_hourly_20261017_20261024_long.csv` (shares, CLR inputs,
  calendar columns) is redistributed under the same licence.

## How to run

```
python -m pip install pandas numpy scipy scikit-learn lightgbm prophet holidays==0.107 ephem matplotlib tabulate jupyter
cd experiment1_mobility && python FACAD_experiment1_mobility.py        # about 5 min on one CPU
cd ../experiment2_cloudflare && python FACAD_cloudflare_experiments.py  # about 4–5 min
cd ../extensions && python ext1_mobility_sensitivity_transfer.py      # about 2 min
python ext2_cloudflare_sensitivity_combined.py                         # about 5 min
python make_fig7.py
python ext3_mobility_coherence_real_events.py                         # about 10 min
python ext4_cloudflare_coherence.py                                    # about 3 min
python make_fig8.py
```

Versions used: Python 3.13, pandas 3.0.5, numpy 2.5.3, scipy 1.18.1, scikit-learn 1.9.1, LightGBM 4.7.0,
Prophet 1.5.0, holidays 0.107, ephem 4.2.1, matplotlib 3.11.2. Seeds 0–9 are fixed in every script.

## Correction note

An early implementation swapped "days before" and "days after" a festival in the context function. All results in
this package and in the article come from the corrected code (`facad_lib.festival_context`).

## Licences

* Code: MIT licence (`LICENSE`).
* Derived Cloudflare data: CC BY-NC 4.0 (see `DATA_LICENSE.md`).
* Results tables and figures: CC BY 4.0.

## Citation

See `CITATION.cff`. Please cite the article once published.

## Experiment 3 (confirmatory, Diwali 2026)

```
cd experiment3_diwali
python build_dataset_exp3.py raw/cloudflare-radar_get-netflows-timeseries-group-by-adm1_20261024-20261111.json data
python prep_mobility_prior_diwali.py      # needs ../experiment1_mobility/raw/gmr_countries.csv
python FACAD_experiment3_diwali.py        # about 10–12 min on one CPU
python test_hypotheses.py                 # H1–H4 as fixed in the analysis plan
```

The raw export was obtained on 11 November 2026 (10:03 UTC) from the Cloudflare Radar data explorer (NetFlows, India, grouped by ADM1, hourly) and is
licensed CC BY-NC 4.0 by Cloudflare. The only change made after the data were seen is the Diwali keyword fix described in
the article (the keyword "bhai" also matched "Sardar Vallabhbhai Patel's Jayanti"); `build_dataset_exp3.py` applies it.

`experiment1_mobility/keyword_fix_rerun/` holds the Experiment 1 results re-run with the same keyword fix; no headline
value changed and no reported difference moved by more than 0.004.

