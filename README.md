# FACAD — Festival-Aware Calendar-Adjusted anomaly Detection

[![tests](https://github.com/ezhilmarch12-jpg/facad/actions/workflows/tests.yml/badge.svg)](https://github.com/ezhilmarch12-jpg/facad/actions/workflows/tests.yml)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24190133.svg)](https://doi.org/10.5281/zenodo.24190133)
![python](https://img.shields.io/badge/python-3.10%2B-blue) ![license](https://img.shields.io/badge/license-MIT-green)

Anomaly detectors raise false alarms on Indian festivals, which follow lunisolar, state-specific calendars,
and whitelisting festival days hides attacks timed to them. **FACAD** is a plug-in for any residual-based
detector. It learns how each festival context **shifts (E)** and **widens (V)** standardised residuals,
pools these effects across states or regions with shrinkage, removes them before scoring, and sets
**Mondrian conformal thresholds** per calendar context.

This repository holds the reusable `facad` Python package and the code, data and results of the article:

> E. Ezhilarasan. *Festival-Aware Anomaly Detection for Indian Network Traffic: Calendar-Adjusted Residuals,
> Regional Pooling and Context-Conditional Thresholds.* Submitted to *Applied Intelligence* (Springer Nature), 2026.

## The method in one formula

For region r, signal s and time t with festival context c(t) (for example `diwali@-1`, one day before Diwali):

```
z*_rs(t) = ( z_rs(t) − E_c,s ) / V_c,s          score_r(t) = max_s | z*_rs(t) |

E_c,s = w · median(z)           V_c,s = max(1, 1 + w · (MAD(z) / MAD_normal − 1))           w = n / (n + λ)
```

| Variant | What changes | Use when |
| --- | --- | --- |
| `FACAD()` | national shift and spread | festivals move all regions the same way |
| `FACAD(gate=True)` (FACAD-CG) | shift kept only where coherence C ≥ τ | default; safe when regions disagree |
| `FACAD(hierarchical=True)` (FACAD-H) | regional tables blended with national | regions differ (South, North, East, West) |
| `FACAD(mode="spread")` | spread only, never adds false alarms | very little training data |
| `FACAD(spread_prior={...})` (FACAD-HP) | V at least an external prior (e.g. mobility) | cold start |

Coherence `C = |Σ_r n_r sign(m_r)| / Σ_r n_r` measures whether regions move the same way in a context.

## Install

```bash
git clone https://github.com/ezhilmarch12-jpg/facad.git
cd facad
pip install -e .                 # core package (numpy, pandas, scikit-learn)
pip install -e ".[calendar]"     # + Indian holiday lists by state
pip install -e ".[experiments]"  # + everything needed to reproduce the article
```

## Quick start

```python
from facad import FACAD, MondrianConformal, festival_contexts, state_festivals, seasonal_z

ctx = festival_contexts(dates, state_festivals("AP", [2025, 2026]))   # e.g. "diwali@+0", None
z = seasonal_z(log_traffic, period=7, mask_scale=ctx.isna().values)    # any detector's residuals work

model = FACAD(lam=5, gate=True, tau=0.5).fit(z_train, ctx_train, region=state_train, day=date_train)
score = model.score(z_test, ctx_test)

mc = MondrianConformal(alpha=0.05).fit(score_cal, category_cal)        # "holiday" / "window" / "normal"
alarms = mc.predict(score, category_test)
```

`python examples/quickstart.py` runs a complete synthetic example (12 states, two years):

```
calendar-blind  window FPR 0.560  PR-AUC 0.543  in-window recall 0.889
FACAD-CG        window FPR 0.198  PR-AUC 0.899  in-window recall 0.889
```

## Repository layout

```
src/facad/            reusable package
  calendar.py         festival groups, state festival lists, festival contexts, hour blocks
  effects.py          EffectTable (shrinkage, coherence gate, fallbacks) and coherence()
  detector.py         FACAD: national, regional (H), gated (CG), spread prior (HP), ablation modes
  baselines.py        robust z-scores and the calendar-blind SeasonalZ baseline
  conformal.py        MondrianConformal thresholds per calendar category
  metrics.py          matched-budget thresholds, metrics, point-anomaly injection
examples/             quickstart.py
tests/                pytest suite (keyword fix, contexts, shrinkage, gate, Proposition 2, coverage)
experiments/          exact code, derived data and results behind every table and figure in the article
  experiment1_mobility/   Google mobility, 36 states, 2020–22
  experiment2_cloudflare/ Cloudflare Radar hourly state shares, Dussehra 2026
  experiment3_diwali/     confirmatory test, Diwali 2026 (analysis plan fixed before the data)
  extensions/             sensitivity, transfer, coherence, real-event benchmark, Figs. 7–9
  REPRODUCE.md            run order, run times and software versions
docs/                 Experiment 3 analysis plan
```

The `experiments/` folder keeps the code exactly as it produced the article's numbers (`facad_lib.py`); the
`src/facad` package is a clean re-implementation of the same method for reuse. See
[experiments/REPRODUCE.md](experiments/REPRODUCE.md) to rerun everything.

## Data

* **Google COVID-19 Community Mobility Reports** — public; download `gmr_countries.csv` (see
  `experiments/experiment1_mobility/raw/README.txt`). Not redistributed.
* **Cloudflare Radar** NetFlows for India by state, hourly — exported 24 October 2026 (Experiment 2) and
  11 November 2026 (Experiment 3). Licensed **CC BY-NC 4.0** by Cloudflare; derived tables are redistributed
  under the same licence (`DATA_LICENSE.md`).

## Main results (matched 5% false-alarm budget)

| Setting | Calendar-blind | FACAD | Notes |
| --- | --- | --- | --- |
| Exp. 1, mobility, 36 states | festival FPR 0.094, PR-AUC 0.190 | 0.056, 0.230 | p = 0.002 |
| Exp. 2, Cloudflare, Dussehra | PR-AUC 0.746 | 0.785 (FACAD-H-CG) | p < 0.001 |
| Exp. 3, Cloudflare, Diwali (confirmatory) | holiday FPR 0.083, PR-AUC 0.590 | 0.043, 0.611 (FACAD-CG) | FPR p = 0.001; PR-AUC n.s. |

## Citation

Code archive: E. Ezhilarasan (2026). *FACAD: Festival-Aware Calendar-Adjusted anomaly Detection for Indian network traffic* (v1.0.1). Zenodo. https://doi.org/10.5281/zenodo.24190133

See `CITATION.cff`. Licence: code MIT (`LICENSE`); Cloudflare-derived data CC BY-NC 4.0 (`DATA_LICENSE.md`).
