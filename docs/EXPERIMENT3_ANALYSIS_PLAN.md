# Experiment 3 — confirmatory analysis plan (fixed before the Diwali data were obtained)

Written: 11 November 2026, before the Cloudflare Radar export for 25 October – 11 November 2026 was received.

## Data
Cloudflare Radar NetFlows, India, grouped by state (ADM1), hourly, 25 October – 11 November 2026 (Diwali: 8 November 2026),
converted with the same dataset builder as Experiment 2 (CLR of state shares; state calendars from `holidays` 0.107).

## Methods (no settings changed after seeing the data)
FACAD-a, FACAD-H-a, FACAD-CG, FACAD-H-CG, SeasonalZ-a (calendar-blind), whitelisting, Isolation Forest without and with
calendar features; mobility spread prior for Diwali learned from the 2020–21 Diwali periods of the Google mobility data.
λ = 5; kernel −3 to +2 days; hour blocks 00–06/06–12/12–18/18–24 IST; τ = 0.5; α = 0.05; causal scale with 24 past residuals;
leave-one-state-out with nested calibration; 10 seeds; injection protocol identical to Experiment 2.

## Hypotheses
* H1 (coherence): the mean national coherence of Diwali contexts is at least 0.5, in contrast to Dussehra (0.12).
* H2 (false alarms): FACAD-CG has a lower holiday false-alarm rate than the calendar-blind detector at the matched 5% budget.
* H3 (detection): FACAD-CG has a PR-AUC at least as high as the calendar-blind detector.
* H4 (whitelisting): whitelisting detects none of the anomalies inside festival windows; FACAD-CG detects most of them.

Tests: paired Wilcoxon signed-rank over held-out states, bootstrap 95% CIs of the median difference, two-sided α = 0.05.
All results will be reported whether or not the hypotheses hold.
