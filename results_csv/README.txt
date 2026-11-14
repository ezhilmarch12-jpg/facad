FACAD - downloadable results (CSV)
Article: Festival-Aware Anomaly Detection for Indian Network Traffic (submitted to Applied Intelligence)
Code and data: https://github.com/ezhilmarch12-jpg/facad   Archive: https://doi.org/10.5281/zenodo.24190133

FACAD_All_Results.pdf  all of the below in one PDF (88 pages)
INDEX.csv          one line per file: what it contains and how many rows
article_tables/    every results table of the article (Tables 3-12, A1, A2), exactly as printed
full_results/      the raw result files the tables were computed from
  exp1_mobility/   Experiment 1: Google mobility, 36 states, 2020-22 (10 seeds)
  exp2_dussehra/   Experiment 2: Cloudflare hourly state shares, Dussehra 2026 (19 held-out states x 10 seeds)
  exp3_diwali/     Experiment 3: confirmatory test, Diwali 2026 (hypothesis_tests.csv = H1-H4)
  extensions/      coherence gate, sensitivity analyses, real-event benchmark

Conventions: FPR = false-alarm rate; matched = threshold set for a 5% overall false-alarm budget;
SeasonalZ = calendar-blind baseline; FACAD-CG = coherence-gated FACAD; p = two-sided Wilcoxon signed-rank.
Files are UTF-8 (article tables with BOM so Excel shows symbols such as lambda and minus correctly).
Licence: results CC BY 4.0; results derived from Cloudflare Radar data follow CC BY-NC 4.0.
