| method                 | vs                        | metric           |   median_diff |   ci95_low |   ci95_high |   p_value |   n_states |
|:-----------------------|:--------------------------|:-----------------|--------------:|-----------:|------------:|----------:|-----------:|
| FACAD-a                | SeasonalZ-a               | FPR_holiday      |         0.006 |      0     |       0.024 |  0.00632  |         19 |
| FACAD-a                | SeasonalZ-a               | PR_AUC           |         0.001 |     -0.008 |       0.026 |  0.465    |         19 |
| FACAD-a                | SeasonalZ-a               | F1               |        -0.004 |     -0.02  |       0.015 |  0.953    |         19 |
| FACAD-a                | SeasonalZ-a               | recall_in_window |        -0.012 |     -0.033 |       0.007 |  0.275    |         19 |
| FACAD-a                | GBM-c (blind)             | FPR_holiday      |        -0.02  |     -0.038 |       0.001 |  0.156    |         19 |
| FACAD-a                | GBM-c (blind)             | PR_AUC           |         0.256 |      0.207 |       0.321 |  3.81e-06 |         19 |
| FACAD-a                | GBM-c (blind)             | F1               |         0.224 |      0.199 |       0.291 |  3.81e-06 |         19 |
| FACAD-a                | GBM-c (blind)             | recall_in_window |         0.323 |      0.277 |       0.377 |  3.81e-06 |         19 |
| FACAD-a                | GBM-cal (calendar inputs) | FPR_holiday      |        -0.017 |     -0.029 |       0.003 |  0.352    |         19 |
| FACAD-a                | GBM-cal (calendar inputs) | PR_AUC           |         0.234 |      0.206 |       0.328 |  3.81e-06 |         19 |
| FACAD-a                | GBM-cal (calendar inputs) | F1               |         0.21  |      0.197 |       0.304 |  3.81e-06 |         19 |
| FACAD-a                | GBM-cal (calendar inputs) | recall_in_window |         0.336 |      0.289 |       0.376 |  3.81e-06 |         19 |
| FACAD-a                | IF                        | FPR_holiday      |         0.012 |      0     |       0.047 |  0.00763  |         19 |
| FACAD-a                | IF                        | PR_AUC           |        -0.02  |     -0.037 |      -0.012 |  0.0124   |         19 |
| FACAD-a                | IF                        | F1               |        -0.031 |     -0.056 |      -0.022 |  0.00169  |         19 |
| FACAD-a                | IF                        | recall_in_window |        -0.023 |     -0.04  |       0     |  0.0495   |         19 |
| FACAD-a                | IF+cal                    | FPR_holiday      |         0     |     -0.014 |       0.012 |  0.638    |         19 |
| FACAD-a                | IF+cal                    | PR_AUC           |        -0.004 |     -0.065 |       0.094 |  0.738    |         19 |
| FACAD-a                | IF+cal                    | F1               |        -0.02  |     -0.067 |       0.075 |  0.709    |         19 |
| FACAD-a                | IF+cal                    | recall_in_window |         0.13  |      0.101 |       0.199 |  7.63e-06 |         19 |
| FACAD-a                | Prophet-hol               | FPR_holiday      |         0     |     -0.027 |       0.026 |  0.948    |         19 |
| FACAD-a                | Prophet-hol               | PR_AUC           |        -0.095 |     -0.133 |      -0.015 |  0.00117  |         19 |
| FACAD-a                | Prophet-hol               | F1               |        -0.067 |     -0.119 |      -0.044 |  9.54e-05 |         19 |
| FACAD-a                | Prophet-hol               | recall_in_window |        -0.025 |     -0.128 |       0.037 |  0.104    |         19 |
| FACAD-a                | Whitelist-a               | FPR_holiday      |         0.029 |      0.013 |       0.075 |  0.000655 |         19 |
| FACAD-a                | Whitelist-a               | PR_AUC           |         0.001 |     -0.008 |       0.026 |  0.465    |         19 |
| FACAD-a                | Whitelist-a               | F1               |         0.148 |      0.066 |       0.279 |  0.00117  |         19 |
| FACAD-a                | Whitelist-a               | recall_in_window |         0.71  |      0.611 |       0.785 |  3.81e-06 |         19 |
| FACAD-a                | FACAD-prior (mobility)    | FPR_holiday      |         0.029 |      0.013 |       0.074 |  0.000655 |         19 |
| FACAD-a                | FACAD-prior (mobility)    | PR_AUC           |         0.012 |     -0.004 |       0.031 |  0.169    |         19 |
| FACAD-a                | FACAD-prior (mobility)    | F1               |         0.037 |     -0.018 |       0.058 |  0.104    |         19 |
| FACAD-a                | FACAD-prior (mobility)    | recall_in_window |         0.08  |      0.004 |       0.102 |  0.00284  |         19 |
| FACAD-H-a              | FACAD-a                   | FPR_holiday      |         0     |     -0.006 |       0.002 |  0.394    |         19 |
| FACAD-H-a              | FACAD-a                   | PR_AUC           |         0.019 |      0.013 |       0.034 |  7.63e-06 |         19 |
| FACAD-H-a              | FACAD-a                   | F1               |         0.026 |      0.012 |       0.042 |  0.000126 |         19 |
| FACAD-H-a              | FACAD-a                   | recall_in_window |         0.016 |      0.011 |       0.047 |  0.00717  |         19 |
| FACAD-H-a              | SeasonalZ-a               | FPR_holiday      |         0.003 |      0     |       0.014 |  0.0268   |         19 |
| FACAD-H-a              | SeasonalZ-a               | PR_AUC           |         0.037 |      0.002 |       0.051 |  0.00141  |         19 |
| FACAD-H-a              | SeasonalZ-a               | F1               |         0.02  |      0.004 |       0.035 |  0.00714  |         19 |
| FACAD-H-a              | SeasonalZ-a               | recall_in_window |         0.008 |     -0.009 |       0.05  |  0.306    |         19 |
| FACAD-prior (mobility) | SeasonalZ-a               | FPR_holiday      |        -0.022 |     -0.036 |      -0.004 |  0.000655 |         19 |
| FACAD-prior (mobility) | SeasonalZ-a               | PR_AUC           |        -0.007 |     -0.035 |       0.007 |  0.49     |         19 |
| FACAD-prior (mobility) | SeasonalZ-a               | F1               |        -0.006 |     -0.03  |       0.015 |  0.145    |         19 |
| FACAD-prior (mobility) | SeasonalZ-a               | recall_in_window |        -0.075 |     -0.102 |      -0.009 |  0.000536 |         19 |
| FACAD-prior (mobility) | Whitelist-a               | FPR_holiday      |         0     |      0     |       0     |  0.109    |         19 |
| FACAD-prior (mobility) | Whitelist-a               | PR_AUC           |        -0.007 |     -0.035 |       0.007 |  0.49     |         19 |
| FACAD-prior (mobility) | Whitelist-a               | F1               |         0.114 |      0.071 |       0.258 |  0.0108   |         19 |
| FACAD-prior (mobility) | Whitelist-a               | recall_in_window |         0.6   |      0.555 |       0.725 |  3.81e-06 |         19 |
| FACAD-b                | SeasonalZ-b               | FPR_holiday      |         0     |     -0.004 |       0.006 |  0.51     |         19 |
| FACAD-b                | SeasonalZ-b               | PR_AUC           |        -0.007 |     -0.011 |       0.001 |  0.0955   |         19 |
| FACAD-b                | SeasonalZ-b               | F1               |        -0.004 |     -0.019 |       0     |  0.0222   |         19 |
| FACAD-b                | SeasonalZ-b               | recall_in_window |        -0.007 |     -0.012 |       0     |  0.0552   |         19 |
| FACAD-c                | GBM-c (blind)             | FPR_holiday      |         0.001 |     -0.004 |       0.004 |  0.984    |         19 |
| FACAD-c                | GBM-c (blind)             | PR_AUC           |         0.016 |      0.007 |       0.022 |  0.00169  |         19 |
| FACAD-c                | GBM-c (blind)             | F1               |         0.021 |      0.006 |       0.031 |  0.0141   |         19 |
| FACAD-c                | GBM-c (blind)             | recall_in_window |        -0.002 |     -0.022 |       0.013 |  0.472    |         19 |
| FACAD-c                | GBM-cal (calendar inputs) | FPR_holiday      |         0.004 |     -0.004 |       0.005 |  0.557    |         19 |
| FACAD-c                | GBM-cal (calendar inputs) | PR_AUC           |         0.001 |     -0.006 |       0.023 |  0.21     |         19 |
| FACAD-c                | GBM-cal (calendar inputs) | F1               |         0.019 |     -0.009 |       0.032 |  0.0258   |         19 |
| FACAD-c                | GBM-cal (calendar inputs) | recall_in_window |        -0.014 |     -0.026 |       0.01  |  0.258    |         19 |