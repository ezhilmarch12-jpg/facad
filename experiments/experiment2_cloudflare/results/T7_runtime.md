| item                                                          |   value | unit   |
|:--------------------------------------------------------------|--------:|:-------|
| FACAD-a: learn table from 18 states + score one state (167 h) |   0.187 | s      |
| FACAD-a peak memory for the step above                        |   0.7   | MiB    |
| LightGBM forecaster fit (mean over cached models)             |   0.131 | s      |
| All non-Prophet methods: score one injected series            |   0.154 | s      |
| Full LOSO fold incl. nested calibration, 10 seeds, Prophet    |   9.6   | s      |