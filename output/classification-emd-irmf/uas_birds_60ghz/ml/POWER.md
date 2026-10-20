# Detectable balanced-accuracy differences

GD60 IRMF minus EMD. All values below are percentage points. se_corrected uses the observed paired fold variance and mean held-out pose-group ratio, with 14 df. threshold is the difference needed for significance at the stated alpha, conditional on that variance. mde80 solves 80% power under the corresponding noncentral-t model.

Nominal alpha is .05; Holm first step uses .05/120 and is a conservative threshold. The actual step-down rank depends on all tests. These plug-in sensitivities are not evidence of equivalence or a new-session power forecast. Zero paired variance leaves planning MDE undefined.

| task       | feature_set    | classifier     |   mean_diff |   se_corrected |   threshold_nominal |   mde80_nominal |   threshold_holm_first_step |   mde80_holm_first_step |
|:-----------|:---------------|:---------------|------------:|---------------:|--------------------:|----------------:|----------------------------:|------------------------:|
| bird_drone | comp           | lightgbm       |       2.753 |          1.099 |               2.356 |           3.311 |                       5.048 |                   6.175 |
| bird_drone | comp           | xgboost        |       2.073 |          1.150 |               2.467 |           3.465 |                       5.284 |                   6.464 |
| bird_drone | comp           | extratrees     |       2.816 |          1.399 |               3.001 |           4.216 |                       6.429 |                   7.864 |
| bird_drone | comp+log_scale | lightgbm       |       1.753 |          1.029 |               2.207 |           3.101 |                       4.728 |                   5.784 |
| bird_drone | comp+log_scale | xgboost        |       1.399 |          0.889 |               1.907 |           2.679 |                       4.085 |                   4.997 |
| bird_drone | comp+log_scale | extratrees     |       2.340 |          1.359 |               2.914 |           4.094 |                       6.243 |                   7.636 |
| bird_drone | comp           | minirocket     |       0.535 |          1.124 |               2.411 |           3.387 |                       5.165 |                   6.318 |
| bird_drone | comp+log_scale | minirocket     |       0.705 |          1.003 |               2.152 |           3.023 |                       4.609 |                   5.638 |
| bird_drone | comp           | multirocket    |      -0.170 |          1.298 |               2.783 |           3.910 |                       5.962 |                   7.293 |
| bird_drone | comp+log_scale | multirocket    |      -0.538 |          0.942 |               2.020 |           2.838 |                       4.327 |                   5.293 |
| bird_drone | comp           | compact_resnet |       0.028 |          1.359 |               2.915 |           4.095 |                       6.245 |                   7.639 |
| bird_drone | comp+log_scale | compact_resnet |       0.736 |          1.749 |               3.752 |           5.271 |                       8.038 |                   9.832 |
| target3    | comp           | lightgbm       |       2.118 |          1.100 |               2.360 |           3.315 |                       5.055 |                   6.183 |
| target3    | comp           | xgboost        |       2.118 |          1.616 |               3.466 |           4.870 |                       7.425 |                   9.083 |
| target3    | comp           | extratrees     |       3.542 |          1.289 |               2.765 |           3.884 |                       5.923 |                   7.245 |
| target3    | comp+log_scale | lightgbm       |       1.736 |          1.192 |               2.557 |           3.593 |                       5.479 |                   6.702 |
| target3    | comp+log_scale | xgboost        |       1.875 |          1.651 |               3.541 |           4.975 |                       7.586 |                   9.280 |
| target3    | comp+log_scale | extratrees     |       3.472 |          1.397 |               2.995 |           4.208 |                       6.417 |                   7.849 |
| target3    | comp           | minirocket     |       3.090 |          1.823 |               3.911 |           5.494 |                       8.378 |                  10.248 |
| target3    | comp+log_scale | minirocket     |       2.917 |          1.958 |               4.199 |           5.899 |                       8.995 |                  11.003 |
| target3    | comp           | multirocket    |       2.847 |          2.791 |               5.987 |           8.411 |                      12.825 |                  15.688 |
| target3    | comp+log_scale | multirocket    |       2.188 |          3.059 |               6.561 |           9.217 |                      14.054 |                  17.192 |
| target3    | comp           | compact_resnet |       5.417 |          2.623 |               5.626 |           7.904 |                      12.052 |                  14.743 |
| target3    | comp+log_scale | compact_resnet |       5.521 |          2.325 |               4.987 |           7.006 |                      10.683 |                  13.068 |

All requested contrasts, including capacity/H checks, are in power.csv. RUN_NOTES.md gives the formula and numerical integration method.
