# esc50_phase3_trees: esc50_animals / bird

450 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 469.4 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.7198 | 0.0460 |
| ('emd', 'comp', 'lightgbm')                            | 0.7581 | 0.0418 |
| ('emd', 'comp', 'xgboost')                             | 0.7516 | 0.0433 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.7225 | 0.0504 |
| ('emd', 'comp+log_scale', 'lightgbm')                  | 0.7545 | 0.0412 |
| ('emd', 'comp+log_scale', 'xgboost')                   | 0.7507 | 0.0463 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.7468 | 0.0425 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')               | 0.7821 | 0.0436 |
| ('imf_linear_a1.41', 'comp', 'xgboost')                | 0.7728 | 0.0545 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.7521 | 0.0416 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')     | 0.7835 | 0.0451 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')      | 0.7746 | 0.0561 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.7530 | 0.0350 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                  | 0.7807 | 0.0346 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                   | 0.7747 | 0.0338 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.7513 | 0.0380 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')        | 0.7765 | 0.0401 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')         | 0.7704 | 0.0378 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.7527 | 0.0390 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')             | 0.7788 | 0.0351 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')              | 0.7743 | 0.0342 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.7540 | 0.0390 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.7762 | 0.0369 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.7708 | 0.0361 |
| ('raw', 'comp', 'extratrees')                          | 0.7266 | 0.0521 |
| ('raw', 'comp', 'lightgbm')                            | 0.7386 | 0.0489 |
| ('raw', 'comp', 'xgboost')                             | 0.7303 | 0.0531 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.7343 | 0.0521 |
| ('raw', 'comp+log_scale', 'lightgbm')                  | 0.7319 | 0.0461 |
| ('raw', 'comp+log_scale', 'xgboost')                   | 0.7319 | 0.0577 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw              | emd                |     -0.0194 |  -0.0401 |    0.0013 |      1.0000 |            0.0702 |
| comp           | lightgbm     | raw              | imf_linear_a1.41   |     -0.0435 |  -0.0932 |    0.0063 |      1.0000 |            0.0846 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41      |     -0.0420 |  -0.0813 |   -0.0028 |      1.0000 |            0.0305 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.0401 |  -0.0839 |    0.0036 |      1.0000 |            0.0702 |
| comp           | lightgbm     | emd              | imf_linear_a1.41   |     -0.0240 |  -0.0625 |    0.0144 |      1.0000 |            0.5962 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0226 |  -0.0557 |    0.0105 |      1.0000 |            0.1666 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0207 |  -0.0554 |    0.0140 |      1.0000 |            0.6895 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0014 |  -0.0304 |    0.0333 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0033 |  -0.0240 |    0.0306 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0019 |  -0.0106 |    0.0144 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw              | emd                |     -0.0213 |  -0.0450 |    0.0024 |      1.0000 |            0.0702 |
| comp           | xgboost      | raw              | imf_linear_a1.41   |     -0.0425 |  -0.0869 |    0.0019 |      1.0000 |            0.0557 |
| comp           | xgboost      | raw              | irmf_h2_a1.41      |     -0.0444 |  -0.0838 |   -0.0050 |      1.0000 |            0.0073 |
| comp           | xgboost      | raw              | irmf_h2_a1.41_conv |     -0.0440 |  -0.0829 |   -0.0051 |      1.0000 |            0.0073 |
| comp           | xgboost      | emd              | imf_linear_a1.41   |     -0.0212 |  -0.0624 |    0.0200 |      1.0000 |            1.0000 |
| comp           | xgboost      | emd              | irmf_h2_a1.41      |     -0.0231 |  -0.0559 |    0.0098 |      1.0000 |            0.5276 |
| comp           | xgboost      | emd              | irmf_h2_a1.41_conv |     -0.0227 |  -0.0544 |    0.0090 |      1.0000 |            0.3094 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0019 |  -0.0433 |    0.0395 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0015 |  -0.0366 |    0.0336 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0004 |  -0.0113 |    0.0121 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw              | emd                |      0.0068 |  -0.0451 |    0.0588 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw              | imf_linear_a1.41   |     -0.0202 |  -0.0909 |    0.0505 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw              | irmf_h2_a1.41      |     -0.0264 |  -0.0803 |    0.0275 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw              | irmf_h2_a1.41_conv |     -0.0261 |  -0.0823 |    0.0300 |      1.0000 |            1.0000 |
| comp           | extratrees   | emd              | imf_linear_a1.41   |     -0.0270 |  -0.0632 |    0.0092 |      1.0000 |            0.1666 |
| comp           | extratrees   | emd              | irmf_h2_a1.41      |     -0.0332 |  -0.0612 |   -0.0053 |      1.0000 |            0.0171 |
| comp           | extratrees   | emd              | irmf_h2_a1.41_conv |     -0.0329 |  -0.0591 |   -0.0068 |      1.0000 |            0.0171 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0062 |  -0.0307 |    0.0183 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0059 |  -0.0271 |    0.0153 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0003 |  -0.0170 |    0.0176 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw              | emd                |     -0.0227 |  -0.0397 |   -0.0057 |      0.7500 |            0.0073 |
| comp+log_scale | lightgbm     | raw              | imf_linear_a1.41   |     -0.0517 |  -0.0997 |   -0.0037 |      1.0000 |            0.0104 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41      |     -0.0446 |  -0.0863 |   -0.0029 |      1.0000 |            0.0702 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.0443 |  -0.0850 |   -0.0037 |      1.0000 |            0.0231 |
| comp+log_scale | lightgbm     | emd              | imf_linear_a1.41   |     -0.0290 |  -0.0692 |    0.0113 |      1.0000 |            0.3094 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0219 |  -0.0579 |    0.0140 |      1.0000 |            0.5276 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0216 |  -0.0552 |    0.0119 |      1.0000 |            0.6895 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0071 |  -0.0222 |    0.0363 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0074 |  -0.0182 |    0.0330 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0003 |  -0.0103 |    0.0109 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw              | emd                |     -0.0188 |  -0.0374 |   -0.0002 |      1.0000 |            0.0557 |
| comp+log_scale | xgboost      | raw              | imf_linear_a1.41   |     -0.0427 |  -0.0882 |    0.0028 |      1.0000 |            0.0846 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41      |     -0.0385 |  -0.0801 |    0.0031 |      1.0000 |            0.0231 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41_conv |     -0.0389 |  -0.0794 |    0.0016 |      1.0000 |            0.0231 |
| comp+log_scale | xgboost      | emd              | imf_linear_a1.41   |     -0.0239 |  -0.0654 |    0.0177 |      1.0000 |            0.6895 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41      |     -0.0197 |  -0.0546 |    0.0153 |      1.0000 |            0.7416 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41_conv |     -0.0201 |  -0.0518 |    0.0117 |      1.0000 |            0.7416 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0042 |  -0.0301 |    0.0386 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0038 |  -0.0286 |    0.0362 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0004 |  -0.0120 |    0.0112 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw              | emd                |      0.0118 |  -0.0456 |    0.0692 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw              | imf_linear_a1.41   |     -0.0178 |  -0.0876 |    0.0521 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41      |     -0.0170 |  -0.0768 |    0.0428 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41_conv |     -0.0198 |  -0.0786 |    0.0391 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | emd              | imf_linear_a1.41   |     -0.0296 |  -0.0646 |    0.0055 |      1.0000 |            0.1343 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41      |     -0.0288 |  -0.0560 |   -0.0015 |      1.0000 |            0.0305 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41_conv |     -0.0315 |  -0.0577 |   -0.0054 |      1.0000 |            0.0231 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0008 |  -0.0171 |    0.0187 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0020 |  -0.0203 |    0.0163 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0028 |  -0.0101 |    0.0046 |      1.0000 |            1.0000 |
