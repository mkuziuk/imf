# trees: synthetic / signal

120 fits on shared split IDs [0, 1, 2, 3, 4]; wall time 280.0 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                        | 0.3540 | 0.0277 |
| ('emd', 'comp', 'lightgbm')                          | 0.4067 | 0.0536 |
| ('emd', 'comp', 'xgboost')                           | 0.3933 | 0.0562 |
| ('emd', 'comp+log_scale', 'extratrees')              | 0.3533 | 0.0348 |
| ('emd', 'comp+log_scale', 'lightgbm')                | 0.3860 | 0.0511 |
| ('emd', 'comp+log_scale', 'xgboost')                 | 0.3587 | 0.0551 |
| ('imf_linear_a1.41', 'comp', 'extratrees')           | 0.6293 | 0.0150 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.7013 | 0.0288 |
| ('imf_linear_a1.41', 'comp', 'xgboost')              | 0.6973 | 0.0583 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees') | 0.6347 | 0.0312 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.6873 | 0.0501 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')    | 0.6653 | 0.0524 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')              | 0.7687 | 0.0421 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.8507 | 0.0477 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                 | 0.8360 | 0.0360 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')    | 0.7627 | 0.0548 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.8460 | 0.0462 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')       | 0.8240 | 0.0348 |
| ('raw', 'comp', 'extratrees')                        | 0.3560 | 0.0293 |
| ('raw', 'comp', 'lightgbm')                          | 0.3367 | 0.0590 |
| ('raw', 'comp', 'xgboost')                           | 0.3300 | 0.0600 |
| ('raw', 'comp+log_scale', 'extratrees')              | 0.3200 | 0.0449 |
| ('raw', 'comp+log_scale', 'lightgbm')                | 0.2900 | 0.0748 |
| ('raw', 'comp+log_scale', 'xgboost')                 | 0.2833 | 0.0796 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b         |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-----------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw              | emd              |     -0.0700 |  -0.2353 |    0.0953 |      1.0000 |            1.0000 |
| comp           | lightgbm     | raw              | imf_linear_a1.41 |     -0.3647 |  -0.4379 |   -0.2914 |      0.0054 |            1.0000 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41    |     -0.5140 |  -0.6369 |   -0.3911 |      0.0101 |            1.0000 |
| comp           | lightgbm     | emd              | imf_linear_a1.41 |     -0.2947 |  -0.4133 |   -0.1760 |      0.0348 |            1.0000 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41    |     -0.4440 |  -0.5859 |   -0.3021 |      0.0242 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.1493 |  -0.2677 |   -0.0309 |      0.2432 |            1.0000 |
| comp           | xgboost      | raw              | emd              |     -0.0633 |  -0.2616 |    0.1349 |      1.0000 |            1.0000 |
| comp           | xgboost      | raw              | imf_linear_a1.41 |     -0.3673 |  -0.4927 |   -0.2420 |      0.0272 |            1.0000 |
| comp           | xgboost      | raw              | irmf_h2_a1.41    |     -0.5060 |  -0.6607 |   -0.3513 |      0.0220 |            1.0000 |
| comp           | xgboost      | emd              | imf_linear_a1.41 |     -0.3040 |  -0.4925 |   -0.1155 |      0.1321 |            1.0000 |
| comp           | xgboost      | emd              | irmf_h2_a1.41    |     -0.4427 |  -0.5731 |   -0.3122 |      0.0198 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.1387 |  -0.2479 |   -0.0295 |      0.2432 |            1.0000 |
| comp           | extratrees   | raw              | emd              |      0.0020 |  -0.0947 |    0.0987 |      1.0000 |            1.0000 |
| comp           | extratrees   | raw              | imf_linear_a1.41 |     -0.2733 |  -0.3126 |   -0.2341 |      0.0015 |            1.0000 |
| comp           | extratrees   | raw              | irmf_h2_a1.41    |     -0.4127 |  -0.4610 |   -0.3643 |      0.0007 |            1.0000 |
| comp           | extratrees   | emd              | imf_linear_a1.41 |     -0.2753 |  -0.3436 |   -0.2071 |      0.0108 |            1.0000 |
| comp           | extratrees   | emd              | irmf_h2_a1.41    |     -0.4147 |  -0.5282 |   -0.3012 |      0.0154 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.1393 |  -0.1966 |   -0.0820 |      0.0348 |            1.0000 |
| comp+log_scale | lightgbm     | raw              | emd              |     -0.0960 |  -0.2180 |    0.0260 |      0.5655 |            1.0000 |
| comp+log_scale | lightgbm     | raw              | imf_linear_a1.41 |     -0.3973 |  -0.4929 |   -0.3017 |      0.0101 |            1.0000 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41    |     -0.5560 |  -0.7618 |   -0.3502 |      0.0310 |            1.0000 |
| comp+log_scale | lightgbm     | emd              | imf_linear_a1.41 |     -0.3013 |  -0.4184 |   -0.1843 |      0.0324 |            1.0000 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41    |     -0.4600 |  -0.6314 |   -0.2886 |      0.0310 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.1587 |  -0.3092 |   -0.0081 |      0.3010 |            1.0000 |
| comp+log_scale | xgboost      | raw              | emd              |     -0.0753 |  -0.1875 |    0.0368 |      0.6780 |            1.0000 |
| comp+log_scale | xgboost      | raw              | imf_linear_a1.41 |     -0.3820 |  -0.4661 |   -0.2979 |      0.0075 |            1.0000 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41    |     -0.5407 |  -0.7389 |   -0.3425 |      0.0310 |            1.0000 |
| comp+log_scale | xgboost      | emd              | imf_linear_a1.41 |     -0.3067 |  -0.4100 |   -0.2033 |      0.0272 |            1.0000 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41    |     -0.4653 |  -0.6229 |   -0.3077 |      0.0272 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.1587 |  -0.2869 |   -0.0304 |      0.2432 |            1.0000 |
| comp+log_scale | extratrees   | raw              | emd              |     -0.0333 |  -0.1313 |    0.0646 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw              | imf_linear_a1.41 |     -0.3147 |  -0.4433 |   -0.1860 |      0.0348 |            1.0000 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41    |     -0.4427 |  -0.5892 |   -0.2961 |      0.0265 |            1.0000 |
| comp+log_scale | extratrees   | emd              | imf_linear_a1.41 |     -0.2813 |  -0.3692 |   -0.1935 |      0.0230 |            1.0000 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41    |     -0.4093 |  -0.5546 |   -0.2640 |      0.0288 |            1.0000 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41    |     -0.1280 |  -0.2150 |   -0.0410 |      0.1653 |            1.0000 |
