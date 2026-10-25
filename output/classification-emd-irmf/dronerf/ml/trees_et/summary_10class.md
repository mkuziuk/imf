# dronerf_phase3_trees_et: dronerf / 10class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 763.3 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.4822 | 0.0159 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.5356 | 0.0106 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.6704 | 0.0158 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.6758 | 0.0155 |
| ('imf_linear_a2', 'comp', 'extratrees')                | 0.6629 | 0.0170 |
| ('imf_linear_a2', 'comp+log_scale', 'extratrees')      | 0.6674 | 0.0196 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.6173 | 0.0165 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.6289 | 0.0160 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.6187 | 0.0193 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.6351 | 0.0161 |
| ('irmf_h2_a2_conv', 'comp', 'extratrees')              | 0.6080 | 0.0183 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'extratrees')    | 0.6293 | 0.0148 |
| ('raw', 'comp', 'extratrees')                          | 0.5720 | 0.0175 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.6290 | 0.0172 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | extratrees   | raw                | emd                |      0.0898 |   0.0741 |    0.1056 |      0.0000 |            0.0026 |
| comp           | extratrees   | raw                | imf_linear_a1.41   |     -0.0984 |  -0.1129 |   -0.0838 |      0.0000 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a1.41      |     -0.0452 |  -0.0532 |   -0.0373 |      0.0000 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0467 |  -0.0560 |   -0.0374 |      0.0000 |            0.0026 |
| comp           | extratrees   | raw                | irmf_h2_a2_conv    |     -0.0360 |  -0.0469 |   -0.0252 |      0.0001 |            0.0026 |
| comp           | extratrees   | raw                | imf_linear_a2      |     -0.0909 |  -0.1067 |   -0.0751 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | imf_linear_a1.41   |     -0.1882 |  -0.2097 |   -0.1667 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a1.41      |     -0.1351 |  -0.1535 |   -0.1167 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.1365 |  -0.1565 |   -0.1166 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | irmf_h2_a2_conv    |     -0.1258 |  -0.1455 |   -0.1061 |      0.0000 |            0.0026 |
| comp           | extratrees   | emd                | imf_linear_a2      |     -0.1807 |  -0.2038 |   -0.1577 |      0.0000 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0531 |   0.0426 |    0.0636 |      0.0000 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0516 |   0.0403 |    0.0630 |      0.0000 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0623 |   0.0510 |    0.0737 |      0.0000 |            0.0026 |
| comp           | extratrees   | imf_linear_a1.41   | imf_linear_a2      |      0.0074 |   0.0016 |    0.0133 |      0.1654 |            0.0031 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0015 |  -0.0094 |    0.0064 |      1.0000 |            0.6753 |
| comp           | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0092 |   0.0000 |    0.0184 |      0.4457 |            0.0055 |
| comp           | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0457 |  -0.0581 |   -0.0332 |      0.0000 |            0.0026 |
| comp           | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0107 |   0.0040 |    0.0174 |      0.0444 |            0.0026 |
| comp           | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0442 |  -0.0575 |   -0.0309 |      0.0001 |            0.0026 |
| comp           | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0549 |  -0.0696 |   -0.0402 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | raw                | emd                |      0.0934 |   0.0793 |    0.1075 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | raw                | imf_linear_a1.41   |     -0.0468 |  -0.0624 |   -0.0313 |      0.0002 |            0.0026 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41      |      0.0001 |  -0.0128 |    0.0129 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a1.41_conv |     -0.0061 |  -0.0186 |    0.0064 |      1.0000 |            0.2063 |
| comp+log_scale | extratrees   | raw                | irmf_h2_a2_conv    |     -0.0003 |  -0.0111 |    0.0105 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw                | imf_linear_a2      |     -0.0384 |  -0.0562 |   -0.0207 |      0.0047 |            0.0026 |
| comp+log_scale | extratrees   | emd                | imf_linear_a1.41   |     -0.1402 |  -0.1568 |   -0.1237 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41      |     -0.0933 |  -0.1091 |   -0.0775 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a1.41_conv |     -0.0995 |  -0.1140 |   -0.0850 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | emd                | irmf_h2_a2_conv    |     -0.0937 |  -0.1054 |   -0.0820 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | emd                | imf_linear_a2      |     -0.1318 |  -0.1522 |   -0.1113 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0469 |   0.0339 |    0.0599 |      0.0000 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0407 |   0.0273 |    0.0541 |      0.0002 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0465 |   0.0328 |    0.0602 |      0.0001 |            0.0026 |
| comp+log_scale | extratrees   | imf_linear_a1.41   | imf_linear_a2      |      0.0084 |  -0.0015 |    0.0184 |      0.7186 |            0.0093 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0062 |  -0.0152 |    0.0028 |      1.0000 |            0.0376 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | irmf_h2_a2_conv    |     -0.0004 |  -0.0097 |    0.0089 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | irmf_h2_a1.41      | imf_linear_a2      |     -0.0385 |  -0.0533 |   -0.0236 |      0.0011 |            0.0026 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0058 |  -0.0031 |    0.0147 |      1.0000 |            0.0615 |
| comp+log_scale | extratrees   | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0323 |  -0.0471 |   -0.0175 |      0.0047 |            0.0026 |
| comp+log_scale | extratrees   | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0381 |  -0.0550 |   -0.0212 |      0.0037 |            0.0026 |
