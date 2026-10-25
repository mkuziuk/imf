# dronerf_phase3_trees_lgbm: dronerf / 10class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 3703.6 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'lightgbm')                          | 0.5649 | 0.0200 |
| ('emd', 'comp+log_scale', 'lightgbm')                | 0.6306 | 0.0184 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.6951 | 0.0202 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.7070 | 0.0179 |
| ('imf_linear_a2', 'comp', 'lightgbm')                | 0.6894 | 0.0195 |
| ('imf_linear_a2', 'comp+log_scale', 'lightgbm')      | 0.7056 | 0.0178 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.6434 | 0.0176 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.6823 | 0.0170 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.6468 | 0.0185 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.6835 | 0.0165 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')              | 0.6378 | 0.0187 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')    | 0.6765 | 0.0155 |
| ('raw', 'comp', 'lightgbm')                          | 0.5704 | 0.0251 |
| ('raw', 'comp+log_scale', 'lightgbm')                | 0.6368 | 0.0200 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw                | emd                |      0.0055 |  -0.0073 |    0.0183 |      1.0000 |            0.2214 |
| comp           | lightgbm     | raw                | imf_linear_a1.41   |     -0.1247 |  -0.1467 |   -0.1027 |      0.0000 |            0.0026 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0729 |  -0.0896 |   -0.0563 |      0.0000 |            0.0026 |
| comp           | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0764 |  -0.0956 |   -0.0572 |      0.0000 |            0.0026 |
| comp           | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0674 |  -0.0897 |   -0.0450 |      0.0003 |            0.0026 |
| comp           | lightgbm     | raw                | imf_linear_a2      |     -0.1190 |  -0.1416 |   -0.0965 |      0.0000 |            0.0026 |
| comp           | lightgbm     | emd                | imf_linear_a1.41   |     -0.1302 |  -0.1489 |   -0.1115 |      0.0000 |            0.0026 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0784 |  -0.0896 |   -0.0673 |      0.0000 |            0.0026 |
| comp           | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0819 |  -0.0949 |   -0.0689 |      0.0000 |            0.0026 |
| comp           | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0729 |  -0.0877 |   -0.0580 |      0.0000 |            0.0026 |
| comp           | lightgbm     | emd                | imf_linear_a2      |     -0.1245 |  -0.1427 |   -0.1064 |      0.0000 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0517 |   0.0408 |    0.0627 |      0.0000 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0483 |   0.0347 |    0.0618 |      0.0001 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0573 |   0.0440 |    0.0707 |      0.0000 |            0.0026 |
| comp           | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |      0.0057 |  -0.0030 |    0.0143 |      1.0000 |            0.0872 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0035 |  -0.0109 |    0.0040 |      1.0000 |            0.1534 |
| comp           | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0056 |  -0.0040 |    0.0152 |      1.0000 |            0.1534 |
| comp           | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |     -0.0461 |  -0.0579 |   -0.0343 |      0.0000 |            0.0026 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0090 |   0.0005 |    0.0176 |      0.3522 |            0.0027 |
| comp           | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0426 |  -0.0574 |   -0.0278 |      0.0004 |            0.0026 |
| comp           | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0516 |  -0.0637 |   -0.0396 |      0.0000 |            0.0026 |
| comp+log_scale | lightgbm     | raw                | emd                |      0.0063 |  -0.0066 |    0.0192 |      1.0000 |            0.2214 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a1.41   |     -0.0702 |  -0.0836 |   -0.0567 |      0.0000 |            0.0026 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41      |     -0.0455 |  -0.0587 |   -0.0323 |      0.0001 |            0.0026 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a1.41_conv |     -0.0467 |  -0.0613 |   -0.0321 |      0.0002 |            0.0026 |
| comp+log_scale | lightgbm     | raw                | irmf_h2_a2_conv    |     -0.0397 |  -0.0547 |   -0.0246 |      0.0008 |            0.0026 |
| comp+log_scale | lightgbm     | raw                | imf_linear_a2      |     -0.0688 |  -0.0832 |   -0.0543 |      0.0000 |            0.0026 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a1.41   |     -0.0764 |  -0.0924 |   -0.0605 |      0.0000 |            0.0026 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41      |     -0.0518 |  -0.0682 |   -0.0354 |      0.0002 |            0.0026 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a1.41_conv |     -0.0530 |  -0.0711 |   -0.0348 |      0.0004 |            0.0026 |
| comp+log_scale | lightgbm     | emd                | irmf_h2_a2_conv    |     -0.0459 |  -0.0634 |   -0.0285 |      0.0008 |            0.0026 |
| comp+log_scale | lightgbm     | emd                | imf_linear_a2      |     -0.0750 |  -0.0927 |   -0.0574 |      0.0000 |            0.0026 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0247 |   0.0166 |    0.0327 |      0.0003 |            0.0026 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0235 |   0.0153 |    0.0316 |      0.0004 |            0.0026 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0305 |   0.0195 |    0.0415 |      0.0005 |            0.0026 |
| comp+log_scale | lightgbm     | imf_linear_a1.41   | imf_linear_a2      |      0.0014 |  -0.0063 |    0.0091 |      1.0000 |            0.6056 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0012 |  -0.0058 |    0.0034 |      1.0000 |            0.6056 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0058 |  -0.0015 |    0.0132 |      0.8863 |            0.0210 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41      | imf_linear_a2      |     -0.0233 |  -0.0304 |   -0.0161 |      0.0002 |            0.0026 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0070 |   0.0013 |    0.0127 |      0.1907 |            0.0026 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0221 |  -0.0310 |   -0.0131 |      0.0013 |            0.0026 |
| comp+log_scale | lightgbm     | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0291 |  -0.0406 |   -0.0175 |      0.0011 |            0.0026 |
