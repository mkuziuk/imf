# trees_lightgbm: birdclef2023 / passerine

150 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 432.0 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'lightgbm')                          | 0.7966 | 0.0155 |
| ('emd', 'comp+log_scale', 'lightgbm')                | 0.7986 | 0.0134 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.8181 | 0.0122 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.8174 | 0.0126 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.8127 | 0.0138 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.8114 | 0.0141 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.8125 | 0.0135 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.8104 | 0.0137 |
| ('raw', 'comp', 'lightgbm')                          | 0.7740 | 0.0187 |
| ('raw', 'comp+log_scale', 'lightgbm')                | 0.7725 | 0.0170 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw              | emd                |     -0.0226 |  -0.0355 |   -0.0096 |      0.0156 |            0.0012 |
| comp           | lightgbm     | raw              | imf_linear_a1.41   |     -0.0441 |  -0.0593 |   -0.0289 |      0.0003 |            0.0012 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41      |     -0.0387 |  -0.0502 |   -0.0271 |      0.0001 |            0.0012 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.0385 |  -0.0511 |   -0.0259 |      0.0002 |            0.0012 |
| comp           | lightgbm     | emd              | imf_linear_a1.41   |     -0.0215 |  -0.0294 |   -0.0137 |      0.0005 |            0.0012 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0161 |  -0.0230 |   -0.0092 |      0.0023 |            0.0012 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0159 |  -0.0237 |   -0.0082 |      0.0054 |            0.0012 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0054 |  -0.0015 |    0.0123 |      0.5309 |            0.0121 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0056 |  -0.0010 |    0.0122 |      0.5309 |            0.0121 |
| comp           | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0002 |  -0.0042 |    0.0046 |      1.0000 |            0.8904 |
| comp+log_scale | lightgbm     | raw              | emd                |     -0.0261 |  -0.0353 |   -0.0169 |      0.0004 |            0.0012 |
| comp+log_scale | lightgbm     | raw              | imf_linear_a1.41   |     -0.0449 |  -0.0600 |   -0.0297 |      0.0003 |            0.0012 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41      |     -0.0389 |  -0.0501 |   -0.0278 |      0.0001 |            0.0012 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.0379 |  -0.0485 |   -0.0274 |      0.0000 |            0.0012 |
| comp+log_scale | lightgbm     | emd              | imf_linear_a1.41   |     -0.0188 |  -0.0273 |   -0.0103 |      0.0034 |            0.0012 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0128 |  -0.0198 |   -0.0059 |      0.0113 |            0.0012 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0119 |  -0.0174 |   -0.0063 |      0.0043 |            0.0012 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0059 |  -0.0010 |    0.0129 |      0.5309 |            0.0121 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0069 |  -0.0013 |    0.0152 |      0.5309 |            0.0121 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0010 |  -0.0044 |    0.0064 |      1.0000 |            0.7788 |
