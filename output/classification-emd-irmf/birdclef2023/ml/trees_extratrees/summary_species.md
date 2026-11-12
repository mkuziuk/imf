# trees_extratrees: birdclef2023 / species

150 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 853.6 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.4731 | 0.0171 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.4714 | 0.0172 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.4982 | 0.0275 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.5004 | 0.0259 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.5031 | 0.0266 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.5032 | 0.0263 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.5011 | 0.0263 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.5041 | 0.0268 |
| ('raw', 'comp', 'extratrees')                          | 0.4537 | 0.0153 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.4509 | 0.0169 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | extratrees   | raw              | emd                |     -0.0193 |  -0.0300 |   -0.0087 |      0.0229 |            0.0012 |
| comp           | extratrees   | raw              | imf_linear_a1.41   |     -0.0444 |  -0.0703 |   -0.0186 |      0.0317 |            0.0012 |
| comp           | extratrees   | raw              | irmf_h2_a1.41      |     -0.0494 |  -0.0701 |   -0.0287 |      0.0028 |            0.0012 |
| comp           | extratrees   | raw              | irmf_h2_a1.41_conv |     -0.0473 |  -0.0684 |   -0.0263 |      0.0044 |            0.0012 |
| comp           | extratrees   | emd              | imf_linear_a1.41   |     -0.0251 |  -0.0470 |   -0.0031 |      0.1953 |            0.0049 |
| comp           | extratrees   | emd              | irmf_h2_a1.41      |     -0.0301 |  -0.0513 |   -0.0088 |      0.0973 |            0.0038 |
| comp           | extratrees   | emd              | irmf_h2_a1.41_conv |     -0.0280 |  -0.0487 |   -0.0073 |      0.0973 |            0.0049 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0050 |  -0.0180 |    0.0081 |      1.0000 |            0.4995 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0029 |  -0.0145 |    0.0086 |      1.0000 |            0.5537 |
| comp           | extratrees   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |      0.0021 |  -0.0047 |    0.0089 |      1.0000 |            0.4995 |
| comp+log_scale | extratrees   | raw              | emd                |     -0.0205 |  -0.0310 |   -0.0099 |      0.0146 |            0.0012 |
| comp+log_scale | extratrees   | raw              | imf_linear_a1.41   |     -0.0495 |  -0.0714 |   -0.0276 |      0.0044 |            0.0012 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41      |     -0.0522 |  -0.0724 |   -0.0321 |      0.0013 |            0.0012 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41_conv |     -0.0531 |  -0.0731 |   -0.0332 |      0.0011 |            0.0012 |
| comp+log_scale | extratrees   | emd              | imf_linear_a1.41   |     -0.0290 |  -0.0500 |   -0.0081 |      0.0973 |            0.0022 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41      |     -0.0318 |  -0.0545 |   -0.0091 |      0.0973 |            0.0034 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41_conv |     -0.0327 |  -0.0544 |   -0.0109 |      0.0740 |            0.0034 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0027 |  -0.0124 |    0.0069 |      1.0000 |            0.5065 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0036 |  -0.0124 |    0.0051 |      1.0000 |            0.4995 |
| comp+log_scale | extratrees   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0009 |  -0.0049 |    0.0032 |      1.0000 |            0.5537 |
