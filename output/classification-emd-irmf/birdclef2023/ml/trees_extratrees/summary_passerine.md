# trees_extratrees: birdclef2023 / passerine

150 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 959.6 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |    std |
|:-------------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'extratrees')                          | 0.7745 | 0.0169 |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.7719 | 0.0198 |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.8021 | 0.0157 |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.7995 | 0.0145 |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.8014 | 0.0168 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.8030 | 0.0173 |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.8015 | 0.0167 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.8035 | 0.0167 |
| ('raw', 'comp', 'extratrees')                          | 0.7827 | 0.0163 |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.7819 | 0.0173 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | extratrees   | raw              | emd                |      0.0082 |  -0.0037 |    0.0200 |      1.0000 |            0.0537 |
| comp           | extratrees   | raw              | imf_linear_a1.41   |     -0.0194 |  -0.0300 |   -0.0088 |      0.0198 |            0.0012 |
| comp           | extratrees   | raw              | irmf_h2_a1.41      |     -0.0187 |  -0.0295 |   -0.0079 |      0.0275 |            0.0012 |
| comp           | extratrees   | raw              | irmf_h2_a1.41_conv |     -0.0188 |  -0.0303 |   -0.0072 |      0.0359 |            0.0012 |
| comp           | extratrees   | emd              | imf_linear_a1.41   |     -0.0276 |  -0.0375 |   -0.0176 |      0.0006 |            0.0012 |
| comp           | extratrees   | emd              | irmf_h2_a1.41      |     -0.0269 |  -0.0361 |   -0.0177 |      0.0004 |            0.0012 |
| comp           | extratrees   | emd              | irmf_h2_a1.41_conv |     -0.0270 |  -0.0371 |   -0.0168 |      0.0009 |            0.0012 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0007 |  -0.0026 |    0.0040 |      1.0000 |            1.0000 |
| comp           | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0006 |  -0.0037 |    0.0049 |      1.0000 |            1.0000 |
| comp           | extratrees   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0001 |  -0.0033 |    0.0032 |      1.0000 |            1.0000 |
| comp+log_scale | extratrees   | raw              | emd                |      0.0100 |  -0.0041 |    0.0242 |      1.0000 |            0.0718 |
| comp+log_scale | extratrees   | raw              | imf_linear_a1.41   |     -0.0175 |  -0.0311 |   -0.0039 |      0.1381 |            0.0016 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41      |     -0.0210 |  -0.0333 |   -0.0087 |      0.0281 |            0.0012 |
| comp+log_scale | extratrees   | raw              | irmf_h2_a1.41_conv |     -0.0215 |  -0.0315 |   -0.0115 |      0.0054 |            0.0012 |
| comp+log_scale | extratrees   | emd              | imf_linear_a1.41   |     -0.0276 |  -0.0401 |   -0.0150 |      0.0051 |            0.0012 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41      |     -0.0311 |  -0.0430 |   -0.0192 |      0.0011 |            0.0012 |
| comp+log_scale | extratrees   | emd              | irmf_h2_a1.41_conv |     -0.0316 |  -0.0425 |   -0.0206 |      0.0004 |            0.0012 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0035 |  -0.0131 |    0.0060 |      1.0000 |            0.6027 |
| comp+log_scale | extratrees   | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0040 |  -0.0113 |    0.0033 |      1.0000 |            0.1534 |
| comp+log_scale | extratrees   | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0005 |  -0.0059 |    0.0050 |      1.0000 |            1.0000 |
