# trees_lightgbm_remaining: birdclef2023 / species

140 fits on shared split IDs [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 1502.4 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'lightgbm')                          | 0.5061 | 0.0193 |
| ('emd', 'comp+log_scale', 'lightgbm')                | 0.5019 | 0.0168 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.5288 | 0.0276 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.5258 | 0.0261 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.5298 | 0.0256 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.5285 | 0.0210 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.5313 | 0.0266 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.5304 | 0.0264 |
| ('raw', 'comp', 'lightgbm')                          | 0.4686 | 0.0203 |
| ('raw', 'comp+log_scale', 'lightgbm')                | 0.4598 | 0.0198 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw              | emd                |     -0.0375 |  -0.0551 |   -0.0200 |      0.0058 |            0.0024 |
| comp           | lightgbm     | raw              | imf_linear_a1.41   |     -0.0601 |  -0.0766 |   -0.0437 |      0.0000 |            0.0024 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41      |     -0.0612 |  -0.0764 |   -0.0459 |      0.0000 |            0.0024 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.0626 |  -0.0798 |   -0.0455 |      0.0000 |            0.0024 |
| comp           | lightgbm     | emd              | imf_linear_a1.41   |     -0.0226 |  -0.0398 |   -0.0055 |      0.0962 |            0.0024 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0237 |  -0.0403 |   -0.0070 |      0.0721 |            0.0026 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0251 |  -0.0415 |   -0.0087 |      0.0506 |            0.0024 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0010 |  -0.0128 |    0.0108 |      1.0000 |            1.0000 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0025 |  -0.0131 |    0.0081 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0015 |  -0.0074 |    0.0044 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | raw              | emd                |     -0.0421 |  -0.0575 |   -0.0267 |      0.0007 |            0.0024 |
| comp+log_scale | lightgbm     | raw              | imf_linear_a1.41   |     -0.0660 |  -0.0868 |   -0.0452 |      0.0002 |            0.0024 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41      |     -0.0687 |  -0.0823 |   -0.0552 |      0.0000 |            0.0024 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.0706 |  -0.0884 |   -0.0528 |      0.0000 |            0.0024 |
| comp+log_scale | lightgbm     | emd              | imf_linear_a1.41   |     -0.0239 |  -0.0390 |   -0.0087 |      0.0470 |            0.0024 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0266 |  -0.0370 |   -0.0163 |      0.0012 |            0.0024 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0285 |  -0.0438 |   -0.0131 |      0.0165 |            0.0024 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0028 |  -0.0148 |    0.0093 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0046 |  -0.0150 |    0.0058 |      1.0000 |            0.4058 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0018 |  -0.0100 |    0.0063 |      1.0000 |            0.9686 |
