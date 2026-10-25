# trees_xgboost: birdclef2023 / species

150 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 5753.4 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                     |   mean |    std |
|:----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'xgboost')                          | 0.5028 | 0.0188 |
| ('emd', 'comp+log_scale', 'xgboost')                | 0.5030 | 0.0188 |
| ('imf_linear_a1.41', 'comp', 'xgboost')             | 0.5278 | 0.0239 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')   | 0.5280 | 0.0228 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                | 0.5288 | 0.0251 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')      | 0.5280 | 0.0238 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')           | 0.5294 | 0.0243 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost') | 0.5292 | 0.0247 |
| ('raw', 'comp', 'xgboost')                          | 0.4621 | 0.0193 |
| ('raw', 'comp+log_scale', 'xgboost')                | 0.4553 | 0.0207 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | xgboost      | raw              | emd                |     -0.0407 |  -0.0523 |   -0.0291 |      0.0000 |            0.0012 |
| comp           | xgboost      | raw              | imf_linear_a1.41   |     -0.0657 |  -0.0802 |   -0.0513 |      0.0000 |            0.0012 |
| comp           | xgboost      | raw              | irmf_h2_a1.41      |     -0.0667 |  -0.0808 |   -0.0526 |      0.0000 |            0.0012 |
| comp           | xgboost      | raw              | irmf_h2_a1.41_conv |     -0.0673 |  -0.0802 |   -0.0543 |      0.0000 |            0.0012 |
| comp           | xgboost      | emd              | imf_linear_a1.41   |     -0.0250 |  -0.0360 |   -0.0140 |      0.0027 |            0.0012 |
| comp           | xgboost      | emd              | irmf_h2_a1.41      |     -0.0260 |  -0.0378 |   -0.0141 |      0.0034 |            0.0012 |
| comp           | xgboost      | emd              | irmf_h2_a1.41_conv |     -0.0266 |  -0.0381 |   -0.0151 |      0.0025 |            0.0012 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0010 |  -0.0095 |    0.0076 |      1.0000 |            1.0000 |
| comp           | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0015 |  -0.0101 |    0.0071 |      1.0000 |            1.0000 |
| comp           | xgboost      | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0006 |  -0.0051 |    0.0040 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | raw              | emd                |     -0.0478 |  -0.0650 |   -0.0306 |      0.0005 |            0.0012 |
| comp+log_scale | xgboost      | raw              | imf_linear_a1.41   |     -0.0727 |  -0.0892 |   -0.0563 |      0.0000 |            0.0012 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41      |     -0.0728 |  -0.0913 |   -0.0542 |      0.0000 |            0.0012 |
| comp+log_scale | xgboost      | raw              | irmf_h2_a1.41_conv |     -0.0740 |  -0.0906 |   -0.0573 |      0.0000 |            0.0012 |
| comp+log_scale | xgboost      | emd              | imf_linear_a1.41   |     -0.0250 |  -0.0374 |   -0.0125 |      0.0066 |            0.0012 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41      |     -0.0250 |  -0.0393 |   -0.0107 |      0.0150 |            0.0012 |
| comp+log_scale | xgboost      | emd              | irmf_h2_a1.41_conv |     -0.0262 |  -0.0407 |   -0.0117 |      0.0133 |            0.0012 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41      |     -0.0000 |  -0.0090 |    0.0089 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | imf_linear_a1.41 | irmf_h2_a1.41_conv |     -0.0012 |  -0.0095 |    0.0070 |      1.0000 |            1.0000 |
| comp+log_scale | xgboost      | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0012 |  -0.0059 |    0.0035 |      1.0000 |            1.0000 |
