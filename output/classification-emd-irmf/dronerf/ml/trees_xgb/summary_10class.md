# dronerf_phase3_trees_xgb: dronerf / 10class

210 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 4667.0 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                     |   mean |    std |
|:----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'xgboost')                          | 0.5255 | 0.0187 |
| ('emd', 'comp+log_scale', 'xgboost')                | 0.5819 | 0.0178 |
| ('imf_linear_a1.41', 'comp', 'xgboost')             | 0.6487 | 0.0184 |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')   | 0.6629 | 0.0203 |
| ('imf_linear_a2', 'comp', 'xgboost')                | 0.6391 | 0.0205 |
| ('imf_linear_a2', 'comp+log_scale', 'xgboost')      | 0.6554 | 0.0188 |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                | 0.5900 | 0.0196 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')      | 0.6265 | 0.0170 |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')           | 0.5941 | 0.0185 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost') | 0.6303 | 0.0184 |
| ('irmf_h2_a2_conv', 'comp', 'xgboost')              | 0.5840 | 0.0176 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'xgboost')    | 0.6202 | 0.0168 |
| ('raw', 'comp', 'xgboost')                          | 0.5209 | 0.0206 |
| ('raw', 'comp+log_scale', 'xgboost')                | 0.5775 | 0.0165 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | xgboost      | raw                | emd                |     -0.0045 |  -0.0177 |    0.0086 |      1.0000 |            0.1912 |
| comp           | xgboost      | raw                | imf_linear_a1.41   |     -0.1278 |  -0.1484 |   -0.1072 |      0.0000 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a1.41      |     -0.0691 |  -0.0841 |   -0.0541 |      0.0000 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0732 |  -0.0872 |   -0.0593 |      0.0000 |            0.0026 |
| comp           | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0631 |  -0.0843 |   -0.0419 |      0.0003 |            0.0026 |
| comp           | xgboost      | raw                | imf_linear_a2      |     -0.1182 |  -0.1406 |   -0.0959 |      0.0000 |            0.0026 |
| comp           | xgboost      | emd                | imf_linear_a1.41   |     -0.1232 |  -0.1395 |   -0.1070 |      0.0000 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a1.41      |     -0.0646 |  -0.0750 |   -0.0541 |      0.0000 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0687 |  -0.0789 |   -0.0584 |      0.0000 |            0.0026 |
| comp           | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0585 |  -0.0736 |   -0.0434 |      0.0000 |            0.0026 |
| comp           | xgboost      | emd                | imf_linear_a2      |     -0.1137 |  -0.1320 |   -0.0953 |      0.0000 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0587 |   0.0450 |    0.0724 |      0.0000 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0546 |   0.0380 |    0.0712 |      0.0001 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0647 |   0.0471 |    0.0823 |      0.0000 |            0.0026 |
| comp           | xgboost      | imf_linear_a1.41   | imf_linear_a2      |      0.0096 |  -0.0025 |    0.0217 |      0.7817 |            0.0068 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0041 |  -0.0116 |    0.0034 |      1.0000 |            0.1650 |
| comp           | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0060 |  -0.0071 |    0.0192 |      1.0000 |            0.1912 |
| comp           | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |     -0.0491 |  -0.0632 |   -0.0350 |      0.0001 |            0.0026 |
| comp           | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0101 |  -0.0005 |    0.0207 |      0.5329 |            0.0055 |
| comp           | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0450 |  -0.0599 |   -0.0300 |      0.0003 |            0.0026 |
| comp           | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0551 |  -0.0685 |   -0.0417 |      0.0000 |            0.0026 |
| comp+log_scale | xgboost      | raw                | emd                |     -0.0043 |  -0.0233 |    0.0147 |      1.0000 |            0.2078 |
| comp+log_scale | xgboost      | raw                | imf_linear_a1.41   |     -0.0853 |  -0.0997 |   -0.0710 |      0.0000 |            0.0026 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41      |     -0.0490 |  -0.0642 |   -0.0337 |      0.0002 |            0.0026 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a1.41_conv |     -0.0527 |  -0.0682 |   -0.0372 |      0.0001 |            0.0026 |
| comp+log_scale | xgboost      | raw                | irmf_h2_a2_conv    |     -0.0426 |  -0.0607 |   -0.0245 |      0.0021 |            0.0026 |
| comp+log_scale | xgboost      | raw                | imf_linear_a2      |     -0.0779 |  -0.0973 |   -0.0584 |      0.0000 |            0.0026 |
| comp+log_scale | xgboost      | emd                | imf_linear_a1.41   |     -0.0810 |  -0.0976 |   -0.0644 |      0.0000 |            0.0026 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41      |     -0.0447 |  -0.0605 |   -0.0288 |      0.0005 |            0.0026 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a1.41_conv |     -0.0484 |  -0.0628 |   -0.0340 |      0.0001 |            0.0026 |
| comp+log_scale | xgboost      | emd                | irmf_h2_a2_conv    |     -0.0383 |  -0.0531 |   -0.0235 |      0.0011 |            0.0026 |
| comp+log_scale | xgboost      | emd                | imf_linear_a2      |     -0.0736 |  -0.0936 |   -0.0535 |      0.0000 |            0.0026 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41      |      0.0364 |   0.0224 |    0.0503 |      0.0010 |            0.0026 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a1.41_conv |      0.0326 |   0.0198 |    0.0454 |      0.0012 |            0.0026 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | irmf_h2_a2_conv    |      0.0427 |   0.0285 |    0.0570 |      0.0003 |            0.0026 |
| comp+log_scale | xgboost      | imf_linear_a1.41   | imf_linear_a2      |      0.0074 |  -0.0051 |    0.0200 |      1.0000 |            0.0903 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a1.41_conv |     -0.0038 |  -0.0097 |    0.0022 |      1.0000 |            0.0747 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | irmf_h2_a2_conv    |      0.0064 |  -0.0013 |    0.0140 |      0.7774 |            0.0081 |
| comp+log_scale | xgboost      | irmf_h2_a1.41      | imf_linear_a2      |     -0.0289 |  -0.0406 |   -0.0172 |      0.0015 |            0.0026 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | irmf_h2_a2_conv    |      0.0101 |   0.0008 |    0.0194 |      0.3505 |            0.0031 |
| comp+log_scale | xgboost      | irmf_h2_a1.41_conv | imf_linear_a2      |     -0.0252 |  -0.0382 |   -0.0121 |      0.0109 |            0.0026 |
| comp+log_scale | xgboost      | irmf_h2_a2_conv    | imf_linear_a2      |     -0.0353 |  -0.0466 |   -0.0240 |      0.0002 |            0.0026 |
