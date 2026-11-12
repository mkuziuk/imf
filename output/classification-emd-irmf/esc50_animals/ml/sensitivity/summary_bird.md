# esc50_phase3_sensitivity: esc50_animals / bird

90 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 127.4 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('irmf_h1_a1.41_conv', 'comp', 'lightgbm')           | 0.7781 | 0.0307 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.7782 | 0.0303 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')              | 0.7800 | 0.0374 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')    | 0.7743 | 0.0391 |
| ('irmf_h4_a1.41_conv', 'comp', 'lightgbm')           | 0.7779 | 0.0381 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.7765 | 0.0418 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0002 |  -0.0211 |    0.0216 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h1_a1.41_conv | irmf_h2_a2_conv    |     -0.0019 |  -0.0192 |    0.0154 |      1.0000 |            1.0000 |
| comp           | lightgbm     | irmf_h4_a1.41_conv | irmf_h2_a2_conv    |     -0.0021 |  -0.0232 |    0.0189 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |      0.0016 |  -0.0228 |    0.0261 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h1_a1.41_conv | irmf_h2_a2_conv    |      0.0038 |  -0.0138 |    0.0215 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | irmf_h4_a1.41_conv | irmf_h2_a2_conv    |      0.0022 |  -0.0196 |    0.0240 |      1.0000 |            1.0000 |
