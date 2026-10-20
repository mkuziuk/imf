# esc50_phase3_sensitivity: esc50_animals / animal

90 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 689.9 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('irmf_h1_a1.41_conv', 'comp', 'lightgbm')           | 0.6523 | 0.0321 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.6616 | 0.0308 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')              | 0.6488 | 0.0372 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')    | 0.6534 | 0.0360 |
| ('irmf_h4_a1.41_conv', 'comp', 'lightgbm')           | 0.6679 | 0.0349 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.6732 | 0.0323 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0155 |  -0.0319 |    0.0008 |      0.2442 |            0.0034 |
| comp           | lightgbm     | irmf_h1_a1.41_conv | irmf_h2_a2_conv    |      0.0035 |  -0.0111 |    0.0182 |      0.6125 |            0.2769 |
| comp           | lightgbm     | irmf_h4_a1.41_conv | irmf_h2_a2_conv    |      0.0191 |   0.0061 |    0.0321 |      0.0432 |            0.0004 |
| comp+log_scale | lightgbm     | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv |     -0.0115 |  -0.0270 |    0.0039 |      0.3955 |            0.0161 |
| comp+log_scale | lightgbm     | irmf_h1_a1.41_conv | irmf_h2_a2_conv    |      0.0082 |  -0.0040 |    0.0205 |      0.3955 |            0.0167 |
| comp+log_scale | lightgbm     | irmf_h4_a1.41_conv | irmf_h2_a2_conv    |      0.0198 |   0.0031 |    0.0365 |      0.1171 |            0.0015 |
