# esc50_phase3_concat: esc50_animals / bird

60 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 147.7 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                           |   mean |    std |
|:----------------------------------------------------------|-------:|-------:|
| ('emd__irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.7789 | 0.0393 |
| ('emd__irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.7782 | 0.0409 |
| ('emd__irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.7796 | 0.0345 |
| ('emd__irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.7768 | 0.0405 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.
EMD and IRMF component statistics are concatenated; shared log_scale appears once.
No new decomposition or feature parquet is generated. Identical grouped tuning/refit and boosting budget.
Fusion importance order is two blocks, EMD then IRMF, each in its true scale order.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a           | method_b                |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-------------------|:------------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | emd__irmf_h2_a1.41 | emd__irmf_h2_a1.41_conv |     -0.0007 |  -0.0138 |    0.0124 |      1.0000 |            1.0000 |
| comp+log_scale | lightgbm     | emd__irmf_h2_a1.41 | emd__irmf_h2_a1.41_conv |      0.0014 |  -0.0139 |    0.0167 |      1.0000 |            1.0000 |
