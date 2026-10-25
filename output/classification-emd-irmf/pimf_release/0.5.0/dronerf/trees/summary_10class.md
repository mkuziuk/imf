# trees: dronerf / 10class

30 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 858.3 s with 3 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.6641 | 0.0167 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.6924 | 0.0167 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
