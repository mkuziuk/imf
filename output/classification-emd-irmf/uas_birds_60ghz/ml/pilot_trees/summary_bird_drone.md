# uas_phase3_pilot_trees: uas_birds_60ghz / bird_drone

12 fits on shared split IDs [0]; wall time 9.8 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                   |   mean |   std |
|:--------------------------------------------------|-------:|------:|
| ('irmf_h2_a1.41', 'comp', 'extratrees')           | 1.0000 |   nan |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')             | 1.0000 |   nan |
| ('irmf_h2_a1.41', 'comp', 'xgboost')              | 0.9948 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees') | 1.0000 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')   | 1.0000 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')    | 0.9948 |   nan |
| ('raw', 'comp', 'extratrees')                     | 0.9297 |   nan |
| ('raw', 'comp', 'lightgbm')                       | 0.9922 |   nan |
| ('raw', 'comp', 'xgboost')                        | 0.9922 |   nan |
| ('raw', 'comp+log_scale', 'extratrees')           | 0.9844 |   nan |
| ('raw', 'comp+log_scale', 'lightgbm')             | 1.0000 |   nan |
| ('raw', 'comp+log_scale', 'xgboost')              | 1.0000 |   nan |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
