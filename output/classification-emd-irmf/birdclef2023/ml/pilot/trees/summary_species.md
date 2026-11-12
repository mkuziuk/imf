# trees: birdclef2023 / species

30 fits on shared split IDs [0]; wall time 204.4 s with 2 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                        |   mean |   std |
|:-------------------------------------------------------|-------:|------:|
| ('emd', 'comp', 'extratrees')                          | 0.4355 |   nan |
| ('emd', 'comp', 'lightgbm')                            | 0.4870 |   nan |
| ('emd', 'comp', 'xgboost')                             | 0.4789 |   nan |
| ('emd', 'comp+log_scale', 'extratrees')                | 0.4333 |   nan |
| ('emd', 'comp+log_scale', 'lightgbm')                  | 0.4855 |   nan |
| ('emd', 'comp+log_scale', 'xgboost')                   | 0.4842 |   nan |
| ('imf_linear_a1.41', 'comp', 'extratrees')             | 0.4627 |   nan |
| ('imf_linear_a1.41', 'comp', 'lightgbm')               | 0.5166 |   nan |
| ('imf_linear_a1.41', 'comp', 'xgboost')                | 0.5077 |   nan |
| ('imf_linear_a1.41', 'comp+log_scale', 'extratrees')   | 0.4632 |   nan |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')     | 0.5039 |   nan |
| ('imf_linear_a1.41', 'comp+log_scale', 'xgboost')      | 0.5056 |   nan |
| ('irmf_h2_a1.41', 'comp', 'extratrees')                | 0.4730 |   nan |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                  | 0.5218 |   nan |
| ('irmf_h2_a1.41', 'comp', 'xgboost')                   | 0.5065 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'extratrees')      | 0.4832 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')        | 0.5180 |   nan |
| ('irmf_h2_a1.41', 'comp+log_scale', 'xgboost')         | 0.5094 |   nan |
| ('irmf_h2_a1.41_conv', 'comp', 'extratrees')           | 0.4794 |   nan |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')             | 0.5201 |   nan |
| ('irmf_h2_a1.41_conv', 'comp', 'xgboost')              | 0.5115 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'extratrees') | 0.4788 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm')   | 0.5251 |   nan |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'xgboost')    | 0.5102 |   nan |
| ('raw', 'comp', 'extratrees')                          | 0.4341 |   nan |
| ('raw', 'comp', 'lightgbm')                            | 0.4644 |   nan |
| ('raw', 'comp', 'xgboost')                             | 0.4611 |   nan |
| ('raw', 'comp+log_scale', 'extratrees')                | 0.4228 |   nan |
| ('raw', 'comp+log_scale', 'lightgbm')                  | 0.4610 |   nan |
| ('raw', 'comp+log_scale', 'xgboost')                   | 0.4582 |   nan |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.
