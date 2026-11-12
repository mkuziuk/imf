# dronerf_phase3_trees_lonly: dronerf / 10class

150 fits on shared split IDs [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]; wall time 1423.7 s with 1 CPU threads.

Balanced accuracy, mean and SD across the selected paired folds:

|                                                      |   mean |    std |
|:-----------------------------------------------------|-------:|-------:|
| ('emd', 'comp', 'lightgbm')                          | 0.5205 | 0.0220 |
| ('emd', 'comp+log_scale', 'lightgbm')                | 0.5809 | 0.0229 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             | 0.6319 | 0.0205 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   | 0.6456 | 0.0224 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                | 0.5832 | 0.0197 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      | 0.6151 | 0.0222 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           | 0.5900 | 0.0208 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') | 0.6218 | 0.0197 |
| ('raw', 'comp', 'lightgbm')                          | 0.5369 | 0.0238 |
| ('raw', 'comp+log_scale', 'lightgbm')                | 0.5950 | 0.0204 |

LightGBM/XGBoost select boosting rounds by weighted inner-validation log loss, then refit on outer train.
ExtraTrees selects min_samples_leaf in {1, 3} by inner balanced accuracy. Median imputation and class weights are fitted inside each training partition.
Importance is normalized per fitted model and summed over features and input channels by storage slot; scale_rank=0 is finest. These model-specific importances are descriptive, not causal.

Comparisons use the corrected resampled t-test and 95% CI, Wilcoxon secondary, Holm across each metric's table. Fewer than two splits produce no inference.

| feature_set    | classifier   | method_a         | method_b           |   mean_diff |   ci_low |   ci_high |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------|:-----------------|:-------------------|------------:|---------:|----------:|------------:|------------------:|
| comp           | lightgbm     | raw              | emd                |      0.0163 |   0.0050 |    0.0277 |      0.0323 |            0.0012 |
| comp           | lightgbm     | raw              | imf_linear_a1.41   |     -0.0950 |  -0.1137 |   -0.0763 |      0.0000 |            0.0012 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41      |     -0.0463 |  -0.0620 |   -0.0307 |      0.0002 |            0.0012 |
| comp           | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.0531 |  -0.0652 |   -0.0411 |      0.0000 |            0.0012 |
| comp           | lightgbm     | emd              | imf_linear_a1.41   |     -0.1113 |  -0.1328 |   -0.0899 |      0.0000 |            0.0012 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0627 |  -0.0782 |   -0.0472 |      0.0000 |            0.0012 |
| comp           | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0695 |  -0.0833 |   -0.0557 |      0.0000 |            0.0012 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0486 |   0.0354 |    0.0619 |      0.0000 |            0.0012 |
| comp           | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0418 |   0.0283 |    0.0554 |      0.0001 |            0.0012 |
| comp           | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0068 |  -0.0154 |    0.0018 |      0.2224 |            0.0067 |
| comp+log_scale | lightgbm     | raw              | emd                |      0.0141 |  -0.0041 |    0.0322 |      0.2224 |            0.0040 |
| comp+log_scale | lightgbm     | raw              | imf_linear_a1.41   |     -0.0506 |  -0.0660 |   -0.0352 |      0.0001 |            0.0012 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41      |     -0.0201 |  -0.0335 |   -0.0067 |      0.0317 |            0.0012 |
| comp+log_scale | lightgbm     | raw              | irmf_h2_a1.41_conv |     -0.0268 |  -0.0399 |   -0.0136 |      0.0047 |            0.0012 |
| comp+log_scale | lightgbm     | emd              | imf_linear_a1.41   |     -0.0647 |  -0.0794 |   -0.0500 |      0.0000 |            0.0012 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41      |     -0.0342 |  -0.0500 |   -0.0184 |      0.0031 |            0.0012 |
| comp+log_scale | lightgbm     | emd              | irmf_h2_a1.41_conv |     -0.0408 |  -0.0573 |   -0.0243 |      0.0010 |            0.0012 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41      |      0.0305 |   0.0195 |    0.0415 |      0.0004 |            0.0012 |
| comp+log_scale | lightgbm     | imf_linear_a1.41 | irmf_h2_a1.41_conv |      0.0239 |   0.0114 |    0.0363 |      0.0064 |            0.0012 |
| comp+log_scale | lightgbm     | irmf_h2_a1.41    | irmf_h2_a1.41_conv |     -0.0067 |  -0.0136 |    0.0003 |      0.1818 |            0.0013 |
