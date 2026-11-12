# slots9, animal: paired balanced accuracy

Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: 20 tests per metric. Wilcoxon is secondary.

| feature_set    | method_a         | method_b           | difference [95% CI], pp   |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-----------------|:-------------------|:--------------------------|------------:|------------------:|
| comp           | raw              | emd                | -7.09 [-12.83, -1.35]     |  0.15268    |         0.0021362 |
| comp           | raw              | imf_linear_a1.41   | -14.01 [-18.55, -9.47]    |  0.00023172 |         0.0012207 |
| comp           | raw              | irmf_h2_a1.41      | -13.17 [-18.75, -7.60]    |  0.0025853  |         0.0012207 |
| comp           | raw              | irmf_h2_a1.41_conv | -13.15 [-18.28, -8.03]    |  0.0013967  |         0.0012207 |
| comp           | emd              | imf_linear_a1.41   | -6.92 [-10.70, -3.13]     |  0.018428   |         0.0012207 |
| comp           | emd              | irmf_h2_a1.41      | -6.09 [-9.77, -2.40]      |  0.029421   |         0.0012207 |
| comp           | emd              | irmf_h2_a1.41_conv | -6.07 [-9.49, -2.65]      |  0.019299   |         0.0012207 |
| comp           | imf_linear_a1.41 | irmf_h2_a1.41      | +0.83 [-0.74, +2.41]      |  0.82773    |         0.10602   |
| comp           | imf_linear_a1.41 | irmf_h2_a1.41_conv | +0.85 [-0.22, +1.92]      |  0.43865    |         0.005127  |
| comp           | irmf_h2_a1.41    | irmf_h2_a1.41_conv | +0.02 [-0.90, +0.94]      |  0.96604    |         0.97797   |
| comp+log_scale | raw              | emd                | -6.12 [-11.07, -1.16]     |  0.15268    |         0.0012207 |
| comp+log_scale | raw              | imf_linear_a1.41   | -13.91 [-18.50, -9.31]    |  0.00027148 |         0.0012207 |
| comp+log_scale | raw              | irmf_h2_a1.41      | -11.73 [-17.08, -6.38]    |  0.0047351  |         0.0012207 |
| comp+log_scale | raw              | irmf_h2_a1.41_conv | -12.04 [-16.99, -7.09]    |  0.0022021  |         0.0012207 |
| comp+log_scale | emd              | imf_linear_a1.41   | -7.79 [-11.02, -4.57]     |  0.0022388  |         0.0012207 |
| comp+log_scale | emd              | irmf_h2_a1.41      | -5.61 [-8.70, -2.53]      |  0.018428   |         0.0012207 |
| comp+log_scale | emd              | irmf_h2_a1.41_conv | -5.92 [-9.01, -2.84]      |  0.013638   |         0.0012207 |
| comp+log_scale | imf_linear_a1.41 | irmf_h2_a1.41      | +2.18 [-0.29, +4.64]      |  0.39475    |         0.01709   |
| comp+log_scale | imf_linear_a1.41 | irmf_h2_a1.41_conv | +1.87 [-0.03, +3.77]      |  0.32052    |         0.0057983 |
| comp+log_scale | irmf_h2_a1.41    | irmf_h2_a1.41_conv | -0.31 [-1.13, +0.51]      |  0.86376    |         0.21399   |
