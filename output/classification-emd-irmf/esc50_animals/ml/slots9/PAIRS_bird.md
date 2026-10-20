# slots9, bird: paired balanced accuracy

Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: 20 tests per metric. Wilcoxon is secondary.

| feature_set    | method_a         | method_b           | difference [95% CI], pp   |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-----------------|:-------------------|:--------------------------|------------:|------------------:|
| comp           | raw              | emd                | -1.94 [-4.01, +0.13]      |        1    |         0.028992  |
| comp           | raw              | imf_linear_a1.41   | -3.48 [-8.22, +1.27]      |        1    |         0.093994  |
| comp           | raw              | irmf_h2_a1.41      | -2.69 [-6.31, +0.93]      |        1    |         0.093994  |
| comp           | raw              | irmf_h2_a1.41_conv | -2.86 [-6.27, +0.55]      |        1    |         0.053711  |
| comp           | emd              | imf_linear_a1.41   | -1.54 [-5.40, +2.32]      |        1    |         0.91577   |
| comp           | emd              | irmf_h2_a1.41      | -0.75 [-3.54, +2.04]      |        1    |         1         |
| comp           | emd              | irmf_h2_a1.41_conv | -0.92 [-3.47, +1.63]      |        1    |         1         |
| comp           | imf_linear_a1.41 | irmf_h2_a1.41      | +0.79 [-1.56, +3.13]      |        1    |         0.76465   |
| comp           | imf_linear_a1.41 | irmf_h2_a1.41_conv | +0.62 [-1.90, +3.14]      |        1    |         1         |
| comp           | irmf_h2_a1.41    | irmf_h2_a1.41_conv | -0.17 [-0.87, +0.53]      |        1    |         1         |
| comp+log_scale | raw              | emd                | -2.27 [-3.97, -0.57]      |        0.25 |         0.0024414 |
| comp+log_scale | raw              | imf_linear_a1.41   | -4.30 [-9.33, +0.73]      |        1    |         0.047241  |
| comp+log_scale | raw              | irmf_h2_a1.41      | -3.08 [-7.05, +0.89]      |        1    |         0.053711  |
| comp+log_scale | raw              | irmf_h2_a1.41_conv | -3.40 [-7.37, +0.56]      |        1    |         0.047241  |
| comp+log_scale | emd              | imf_linear_a1.41   | -2.03 [-6.42, +2.36]      |        1    |         0.91577   |
| comp+log_scale | emd              | irmf_h2_a1.41      | -0.81 [-4.12, +2.50]      |        1    |         1         |
| comp+log_scale | emd              | irmf_h2_a1.41_conv | -1.14 [-4.49, +2.22]      |        1    |         1         |
| comp+log_scale | imf_linear_a1.41 | irmf_h2_a1.41      | +1.22 [-1.77, +4.21]      |        1    |         0.91577   |
| comp+log_scale | imf_linear_a1.41 | irmf_h2_a1.41_conv | +0.89 [-2.02, +3.81]      |        1    |         1         |
| comp+log_scale | irmf_h2_a1.41    | irmf_h2_a1.41_conv | -0.33 [-1.26, +0.60]      |        1    |         1         |
