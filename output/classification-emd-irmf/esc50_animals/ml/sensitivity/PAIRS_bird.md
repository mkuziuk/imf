# sensitivity, bird: paired balanced accuracy

Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: 56 tests per metric. Wilcoxon is secondary.

| feature_set    | method_a           | method_b           | difference [95% CI], pp   |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------------|:-------------------|:--------------------------|------------:|------------------:|
| comp           | raw                | emd                | -1.94 [-4.01, +0.13]      |     1       |         0.073242  |
| comp           | raw                | imf_linear_a1.41   | -4.35 [-9.32, +0.63]      |     1       |         0.090637  |
| comp           | raw                | irmf_h2_a1.41      | -4.20 [-8.13, -0.28]      |     1       |         0.031128  |
| comp           | raw                | irmf_h2_a1.41_conv | -4.01 [-8.39, +0.36]      |     1       |         0.073242  |
| comp           | raw                | irmf_h1_a1.41_conv | -3.95 [-8.20, +0.29]      |     1       |         0.056824  |
| comp           | raw                | irmf_h4_a1.41_conv | -3.93 [-8.86, +1.00]      |     1       |         0.11285   |
| comp           | raw                | irmf_h2_a2_conv    | -4.14 [-8.18, -0.10]      |     1       |         0.090637  |
| comp           | emd                | imf_linear_a1.41   | -2.40 [-6.25, +1.44]      |     1       |         0.65039   |
| comp           | emd                | irmf_h2_a1.41      | -2.26 [-5.57, +1.05]      |     1       |         0.17944   |
| comp           | emd                | irmf_h2_a1.41_conv | -2.07 [-5.54, +1.40]      |     1       |         0.75409   |
| comp           | emd                | irmf_h1_a1.41_conv | -2.01 [-5.20, +1.19]      |     1       |         0.58795   |
| comp           | emd                | irmf_h4_a1.41_conv | -1.98 [-5.66, +1.69]      |     1       |         1         |
| comp           | emd                | irmf_h2_a2_conv    | -2.20 [-5.27, +0.88]      |     1       |         0.34283   |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41      | +0.14 [-3.04, +3.33]      |     1       |         1         |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41_conv | +0.33 [-2.40, +3.06]      |     1       |         1         |
| comp           | imf_linear_a1.41   | irmf_h1_a1.41_conv | +0.39 [-2.62, +3.41]      |     1       |         1         |
| comp           | imf_linear_a1.41   | irmf_h4_a1.41_conv | +0.42 [-1.61, +2.45]      |     1       |         1         |
| comp           | imf_linear_a1.41   | irmf_h2_a2_conv    | +0.21 [-3.04, +3.45]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | irmf_h2_a1.41_conv | +0.19 [-1.06, +1.44]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | irmf_h1_a1.41_conv | +0.25 [-1.20, +1.71]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | irmf_h4_a1.41_conv | +0.28 [-1.92, +2.47]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | irmf_h2_a2_conv    | +0.06 [-1.88, +2.01]      |     1       |         1         |
| comp           | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv | +0.06 [-1.52, +1.64]      |     1       |         1         |
| comp           | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv | +0.09 [-1.79, +1.96]      |     1       |         1         |
| comp           | irmf_h2_a1.41_conv | irmf_h2_a2_conv    | -0.13 [-1.93, +1.68]      |     1       |         1         |
| comp           | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv | +0.02 [-2.11, +2.16]      |     1       |         1         |
| comp           | irmf_h1_a1.41_conv | irmf_h2_a2_conv    | -0.19 [-1.92, +1.54]      |     1       |         1         |
| comp           | irmf_h4_a1.41_conv | irmf_h2_a2_conv    | -0.21 [-2.32, +1.89]      |     1       |         1         |
| comp+log_scale | raw                | emd                | -2.27 [-3.97, -0.57]      |     0.70001 |         0.0068359 |
| comp+log_scale | raw                | imf_linear_a1.41   | -5.17 [-9.97, -0.37]      |     1       |         0.010071  |
| comp+log_scale | raw                | irmf_h2_a1.41      | -4.46 [-8.63, -0.29]      |     1       |         0.073242  |
| comp+log_scale | raw                | irmf_h2_a1.41_conv | -4.43 [-8.50, -0.37]      |     1       |         0.022644  |
| comp+log_scale | raw                | irmf_h1_a1.41_conv | -4.63 [-8.59, -0.67]      |     1       |         0.016479  |
| comp+log_scale | raw                | irmf_h4_a1.41_conv | -4.47 [-8.97, +0.03]      |     1       |         0.022644  |
| comp+log_scale | raw                | irmf_h2_a2_conv    | -4.25 [-8.30, -0.20]      |     1       |         0.042725  |
| comp+log_scale | emd                | imf_linear_a1.41   | -2.90 [-6.92, +1.13]      |     1       |         0.34283   |
| comp+log_scale | emd                | irmf_h2_a1.41      | -2.19 [-5.79, +1.40]      |     1       |         0.58795   |
| comp+log_scale | emd                | irmf_h2_a1.41_conv | -2.16 [-5.52, +1.19]      |     1       |         0.75409   |
| comp+log_scale | emd                | irmf_h1_a1.41_conv | -2.36 [-5.85, +1.13]      |     1       |         0.58795   |
| comp+log_scale | emd                | irmf_h4_a1.41_conv | -2.20 [-6.17, +1.78]      |     1       |         1         |
| comp+log_scale | emd                | irmf_h2_a2_conv    | -1.98 [-5.74, +1.79]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41      | +0.71 [-2.22, +3.63]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41_conv | +0.74 [-1.82, +3.30]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h1_a1.41_conv | +0.54 [-3.15, +4.22]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h4_a1.41_conv | +0.70 [-1.97, +3.37]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a2_conv    | +0.92 [-2.84, +4.68]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a1.41_conv | +0.03 [-1.03, +1.09]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h1_a1.41_conv | -0.17 [-1.77, +1.43]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h4_a1.41_conv | -0.00 [-1.67, +1.66]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a2_conv    | +0.21 [-1.49, +1.92]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h1_a1.41_conv | -0.20 [-1.77, +1.38]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h4_a1.41_conv | -0.04 [-1.67, +1.60]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h2_a2_conv    | +0.18 [-1.79, +2.15]      |     1       |         1         |
| comp+log_scale | irmf_h1_a1.41_conv | irmf_h4_a1.41_conv | +0.16 [-2.28, +2.61]      |     1       |         1         |
| comp+log_scale | irmf_h1_a1.41_conv | irmf_h2_a2_conv    | +0.38 [-1.38, +2.15]      |     1       |         1         |
| comp+log_scale | irmf_h4_a1.41_conv | irmf_h2_a2_conv    | +0.22 [-1.96, +2.40]      |     1       |         1         |
