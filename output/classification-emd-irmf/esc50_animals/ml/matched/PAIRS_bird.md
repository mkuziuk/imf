# matched, bird: paired balanced accuracy

Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: 56 tests per metric. Wilcoxon is secondary.

| feature_set    | method_a           | method_b           | difference [95% CI], pp   |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------------|:-------------------|:--------------------------|------------:|------------------:|
| comp           | raw                | emd                | -1.94 [-4.01, +0.13]      |     1       |         0.074768  |
| comp           | raw                | imf_linear_a1.41   | -4.35 [-9.32, +0.63]      |     1       |         0.088623  |
| comp           | raw                | irmf_h2_a1.41      | -4.20 [-8.13, -0.28]      |     1       |         0.032349  |
| comp           | raw                | irmf_h2_a1.41_conv | -4.01 [-8.39, +0.36]      |     1       |         0.074768  |
| comp           | raw                | irmf_h2_a2_conv    | -4.14 [-8.18, -0.10]      |     1       |         0.088623  |
| comp           | raw                | imf_linear_a2      | -3.96 [-8.02, +0.11]      |     1       |         0.044434  |
| comp           | raw                | irmf_h2_a2         | -3.81 [-7.75, +0.13]      |     1       |         0.074768  |
| comp           | emd                | imf_linear_a1.41   | -2.40 [-6.25, +1.44]      |     1       |         0.65039   |
| comp           | emd                | irmf_h2_a1.41      | -2.26 [-5.57, +1.05]      |     1       |         0.17944   |
| comp           | emd                | irmf_h2_a1.41_conv | -2.07 [-5.54, +1.40]      |     1       |         0.75409   |
| comp           | emd                | irmf_h2_a2_conv    | -2.20 [-5.27, +0.88]      |     1       |         0.34283   |
| comp           | emd                | imf_linear_a2      | -2.01 [-5.17, +1.14]      |     1       |         0.4856    |
| comp           | emd                | irmf_h2_a2         | -1.87 [-4.93, +1.19]      |     1       |         0.75409   |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41      | +0.14 [-3.04, +3.33]      |     1       |         1         |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41_conv | +0.33 [-2.40, +3.06]      |     1       |         1         |
| comp           | imf_linear_a1.41   | irmf_h2_a2_conv    | +0.21 [-3.04, +3.45]      |     1       |         1         |
| comp           | imf_linear_a1.41   | imf_linear_a2      | +0.39 [-2.02, +2.79]      |     1       |         1         |
| comp           | imf_linear_a1.41   | irmf_h2_a2         | +0.53 [-2.81, +3.88]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | irmf_h2_a1.41_conv | +0.19 [-1.06, +1.44]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | irmf_h2_a2_conv    | +0.06 [-1.88, +2.01]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | imf_linear_a2      | +0.25 [-2.95, +3.44]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | irmf_h2_a2         | +0.39 [-1.17, +1.95]      |     1       |         1         |
| comp           | irmf_h2_a1.41_conv | irmf_h2_a2_conv    | -0.13 [-1.93, +1.68]      |     1       |         1         |
| comp           | irmf_h2_a1.41_conv | imf_linear_a2      | +0.06 [-2.98, +3.10]      |     1       |         1         |
| comp           | irmf_h2_a1.41_conv | irmf_h2_a2         | +0.20 [-1.62, +2.02]      |     1       |         1         |
| comp           | irmf_h2_a2_conv    | imf_linear_a2      | +0.18 [-2.88, +3.25]      |     1       |         1         |
| comp           | irmf_h2_a2_conv    | irmf_h2_a2         | +0.33 [-0.72, +1.37]      |     1       |         1         |
| comp           | imf_linear_a2      | irmf_h2_a2         | +0.14 [-2.85, +3.13]      |     1       |         1         |
| comp+log_scale | raw                | emd                | -2.27 [-3.97, -0.57]      |     0.70001 |         0.0068359 |
| comp+log_scale | raw                | imf_linear_a1.41   | -5.17 [-9.97, -0.37]      |     1       |         0.010071  |
| comp+log_scale | raw                | irmf_h2_a1.41      | -4.46 [-8.63, -0.29]      |     1       |         0.074768  |
| comp+log_scale | raw                | irmf_h2_a1.41_conv | -4.43 [-8.50, -0.37]      |     1       |         0.023071  |
| comp+log_scale | raw                | irmf_h2_a2_conv    | -4.25 [-8.30, -0.20]      |     1       |         0.044434  |
| comp+log_scale | raw                | imf_linear_a2      | -4.53 [-9.53, +0.46]      |     1       |         0.074768  |
| comp+log_scale | raw                | irmf_h2_a2         | -4.42 [-8.33, -0.51]      |     1       |         0.044434  |
| comp+log_scale | emd                | imf_linear_a1.41   | -2.90 [-6.92, +1.13]      |     1       |         0.34283   |
| comp+log_scale | emd                | irmf_h2_a1.41      | -2.19 [-5.79, +1.40]      |     1       |         0.57288   |
| comp+log_scale | emd                | irmf_h2_a1.41_conv | -2.16 [-5.52, +1.19]      |     1       |         0.75409   |
| comp+log_scale | emd                | irmf_h2_a2_conv    | -1.98 [-5.74, +1.79]      |     1       |         1         |
| comp+log_scale | emd                | imf_linear_a2      | -2.26 [-6.60, +2.07]      |     1       |         0.81836   |
| comp+log_scale | emd                | irmf_h2_a2         | -2.15 [-5.57, +1.26]      |     1       |         0.57288   |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41      | +0.71 [-2.22, +3.63]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41_conv | +0.74 [-1.82, +3.30]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a2_conv    | +0.92 [-2.84, +4.68]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | imf_linear_a2      | +0.64 [-2.35, +3.62]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a2         | +0.75 [-2.68, +4.18]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a1.41_conv | +0.03 [-1.03, +1.09]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a2_conv    | +0.21 [-1.49, +1.92]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | imf_linear_a2      | -0.07 [-4.49, +4.34]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a2         | +0.04 [-1.45, +1.53]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h2_a2_conv    | +0.18 [-1.79, +2.15]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | imf_linear_a2      | -0.10 [-4.24, +4.04]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | irmf_h2_a2         | +0.01 [-1.79, +1.81]      |     1       |         1         |
| comp+log_scale | irmf_h2_a2_conv    | imf_linear_a2      | -0.28 [-5.02, +4.45]      |     1       |         1         |
| comp+log_scale | irmf_h2_a2_conv    | irmf_h2_a2         | -0.17 [-1.67, +1.32]      |     1       |         1         |
| comp+log_scale | imf_linear_a2      | irmf_h2_a2         | +0.11 [-4.26, +4.48]      |     1       |         1         |
