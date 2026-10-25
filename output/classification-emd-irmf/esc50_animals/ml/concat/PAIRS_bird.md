# concat, bird: paired balanced accuracy

Nadeau-Bengio corrected t-test and ordinary corrected 95% intervals. Holm family: 42 tests per metric. Wilcoxon is secondary.

| feature_set    | method_a           | method_b                | difference [95% CI], pp   |   p_nb_holm |   p_wilcoxon_holm |
|:---------------|:-------------------|:------------------------|:--------------------------|------------:|------------------:|
| comp           | raw                | emd                     | -1.94 [-4.01, +0.13]      |     1       |         0.05188   |
| comp           | raw                | imf_linear_a1.41        | -4.35 [-9.32, +0.63]      |     1       |         0.062439  |
| comp           | raw                | irmf_h2_a1.41           | -4.20 [-8.13, -0.28]      |     1       |         0.022583  |
| comp           | raw                | irmf_h2_a1.41_conv      | -4.01 [-8.39, +0.36]      |     1       |         0.05188   |
| comp           | raw                | emd__irmf_h2_a1.41      | -4.03 [-7.98, -0.07]      |     1       |         0.030762  |
| comp           | raw                | emd__irmf_h2_a1.41_conv | -4.10 [-7.98, -0.22]      |     1       |         0.030762  |
| comp           | emd                | imf_linear_a1.41        | -2.40 [-6.25, +1.44]      |     1       |         0.41553   |
| comp           | emd                | irmf_h2_a1.41           | -2.26 [-5.57, +1.05]      |     1       |         0.12817   |
| comp           | emd                | irmf_h2_a1.41_conv      | -2.07 [-5.54, +1.40]      |     1       |         0.474     |
| comp           | emd                | emd__irmf_h2_a1.41      | -2.08 [-5.20, +1.03]      |     1       |         0.37689   |
| comp           | emd                | emd__irmf_h2_a1.41_conv | -2.15 [-5.10, +0.79]      |     1       |         0.15576   |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41           | +0.14 [-3.04, +3.33]      |     1       |         1         |
| comp           | imf_linear_a1.41   | irmf_h2_a1.41_conv      | +0.33 [-2.40, +3.06]      |     1       |         1         |
| comp           | imf_linear_a1.41   | emd__irmf_h2_a1.41      | +0.32 [-2.40, +3.04]      |     1       |         1         |
| comp           | imf_linear_a1.41   | emd__irmf_h2_a1.41_conv | +0.25 [-3.04, +3.53]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | irmf_h2_a1.41_conv      | +0.19 [-1.06, +1.44]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | emd__irmf_h2_a1.41      | +0.18 [-0.91, +1.26]      |     1       |         1         |
| comp           | irmf_h2_a1.41      | emd__irmf_h2_a1.41_conv | +0.11 [-1.01, +1.22]      |     1       |         1         |
| comp           | irmf_h2_a1.41_conv | emd__irmf_h2_a1.41      | -0.01 [-1.35, +1.32]      |     1       |         1         |
| comp           | irmf_h2_a1.41_conv | emd__irmf_h2_a1.41_conv | -0.08 [-1.34, +1.17]      |     1       |         1         |
| comp           | emd__irmf_h2_a1.41 | emd__irmf_h2_a1.41_conv | -0.07 [-1.38, +1.24]      |     1       |         1         |
| comp+log_scale | raw                | emd                     | -2.27 [-3.97, -0.57]      |     0.525   |         0.0050049 |
| comp+log_scale | raw                | imf_linear_a1.41        | -5.17 [-9.97, -0.37]      |     1       |         0.0071411 |
| comp+log_scale | raw                | irmf_h2_a1.41           | -4.46 [-8.63, -0.29]      |     1       |         0.05188   |
| comp+log_scale | raw                | irmf_h2_a1.41_conv      | -4.43 [-8.50, -0.37]      |     1       |         0.016235  |
| comp+log_scale | raw                | emd__irmf_h2_a1.41      | -4.64 [-8.66, -0.61]      |     1       |         0.0025635 |
| comp+log_scale | raw                | emd__irmf_h2_a1.41_conv | -4.50 [-8.14, -0.85]      |     0.78463 |         0.0050049 |
| comp+log_scale | emd                | imf_linear_a1.41        | -2.90 [-6.92, +1.13]      |     1       |         0.22577   |
| comp+log_scale | emd                | irmf_h2_a1.41           | -2.19 [-5.79, +1.40]      |     1       |         0.37689   |
| comp+log_scale | emd                | irmf_h2_a1.41_conv      | -2.16 [-5.52, +1.19]      |     1       |         0.474     |
| comp+log_scale | emd                | emd__irmf_h2_a1.41      | -2.37 [-5.71, +0.98]      |     1       |         0.2666    |
| comp+log_scale | emd                | emd__irmf_h2_a1.41_conv | -2.23 [-5.10, +0.64]      |     1       |         0.15576   |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41           | +0.71 [-2.22, +3.63]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | irmf_h2_a1.41_conv      | +0.74 [-1.82, +3.30]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | emd__irmf_h2_a1.41      | +0.53 [-2.28, +3.34]      |     1       |         1         |
| comp+log_scale | imf_linear_a1.41   | emd__irmf_h2_a1.41_conv | +0.67 [-1.60, +2.95]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | irmf_h2_a1.41_conv      | +0.03 [-1.03, +1.09]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | emd__irmf_h2_a1.41      | -0.18 [-1.36, +1.00]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41      | emd__irmf_h2_a1.41_conv | -0.04 [-1.39, +1.32]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | emd__irmf_h2_a1.41      | -0.21 [-1.33, +0.92]      |     1       |         1         |
| comp+log_scale | irmf_h2_a1.41_conv | emd__irmf_h2_a1.41_conv | -0.07 [-1.29, +1.16]      |     1       |         1         |
| comp+log_scale | emd__irmf_h2_a1.41 | emd__irmf_h2_a1.41_conv | +0.14 [-1.39, +1.67]      |     1       |         1         |
