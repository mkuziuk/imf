# sensitivity, animal

Percentages. References reuse identical main LightGBM fits.

|                                                      |   ('balanced_accuracy', 'mean') |   ('balanced_accuracy', 'std') |   ('macro_f1', 'mean') |   ('macro_f1', 'std') |   ('accuracy', 'mean') |   ('accuracy', 'std') |
|:-----------------------------------------------------|--------------------------------:|-------------------------------:|-----------------------:|----------------------:|-----------------------:|----------------------:|
| ('emd', 'comp', 'lightgbm')                          |                           59.46 |                           4.01 |                  58.75 |                  3.99 |                  59.49 |                  4.11 |
| ('emd', 'comp+log_scale', 'lightgbm')                |                           59.70 |                           2.86 |                  59.05 |                  2.77 |                  59.81 |                  2.96 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             |                           67.18 |                           3.23 |                  66.75 |                  3.27 |                  67.40 |                  3.43 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   |                           67.48 |                           2.70 |                  67.07 |                  2.82 |                  67.92 |                  2.95 |
| ('irmf_h1_a1.41_conv', 'comp', 'lightgbm')           |                           65.23 |                           3.21 |                  64.59 |                  3.22 |                  65.61 |                  3.39 |
| ('irmf_h1_a1.41_conv', 'comp+log_scale', 'lightgbm') |                           66.16 |                           3.08 |                  65.52 |                  3.28 |                  66.70 |                  3.17 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                |                           66.32 |                           3.05 |                  65.72 |                  3.12 |                  66.68 |                  3.19 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      |                           67.08 |                           3.14 |                  66.54 |                  3.31 |                  67.59 |                  3.32 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           |                           65.89 |                           3.12 |                  65.28 |                  3.19 |                  66.20 |                  3.29 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') |                           66.40 |                           3.09 |                  65.89 |                  3.31 |                  66.91 |                  3.27 |
| ('irmf_h2_a2_conv', 'comp', 'lightgbm')              |                           64.88 |                           3.72 |                  64.35 |                  3.88 |                  65.23 |                  3.98 |
| ('irmf_h2_a2_conv', 'comp+log_scale', 'lightgbm')    |                           65.34 |                           3.60 |                  64.70 |                  3.85 |                  65.79 |                  3.74 |
| ('irmf_h4_a1.41_conv', 'comp', 'lightgbm')           |                           66.79 |                           3.49 |                  66.13 |                  3.79 |                  67.18 |                  3.74 |
| ('irmf_h4_a1.41_conv', 'comp+log_scale', 'lightgbm') |                           67.32 |                           3.23 |                  66.70 |                  3.51 |                  67.88 |                  3.44 |
| ('raw', 'comp', 'lightgbm')                          |                           52.37 |                           3.12 |                  51.84 |                  2.96 |                  52.88 |                  3.05 |
| ('raw', 'comp+log_scale', 'lightgbm')                |                           53.58 |                           3.79 |                  53.17 |                  3.55 |                  54.30 |                  3.45 |

Nadeau-Bengio corrected t-test and CI, Wilcoxon secondary. Holm across all pairs and both feature sets per metric.
