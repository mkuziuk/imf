# slots9, animal

Percentages. References reuse identical main LightGBM fits.

|                                                      |   ('balanced_accuracy', 'mean') |   ('balanced_accuracy', 'std') |   ('macro_f1', 'mean') |   ('macro_f1', 'std') |   ('accuracy', 'mean') |   ('accuracy', 'std') |
|:-----------------------------------------------------|--------------------------------:|-------------------------------:|-----------------------:|----------------------:|-----------------------:|----------------------:|
| ('emd', 'comp', 'lightgbm')                          |                           59.46 |                           4.01 |                  58.75 |                  3.99 |                  59.49 |                  4.11 |
| ('emd', 'comp+log_scale', 'lightgbm')                |                           59.70 |                           2.86 |                  59.05 |                  2.77 |                  59.81 |                  2.96 |
| ('imf_linear_a1.41', 'comp', 'lightgbm')             |                           66.37 |                           2.52 |                  65.86 |                  2.51 |                  66.55 |                  2.59 |
| ('imf_linear_a1.41', 'comp+log_scale', 'lightgbm')   |                           67.49 |                           2.63 |                  67.02 |                  2.57 |                  67.87 |                  2.73 |
| ('irmf_h2_a1.41', 'comp', 'lightgbm')                |                           65.54 |                           3.07 |                  64.87 |                  3.20 |                  65.91 |                  3.15 |
| ('irmf_h2_a1.41', 'comp+log_scale', 'lightgbm')      |                           65.31 |                           2.56 |                  64.72 |                  2.85 |                  65.89 |                  2.69 |
| ('irmf_h2_a1.41_conv', 'comp', 'lightgbm')           |                           65.52 |                           2.92 |                  64.85 |                  3.09 |                  65.89 |                  3.04 |
| ('irmf_h2_a1.41_conv', 'comp+log_scale', 'lightgbm') |                           65.62 |                           2.46 |                  64.97 |                  2.78 |                  66.19 |                  2.50 |
| ('raw', 'comp', 'lightgbm')                          |                           52.37 |                           3.12 |                  51.84 |                  2.96 |                  52.88 |                  3.05 |
| ('raw', 'comp+log_scale', 'lightgbm')                |                           53.58 |                           3.79 |                  53.17 |                  3.55 |                  54.30 |                  3.45 |

Nadeau-Bengio corrected t-test and CI, Wilcoxon secondary. Holm across all pairs and both feature sets per metric.
