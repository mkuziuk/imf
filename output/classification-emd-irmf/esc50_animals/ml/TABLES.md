# Phase-3 score tables

Percentages, outer-fold means. SDs and every metric are in scores.csv.

## animal, comp

| method             |   compact_resnet |   extratrees |   lightgbm |   minirocket |   multirocket |   xgboost |
|:-------------------|-----------------:|-------------:|-----------:|-------------:|--------------:|----------:|
| emd                |            51.46 |        56.22 |      59.46 |        57.28 |         54.27 |     57.24 |
| imf_linear_a1.41   |            52.49 |        65.79 |      67.18 |        60.81 |         59.63 |     65.77 |
| irmf_h2_a1.41      |            50.84 |        64.94 |      66.32 |        58.93 |         58.92 |     64.08 |
| irmf_h2_a1.41_conv |            51.03 |        65.03 |      65.89 |        59.25 |         58.22 |     64.05 |
| raw                |            52.59 |        52.14 |      52.37 |        56.49 |         55.87 |     50.28 |

Raw log-STFT 2-D reference: 53.93% BA.

## animal, comp+log_scale

| method             |   compact_resnet |   extratrees |   lightgbm |   minirocket |   multirocket |   xgboost |
|:-------------------|-----------------:|-------------:|-----------:|-------------:|--------------:|----------:|
| emd                |            54.34 |        57.09 |      59.70 |        57.95 |         54.93 |     57.33 |
| imf_linear_a1.41   |            53.05 |        65.81 |      67.48 |        60.73 |         60.42 |     66.07 |
| irmf_h2_a1.41      |            55.96 |        64.93 |      67.08 |        59.32 |         60.01 |     64.88 |
| irmf_h2_a1.41_conv |            54.31 |        65.12 |      66.40 |        59.47 |         59.24 |     64.45 |
| raw                |            53.83 |        53.16 |      53.58 |        57.19 |         57.92 |     51.51 |

Raw log-STFT 2-D reference: 55.47% BA.

## bird, comp

| method             |   compact_resnet |   extratrees |   lightgbm |   minirocket |   multirocket |   xgboost |
|:-------------------|-----------------:|-------------:|-----------:|-------------:|--------------:|----------:|
| emd                |            71.98 |        71.98 |      75.81 |        74.93 |         73.73 |     75.16 |
| imf_linear_a1.41   |            73.92 |        74.68 |      78.21 |        76.15 |         74.97 |     77.28 |
| irmf_h2_a1.41      |            75.91 |        75.30 |      78.07 |        75.47 |         74.81 |     77.47 |
| irmf_h2_a1.41_conv |            74.93 |        75.27 |      77.88 |        75.34 |         74.82 |     77.43 |
| raw                |            73.04 |        72.66 |      73.86 |        75.46 |         74.98 |     73.03 |

Raw log-STFT 2-D reference: 68.13% BA.

## bird, comp+log_scale

| method             |   compact_resnet |   extratrees |   lightgbm |   minirocket |   multirocket |   xgboost |
|:-------------------|-----------------:|-------------:|-----------:|-------------:|--------------:|----------:|
| emd                |            74.43 |        72.25 |      75.45 |        74.50 |         73.39 |     75.07 |
| imf_linear_a1.41   |            72.20 |        75.21 |      78.35 |        76.17 |         74.82 |     77.46 |
| irmf_h2_a1.41      |            74.93 |        75.13 |      77.65 |        74.89 |         74.10 |     77.04 |
| irmf_h2_a1.41_conv |            73.19 |        75.40 |      77.62 |        74.74 |         73.62 |     77.08 |
| raw                |            72.53 |        73.43 |      73.19 |        75.25 |         74.15 |     73.19 |

Raw log-STFT 2-D reference: 65.85% BA.
