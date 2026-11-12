# Errors on held-out poses

Each cell is evaluated once per seed, three times overall. Denominators are 2,352 cell evaluations or 147 recording evaluations per method/model/task/feature set. Counts are correlated, not independent samples.
Groups ever wrong count distinct poses across all three seeds, out of 45. Recording aggregation averages probabilities; majority-vote alternatives are in error_counts.csv and recording_errors.csv.
LightGBM is fixed below. Every classifier, class and recording is retained in the CSV files.

## bird_drone, comp

| method             |   bird |   uas |   cell-error poses /45 |   recording errors /147 |   record-error poses /45 |
|:-------------------|-------:|------:|-----------------------:|------------------------:|-------------------------:|
| raw                |     47 |    45 |                     14 |                       0 |                        0 |
| emd                |     40 |    47 |                     18 |                       0 |                        0 |
| imf_linear_a1.41   |     25 |    23 |                      9 |                       0 |                        0 |
| irmf_h2_a1.41      |     13 |    10 |                      7 |                       0 |                        0 |
| irmf_h2_a1.41_conv |     13 |     6 |                      6 |                       0 |                        0 |
| irmf_h2_a2_conv    |     12 |    13 |                      8 |                       0 |                        0 |
| imf_linear_a2      |     21 |    16 |                      6 |                       0 |                        0 |
| irmf_h1_a1.41_conv |     13 |     8 |                      6 |                       0 |                        0 |
| irmf_h4_a1.41_conv |     16 |    11 |                      6 |                       0 |                        0 |

## bird_drone, comp+log_scale

| method             |   bird |   uas |   cell-error poses /45 |   recording errors /147 |   record-error poses /45 |
|:-------------------|-------:|------:|-----------------------:|------------------------:|-------------------------:|
| raw                |     38 |    25 |                     12 |                       0 |                        0 |
| emd                |     27 |    37 |                     18 |                       0 |                        0 |
| imf_linear_a1.41   |     23 |    22 |                      9 |                       0 |                        0 |
| irmf_h2_a1.41      |     12 |    11 |                      7 |                       0 |                        0 |
| irmf_h2_a1.41_conv |     13 |     7 |                      7 |                       0 |                        0 |
| irmf_h2_a2_conv    |     15 |    14 |                      8 |                       0 |                        0 |
| imf_linear_a2      |     23 |    15 |                      6 |                       0 |                        0 |
| irmf_h1_a1.41_conv |     11 |     9 |                      5 |                       0 |                        0 |
| irmf_h4_a1.41_conv |     16 |    14 |                      7 |                       0 |                        0 |

## bird_drone, recordings ever wrong for GD60 IRMF LightGBM

| group_recording                         | group                      | y_true   | feature_set    |   n_frame_errors |   n_frame_evaluations |   errors |
|:----------------------------------------|:---------------------------|:---------|:---------------|-----------------:|----------------------:|---------:|
| cut_bird_0_100pct                       | bird_0                     | bird     | comp           |               10 |                    48 |        0 |
| cut_p3p_0_rotation                      | p3p_0                      | uas      | comp           |                5 |                    48 |        0 |
| cut_mavik_tripod_az_0_tilt_-45_rotation | mavic_tripod_az_0_tilt_-45 | uas      | comp           |                3 |                    48 |        0 |
| cut_bird_tripod_az_0_tilt_-45_50pct     | bird_tripod_az_0_tilt_-45  | bird     | comp           |                2 |                    48 |        0 |
| cut_bird_tripod_az_0_tilt_+45_50pct     | bird_tripod_az_0_tilt_+45  | bird     | comp           |                1 |                    48 |        0 |
| cut_p3p_45_rotation                     | p3p_45                     | uas      | comp           |                1 |                    48 |        0 |
| cut_p3p_90_rotation                     | p3p_90                     | uas      | comp           |                1 |                    48 |        0 |
| cut_bird_0_100pct                       | bird_0                     | bird     | comp+log_scale |               10 |                    48 |        0 |
| cut_p3p_0_rotation                      | p3p_0                      | uas      | comp+log_scale |                6 |                    48 |        0 |
| cut_mavik_tripod_az_0_tilt_-45_rotation | mavic_tripod_az_0_tilt_-45 | uas      | comp+log_scale |                3 |                    48 |        0 |
| cut_bird_tripod_az_0_tilt_+45_50pct     | bird_tripod_az_0_tilt_+45  | bird     | comp+log_scale |                1 |                    48 |        0 |
| cut_bird_tripod_az_0_tilt_-45_50pct     | bird_tripod_az_0_tilt_-45  | bird     | comp+log_scale |                1 |                    48 |        0 |
| cut_p3p_45_rotation                     | p3p_45                     | uas      | comp+log_scale |                1 |                    48 |        0 |
| cut_p3p_90_rotation                     | p3p_90                     | uas      | comp+log_scale |                1 |                    48 |        0 |

## target3, comp

| method             |   bird |   mavic |   p3p |   cell-error poses /45 |   recording errors /147 |   record-error poses /45 |
|:-------------------|-------:|--------:|------:|-----------------------:|------------------------:|-------------------------:|
| raw                |     51 |      72 |    57 |                     28 |                       0 |                        0 |
| emd                |     28 |      45 |    43 |                     20 |                       0 |                        0 |
| imf_linear_a1.41   |     28 |      44 |    41 |                     18 |                       0 |                        0 |
| irmf_h2_a1.41      |     18 |      26 |    29 |                     12 |                       0 |                        0 |
| irmf_h2_a1.41_conv |     20 |      26 |    26 |                     13 |                       0 |                        0 |
| irmf_h2_a2_conv    |     16 |      29 |    35 |                     15 |                       0 |                        0 |
| imf_linear_a2      |     29 |      38 |    43 |                     16 |                       0 |                        0 |
| irmf_h1_a1.41_conv |     24 |      17 |    19 |                     11 |                       0 |                        0 |
| irmf_h4_a1.41_conv |     24 |      36 |    35 |                     16 |                       0 |                        0 |

## target3, comp+log_scale

| method             |   bird |   mavic |   p3p |   cell-error poses /45 |   recording errors /147 |   record-error poses /45 |
|:-------------------|-------:|--------:|------:|-----------------------:|------------------------:|-------------------------:|
| raw                |     31 |      47 |    50 |                     23 |                       0 |                        0 |
| emd                |     18 |      48 |    41 |                     21 |                       0 |                        0 |
| imf_linear_a1.41   |     27 |      46 |    43 |                     16 |                       0 |                        0 |
| irmf_h2_a1.41      |     18 |      26 |    28 |                     12 |                       0 |                        0 |
| irmf_h2_a1.41_conv |     20 |      25 |    28 |                     13 |                       0 |                        0 |
| irmf_h2_a2_conv    |     17 |      32 |    35 |                     16 |                       0 |                        0 |
| imf_linear_a2      |     28 |      38 |    42 |                     15 |                       0 |                        0 |
| irmf_h1_a1.41_conv |     24 |      16 |    18 |                     10 |                       0 |                        0 |
| irmf_h4_a1.41_conv |     25 |      35 |    38 |                     17 |                       0 |                        0 |

## target3, recordings ever wrong for GD60 IRMF LightGBM

| group_recording                          | group                       | y_true   | feature_set    |   n_frame_errors |   n_frame_evaluations |   errors |
|:-----------------------------------------|:----------------------------|:---------|:---------------|-----------------:|----------------------:|---------:|
| cut_bird_0_100pct                        | bird_0                      | bird     | comp           |               18 |                    48 |        0 |
| cut_mavik_tripod_az_0_tilt_-45_rotation  | mavic_tripod_az_0_tilt_-45  | mavic    | comp           |                9 |                    48 |        0 |
| cut_p3p_135_rotation                     | p3p_135                     | p3p      | comp           |                8 |                    48 |        0 |
| cut_mavik_tripod_az_90_tilt_0_rotation   | mavic_tripod_az_90_tilt_0   | mavic    | comp           |                6 |                    48 |        0 |
| cut_p3p_180_rotation                     | p3p_180                     | p3p      | comp           |                6 |                    48 |        0 |
| cut_mavik_67.5_rotation                  | mavic_67.5                  | mavic    | comp           |                5 |                    48 |        0 |
| cut_p3p_0_rotation                       | p3p_0                       | p3p      | comp           |                5 |                    48 |        0 |
| cut_p3p_22.5_rotation                    | p3p_22.5                    | p3p      | comp           |                5 |                    48 |        0 |
| cut_mavik_0_rotation                     | mavic_0                     | mavic    | comp           |                4 |                    48 |        0 |
| cut_p3p_157.5_rotation                   | p3p_157.5                   | p3p      | comp           |                3 |                    48 |        0 |
| cut_mavik_tripod_az_45_tilt_-45_rotation | mavic_tripod_az_45_tilt_-45 | mavic    | comp           |                2 |                    48 |        0 |
| cut_p3p_67.5_rotation                    | p3p_67.5                    | p3p      | comp           |                2 |                    48 |        0 |
| cut_bird_0_100pct                        | bird_0                      | bird     | comp+log_scale |               18 |                    48 |        0 |
| cut_mavik_tripod_az_0_tilt_-45_rotation  | mavic_tripod_az_0_tilt_-45  | mavic    | comp+log_scale |                9 |                    48 |        0 |
| cut_p3p_135_rotation                     | p3p_135                     | p3p      | comp+log_scale |                8 |                    48 |        0 |
| cut_mavik_tripod_az_90_tilt_0_rotation   | mavic_tripod_az_90_tilt_0   | mavic    | comp+log_scale |                6 |                    48 |        0 |
| cut_p3p_180_rotation                     | p3p_180                     | p3p      | comp+log_scale |                6 |                    48 |        0 |
| cut_mavik_67.5_rotation                  | mavic_67.5                  | mavic    | comp+log_scale |                5 |                    48 |        0 |
| cut_p3p_0_rotation                       | p3p_0                       | p3p      | comp+log_scale |                5 |                    48 |        0 |
| cut_p3p_22.5_rotation                    | p3p_22.5                    | p3p      | comp+log_scale |                5 |                    48 |        0 |
| cut_mavik_0_rotation                     | mavic_0                     | mavic    | comp+log_scale |                4 |                    48 |        0 |
| cut_p3p_157.5_rotation                   | p3p_157.5                   | p3p      | comp+log_scale |                3 |                    48 |        0 |
| cut_mavik_tripod_az_45_tilt_-45_rotation | mavic_tripod_az_45_tilt_-45 | mavic    | comp+log_scale |                2 |                    48 |        0 |
| cut_p3p_67.5_rotation                    | p3p_67.5                    | p3p      | comp+log_scale |                1 |                    48 |        0 |
