# ESC-50 animals: phase-3

Animal LightGBM IRMF−EMD: +6.86 pp, corrected 95% CI [3.30,10.42], Holm p=.0666 without log scale; +7.39 [4.63,10.15], p=.00421 with it.

Frame balanced accuracy %, without/with log_scale. Mini/Multi are ROCKET variants; IRMF is GD60, Conv is converged Newton IRMF. All 5,100 frames use uploader-held-out splits.

| Task/method | LGBM | XGB | ET | Mini | CNN | Multi |
|---|---:|---:|---:|---:|---:|---:|
| animal/Raw | 52.37/53.58 | 50.28/51.51 | 52.14/53.16 | 56.49/57.19 | 52.59/53.83 | 55.87/57.92 |
| animal/EMD | 59.46/59.70 | 57.24/57.33 | 56.22/57.09 | 57.28/57.95 | 51.46/54.34 | 54.27/54.93 |
| animal/Linear | 67.18/67.48 | 65.77/66.07 | 65.79/65.81 | 60.81/60.73 | 52.49/53.05 | 59.63/60.42 |
| animal/IRMF | 66.32/67.08 | 64.08/64.88 | 64.94/64.93 | 58.93/59.32 | 50.84/55.96 | 58.92/60.01 |
| animal/Conv | 65.89/66.40 | 64.05/64.45 | 65.03/65.12 | 59.25/59.47 | 51.03/54.31 | 58.22/59.24 |
| bird/Raw | 73.86/73.19 | 73.03/73.19 | 72.66/73.43 | 75.46/75.25 | 73.04/72.53 | 74.98/74.15 |
| bird/EMD | 75.81/75.45 | 75.16/75.07 | 71.98/72.25 | 74.93/74.50 | 71.98/74.43 | 73.73/73.39 |
| bird/Linear | 78.21/78.35 | 77.28/77.46 | 74.68/75.21 | 76.15/76.17 | 73.92/72.20 | 74.97/74.82 |
| bird/IRMF | 78.07/77.65 | 77.47/77.04 | 75.30/75.13 | 75.47/74.89 | 75.91/74.93 | 74.81/74.10 |
| bird/Conv | 77.88/77.62 | 77.43/77.08 | 75.27/75.40 | 75.34/74.74 | 74.93/73.19 | 74.82/73.62 |

For animal, tree ordering persists in means. GD60 IRMF−EMD passes Holm only with level; converged IRMF also passes without level for LightGBM/XGBoost, p=.040/.031. All tree IRMF−raw contrasts pass. Neither ROCKET nor CNN confirms an IRMF−EMD gain. No common model establishes IRMF over linear IMF. Bird has no corrected IRMF−EMD or IRMF−raw difference.

Nadeau-Bengio is primary, Wilcoxon secondary; Holm covers 100 main tests/task/metric, 20 for supplemental MultiRocket. Intervals are individual corrected intervals. [All three metrics](SCORES_ALL_METRICS.md), [paired effects](PAIRS.md), and [all-metric tests](paired_requested.csv) cover every model and contrast.

Best main family, LightGBM: animal clip BA without/with is raw 65.83/64.92, EMD 72.17/71.42, linear 76.92/77.17, IRMF 77.83/78.58, Conv 77.17/78.50. These pool 400 clips/seed then average three seeds; selection is exploratory. [Both tasks/all models](clip_scores.csv).

Animal ten/nine-slot IRMF scores 65.50/65.54% without level versus EMD 59.46%; linear controls remain competitive. Fusion adds no corrected gain. H=1/4, convergence and capacity [tables](ABLATIONS.md) include all ready tags. Raw log-STFT CNN scores 53.93/55.47% animal and 68.13/65.85% bird.

Animal IRMF without level: largest scale groups are finest residual, 10.9% LightGBM split count; w13, 10.0% XGBoost gain; w9, 7.1% ExtraTrees impurity. [Full importance](IMPORTANCE.md).

Timing pilots fixed identical per-method budgets. Trees/MiniRocket use 15 folds; CNN/MultiRocket five. Caps: LightGBM 150 rounds, XGBoost 60, ExtraTrees 64 trees; Mini/Multi 1,008/672 features and five head passes; CNN width 8, downsampling 2, at most 20 epochs. The 2-D reference uses width 16 and 20 epochs.

CPU decomposition s/frame: raw 0, EMD .0140, linear .0062, GD60 IRMF 1.759, Newton IRMF .0644. Decomposition costs exclude fitting. Elapsed through audit: 102.4 min, within the two-hour budget. [Costs](runner_costs.csv), [supplemental costs](supplemental_costs.csv), [reproduction](RUN_NOTES.md).

All 2,020 evaluation fits and 2,060,400 predictions passed audits. Local export checks preserve native saturated-probability ties; shared ML code is unchanged. One missing GD60 a=2 feature cache was exported. No commit.
