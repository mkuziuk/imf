# UAS/birds at 60 GHz: phase-3

IRMF−EMD LightGBM gains without log_scale: **+2.75 pp binary, +2.12 pp target3; Holm p=1.00 for both**.

Balanced accuracy %, without/with log_scale. IRMF is GD60; Conv is converged Newton H=2. Linear/IRMF/Conv use a=sqrt(2).

| Task/input        | LGBM        | XGB         | ET          | Mini        | CNN         | Multi       |
|:------------------|:------------|:------------|:------------|:------------|:------------|:------------|
| bird_drone/Raw    | 95.92/97.11 | 95.76/97.34 | 93.68/95.51 | 96.20/96.64 | 95.08/95.52 | 93.79/95.66 |
| bird_drone/EMD    | 96.22/97.24 | 96.79/97.40 | 95.43/95.85 | 96.80/96.89 | 94.97/93.79 | 96.21/97.27 |
| bird_drone/Linear | 97.89/98.02 | 97.99/97.90 | 97.10/97.30 | 97.45/97.80 | 95.23/95.93 | 96.24/97.09 |
| bird_drone/IRMF   | 98.98/98.99 | 98.86/98.80 | 98.25/98.19 | 97.33/97.59 | 95.00/94.52 | 96.04/96.73 |
| bird_drone/Conv   | 99.11/99.07 | 98.70/98.58 | 98.19/98.31 | 97.36/97.63 | 94.63/95.03 | 95.45/96.71 |
| target3/Raw       | 91.49/93.40 | 91.39/93.58 | 89.58/91.60 | 94.44/94.62 | 90.56/91.25 | 89.17/90.31 |
| target3/EMD       | 94.34/94.79 | 94.24/94.34 | 92.92/92.78 | 93.19/93.54 | 85.59/85.87 | 88.16/88.85 |
| target3/Linear    | 94.51/94.34 | 93.78/93.65 | 95.45/95.17 | 95.97/96.49 | 90.90/91.84 | 91.91/92.64 |
| target3/IRMF      | 96.46/96.53 | 96.35/96.22 | 96.46/96.25 | 96.28/96.46 | 91.01/91.39 | 91.01/91.04 |
| target3/Conv      | 96.60/96.49 | 96.15/96.01 | 95.94/96.04 | 96.22/96.49 | 90.59/90.94 | 90.56/91.77 |

No balanced-accuracy contrast passes Nadeau-Bengio plus Holm, including IRMF versus EMD, linear IMF or raw. Capacity, H and convergence checks also give no corrected gain. Non-significance is not equivalence. Secondary Wilcoxon flags three contrasts. Holm covers 120 primary or 84 sensitivity tests per task/metric/aggregation.

Poses with any LightGBM cell error, without scale, Raw/EMD/Linear/IRMF/Conv: **14/18/9/7/6** binary; **28/20/18/12/13** target3, out of 45. IRMF cell errors pooled over three seeds are bird/UAS **13/10**, and bird/Mavic/P3P **18/26/29**. LightGBM probability-mean recording BA is **100% for all five inputs**, both tasks/scales. Voting adds one EMD target3 error with scale. [Class/recording errors](ERRORS.md).

Using observed paired-fold variance, LightGBM IRMF−EMD needs **3.31 pp nominally or 6.18 pp at the conservative Holm first step for 80% power**, without scale, on either task. The significance thresholds are 2.36/5.05 pp. [All power estimates](POWER.md). 49 recordings, 45 poses, one campaign.

Nine inputs, 15 folds, full tree budgets; Mini/Multi use 1,008/672 features; CNN width 8, at most 20 epochs. CPU decomposition seconds/frame: raw 0, EMD .0305, linear .0044, GD60 .3547, Conv .0314. [Costs](decomposition_costs.csv).

Written: [BA/macro-F1/accuracy](SCORES_ALL_METRICS.md), [tests](PAIRS.md), [ablations](ABLATIONS.md), [interactive scores](overview.html), error/power CSVs, [reproduction](RUN_NOTES.md), and four dataset-local scripts. All **3,240 fits / 508,032 predictions passed audits**. Predictions remain external. No commit.
