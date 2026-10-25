# DroneRF: phase 3

10-class LightGBM IRMF−EMD: **+7.84 pp** (corrected 95% CI [6.73,8.96], Holm p=4.31e−8) without log scale; **+5.18** [3.54,6.82], p=.000537 with it.

Frame balanced accuracy %, without/with log_scale. IRMF is GD60; Conv is converged Newton IRMF. All methods use 12,000 frames and segment-held-out splits. L/H are nonsimultaneous views.

| Input | LGBM | XGB | ExtraTrees | MiniRocket | CNN |
|---|---:|---:|---:|---:|---:|
| Raw | 57.04/63.68 | 52.09/57.75 | 57.20/62.90 | 46.27/48.21 | 28.92/31.36 |
| EMD | 56.49/63.06 | 52.55/58.19 | 48.22/53.56 | 41.98/44.31 | 28.54/31.07 |
| Linear IMF | 69.51/70.70 | 64.87/66.29 | 67.04/67.58 | 49.54/50.89 | 30.63/33.96 |
| IRMF | 64.34/68.23 | 59.00/62.65 | 61.73/62.89 | 47.07/49.70 | 30.48/33.47 |
| Conv | 64.68/68.35 | 59.41/63.03 | 61.87/63.51 | 47.34/49.84 | 29.38/33.16 |
| Conv a=2 | 63.78/67.65 | 58.40/62.02 | 60.80/62.93 | 46.93/49.86 | 29.85/32.22 |
| Linear a=2 | 68.94/70.56 | 63.91/65.54 | 66.29/66.74 | 49.22/50.94 | 30.21/33.40 |

Linear LightGBM macro-F1/accuracy: 69.25/69.58% without scale, 70.47/70.74% with it.

All three tree families establish IRMF over EMD with both scale variants. MiniRocket also passes primary Nadeau–Bengio tests (+5.08/+5.40 pp, Holm p=.00160/.000250); secondary Wilcoxon does not (p=.0996). CNN does not confirm a gain. **Convergence does not close the linear-IMF gap.** LightGBM Conv−GD60 is +0.35/+0.12 pp (both p=1); Conv−linear is −4.83 [−6.18,−3.47], p=.000155, and −2.35 [−3.16,−1.53], p=.00128. Newton leaves the GD60-unconverged subset difficult ([ablations](ABLATIONS.md)); this association does not establish causation.

The a=2 controls retain the ordering: LightGBM linear−Conv is +5.16/+2.91 pp, both corrected p<.002. Their ten slots per view approach EMD’s nine; default IMF uses eighteen. H=4 improves converged IRMF slightly but remains below linear IMF.

Secondary LightGBM IRMF−EMD shrinks to +3.81/+0.97 pp (4-class), +1.90/+0.80 (binary); only 4-class without scale survives Holm.

Pooling native OOF probabilities over each segment gives LightGBM BA with log scale: raw 90.61%, EMD 90.85%, linear 95.32%, IRMF 95.34%, Conv 95.00%. These pool 227 segments/seed then average three seeds. L-only linear scores 64.56% versus 70.70% with both views. Raw log-STFT 2-D CNN scores 43.40/44.57%. Published near-99% results use different tasks/context; segment grouping here still shares BUI sessions. [Scope](LITERATURE_CONTEXT.md).

Timing pilots fixed equal per-method budgets: trees/primary CNN use 15 folds; primary MiniRocket ten; secondary stack models five. Caps are 60 boosting rounds, 64 ExtraTrees trees, 1,008 MiniRocket features/five head passes, and width-8 CNN/eight epochs/downsampling two. A reader cache with bit-identical pilot probabilities enabled additional CNN seeds. These budgets do not establish accuracy ceilings.

Nadeau–Bengio is primary, Wilcoxon secondary; Holm covers 100 main tests/task/metric and 30 capacity tests. CIs are individual corrected intervals. [All tasks/three metrics](SCORES_ALL_METRICS.md), [effects](PAIRS.md), [segment results](segment_scores.csv), [costs](runner_costs.csv), and [reproduction/files](RUN_NOTES.md) accompany the local runners and audit. Cached decomposition CPU seconds/two-view frame: EMD .0294, linear .0126, GD60 1.279, Newton .108. All 2,740 evaluation fits and 6,576,000 predictions passed [audits](validation.json); 176.3 minutes elapsed from the first pilot. No commit.
