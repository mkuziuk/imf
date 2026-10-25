Best balanced accuracy (%), **without / with `log_scale`**. Maxima over six classifier families are descriptive.

| Method | 2-class | 4-class | 10-class (primary) |
|---|---:|---:|---:|
| raw | 86.63 / 90.25 | 76.88 / 82.71 | 41.94 / 47.69 |
| EMD | 82.53 / 86.51 | 73.70 / 79.87 | 38.78 / 42.92 |
| linear IMF | 94.14 / 95.37 | 85.02 / 87.42 | 57.47 / 59.49 |
| IRMF | 92.89 / 94.19 | 80.91 / 86.21 | 49.53 / 53.61 |

All 15 grouped splits and six families were retained; [METHODS.md](METHODS.md) documents reduced grids. [Tables](tables.md) include macro-F1, accuracy, corrected-t and secondary Wilcoxon p-values for every task and both feature sets.

Primary linear-SVM contrasts: percentage-point difference [95% Nadeau–Bengio CI]. Holm: 72 tests/task/metric.

| Contrast | Without scale | With scale | Holm p, without / with |
|---|---:|---:|---:|
| IRMF − EMD | +10.80 [+9.91,+11.69] | +10.67 [+9.80,+11.53] | 2e-11 / 1.6e-11 |
| IRMF − linear IMF | -7.94 [-9.17,-6.72] | -5.90 [-7.01,-4.80] | 5.8e-08 / 6.3e-07 |
| linear IMF − EMD | +18.74 [+17.05,+20.43] | +16.57 [+15.30,+17.84] | 6.4e-11 / 7.5e-12 |
| IRMF − raw | +14.58 [+12.98,+16.18] | +14.54 [+13.16,+15.91] | 8e-10 / 1.2e-10 |
| linear IMF − raw | +22.52 [+20.33,+24.71] | +20.44 [+18.75,+22.13] | 1.8e-10 / 2.1e-11 |
| EMD − raw | +3.78 [+2.25,+5.32] | +3.87 [+2.47,+5.27] | 0.0022 / 0.00081 |

**The IRMF–EMD gain survives without `log_scale`; linear IMF remains better.** IRMF beats EMD with five families; QDA reverses by −2.12/−2.15 points (Holm p<.001). Secondary SVM IRMF–EMD gains are +10.36/+7.84 points (2-class) and +7.21/+6.35 (4-class).

SVM mean recalls improve versus EMD in all ten classes, largest for Bebop connected: +20.81/+29.28 points. IRMF trails linear IMF there by 22.77/15.49; smaller EMD gains often fail recall-wide Holm correction.

L-band information peaks at w47 by mean MI (linear .293, IRMF .251 nats) and w93 by mean Kruskal–Wallis ε² (.256/.204); EMD peaks at IMF1 (.220 MI, .131 ε²). Scales follow `scale_order`. H peaks at w9 for IMF methods.

L-only IRMF scores 43.57/47.58 versus both-view SVM 49.53/53.58; the ranking persists. Dropping H significantly lowers every method. H is nonsimultaneous, mostly noise for AR/Bebop. Log-scale-only SVM reaches 19.57%.

At max_iter=60, 6.70% of stages and 16.64% of frames are unconverged. [Stage/class rates](convergence_stages.csv) accompany the [class summary](convergence_classes.csv). IRMF BA is 50.53/55.31 on converged versus 41.30/41.96 on unconverged frames; this association does not establish cap causality.

Power/SNR and quantization cues remain after normalization. Each BUI is one session: segment CV cannot establish unseen-session recognition. [Published recording-held-out AR/Bebop macro-F1](https://arxiv.org/html/2607.01025v1) is .455 [.439,.470], versus .742 randomly, with different task/grouping. Our 51.2-µs frames also differ from the original nominal 2.5-ms blocks; direct ranking is invalid.

Cached decomposition CPU ms/two-band frame: raw 0, EMD 29.4, linear 12.6, IRMF 1278.7. [Computation](runtime.json): 2.86 h, ≤3 workers, including interrupted phases/probe. [Validation](validation.json) passed despite worker-replacement warnings; no shared-code bug found. Wrote report, methods, CSVs, three DroneRF scripts and three figures: [accuracy](figures/balanced_accuracy.png), [scales](figures/discriminability_scales.png), [convergence](figures/convergence.png). Caches/predictions remain under DATA_ROOT; no commits.
