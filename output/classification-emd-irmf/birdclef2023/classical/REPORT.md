| Method | Species −log | Species +log | Passerine −log | Passerine +log |
|---|---:|---:|---:|---:|
| Raw | 42.3/42.3/42.3 K | 40.4/40.4/40.4 K | 76.1/76.6/78.5 K | 75.2/75.6/77.7 K |
| EMD | 43.5/43.2/43.4 R | 43.3/43.1/43.3 R | 75.2/75.5/77.2 R | 75.2/75.4/77.1 R |
| Linear IMF | 49.3/48.5/49.3 S | 49.3/48.5/49.3 S | 78.0/78.2/79.8 R | 77.8/78.1/79.6 R |
| IRMF | 49.6/48.9/49.6 S | 49.7/49.0/49.7 S | 77.6/77.8/79.4 R | 77.6/77.8/79.4 R |

Best-BA classifier’s BA/macro-F1/accuracy (%). K=kNN, R=LogReg, S=linear SVM; best-of-six is descriptive. log=`log_scale`.

LogReg Δpp[95% CI]; Nadeau-Bengio p/Holm q, 72 comparisons/task. Secondary Wilcoxon and [all metrics/classifiers](paired_all_metrics.csv).

Species:

| A−B | −log: Δ[CI];p/q | +log: Δ[CI];p/q |
|---|---:|---:|
| IRMF−EMD | +6.05[4.72,7.39];1.3e-07/6.6e-06 | +6.05[4.49,7.60];8.3e-07/3.8e-05 |
| IRMF−Linear IMF | +0.65[-0.60,1.89];0.28/1 | +0.82[-0.41,2.05];0.18/1 |
| Linear IMF−EMD | +5.41[4.24,6.57];1e-07/5.4e-06 | +5.23[4.02,6.44];2.3e-07/1.1e-05 |
| IRMF−Raw | +11.89[9.84,13.94];5.8e-09/3.8e-07 | +11.82[9.98,13.66];1.6e-09/1e-07 |
| Linear IMF−Raw | +11.24[9.19,13.29];1.2e-08/7.5e-07 | +11.00[9.31,12.69];1.3e-09/8.7e-08 |
| EMD−Raw | +5.84[4.33,7.34];8.6e-07/3.9e-05 | +5.77[4.27,7.27];9.5e-07/4.2e-05 |

Passerine:

| A−B | −log: Δ[CI];p/q | +log: Δ[CI];p/q |
|---|---:|---:|
| IRMF−EMD | +2.31[0.88,3.74];0.0037/0.12 | +2.39[1.32,3.45];0.00028/0.014 |
| IRMF−Linear IMF | -0.42[-1.86,1.02];0.54/1 | -0.27[-1.60,1.06];0.67/1 |
| Linear IMF−EMD | +2.73[1.36,4.09];0.00075/0.033 | +2.66[1.34,3.98];0.00068/0.031 |
| IRMF−Raw | +6.95[4.55,9.35];2.3e-05/0.0014 | +7.02[4.68,9.37];1.6e-05/0.001 |
| Linear IMF−Raw | +7.37[4.72,10.02];3.5e-05/0.0021 | +7.29[4.50,10.09];6.5e-05/0.0037 |
| EMD−Raw | +4.64[2.10,7.18];0.0016/0.061 | +4.63[2.19,7.08];0.0012/0.048 |

Species IRMF−EMD survives removing loudness for all six classifiers. IRMF−linear is significant only for kNN (+2.53/+2.62 points). Raw kNN scores 42.3%, versus IRMF kNN 41.7%, without log. Passerine LogReg IRMF−EMD survives Holm only with log.

Recording BA (%), LogReg probability averaging over selected frames; recordist folds retained. Species IRMF−EMD gap shrinks 6.1→3.1 points:

| Method | Species −log/+log | Passerine −log/+log |
|---|---:|---:|
| Raw | 46.1/44.9 | 70.6/70.3 |
| EMD | 57.2/57.0 | 78.0/77.8 |
| Linear IMF | 58.9/59.2 | 81.8/81.6 |
| IRMF | 60.3/60.3 | 82.1/82.1 |

LogReg IRMF−EMD recall spans −4.5 barswa to +13.9 eubeat1 points without log; −5.5/+15.9 with. All fail Holm. [All species/pairs](per_species_recall_comparisons.csv).

Mean feature discriminability peaks at w13–w33, median centroids 1.8–2.65 kHz at 22,050 Hz. IRMF: w13 ε²=.343; w33 MI=.375 nats. Linear: .356/.387 at w13/w25. EMD: imf2 ε²=.273 at 1.43 kHz; imf1 MI=.288 at 3.34 kHz. [Every scale](scale_discriminability_species.csv) follows `scale_order`, finest residual first; centroids are not passbands. Statistics are descriptive.

IRMF: 7,888/12,000 frames converge completely; 14.05% of stages do not. LogReg accuracy, converged/unconverged: species 51.1/46.3% without log, 51.1/46.0% with; passerine 79.0/80.2%, 79.0/80.1%. Label mixtures differ; some unconverged subsets lack one species. Observational differences. [Details](convergence_subset_metrics.csv).

Confounds: 8.96 dB species median RMS span, weak 93 ms labels, 451 recordists including 175 with multiple species, and 20/180/360 component features for raw/EMD/IMF.

CPU seconds/frame: raw 0, EMD .01165, linear .00632, IRMF 1.32049. Runtime: species 26.4 min, passerine 8.3 min; 46.7 min including pilot/analysis. All 15 folds retained; only SVM C narrowed identically to {.001,.01} after partial first-task timing. Other grids unchanged.

Files: this report, [reproduction](REPRODUCE.md), CSV/JSON tables, interactive HTML, two dataset-local scripts, and three figures: [classifiers](figures/classifier_balanced_accuracy.png), [recall](figures/per_species_recall.png), [scales/convergence](figures/scales_and_convergence.png). Frame caches: DATA_ROOT. Fixed wrapper object-label serialization; no shared numerical bug observed.
