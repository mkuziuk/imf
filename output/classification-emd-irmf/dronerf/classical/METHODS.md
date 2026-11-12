# DroneRF classical baseline methods

The [protocol](../../../../experiments/classification-emd-irmf/PROTOCOL.md) is binding. The four tags are `raw`, `emd`, `imf_linear_a1.41`, and `irmf_h2_a1.41`. IRMF uses the cached package defaults, including max_iter=60. Shared source files and other datasets were left unchanged. Data, predictions, feature memmaps and fit checkpoints stay under `/Users/mikhail/Datasets/imf-classification/dronerf/`; only code, small result tables and figures are in the repository.

All three tasks use five segment-grouped folds repeated with seeds 0, 1 and 2. Every method and classifier sees the same outer test frames. Imputation, constant-feature removal and standardization fit inside each training fold. Searches use three grouped inner folds with seed 0 and balanced accuracy scoring. Fixed models need no inner selection. The runner's class-prior defaults are retained, including unweighted logistic regression and SVM. Scores are means of fold scores, rather than scores from concatenating repeated predictions.

| Classifier | 2-class candidates | 4-class candidates | 10-class candidates |
|---|---|---|---|
| LDA | Ledoit-Wolf automatic shrinkage | same | same |
| QDA | shrinkage auto, .05, .2, .5, .8 | auto, .2, .8 | auto, .2, .8 |
| Gaussian NB | defaults | same | same |
| Logistic regression | C=.001, .01, .1, 1, 10 | .01, .1, 1 | .01, .1, 1 |
| kNN | k=1, 5, 15, 35 | 1, 15 | 1, 15 |
| Linear SVM | C=.001, .01, .1, 1 | .01, .1 | .1 fixed |

The full 2-class task took 3279.2 s with three workers. SVM fitting alone accumulated 7500.7 worker-seconds. The smaller grids apply identically to every method and both feature sets within a task. The initial multiclass SVM range {.001,.01} was replaced by {.01,.1} after inspecting the full binary tuning. Sixty earlier SVM fits were discarded; 351 non-SVM checkpoints were retained after verifying that their data, splits and model definitions were unchanged. `grid_revision.json` records this 328 s partial attempt.

The 10-class SVM was then restricted to the single candidate C=.1 before primary-task results, after the secondary-task fits selected it almost throughout. Omitting that candidate's redundant inner fits produces exactly the same final estimator as a one-candidate GridSearchCV refit. It preserves all six classifier families and all 15 outer splits. `primary_svm_grid.json` and the run metadata record this choice. Other 10-class grids remain as shown above.

Every worker uses one BLAS/OpenMP thread. The maximum pool size is three; GridSearchCV itself uses n_jobs=1. Resumed jobs use one fit per joblib batch to balance long fits. Checkpoints fingerprint input features, metadata, splits, grids and modeling source. Scheduling-only changes and the cache migration are recorded in `batch_scheduling.json`. The wrapper invokes the shared runner and changes only explicit grids, checkpointing and scheduling.

A representative 4-class SVM fit converged in 51 iterations, took 14.31 s, and reproduced its saved predictions exactly. A dual-solver trial took 88.54 s and reached its iteration cap; a relaxed primal tolerance gave only a small speed gain. Both alternatives were rejected. The original solver, tolerance and iteration limit were retained. `svm_solver_probe.json` records this check; [benchmark_svm_solver.py](../../../../experiments/classification-emd-irmf/datasets/dronerf/benchmark_svm_solver.py) reproduces it after the main caches exist. The preceding resumed 4-class phase had run 1396 s, recorded in `svm_probe_pause.json`. Total reported runtime includes both partial phases and the numerical probe, rather than only the final resumed run's timer.

The per-method best-family table is descriptive. Maximizing outer means over families gives optimistic point estimates. For compact comparisons and the band check, one common classifier is selected per task: the best method/classifier combination averaged over both feature sets. That selection is exploratory. Full classifier-specific comparisons remain available; paired tests hold the classifier fixed rather than comparing separately selected maxima.

The primary paired test uses Nadeau-Bengio SE = sqrt[(1/J + mean(n_test/n_train)) s_d²], J=15, with Student t on 14 degrees of freedom. Its 95% intervals are marginal. Holm correction covers 72 tests per task and metric: six pairs, six classifiers, two feature sets. Wilcoxon is secondary because repeated-CV folds overlap. Recall tests use one task-wide Holm family over every pair, classifier, feature set and class, giving 720 tests for 10-class. Band tests use one eight-test family per task and metric. The band contrast is L-only minus both-bands. Balanced accuracy is primary; macro-F1 and accuracy are reported for both feature sets as well.

Per-feature Kruskal-Wallis epsilon²=H/(N-1) is tie-corrected. Mutual information is the runner's 16-quantile-bin Miller-Madow estimate, in nats. These are descriptive all-frame statistics, never feature-selection criteria. Correlated frames preclude interpreting their ordinary p-values as independent-observation tests. Scale summaries average the 20 features separately for L and H; those means are not additive information measures. `scale_order` puts IMF's `residual_fine` first and EMD's residue last. Window labels and EMD indices do not identify matched frequency bands. EMD has nine slots per band, IMF/IRMF 18 and raw one; boundary handling also differs.

The band sensitivity check focuses on the primary 10-class task. L/H are non-simultaneous views. L-only models drop all H columns and use the same model/grid as the corresponding main task. Energies normalize within each channel, so cached L columns are valid L-only inputs. The original gate accepted activity in either band; the test therefore retains H-only-active frames. It does not prepare a new L-gated sample. There are 135 outer control fits: four methods, both feature sets, all 15 splits, plus a two-feature log-scale-only control. Optional `--band-tasks 2class 4class 10class` expands the sensitivity check.

Convergence status means any IRMF stage failed in either band. Status-specific accuracy is descriptive: frames and class mixtures differ, and convergence is not an intervention. Common-class balanced accuracy and per-class counts expose composition differences. Cached decomposition CPU costs are per two-band frame; historical decomposition wall times came from six-worker preprocessing, which was not rerun here. Raw has zero additional decomposition cost, excluding normalization and feature extraction.

The task is to classify activity-gated frames. Excluding log_scale still leaves SNR, burst-strength and quantization cues in normalized features. Each BUI is one session, so segment grouping cannot establish unseen-session flight-mode recognition. [Shulman 2026, Table 3](https://arxiv.org/html/2607.01025v1) reports AR/Bebop recording-held-out macro-F1 .455 [.439,.470], versus .742 [.728,.755] randomly. Its task and grouping differ from ours, so scores cannot be ranked directly. Our frames are 51.2 µs versus nominal 2.5 ms literature blocks. The [original aggregation code](https://github.com/Al-Sad/DroneRF/blob/master/Matlab/Main_1_Data_aggregation.m#L39-L66) centers all 100,000 samples and then truncates the FFT to 2048; the 48.8-fold duration ratio is not a claim about every original FFT feature's effective support.

To reproduce from the worktree root:

```bash
study_root=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594
study_python="$study_root/.venv/bin/python"
study_out="$study_root/output/classification-emd-irmf/dronerf/classical"
export IMF_CLS_MAX_WORKERS=3 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TMPDIR=/Users/mikhail/Datasets/imf-classification/dronerf/work/classical/tmp
mkdir -p "$TMPDIR"
cd "$study_root"
"$study_python" "$study_root/experiments/classification-emd-irmf/baselines/run_classical.py" \
    --dataset dronerf --task 2class --workers 3 --out "$study_out"
for task in 4class 10class; do
    "$study_python" "$study_root/experiments/classification-emd-irmf/datasets/dronerf/run_classical_logged.py" \
        --dataset dronerf --task "$task" --workers 3 --out "$study_out" --reduced-grid
done
"$study_python" "$study_root/experiments/classification-emd-irmf/datasets/dronerf/analyze_classical.py" \
    --static --bands --band-tasks 10class --report
```

`--bands` refits the controls; `--report` alone rebuilds numerical tables and figures. `validation.json` checks group separation and prediction coverage for all models, recomputes the selected-family and all band/control metrics, verifies identical shared log-scale values across methods, and records software versions and input/source hashes. `tables.md` contains detailed numerical results; `REPORT.md` states the findings.
