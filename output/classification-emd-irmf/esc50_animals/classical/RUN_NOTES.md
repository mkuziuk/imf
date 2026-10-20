# Reproduction and interpretation

Run from `/Users/mikhail/.t3/worktrees/imf/t3-44ed3594`. The isolated launcher
calls `baselines/run_classical.py`'s `run` function with `quick=False`, all six
classifiers, both feature sets, all 15 stored splits, and three workers. Each
worker has one BLAS/OpenMP thread. It uses the driver's original estimators and
hyperparameter grids and records probabilities immediately after the same fit.
The shared driver and `emdirmf` files are not edited.

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/esc50_animals/run_phase2.py \
  --batch main
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/esc50_animals/analyse_classical.py
```

At the end of the main analysis, run only the tags listed in `ready_ablations`
in `analysis_context.json`, using `run_phase2.py --batch ablations --methods TAG...`.
Skip unfinished tags and do not wait for later arrivals. Then run
`analyse_classical.py --ablations-only`. A decomposition array or `.parts` folder
alone does not establish completion. The check requires a nonpartial
decomposition JSON for 5,100 frames and the feature parquet and JSON.

Large caches and OOF probabilities live under
`/Users/mikhail/Datasets/imf-classification/esc50_animals/results/phase2_main/`.
There are separate prediction paths for main and ablation runs. Each completed
fit has a resumable NPZ with held-out frame IDs, labels, probabilities when
available, class order, fit timing, best parameters, and warnings. Feature file
modification time, estimator parameters, and test indices enter the cache key.

`validation.json` records a direct upstream LDA fit that must reproduce the
wrapper's labels exactly. It also checks identical shared `log_scale` values,
aligned frames, 720 fits per task, finite normalized native probabilities, and
agreement between probability argmax and the saved class label. Loading the
saved splits checks uploader separation and a complete test partition per seed.

Headline best-classifier scores use the best mean outer-CV balanced accuracy
within each method and feature set. This selection makes those scores
exploratory. All inferential comparisons keep classifier and split fixed. Holm
correction covers 72 tests per task and metric, six method pairs times six
classifiers times two feature sets. Corrected intervals themselves are ordinary
95% intervals, not simultaneous Holm-adjusted intervals. Wilcoxon is secondary
because repeated grouped-CV folds overlap. Accuracy comparisons are also saved,
although the upstream driver only makes comparison tables for balanced accuracy
and macro-F1.

Feature statistics describe all 5,100 frames and do not establish group-level
significance. Epsilon squared is the driver's `H / (N - 1)` convention. Mutual
information uses 16 quantile bins and the Miller-Madow correction, in nats.
Component aggregates average the same 20 features, excluding `log_scale`.
`scale_order` and `slot_labels` preserve the fine residual of IMF/IRMF as the
finest component. Hz are observed median spectral centroids among nonzero
components, with the 10th and 90th percentiles. `fs / window_samples` is also
provided as a nominal inverse duration; it is not a filter cutoff. EMD slot
indices do not identify fixed frequency bands.

Clip probabilities are averaged within one outer fold and seed, never across
different trained models. Every clip stays within its uploader's held-out fold.
Logistic regression is fixed for the clip comparison across all methods, using
its native cached held-out probabilities. This choice was made before examining
clip scores. Linear SVM has no native probabilities; none are fabricated, and no
extra calibration study is added. The headline clip score pools all 400 held-out
clips within a seed and averages the three seed scores. The mean over 15 outer
folds is also saved separately. Every clip has one prediction per seed. The
matched logistic frame scores are saved alongside clip scores in
`logreg_frame_clip_metrics.csv`.

Convergence subsets describe predictions from models trained on all eligible
training frames. They do not refit on just converged or unconverged frames.
Their accuracy difference is observational, with different animals and signal
types in the two subsets. Class-specific accuracy and balanced accuracy expose
that compositional difference. The supplemental uploader-balanced metric first
averages frame correctness within uploader and true class, then averages
uploaders within class, then classes within each held-out fold.

## Driver limitations and minimal reproductions

The driver saves `y_pred` without probabilities. Its unchanged table writer is
used by this run too. Inspect its actual output to expose the missing fields;
the wrapper's probability files are separate.

```python
import pandas as pd

path = "/Users/mikhail/Datasets/imf-classification/esc50_animals/results/phase2_main/classical_animal_predictions.parquet"
columns = pd.read_parquet(path).columns
assert "y_pred" in columns
assert not any(column.startswith("prob_") for column in columns)
```

Also, `--out` changes result tables but not the prediction filename. These two
commands both target `results/classical_animal_predictions.parquet`; the second
replaces the first. These are reproduction instructions, not additional runs
used in the study.

```bash
PY=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python
DRIVER=/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/baselines/run_classical.py
$PY "$DRIVER" --dataset esc50_animals --task animal --methods raw --classifiers lda --quick --workers 1 --out /tmp/esc50_repro_raw
$PY "$DRIVER" --dataset esc50_animals --task animal --methods emd --classifiers lda --quick --workers 1 --out /tmp/esc50_repro_emd
```

`run_phase2.py` works around both limitations locally by recording held-out
probabilities and isolating each batch's prediction directory.

The audit also retains numerical warnings suppressed by the upstream driver.
Automatic QDA shrinkage can retain very small within-class variances and fail
the eigenvalue tolerance check. This small example reproduces it on the
installed scikit-learn 1.9. A fixed isotropic shrinkage value resolves it, as in
the driver's existing tuned alternatives. No outer-fold samples enter that choice.

```python
import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

rng = np.random.default_rng(0)
y = np.repeat(np.arange(3), 20)
X = rng.normal(size=(60, 4))
X[:, 0] = y + 1e-3 * rng.normal(size=60)
QuadraticDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(X, y)
# LinAlgError: the covariance matrix of class 0 is not full rank.
QuadraticDiscriminantAnalysis(solver="eigen", shrinkage=0.05).fit(X, y)
# Succeeds. Run this line after handling the expected exception above.
```

For SVM grids with convergence warnings, `svm_selected_fit_audit.csv` reports a
refit at the selected C, the iteration count, warnings, and agreement with the
original held-out labels. It separates a rejected grid candidate's warning from
a warning in the chosen model. These diagnostic refits do not replace primary
scores or tune against held-out labels. Full warning messages are in
`fit_warnings.csv`.

The finished run has 720 unique fits per task. There are 222 QDA grid warning
records and 42 SVM grid warning records. All 42 selected-SVM diagnostic refits
reproduce the original labels exactly. Two bird fits at C=1, split 13, with and
without `log_scale`, hit 5,000 iterations. The sensitivity fit permits 20,000
iterations and converges at 5,737 and 5,878 respectively, with unchanged balanced
accuracy. All audited animal SVM fits converge at the default iteration limit.
One worker-restart warning appears in the joblib log; no result is missing or
duplicated. Primary CV takes 2,520.9 seconds. The diagnostic refits span 97.8
seconds on three workers; the successful remaining analysis takes 24.3 seconds.

At the main-analysis readiness checkpoint, 2026-10-20 19:29:14 UTC, none of the
four ablation tags has a feature parquet. Both a=2 decomposition JSONs exist;
the H=1 and H=4 decomposition JSONs do not. All ablations are therefore skipped.
The snapshot in `analysis_context.json` is preserved rather than extended to
include files produced later by the background queue.
