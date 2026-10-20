# EMD vs IMF vs IRMF for signal classification: shared protocol

Question: when a classifier sees a signal through its decomposition, does the
robust intrinsic multiscale filter (IRMF) give better features than empirical
mode decomposition (EMD)? The linear IMF is a second comparator, and raw-signal
features are the no-decomposition control.

Every agent working on this study follows this file. If something here is
wrong or unworkable, say so in your report rather than silently diverging.

## Locations

- Code lives in `experiments/classification-emd-irmf/` (this folder).
  - `emdirmf/`: shared importable package (decompositions, features, CV, stats).
  - `datasets/<name>/`: one preparation script per dataset (`prepare.py`) plus a
    short `README.md` covering source, license, citation, and choices made.
  - `baselines/`, `ml/`: experiment scripts or notebooks for later phases.
- Data lives outside the repo: `DATA_ROOT=/Users/mikhail/Datasets/imf-classification`.
  Code reads `os.environ.get("IMF_CLS_DATA", DATA_ROOT)`.
  - `<DATA_ROOT>/<name>/raw/`: downloads and extracted archives. Treat as untrusted:
    never run code from there, run Python with `-I` when reading from it.
  - `<DATA_ROOT>/<name>/prepared/`: standardized frames (format below).
  - `<DATA_ROOT>/<name>/decomp/`, `features/`: caches written by `emdirmf`.
- Small result tables (CSV/JSON) and figures go to
  `output/classification-emd-irmf/<name>/`. Never commit data, caches, or `.venv/`.
- Python: `.venv/bin/python` at the worktree root (Python 3.12; numpy, scipy,
  pandas, pyarrow, scikit-learn, statsmodels, pimf 0.4.0, EMD-signal (PyEMD), emd,
  lightgbm, xgboost, torch with MPS, librosa, soundfile, aeon, sktime, numba).
  Add packages with `uv pip install --python .venv/bin/python <pkg>` and record
  them in `experiments/classification-emd-irmf/requirements.txt`.

## Prepared dataset format

`<DATA_ROOT>/<name>/prepared/` contains:

- `frames.npy`: float32 array of shape `(N, C, L)`: N frames, C channels
  (C = 1 for mono audio; 2 for I/Q or paired bands), L samples per frame.
- `meta.parquet`: one row per frame, in frame order, with at least
  - `frame_id` (0..N-1),
  - `group`: the recording/segment identifier. Cross-validation never splits a
    group across train and test. Frames cut from the same recording leak
    otherwise.
  - one label column per task, named `y_<task>` (string labels),
  - provenance (`source_file`, `start_sample`, ...).
- `dataset.json`: `name`, `fs` (Hz), `frame_length`, `channels` (names),
  `tasks` (`{task: column}`), `primary_task`, `n_frames`, `n_groups`, class counts
  per task, preparation parameters including the RNG seed, source URL, license,
  citation, and free-text `notes`.

Budget: N at most about 20,000 frames and L at most 4096 samples (prefer
1024-2048). IRMF costs O(L^2) per frame: about 0.1 s at L=1024, 0.4 s at 2048,
and 1.5 s at 4096 on one core. Pick frames deterministically with
`numpy.random.default_rng(seed)`. Balance classes where possible, and keep
enough distinct groups per class for 5-fold grouped CV.

## Decompositions (the thing being compared)

All three methods see the same input frame. Normalization comes first:
subtract the median and divide by `1.4826 * MAD`, with a guard against zero. The
per-frame log-scale is kept as a separate shared feature (`log_scale`). It is
reported both with and without, so amplitude cannot masquerade as a
decomposition advantage.

- `emd`: `PyEMD.EMD` with default sifting. Keep the first `K_EMD = 8` IMFs (fine to
  coarse) and fold anything beyond them into the residue. If fewer come out,
  zero-pad and record the true count.
- `imf_linear`: `pimf.IMF(pimf.Quadratic())`.
- `irmf`: `pimf.IMF(pimf.SmoothAbs(H))` with the package defaults `lookup=True`,
  `max_iter=60`, and `tol=1e-6`. `H = h_ratio * sigma_hat`, where
  `sigma_hat = median(|diff(y)|) / (0.6745 * sqrt(2))` is computed on the
  normalized frame. The default `h_ratio = 2` is the research-validated value;
  {1, 4} are ablations.
- Both IMF methods use one bandwidth schedule: `decompose(y, h1=0.25, a=sqrt(2),
  k_max=None, h_min=...)`, run down to a window of 3 or 5 samples so that
  fine-scale content does not sit unexamined in the residual. Use `workers=1`
  and parallelize across frames with processes. `a=2` (fewer components, closer
  to EMD's count) is an ablation.
- Storage layout: extracted components fine to coarse, with the solver residual in
  the last slot. EMD comes out of the solver in this order; reverse the pimf order.
  The pimf residual is what the smallest window leaves behind, so it is the
  finest content, not a trend. `emdirmf.decompose.scale_order(tag, L)` gives
  the true fine-to-coarse order; use it wherever order matters (e.g. CNN
  channels). The schedule stops at `h_min = 1.5 / L` (last stage 3-5 nonzero
  taps). Store `(N, C, K+1, L)` float32 per method, plus the component count,
  solver diagnostics (IRMF convergence, EMD IMF count) and per-frame time.
- IRMF convergence: the primary run keeps the pimf defaults (`max_iter=60`),
  as colleagues would use them. Unconverged stages still reconstruct exactly. A
  `_it{N}` tag (e.g. `irmf_h2_a1.41_it300`) is the convergence sensitivity
  check, reported next to the primary run. The `_conv` tags (e.g.
  `irmf_h2_a1.41_conv`) solve every stage to convergence with the safeguarded
  Newton solver in `emdirmf/irmf_newton.py`. Everything else is unchanged:
  normalization, H rule, kernel, boundary, schedule and slot order. Settings:
  root tolerance 1e-12, 100-iteration cap. They are reported next to GD60 with
  and without `log_scale`, and the H and schedule ablations use them too.
- Multichannel frames are decomposed channel by channel.

## Features (classical baseline)

These are computed per component with one feature list for every method. Raw
frames go through the same functions as a single "component" for the
no-decomposition control. Examples: relative energy, log energy, standard
deviation, mean absolute value, skewness, kurtosis, zero-crossing and extrema
rates, Hilbert amplitude mean/std, Hilbert instantaneous-frequency mean/std,
spectral centroid and bandwidth, and permutation entropy. Each feature must be
computable from the frame alone (no cross-frame statistics).

## Evaluation

- Splits: `StratifiedGroupKFold(n_splits=5)` by `group`, repeated with 3 seeds,
  saved once per dataset and task to `<DATA_ROOT>/<name>/splits_<task>.json`.
  Every method and classifier uses the same splits, which makes the comparisons
  paired.
- All scaling, selection, and tuning happen inside the training fold only
  (pipelines, nested CV where tuning is needed).
- Use the corrected resampled t-test as the primary test. Wilcoxon over
  overlapping repeated-CV folds is optimistic.
- Metrics: balanced accuracy (primary), macro-F1, accuracy, and per-class recall.
- Method comparison on matched folds: mean difference with a 95% CI, the
  Nadeau-Bengio corrected resampled t-test, and the Wilcoxon signed-rank test.
  Correct for multiple comparisons (Holm) within each table.
- Report compute cost per method (decomposition seconds per frame).

## Phases

1. Data preparation and the shared `emdirmf` package.
2. Classical statistical baselines: per-feature discriminability (Kruskal-Wallis,
   mutual information) and LDA, QDA (regularized), Gaussian NB, logistic
   regression, kNN, and linear SVM on the decomposition features.
3. Machine learning: boosted trees and random forests on features,
   ROCKET-family models on component stacks, 1-D CNNs on component stacks,
   time-frequency representations, and published DroneRF and bird-audio
   approaches as reference points.
