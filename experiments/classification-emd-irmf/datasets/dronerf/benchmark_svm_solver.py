"""One-fold numerical/runtime check of equivalent linear-SVM solvers."""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
root = Path('/Users/mikhail/.t3/worktrees/imf/t3-44ed3594')
study = root / 'experiments/classification-emd-irmf'
sys.path.insert(0, str(study))
sys.path.insert(0, str(study / 'baselines'))
import numpy as np
import pandas as pd
from sklearn.svm import LinearSVC
import run_classical as baseline
from emdirmf.cv import load_splits, split_metrics
from emdirmf.features import feature_columns

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--variant', choices=['primal_default', 'dual', 'primal_relaxed'], required=True)
args = parser.parse_args()
data = Path('/Users/mikhail/Datasets/imf-classification/dronerf')
meta = pd.read_parquet(data / 'prepared/meta.parquet')
tag, feature_set, split_id = 'imf_linear_a1.41', 'comp', 0
cache = data / 'work/classical/checkpoints_reduced_4class'
stem = cache / f'{tag}__{feature_set}__linsvm__{split_id}'
reference = json.loads(Path(str(stem) + '.json').read_text())
reference_pred = np.load(Path(str(stem) + '.npy'), allow_pickle=False)
c_value = json.loads(reference['best_params'])['clf__C']
table = pd.read_parquet(data / f'features/{tag}.parquet')
x = table[feature_columns(table.columns, include_log_scale=False)].to_numpy(dtype=float)
y = meta.y_4class.astype(str).to_numpy()
split = load_splits('dronerf', '4class')[split_id]
model = baseline._pipeline(LinearSVC(C=c_value, dual=args.variant == 'dual', max_iter=5000,
                                    tol=1e-3 if args.variant == 'primal_relaxed' else 1e-4,
                                    random_state=0))
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    started = time.perf_counter()
    model.fit(x[split.train], y[split.train])
    fit_seconds = time.perf_counter() - started
pred = model.predict(x[split.test])
clf = model.named_steps['clf']
z_train = model[:-1].transform(x[split.train])
scores = clf.decision_function(z_train)
targets = np.where(y[split.train, None] == clf.classes_[None, :], 1.0, -1.0)
hinge = np.maximum(1 - targets * scores, 0)
objective = 0.5 * (np.sum(clf.coef_**2) + np.sum(clf.intercept_**2)) + c_value * np.sum(hinge**2)
record = {
    'variant': args.variant, 'method': tag, 'feature_set': feature_set, 'split_id': split_id,
    'C': c_value, 'fit_seconds': fit_seconds, 'n_iter': int(clf.n_iter_),
    'prediction_agreement': float(np.mean(pred == reference_pred)),
    'n_prediction_differences': int(np.sum(pred != reference_pred)),
    'objective': float(objective), 'reference_balanced_accuracy': reference['balanced_accuracy'],
    **split_metrics(y[split.test], pred, sorted(set(y))),
    'warnings': [str(item.message) for item in captured],
}
print(json.dumps(record, indent=2), flush=True)
