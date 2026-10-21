"""Paired DiD vectors and pre-specified multiplicity families."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from emdirmf.io import label_column, load_info, load_meta
from emdirmf.stats import corrected_resampled_ttest, holm, wilcoxon_signed_rank
from robustness.classification import CLASSICAL
from robustness.contamination import CONDITIONS, DATASETS
from robustness.environment import verify
from robustness.pilot import METHODS
from robustness.storage import root, save_json

ROBUST = 'irmf_h2_a1.41_conv'
LINEAR = 'imf_linear_a1.41'
METRICS = ('balanced_accuracy', 'macro_f1', 'accuracy')


def load_scores(dataset):
    rows, costs = [], []
    checkpoints = sorted((root(dataset)/'classification').rglob('complete.json'))
    for p in checkpoints:
        value = json.loads(p.read_text())
        rows.extend(value['rows'])
        key = value['rows'][0]
        costs.append({k: key[k] for k in ('dataset','scenario','classifier','method','feature_set','fit_key')}
                     | {k: value[k] for k in ('fit_seconds','fit_cpu_seconds','predict_seconds','predict_cpu_seconds')}
                     | {'warning_count': sum(value['warnings'].values())})
    table = pd.DataFrame(rows)
    if len(checkpoints) != 600 or len(table) != 2760:
        raise RuntimeError(f'{dataset}: expected 600 fits / 2760 scores; found {len(checkpoints)} / {len(table)}')
    keys = ['scenario','classifier','method','feature_set','condition','split_id']
    if table.duplicated(keys).any():
        raise AssertionError('duplicate result key')
    for _, cell in table.groupby(keys[:-1]):
        if sorted(cell.split_id) != list(range(15)):
            raise AssertionError('incomplete paired splits')
    return table, pd.DataFrame(costs)


def did_vector(scores, dataset, classifier, feature_set, scenario, condition, comparator):
    cell = scores[(scores.dataset == dataset) & (scores.classifier == classifier)
                  & (scores.feature_set == feature_set) & (scores.scenario == scenario)
                  & (scores.condition == condition)]
    a = cell[cell.method == ROBUST].set_index('split_id').sort_index()
    b = cell[cell.method == comparator].set_index('split_id').sort_index()
    if not np.array_equal(a.index, b.index) or len(a) != 15:
        raise AssertionError('unpaired DiD')
    for c in ('n_train','n_test'):
        np.testing.assert_array_equal(a[c], b[c])
    d = (a.balanced_accuracy-b.balanced_accuracy) - (a.clean_balanced_accuracy-b.clean_balanced_accuracy)
    return d.to_numpy(), a.n_train.to_numpy(), a.n_test.to_numpy()


def specs(dataset):
    """Original real-data families plus exact counterparts for amended synthetic."""
    classical = CLASSICAL[dataset]
    impulses = [c for c in CONDITIONS if c.startswith('impulse')]
    gauss = [c for c in CONDITIONS if c.startswith('gauss')]
    for family, clf, fs, scenario, comparator, conditions in [
        ('Primary', classical, 'comp', 'shift', LINEAR, impulses),
        ('S1', classical, 'comp', 'shift', 'emd', impulses),
        ('S2', classical, 'comp', 'shift', LINEAR, gauss),
        *[('S3', 'lightgbm', 'comp', 'shift', comp, CONDITIONS[1:]) for comp in (LINEAR, 'emd')],
        *[('S4', clf, 'comp+log_scale', 'shift', comp, CONDITIONS[1:])
          for clf in (classical, 'lightgbm') for comp in (LINEAR, 'emd')],
        *[('S5', classical, fs, 'matched', comp, [c for c in CONDITIONS[1:] if c.endswith('_0.04')])
          for fs in ('comp','comp+log_scale') for comp in (LINEAR,'emd')],
    ]:
        for condition in conditions:
            yield dict(dataset=dataset, family=family, classifier=clf, feature_set=fs,
                       scenario=scenario, comparator=comparator, condition=condition, contrast='DiD')
    for condition in impulses:
        yield dict(dataset=dataset, family='S2', classifier=classical, feature_set='comp',
                   scenario='shift', comparator=LINEAR, condition=condition, contrast='impulse_minus_gauss_DiD')


def paired_tests(scores, datasets=DATASETS):
    verify()
    rows, vectors = [], []
    for dataset in datasets:
        for spec in specs(dataset):
            args = {k: spec[k] for k in ('dataset','classifier','feature_set','scenario','condition','comparator')}
            d, train, test = did_vector(scores, **args)
            if spec['contrast'] == 'impulse_minus_gauss_DiD':
                other = dict(args, condition='gauss_'+spec['condition'].rsplit('_',1)[1])
                dg, nt, ne = did_vector(scores, **other)
                if not np.array_equal(test, ne):
                    raise RuntimeError('Differential failures require common-frame rescoring before impulse/Gaussian tests')
                d = d-dg
            row = dict(spec, n_splits=len(d), holm_family=('synthetic_added' if dataset == 'synthetic' else spec['family']),
                       status=('synthetic, added after pre-registration at the user\'s request'
                               if dataset == 'synthetic' else ('confirmatory' if spec['family']=='Primary' else 'exploratory')))
            row.update(corrected_resampled_ttest(d, train, test))
            row.update(wilcoxon_signed_rank(d))
            rows.append(row)
            for sid, value, n_train, n_test in zip(range(15), d, train, test):
                vectors.append(dict(spec, split_id=sid, did=float(value), n_train=int(n_train), n_test=int(n_test)))
    result = pd.DataFrame(rows)
    for _, indices in result.groupby('holm_family').groups.items():
        result.loc[indices,'p_nb_holm'] = holm(result.loc[indices,'p_nb'])
        result.loc[indices,'p_wilcoxon_holm'] = holm(result.loc[indices,'p_wilcoxon'])
    expected = dict(Primary=18, S1=18, S2=27, S3=54, S4=108, S5=36, synthetic_added=87)
    actual = result.groupby('holm_family').size().to_dict()
    for family, n in actual.items():
        required = 87 if family == 'synthetic_added' else expected[family]
        if n != required:
            raise AssertionError(f'{family}: {n} instead of {required}')
    return result, pd.DataFrame(vectors)


def smoke():
    current = verify()
    scores, costs = load_scores('synthetic')
    tests, vectors = paired_tests(scores, ('synthetic',))
    if not np.isfinite(tests[['mean_diff','ci_low','ci_high','p_nb_holm']].to_numpy()).all():
        raise AssertionError('nonfinite smoke tests')
    from robustness.processing import VERIFICATION
    check = json.loads((root('synthetic')/VERIFICATION).read_text())
    if not check['ok']:
        raise AssertionError('clean verification failed')
    # Check score and prediction agreement for every fit and every condition.
    from emdirmf.cv import split_metrics
    meta, info = load_meta('synthetic'), load_info('synthetic')
    y = meta[label_column(info)].to_numpy(dtype=str)
    for path in sorted((root('synthetic')/'classification').rglob('complete.json')):
        payload = json.loads(path.read_text())
        with np.load(path.with_name('predictions.npz'), allow_pickle=False) as z:
            for row in payload['rows']:
                c = row['condition']
                actual = split_metrics(y[z[c+'__ids']], z[c+'__pred'], sorted(set(y)))
                for metric in METRICS:
                    if abs(actual[metric]-row[metric]) > 1e-12:
                        raise AssertionError('prediction-score mismatch')
    out = root('synthetic')/'smoke'
    out.mkdir(parents=True, exist_ok=True)
    for name, table in [('per_split_scores.csv', scores), ('paired_tests.csv', tests),
                        ('paired_vectors.csv', vectors), ('fit_costs.csv', costs)]:
        if not (out/name).exists():
            table.to_csv(out/name, index=False, mode='x')
    path = out/'validation.json'
    if not path.exists():
        save_json(path, dict(passed=True, n_fits=len(costs), n_scores=len(scores), n_tests=len(tests),
                             prediction_scores_recomputed=True, pimf=current))
    print(f'Synthetic smoke: {len(costs)} fits, {len(scores)} scores, {len(tests)} amended tests validated', flush=True)
