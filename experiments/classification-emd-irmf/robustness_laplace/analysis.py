"""Paired DiD vectors and the protocol's multiplicity families."""
import json

import numpy as np
import pandas as pd

from emdirmf.stats import corrected_resampled_ttest, holm, wilcoxon_signed_rank
from robustness.analysis import LINEAR, METRICS, ROBUST, did_vector
from robustness.classification import CLASSICAL
from robustness.environment import verify
from robustness_laplace.contamination import CONDITIONS, DATASETS, MATCHED
from robustness_laplace.storage import root

__all__ = ['CLASSICAL', 'DATASETS', 'FAMILY_SIZES', 'LINEAR', 'METRICS', 'ROBUST', 'load_scores',
           'paired_tests', 'specs']

FITS, SCORES = 360, 1320
FAMILY_SIZES = dict(Primary=12, S1=12, S3=24, S4=48, S5=12, synthetic=36)
SYNTHETIC_STATUS = 'synthetic, separate family'


def load_scores(dataset):
    rows, costs = [], []
    checkpoints = sorted((root(dataset)/'classification').rglob('complete.json'))
    for p in checkpoints:
        value = json.loads(p.read_text())
        rows.extend(value['rows'])
        key = value['rows'][0]
        costs.append({k: key[k] for k in ('dataset', 'scenario', 'classifier', 'method', 'feature_set', 'fit_key')}
                     | {k: value[k] for k in ('fit_seconds', 'fit_cpu_seconds', 'predict_seconds',
                                              'predict_cpu_seconds', 'selection', 'clean_matches_impulse_run')}
                     | {'warning_count': sum(value['warnings'].values())})
    table = pd.DataFrame(rows)
    if len(checkpoints) != FITS or len(table) != SCORES:
        raise RuntimeError(f'{dataset}: expected {FITS} fits / {SCORES} scores; '
                           f'found {len(checkpoints)} / {len(table)}')
    keys = ['scenario', 'classifier', 'method', 'feature_set', 'condition', 'split_id']
    if table.duplicated(keys).any():
        raise AssertionError('duplicate result key')
    for _, cell in table.groupby(keys[:-1]):
        if sorted(cell.split_id) != list(range(15)):
            raise AssertionError('incomplete paired splits')
    return table, pd.DataFrame(costs)


def specs(dataset):
    classical = CLASSICAL[dataset]
    laplace = list(CONDITIONS[1:])
    for family, clf, fs, scenario, comparator, conditions in [
        ('Primary', classical, 'comp', 'shift', LINEAR, laplace),
        ('S1', classical, 'comp', 'shift', 'emd', laplace),
        *[('S3', 'lightgbm', 'comp', 'shift', comp, laplace) for comp in (LINEAR, 'emd')],
        *[('S4', c, 'comp+log_scale', 'shift', comp, laplace) for c in (classical, 'lightgbm') for comp in (LINEAR, 'emd')],
        *[('S5', classical, fs, 'matched', comp, list(MATCHED)) for fs in ('comp', 'comp+log_scale')
          for comp in (LINEAR, 'emd')],
    ]:
        for condition in conditions:
            yield dict(dataset=dataset, family=family, classifier=clf, feature_set=fs,
                       scenario=scenario, comparator=comparator, condition=condition, contrast='DiD')


def holm_padded(pvalues, size):
    """Holm over a family of `size` tests; untested members count as p = 1.

    A complete family gives plain Holm. An incomplete one (a preview of some real
    datasets) never uses a smaller family than the protocol specifies.
    """
    p = np.asarray(pvalues, dtype=float)
    if len(p) > size:
        raise AssertionError(f'{len(p)} tests in a family of {size}')
    return holm(np.concatenate([p, np.ones(size - len(p))]))[:len(p)]


def paired_tests(scores, datasets=DATASETS):
    verify()
    rows, vectors = [], []
    for dataset in datasets:
        for spec in specs(dataset):
            args = {k: spec[k] for k in ('dataset', 'classifier', 'feature_set', 'scenario', 'condition', 'comparator')}
            d, train, test = did_vector(scores, **args)
            row = dict(spec, n_splits=len(d), holm_family='synthetic' if dataset == 'synthetic' else spec['family'],
                       status=SYNTHETIC_STATUS if dataset == 'synthetic' else
                       ('primary' if spec['family'] == 'Primary' else 'exploratory'))
            row.update(corrected_resampled_ttest(d, train, test))
            row.update(wilcoxon_signed_rank(d))
            rows.append(row)
            for sid, value, n_train, n_test in zip(range(15), d, train, test):
                vectors.append(dict(spec, split_id=sid, did=float(value), n_train=int(n_train), n_test=int(n_test)))
    result = pd.DataFrame(rows)
    for family, indices in result.groupby('holm_family').groups.items():
        size = FAMILY_SIZES[family]
        result.loc[indices, 'p_nb_holm'] = holm_padded(result.loc[indices, 'p_nb'], size)
        result.loc[indices, 'p_wilcoxon_holm'] = holm_padded(result.loc[indices, 'p_wilcoxon'], size)
        result.loc[indices, 'holm_complete'] = len(indices) == size
    if set(datasets) == set(DATASETS):
        actual = result.groupby('holm_family').size().to_dict()
        if actual != FAMILY_SIZES:
            raise AssertionError(f'family sizes {actual} instead of {FAMILY_SIZES}')
    return result, pd.DataFrame(vectors)
