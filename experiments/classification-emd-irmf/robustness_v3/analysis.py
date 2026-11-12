"""Clean-selected K comparisons and the amended multiplicity families."""
import json

import numpy as np
import pandas as pd

from emdirmf.stats import corrected_resampled_ttest, holm, wilcoxon_signed_rank
from robustness.classification import CLASSICAL
from robustness.environment import verify
from robustness_v3.classification import dataset_variants, select_k
from robustness_v3.construction import METHODS, parse_variant
from robustness_v3.contamination import CONDITIONS, DATASETS, MATCHED
from robustness_v3.storage import root

METRICS = ('balanced_accuracy', 'macro_f1', 'accuracy')
FAMILY_SIZES = dict(Primary=12, S1=12, S3=24, S5=6, synthetic=18)


def selections(scores):
    rows = []
    clean = scores[(scores.scenario == 'shift') & (scores.condition == 'clean')]
    for (dataset, classifier, method), cell in clean.groupby(['dataset', 'classifier', 'family']):
        rows.append(dict(dataset=dataset, classifier=classifier, method=method, **select_k(cell.to_dict('records'))))
    return pd.DataFrame(rows)


def selected_scores(scores):
    selected = selections(scores)[['dataset', 'classifier', 'variant']]
    return scores.merge(selected, on=['dataset', 'classifier', 'variant'], validate='many_to_one')


def load_scores(dataset):
    rows, costs = [], []
    checkpoints = sorted((root(dataset)/'classification').rglob('complete.json'))
    for p in checkpoints:
        value = json.loads(p.read_text())
        rows.extend(value['rows'])
        key = value['rows'][0]
        costs.append({k: key[k] for k in ('dataset', 'scenario', 'classifier', 'method', 'feature_set', 'fit_key')}
            | {k: value[k] for k in ('fit_seconds', 'fit_cpu_seconds', 'predict_seconds', 'predict_cpu_seconds', 'selection')}
            | {'warning_count': sum(value['warnings'].values())})
    table = pd.DataFrame(rows)
    vs = dataset_variants(dataset)
    expected_fits, expected_scores = len(vs)*30 + 60, len(vs)*150 + 60
    if len(checkpoints) != expected_fits or len(table) != expected_scores:
        raise RuntimeError(f'{dataset}: expected {expected_fits} fits / {expected_scores} scores; '
                           f'found {len(checkpoints)} / {len(table)}')
    keys = ['scenario', 'classifier', 'method', 'feature_set', 'condition', 'split_id']
    expected = {('shift', clf, v, 'comp', c, sid) for clf in (CLASSICAL[dataset], 'lightgbm')
                for v in vs for c in CONDITIONS for sid in range(15)}
    selected = selections(table)
    expected |= {('matched', CLASSICAL[dataset], v, 'comp', MATCHED[0], sid)
                 for v in selected[selected.classifier == CLASSICAL[dataset]].variant for sid in range(15)}
    if table.duplicated(keys).any() or set(table[keys].itertuples(index=False, name=None)) != expected:
        raise AssertionError('duplicate, missing or unexpected result key')
    if not np.isfinite(table[list(METRICS)+['clean_'+m for m in METRICS]]).all().all():
        raise AssertionError('nonfinite score')
    return table, pd.DataFrame(costs)


def specs(dataset):
    for family, clf, scenario, comparator, conditions in [
        ('Primary', CLASSICAL[dataset], 'shift', 'linear', CONDITIONS[1:]),
        ('S1', CLASSICAL[dataset], 'shift', 'emd', CONDITIONS[1:]),
        *[('S3', 'lightgbm', 'shift', comp, CONDITIONS[1:]) for comp in ('linear', 'emd')],
        *[('S5', CLASSICAL[dataset], 'matched', comp, MATCHED) for comp in ('linear', 'emd')],
    ]:
        for condition in conditions:
            yield dict(dataset=dataset, family=family, classifier=clf, feature_set='comp',
                       scenario=scenario, comparator=comparator, condition=condition, contrast='DiD')


def did_vector(scores, dataset, classifier, scenario, condition, comparator, **unused):
    cell = scores[(scores.dataset == dataset) & (scores.classifier == classifier)
                  & (scores.scenario == scenario) & (scores.condition == condition)]
    a = cell[cell.family == 'irmf'].set_index('split_id').sort_index()
    b = cell[cell.family == comparator].set_index('split_id').sort_index()
    if list(a.index) != list(range(15)) or not a.index.equals(b.index):
        raise AssertionError('unpaired DiD')
    for name in ('n_train', 'n_test'):
        np.testing.assert_array_equal(a[name], b[name])
    d = (a.balanced_accuracy-b.balanced_accuracy) - (a.clean_balanced_accuracy-b.clean_balanced_accuracy)
    return d.to_numpy(), a.n_train.to_numpy(), a.n_test.to_numpy(), int(a.k.iloc[0]), int(b.k.iloc[0])


def holm_padded(pvalues, size):
    p = np.asarray(pvalues, dtype=float)
    if len(p) > size:
        raise AssertionError('family larger than specified')
    return holm(np.r_[p, np.ones(size-len(p))])[:len(p)]


def paired_tests(scores, datasets=DATASETS):
    verify()
    scores = selected_scores(scores)
    rows, vectors = [], []
    for dataset in datasets:
        for spec in specs(dataset):
            d, train, test, ka, kb = did_vector(scores, **spec)
            row = dict(spec, irmf_k=ka, comparator_k=kb, n_splits=len(d),
                       holm_family='synthetic' if dataset == 'synthetic' else spec['family'],
                       status='synthetic, separate family' if dataset == 'synthetic' else
                       ('primary' if spec['family'] == 'Primary' else 'exploratory'))
            row.update(corrected_resampled_ttest(d, train, test))
            row.update(wilcoxon_signed_rank(d))
            rows.append(row)
            vectors.extend(dict(spec, irmf_k=ka, comparator_k=kb, split_id=sid, did=float(value),
                                n_train=int(nt), n_test=int(ne))
                           for sid, value, nt, ne in zip(range(15), d, train, test))
    result = pd.DataFrame(rows)
    for family, indices in result.groupby('holm_family').groups.items():
        size = FAMILY_SIZES[family]
        for field in ('p_nb', 'p_wilcoxon'):
            result.loc[indices, field+'_holm'] = holm_padded(result.loc[indices, field], size)
        result.loc[indices, 'holm_complete'] = len(indices) == size
    if set(datasets) == set(DATASETS) and result.groupby('holm_family').size().to_dict() != FAMILY_SIZES:
        raise AssertionError('incorrect complete family sizes')
    return result, pd.DataFrame(vectors)


def descriptive_tables(scores):
    keys = ['dataset', 'scenario', 'classifier', 'family', 'variant', 'k', 'condition']
    every = scores.groupby(keys).balanced_accuracy.agg(['mean', 'std', 'size']).reset_index()
    best = every[every.scenario == 'shift'].sort_values(['mean', 'k'], ascending=[False, True])
    best = best.drop_duplicates(['dataset', 'classifier', 'family', 'condition'])
    chosen = selections(scores)
    selected = selected_scores(scores).groupby(keys).balanced_accuracy.agg(['mean', 'std']).reset_index()
    return every, best, selected, chosen
