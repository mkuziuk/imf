"""Full grouped grids, shift predictions and method-wise K selection."""
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder

from baselines.run_classical import classifiers
from emdirmf.cv import load_splits, split_metrics
from emdirmf.io import label_column, load_info, load_meta
from emdirmf.paths import OUTPUT_ROOT, data_root
from ml.run_trees import fit_tree
from robustness.classification import CLASSICAL, excluded_split, predict_conditions
from robustness.environment import verify
from robustness_v3.construction import METHODS, n_slots, parse_variant, variants
from robustness_v3.contamination import CONDITIONS, MATCHED, subset_ids
from robustness_v3.processing import MANIFEST, materialize_condition
from robustness_v3.storage import check_disk, digest, publish, root, save_json, save_npz

FEATURE_SETS = ('comp',)
CLASSIFICATION_MANIFEST = 'classification_manifest_v2.json'
SEED = 20261021
_STATE = {}


def checkpoint(dataset, scenario, classifier, method, feature_set, split_id, condition='clean'):
    return root(dataset)/'classification'/scenario/classifier/method/feature_set/f'split_{split_id:02d}_{condition}'


def materialize(dataset):
    out = root(dataset)
    if (out/'matrices/complete.json').exists():
        return dataset
    for condition in CONDITIONS:
        columns = materialize_condition(dataset, condition)
    save_json(out/'matrices/complete.json', dict(columns=columns, feature_manifest_sha256=digest(out/MANIFEST),
        hashes={str(p.relative_to(out)): digest(p) for p in sorted((out/'matrices').rglob('*.npy'))}))
    check_disk()
    return dataset


def state(dataset):
    if dataset not in _STATE:
        out = root(dataset)
        meta, info = load_meta(dataset), load_info(dataset)
        keep = np.ones(len(meta), dtype=bool)
        keep[subset_ids(dataset, len(meta))] = False
        xs = {c: {m: np.load(out/'matrices'/c/f'{m}.npy', mmap_mode='r', allow_pickle=False) for m in variants(info['frame_length'])}
              for c in CONDITIONS}
        valid = {c: np.load(out/'matrices'/c/'valid.npy', allow_pickle=False) for c in CONDITIONS}
        _STATE[dataset] = dict(y=meta[label_column(info)].to_numpy(dtype=str),
                               groups=meta.group.to_numpy(dtype=str), keep=keep, valid=valid, xs=xs,
                               splits={s.split_id: s for s in load_splits(dataset)},
                               channels=len(info['channels']), task=info['primary_task'])
    return _STATE[dataset]


def fit_classical(classifier, X, y, groups):
    search = clone(classifiers()[classifier])
    search.fit(X, y, groups=groups)
    return search, search.best_params_


def job(scenario, dataset, classifier, method, feature_set, sid, condition):
    s = state(dataset)
    y = s['y']
    where = checkpoint(dataset, scenario, classifier, method, feature_set, sid, condition)
    if (where/'complete.json').exists():
        return str(where)
    split = excluded_split(s['splits'][sid], s['keep'] & s['valid'][condition])
    labels = set(y[s['keep']])
    if set(y[split.train]) != labels or set(y[split.test]) != labels:
        raise RuntimeError(f'{dataset}/{sid}/{condition}: excluded fold loses classes')
    xs = {c: s['xs'][c][method] for c in CONDITIONS}
    if classifier == 'lightgbm':
        xs = {c: np.asarray(x, dtype=np.float32) for c, x in xs.items()}
    conditions = CONDITIONS if scenario == 'shift' else (condition,)
    selection = 'grid_search'
    np.random.seed(SEED+sid)
    wall, cpu = time.perf_counter(), time.process_time()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        if classifier == 'lightgbm':
            encoder = LabelEncoder().fit(y)
            model, imputer, params = fit_tree('lightgbm', SimpleNamespace(workers=1, quick=False),
                                             xs[condition], encoder.transform(y), s['groups'], split)
        else:
            imputer = None
            xs = {c: np.asarray(x, dtype=np.float64) for c, x in xs.items()}
            model, params = fit_classical(classifier, xs[condition][split.train], y[split.train],
                                          s['groups'][split.train])
        fit_seconds, fit_cpu = time.perf_counter()-wall, time.process_time()-cpu
        cpu, wall = time.process_time(), time.perf_counter()
        predicted = predict_conditions(model, imputer, xs, split.test, s['valid'], conditions)
    metrics, arrays = [], dict(test_ids=split.test)
    for c in conditions:
        ids, pred = predicted[c]
        pred = encoder.inverse_transform(pred.astype(int)) if classifier == 'lightgbm' else pred.astype(str)
        predicted[c] = (ids, np.asarray(pred, dtype=str))
    if scenario == 'shift':
        baseline = dict(zip(predicted['clean'][0].tolist(), predicted['clean'][1].tolist()))
    else:
        clean_where = checkpoint(dataset, 'shift', classifier, method, feature_set, sid)
        with np.load(clean_where/'predictions.npz', allow_pickle=False) as z:
            baseline = dict(zip(z['clean__ids'].tolist(), z['clean__pred'].tolist()))
    for c in conditions:
        ids, pred = predicted[c]
        if set(y[ids]) != labels:
            raise RuntimeError('condition-valid test fold loses classes')
        arrays[c+'__ids'], arrays[c+'__pred'] = ids, pred
        row = dict(dataset=dataset, task=s['task'], scenario=scenario, classifier=classifier, method=method,
                   feature_set=feature_set, condition=c, split_id=sid, seed=split.seed, fold=split.fold,
                   n_train=len(split.train), n_test=len(ids), fit_key=str(where.relative_to(root(dataset))),
                   best_params=json.dumps(params), selection=selection,
                   variant=method, family=parse_variant(method)[0], k=parse_variant(method)[1])
        row.update({k: float(v) for k, v in split_metrics(y[ids], pred, sorted(labels)).items()})
        baseline_pred = np.array([baseline[int(i)] for i in ids], dtype=str)
        row.update({'clean_'+k: float(v) for k, v in split_metrics(y[ids], baseline_pred, sorted(labels)).items()})
        metrics.append(row)
    if not (where/'predictions.npz').exists():
        save_npz(where/'predictions.npz', **arrays)
    save_json(where/'complete.json', dict(
        rows=metrics, fit_seconds=fit_seconds, fit_cpu_seconds=fit_cpu,
        predict_seconds=time.perf_counter()-wall, predict_cpu_seconds=time.process_time()-cpu,
        selection=selection,
        warnings={str(w.message): sum(str(z.message) == str(w.message) for z in notices) for w in notices},
        pimf=verify()))
    return str(where)


def write_manifest(dataset):
    out = root(dataset)
    study = Path(__file__).parents[1]
    source = [Path(__file__), study/'robustness/classification.py', study/'baselines/run_classical.py',
              study/'ml/run_trees.py', study/'ml/common.py', study/'emdirmf/cv.py']
    manifest = dict(feature_manifest_sha256=digest(out/MANIFEST),
                    source_hashes={str(p.relative_to(study)): digest(p) for p in source},
                    matched_conditions=list(MATCHED), matched_lightgbm=False, seed=SEED,
                    shift_classical_selection='full shared grouped C grid for every variant',
                    k_selection='mean clean outer BA over 15 splits; ties smaller K',
                    pimf=verify())
    path = out/CLASSIFICATION_MANIFEST
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f'{dataset}: classification manifest mismatch; refusing to mix results')
    else:
        save_json(path, manifest)
    return manifest


def dataset_variants(dataset):
    return tuple(json.loads((root(dataset)/MANIFEST).read_text())['methods'])


def shift_dependencies(dataset, method, classifier):
    return tuple(('shift', dataset, classifier, v, 'comp', sid, 'clean')
                 for v in dataset_variants(dataset) if parse_variant(v)[0] == method for sid in range(15))


def select_k(rows):
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError('no clean rows')
    means = []
    for variant, cell in table.groupby('variant'):
        if sorted(cell.split_id) != list(range(15)) or not np.isfinite(cell.balanced_accuracy).all():
            raise ValueError('K selection requires all 15 clean split scores for every variant')
        means.append((float(cell.balanced_accuracy.mean()), parse_variant(variant)[1], variant))
    score, k, variant = sorted(means, key=lambda x: (-x[0], x[1]))[0]
    return dict(variant=variant, k=k, mean_clean_ba=score,
                mean_clean_ba_by_variant={v: a for a, _, v in means})


def selection_path(dataset, method, classifier):
    return root(dataset)/'k_selection'/f'{classifier}__{method}.json'


def maybe_select(dataset, method, classifier):
    path = selection_path(dataset, method, classifier)
    if path.exists():
        return json.loads(path.read_text())
    dependencies = shift_dependencies(dataset, method, classifier)
    if any(not (job_checkpoint(a)/'complete.json').exists() for a in dependencies):
        return None
    rows = []
    for args in dependencies:
        value = json.loads((job_checkpoint(args)/'complete.json').read_text())
        rows.extend(r for r in value['rows'] if r['condition'] == 'clean')
    selection = select_k(rows)
    record = dict(dataset=dataset, method=method, classifier=classifier, **selection,
                  criterion='argmax mean clean BA over all 15 splits; ties smaller K',
                  dependencies={str(job_checkpoint(a).relative_to(root(dataset))): digest(job_checkpoint(a)/'complete.json')
                                for a in dependencies})
    save_json(path, record)
    return record


def write_k_summary(dataset):
    selected = {}
    for clf in (CLASSICAL[dataset], 'lightgbm'):
        selected[clf] = {}
        for method in METHODS:
            value = maybe_select(dataset, method, clf)
            if value is None:
                return
            selected[clf][method] = value
    path = root(dataset)/'k_star.json'
    if not path.exists():
        save_json(path, dict(dataset=dataset, classifiers=selected))


def expected_cost(dataset, classifier, variant, scenario='shift'):
    method, k = parse_variant(variant)
    # Prior measured CPU costs give the ordering; each new fit still runs its full search.
    path = OUTPUT_ROOT/('robustness_laplace' if classifier == 'lightgbm' else 'robustness')/'fit_costs.csv'
    if path.exists():
        costs = pd.read_csv(path)
        old = {'linear': 'imf_linear_a1.41', 'irmf': 'irmf_h2_a1.41_conv', 'emd': 'emd', 'raw': 'raw'}[method]
        cell = costs[(costs.dataset == dataset) & (costs.scenario == scenario) &
                     (costs.classifier == classifier) & (costs.method == old) & (costs.feature_set == 'comp')]
        if len(cell):
            return max(.01, float(cell.fit_cpu_seconds.mean()) * max(1, k)/8)
    return max(1, k) * (10 if classifier == 'lightgbm' else 3)


def fit_jobs(dataset):
    jobs = []
    for variant in dataset_variants(dataset):
        for clf in (CLASSICAL[dataset], 'lightgbm'):
            cost = expected_cost(dataset, clf, variant)
            for sid in range(15):
                args = ('shift', dataset, clf, variant, 'comp', sid, 'clean')
                if not (job_checkpoint(args)/'complete.json').exists():
                    jobs.append((cost, args))
    return jobs


def matched_jobs(dataset, method):
    selection = maybe_select(dataset, method, CLASSICAL[dataset])
    if selection is None:
        return []
    variant = selection['variant']
    cost = expected_cost(dataset, CLASSICAL[dataset], variant, 'matched')
    return [(cost, a) for sid in range(15)
            if not (job_checkpoint(a := ('matched', dataset, CLASSICAL[dataset], variant, 'comp', sid, MATCHED[0]))/'complete.json').exists()]


def job_checkpoint(args):
    scenario, dataset, clf, method, fs, sid, condition = args
    return checkpoint(dataset, scenario, clf, method, fs, sid, condition)
