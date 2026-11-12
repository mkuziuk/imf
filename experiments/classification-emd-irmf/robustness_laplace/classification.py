"""Clean-trained shift and matched fits, as in the impulse run.

Shift classical fits refit the pipeline at the C the impulse run selected on the
same clean training rows. The refit counts only if its clean predictions equal
the stored ones; otherwise the full grouped grid search runs.
"""
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
from robustness.classification import CLASSICAL, FEATURE_SETS, excluded_split, predict_conditions
from robustness.environment import verify
from robustness.pilot import METHODS
from robustness_laplace.contamination import CONDITIONS, MATCHED, subset_ids
from robustness_laplace.processing import MANIFEST, matrices
from robustness_laplace.storage import check_disk, digest, publish, root, save_json, save_npz

SEED = 20261021
_STATE = {}


def checkpoint(dataset, scenario, classifier, method, feature_set, split_id, condition='clean'):
    return root(dataset)/'classification'/scenario/classifier/method/feature_set/f'split_{split_id:02d}_{condition}'


def impulse_checkpoint(dataset, classifier, method, feature_set, split_id):
    return (data_root()/dataset/'robustness/classification/shift'/classifier/method/feature_set
            / f'split_{split_id:02d}_clean')


def materialize(dataset):
    out = root(dataset)
    if (out/'matrices/complete.json').exists():
        return dataset
    values, valid, columns = matrices(dataset)
    for condition in CONDITIONS:
        for method in METHODS:
            path = out/'matrices'/condition/f'{method}.npy'
            if not path.exists():
                publish(path, lambda f, a=values[condition][method]: np.save(f, a, allow_pickle=False))
        path = out/'matrices'/condition/'valid.npy'
        if not path.exists():
            publish(path, lambda f, a=valid[condition]: np.save(f, a, allow_pickle=False))
    save_json(out/'matrices/complete.json', dict(columns=columns, feature_manifest_sha256=digest(out/MANIFEST)))
    check_disk()
    return dataset


def state(dataset):
    if dataset not in _STATE:
        out = root(dataset)
        meta, info = load_meta(dataset), load_info(dataset)
        keep = np.ones(len(meta), dtype=bool)
        keep[subset_ids(dataset, len(meta))] = False
        xs = {c: {m: np.load(out/'matrices'/c/f'{m}.npy', mmap_mode='r', allow_pickle=False) for m in METHODS}
              for c in CONDITIONS}
        valid = {c: np.load(out/'matrices'/c/'valid.npy', allow_pickle=False) for c in CONDITIONS}
        _STATE[dataset] = dict(y=meta[label_column(info)].to_numpy(dtype=str),
                               groups=meta.group.to_numpy(dtype=str), keep=keep, valid=valid, xs=xs,
                               splits={s.split_id: s for s in load_splits(dataset)},
                               channels=len(info['channels']), task=info['primary_task'])
    return _STATE[dataset]


def reference_clean(dataset, classifier, method, feature_set, sid):
    where = impulse_checkpoint(dataset, classifier, method, feature_set, sid)
    value = json.loads((where/'complete.json').read_text())
    with np.load(where/'predictions.npz', allow_pickle=False) as z:
        ids, pred = z['clean__ids'], z['clean__pred']
    return json.loads(value['rows'][0]['best_params']), ids, pred


def fit_classical(classifier, X, y, groups, params=None):
    """Grid search when params is None, else the pipeline refit at params."""
    search = clone(classifiers()[classifier])
    if params is None:
        search.fit(X, y, groups=groups)
        return search, search.best_params_
    model = clone(search.estimator).set_params(**params)
    model.fit(X, y)
    return model, params


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
    n_comp = s['xs']['clean'][method].shape[1]-s['channels']
    xs = {c: s['xs'][c][method] if feature_set.endswith('log_scale') else s['xs'][c][method][:, :n_comp]
          for c in CONDITIONS}
    if classifier == 'lightgbm':
        xs = {c: np.asarray(x, dtype=np.float32) for c, x in xs.items()}
    conditions = CONDITIONS if scenario == 'shift' else (condition,)
    selection, reference_match = 'grid_search', None
    np.random.seed(SEED+sid)
    wall, cpu = time.perf_counter(), time.process_time()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        if classifier == 'lightgbm':
            encoder = LabelEncoder().fit(y)
            model, imputer, params = fit_tree('lightgbm', SimpleNamespace(workers=1, quick=False),
                                             xs[condition], encoder.transform(y), s['groups'], split)
        else:
            imputer, params = None, None
            if scenario == 'shift':
                params, reference_ids, reference_pred = reference_clean(dataset, classifier, method, feature_set, sid)
                selection = 'impulse_run_choice'
            model, params = fit_classical(classifier, xs[condition][split.train], y[split.train],
                                          s['groups'][split.train], params)
        predicted = predict_conditions(model, imputer, xs, split.test, s['valid'], conditions)
        if classifier != 'lightgbm' and scenario == 'shift':
            ids, pred = predicted['clean']
            reference_match = bool(np.array_equal(ids, reference_ids)
                                   and np.array_equal(pred.astype(str), reference_pred))
            if not reference_match:
                np.random.seed(SEED+sid)
                model, params = fit_classical(classifier, xs[condition][split.train], y[split.train],
                                              s['groups'][split.train])
                predicted = predict_conditions(model, imputer, xs, split.test, s['valid'], conditions)
                selection = 'grid_search_fallback'
        fit_seconds, fit_cpu = time.perf_counter()-wall, time.process_time()-cpu
    cpu, wall = time.process_time(), time.perf_counter()
    metrics, arrays = [], dict(test_ids=split.test)
    for c in conditions:
        ids, pred = predicted[c]
        pred = encoder.inverse_transform(pred.astype(int)) if classifier == 'lightgbm' else pred.astype(str)
        predicted[c] = (ids, np.asarray(pred, dtype=str))
    if scenario == 'shift':
        baseline = dict(zip(predicted['clean'][0].tolist(), predicted['clean'][1].tolist()))
        if classifier == 'lightgbm':
            _, reference_ids, reference_pred = reference_clean(dataset, classifier, method, feature_set, sid)
            reference_match = bool(np.array_equal(predicted['clean'][0], reference_ids)
                                   and np.array_equal(predicted['clean'][1], reference_pred))
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
                   best_params=json.dumps(params), selection=selection)
        row.update({k: float(v) for k, v in split_metrics(y[ids], pred, sorted(labels)).items()})
        baseline_pred = np.array([baseline[int(i)] for i in ids], dtype=str)
        row.update({'clean_'+k: float(v) for k, v in split_metrics(y[ids], baseline_pred, sorted(labels)).items()})
        metrics.append(row)
    if not (where/'predictions.npz').exists():
        save_npz(where/'predictions.npz', **arrays)
    save_json(where/'complete.json', dict(
        rows=metrics, fit_seconds=fit_seconds, fit_cpu_seconds=fit_cpu,
        predict_seconds=time.perf_counter()-wall, predict_cpu_seconds=time.process_time()-cpu,
        selection=selection, clean_matches_impulse_run=reference_match,
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
                    shift_classical_selection='impulse run choice, verified on clean predictions',
                    pimf=verify())
    path = out/'classification_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError(f'{dataset}: classification manifest mismatch; refusing to mix results')
    else:
        save_json(path, manifest)
    return manifest


def fit_jobs(dataset):
    """(priority seconds, args) for every unfinished fit; matched fits depend on shift."""
    costs = pd.read_csv(OUTPUT_ROOT/'robustness/fit_costs.csv')
    mean = costs.groupby(['dataset', 'scenario', 'classifier', 'method', 'feature_set']).fit_cpu_seconds.mean()
    jobs = []
    for split in load_splits(dataset):
        for method in METHODS:
            for fs in FEATURE_SETS:
                for clf in (CLASSICAL[dataset], 'lightgbm'):
                    cost = mean[(dataset, 'shift', clf, method, fs)]
                    jobs.append((cost/8 if clf != 'lightgbm' else cost,
                                 ('shift', dataset, clf, method, fs, split.split_id, 'clean')))
                for c in MATCHED:
                    jobs.append((mean[(dataset, 'matched', CLASSICAL[dataset], method, fs)],
                                 ('matched', dataset, CLASSICAL[dataset], method, fs, split.split_id, c)))
    return [(cost, args) for cost, args in jobs if not (job_checkpoint(args)/'complete.json').exists()]


def job_checkpoint(args):
    scenario, dataset, clf, method, fs, sid, condition = args
    return checkpoint(dataset, scenario, clf, method, fs, sid, condition)


def shift_dependency(args):
    scenario, dataset, clf, method, fs, sid, _ = args
    return ('shift', dataset, clf, method, fs, sid, 'clean') if scenario == 'matched' else None
