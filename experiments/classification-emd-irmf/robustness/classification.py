"""Paired clean-to-corrupt shift and matched-condition classification."""
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection._search import BaseSearchCV
from sklearn.preprocessing import LabelEncoder

from baselines.run_classical import classifiers
from emdirmf.cv import Split, load_splits, split_metrics
from emdirmf.io import label_column, load_info, load_meta
from emdirmf.parallel import limit_threads, single_thread_env
from ml.run_trees import fit_tree
from robustness.contamination import CONDITIONS, subset_ids
from robustness.environment import verify
from robustness.pilot import METHODS
from robustness.processing import MANIFEST, matrices
from robustness.storage import check_disk, digest, publish, root, save_json, save_npz

FEATURE_SETS = ('comp', 'comp+log_scale')
CLASSICAL = dict(synthetic='logreg', dronerf='linsvm', birdclef2023='logreg', esc50_animals='linsvm')
_STATE = {}


def excluded_split(split, keep):
    return Split(split.split_id, split.seed, split.fold, split.train[keep[split.train]], split.test[keep[split.test]])


def checkpoint(dataset, scenario, classifier, method, feature_set, split_id, condition='clean'):
    return root(dataset)/'classification'/scenario/classifier/method/feature_set/f'split_{split_id:02d}_{condition}'


def materialize(dataset):
    out = root(dataset)
    if (out/'matrices/complete.json').exists():
        return
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


def initialize(dataset):
    limit_threads()
    verify()
    out = root(dataset)
    meta, info = load_meta(dataset), load_info(dataset)
    y = meta[label_column(info)].to_numpy(dtype=str)
    groups = meta.group.to_numpy(dtype=str)
    excluded = subset_ids(dataset, len(meta))
    keep = np.ones(len(meta), dtype=bool)
    keep[excluded] = False
    xs, valid = {}, {}
    for condition in CONDITIONS:
        xs[condition] = {m: np.load(out/'matrices'/condition/f'{m}.npy', mmap_mode='r', allow_pickle=False)
                         for m in METHODS}
        valid[condition] = np.load(out/'matrices'/condition/'valid.npy', allow_pickle=False)
    _STATE.clear()
    _STATE.update(dataset=dataset, y=y, groups=groups, keep=keep, valid=valid, xs=xs,
                  splits={s.split_id: s for s in load_splits(dataset)}, channels=len(info['channels']))


def predict_conditions(model, imputer, xs, test, valid, conditions):
    """One fitted estimator, many paired held-out conditions; no fit occurs here."""
    result = {}
    for condition in conditions:
        mask = valid[condition][test]
        X = xs[condition][test[mask]]
        if imputer is not None:
            X = imputer.transform(X)
        pred = np.asarray(model.predict(X))
        result[condition] = (test[mask], pred)
    return result


def job(args):
    scenario, classifier, method, feature_set, sid, condition = args
    s = _STATE
    dataset, y = s['dataset'], s['y']
    where = checkpoint(dataset, scenario, classifier, method, feature_set, sid, condition)
    if (where/'complete.json').exists():
        return str(where), True
    split = excluded_split(s['splits'][sid], s['keep'] & s['valid'][condition])
    all_labels = set(y[s['keep']])
    if set(y[split.train]) != all_labels or set(y[split.test]) != all_labels:
        raise RuntimeError(f'{dataset}/{sid}/{condition}: excluded fold loses classes')
    n_comp = s['xs']['clean'][method].shape[1]-s['channels']
    xs = {c: s['xs'][c][method] if feature_set.endswith('log_scale') else s['xs'][c][method][:, :n_comp]
          for c in CONDITIONS}
    # The tree runner consumes float32; the classical runner consumes float64.
    # Keep shared cache matrices read-only; cast only where needed.
    if classifier == 'lightgbm':
        xs = {c: np.asarray(x, dtype=np.float32) for c, x in xs.items()}
    np.random.seed(20261021+sid)
    wall, cpu = time.perf_counter(), time.process_time()
    with warnings.catch_warnings(record=True) as notices:
        warnings.simplefilter('always')
        if classifier == 'lightgbm':
            encoder = LabelEncoder().fit(y)
            model, imputer, params = fit_tree('lightgbm', SimpleNamespace(workers=1, quick=False),
                                             xs[condition], encoder.transform(y), s['groups'], split)
        else:
            model = clone(classifiers()[classifier])
            options = {'groups': s['groups'][split.train]} if isinstance(model, BaseSearchCV) else {}
            model.fit(xs[condition][split.train], y[split.train], **options)
            imputer = None
            params = getattr(model, 'best_params_', {})
        fit_seconds, fit_cpu = time.perf_counter()-wall, time.process_time()-cpu
        cpu, wall = time.process_time(), time.perf_counter()
        conditions = CONDITIONS if scenario == 'shift' else (condition,)
        predicted = predict_conditions(model, imputer, xs, split.test, s['valid'], conditions)
    metrics, arrays = [], dict(test_ids=split.test)
    clean_pred = None
    if scenario == 'matched':
        clean_where = checkpoint(dataset, 'shift', classifier, method, feature_set, sid)
        with np.load(clean_where/'predictions.npz', allow_pickle=False) as z:
            clean_lookup = dict(zip(z['clean__ids'], z['clean__pred']))
        clean_pred = np.array([clean_lookup[int(i)] for i in split.test], dtype=str)
    for c in conditions:
        ids, pred = predicted[c]
        if classifier == 'lightgbm':
            pred = encoder.inverse_transform(pred.astype(int))
        else:
            pred = pred.astype(str)
        if set(y[ids]) != all_labels:
            raise RuntimeError('condition-valid test fold loses classes')
        arrays[c+'__ids'], arrays[c+'__pred'] = ids, pred
        row = dict(dataset=dataset, task=load_info(dataset)['primary_task'], scenario=scenario,
                   classifier=classifier, method=method, feature_set=feature_set, condition=c,
                   split_id=sid, seed=split.seed, fold=split.fold, n_train=len(split.train), n_test=len(ids),
                   fit_key=str(where.relative_to(root(dataset))), best_params=json.dumps(params))
        row.update({k: float(v) for k, v in split_metrics(y[ids], pred, sorted(all_labels)).items()})
        if scenario == 'shift':
            clean_ids, raw_clean_pred = predicted['clean']
            if classifier == 'lightgbm':
                raw_clean_pred = encoder.inverse_transform(raw_clean_pred.astype(int))
            lookup = dict(zip(clean_ids, raw_clean_pred))
            baseline_pred = np.array([lookup[int(i)] for i in ids], dtype=str)
        else:
            baseline_pred = clean_pred
        baseline = split_metrics(y[ids], baseline_pred, sorted(all_labels))
        row.update({'clean_'+k: float(v) for k, v in baseline.items()})
        metrics.append(row)
    if not (where/'predictions.npz').exists():
        save_npz(where/'predictions.npz', **arrays)
    save_json(where/'complete.json', dict(rows=metrics, fit_seconds=fit_seconds, fit_cpu_seconds=fit_cpu,
              predict_seconds=time.perf_counter()-wall, predict_cpu_seconds=time.process_time()-cpu,
              warnings={str(w.message): sum(str(z.message)==str(w.message) for z in notices) for w in notices},
              pimf=verify()))
    return str(where), False


def run(dataset, workers=6):
    if not 1 <= workers <= 6:
        raise ValueError('workers must be 1..6')
    verify()
    out = root(dataset)
    materialize(dataset)
    source = [Path(__file__), Path(__file__).parents[1]/'baselines/run_classical.py',
              Path(__file__).parents[1]/'ml/run_trees.py', Path(__file__).parents[1]/'ml/common.py']
    manifest = dict(feature_manifest_sha256=digest(out/MANIFEST),
                    source_hashes={str(p.relative_to(Path(__file__).parents[1])): digest(p) for p in source},
                    matched_conditions=[c for c in CONDITIONS[1:] if c.endswith('_0.04')],
                    matched_lightgbm=False, seed=20261021, pimf=verify())
    path = out/'classification_manifest.json'
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise RuntimeError('Classification manifest mismatch; refusing to replace results')
    else:
        save_json(path, manifest)
    single_thread_env()
    limit_threads()
    started = time.perf_counter()
    splits = load_splits(dataset)
    with ProcessPoolExecutor(max_workers=workers, initializer=initialize, initargs=(dataset,)) as pool:
        for scenario in ('shift', 'matched'):
            jobs = []
            for clf in ((CLASSICAL[dataset], 'lightgbm') if scenario == 'shift' else (CLASSICAL[dataset],)):
                for method in METHODS:
                    for fs in FEATURE_SETS:
                        for split in splits:
                            for c in (('clean',) if scenario == 'shift' else manifest['matched_conditions']):
                                if not (checkpoint(dataset, scenario, clf, method, fs, split.split_id, c)/'complete.json').exists():
                                    jobs.append((scenario, clf, method, fs, split.split_id, c))
            futures = [pool.submit(job, args) for args in jobs]
            for done, future in enumerate(as_completed(futures), 1):
                path, reused = future.result()
                if done % 10 == 0 or done == len(futures):
                    print(f'{dataset}/{scenario}: {done}/{len(futures)} fits complete', flush=True)
                if done % 50 == 0:
                    check_disk()
    path = out/'runs/classification.json'
    if not path.exists():
        save_json(path, dict(wall_seconds=time.perf_counter()-started, workers=workers, pimf=verify()))
