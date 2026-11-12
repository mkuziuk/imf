"""Validate the completed run, then write the report artifacts.

    $PY -m robustness_v3.analyse [--out DIR]
"""
import argparse
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf.cv import load_splits, split_metrics
from emdirmf.io import frames_fingerprint, label_column, load_info, load_meta
from emdirmf.paths import OUTPUT_ROOT, data_root
from robustness.environment import START, verify
from robustness_v3.analysis import DATASETS, FAMILY_SIZES, METRICS, descriptive_tables, load_scores, paired_tests, selections
from robustness_v3.classification import CLASSIFICATION_MANIFEST, checkpoint, selection_path
from robustness_v3.processing import MANIFEST, VERIFICATION, condition_matrices
from robustness_v3.estimators import DECISION, PILOT_SHA256, calibrate_h
from robustness_v3.reporting import aggregate_representations, figures, readings, render_report
from robustness_v3.run import runs_dir
from robustness_v3.storage import digest, root, save_json

STUDY = Path(__file__).parents[1]


def validate_predictions(dataset):
    meta, info = load_meta(dataset), load_info(dataset)
    y = meta[label_column(info)].to_numpy(dtype=str)
    labels = sorted(set(y))
    manifest = json.loads((root(dataset)/MANIFEST).read_text())
    excluded = set(manifest['subset_ids'])
    splits = {s.split_id: s for s in load_splits(dataset)}
    valid = {c: np.load(root(dataset)/'matrices'/c/'valid.npy', allow_pickle=False) for c in manifest['conditions']}
    keep = np.ones(len(y), dtype=bool)
    keep[list(excluded)] = False
    n_predictions = 0
    for path in sorted((root(dataset)/'classification').rglob('complete.json')):
        value = json.loads(path.read_text())
        with np.load(path.with_name('predictions.npz'), allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        for row in value['rows']:
            c, sid = row['condition'], row['split_id']
            ids, pred = arrays[c+'__ids'], arrays[c+'__pred']
            if row['scenario'] == 'shift':
                clean_ids, clean_pred = arrays['clean__ids'], arrays['clean__pred']
            else:
                shift = checkpoint(dataset, 'shift', row['classifier'], row['method'], row['feature_set'], sid)
                with np.load(shift/'predictions.npz', allow_pickle=False) as base:
                    clean_ids, clean_pred = base['clean__ids'], base['clean__pred']
            baseline = dict(zip(clean_ids.tolist(), clean_pred.tolist()))
            check_row(row, y, labels, ids, pred, baseline)
            if set(map(int, ids)) & excluded or not np.isin(ids, splits[sid].test).all():
                raise AssertionError('pilot inclusion or test-fold mismatch')
            np.testing.assert_array_equal(ids, splits[sid].test[(keep & valid[c])[splits[sid].test]])
            train_condition = 'clean' if row['scenario'] == 'shift' else c
            if row['n_train'] != int((keep & valid[train_condition])[splits[sid].train].sum()):
                raise AssertionError('training population mismatch')
            n_predictions += len(ids)
    verification = json.loads((root(dataset)/VERIFICATION).read_text())
    if not verification['ok']:
        raise AssertionError('clean cache reproduction failed')
    for c in manifest['conditions']:
        for m in manifest['methods']:
            ids = []
            for p in (root(dataset)/'figures'/c/m).glob('part_*.npz'):
                with np.load(p, allow_pickle=False) as z:
                    ids.extend(z['frame_id'].tolist())
            if set(ids) != excluded or len(ids) != len(excluded):
                raise AssertionError('incomplete or duplicate retained subset')
    return dict(dataset=dataset, predictions_recomputed=n_predictions,
                clean_exact_checks=sum(r['exact'] for r in verification['rows']),
                clean_checks=len(verification['rows']),
                clean_max_absolute_error=max(r['max_absolute_error'] for r in verification['rows']),
                pilot_exclusions=len(excluded), grouped_splits_verified=len(splits),
                feature_manifest_sha256=digest(root(dataset)/MANIFEST),
                classification_manifest_sha256=digest(root(dataset)/CLASSIFICATION_MANIFEST))


def check_row(row, y, labels, ids, pred, baseline):
    """Recompute a row's metrics and its clean-baseline metrics from saved predictions."""
    if ids.dtype.kind not in 'iu' or len(np.unique(ids)) != len(ids) or len(pred) != len(ids):
        raise AssertionError('invalid prediction IDs or length')
    if not set(pred) <= set(labels):
        raise AssertionError('invalid prediction labels')
    if row['n_test'] != len(ids):
        raise AssertionError('test count mismatch')
    actual = split_metrics(y[ids], pred, labels)
    clean = split_metrics(y[ids], np.array([baseline[int(i)] for i in ids], dtype=str), labels)
    for metric in METRICS:
        if abs(actual[metric]-row[metric]) > 1e-12:
            raise AssertionError('prediction-score mismatch')
        if abs(clean[metric]-row['clean_'+metric]) > 1e-12:
            raise AssertionError('clean-baseline mismatch')


def check_provenance(dataset):
    """Verify immutable inputs, numeric matrices, fixed H, selected K and manifest links."""
    base, out = data_root()/dataset, root(dataset)
    manifest = json.loads((out/MANIFEST).read_text())
    if manifest['version'] != 2 or manifest['verification_file'] != VERIFICATION:
        raise RuntimeError('wrong amended manifest version')
    for name, expected in manifest['inputs'].items():
        if digest(base/name) != expected:
            raise RuntimeError(f'{dataset}: changed input: {name}')
    frames = np.load(base/'prepared/frames.npy', mmap_mode='r', allow_pickle=False)
    if frames_fingerprint(frames) != manifest['frames_fingerprint']:
        raise RuntimeError('prepared frames changed')
    if calibrate_h(frames, manifest['subset_ids']) != manifest['fixed_h']:
        raise RuntimeError('fixed H calibration mismatch')
    if manifest['h_decision'] != json.loads(DECISION.read_text()) or manifest['pilot_report_sha256'] != PILOT_SHA256:
        raise RuntimeError('amended pilot decision mismatch')
    feature_sha = digest(out/MANIFEST)
    for path in (out/CLASSIFICATION_MANIFEST, out/'matrices/complete.json'):
        if json.loads(path.read_text())['feature_manifest_sha256'] != feature_sha:
            raise RuntimeError('feature manifest link mismatch')
    complete = json.loads((out/'matrices/complete.json').read_text())
    for name, expected in complete['hashes'].items():
        if digest(out/name) != expected:
            raise RuntimeError(f'matrix changed: {name}')
    for condition in manifest['conditions']:
        values, valid = condition_matrices(dataset, condition)
        np.testing.assert_array_equal(valid, np.load(out/'matrices'/condition/'valid.npy', allow_pickle=False))
        for variant, value in values.items():
            np.testing.assert_array_equal(value, np.load(out/'matrices'/condition/f'{variant}.npy', allow_pickle=False))
    scores, _ = load_scores(dataset)
    summary = json.loads((out/'k_star.json').read_text())['classifiers']
    for row in selections(scores).itertuples():
        saved = json.loads(selection_path(dataset, row.method, row.classifier).read_text())
        if (saved['variant'], saved['k'], saved['mean_clean_ba']) != (row.variant, row.k, row.mean_clean_ba):
            raise RuntimeError('K* does not match all clean split scores')
        if summary[row.classifier][row.method] != saved:
            raise RuntimeError('K* summary mismatch')
        for name, expected in saved['dependencies'].items():
            if digest(out/name/'complete.json') != expected:
                raise RuntimeError('K* dependency changed')
    return dict(inputs_verified=len(manifest['inputs']), fixed_h_recalibrated=True,
                matrices_recomputed=True, k_selection_recomputed=True)


def run_record(datasets):
    """Completed run records that together cover the requested datasets."""
    records = [json.loads(p.read_text()) for p in sorted(runs_dir().glob('run_*.json'))]
    records = [r for r in records if set(r['datasets']) & set(datasets)]
    covered = set().union(*(set(r['datasets']) for r in records))
    if not set(datasets) <= covered:
        raise RuntimeError(f'no completed run record for {sorted(set(datasets) - covered)}')
    return dict(workers=max(r['workers'] for r in records), records=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=OUTPUT_ROOT/'robustness_v3')
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=list(DATASETS),
                        help='a subset gives a preview, which must go to a non-default --out')
    args = parser.parse_args()
    partial = set(args.datasets) != set(DATASETS)
    if partial and args.out == OUTPUT_ROOT/'robustness_v3':
        raise SystemExit('a dataset subset is a preview; pass a scratch --out')
    current = verify()
    if args.out.exists():
        raise FileExistsError(f'refusing to replace {args.out}')
    scores, fit_costs, stability, costs, diagnostics, failures, validation = [], [], [], [], [], [], []
    for d in args.datasets:
        print(f'validating {d}', flush=True)
        for provenance, field in ((MANIFEST, 'sources'), (CLASSIFICATION_MANIFEST, 'source_hashes')):
            for filename, expected in json.loads((root(d)/provenance).read_text())[field].items():
                if digest(STUDY/filename) != expected:
                    raise RuntimeError(f'source changed since the run: {filename}')
        provenance = check_provenance(d)
        s, f = load_scores(d)
        scores.append(s)
        fit_costs.append(f)
        validation.append(validate_predictions(d) | provenance)
        st, co, di, fa = aggregate_representations(d)
        stability.append(st)
        costs.append(co)
        diagnostics.append(di)
        failures.append(fa)
    scores, fit_costs = pd.concat(scores, ignore_index=True), pd.concat(fit_costs, ignore_index=True)
    stability, costs = pd.concat(stability, ignore_index=True), pd.concat(costs, ignore_index=True)
    diagnostics, failures = pd.concat(diagnostics, ignore_index=True), pd.concat(failures, ignore_index=True)
    tests, vectors = paired_tests(scores, args.datasets)
    reading = readings(tests)
    run = None if partial else run_record(args.datasets)
    verify()
    out = args.out
    out.mkdir(parents=True)
    for name, table in [('per_split_scores.csv', scores), ('paired_tests.csv', tests), ('paired_vectors.csv', vectors),
                        ('primary_tests.csv', tests[tests.family == 'Primary']), ('stability_metrics.csv', stability),
                        ('costs.csv', costs), ('fit_costs.csv', fit_costs), ('solver_diagnostics.csv', diagnostics),
                        ('failed_frames.csv', failures), ('prespecified_reading.csv', reading)]:
        table.to_csv(out/name, index=False, mode='x')
    summary = scores.groupby(['dataset', 'scenario', 'classifier', 'feature_set', 'method', 'condition'])[
        list(METRICS)].agg(['mean', 'std']).reset_index()
    summary.columns = ['_'.join(k).rstrip('_') if isinstance(k, tuple) else k for k in summary.columns]
    summary.to_csv(out/'score_summary.csv', index=False, mode='x')
    versions = {'python': platform.python_version()}
    for package in ('numpy', 'scipy', 'pandas', 'scikit-learn', 'lightgbm', 'EMD-signal', 'numba', 'pimf'):
        versions[package] = importlib.metadata.version(package)
    stamp = datetime.now(timezone.utc).isoformat()
    save_json(out/'environment_audit.json', dict(start=json.loads(START.read_text()), final=dict(utc=stamp, pimf=current),
                                                 unchanged=True, versions=versions,
                                                 runs=run['records'] if run else None))
    save_json(out/'validation.json', dict(datasets=validation, families=tests.groupby('holm_family').size().to_dict(),
                                          expected_families=FAMILY_SIZES, all_prediction_metrics_recomputed=True,
                                          shift_classical_selection=fit_costs.groupby(['dataset', 'selection']).size()
                                          .rename('fits').reset_index().to_dict('records')))
    every, best, selected, chosen = descriptive_tables(scores)
    for name, table in [('ba_every_k_condition.csv', every), ('best_k_per_condition.csv', best),
                        ('best_vs_best_ba.csv', selected), ('k_star.csv', chosen)]:
        table.to_csv(out/name, index=False, mode='x')
    figures(out, scores, tests, stability)
    render_report(out, scores, tests, stability, costs, diagnostics, fit_costs, reading, run)
    print(out/'REPORT.md', flush=True)


if __name__ == '__main__':
    main()
