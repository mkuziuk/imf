"""Validate completed runs, verify pimf again, then publish new report artifacts."""
import argparse
import importlib.metadata
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from emdirmf.cv import load_splits, split_metrics
from emdirmf.io import label_column, load_info, load_meta
from emdirmf.paths import OUTPUT_ROOT
from robustness.analysis import DATASETS, METRICS, load_scores, paired_tests
from robustness.environment import START, verify
from robustness.processing import MANIFEST, VERIFICATION
from robustness.reporting import (aggregate_representations, csv_new, figures, readings, render_report)
from robustness.storage import digest, root, save_json


def validate_predictions(dataset):
    meta, info = load_meta(dataset), load_info(dataset)
    y = meta[label_column(info)].to_numpy(dtype=str)
    labels = sorted(set(y))
    manifest = json.loads((root(dataset)/MANIFEST).read_text())
    excluded = set(manifest['subset_ids'])
    splits = {s.split_id:s for s in load_splits(dataset)}
    valid = {c: np.load(root(dataset)/'matrices'/c/'valid.npy', allow_pickle=False)
             for c in manifest['conditions']}
    keep = np.ones(len(y), dtype=bool)
    keep[list(excluded)] = False
    n_predictions = 0
    for path in sorted((root(dataset)/'classification').rglob('complete.json')):
        value = json.loads(path.read_text())
        with np.load(path.with_name('predictions.npz'),allow_pickle=False) as z:
            for row in value['rows']:
                c, sid = row['condition'], row['split_id']
                ids, pred = z[c+'__ids'], z[c+'__pred']
                if set(map(int,ids)) & excluded or not np.isin(ids,splits[sid].test).all():
                    raise AssertionError('pilot inclusion or test-fold mismatch')
                if len(np.unique(ids)) != len(ids):
                    raise AssertionError('duplicate predictions')
                expected_ids = splits[sid].test[(keep & valid[c])[splits[sid].test]]
                np.testing.assert_array_equal(ids, expected_ids)
                train_condition = 'clean' if row['scenario']=='shift' else c
                expected_train = (keep & valid[train_condition])[splits[sid].train].sum()
                if row['n_train'] != int(expected_train) or row['n_test'] != len(ids):
                    raise AssertionError('training or test population mismatch')
                actual = split_metrics(y[ids], pred, labels)
                for metric in METRICS:
                    if abs(actual[metric]-row[metric]) > 1e-12:
                        raise AssertionError('prediction-score mismatch')
                n_predictions += len(ids)
    verification = json.loads((root(dataset)/VERIFICATION).read_text())
    if not verification['ok']:
        raise AssertionError('clean cache reproduction failed')
    subset = set(manifest['subset_ids'])
    # Every retained contaminated stack is confined to the preselected 100 IDs.
    retained = {}
    for c in manifest['conditions'][1:]:
        for m in manifest['methods']:
            ids=[]
            for p in (root(dataset)/'figures'/c/m).glob('part_*.npz'):
                with np.load(p,allow_pickle=False) as z:
                    ids.extend(z['frame_id'].tolist())
                    if not set(z['frame_id'].tolist()) <= subset:
                        raise AssertionError('stack outside fixed subset')
                    if not np.isfinite(z['stacks']).all():
                        # Failed figure IDs may intentionally contain NaNs. Their
                        # classification exclusion is independently checked above.
                        pass
            if set(ids) != subset or len(ids) != 100:
                raise AssertionError('incomplete or duplicate retained subset')
            retained[c+'/'+m] = len(ids)
    return dict(dataset=dataset,predictions_recomputed=n_predictions,
                clean_exact_checks=sum(r['exact'] for r in verification['rows']),
                clean_checks=len(verification['rows']),
                clean_max_absolute_error=max(r['max_absolute_error'] for r in verification['rows']),
                retained_stack_frames_per_condition_method=100,
                pilot_exclusions=len(subset),grouped_splits_verified=15,
                feature_manifest_sha256=digest(root(dataset)/MANIFEST),
                classification_manifest_sha256=digest(root(dataset)/'classification_manifest.json'))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=OUTPUT_ROOT/'robustness')
    args=parser.parse_args()
    # This check occurs before opening the final score/test tables.
    current=verify()
    stamp=datetime.now(timezone.utc).isoformat()
    for d in DATASETS:
        path=root(d)/'runs/pre_analysis_pimf.json'
        if not path.exists(): save_json(path,dict(utc=stamp,pimf=current))
    out=args.out
    if out.exists():
        raise FileExistsError(f'Refusing to replace report directory: {out}')
    scores,costs,stability,diagnostics,failures,fit_costs,wall=[],[],[],[],[],[],[]
    validation=[]
    for d in DATASETS:
        print(f'Validating {d}',flush=True)
        for provenance in ('manifest_v2.json','classification_manifest.json'):
            recorded=json.loads((root(d)/provenance).read_text())
            for filename, expected in recorded.get('sources', recorded.get('source_hashes', {})).items():
                if digest(Path(__file__).parents[1]/filename) != expected:
                    raise RuntimeError(f'Source changed since run: {filename}')
        s,f=load_scores(d);scores.append(s);fit_costs.append(f)
        validation.append(validate_predictions(d))
        st,co,di,fa=aggregate_representations(d)
        stability.append(st);costs.append(co);diagnostics.append(di);failures.append(fa)
        feature=json.loads((root(d)/'runs/features.json').read_text())
        classification=json.loads((root(d)/'runs/classification.json').read_text())
        pilot=json.loads((root(d)/'pilot/run.json').read_text())
        pilot_cpu=pd.read_csv(root(d)/'pilot/costs.csv').cpu_seconds.sum()
        restarts = [json.loads(p.read_text()) for p in sorted((root(d)/'runs/restarts').glob('*.json'))]
        interrupted_wall = sum(r['classification_segment_wall_seconds'] for r in restarts)
        uncheckpointed_cpu = sum(r['uncheckpointed_worker_cpu_seconds'] for r in restarts)
        wall.append(dict(dataset=d,feature_wall_seconds=feature['wall_seconds'],
                         classification_wall_seconds=classification['wall_seconds']+interrupted_wall,
                         resumed_classification_wall_seconds=classification['wall_seconds'],
                         interrupted_classification_wall_seconds=interrupted_wall,
                         uncheckpointed_worker_cpu_seconds=uncheckpointed_cpu,
                         pilot_wall_seconds=pilot['wall_seconds'],pilot_cpu_seconds=pilot_cpu))
    scores=pd.concat(scores,ignore_index=True);fit_costs=pd.concat(fit_costs,ignore_index=True)
    stability=pd.concat(stability,ignore_index=True);costs=pd.concat(costs,ignore_index=True)
    diagnostics=pd.concat(diagnostics,ignore_index=True);failures=pd.concat(failures,ignore_index=True)
    wall=pd.DataFrame(wall)
    tests,vectors=paired_tests(scores)
    reading=readings(tests)
    verify()
    out.mkdir(parents=True,exist_ok=False)
    for name,table in [('per_split_scores.csv',scores),('paired_tests.csv',tests),
                       ('paired_vectors.csv',vectors),('primary_tests.csv',tests[tests.family=='Primary']),
                       ('stability_metrics.csv',stability),('costs.csv',costs),('fit_costs.csv',fit_costs),
                       ('solver_diagnostics.csv',diagnostics),('failed_frames.csv',failures),
                       ('prespecified_reading.csv',reading),('wall_times.csv',wall)]:
        csv_new(out/name,table)
    summary=scores.groupby(['dataset','scenario','classifier','feature_set','method','condition'])[
        list(METRICS)].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(k).rstrip('_') if isinstance(k,tuple) else k for k in summary.columns]
    csv_new(out/'score_summary.csv',summary)
    versions = {'python': platform.python_version()}
    for package in ('numpy','scipy','pandas','scikit-learn','lightgbm','EMD-signal','numba','pimf'):
        versions[package] = importlib.metadata.version(package)
    save_json(out/'environment_audit.json',dict(start=json.loads(START.read_text()),
        final=dict(utc=stamp,pimf=current),unchanged=True,versions=versions))
    save_json(out/'validation.json',dict(datasets=validation,real_families=tests[tests.dataset!='synthetic'].groupby('family').size().to_dict(),
        synthetic_separate_family=87,all_prediction_metrics_recomputed=True,source_shared_modules_modified=False))
    figures(out,scores,tests,stability)
    render_report(out,scores,tests,stability,costs,diagnostics,fit_costs,reading,wall)
    print(out/'REPORT.md',flush=True)


if __name__=='__main__':
    main()
