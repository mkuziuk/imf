"""Finish validation and publication after the detached real-data batch succeeds."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

from emdirmf.paths import OUTPUT_ROOT
from robustness.contamination import DATASETS
from robustness.environment import verify
from robustness.storage import root, save_json


def wait_for_real_batch():
    record = json.loads((root('dronerf')/'runs/parallel_schedule.json').read_text())
    pids = {'dronerf': record['preserved_dronerf_runner_pid']}
    pids.update({dataset: launch['pid']
                 for dataset, launch in record['detached_launches'].items()})
    for dataset in DATASETS[1:]:
        if dataset not in pids:
            raise RuntimeError(f'Missing detached run for {dataset}')
    while True:
        verify()
        pending = [dataset for dataset in DATASETS[1:]
                   if not (root(dataset)/'runs/classification.json').exists()]
        if not pending:
            break
        for dataset in pending:
            if not psutil.pid_exists(pids[dataset]):
                raise RuntimeError(f'{dataset} runner {pids[dataset]} exited before success')
        print(f'{datetime.now(timezone.utc).isoformat()} waiting for {pending}', flush=True)
        time.sleep(30)
    verify()


def main():
    wait_for_real_batch()
    print('All real datasets complete; running required tests', flush=True)
    subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests', 'robustness/tests'], check=True)
    verify()
    qa = root('dronerf')/'runs'/('report_qa_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    print(f'Validating full report in {qa}', flush=True)
    subprocess.run([sys.executable, '-m', 'robustness.analyse', '--out', str(qa)], check=True)
    final = OUTPUT_ROOT/'robustness'
    if final.exists():
        raise FileExistsError(f'Final report already exists: {final}')
    print(f'Publishing final report in {final}', flush=True)
    subprocess.run([sys.executable, '-m', 'robustness.analyse'], check=True)
    save_json(root('dronerf')/'runs/report_ready.json', dict(
        utc=datetime.now(timezone.utc).isoformat(), report=str(final/'REPORT.md'),
        qa=str(qa), pimf=verify()))
    print(f'Report ready: {final/"REPORT.md"}', flush=True)


if __name__=='__main__':
    main()
