"""Finish validation and publication after the detached real-data batch succeeds."""
import json
import subprocess
import sys
import time
from datetime import datetime, timezone

import psutil

from emdirmf.paths import OUTPUT_ROOT
from robustness.environment import verify
from robustness.storage import root, save_json


def wait_for_real_batch():
    runs = root('dronerf')/'runs'
    while not (runs/'real_batch.json').exists():
        verify()
        records = sorted((runs/'detached').glob('*.json'))
        if records:
            record = json.loads(records[-1].read_text())
            if not psutil.pid_exists(record['pid']):
                raise RuntimeError(f'Detached real-data batch exited before success: {record}')
        print(f'{datetime.now(timezone.utc).isoformat()} waiting for real batch', flush=True)
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
