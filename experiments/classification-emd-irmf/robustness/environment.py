"""Record and enforce the installed, unmodified pimf dependency."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pimf

START = Path(__file__).parent / 'provenance/pimf_start.json'


def snapshot():
    return {'version': pimf.__version__, 'files': {
        f.name: hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(Path(pimf.__file__).parent.glob('*.py'))}}


def verify():
    current = snapshot()
    expected = json.loads(START.read_text())['pimf']
    if current != expected or current['version'] != '0.4.0':
        raise RuntimeError('pimf version or source hashes changed; STOP')
    return current


def record_start():
    with START.open('x') as f:
        json.dump({'utc': datetime.now(timezone.utc).isoformat(), 'pimf': snapshot()}, f, indent=2)
