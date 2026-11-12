import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

STUDY_DIR = Path(__file__).resolve().parents[1]
if str(STUDY_DIR) not in sys.path:
    sys.path.insert(0, str(STUDY_DIR))


def multiscale_signal(length, seed=0, impulses=True):
    """Trend + two tones + Gaussian noise (+ two-sided exponential impulses)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length) / length
    y = 0.8 * np.sin(2 * np.pi * 2 * t) + 0.6 * np.sin(2 * np.pi * 40 * t) + 0.3 * (t - 0.5)
    y += rng.normal(scale=0.4, size=length)
    if impulses:
        y += (rng.random(length) < 0.05) * rng.exponential(2.0, length) * rng.choice([-1, 1], length)
    return y


def write_prepared(root, name, frames, meta, tasks=None):
    out = Path(root) / name / "prepared"
    out.mkdir(parents=True)
    np.save(out / "frames.npy", frames.astype(np.float32))
    meta.to_parquet(out / "meta.parquet", index=False)
    tasks = {"cls": "y_cls"} if tasks is None else tasks
    info = {
        "name": name,
        "fs": 1000.0,
        "frame_length": frames.shape[2],
        "channels": [f"ch{c}" for c in range(frames.shape[1])],
        "tasks": tasks,
        "primary_task": next(iter(tasks)),
        "n_frames": len(meta),
        "n_groups": int(meta["group"].nunique()),
        "class_counts": {task: meta[column].value_counts().to_dict() for task, column in tasks.items()},
        "preparation": {"seed": 0},
        "source_url": "test",
        "license": "test",
        "citation": "test",
        "notes": "pytest fixture",
    }
    with open(out / "dataset.json", "w") as handle:
        json.dump(info, handle)


@pytest.fixture
def tiny_dataset(tmp_path, monkeypatch):
    """40 frames, 2 channels, L=256, 10 groups of 4 frames, 2 balanced classes."""
    monkeypatch.setenv("IMF_CLS_DATA", str(tmp_path))
    rng = np.random.default_rng(1)
    n, channels, length = 40, 2, 256
    frames = np.empty((n, channels, length))
    rows = []
    t = np.arange(length)
    for index in range(n):
        group = index // 4
        label = "a" if group % 2 == 0 else "b"
        freq = 0.05 if label == "a" else 0.11
        for channel in range(channels):
            frames[index, channel] = np.sin(2 * np.pi * freq * t + rng.uniform(0, 6)) + rng.normal(0, 0.5, length)
        rows.append({"frame_id": index, "group": f"g{group}", "y_cls": label,
                     "source_file": f"rec{group}", "start_sample": (index % 4) * length})
    meta = pd.DataFrame(rows)
    write_prepared(tmp_path, "tiny", frames, meta)
    return "tiny"
