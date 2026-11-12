#!/usr/bin/env python3
"""Raw log-STFT reference using the same grouped CNN training/epoch selection."""

import sys
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from scipy.signal import stft
import torch
from torch import nn

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

from emdirmf.stacks import StackReader
from emdirmf.paths import results_dir
from ml import run_cnn
import run_phase3


class SpectrogramReader(StackReader):
    def __init__(self, dataset, method, downsample=1):
        if method != "raw" or downsample != 1:
            raise ValueError("this reference uses full-resolution raw frames only")
        super().__init__(dataset, method, downsample)
        directory = results_dir(dataset) / "representations"
        directory.mkdir(parents=True, exist_ok=True)
        source = Path(self.frames.filename).stat()
        signature = {"bytes": source.st_size, "mtime_ns": source.st_mtime_ns,
                     "normalization": hashlib.sha256((STUDY / "emdirmf/decompose.py").read_bytes()).hexdigest(),
                     "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:16]
        cache = directory / f"raw_log_stft_256_64_{key}.npy"
        expected = (len(self.frames), self.frames.shape[1], 129, 33)
        if not cache.exists():
            temporary = cache.with_suffix(f".{os.getpid()}.tmp.npy")
            values = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=expected)
            for start in range(0, len(self.frames), 64):
                indices = np.arange(start, min(start + 64, len(self.frames)))
                normalized = super().take(indices)
                _, _, spectrum = stft(normalized, fs=40_000_000, window="hann", nperseg=256,
                                      noverlap=192, nfft=256, boundary="zeros", padded=True,
                                      axis=-1, scaling="spectrum")
                values[indices] = np.log1p(np.abs(spectrum) ** 2)
                assert np.isfinite(values[indices]).all()
            values.flush()
            del values
            temporary.replace(cache)
        self.values = np.load(cache, mmap_mode="r")
        assert self.values.shape == expected and self.values.dtype == np.float32
        self.cache = str(cache)
        self.shape = self.values.shape

    def take(self, indices):
        return np.ascontiguousarray(self.values[indices])

    def description(self):
        return {"shape": self.shape, "representation": "log1p squared STFT magnitude",
                "normalization": "shared median/MAD per raw frame", "n_fft": 256,
                "hop": 64, "window": "hann", "boundary": "zeros", "padding": True,
                "cross_frame_preprocessing": False, "fs": 40_000_000, "cache": self.cache}


class SmallSpectrogramCNN(nn.Module):
    def __init__(self, channels, classes, width=8, scale_features=0):
        super().__init__()
        blocks = []
        for outgoing in (width, 2 * width, 4 * width):
            blocks.extend([nn.Conv2d(channels, outgoing, 3, padding=1, bias=False),
                           nn.GroupNorm(4, outgoing), nn.ReLU(), nn.MaxPool2d(2)])
            channels = outgoing
        self.features = nn.Sequential(*blocks)
        self.head = nn.Linear(8 * width + scale_features, classes)

    def forward(self, x, scales):
        x = self.features(x)
        return self.head(torch.cat([x.mean(dim=(-2, -1)), x.amax(dim=(-2, -1)), scales], dim=1))


def main():
    run_cnn.StackReader = SpectrogramReader
    run_cnn.CompactResNet = SmallSpectrogramCNN
    original_finish = run_cnn.finish

    def finish(context, out, rows, predictions, notes=(), extra=None):
        for row in rows:
            row["classifier"] = "log_stft_cnn2d"
        for prediction in predictions:
            prediction["classifier"] = "log_stft_cnn2d"
        extra = dict(extra or {})
        width = context["params"]["width"]
        extra["architecture"] = {"name": "SmallSpectrogramCNN", "widths": [width, 2 * width, 4 * width],
                                 "kernel_size": 3, "normalization": "GroupNorm(4)",
                                 "pooling": "three MaxPool2d(2), then global mean/max"}
        return original_finish(context, out, rows, predictions, notes=[
            "Raw log-STFT reference, deterministic per-frame transform with no train/test-fitted moments.",
            "Three Conv2d/GroupNorm/ReLU/MaxPool2d blocks, global mean/max pooling, optional train-standardized log_scale.",
            "Same grouped inner validation, balanced cross entropy, AdamW, epoch selection and fresh outer refit as 1-D CNN.",
            "Different architecture and representation: an exploratory time-frequency reference, not a decomposition pair.",
        ], extra=extra)

    run_cnn.finish = finish
    sys.argv.insert(1, "cnn")
    return run_phase3.main()


if __name__ == "__main__":
    sys.exit(main())
