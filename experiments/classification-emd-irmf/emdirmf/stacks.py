"""Read bounded batches of normalized component stacks without changing caches."""

import json

import numpy as np
from scipy.signal import resample_poly

from .decompose import canonical_tag, n_slots, normalize_frame, scale_order, slot_labels
from .io import load_frames
from .paths import decomp_dir


class StackReader:
    """Channel-major, then fine-to-coarse slots; raw uses the shared normalization.

    Only ``take`` materializes samples. Downsampling uses an anti-aliasing FIR
    on *every* channel of every method, after normalization/decomposition.
    """

    def __init__(self, dataset, method, downsample=1):
        self.method = canonical_tag(method)
        self.downsample = int(downsample)
        if self.downsample < 1:
            raise ValueError("downsample must be positive")
        self.frames = load_frames(dataset)
        n, channels, length = self.frames.shape
        self.order = scale_order(self.method, length)
        self.labels = slot_labels(self.method, length)
        self.source = None
        if self.method != "raw":
            directory = decomp_dir(dataset)
            with open(directory / f"{self.method}.json") as handle:
                summary = json.load(handle)
            if summary.get("partial", False):
                raise ValueError(f"{self.method}: partial decomposition cache; wait for completion")
            if summary.get("n_failed", 0):
                raise ValueError(f"{self.method}: failed decompositions; repair before paired ML")
            self.source = np.load(directory / f"{self.method}.npy", mmap_mode="r")
            expected = (n, channels, n_slots(self.method, length), length)
            if self.source.shape != expected or tuple(summary["shape"]) != expected:
                raise ValueError(f"{self.method}: stale or incomplete stack shape, expected {expected}")
            if self.source.dtype != np.float32:
                raise ValueError("decomposition must be float32")
        self.shape = (n, channels * len(self.order), (length + self.downsample - 1) // self.downsample)
        if self.shape[-1] < 9:
            raise ValueError("at least 9 timepoints are required")

    def take(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if self.source is None:
            values, _ = normalize_frame(self.frames[indices])
        else:
            values = np.asarray(self.source[indices])[:, :, self.order, :]
            values = values.reshape(len(indices), self.shape[1], self.frames.shape[-1])
        values = np.asarray(values, dtype=np.float32)
        if not np.isfinite(values).all():
            raise ValueError(f"{self.method}: non-finite stack samples")
        if self.downsample > 1:
            values = resample_poly(values, up=1, down=self.downsample, axis=-1)
        return np.ascontiguousarray(values, dtype=np.float32)

    def log_scales(self, indices):
        _, values = normalize_frame(self.frames[np.asarray(indices, dtype=np.int64)])
        return values.astype(np.float32)

    def batches(self, indices, batch_size=64, *, shuffle=False, seed=0):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        indices = np.asarray(indices, dtype=np.int64).copy()
        if shuffle:
            np.random.default_rng(seed).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch = indices[start : start + batch_size]
            yield batch, self.take(batch)

    def description(self):
        return {"shape": self.shape, "storage_slots_in_scale_order": self.order,
                "slot_labels_in_scale_order": [self.labels[k] for k in self.order],
                "channel_order": "input channel, then fine-to-coarse component",
                "downsample": self.downsample, "resampling": "scipy.signal.resample_poly"}
