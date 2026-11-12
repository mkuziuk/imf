#!/usr/bin/env python3
"""Raw log-STFT reference using the shared grouped CNN selection/refit loop."""

import argparse
import json
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(STUDY))

import numpy as np
import pandas as pd
from torch import nn
import torch

from emdirmf.stacks import StackReader
from emdirmf.cv import load_splits, split_metrics
from emdirmf.io import load_meta
from emdirmf.paths import output_dir, results_dir
from ml import run_cnn as cnn
import run_phase3


def audit_reference(options):
    meta = load_meta("birdclef2023")
    labels = np.unique(meta[f"y_{options.task}"].astype(str))
    columns = [f"prob_{label}" for label in labels]
    predictions = pd.read_parquet(results_dir("birdclef2023") /
                                  f"{options.run_name}_{options.task}_predictions.parquet")
    assert not predictions.duplicated(["feature_set", "split_id", "frame_id"]).any()
    folds = pd.read_csv(Path(options.out) / f"results_folds_{options.task}.csv")
    splits = {s.split_id: s for s in load_splits("birdclef2023", options.task)}
    rows, recording_predictions = [], []
    for (side, split_id), cell in predictions.groupby(["feature_set", "split_id"]):
        split = splits[split_id]
        cell = cell.sort_values("frame_id").copy()
        np.testing.assert_array_equal(cell.frame_id, np.sort(split.test))
        np.testing.assert_array_equal(cell.y_true, meta.iloc[cell.frame_id][f"y_{options.task}"].astype(str))
        probabilities = cell[columns].to_numpy()
        assert np.isfinite(probabilities).all()
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-5)
        np.testing.assert_array_equal(labels[probabilities.argmax(axis=1)], cell.y_pred)
        assert not set(meta.iloc[split.train].group) & set(meta.iloc[split.test].group)
        expected = split_metrics(cell.y_true.to_numpy(), cell.y_pred.to_numpy(), labels)
        saved = folds[(folds.feature_set == side) & (folds.split_id == split_id)].iloc[0]
        for metric in ("balanced_accuracy", "macro_f1", "accuracy"):
            np.testing.assert_allclose(expected[metric], saved[metric], atol=1e-12)
        cell["recording_id"] = meta.iloc[cell.frame_id].recording_id.to_numpy()
        assert cell.groupby("recording_id").y_true.nunique().max() == 1
        averaged = cell.groupby("recording_id")[columns].mean()
        truth = cell.groupby("recording_id").y_true.first().to_numpy()
        pred = labels[averaged.to_numpy().argmax(axis=1)]
        row = {"feature_set": side, "split_id": split_id, "seed": split.seed, "fold": split.fold,
               "n_train": meta.iloc[split.train].recording_id.nunique(), "n_test": len(averaged)}
        row.update(split_metrics(truth, pred, labels))
        rows.append(row)
        averaged["y_true"], averaged["y_pred"] = truth, pred
        for column in ("feature_set", "split_id", "seed", "fold"):
            averaged[column] = row[column]
        recording_predictions.append(averaged.reset_index())
    recordings = pd.DataFrame(rows)
    recordings.to_csv(Path(options.out) / f"recording_folds_{options.task}.csv", index=False)
    recordings.groupby("feature_set")[["balanced_accuracy", "macro_f1", "accuracy"]].agg(
        ["mean", "std"]).to_csv(Path(options.out) / f"recording_summary_{options.task}.csv")
    rec = pd.concat(recording_predictions, ignore_index=True)
    assert not rec.duplicated(["feature_set", "seed", "recording_id"]).any()
    rec.to_parquet(results_dir("birdclef2023") /
                   f"{options.run_name}_{options.task}_recording_predictions.parquet", index=False)
    pooled = []
    for (side, seed), cell in rec.groupby(["feature_set", "seed"]):
        complete = cell.fold.nunique() == 5
        if complete:
            assert len(cell) == meta.recording_id.nunique()
        row = {"feature_set": side, "seed": seed, "n_recordings": len(cell), "complete_seed": complete}
        row.update(split_metrics(cell.y_true.to_numpy(), cell.y_pred.to_numpy(), labels))
        pooled.append(row)
    pd.DataFrame(pooled).to_csv(Path(options.out) / f"recording_pooled_seeds_{options.task}.csv", index=False)
    (Path(options.out) / f"audit_{options.task}.json").write_text(json.dumps(
        {"status": "passed", "n_frame_predictions": len(predictions),
         "n_fits": len(folds), "split_ids": sorted(folds.split_id.unique().tolist())}, indent=2) + "\n")


class SpectrogramReader:
    def __init__(self, dataset, tag, downsample=1):
        if tag != "raw" or downsample != 1:
            raise ValueError("this reference uses normalized raw frames without downsampling")
        self.base = StackReader(dataset, tag)
        self.frames = self.base.frames
        self.window = np.hanning(257)[:-1].astype(np.float32)
        self.shape = (len(self.frames), 1, 129, 15)

    def take(self, indices):
        values = self.base.take(indices)
        windows = np.lib.stride_tricks.sliding_window_view(values, 256, axis=-1)[..., ::128, :]
        spectrum = np.fft.rfft(windows * self.window, axis=-1)
        power = np.abs(spectrum) ** 2 / self.window.sum() ** 2
        logged = np.log10(np.maximum(power, 1e-12)).swapaxes(-1, -2)
        return np.ascontiguousarray(logged, dtype=np.float32)

    def log_scales(self, indices):
        return self.base.log_scales(indices)

    def description(self):
        return {"shape": self.shape, "source": "robust-normalized raw frame",
                "representation": "log10(max(abs(STFT)**2 / sum(window)**2, 1e-12))",
                "n_fft": 256, "hop": 128, "window": "periodic Hann", "padding": False,
                "frequency_bins": 129, "time_bins": 15, "fs": 22050}


class ResidualBlock2D(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(incoming, outgoing, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(4, outgoing), nn.ReLU(),
            nn.Conv2d(outgoing, outgoing, 3, padding=1, bias=False), nn.GroupNorm(4, outgoing))
        self.skip = (nn.Conv2d(incoming, outgoing, 1, stride=stride, bias=False)
                     if incoming != outgoing or stride != 1 else nn.Identity())

    def forward(self, values):
        return torch.relu(self.main(values) + self.skip(values))


class CompactResNet2D(nn.Module):
    def __init__(self, channels, classes, width=16, scale_features=0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(channels, width, 3, stride=2, padding=1, bias=False),
                                  nn.GroupNorm(4, width), nn.ReLU())
        self.blocks = nn.Sequential(ResidualBlock2D(width, width, 1),
                                    ResidualBlock2D(width, 2 * width, 2),
                                    ResidualBlock2D(2 * width, 4 * width, 2))
        self.head = nn.Linear(8 * width + scale_features, classes)

    def forward(self, values, scales):
        values = self.blocks(self.stem(values))
        pooled = torch.cat([values.mean(dim=(2, 3)), values.amax(dim=(2, 3)), scales], dim=1)
        return self.head(pooled)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--out", default=str(output_dir("birdclef2023", "ml", "stft_reference")))
    parser.add_argument("--run-name", default="cnn_stft")
    parser.add_argument("--task", default="species")
    parser.add_argument("--audit-only", action="store_true")
    options, _ = parser.parse_known_args()
    if options.audit_only:
        audit_reference(options)
        return 0
    if "--out" not in sys.argv:
        sys.argv.extend(["--out", options.out])
    if "--run-name" not in sys.argv:
        sys.argv.extend(["--run-name", options.run_name])
    original_setup, original_finish = cnn.setup, cnn.finish

    def setup(args, name):
        result = original_setup(args, name)
        result[0]["reference_only"] = True
        result[0]["representation"] = SpectrogramReader(args.dataset, "raw").description()
        return result

    def finish(context, out, *args, notes=(), extra=None):
        notes = ["Raw log-STFT reference: a 2-D residual CNN, 3x3 kernels, widths 16/32/64, "
                 "GroupNorm and frequency/time mean/max pooling. All frames are 93 ms. "
                 "Fixed STFT parameters do not use training, validation or test labels.", *notes[1:]]
        extra["architecture"] = {"name": "CompactResNet2D", "widths": [16, 32, 64],
                                 "kernel_size": [3, 3], "normalization": "GroupNorm(4)",
                                 "pooling": ["mean over frequency/time", "max over frequency/time"]}
        return original_finish(context, out, *args, notes=notes, extra=extra)

    cnn.setup, cnn.finish = setup, finish
    cnn.StackReader, cnn.CompactResNet = SpectrogramReader, CompactResNet2D
    sys.argv.insert(1, "cnn")
    result = run_phase3.main()
    audit_reference(options)
    return result


if __name__ == "__main__":
    sys.exit(main())
