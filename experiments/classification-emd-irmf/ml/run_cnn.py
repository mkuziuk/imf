#!/usr/bin/env python3
"""Compact residual 1-D CNN on streamed, scale-ordered component stacks."""

import sys
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(STUDY_DIR))

from ml.common import feature_sets, finish, inner_split, parser_for, record, setup

import json
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch import nn

from emdirmf.stacks import StackReader


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device(requested):
    available = torch.backends.mps.is_available()
    if requested == "mps" and not available:
        raise ValueError("MPS was requested but is unavailable")
    return torch.device("mps" if requested == "mps" or requested == "auto" and available else "cpu")


class ResidualBlock(nn.Module):
    def __init__(self, incoming, outgoing, stride):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv1d(incoming, outgoing, 7, stride=stride, padding=3, bias=False),
            nn.GroupNorm(4, outgoing), nn.ReLU(),
            nn.Conv1d(outgoing, outgoing, 7, padding=3, bias=False),
            nn.GroupNorm(4, outgoing),
        )
        self.skip = nn.Conv1d(incoming, outgoing, 1, stride=stride, bias=False) if (
            incoming != outgoing or stride != 1) else nn.Identity()

    def forward(self, x):
        return torch.relu(self.main(x) + self.skip(x))


class CompactResNet(nn.Module):
    """Same network for every method; only input channels and optional scale vary."""

    def __init__(self, channels, classes, width=16, scale_features=0):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(channels, width, 7, stride=2, padding=3, bias=False),
                                  nn.GroupNorm(4, width), nn.ReLU())
        self.blocks = nn.Sequential(ResidualBlock(width, width, 1),
                                    ResidualBlock(width, width * 2, 2),
                                    ResidualBlock(width * 2, width * 4, 2))
        self.head = nn.Linear(width * 8 + scale_features, classes)

    def forward(self, x, scales):
        x = self.blocks(self.stem(x))
        pooled = torch.cat([x.mean(dim=-1), x.amax(dim=-1), scales], dim=1)
        return self.head(pooled)


class FrameDataset(torch.utils.data.Dataset):
    """Read one frame per item. No worker processes or preloaded stack copies."""

    def __init__(self, reader, indices, targets, scales=None):
        self.reader = reader
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.targets = np.asarray(targets, dtype=np.int64)
        self.scales = scales

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        frame_id = self.indices[index]
        scale = np.empty(0, dtype=np.float32) if self.scales is None else self.scales[frame_id]
        return (torch.from_numpy(self.reader.take([frame_id])[0]),
                torch.from_numpy(np.asarray(scale, dtype=np.float32)), int(self.targets[frame_id]))


def make_loader(reader, indices, targets, scales, batch_size, seed, shuffle):
    generator = torch.Generator().manual_seed(seed)
    return torch.utils.data.DataLoader(FrameDataset(reader, indices, targets, scales),
                                       batch_size=batch_size, shuffle=shuffle, generator=generator,
                                       num_workers=0, drop_last=False)


def scale_inputs(reader, train, enabled, batch_size):
    if not enabled:
        return None
    values = np.empty((reader.shape[0], reader.frames.shape[1]), dtype=np.float32)
    for start in range(0, len(values), batch_size):
        indices = np.arange(start, min(start + batch_size, len(values)))
        values[indices] = reader.log_scales(indices)
    scaler = StandardScaler().fit(values[train])
    return scaler.transform(values).astype(np.float32)


def synchronize(device):
    if device.type == "mps":
        torch.mps.synchronize()


def epoch_pass(model, loader, loss_fn, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, n, predictions, targets = 0.0, 0, [], []
    with torch.set_grad_enabled(optimizer is not None):
        for x, scale, y in loader:
            x, scale, y = x.to(device), scale.to(device), y.to(device)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x, scale)
            loss = loss_fn(logits, y)
            if optimizer is not None:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(y)
            n += len(y)
            predictions.append(logits.detach().argmax(dim=1).cpu().numpy())
            targets.append(y.cpu().numpy())
    return total_loss / n, np.concatenate(targets), np.concatenate(predictions)


def train_model(reader, train, valid, encoded, use_scale, args, device, seed, epochs,
                curves, key, stage):
    from sklearn.metrics import balanced_accuracy_score

    set_seed(seed)
    scales = scale_inputs(reader, train, use_scale, args.batch_size)
    train_loader = make_loader(reader, train, encoded, scales, args.batch_size, seed, True)
    valid_loader = None if valid is None else make_loader(
        reader, valid, encoded, scales, args.batch_size, seed, False)
    model = CompactResNet(reader.shape[1], len(np.unique(encoded)), args.width,
                          reader.frames.shape[1] if use_scale else 0).to(device)
    weights = compute_class_weight("balanced", classes=np.unique(encoded), y=encoded[train])
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-3)
    best_loss, best_epoch, stale = np.inf, 1, 0
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        train_loss, _, _ = epoch_pass(model, train_loader, loss_fn, device, optimizer)
        if not np.isfinite(train_loss):
            raise FloatingPointError("CNN training loss is non-finite")
        valid_loss, valid_ba = np.nan, np.nan
        if valid_loader is not None:
            valid_loss, target, pred = epoch_pass(model, valid_loader, loss_fn, device)
            if not np.isfinite(valid_loss):
                raise FloatingPointError("CNN validation loss is non-finite")
            valid_ba = balanced_accuracy_score(target, pred)
            if valid_loss < best_loss - 1e-4:
                best_loss, best_epoch, stale = valid_loss, epoch, 0
            else:
                stale += 1
        synchronize(device)
        curves.append({**key, "stage": stage, "epoch": epoch, "train_loss": train_loss,
                       "valid_loss": valid_loss, "valid_balanced_accuracy": valid_ba,
                       "seconds": time.perf_counter() - started})
        if valid_loader is not None and stale >= args.patience:
            break
    return model, scales, best_epoch if valid_loader is not None else epochs


def run(args):
    context, _, y, groups, splits, methods, out = setup(args, "cnn")
    torch.set_num_threads(args.workers)
    device = choose_device(args.device)
    encoder = LabelEncoder().fit(y)
    encoded = encoder.transform(y)
    rows, predictions, curves, descriptions = [], [], [], {}
    for tag in methods:
        reader = StackReader(args.dataset, tag, args.downsample)
        descriptions[tag] = reader.description()
        for feature_set in feature_sets(args):
            for split in splits:
                fit, valid = inner_split(split.train, y, groups, split.seed)
                seed = split.seed * 100 + split.fold
                key = {"method": tag, "feature_set": feature_set, "split_id": split.split_id, "seed": seed}
                use_scale = feature_set.endswith("log_scale")
                synchronize(device)
                started = time.perf_counter()
                model, scales, best_epoch = train_model(reader, fit, valid, encoded, use_scale, args,
                                                       device, seed, args.epochs, curves, key, "selection")
                del model, scales
                # Validation selects only the epoch count. Reinitialize and refit
                # all training-fold samples, with scale moments fitted there too.
                model, scales, _ = train_model(reader, split.train, None, encoded, use_scale, args,
                                               device, seed, best_epoch, curves, key, "refit")
                synchronize(device)
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                loader = make_loader(reader, split.test, encoded, scales, args.batch_size, seed, False)
                _, _, pred = epoch_pass(model, loader, nn.CrossEntropyLoss(), device)
                synchronize(device)
                predict_seconds = time.perf_counter() - started
                pred = encoder.inverse_transform(pred)
                record(rows, predictions, split, y, pred, tag, feature_set, "compact_resnet",
                       fit_seconds, predict_seconds, best_epoch=best_epoch,
                       n_parameters=sum(p.numel() for p in model.parameters()),
                       best_params=json.dumps({"epochs": best_epoch, "seed": seed}))
                pd.DataFrame(curves).to_csv(out / f"curves_{context['task']}.csv", index=False)
                del model, scales
                if device.type == "mps":
                    torch.mps.empty_cache()
    return finish(context, out, rows, predictions, notes=[
        "Compact residual CNN: width 16 by default, three residual blocks, kernels 7, GroupNorm, "
        "temporal mean/max pooling, AdamW, gradient norm cap 5, weighted cross entropy. Architecture is identical across methods "
        "apart from the input channel count; optional standardized log_scale joins the pooled head.",
        "Grouped inner-validation loss selects the epoch count with patience. A fresh model then trains on "
        "all outer training samples for that many epochs. Test samples never select epochs or scale statistics.",
        "Read-only stacks are loaded one frame at a time with DataLoader num_workers=0. CPU threads are capped "
        "by --workers. Fixed CPU/MPS seeds and deterministic algorithms are enabled; numerical results may "
        "vary across devices or library versions. Training curves include selection and refit stages.",
    ], extra={"device": str(device), "stacks": descriptions, "feature_sets": list(feature_sets(args)),
              "architecture": {"name": "CompactResNet", "widths": [args.width, 2 * args.width, 4 * args.width],
                  "kernel_size": 7, "normalization": "GroupNorm(4)", "pooling": ["mean", "max"]},
              "optimizer": {"name": "AdamW", "learning_rate": args.learning_rate, "weight_decay": 1e-3,
                  "gradient_norm_cap": 5.0, "loss": "balanced cross entropy"}})


def main(argv=None):
    parser = parser_for("Phase-3 compact 1-D residual CNN")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--downsample", type=int, default=1)
    args = parser.parse_args(argv)
    args.epochs = args.epochs if args.epochs is not None else (6 if args.quick else 30)
    args.patience = args.patience if args.patience is not None else (2 if args.quick else 5)
    if min(args.epochs, args.patience, args.batch_size, args.width) < 1 or args.width % 4:
        parser.error("epochs/patience/batch-size must be positive; width must be a positive multiple of 4")
    if args.learning_rate <= 0:
        parser.error("learning-rate must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
