"""Diagnostic figures for the prepared birdclef2023 frames.

Writes to output/classification-emd-irmf/birdclef2023/data/:
  example_frames.png  one frame per species: 1 s context spectrogram + the frame waveform
  class_spectra.png   per-species median log power spectrum of frames (IQR band)
  frame_energy.png    frame score relative to recording max, frame level per species,
                      frames kept per recording

    .venv/bin/python -I experiments/classification-emd-irmf/datasets/birdclef2023/figures.py
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator
from scipy.signal import spectrogram, welch

NAME = "birdclef2023"
DEFAULT_DATA_ROOT = "/Users/mikhail/Datasets/imf-classification"
REPO = Path(__file__).resolve().parents[4]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES, REFERENCE = "#2a78d6", "#9a9994"
SEQ = LinearSegmentedColormap.from_list(
    "blue_seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.titlesize": 9, "axes.labelsize": 8, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 8,
})


def birdclef_context(raw_dir, fs, length):
    """Loader returning a recording's audio at fs in the stored start_sample coordinates."""
    import librosa
    import soundfile as sf

    def load(row):
        with sf.SoundFile(raw_dir / "train_audio" / row["source_file"]) as handle:
            sr = handle.samplerate
            stop = int((row["start_sample"] + length + fs) / fs * sr) + 1
            audio = handle.read(min(stop, handle.frames), dtype="float64", always_2d=True).mean(axis=1)
        audio = librosa.resample(audio, orig_sr=sr, target_sr=fs, res_type="soxr_hq")
        return audio - audio.mean()

    return load


def _grid(n):
    ncols = 4
    return (n + ncols - 1) // ncols, ncols


def example_frames(frames, meta, info, load_audio, label_col, name_col, out, rng):
    fs, length = info["fs"], info["frame_length"]
    classes = sorted(meta[label_col].unique())
    nrows, ncols = _grid(len(classes))
    fig = plt.figure(figsize=(13, 2.75 * nrows))
    outer = fig.add_gridspec(nrows, ncols, hspace=0.55, wspace=0.28)
    context = fs  # 1 s window centred on the frame
    for k, code in enumerate(classes):
        rows = meta.index[(meta[label_col] == code) & (meta["energy_rank"] == 0)].to_numpy()
        i = int(rng.choice(rows))
        row = meta.loc[i]
        audio = load_audio(row)
        centre = row["start_sample"] + length // 2
        lo = int(max(0, centre - context // 2))
        seg = audio[lo: lo + context]
        f, t, s = spectrogram(seg, fs=fs, nperseg=512, noverlap=384)
        inner = outer[k // ncols, k % ncols].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.08)
        ax = fig.add_subplot(inner[0])
        db = 10 * np.log10(s + 1e-12)
        ax.pcolormesh(t + lo / fs, f / 1000, db, cmap=SEQ, vmin=db.max() - 70, vmax=db.max(),
                      shading="auto", rasterized=True)
        for edge in (row["start_sample"], row["start_sample"] + length):
            ax.axvline(edge / fs, color=INK, lw=0.8, ls="--")
        title = f"{row[name_col]} ({code})" if name_col != label_col else str(code)
        ax.set_title(title, color=INK, loc="left")
        ax.set_ylabel("kHz")
        ax.tick_params(labelbottom=False)
        axw = fig.add_subplot(inner[1])
        tt = np.arange(length) / fs * 1000
        axw.plot(tt, frames[i, 0], color=SERIES, lw=0.6)
        axw.set_xlim(tt[0], tt[-1])
        axw.set_xlabel(f"ms in frame ({row['recording_id']} at {row['start_sample'] / fs:.2f} s)")
        axw.set_yticks([])
    fig.suptitle(
        f"One top-energy frame per class: 1 s context spectrogram (frame between dashed lines) "
        f"and the stored frame, fs = {fs} Hz, L = {length}", x=0.07, ha="left", color=INK, fontsize=11)
    fig.savefig(out / "example_frames.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def class_spectra(frames, meta, info, label_col, name_col, out):
    fs = info["fs"]
    f, p = welch(frames[:, 0].astype(np.float64), fs=fs, nperseg=512, axis=-1)
    db = 10 * np.log10(p + 1e-20)
    db -= db.max(axis=1, keepdims=True)  # each frame's spectrum relative to its own peak
    overall = np.median(db, axis=0)
    classes = sorted(meta[label_col].unique())
    nrows, ncols = _grid(len(classes))
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 2.25 * nrows), sharex=True, sharey=True,
                             squeeze=False)
    for ax, code in zip(axes.flat, classes):
        sel = (meta[label_col] == code).to_numpy()
        q25, q50, q75 = np.percentile(db[sel], [25, 50, 75], axis=0)
        ax.plot(f / 1000, overall, color=REFERENCE, lw=1.0, label="all classes")
        ax.fill_between(f / 1000, q25, q75, color=SERIES, alpha=0.2, lw=0, label="IQR")
        ax.plot(f / 1000, q50, color=SERIES, lw=1.6, label="median")
        name = meta.loc[sel, name_col].iloc[0]
        ax.set_title(f"{name} ({code})" if name_col != label_col else str(code), loc="left")
        ax.grid(True, axis="y")
        ax.set_ylim(-70, 2)
    for ax in axes.flat[len(classes):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("frequency (kHz)")
        ax.xaxis.set_tick_params(labelbottom=True)
    for ax in axes[:, 0]:
        ax.set_ylabel("dB re frame peak")
    axes[0, 0].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle("Frame power spectra per class (Welch, 512-point), median and interquartile band; "
                 "gray: median over all frames", x=0.07, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out / "class_spectra.png", dpi=130)
    plt.close(fig)


def frame_energy(meta, info, label_col, out):
    species = sorted(meta[label_col].unique())
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"width_ratios": [1, 1.8, 1]})
    ax = axes[0]
    floor = info["preparation"]["db_floor"]
    ax.hist(meta["score_db_rel"], bins=np.linspace(-floor, 0, 31), color=SERIES, edgecolor=SURFACE, lw=1)
    ax.set_xlabel("frame score re loudest frame of its recording (dB)")
    ax.set_ylabel("frames")
    ax.set_title("Selection score (high-passed power)", loc="left")
    ax.grid(True, axis="y")
    ax = axes[1]
    data = [meta.loc[meta[label_col] == c, "frame_rms_dbfs"].to_numpy() for c in species]
    parts = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                       medianprops={"color": INK, "lw": 1.2}, whiskerprops={"color": INK2},
                       capprops={"color": INK2}, boxprops={"edgecolor": SERIES})
    for box in parts["boxes"]:
        box.set_facecolor("#cde2fb")
    ax.set_xticks(range(1, len(species) + 1), species, rotation=60, ha="right")
    ax.set_ylabel("frame RMS (dBFS)")
    ax.set_title("Stored frame level per class (whiskers 1.5 IQR)", loc="left")
    ax.grid(True, axis="y")
    ax = axes[2]
    per_rec = meta.groupby("recording_id").size()
    cap = info["preparation"]["max_frames_per_recording"]
    ax.hist(per_rec, bins=np.arange(0.5, cap + 1.5), color=SERIES, edgecolor=SURFACE, lw=1)
    ax.set_xlabel("frames kept per recording")
    ax.set_ylabel("recordings")
    ax.set_xlim(0, cap + 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(f"{per_rec.size} recordings, median {per_rec.median():.0f} frames", loc="left")
    ax.grid(True, axis="y")
    fig.tight_layout()
    fig.savefig(out / "frame_energy.png", dpi=130)
    plt.close(fig)


def render(name, load_audio_factory, label_col, name_col, argv=None):
    parser = argparse.ArgumentParser(description=f"{name} diagnostic figures")
    parser.add_argument("--data-root", default=os.environ.get("IMF_CLS_DATA", DEFAULT_DATA_ROOT))
    parser.add_argument("--out", default=str(REPO / "output" / "classification-emd-irmf" / name / "data"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    base = Path(args.data_root) / name
    frames = np.load(base / "prepared" / "frames.npy", mmap_mode="r")
    meta = pd.read_parquet(base / "prepared" / "meta.parquet")
    info = json.loads((base / "prepared" / "dataset.json").read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    load_audio = load_audio_factory(base / "raw", info["fs"], info["frame_length"])
    example_frames(frames, meta, info, load_audio, label_col, name_col, out, rng)
    class_spectra(np.asarray(frames), meta, info, label_col, name_col, out)
    frame_energy(meta, info, label_col, out)
    print("wrote", out, sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    render(NAME, birdclef_context, "y_species", "species_common")
