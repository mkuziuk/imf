"""Diagnostic tables and figures for the prepared DroneRF frames.

Reads `<DATA_ROOT>/dronerf/work/segment_stats.npz` (from `segment_stats.py`)
and `<DATA_ROOT>/dronerf/prepared/` (from `prepare.py`), and writes to
`output/classification-emd-irmf/dronerf/data/`:

- `lh_lag.csv`: per-segment lag of the H envelope relative to L (256-sample
  window energies, full-range cross-correlation), with peak strength;
- `activity_by_class.csv`: active-frame fractions per class and task;
- `fig_example_frames.png`, `fig_activity.png`, `fig_spectra_lag.png`.

Run with `python -I` from the worktree root.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
REPO = Path(__file__).resolve().parents[4]
FS = 40e6

# reference palette (dataviz skill), categorical slots 1-4, fixed order
CLASS_ORDER = ["background", "bebop", "ar", "phantom"]
CLASS_COLOR = {"background": "#2a78d6", "bebop": "#eb6834", "ar": "#1baf7a", "phantom": "#eda100"}
CLASS_LABEL = {"background": "Background", "bebop": "Bebop", "ar": "AR Drone", "phantom": "Phantom"}
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"


def style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.titlecolor": INK,
        "axes.titlesize": 10, "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "font.family": "sans-serif", "lines.linewidth": 1.2, "legend.frameon": False, "legend.fontsize": 8,
    })


def drone_of(bui: str) -> str:
    return "background" if bui == "00000" else {"100": "bebop", "101": "ar", "110": "phantom"}[bui[:3]]


def xcorr_peak(a: np.ndarray, h: np.ndarray):
    """Lag k maximizing corr(a[t + k], h[t]) over the full range; also the best peak >= 50 windows away."""
    a = a - a.mean()
    h = h - h.mean()
    n = a.size
    c = np.fft.irfft(np.fft.rfft(a, 2 * n) * np.conj(np.fft.rfft(h, 2 * n)), 2 * n)
    c = np.r_[c[-(n - 1):], c[:n]] / np.sqrt((a * a).sum() * (h * h).sum())
    lags = np.arange(-(n - 1), n)
    j = int(np.argmax(c))
    far = np.abs(lags - lags[j]) >= 50
    return int(lags[j]), float(c[j]), float(c[far].max()), float(c[lags == 0][0])


def lag_table(stats_path: Path) -> pd.DataFrame:
    d = np.load(stats_path)
    win = int(d["win"])
    key = {(b, bd, int(i)): k for k, (b, bd, i) in enumerate(zip(d["bui"], d["band"], d["idx"]))}
    rows = []
    for (b, bd, i), k in sorted(key.items()):
        if bd != "L":
            continue
        lag, peak, second, zero = xcorr_peak(d["log_energy"][k].astype(np.float64),
                                             d["log_energy"][key[(b, "H", i)]].astype(np.float64))
        rows.append(dict(bui=b, segment_index=i, lag_windows=lag, lag_samples=lag * win,
                         lag_ms=lag * win / FS * 1e3, peak_corr=peak, second_peak_corr=second, corr_at_zero=zero))
    df = pd.DataFrame(rows)
    # a session's lag is "consistent" when >= half of its segments agree within 2 windows of the modal lag
    out = []
    for b, g in df.groupby("bui"):
        mode = g.lag_windows.mode().iloc[0]
        agree = (g.lag_windows - mode).abs() <= 2
        g = g.assign(session_modal_lag_windows=mode, agrees_with_session=agree,
                     session_lag_consistent=bool(agree.mean() >= 0.5))
        out.append(g)
    return pd.concat(out, ignore_index=True)


def fig_examples(frames, meta, out: Path):
    fig, axes = plt.subplots(4, 2, figsize=(10, 8.2), sharex=True)
    t = np.arange(frames.shape[2]) / FS * 1e6
    for r, cls in enumerate(CLASS_ORDER):
        m = meta[meta.y_4class == cls]
        # deterministic pick: the frame whose activity is closest to the class median
        j = int((m.activity_db - m.activity_db.median()).abs().idxmin())
        row = meta.loc[j]
        for c, ch in enumerate(["L", "H"]):
            ax = axes[r, c]
            ax.plot(t, frames[j, c], color=CLASS_COLOR[cls], lw=0.6)
            ax.set_title(f"{CLASS_LABEL[cls]} ({row.bui}), band {ch}: "
                         f"{int(round(row[f'activity_db_{ch}'])) + 0:+d} dB vs noise floor", loc="left")
            if c == 0:
                ax.set_ylabel("ADC counts")
        axes[r, 0].text(0.0, -0.02, f"segment {row.group}, start {row.start_sample}", transform=axes[r, 0].transAxes,
                        fontsize=7, color=MUTED, va="top")
    for ax in axes[-1]:
        ax.set_xlabel("time in frame (µs)")
    fig.suptitle(f"Example prepared frames (L = {frames.shape[2]} samples at 40 MS/s), one per drone class",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_activity(segs, cand, thr, out: Path):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
    buis = sorted(segs.bui.unique())
    for y, b in enumerate(buis):
        s = segs[segs.bui == b]
        jitter = (np.random.default_rng(y).random(len(s)) - 0.5) * 0.35
        a1.scatter(s.active_fraction_10db * 100, y + jitter, s=12, color=CLASS_COLOR[drone_of(b)], alpha=0.75, lw=0)
        med = s.active_fraction_10db.median() * 100
        a1.plot([med, med], [y - 0.32, y + 0.32], color=INK, lw=2)
        a1.text(101, y, f"{med:.0f}%", va="center", fontsize=8, color=INK2)
    a1.set_yticks(range(len(buis)), [f"{b} {CLASS_LABEL[drone_of(b)]}" for b in buis])
    a1.invert_yaxis()
    a1.set_xlim(0, 100)
    a1.set_xlabel(f"active frames per segment (%), uniform grid, ≥ {thr:.0f} dB above floor")
    a1.set_title("Share of active frames, one dot per segment (bar = median)", loc="left")

    g2c = dict(zip(segs.group, segs.bui.map(drone_of)))
    cls = pd.Series(cand["group"]).map(g2c).to_numpy()
    act = cand["activity_db"].max(axis=1)
    x = np.linspace(-2, 75, 400)
    for c in CLASS_ORDER:
        v = np.sort(act[cls == c])
        ecdf = 1 - np.searchsorted(v, x, side="left") / v.size
        a2.plot(x, ecdf * 100, color=CLASS_COLOR[c], lw=2)
        k = np.searchsorted(x, {"background": 14, "bebop": 54, "ar": 41, "phantom": 27}[c])
        a2.text(x[k] + 0.8, ecdf[k] * 100 * 1.12, CLASS_LABEL[c], color=INK2, fontsize=8)
    a2.axvline(thr, color=INK2, lw=1)
    a2.text(thr + 0.8, 97, f"gate {thr:.0f} dB", fontsize=8, color=INK2, va="top")
    a2.set_yscale("log")
    a2.set_ylim(0.5, 100)
    a2.set_yticks([1, 3, 10, 30, 100], ["1", "3", "10", "30", "100"])
    a2.set_xlabel("frame activity: max over L/H of 10·log10(var / noise floor) (dB)")
    a2.set_ylabel("candidate frames above x (%)")
    a2.set_title("Activity distribution of all candidate frames", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def fig_spectra_lag(stats_path: Path, lags: pd.DataFrame, out: Path):
    d = np.load(stats_path)
    f = d["freqs"] / 1e6
    cls = np.array([drone_of(b) for b in d["bui"]])
    fig = plt.figure(figsize=(11, 7.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.05])
    for c, band in enumerate(["L", "H"]):
        ax = fig.add_subplot(gs[0, c])
        for k in CLASS_ORDER:
            p = 10 * np.log10(np.median(d["psd"][(cls == k) & (d["band"] == band)], axis=0))
            ax.plot(f, p, color=CLASS_COLOR[k], lw=1.4, label=CLASS_LABEL[k])
        ax.set_xlim(0, 20)
        if c == 0:
            ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=1, handlelength=1.5)
        ax.set_xlabel("frequency (MHz), one-sided spectrum of the real samples")
        ax.set_ylabel("PSD (dB, arbitrary ref.)")
        ax.set_title(f"Band {band}: median segment PSD per class", loc="left")
    ax = fig.add_subplot(gs[1, :])
    buis = sorted(lags.bui.unique())
    for y, b in enumerate(buis):
        g = lags[lags.bui == b]
        jitter = (np.random.default_rng(y).random(len(g)) - 0.5) * 0.35
        ax.scatter(g.lag_ms, y + jitter, s=10 + 120 * np.clip(g.peak_corr, 0, 1), lw=0, alpha=0.7,
                   color=CLASS_COLOR[drone_of(b)])
        if g.session_lag_consistent.iloc[0]:
            txt = f"{g.session_modal_lag_windows.iloc[0] * 256 / FS * 1e3:+.1f} ms in {int(g.agrees_with_session.sum())}/{len(g)}"
        else:
            txt = "no consistent lag"
        ax.text(258, y, txt, va="center", fontsize=8, color=INK2)
    ax.set_yticks(range(len(buis)), [f"{b} {CLASS_LABEL[drone_of(b)]}" for b in buis])
    ax.invert_yaxis()
    ax.set_xlim(-250, 345)
    ax.set_xticks([-200, -100, 0, 100, 200])
    ax.axvline(0, color=AXIS, lw=1)
    ax.set_xlabel("envelope cross-correlation peak (ms): an event at time t of the H file appears at t + lag "
                  "in the L file; dot size = peak correlation")
    ax.set_title("L and H files of one segment are offset in time by a session-specific lag", loc="left")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", type=Path, default=DATA_ROOT / "dronerf/prepared")
    p.add_argument("--stats", type=Path, default=DATA_ROOT / "dronerf/work/segment_stats.npz")
    p.add_argument("--out", type=Path, default=REPO / "output/classification-emd-irmf/dronerf/data")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    style()

    info = json.loads((a.prepared / "dataset.json").read_text())
    meta = pd.read_parquet(a.prepared / "meta.parquet")
    frames = np.load(a.prepared / "frames.npy", mmap_mode="r")
    segs = pd.read_csv(a.prepared / "segments.csv", dtype={"bui": str})
    cand = dict(np.load(a.prepared / "candidates.npz"))
    thr = info["preparation"]["threshold_db"] or 10.0

    lags = lag_table(a.stats)
    lags.to_csv(a.out / "lh_lag.csv", index=False)
    rows = []
    for task, d in info["activity"].items():
        for lab, v in d.items():
            rows.append({"task": task, "class": lab, "frames": info["class_counts"][task][lab], **v})
    pd.DataFrame(rows).to_csv(a.out / "activity_by_class.csv", index=False)

    fig_examples(frames, meta, a.out / "fig_example_frames.png")
    fig_activity(segs, cand, thr, a.out / "fig_activity.png")
    fig_spectra_lag(a.stats, lags, a.out / "fig_spectra_lag.png")
    s = lags.groupby("bui").agg(modal_lag_ms=("session_modal_lag_windows", lambda v: v.iloc[0] * 256 / FS * 1e3),
                                agree=("agrees_with_session", "sum"), n=("bui", "size"),
                                consistent=("session_lag_consistent", "first"), peak_med=("peak_corr", "median"))
    print(s.round(3).to_string())
    print("wrote", sorted(x.name for x in a.out.iterdir()))


if __name__ == "__main__":
    main()
