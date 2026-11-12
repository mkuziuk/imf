"""Prepare standardized DroneRF frames for the EMD/IMF/IRMF classification study.

Input: the float32 segment cache written by `convert_csv.py`
(`<DATA_ROOT>/dronerf/raw/npy/<BUI><band>_<idx>.npy`, 10^7 real samples per
file, 40 MS/s, band L = lower and H = upper half of the 2.4 GHz ISM band).

Output (`<DATA_ROOT>/dronerf/prepared/`, see PROTOCOL.md):
`frames.npy` (N, 2, L) float32 with channels [L, H], `meta.parquet`, and
`dataset.json`.

Frame selection (label-free, identical for every segment):

1. Each segment (one L file and the H file with the same BUI and index) is cut
   into a grid of non-overlapping candidate frames of `--frame-length` samples.
   The first and last grid frames are dropped (files start with up to 12 and
   end with up to 3 zero samples).
2. For every candidate and channel, the frame variance is compared with that
   channel's noise floor, which is the median candidate variance pooled over all
   segments of all classes (most candidates are noise-only, so the pooled median
   is the receiver noise level). `activity_db_c = 10 log10(var_c / floor_c)`.
3. A candidate is *active* when `max(activity_db_L, activity_db_H) >=
   --threshold-db` (default 10 dB). Noise-only frames of 2048 samples fluctuate
   by well under 1 dB, so 10 dB means a transmission occupies a substantial
   part of the frame.
4. Frames are drawn without replacement from the active candidates of each
   segment with `numpy.random.default_rng(seed)`. Labels are used only for the
   per-class frame budget: by default every 10-class label gets
   `--frames-per-class` frames, spread as evenly as possible over its segments
   (water-filling if a segment runs out of active candidates). With
   `--frames-per-segment K` every segment instead contributes min(K, active).
   `--threshold-db -inf` gives uniform random frames (no activity gate).

L and H files of the same segment are NOT simultaneous: their envelopes line up
only after a session-specific shift of 1.5-3.4 M samples (see README.md). Both
channels are cut at the same `start_sample` of their own file, which is how the
dataset authors pair them. Treat the two channels as two views of the same
segment, not as a phase-coherent pair.

Run (reads only .npy files from the cache; still run with `-I`):

    .venv/bin/python -I experiments/classification-emd-irmf/datasets/dronerf/prepare.py
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path(os.environ.get("IMF_CLS_DATA", "/Users/mikhail/Datasets/imf-classification"))
FS = 40_000_000
SEGMENT_SAMPLES = 10_000_000
SUBWIN = 128  # sub-window length used for the occupancy statistic
CHANNELS = ["L", "H"]
NAME_RE = re.compile(r"^(?P<bui>[01]{5})(?P<band>[LH])_(?P<idx>\d+)\.npy$")

DRONE = {"00000": "background", "100": "bebop", "101": "ar", "110": "phantom"}
MODE = {"00": "on_connected", "01": "hovering", "10": "flying", "11": "video_recording"}
ZIP_FOLDER = {"background": "Background RF activites", "bebop": "Bepop drone",
              "ar": "AR drone", "phantom": "Phantom drone"}

SOURCE_URL = "https://data.mendeley.com/datasets/f4c2b4n755/1"
LICENSE = "CC BY 4.0"
CITATION = (
    "M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, A. Erbad. RF-based drone detection and "
    "identification using deep learning approaches: An initiative towards a large open source drone "
    "database. Future Generation Computer Systems 100 (2019) 86-97, doi:10.1016/j.future.2019.05.007. "
    "Data: M. S. Allahham, M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, A. Erbad. DroneRF dataset: "
    "A dataset of drones for RF-based detection, classification and identification. Mendeley Data, v1 "
    "(2019), doi:10.17632/f4c2b4n755.1."
)


def drone_of(bui: str) -> str:
    return "background" if bui == "00000" else DRONE[bui[:3]]


def mode_of(bui: str) -> str:
    return "none" if bui == "00000" else MODE[bui[3:]]


# --------------------------------------------------------------------------- #
# Pass 1: candidate statistics per segment (label-free)
# --------------------------------------------------------------------------- #
def candidate_stats(args):
    paths, frame_length = args
    n_frames = SEGMENT_SAMPLES // frame_length
    starts = np.arange(1, n_frames - 1) * frame_length
    var = np.empty((starts.size, 2), np.float64)
    sub = np.empty((starts.size, 2, frame_length // SUBWIN), np.float32)
    for c, p in enumerate(paths):
        x = np.load(p, mmap_mode="r")
        if x.shape != (SEGMENT_SAMPLES,):
            raise ValueError(f"{p}: unexpected shape {x.shape}")
        fr = np.asarray(x[starts[0]: starts[-1] + frame_length], dtype=np.float64)
        fr = fr.reshape(starts.size, frame_length)
        fr = fr - fr.mean(axis=1, keepdims=True)
        var[:, c] = (fr ** 2).mean(axis=1)
        sub[:, c] = (fr.reshape(starts.size, -1, SUBWIN) ** 2).mean(axis=2)
    return starts, var, sub


def extract_frames(args):
    paths, starts, frame_length = args
    out = np.empty((starts.size, 2, frame_length), np.float32)
    for c, p in enumerate(paths):
        x = np.load(p, mmap_mode="r")
        for i, s in enumerate(starts):
            out[i, c] = x[s: s + frame_length]
    return out


def allocate(total: int, caps: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Spread `total` frames over segments as evenly as possible under caps."""
    q = np.zeros(caps.size, dtype=np.int64)
    remaining = int(total)
    open_ = q < caps
    while remaining > 0 and open_.any():
        idx = np.flatnonzero(open_)
        share = remaining // idx.size
        if share == 0:
            pick = rng.choice(idx, size=remaining, replace=False)
            q[pick] += 1
            remaining = 0
            break
        add = np.minimum(share, caps[idx] - q[idx])
        q[idx] += add
        remaining -= int(add.sum())
        open_ = q < caps
    return q


def source_names(npy_dir: Path) -> dict[str, str]:
    """Map cached .npy file name -> 'zip folder/archive.rar::member.csv'."""
    log_path = npy_dir / "conversion_log.json"
    out = {}
    if not log_path.exists():
        return out
    for r in json.loads(log_path.read_text())["files"]:
        src = Path(r["src"])
        # src = .../extracted/rar/<archive stem>/<member path>
        parts = src.parts
        k = parts.index("rar")
        stem = parts[k + 1]
        member = "/".join(parts[k + 2:])
        bui = NAME_RE.match(Path(r["dst"]).name)["bui"]
        out[Path(r["dst"]).name] = f"DroneRF/{ZIP_FOLDER[drone_of(bui)]}/{stem}.rar::{member}"
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npy", type=Path, default=DATA_ROOT / "dronerf/raw/npy")
    p.add_argument("--out", type=Path, default=DATA_ROOT / "dronerf/prepared")
    p.add_argument("--frame-length", type=int, default=2048)
    p.add_argument("--frames-per-class", type=int, default=1200,
                   help="frames per 10-class label (BUI), spread evenly over its segments")
    p.add_argument("--frames-per-segment", type=int, default=None,
                   help="if set, take min(K, n_active) frames from every segment instead")
    p.add_argument("--threshold-db", type=float, default=10.0,
                   help="activity gate on max-channel frame SNR; -inf disables the gate")
    p.add_argument("--seed", type=int, default=20261020)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    if a.frame_length % SUBWIN or a.frame_length > 4096:
        raise SystemExit(f"--frame-length must be a multiple of {SUBWIN} and <= 4096")

    t_start = time.perf_counter()
    rng = np.random.default_rng(a.seed)

    # ---- discover segments -------------------------------------------------
    files = {}
    for f in a.npy.glob("*.npy"):
        m = NAME_RE.match(f.name)
        if m:
            files[(m["bui"], int(m["idx"]), m["band"])] = f
    seg_keys = sorted({(b, i) for (b, i, _) in files})
    missing = [k for k in seg_keys if (k[0], k[1], "L") not in files or (k[0], k[1], "H") not in files]
    if missing:
        raise SystemExit(f"segments without both bands: {missing}")
    segs = pd.DataFrame({"bui": [b for b, _ in seg_keys], "segment_index": [i for _, i in seg_keys]})
    segs["group"] = [f"{b}_{i:02d}" for b, i in seg_keys]
    segs["drone"] = segs.bui.map(drone_of)
    paths = [(str(files[(b, i, "L")]), str(files[(b, i, "H")])) for b, i in seg_keys]
    print(f"{len(segs)} segments ({len(files)} files)", flush=True)

    # ---- pass 1: candidate statistics -------------------------------------
    t0 = time.perf_counter()
    with ProcessPoolExecutor(a.workers) as ex:
        stats = list(ex.map(candidate_stats, [(pp, a.frame_length) for pp in paths], chunksize=2))
    t_stats = time.perf_counter() - t0
    print(f"candidate statistics: {t_stats:.0f} s", flush=True)

    all_var = np.concatenate([s[1] for s in stats])
    floor = np.median(all_var, axis=0)  # per channel, pooled over all segments (label-free)
    floor_db = 10 * np.log10(floor)
    gate = a.threshold_db

    cand = []
    for k, (starts, var, sub) in enumerate(stats):
        act = 10 * np.log10(var / floor)
        occ = (10 * np.log10(sub / floor[None, :, None] + 1e-30) >= (gate if np.isfinite(gate) else 10.0)).mean(axis=2)
        cand.append(dict(starts=starts, act=act, occ=occ, var=var, active=act.max(axis=1) >= gate))
    segs["n_candidates"] = [c["starts"].size for c in cand]
    segs["n_active"] = [int(c["active"].sum()) for c in cand]
    # activity at the default 10 dB, regardless of the gate in use, for reporting
    segs["active_fraction_10db"] = [float((c["act"].max(axis=1) >= 10).mean()) for c in cand]
    segs["active_fraction_10db_L"] = [float((c["act"][:, 0] >= 10).mean()) for c in cand]
    segs["active_fraction_10db_H"] = [float((c["act"][:, 1] >= 10).mean()) for c in cand]

    # ---- per-segment quotas ------------------------------------------------
    caps = segs.n_active.to_numpy()
    quota = np.zeros(len(segs), dtype=np.int64)
    if a.frames_per_segment is not None:
        quota = np.minimum(caps, a.frames_per_segment)
    else:
        for bui, idx in segs.groupby("bui").indices.items():
            quota[idx] = allocate(a.frames_per_class, caps[idx], rng)
    segs["n_selected"] = quota

    # ---- draw frames ---------------------------------------------------------
    picks = []
    for k in range(len(segs)):
        ok = np.flatnonzero(cand[k]["active"])
        sel = np.sort(rng.choice(ok, size=quota[k], replace=False)) if quota[k] else np.array([], int)
        picks.append(sel)

    n_total = int(quota.sum())
    a.out.mkdir(parents=True, exist_ok=True)
    tmp_frames = a.out / "frames.tmp.npy"
    frames = np.lib.format.open_memmap(tmp_frames, mode="w+", dtype=np.float32,
                                       shape=(n_total, 2, a.frame_length))
    t0 = time.perf_counter()
    jobs = [(paths[k], cand[k]["starts"][picks[k]], a.frame_length) for k in range(len(segs))]
    pos = 0
    with ProcessPoolExecutor(a.workers) as ex:
        for k, fr in enumerate(ex.map(extract_frames, jobs, chunksize=2)):
            frames[pos: pos + len(fr)] = fr
            pos += len(fr)
    frames.flush()
    t_extract = time.perf_counter() - t0
    print(f"extracted {n_total} frames: {t_extract:.0f} s", flush=True)

    # ---- metadata --------------------------------------------------------------
    srcmap = source_names(a.npy)
    rows = []
    for k, seg in segs.iterrows():
        c = cand[k]
        for j in picks[k]:
            rows.append({
                "group": seg.group,
                "bui": seg.bui,
                "drone": seg.drone,
                "flight_mode": mode_of(seg.bui),
                "segment_index": int(seg.segment_index),
                "start_sample": int(c["starts"][j]),
                "candidate_index": int(j),
                "activity_db": float(c["act"][j].max()),
                "activity_db_L": float(c["act"][j, 0]),
                "activity_db_H": float(c["act"][j, 1]),
                "occupancy_L": float(c["occ"][j, 0]),
                "occupancy_H": float(c["occ"][j, 1]),
                "rms_L": float(np.sqrt(c["var"][j, 0])),
                "rms_H": float(np.sqrt(c["var"][j, 1])),
                "segment_active_fraction": float(seg.active_fraction_10db),
                "source_file": srcmap.get(Path(paths[k][0]).name, Path(paths[k][0]).name),
                "source_file_H": srcmap.get(Path(paths[k][1]).name, Path(paths[k][1]).name),
                "cache_file_L": Path(paths[k][0]).name,
                "cache_file_H": Path(paths[k][1]).name,
            })
    meta = pd.DataFrame(rows)
    meta.insert(0, "frame_id", np.arange(len(meta), dtype=np.int64))
    meta.insert(2, "y_2class", np.where(meta.bui == "00000", "background", "drone"))
    meta.insert(3, "y_4class", meta.drone.to_numpy())
    meta.insert(4, "y_10class", meta.bui.to_numpy())
    for col in ["group", "y_2class", "y_4class", "y_10class", "bui", "drone", "flight_mode"]:
        meta[col] = meta[col].astype(str)
    assert len(meta) == n_total

    # ---- summaries for dataset.json ---------------------------------------------
    tasks = {"2class": "y_2class", "4class": "y_4class", "10class": "y_10class"}
    seg_lab = segs.assign(y_2class=np.where(segs.bui == "00000", "background", "drone"),
                          y_4class=segs.drone, y_10class=segs.bui)
    class_counts = {t: meta[col].value_counts().sort_index().to_dict() for t, col in tasks.items()}
    group_counts = {t: meta.groupby(col).group.nunique().sort_index().to_dict() for t, col in tasks.items()}
    activity = {}
    for t, col in tasks.items():
        g = seg_lab.groupby(col)
        activity[t] = {
            lab: {
                "segments": int(len(d)),
                "uniform_active_fraction_pooled": float((d.active_fraction_10db * d.n_candidates).sum() / d.n_candidates.sum()),
                "uniform_active_fraction_L": float((d.active_fraction_10db_L * d.n_candidates).sum() / d.n_candidates.sum()),
                "uniform_active_fraction_H": float((d.active_fraction_10db_H * d.n_candidates).sum() / d.n_candidates.sum()),
                "segment_active_fraction_min": float(d.active_fraction_10db.min()),
                "segment_active_fraction_median": float(d.active_fraction_10db.median()),
                "segment_active_fraction_max": float(d.active_fraction_10db.max()),
                "selected_occupancy_L_median": float(meta.loc[meta[col] == lab, "occupancy_L"].median()),
                "selected_occupancy_H_median": float(meta.loc[meta[col] == lab, "occupancy_H"].median()),
                "selected_activity_db_median": float(meta.loc[meta[col] == lab, "activity_db"].median()),
            }
            for lab, d in g
        }

    bui_desc = {b: ("background RF activity (no drone)" if b == "00000" else f"{drone_of(b)} drone, {mode_of(b)}")
                for b in sorted(segs.bui.unique())}
    info = {
        "name": "dronerf",
        "fs": FS,
        "frame_length": a.frame_length,
        "frame_duration_us": a.frame_length / FS * 1e6,
        "channels": ["L", "H"],
        "channel_descriptions": {
            "L": "receiver 1, lower half of the 2.4 GHz ISM band (per the paper), real-valued ADC counts",
            "H": "receiver 2, upper half of the 2.4 GHz ISM band (per the paper), real-valued ADC counts",
        },
        "tasks": tasks,
        "primary_task": "10class",
        "n_frames": int(len(meta)),
        "n_groups": int(meta.group.nunique()),
        "group_definition": "one DroneRF segment (BUI + segment index) = one L file + one H file of 10^7 samples (0.25 s)",
        "class_counts": class_counts,
        "group_counts": group_counts,
        "segments_per_class": {t: seg_lab.groupby(col).size().sort_index().to_dict() for t, col in tasks.items()},
        "class_descriptions": {"10class": bui_desc,
                               "4class": {"background": "no drone", "bebop": "Parrot Bebop", "ar": "Parrot AR Drone",
                                          "phantom": "DJI Phantom 3"},
                               "2class": {"background": "no drone", "drone": "any drone"}},
        "activity": activity,
        "preparation": {
            "script": "experiments/classification-emd-irmf/datasets/dronerf/prepare.py",
            "seed": a.seed,
            "frame_length": a.frame_length,
            "frames_per_class": None if a.frames_per_segment is not None else a.frames_per_class,
            "frames_per_segment": a.frames_per_segment,
            "candidate_grid": "non-overlapping, start = k*frame_length for k = 1 .. floor(1e7/frame_length)-2",
            "activity_statistic": "activity_db_c = 10*log10(var(frame_c - mean) / floor_c); active if max_c >= threshold_db",
            "threshold_db": gate if np.isfinite(gate) else None,
            "noise_floor_var": {"L": float(floor[0]), "H": float(floor[1])},
            "noise_floor_db": {"L": float(floor_db[0]), "H": float(floor_db[1])},
            "occupancy_definition": f"fraction of {SUBWIN}-sample sub-windows with energy >= floor + threshold "
                                    "(10 dB if the gate is disabled)",
            "selection": "uniform without replacement among active candidates of each segment; per-class budget "
                         "spread evenly over segments (water-filling)",
            "channel_pairing": "same start_sample in the L and H file of a segment (not time-aligned, see notes)",
            "frame_order": "sorted by group, then start_sample",
            "dtype_note": "raw integer ADC counts stored exactly as float32; no scaling or DC removal applied",
            "timing_seconds": None,  # filled below
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "source_url": SOURCE_URL,
        "doi": "10.17632/f4c2b4n755.1",
        "license": LICENSE,
        "citation": CITATION,
        "notes": (
            "Raw files: one comma-separated line of 10^7 integer-valued real samples per segment and band "
            "(40 MS/s, 0.25 s). 227 segments: background 41, Bebop 4x21, AR 3x21 + 18 (10111), Phantom 21. "
            "Each BUI is a single recording session (one consistent L/H lag per BUI, also across the background "
            "L1/L2 archive split), so the 10-class label is confounded with the session; "
            "grouping by segment stops frame-level leakage but not session-level cues (received power, "
            "interference). L and H files of one segment are not time-aligned: where both bands carry common "
            "signal (00000, 10001, 10010, 10011, 11000) the envelope cross-correlation peaks at a session-specific "
            "shift of 0.74-3.40 M samples (19-85 ms) that drifts by a few samples per segment; for the other "
            "sessions H is almost pure receiver noise and no shift can be estimated. Channels are therefore paired "
            "at equal start_sample like the dataset authors' code, and should be treated as two views. For Bebop "
            "and AR the H channel is essentially noise (integer-valued, std about 2.4 counts). Signal power differs "
            "by up to 60 dB between classes; keep the per-frame log_scale feature separate as PROTOCOL.md requires. "
            "Frames are activity-gated (label-free): uniform random frames would be noise-only for 69% (10111) "
            "to 95% (background) of frames, and 54% for Phantom. The gated task is 'classify an active frame'."
        ),
    }

    os.replace(tmp_frames, a.out / "frames.npy")
    meta.to_parquet(a.out / "meta.parquet", index=False)
    segs.to_csv(a.out / "segments.csv", index=False)
    # candidate-level activity (all grid frames, before selection) for diagnostics
    np.savez_compressed(
        a.out / "candidates.npz",
        group=np.repeat(segs.group.to_numpy().astype(str), segs.n_candidates.to_numpy()),
        start_sample=np.concatenate([c["starts"] for c in cand]),
        activity_db=np.concatenate([c["act"] for c in cand]).astype(np.float32),
        selected=np.concatenate([np.isin(np.arange(c["starts"].size), picks[k]) for k, c in enumerate(cand)]),
    )
    info["preparation"]["timing_seconds"] = {
        "candidate_statistics": round(t_stats, 1),
        "frame_extraction": round(t_extract, 1),
        "total": round(time.perf_counter() - t_start, 1),
    }
    (a.out / "dataset.json").write_text(json.dumps(info, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({"n_frames": info["n_frames"], "n_groups": info["n_groups"],
                      "class_counts_10": class_counts["10class"], "floor_db": info["preparation"]["noise_floor_db"],
                      "timing": info["preparation"]["timing_seconds"]}, indent=1))


if __name__ == "__main__":
    main()
