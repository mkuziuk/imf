"""Prepare the ESC-50 "Animals" classes as frames for the EMD vs IMF vs IRMF study.

Source: ESC-50 (Piczak 2015), GitHub archive of karolpiczak/ESC-50 (master). The
10 classes of the "Animals" major category (dog, rooster, pig, cow, frog, cat,
hen, insects, sheep, crow), 40 five-second clips each, 44.1 kHz mono, all derived
from Freesound recordings. The zip is read in place (members are never extracted).

Frame selection is the same rule as birdclef2023 (imported from its prepare.py):
mono, resampled to --fs, mean removed, top-energy non-overlapping frames scored
above a 250 Hz high-pass, within --db-floor dB of the clip's loudest frame, at
most --max-frames-per-recording per clip. ESC-50 pads short clips with digital
silence, so leading/trailing runs of exact zeros are trimmed before resampling.
Classes are balanced to min(--frames-per-class, smallest class) by the same
round-robin over energy ranks.

Group = Freesound uploader parsed from the archive's LICENSE attribution list
(stricter than the ESC-50 fold rule, which keeps clips of one source file
together); src_file and esc50_fold are kept for the looser grouping.

    .venv/bin/python -I experiments/classification-emd-irmf/datasets/esc50_animals/prepare.py
"""

import argparse
import importlib.util
import io
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

NAME = "esc50_animals"
DEFAULT_DATA_ROOT = "/Users/mikhail/Datasets/imf-classification"
ARCHIVE = "ESC-50-master.zip"
ARCHIVE_URL = "https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master"
SOURCE_URL = "https://github.com/karolpiczak/ESC-50"
ANIMALS = ("dog", "rooster", "pig", "cow", "frog", "cat", "hen", "insects", "sheep", "crow")
BIRDS = ("rooster", "hen", "crow")
SAFE_WAV = re.compile(r"^\d-\d{1,7}-[A-Z]-\d{1,2}\.wav$")
LICENSE_LINE = re.compile(
    r"^- \[(?P<clip>\d-\d+-[A-Z])\.ogg\]:\s+clip derived from .* \((?P<url>https?://[^ )]*freesound[^ )]*)\) "
    r"by (?P<user>.+?) \[(?P<lic>[^\]]+)\]\s*$")
CITATION = (
    "Piczak, K. J. (2015). ESC: Dataset for Environmental Sound Classification. Proceedings of "
    "the 23rd ACM International Conference on Multimedia, 1015-1018. doi:10.1145/2733373.2806390. "
    "Data: https://github.com/karolpiczak/ESC-50, doi:10.7910/DVN/YDEPUT."
)


_SHARED = None


def _shared():
    """Load the frame-selection rule from ../birdclef2023/prepare.py (single source of truth)."""
    global _SHARED
    if _SHARED is None:
        path = Path(__file__).resolve().parents[1] / "birdclef2023" / "prepare.py"
        spec = importlib.util.spec_from_file_location("birdclef2023_prepare", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SHARED = module
    return _SHARED


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data-root", default=os.environ.get("IMF_CLS_DATA", DEFAULT_DATA_ROOT))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fs", type=int, default=22050)
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512)
    parser.add_argument("--score-highpass-hz", type=float, default=250.0)
    parser.add_argument("--db-floor", type=float, default=15.0)
    parser.add_argument("--max-frames-per-recording", type=int, default=20)
    parser.add_argument("--frames-per-class", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    return parser.parse_args(argv)


def read_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        root = names[0].split("/")[0] + "/"
        meta = pd.read_csv(io.BytesIO(archive.read(root + "meta/esc50.csv")))
        license_text = archive.read(root + "LICENSE").decode("utf-8", "replace")
        meta = meta[meta["category"].isin(ANIMALS)]
        meta = meta[meta["filename"].map(lambda f: bool(SAFE_WAV.match(f)))]
        audio = {f: archive.read(root + "audio/" + f) for f in meta["filename"]}
    return meta.sort_values("filename", kind="mergesort").reset_index(drop=True), license_text, audio


def parse_license(text):
    rows = []
    for line in text.splitlines():
        match = LICENSE_LINE.match(line.strip())
        if match:
            rows.append(match.groupdict())
    table = pd.DataFrame(rows).drop_duplicates("clip")
    return table.set_index("clip")


def frame_clip(job):
    payload, kwargs = job
    import librosa
    import soundfile as sf

    shared = _shared()
    audio, sr = sf.read(io.BytesIO(payload), dtype="float64", always_2d=True)
    audio = audio.mean(axis=1)
    nonzero = np.flatnonzero(audio != 0.0)
    trimmed = (0, 0)
    if nonzero.size:
        trimmed = (int(nonzero[0]), int(audio.size - nonzero[-1] - 1))
        audio = audio[nonzero[0]: nonzero[-1] + 1]
    else:
        audio = audio[:0]
    info = {"orig_sr": int(sr), "trim_lead_s": trimmed[0] / sr, "trim_tail_s": trimmed[1] / sr}
    if audio.size and sr != kwargs["fs"]:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=kwargs["fs"], res_type="soxr_hq")
    audio = audio - audio.mean() if audio.size else audio
    info["analysed_s"] = audio.size / kwargs["fs"]
    frames, records = shared.select_frames(audio, **kwargs)
    return frames, info, records


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    base = Path(args.data_root) / NAME
    raw_path, out_dir = base / "raw" / ARCHIVE, base / "prepared"
    if not raw_path.exists():
        raise SystemExit(f"download {ARCHIVE_URL} to {raw_path} first (see README)")
    shared = _shared()

    meta, license_text, audio = read_archive(raw_path)
    lic = parse_license(license_text)
    meta["clip"] = meta["filename"].str.rsplit("-", n=1).str[0]
    meta["uploader"] = lic.reindex(meta["clip"])["user"].to_numpy()
    meta["clip_license"] = lic.reindex(meta["clip"])["lic"].to_numpy()
    meta["freesound_url"] = lic.reindex(meta["clip"])["url"].to_numpy()
    missing = meta["uploader"].isna()
    if missing.any():
        print(f"warning: {int(missing.sum())} clips without a LICENSE attribution; group = src_file")
    meta["group"] = np.where(missing, "src:" + meta["src_file"].astype(str),
                             "fs:" + meta["uploader"].fillna("").str.strip().str.lower())

    kwargs = dict(fs=args.fs, frame_length=args.frame_length, hop=args.hop,
                  highpass_hz=args.score_highpass_hz, db_floor=args.db_floor,
                  max_frames=args.max_frames_per_recording)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(frame_clip, [(audio[f], kwargs) for f in meta["filename"]], chunksize=8))
    print(f"framed {len(results)} clips in {time.time() - t0:.0f} s", flush=True)
    for key in ("orig_sr", "trim_lead_s", "trim_tail_s", "analysed_s"):
        meta[key] = [info[key] for _, info, _ in results]
    meta["n_candidate_frames"] = [len(r) for _, _, r in results]

    rows, blocks = [], []
    for rec_index, ((frames, _, records), (_, rec)) in enumerate(zip(results, meta.iterrows())):
        for local, record in enumerate(records):
            rows.append({"rec_index": rec_index, "local_index": local, **record,
                         "y_species": rec["category"], "recording_id": rec["clip"]})
        blocks.append(frames)
    candidates = pd.DataFrame(rows)
    available = candidates.groupby("y_species").size()
    quota = int(min(args.frames_per_class, available.min()))
    kept = shared.balance(candidates, quota, rng)

    offsets = np.cumsum([0] + [len(b) for b in blocks])
    frames = np.concatenate(blocks)[offsets[kept["rec_index"].to_numpy()] + kept["local_index"].to_numpy()]
    frames = frames[:, None, :].astype(np.float32)
    rec = meta.iloc[kept["rec_index"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({
        "frame_id": np.arange(len(kept)),
        "group": rec["group"].to_numpy(),
        "y_animal": rec["category"].to_numpy(),
        "y_bird": np.where(rec["category"].isin(BIRDS), "bird", "non_bird"),
        "recording_id": rec["clip"].to_numpy(),
        "source_file": rec["filename"].to_numpy(),
        "start_sample": kept["start_sample"].to_numpy().astype(np.int64),
        "start_seconds": kept["start_sample"].to_numpy() / args.fs,
        "trim_lead_s": rec["trim_lead_s"].to_numpy().astype(np.float32),
        "energy_rank": kept["energy_rank"].to_numpy().astype(np.int16),
        "score_db_rel": kept["score_db_rel"].to_numpy().astype(np.float32),
        "score_dbfs": kept["score_dbfs"].to_numpy().astype(np.float32),
        "frame_rms_dbfs": (10 * np.log10(np.maximum(
            np.mean(frames[:, 0].astype(np.float64) ** 2, axis=1), 1e-20))).astype(np.float32),
        "src_file": rec["src_file"].astype(str).to_numpy(),
        "esc50_fold": rec["fold"].to_numpy().astype(np.int8),
        "esc10": rec["esc10"].astype(bool).to_numpy(),
        "uploader": rec["uploader"].fillna("").astype(str).to_numpy(),
        "license": rec["clip_license"].fillna("").astype(str).to_numpy(),
        "freesound_url": rec["freesound_url"].fillna("").astype(str).to_numpy(),
    })

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "frames.npy", frames)
    out.to_parquet(out_dir / "meta.parquet", index=False)
    meta.to_csv(out_dir / "recordings.csv", index=False)

    def counts(column):
        return {str(k): int(v) for k, v in out[column].value_counts().sort_index().items()}

    info = {
        "name": NAME,
        "fs": args.fs,
        "frame_length": args.frame_length,
        "frame_seconds": args.frame_length / args.fs,
        "channels": ["mono"],
        "tasks": {"animal": "y_animal", "bird": "y_bird"},
        "primary_task": "animal",
        "n_frames": int(len(out)),
        "n_groups": int(out["group"].nunique()),
        "group_definition": "Freesound uploader from the ESC-50 LICENSE attribution list "
                            "(fallback src_file); src_file / esc50_fold give the ESC-50 grouping",
        "n_recordings": int(out["recording_id"].nunique()),
        "class_counts": {"animal": counts("y_animal"), "bird": counts("y_bird")},
        "groups_per_class": {"animal": {str(k): int(v) for k, v in
                                        out.groupby("y_animal")["group"].nunique().items()}},
        "preparation": {
            "seed": args.seed, "quota_per_class": quota,
            **{k: v for k, v in vars(args).items() if k not in ("data_root", "seed")},
            "classes": list(ANIMALS), "bird_classes": list(BIRDS),
            "frame_rule": "shared with birdclef2023 (select_frames, balance)",
            "silence_trim": "leading/trailing exact-zero runs removed before resampling",
            "resampler": "librosa.resample(res_type='soxr_hq')",
            "stored_frames": "resampled audio, clip mean removed, no filtering, float32",
            "available_frames_per_class": {str(k): int(v) for k, v in available.items()},
        },
        "source_url": SOURCE_URL,
        "source_archive": ARCHIVE_URL,
        "license": "CC BY-NC 3.0 (ESC-50 as a whole); per-clip Freesound licenses (CC0 / CC BY / "
                   "CC BY-NC) and uploader attribution in meta.parquet",
        "citation": CITATION,
        "notes": (
            "400 clips (40 per class), 5 s each; many contain one or a few short events, so frames "
            "per clip vary. Several clips per class come from the same Freesound source file "
            "(src_file) or uploader; group = uploader keeps both together. Bird task: rooster, "
            "hen, crow vs the 7 other animals."
        ),
    }
    with open(out_dir / "dataset.json", "w") as handle:
        json.dump(info, handle, indent=2, default=shared._json_default)
    print(json.dumps({k: info[k] for k in ("n_frames", "n_groups", "n_recordings", "class_counts",
                                           "groups_per_class")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
