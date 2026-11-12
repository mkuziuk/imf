"""Prepare BirdCLEF 2023 bird-species frames for the EMD vs IMF vs IRMF study.

Source: the Kaggle BirdCLEF 2023 competition training set (xeno-canto recordings
of 264 East African bird species), read from the public Hugging Face mirror
`sebasgar/birdclef-2023` at a pinned revision. Only the metadata CSVs and the
audio files of the selected recordings are fetched, using HTTP range reads into
the 5.3 GB zip, so the full archive is never downloaded.

Pipeline (deterministic for a given --seed):

1. Eligible recordings: rating >= --min-rating, no secondary labels.
2. Species: the top --n-species by eligible recordings, after capping each
   recordist at --max-per-author recordings per species.
3. Recordings: per species, a seeded permutation of the eligible recordings,
   taking at most --max-per-author per recordist, up to --recordings-per-species.
4. Audio: mono, first --max-seconds, resampled to --fs (soxr_hq), mean removed.
5. Frames: candidate starts every --hop samples. Score = mean power of the
   recording after a zero-phase Butterworth high-pass at --score-highpass-hz
   (scoring only; stored frames are not filtered). Greedy picks in descending
   score, no overlap between picked frames, keep only frames within --db-floor dB
   of the recording's loudest frame, at most --max-frames-per-recording.
6. Class balance: per species, round-robin over recordings by energy rank (all
   rank-1 frames, then rank-2, ...) until --frames-per-class; the last partial
   rank level is subsampled with the seeded RNG.

Run from the worktree root (python -I keeps the untrusted raw folder off sys.path):

    .venv/bin/python -I experiments/classification-emd-irmf/datasets/birdclef2023/prepare.py
"""

import argparse
import ast
import json
import os
import re
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

NAME = "birdclef2023"
DEFAULT_DATA_ROOT = "/Users/mikhail/Datasets/imf-classification"
HF_REPO = "sebasgar/birdclef-2023"
HF_REVISION = "b43091aff6033e548e57e71ef3b4a597ee55011a"
ZIP_NAME = "birdclef-2023.zip"
ZIP_PREFIX = "birdclef-2023/"
METADATA_FILES = ("train_metadata.csv", "eBird_Taxonomy_v2021.csv")
SAFE_AUDIO_NAME = re.compile(r"^[a-z0-9]{3,12}/XC\d{1,8}\.ogg$")
KAGGLE_URL = "https://www.kaggle.com/competitions/birdclef-2023"
MIRROR_URL = f"https://huggingface.co/datasets/{HF_REPO}/tree/{HF_REVISION}"
CITATION = (
    "Kahl, S., Denton, T., Klinck, H., Reers, H., Cherutich, F., Glotin, H., Goeau, H., "
    "Vellinga, W.-P., Planque, R., Joly, A. (2023). Overview of BirdCLEF 2023: Automated "
    "bird species identification in Eastern Africa. Working Notes of CLEF 2023, CEUR "
    "Workshop Proceedings 3497, 1934-1942. Data: BirdCLEF 2023 Kaggle competition, " + KAGGLE_URL + "; "
    "recordings from xeno-canto.org (per-recording recordist and XC URL in meta.parquet)."
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    data_root = os.environ.get("IMF_CLS_DATA", DEFAULT_DATA_ROOT)
    parser.add_argument("--data-root", default=data_root)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-species", type=int, default=16)
    parser.add_argument("--min-rating", type=float, default=3.0)
    parser.add_argument("--max-per-author", type=int, default=2,
                        help="max recordings per recordist per species")
    parser.add_argument("--recordings-per-species", type=int, default=80)
    parser.add_argument("--min-recordings", type=int, default=30,
                        help="species with fewer selected recordings are rejected")
    parser.add_argument("--fs", type=int, default=22050)
    parser.add_argument("--frame-length", type=int, default=2048)
    parser.add_argument("--hop", type=int, default=512, help="candidate start spacing")
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--score-highpass-hz", type=float, default=250.0)
    parser.add_argument("--db-floor", type=float, default=15.0)
    parser.add_argument("--max-frames-per-recording", type=int, default=20)
    parser.add_argument("--frames-per-class", type=int, default=750)
    parser.add_argument("--download-threads", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6, help="decode/frame processes")
    parser.add_argument("--select-only", action="store_true",
                        help="print the species/recording selection and exit")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------- download

def _open_remote_zip():
    from huggingface_hub import HfFileSystem

    path = f"datasets/{HF_REPO}@{HF_REVISION}/{ZIP_NAME}"
    handle = HfFileSystem().open(path, "rb", block_size=1 << 20)
    return handle, zipfile.ZipFile(handle)


def _write_atomic(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(payload)
    tmp.replace(path)


def fetch_metadata(raw_dir):
    missing = [name for name in METADATA_FILES if not (raw_dir / name).exists()]
    if not missing:
        return
    handle, archive = _open_remote_zip()
    with handle, archive:
        for name in missing:
            _write_atomic(raw_dir / name, archive.read(ZIP_PREFIX + name))  # CRC-checked


def _fetch_chunk(raw_dir, names):
    handle, archive = _open_remote_zip()
    with handle, archive:
        for name in names:
            # Output paths come from the validated metadata name, never from the archive.
            _write_atomic(raw_dir / "train_audio" / name, archive.read(ZIP_PREFIX + "train_audio/" + name))
    return len(names)


def fetch_audio(raw_dir, names, threads):
    todo = [n for n in names if not (raw_dir / "train_audio" / n).exists()]
    if not todo:
        return 0
    print(f"fetching {len(todo)} audio files from {HF_REPO}@{HF_REVISION[:8]}", flush=True)
    chunks = [todo[i::threads] for i in range(threads)]
    t0 = time.time()
    with ThreadPoolExecutor(threads) as pool:
        done = sum(pool.map(lambda c: _fetch_chunk(raw_dir, c), chunks))
    print(f"fetched {done} files in {time.time() - t0:.0f} s", flush=True)
    return done


# ---------------------------------------------------------------------------- selection

def _n_labels(text):
    try:
        return len(ast.literal_eval(text)) if isinstance(text, str) else 0
    except (ValueError, SyntaxError):
        return -1


def load_tables(raw_dir):
    meta = pd.read_csv(raw_dir / "train_metadata.csv")
    taxonomy = pd.read_csv(raw_dir / "eBird_Taxonomy_v2021.csv")
    taxonomy = taxonomy.drop_duplicates("SPECIES_CODE").set_index("SPECIES_CODE")
    return meta, taxonomy


def select_recordings(meta, taxonomy, args, rng):
    meta = meta.copy()
    meta["author_key"] = meta["author"].fillna("unknown").str.strip().str.lower().str.split().str.join(" ")
    meta["n_secondary"] = meta["secondary_labels"].map(_n_labels)
    eligible = meta[
        (meta["rating"] >= args.min_rating)
        & (meta["n_secondary"] == 0)
        & meta["filename"].map(lambda f: bool(SAFE_AUDIO_NAME.match(str(f))))
    ].sort_values("filename", kind="mergesort")

    capped = (
        eligible.groupby(["primary_label", "author_key"]).size().clip(upper=args.max_per_author)
        .groupby("primary_label").sum()
    )
    ranking = (
        pd.DataFrame({"n": capped})
        .reset_index()
        .sort_values(["n", "primary_label"], ascending=[False, True], kind="mergesort")
    )
    species = sorted(ranking["primary_label"].head(args.n_species))

    chosen = []
    for code in species:
        rows = eligible[eligible["primary_label"] == code]
        per_author, picked = {}, []
        for index in rng.permutation(len(rows)):
            row = rows.iloc[index]
            if per_author.get(row["author_key"], 0) >= args.max_per_author:
                continue
            per_author[row["author_key"]] = per_author.get(row["author_key"], 0) + 1
            picked.append(row)
            if len(picked) == args.recordings_per_species:
                break
        if len(picked) < args.min_recordings:
            raise SystemExit(f"species {code}: only {len(picked)} recordings < {args.min_recordings}")
        chosen.extend(picked)

    selection = pd.DataFrame(chosen).reset_index(drop=True)
    tax = taxonomy.reindex(selection["primary_label"])
    selection["order"] = tax["ORDER1"].to_numpy()
    selection["family"] = tax["FAMILY"].to_numpy()
    selection["species_common"] = tax["PRIMARY_COM_NAME"].to_numpy()
    selection["species_scientific"] = tax["SCI_NAME"].to_numpy()
    if selection["order"].isna().any():
        raise SystemExit("taxonomy lookup failed for some species")
    return selection, ranking


# ------------------------------------------------------------------------------ framing

def frame_recording(path, fs, frame_length, hop, max_seconds, highpass_hz, db_floor, max_frames):
    """Decode one recording and return (frames (F, L) float32, info dict, frame records)."""
    import librosa
    import soundfile as sf

    with sf.SoundFile(path) as handle:
        orig_sr = handle.samplerate
        n_total = handle.frames
        audio = handle.read(min(n_total, int(round(max_seconds * orig_sr))), dtype="float64", always_2d=True)
    audio = audio.mean(axis=1)
    if orig_sr != fs:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=fs, res_type="soxr_hq")
    audio = audio - audio.mean()
    info = {"orig_sr": int(orig_sr), "duration_s": n_total / orig_sr, "analysed_s": audio.size / fs}
    frames, records = select_frames(audio, fs, frame_length, hop, highpass_hz, db_floor, max_frames)
    return frames, info, records


def select_frames(audio, fs, frame_length, hop, highpass_hz, db_floor, max_frames):
    """Label-free frame rule shared by all audio datasets of the study.

    Candidate starts every `hop` samples; score = mean power of the signal after a
    4th-order zero-phase Butterworth high-pass at `highpass_hz`; greedy picks in
    descending score with no overlap between picked frames; stop below the
    loudest score minus `db_floor` dB or after `max_frames`. Returns the unfiltered
    frames (F, L) float32 and one record per frame (start, rank, scores).
    """
    from scipy.signal import butter, sosfiltfilt

    n_starts = (audio.size - frame_length) // hop + 1
    if n_starts <= 0:
        return np.zeros((0, frame_length), np.float32), []

    sos = butter(4, highpass_hz, btype="highpass", fs=fs, output="sos")
    power = np.concatenate([[0.0], np.cumsum(sosfiltfilt(sos, audio) ** 2)])
    starts = np.arange(n_starts) * hop
    score = (power[starts + frame_length] - power[starts]) / frame_length

    order = np.argsort(-score, kind="stable")
    best = score[order[0]]
    floor = best * 10.0 ** (-db_floor / 10.0)
    blocked = np.zeros(n_starts, bool)
    picks = []
    radius = (frame_length - 1) // hop  # starts closer than L samples would overlap
    for index in order:
        if score[index] < floor or score[index] <= 0 or len(picks) == max_frames:
            break
        if blocked[index]:
            continue
        picks.append(int(index))
        blocked[max(0, index - radius): index + radius + 1] = True
    records = [
        {"start_sample": int(starts[i]), "energy_rank": rank,
         "score_db_rel": float(10 * np.log10(score[i] / best)),
         "score_dbfs": float(10 * np.log10(max(score[i], 1e-20)))}
        for rank, i in enumerate(picks)
    ]
    frames = np.stack([audio[r["start_sample"]: r["start_sample"] + frame_length] for r in records]) \
        if records else np.zeros((0, frame_length))
    return frames.astype(np.float32), records


def _frame_job(job):
    path, kwargs = job
    return frame_recording(path, **kwargs)


def balance(candidates, quota, rng):
    """Round-robin over recordings by energy rank; returns the kept candidate rows."""
    kept = []
    for code in sorted(candidates["y_species"].unique()):
        rows = candidates[candidates["y_species"] == code]
        total = 0
        for rank in sorted(rows["energy_rank"].unique()):
            level = rows[rows["energy_rank"] == rank]
            if total + len(level) <= quota:
                kept.append(level)
                total += len(level)
            else:
                pick = np.sort(rng.choice(len(level), quota - total, replace=False))
                kept.append(level.iloc[pick])
                total = quota
            if total == quota:
                break
    return pd.concat(kept).sort_values(["y_species", "recording_id", "start_sample"], kind="mergesort")


# --------------------------------------------------------------------------------- main

def _json_default(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"not JSON serialisable: {type(value)}")


def main(argv=None):
    args = parse_args(argv)
    rng = np.random.default_rng(args.seed)
    base = Path(args.data_root) / NAME
    raw_dir, out_dir = base / "raw", base / "prepared"
    raw_dir.mkdir(parents=True, exist_ok=True)

    fetch_metadata(raw_dir)
    meta, taxonomy = load_tables(raw_dir)
    selection, ranking = select_recordings(meta, taxonomy, args, rng)
    summary = selection.groupby("primary_label").agg(
        n=("filename", "size"), authors=("author_key", "nunique"), order=("order", "first"),
        name=("species_common", "first"))
    print(summary.to_string(), flush=True)
    if args.select_only:
        return 0

    fetch_audio(raw_dir, selection["filename"].tolist(), args.download_threads)

    kwargs = dict(fs=args.fs, frame_length=args.frame_length, hop=args.hop, max_seconds=args.max_seconds,
                  highpass_hz=args.score_highpass_hz, db_floor=args.db_floor,
                  max_frames=args.max_frames_per_recording)
    jobs = [(str(raw_dir / "train_audio" / f), kwargs) for f in selection["filename"]]
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        results = list(pool.map(_frame_job, jobs, chunksize=4))
    print(f"framed {len(jobs)} recordings in {time.time() - t0:.0f} s", flush=True)

    rows, frame_blocks = [], []
    for key in ("orig_sr", "duration_s", "analysed_s"):
        selection[key] = [info[key] for _, info, _ in results]
    selection["n_candidate_frames"] = [len(records) for _, _, records in results]
    for rec_index, ((frames, info, records), (_, rec)) in enumerate(zip(results, selection.iterrows())):
        for local, record in enumerate(records):
            rows.append({
                "rec_index": rec_index, "local_index": local, **record,
                "y_species": rec["primary_label"], "recording_id": Path(rec["filename"]).stem,
            })
        frame_blocks.append(frames)
    candidates = pd.DataFrame(rows)
    kept = balance(candidates, args.frames_per_class, rng)

    offsets = np.cumsum([0] + [len(b) for b in frame_blocks])
    all_frames = np.concatenate(frame_blocks)
    frames = all_frames[offsets[kept["rec_index"].to_numpy()] + kept["local_index"].to_numpy()]
    frames = frames[:, None, :].astype(np.float32)

    rec = selection.iloc[kept["rec_index"].to_numpy()].reset_index(drop=True)
    out = pd.DataFrame({
        "frame_id": np.arange(len(kept)),
        "group": rec["author_key"].to_numpy(),
        "y_species": kept["y_species"].to_numpy(),
        "y_passerine": np.where(rec["order"].to_numpy() == "Passeriformes", "passerine", "non_passerine"),
        "recording_id": kept["recording_id"].to_numpy(),
        "source_file": rec["filename"].to_numpy(),
        "start_sample": kept["start_sample"].to_numpy().astype(np.int64),
        "start_seconds": kept["start_sample"].to_numpy() / args.fs,
        "energy_rank": kept["energy_rank"].to_numpy().astype(np.int16),
        "score_db_rel": kept["score_db_rel"].to_numpy().astype(np.float32),
        "score_dbfs": kept["score_dbfs"].to_numpy().astype(np.float32),
        "frame_rms_dbfs": (10 * np.log10(np.maximum(np.mean(frames[:, 0].astype(np.float64) ** 2, axis=1), 1e-20))).astype(np.float32),
        "species_common": rec["species_common"].to_numpy(),
        "species_scientific": rec["species_scientific"].to_numpy(),
        "order": rec["order"].to_numpy(),
        "family": rec["family"].to_numpy(),
        "author": rec["author"].to_numpy(),
        "license": rec["license"].to_numpy(),
        "rating": rec["rating"].to_numpy().astype(np.float32),
        "call_type": rec["type"].astype(str).to_numpy(),
        "latitude": rec["latitude"].to_numpy().astype(np.float32),
        "longitude": rec["longitude"].to_numpy().astype(np.float32),
        "xc_url": rec["url"].to_numpy(),
        "orig_sr": rec["orig_sr"].to_numpy().astype(np.int32),
        "recording_seconds": rec["duration_s"].to_numpy().astype(np.float32),
    })
    for column in out.columns:
        if out[column].dtype == object:
            out[column] = out[column].astype(str)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "frames.npy", frames)
    out.to_parquet(out_dir / "meta.parquet", index=False)
    selection.drop(columns=["secondary_labels"]).to_csv(out_dir / "recordings.csv", index=False)

    def counts(column):
        return {str(k): int(v) for k, v in out[column].value_counts().sort_index().items()}

    groups_per_class = {str(k): int(v) for k, v in out.groupby("y_species")["group"].nunique().items()}
    recs_per_class = {str(k): int(v) for k, v in out.groupby("y_species")["recording_id"].nunique().items()}
    species_info = (
        out.groupby("y_species")[["species_common", "species_scientific", "order", "family"]].first()
        .to_dict(orient="index")
    )
    info = {
        "name": NAME,
        "fs": args.fs,
        "frame_length": args.frame_length,
        "frame_seconds": args.frame_length / args.fs,
        "channels": ["mono"],
        "tasks": {"species": "y_species", "passerine": "y_passerine"},
        "primary_task": "species",
        "n_frames": int(len(out)),
        "n_groups": int(out["group"].nunique()),
        "group_definition": "recordist (xeno-canto author, case/space-normalised); "
                            "recording_id gives the looser per-recording grouping",
        "n_recordings": int(out["recording_id"].nunique()),
        "class_counts": {"species": counts("y_species"), "passerine": counts("y_passerine")},
        "groups_per_class": {"species": groups_per_class},
        "recordings_per_class": {"species": recs_per_class},
        "species": species_info,
        "preparation": {
            "seed": args.seed,
            **{k: v for k, v in vars(args).items() if k not in ("data_root", "seed", "select_only")},
            "eligibility": "rating >= min_rating and no secondary labels",
            "resampler": "librosa.resample(res_type='soxr_hq') after mono mean of channels",
            "stored_frames": "resampled audio, recording mean removed, no filtering, float32 full-scale "
                             "units (|x| can exceed 1 slightly: resampler overshoot on clipped input)",
            "frame_score": "mean power after 4th-order zero-phase Butterworth high-pass at score_highpass_hz",
            "species_rank_top": ranking.head(args.n_species + 4).to_dict(orient="records"),
        },
        "source_url": KAGGLE_URL,
        "source_mirror": MIRROR_URL,
        "source_archive": f"{HF_REPO}@{HF_REVISION}/{ZIP_NAME}",
        "license": "CC BY-NC-SA 4.0 (BirdCLEF 2023 competition data); per-recording xeno-canto "
                   "licenses in meta.parquet column 'license' (CC BY-NC-SA 4.0/3.0, CC BY-SA 4.0/3.0). "
                   "Attribution: recordist + XC URL per frame. Non-commercial research use.",
        "citation": CITATION,
        "notes": (
            "Weakly labelled: the label is the recording's primary species; frames are the "
            "loudest non-overlapping windows above 250 Hz and can still contain background or "
            "other species. Recordings with listed secondary species were excluded. Recordists "
            "upload from many countries (not only Kenya); devices and pre-filtering differ by "
            "recordist, which is why group = recordist."
        ),
    }
    with open(out_dir / "dataset.json", "w") as handle:
        json.dump(info, handle, indent=2, default=_json_default)
    print(json.dumps({k: info[k] for k in ("n_frames", "n_groups", "n_recordings", "class_counts")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
