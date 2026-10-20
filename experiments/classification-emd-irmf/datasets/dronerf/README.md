# dronerf: drone RF activity in the 2.4 GHz ISM band

12,000 frames of 2048 samples at 40 MS/s (51.2 µs), two channels (L and H band),
cut from all 227 DroneRF segments. Tasks `y_2class` (drone vs background),
`y_4class` (background, Bebop, AR, Phantom) and `y_10class` (BUI code, primary).
Each 10-class label gets 1,200 frames. `group` is the segment (`<BUI>_<index>`), so
frames from one segment never land on both sides of a split.

## Source, license, citation

- Data: DroneRF, Mendeley Data v1, doi:10.17632/f4c2b4n755.1,
  <https://data.mendeley.com/datasets/f4c2b4n755/1>. License CC BY 4.0.
- Paper: M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, A. Erbad. "RF-based drone
  detection and identification using deep learning approaches: An initiative
  towards a large open source drone database." Future Generation Computer Systems
  100 (2019) 86-97, doi:10.1016/j.future.2019.05.007.
- Dataset paper: M. S. Allahham et al. "DroneRF dataset: A dataset of drones for
  RF-based detection, classification and identification." Data in Brief (2019).
- Authors' code: <https://github.com/Al-Sad/DroneRF>.

The download is `raw/f4c2b4n755-1.zip` (4,028,694,325 bytes). It holds 23 RAR v4
archives, not 27. `bsdtar` (libarchive) extracts all of them.

## What the raw files look like

- One CSV per segment and band, holding one comma-separated line of exactly 10^7
  values (0.25 s at 40 MS/s). All 454 files have this length.
- Every value is an integer written as a float (`15.000000`). These are raw ADC
  counts, real-valued, with no I/Q. float32 holds them exactly. Observed range is
  -15,342 to 15,080.
- Files start with up to 12 zero samples and end with up to 3.
- The receiver noise floor is about 2.4 counts RMS in both bands, so noise-only
  stretches take only about 17 distinct integer values per 2048-sample frame.
- Each L file has an H file with the same BUI and index (227 pairs). Background and
  Phantom L files are split across two archives (`_L1`, `_L2`, and the misnamed
  `FR Data_00000_H2.rar`), but the indices continue without gaps or duplicates.

The paper describes the two bands as 40 MHz each. The samples are real at 40 MS/s,
so the one-sided spectrum covers 0 to 20 MHz. The authors' code also keeps only
the one-sided FFT. I did not try to resolve this.

Segments per class, which match the paper's 227:

| BUI | drone | mode | segments | frames |
|---|---|---|---|---|
| 00000 | none (background) | | 41 | 1200 |
| 10000 | Bebop | on and connected | 21 | 1200 |
| 10001 | Bebop | hovering | 21 | 1200 |
| 10010 | Bebop | flying | 21 | 1200 |
| 10011 | Bebop | video recording | 21 | 1200 |
| 10100 | AR Drone | on and connected | 21 | 1200 |
| 10101 | AR Drone | hovering | 21 | 1200 |
| 10110 | AR Drone | flying | 21 | 1200 |
| 10111 | AR Drone | video recording | 18 | 1200 |
| 11000 | Phantom | on and connected | 21 | 1200 |

Coarser tasks: 4-class has background 41, Bebop 84, AR 81 and Phantom 21 segments
(1200/4800/4800/1200 frames). 2-class has 41 background and 186 drone segments
(1200/10800 frames). Every class in every task has at least 18 segments, enough
for 5-fold grouped CV.

## L and H are not simultaneous

I expected the two bands of a segment to be time-aligned. They are not. I
cross-correlated the 256-sample log-energy envelopes of each L/H pair over all
lags (`diagnostics.py`, table `output/.../lh_lag.csv`, figure `fig_spectra_lag.png`).
Five sessions show a sharp peak at a lag that every segment of the session shares:

| BUI | lag (L relative to H) | segments agreeing | median peak r |
|---|---|---|---|
| 00000 | +18.6 ms (+743 k samples) | 28/41 | 0.50 |
| 10001 | -67.8 ms | 19/21 | 0.29 |
| 10010 | -85.0 ms | 17/21 | 0.19 |
| 10011 | -38.3 ms | 21/21 | 0.30 |
| 11000 | +62.0 ms (+2.48 M samples) | 21/21 | 0.62 |

At 16-sample resolution, the lag drifts by 3 to 5 samples per segment index
within a session (11000 goes from 2,480,448 to 2,480,544). That looks like two
free-running receivers started at different times in each session. For the other
five sessions (10000 and all AR) the H band holds almost nothing but noise, so no
lag can be estimated. At lag 0 the envelope correlation is near zero everywhere
except Phantom (0.15).

So aligning H to L is possible in only half of the sessions, and only inside the
overlapping part of each file. I kept C = 2, but paired the channels the way the
authors' code does, at the same `start_sample` of the L and H files. Treat them as
two views of the same segment. Cross-channel phase or coherence features would
measure nothing physical.

## Activity and frame choice

Drone links are bursty. In a typical drone L file, more than half of the
256-sample windows sit at the noise floor. Median burst length is about 100 µs for Bebop and AR
and 70 µs for Phantom. A random 51 µs frame is therefore usually noise.

Share of non-overlapping 2048-sample grid frames that are "active" (frame variance
at least 10 dB above the noise floor), which is what uniform random offsets would
give:

| class | any channel | L | H | per-segment range (any) |
|---|---|---|---|---|
| 00000 background | 4.8% | 3.1% | 1.7% | 2.5-8.7% |
| 10000 Bebop | 21.9% | 20.0% | 2.6% | 9.4-53.9% |
| 10001 Bebop | 14.9% | 13.7% | 1.3% | 7.4-19.9% |
| 10010 Bebop | 15.0% | 14.0% | 1.1% | 11.7-17.5% |
| 10011 Bebop | 14.8% | 13.8% | 1.1% | 11.9-17.3% |
| 10100 AR | 18.2% | 17.1% | 1.3% | 8.2-60.2% |
| 10101 AR | 10.9% | 10.0% | 1.0% | 8.1-17.6% |
| 10110 AR | 9.9% | 9.1% | 0.9% | 7.6-13.8% |
| 10111 AR | 31.1% | 30.3% | 1.2% | 16.4-87.5% |
| 11000 Phantom | 45.9% | 36.2% | 21.2% | 34.6-57.0% |

With uniform offsets, 69% (10111) to 95% (background) of frames would be receiver
noise, and 54% even for Phantom. Those frames look the same in all classes. A comparison of
decompositions on them would mostly measure noise. So the frames are
activity-gated. The rule uses no labels and is the same for every segment:

1. Cut each segment into a grid of non-overlapping 2048-sample candidates, starts
   `k * 2048` for k = 1 .. 4880. The first and last grid frames are dropped
   because of the zero padding.
2. For each candidate and channel compute `activity_db = 10 log10(var / floor)`.
   Here `floor` is the median candidate variance of that channel pooled over all
   227 segments: 5.61 counts^2 (7.48 dB) for L and 5.95 (7.74 dB) for H. Most
   candidates are noise, so this median is the receiver noise level.
3. A candidate is active if `max(activity_db_L, activity_db_H) >= 10 dB`. A
   noise-only 2048-sample frame varies by about 0.14 dB (one standard deviation), so a 10 dB excess means
   a transmission fills a real part of the frame.
4. Draw frames uniformly without replacement from each segment's active candidates
   with `default_rng(20261020)`. Each BUI gets 1,200 frames, spread evenly over its
   segments (29-30 per background segment, 57-58 per segment for 21-segment BUIs,
   66-67 for 10111). The fewest active candidates in any segment is 121
   (background), so no segment hits its cap.

Labels enter only through the per-class budget. The gate itself is label-free.

What the selected frames contain:

- Median activity: background 12 dB, Bebop 52-58 dB, AR 36-40 dB, Phantom 35 dB.
- L is the active channel in 93% of Bebop and AR frames, 78% of Phantom frames and
  65% of background frames. H is active in 8%, 8%, 49% and 36% of them.
- Median L occupancy (share of 128-sample sub-windows 10 dB above the floor) is
  1.0 in every class. The 10th percentile is 0.38 for Bebop, 0.19 for AR and 0 for
  background and Phantom, so some frames hold a burst edge or only H activity.

Why 2048 samples? Bursts last about 70-100 µs, so a 51 µs frame usually sits
inside one burst, and IRMF costs about 0.4 s per channel at this length (PROTOCOL).
`--frame-length` accepts any multiple of 128 up to 4096.

A quick sanity check backs the gate. Log Welch PSD (128 points) of MAD-normalized
frames plus logistic regression, `StratifiedGroupKFold(5)` by segment, gives
balanced accuracy 0.92 / 0.82 / 0.55 (2/4/10-class) on the gated frames. The same
pipeline on uniform frames (`--threshold-db=-inf`) gives 0.76 / 0.55 / 0.30. This is a
throwaway check, not a result of the study.

## Files

`<DATA_ROOT>/dronerf/prepared/`:

- `frames.npy`: float32 (12000, 2, 2048), channels `[L, H]`, raw ADC counts with no
  scaling or DC removal. Frames are sorted by group, then `start_sample`.
- `meta.parquet`: `frame_id`, `group`, `y_2class`, `y_4class`, `y_10class`, `bui`,
  `drone`, `flight_mode`, `segment_index`, `start_sample`, `candidate_index`,
  `activity_db`, `activity_db_L`, `activity_db_H`, `occupancy_L`, `occupancy_H`,
  `rms_L`, `rms_H`, `segment_active_fraction`, `source_file` (L, as
  `DroneRF/<folder>/<archive>.rar::<member>.csv`), `source_file_H`, `cache_file_L`,
  `cache_file_H`.
- `dataset.json`: the PROTOCOL fields plus per-class activity statistics, noise
  floors, and preparation parameters and timings.
- `segments.csv`: one row per segment with candidate, active and selected counts.
- `candidates.npz`: activity of every grid candidate (for diagnostics).

Other locations: `raw/extracted/{zip,rar}` (extracted archives, 44 GB),
`raw/npy/` (float32 cache, 17 GB, plus `conversion_log.json`), and
`work/segment_stats.npz` (envelopes and PSDs). Tables and figures are in
`output/classification-emd-irmf/dronerf/data/`.

## Reproduce

From the worktree root (the raw folder is untrusted, so use `-I`):

```bash
D=/Users/mikhail/Datasets/imf-classification/dronerf/raw
mkdir -p $D/extracted/zip && (cd $D/extracted/zip && bsdtar -xf $D/f4c2b4n755-1.zip)
for f in $D/extracted/zip/DroneRF/*/*.rar; do s=$(basename "$f" .rar); \
  mkdir -p "$D/extracted/rar/$s" && (cd "$D/extracted/rar/$s" && bsdtar -xf "$f"); done
S=experiments/classification-emd-irmf/datasets/dronerf
.venv/bin/python -I $S/convert_csv.py --workers 6    # 454 CSV -> .npy, 200 s
.venv/bin/python -I $S/segment_stats.py --workers 6  # envelopes + PSDs, 16 s
.venv/bin/python -I $S/prepare.py                    # frames, 4 s
.venv/bin/python -I $S/diagnostics.py                # tables + figures, 6 s
(cd experiments/classification-emd-irmf && ../../.venv/bin/python -m emdirmf.validate dronerf)
```

`prepare.py` is deterministic. Two runs produced bit-identical `frames.npy` and
`meta.parquet`. Options: `--frame-length`, `--frames-per-class`,
`--frames-per-segment K` (instead of the class budget), `--threshold-db`, `--seed`.

## Caveats

- Each BUI is one recording session. The L/H lag is constant within a BUI and
  differs between BUIs, and background keeps one lag across its L1/L2 archive
  split. Grouping by segment prevents frame-level
  leakage, but train and test segments of a class always come from the same
  session. Session-level cues (distance, gain, background WiFi at that time) can
  therefore pass for flight-mode or drone-type information. No split can fix this
  with this dataset. This is the main reason to read 10-class results with care.
- The paper's 99.7 / 84.5 / 46.8% come from random splits over 2.5 ms (10^5-sample)
  spectra. Our frames are 49 times shorter, activity-gated, and split by segment,
  so the numbers are not comparable.
- Power differs by up to 60 dB between classes (segment std about 4 counts for
  background against 80-800 for drone L bands). PROTOCOL's per-frame
  normalization and the separate `log_scale` feature handle this. Report both.
- After normalization, SNR still differs: background frames are low-SNR bursts
  (median 12 dB), drone frames are 35-58 dB. A method can win by sensing how much
  noise is in the frame rather than the waveform shape.
- For Bebop and AR the H channel is quantized receiver noise. In 83% of all frames
  the H channel's MAD is exactly 2 counts, so the H `log_scale` is nearly constant
  and `sigma_hat` takes few values. The H channel doubles the decomposition cost
  and mostly helps for Phantom and background. Downstream code can use
  `frames[:, :1]` for an L-only run at half the cost.
- A few segments are much more active than the rest of their class (10000, 10100
  and 10111 have segments at 35-88% activity). Equal per-segment quotas keep them
  from dominating.
- The gate defines the task as "classify an active frame". It does not test
  whether a method can detect a drone in noise-only frames.
