# 60 GHz radar: UAS and birds (Kang, Forstén, Semkin, Rangan)

Status as of 2026-10-20:

- **Full release (IEEE DataPort `Data.zip`, 15.21 GB): not downloaded.** The files
  sit behind an IEEE login. The landing page exposes no public file URL. The
  only S3 reference in its HTML is the uploader help link. No Kaggle, Hugging
  Face, Zenodo, or other mirror turned up.
- **Public subset: downloaded and verified.** The first author's GitHub repository
  publishes a cropped copy of the same measurement campaign: 97 recordings,
  each a complex64 cube of shape (1000, 20, 30). It lives under
  `<DATA_ROOT>/uas_birds_60ghz/raw/github_sk8053_drone_bird_detection_with_radar/`.
  All 160 downloaded files match their git blob SHA-1 at the pinned commit.
  That is enough to write and test `prepare.py` now. See "Public subset" below.

## Sources

| What | Where |
|---|---|
| Dataset landing page | https://ieee-dataport.org/open-access/millimeter-wave-60-ghz-radar-measurements-uas-and-birds (DOI [10.21227/c8r5-6h79](https://dx.doi.org/10.21227/c8r5-6h79)) |
| Paper | S. Kang, H. Forstén, P. Skrimponis, M. Ezuma, M. Mezzavilla, I. Guvenc, S. Rangan, V. Semkin, "Millimeter Wave Radar Measurements: Distinguishing UAS and Birds Based on 60 GHz micro-Doppler Signatures," IEEE VTC2024-Fall, doi:[10.1109/VTC2024-Fall63153.2024.10757735](https://doi.org/10.1109/VTC2024-Fall63153.2024.10757735). Closed access, no arXiv version found. |
| Authors' code and data subset | https://github.com/sk8053/drone_bird_detection_with_radar, commit `0e991a1b99f8eac2acab420254f656dc434d7f38` (2024-12-14). No license file. |
| Third-party use | M. Larrat, C. Sales, "Classification of Flying Drones Using Millimeter-Wave Radar: Comparative Analysis of Algorithms Under Noisy Conditions," *Sensors* 25(3):721, 2025, doi:[10.3390/s25030721](https://doi.org/10.3390/s25030721) ([PMC full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC11821064/)) |
| Probable radar hardware (inferred from authorship, not stated in any source found) | H. Forstén et al., "Scalable 60 GHz FMCW Frequency-Division Multiplexing MIMO Radar," IEEE TMTT 68(7):2845-2855, 2020, doi:[10.1109/TMTT.2020.2980521](https://doi.org/10.1109/TMTT.2020.2980521) |

License: the DataPort page states none. Its "Open Access" files come under
DataPort's terms of use. The GitHub repository has no license, so treat the
subset as all rights reserved. Use it for research, cite the DataPort DOI and
the VTC paper, and do not redistribute either copy.

Citation (DataPort IEEE style): Seongjoon Kang, Henrik Forsten, Vasilii Semkin,
Sundeep Rangan, "Millimeter Wave 60 GHz Radar Measurements: UAS and Birds,"
IEEE Dataport, March 15, 2024, doi:10.21227/c8r5-6h79.

## User action: getting the full release

1. Open https://ieee-dataport.org/open-access/millimeter-wave-60-ghz-radar-measurements-uas-and-birds.
   Log in with a free IEEE account ("LOGIN TO ACCESS DATASET FILES"). The page says
   open-access files are available to every logged-in user. It also carries a
   generic "requires an IEEE DataPort Subscription" banner, which does not seem
   to apply to open-access entries.
2. Download the single file **`Data.zip` (15.21 GB)**. DataPort usually serves a
   time-limited signed S3 link. To resume an interrupted transfer, copy that link
   from the browser and run
   `curl -L -C - -o Data.zip "<signed link>"`. Generate a fresh link if it expires.
3. Put it at
   `/Users/mikhail/Datasets/imf-classification/uas_birds_60ghz/raw/dataport/Data.zip`.
4. Check the size, then list the contents before extracting:
   `unzip -l Data.zip | head -50`. Extract into
   `raw/dataport/extracted/`. Never run anything from that directory.
5. Tell the data agent. `prepare.py` should accept either the DataPort raw cubes
   or the GitHub crops (below).

## What is known about the format

Sources: the authors' code (`cut_drone_bird.py`, `extract_angles.py`,
`matlab_files/plot_raw_and_cut_images.py`, the notebooks, and `README.docx`), the
dataset description in Larrat & Sales (2025, Section 3.4), and direct
inspection of the GitHub pickles.

**Campaign.** The data comes from one indoor session. The notebook path is
`2021-05-19_Drone_vs_bird_detection_indoor/Radar_measurements_60GHz_Mavic+P3P+bird`.
The targets are a DJI Mavic (filenames say `mavik`), a DJI Phantom 3 Pro (`p3p`),
a bionic (flapping-wing) toy bird (`bird`), and one empty pole (`pole`). Targets
were mounted on a turntable or tripod in front of absorbers.

**Filenames encode the conditions** (`<target>_<pose>_<state>.pkl`):

- Turntable pose: aspect angle `0, 22.5, ..., 180` degrees (9 values).
- Tripod pose: `tripod_az_{0,45,90}_tilt_{-45,0,+45}` (9 values).
- Drone state: `still` (propellers stopped) or `rotation` (propellers spinning).
  Two extra Mavic pairs exist: `_no_absorber` and `_old_absorber_configuration`.
- Bird state: `still` or a flapping setting `5pct`, `50pct`, `100pct`
  (presumably a percentage of flapping drive; not documented).

**Raw DataPort cube (inferred, verify on arrival).** One `.pkl` per recording,
holding a pickled numpy complex ndarray of shape **(1000, 128, 168) = (time
stamps, angle bins, range bins)**. The evidence:

- `plot_raw_and_cut_images.py` plots `data[500].T` (168 x 128). It labels the
  128-bin axis as angle from -90 to +90 degrees and the 168-bin axis as
  distance from about 1.0 to 5.0 m (about 2.5 cm per bin).
- Larrat & Sales describe the cube as 1000 x 128 x 168. They call the last axis
  "functional attributes ... real and imaginary parts, amplitude, phase". That
  conflicts with the authors' code, which treats it as range. Trust the code.
- The dtype is probably complex64, because the crops are complex64 and
  `cut_drone_bird.py` slices without casting. That gives about 172 MB per
  recording. Ninety-five recordings come to about 15.2 GiB, which matches the
  15.21 GB `Data.zip`. The archive may therefore hold roughly the same 95-97
  recordings as the GitHub subset, uncropped. This is a size-based inference only.
- A 128-bin angle axis is consistent with a zero-padded angle FFT over the
  4 TX x 8 RX virtual array of the VTT radar. The 2.5 cm bins are consistent with
  a 2x zero-padded range FFT over a 3 GHz sweep (5 cm resolution). Both
  hardware links are inferences.

**Slow-time sampling rate: not documented.** The authors' figures label the
1000 time stamps as 0-1000 ms, which implies about 1 kHz. A Microwave Journal
article on the VTT 8x8 radar ([Kiuru & Forstén](https://microwavejournal.com/articles/35244-design-of-a-mmwave-mimo-radar))
quotes 200 frames/s imaging and uses a 1 ms chirp in its simulation example. Treat `fs` as unknown. Store
`fs_assumed=1000` in `dataset.json` with a note, and keep every feature
dimensionless (per-sample frequencies). At 60 GHz (wavelength 5 mm), a 1 kHz
frame rate leaves an unambiguous Doppler of only +/- 0.25 m/s. Propeller-blade
micro-Doppler is therefore heavily aliased in these series. That is fine for
classification but rules out physical Doppler interpretation.

**GitHub crop.** `cut_drone_bird.py` computes `data[:, 60:80, :][:, :, 60:90]`,
i.e. angle bins 60-79 and range bins 60-89 (about 2.3-3.05 m if the plot axis is
exact), so the target fills the window. With the DataPort cube, apply the same
crop to one file and check that it reproduces the GitHub crop exactly. That
confirms the inferred layout.

**Values.** These are uncalibrated complex amplitudes. Median |x| per recording
is 3.6e3-12.8e3, and the strongest cell's mean |x| is 7e4-8.6e5. Static
returns dominate. Remove the per-cell complex mean (static clutter) before
looking at micro-Doppler.

## Public subset (downloaded)

Location: `<DATA_ROOT>/uas_birds_60ghz/raw/github_sk8053_drone_bird_detection_with_radar/`

- `data/cut_data/cut_*.pkl`: 97 recordings, each **complex64 (1000, 20, 30)**,
  C-order little-endian. The pickles are protocol 4 and reference only
  `numpy.core.multiarray._reconstruct`, `numpy.ndarray`, and `numpy.dtype`.
  `cut_bird_data_total.pkl` (153 bytes) is an empty `(38000, 0, 0)` placeholder;
  skip it.
- `data/sampled_phase_data/*_angle_moving_*.txt` and `data/*_angle_*.txt`: 1000
  phase values in degrees per file, taken at the authors' hand-picked
  propeller/wing cell. The `.mat` files hold their spectrograms.
- `MANIFEST_git_blobs.tsv`: path, size, and git blob SHA-1 for every file
  (verified). `README.docx`: the authors' code notes.

Recordings by class and state:

| target | moving | still | total |
|---|---|---|---|
| bird | 20 (`50pct` x17, `100pct` x2, `5pct` x1) | 18 | 38 |
| mavik | 20 (`rotation`, incl. 2 absorber variants) | 20 (incl. 2 absorber variants) | 40 |
| p3p | 9 (`rotation`) | 9 | 18 |
| pole | - | 1 | 1 |

Quick check: the ratio of the strongest cell's dynamic std to its static
level separates moving from still in most recordings. Still recordings are
mostly below 0.025 and moving ones mostly above 0.035. The exceptions are
`mavik_0_still` at 0.048 and the bird tripod `az_0` flapping runs at
0.022-0.027. Motion state is therefore close to trivial. The useful question
is bird vs drone among *moving* recordings.

Safe loading. `pickle.load` can execute code, so never call it on these
files. Use a whitelist unpickler (Python docs, "Restricting Globals"):

```python
import pickle
import numpy as np

_ALLOWED = {("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct"),
            ("numpy", "ndarray"), ("numpy", "dtype")}

class _NdarrayOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) not in _ALLOWED:
            raise pickle.UnpicklingError(f"blocked global {module}.{name}")
        if name == "_reconstruct":
            return np._core.multiarray._reconstruct  # numpy >= 2
        return getattr(np, name)

def load_ndarray(path):
    with open(path, "rb") as f:
        arr = _NdarrayOnly(f).load()
    assert isinstance(arr, np.ndarray)
    return arr
```

Run it with `.venv/bin/python -I` from outside the data directory.

The repo's `utils/download_data.py` also points to two public Google Drive files,
`bird_data_total.pkl` and `mavik_data_total.pkl`. These are the crops
concatenated, with recording boundaries lost: (38000, 20, 30) and
(46000, 20, 30). They are useless for grouped CV and were not downloaded. The
repo also contains a committed server credential file. It was not opened or
used.

## Published protocols on this data (all leak across recordings)

- **Kang et al. 2024 (repo code).** Binary bird vs "mavik". The non-bird
  array also absorbs P3P and pole files through an `else` branch. A CNN runs on
  |x| 20 x 30 maps, and a ConvLSTM on phase over 5 consecutive time stamps.
  Frames from all recordings are randomly permuted and then split 70/30 (CNN),
  30/70, or 80/20 (ConvLSTM). Validation accuracy is about 0.99 after one epoch.
  Neighbouring time stamps from one recording land on both sides of the split,
  so these numbers measure interpolation, not generalisation. Robustness is
  shown against added noise (SNR -5 to 30 dB) with a random attenuation
  augmentation in [0.05, 1].
- **Larrat & Sales 2025.** Three classes (Mavic, P3P, bird). LSTM, GRU,
  Conv1D, and Transformer on amplitude and phase, plus a "multimodal
  Transformer" that adds skewness and kurtosis. Synthetic white, Pareto,
  impulsive, and multipath noise is added. Under white noise the Transformer
  reaches per-class AUC 0.92-0.98, against about 0.5 for the others. They hold
  out a random 10% for validation and do not group by recording. Their data is
  "available upon request".
- **Sirisilla et al., INDICON 2025** ("Multi-Channel 1D-CNN Architecture for
  Drone vs Bird Classification Using mmWave Radar ADC Data",
  doi:[10.1109/INDICON68490.2025.11392899](https://doi.org/10.1109/INDICON68490.2025.11392899)).
  Reports 96% for bird vs drone on "raw 60 GHz ADC data". It cites the VTC
  paper, but this release holds processed cubes rather than ADC samples, so
  their source dataset is unconfirmed.

No grouped (leave-recording-out) result exists, so ours would be the first.

## Recommendations for `prepare.py`

- **Group:** one recording (`.pkl` file). Turntable `0_100pct`/`0_50pct` and
  tripod `az_0_tilt_0_{100pct,50pct}` share a pose. A stricter
  `group_pose = target + pose` is also worth storing, so that CV can optionally
  keep one physical setup on one side.
- **Tasks:**
  - `y_bird_drone`: bird vs drone on moving recordings only (20 vs 29). Make
    this the primary task.
  - `y_target3`: bird / mavic / p3p, moving only (20 / 20 / 9). P3P has 9
    groups, enough for 5-fold grouped CV but thin.
  - `y_motion`: still vs moving. Optional and near-trivial; useful as a
    sanity check.
  - Drop `pole`. Keep the absorber variants but flag them in `meta`.
- **1-D signal:** the slow-time series at a range-angle cell. Use C = 2
  channels, real and imaginary after removing the complex mean; amplitude and
  unwrapped phase is an alternative. The authors found phase the more
  informative channel. Pick cells within each recording, for example the top-k
  cells by dynamic std. That uses only the recording itself and never crosses
  recordings.
- **Frame length:** each recording has only 1000 time stamps. Use either
  L = 1000 per cell (one frame per cell) or windows of L = 512 with hop 128
  (about 4 windows per cell). With k = 16 cells per recording, that is
  roughly 16-64 frames per recording, or 0.8k-3k frames for the moving
  subset. Frames from one recording are strongly correlated, so cap them and
  report group counts. The full DataPort cube (128 x 168 cells) would allow
  better cell selection but not more independent groups.
- **Confounds:** the background level differs by session and target. Median
  |x| is about 3.7-5.3e3 for bird, 5.5-9e3 for mavic, and 7.6-12.8e3 for p3p.
  Per-frame robust normalisation and the separate `log_scale` feature from the
  protocol are therefore essential. Also report a "noise-floor only" control
  (features from a target-free cell) to show that class information is not
  just session gain.
