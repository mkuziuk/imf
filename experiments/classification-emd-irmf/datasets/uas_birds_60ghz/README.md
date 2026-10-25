# 60 GHz radar: UAS and birds (Kang, Forstén, Semkin, Rangan)

Status as of 2026-10-20: the authorized NumPy-only conversion and deterministic
preparation have completed. The public subset supplies 49 moving recordings,
784 I/Q frames and 45 target-plus-pose CV groups. The full DataPort release has
not been downloaded. Numerical results and reproducibility commands are in
[`REPORT.md`](../../../../output/classification-emd-irmf/uas_birds_60ghz/classical/REPORT.md).

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
5. Scan, convert and use the explicitly untested DataPort preparation path below.
   Verify the inferred cube axes and crop equality before interpreting results.

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
quotes 200 frames/s imaging and uses a 1 ms chirp in its simulation example.
`dataset.json` therefore stores `fs: null` and `fs_assumed: 1000`, with an explicit
uncertainty note. All frequency features use cycles/sample. If the slow-time
sampling rate were 1 kHz, Nyquist would be 500 Hz and the monostatic unambiguous
velocity at 60 GHz would be approximately +/- 1.25 m/s. This is conditional, not
a measured PRF or a claim that blade velocities are resolved.

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

## Authorized loader and conversion

The round-2 authorization permits a strict allow-list unpickler after a
nonexecuting opcode scan. [`convert_pickles.py`](convert_pickles.py) subclasses
`pickle.Unpickler` and overrides `find_class`. The only allowed pairs are:

- `numpy.core.multiarray`, `_reconstruct`
- `numpy._core.multiarray`, `_reconstruct`
- `numpy`, `ndarray`
- `numpy`, `dtype`

Both reconstruct names map directly to `numpy._core.multiarray._reconstruct`
for NumPy 2. Every other pair raises `pickle.UnpicklingError`. Persistent IDs,
extension opcodes, legacy constructors and external buffers are rejected.
After loading, the converter accepts only ndarrays, flat dict/list/tuple
containers of ndarrays or plain scalar/string values. It rejects every array
whose `dtype.hasobject` is true, including container members. No downloaded
Python code is imported or executed.

The `pickletools.genops` scan executes no pickle instructions. It symbolically
tracks stack strings and memo references to resolve `GLOBAL` and `STACK_GLOBAL`;
all 98 local files reference exactly `numpy.core.multiarray._reconstruct`,
`numpy.ndarray`, and `numpy.dtype`. The runtime allow-list is a second check.
This restricts the accepted inputs to the audited NumPy construction path;
`-I` alone would not make arbitrary pickle input safe. It is not a general
sandbox or a guarantee against resource exhaustion in arbitrary future files.

Run from the repository root, outside `raw/`, with the absolute worktree Python:

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python -I \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/uas_birds_60ghz/convert_pickles.py --scan-only
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python -I \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/uas_birds_60ghz/convert_pickles.py
```

Each source is converted once to `converted/<stem>.npy`, or numeric-only `.npz`
for a container. `converted/manifest.json` records source-relative paths,
source SHA-256, globals, output SHA-256, shape and dtype. A rerun verifies
hashes and reuses completed exports without unpickling. Conversion included the
empty placeholder, with shape `(38000, 0, 0)` and dtype complex128; preparation
excludes it. The 97 real cubes are complex64 `(1000, 20, 30)`.

Re-runnable opcode-scan code, including for an extracted full release:

```python
# Run with /absolute/worktree/.venv/bin/python -I from outside raw/.
import importlib.util
from pathlib import Path

converter = Path('/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/uas_birds_60ghz/convert_pickles.py')
spec = importlib.util.spec_from_file_location('numeric_converter', converter)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)  # local audited converter, never raw code
source = Path('/Users/mikhail/Datasets/imf-classification/uas_birds_60ghz/raw/dataport/extracted')
for path in sorted(source.rglob('*.pkl')):
    globals_used = module.scan_globals(path.read_bytes())
    print(path.relative_to(source), globals_used)
```

The full symbolic implementation is `scan_globals` in the linked converter.
The equivalent CLI is `convert_pickles.py --scan-only --source-root <extracted>`.
An unexpected global or unsupported opcode stops before conversion.

Everything downstream reads only numeric exports using `allow_pickle=False`.
Preparation also checks the converted-file SHA-256 against the manifest.

The repo's `utils/download_data.py` also points to two public Google Drive files,
`bird_data_total.pkl` and `mavik_data_total.pkl`. These are the crops
concatenated, with recording boundaries lost: (38000, 20, 30) and
(46000, 20, 30). They are useless for grouped CV and were not downloaded. The
repo also contains a committed server credential file. It was not opened or
used.

## Published protocols on this data

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

The protocols above do not establish performance on held-out recordings.
This study uses target-plus-pose groups; no claim of literature priority is made.

## Prepared data

[`prepare.py`](prepare.py) uses moving targets only and removes the complex
mean independently at each cell. It ranks the 600 cropped cells by mean squared
complex fluctuation within that recording and retains the top 16. Ties resolve
by flattened cell index. Selection uses no class labels or other recordings.
Each chosen cell supplies one complete 1000-sample frame, with I and Q as two
float32 channels. Shared decomposition code then applies the protocol's median
and MAD normalization and preserves `log_scale` separately. No padding,
resampling or random windows are used. The recorded seed is 0; selection itself
has no random step.

| task/class | frames | recordings | CV groups |
|---|---:|---:|---:|
| bird_drone: bird | 320 | 20 | 18 |
| bird_drone: uas | 464 | 29 | 27 |
| target3: bird | 320 | 20 | 18 |
| target3: mavic | 320 | 20 | 18 |
| target3: p3p | 144 | 9 | 9 |
| total | 784 | 49 | 45 |

Primary task is `bird_drone`; `target3` is secondary. Static recordings and
pole are excluded, so a classifier cannot solve the binary task by detecting
motion alone. P3P has only nine groups. No oversampling or class balancing is
applied during preparation.

`group` is target plus pose, merging bird flapping settings at turntable 0 and
tripod azimuth 0/tilt 0, and the three Mavic absorber configurations at that
tripod pose. This leaves 45 groups and is stricter than recording-only CV.
`group_recording`, `group_pose`, mount, state, variant, original/cropped cell
indices and source SHA-256 remain in `meta.parquet`. Equal pose angles on
different targets do not establish a shared flight. All measurements belong to
one indoor campaign, so grouping the whole session would leave one group and
make CV impossible. New-session generalization remains untested.

The authors' `README.docx` was read as ZIP/XML, never as executable content.
It confirms the 20 x 30 crop and 1000 samples. The 33 phase-angle text files
contain 1000 values each; `example.txt` is a separate spectrogram table. They
are extracted signal phase, not turntable encoder traces or flight/session
identifiers. They supply neither a PRF nor additional independent recordings.

Preparation writes only to `<DATA_ROOT>/uas_birds_60ghz/prepared/`. Validation
passed with zero nonfinite frames, constant channels or zero-MAD channels:

```bash
/Users/mikhail/.t3/worktrees/imf/t3-44ed3594/.venv/bin/python -I \
  /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf/datasets/uas_birds_60ghz/prepare.py
cd /Users/mikhail/.t3/worktrees/imf/t3-44ed3594/experiments/classification-emd-irmf
../../.venv/bin/python -m emdirmf.validate uas_birds_60ghz
```

The 16 spatial cells are strongly correlated views of one target. Reported
sample size is 49 recordings in 45 groups, not 784 independent observations.
Target identity, background gain and mounting may remain confounded even after
normalization. A verified target-free background control was not constructed.
This is a bionic bird and two mounted drone models, not an airborne wildlife
benchmark. The unknown PRF prevents unconditional Hz or velocity conclusions.

## Full DataPort release: untested path

Expected archive is `raw/dataport/Data.zip`; list and manually extract it to
`raw/dataport/extracted/` without executing its contents. After a successful
opcode scan, run the isolated converter with that source root and
`--destination <DATA_ROOT>/uas_birds_60ghz/converted_dataport`. The converter
requires unique stems; if the release has nested repeated names, inspect its
provenance and adapt the mapping rather than silently merging recordings.

`prepare.py --source dataport` is a clearly separated, untested path. It expects
one numeric complex `(T, 128, 168)` cube per original recording, `T >= 1000`,
with filenames matching the public target/pose/state convention. It checks the
shape, applies `[:, 60:80, 60:90]`, uses the same label-free cell selection and
cuts longer records into nonoverlapping 1000-sample windows. Before use, verify
the axis order and exact crop equality on a matching public recording. Different
filenames or session metadata require a reviewed adapter, not guesses.

This path replaces `prepared/`, so preserve the public result and caches in a
separate external data root before using it. The full cubes would add spatial
context and allow target/background controls and a test of crop selection.
They may contain the same campaign and recordings rather than more independent
samples. Only verified new sessions or recordings would improve statistical
power. The release may also clarify PRF and acquisition provenance; neither is
assumed here.
