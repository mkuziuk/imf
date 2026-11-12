# birdclef2023: bird species from the Kaggle BirdCLEF 2023 training set

12,000 mono frames of 2048 samples at 22,050 Hz (92.9 ms), 16 species, 750 frames
each, cut from 1,280 xeno-canto recordings by 451 recordists. Primary task
`y_species` (16 classes). Coarse task `y_passerine` (6 passerine species vs 10
non-passerine, 4,500 vs 7,500 frames). `group` is the recordist.

## Which competition, and could we get it

The Kaggle bird competition the user remembers is almost certainly BirdCLEF. It has
run on Kaggle every year since 2021, after the 2020 "Cornell Birdcall
Identification" competition. The phrase "animal classification (or birds)" fits
BirdCLEF+ 2025 best, because that edition added amphibians, mammals and insects.
Kaggle competition data needs a login and acceptance of the rules. This machine has
no Kaggle credentials, so I looked for login-free mirrors (checked 2026-10-20):

- BirdCLEF+ 2025: the Hugging Face mirror `christopher/birdclef-2025` now returns
  HTTP 401 (removed or private). The only other copy I found
  (`VictorJuiz/BirdCLEF2025-1000-sample`) holds 1,000 mel spectrograms and no waveforms.
- BirdCLEF 2024: no mirror found.
- BirdCLEF 2023: complete, public copies exist on Hugging Face. `sebasgar/birdclef-2023`
  is the original Kaggle zip (5.27 GB, 16,941 training recordings). This dataset uses it.

## Candidates

| Name | Competition | Modality | Classes | Size | Login-free | License | Fit for 1-D decomposition |
|---|---|---|---|---|---|---|---|
| BirdCLEF 2023 (Kenya) | Kaggle 2023 | bird audio, 32 kHz ogg | 264 species | 16,941 recs, 5.3 GB | yes, HF mirror `sebasgar/birdclef-2023` | CC BY-NC-SA 4.0, per-recording CC BY-NC-SA / BY-SA | good. Recordist metadata allows strict grouping. **Chosen** |
| BirdCLEF+ 2025 (Colombia) | Kaggle 2025 | birds, amphibians, mammals, insects | 206 taxa | 28,564 recs, ~12 GB | no, mirror gone | CC BY-NC-SA 4.0 | good, best match to "animals", not reachable |
| BirdCLEF 2024 (Western Ghats) | Kaggle 2024 | bird audio | 182 species | ~24k recs | no | CC BY-NC-SA 4.0 | good, not reachable |
| BirdCLEF 2021 / 2022 | Kaggle | bird audio | 397 / 152 species | 42 GB / 6.6 GB zips | yes, HF `sebasgar/birdclef-2021`, `-2022` | CC BY-NC-SA 4.0 | good. 2023 is smaller per useful species |
| Cornell Birdcall Identification | Kaggle 2020 | bird audio, mp3 | 264 species | ~21k recs | Kaggle no; BEANS copy `DBD-research-group/beans_cbi` yes | xeno-canto CC licenses | fair. The BEANS copy is 17 GB parquet with labels only, no recordist |
| Rainforest Connection Species Audio Detection | Kaggle 2020-21 | soundscapes, birds + frogs | 24 species | ~4.7k one-minute files, ~1.2k true-positive call labels | Kaggle no; BEANS `beans_rfcx` yes | Kaggle competition terms | fair. Short labelled calls inside 60 s soundscapes |
| MAFAT Radar Challenge | CodaLab 2020 | pulse-Doppler radar I/Q | human vs animal | 6,656 train + 49k aux segments | no, CodaLab account; access looks restricted | competition terms | poor. Slow-time length per segment is 32 pulses (128 x 32 I/Q), far too short for a multiscale decomposition |
| DCASE 2018 Bird Audio Detection | DCASE challenge | 10 s field audio | bird present / absent | ~36k clips, several GB | yes (freefield1010, warblrb10k, BirdVox-DCASE-20k) | CC BY, per subset | fair. Binary detection only, labels are clip-level |
| ESC-50, Animals category | no (benchmark) | 5 s clips, 44.1 kHz | 10 animals | 400 clips, 646 MB zip | yes, GitHub | CC BY-NC 3.0 | fair. Small, but cheap. Prepared as `esc50_animals` |
| xeno-canto direct | no | bird audio | ~11k species | huge | no, API v3 needs an account key | per recording CC | good, but needs an account |
| BirdSet | no (benchmark) | xeno-canto + soundscapes | up to ~10k species | tens to hundreds of GB | yes, HF | CC BY-NC 4.0 | good, but far larger than this study needs |

## Selection rules

All rules are label-free inside a recording and identical for every recording.
`prepare.py` applies them in this order with `numpy.random.default_rng(seed=0)`.

1. **Eligible recordings.** Rating at least 3.0 out of 5, and no listed secondary
   species. Recordings with background species are left out so the primary label is
   less likely to be wrong for a given frame.
2. **Species.** Rank species by eligible recordings after capping each recordist at
   2 recordings per species, then take the top 16. Every chosen species keeps 80
   recordings from 55 to 66 recordists.
3. **Recordings.** A seeded permutation per species, at most 2 recordings per
   recordist, stopping at 80.
4. **Audio.** Average channels to mono, read at most the first 120 s, resample from
   32 kHz to 22,050 Hz with `soxr_hq`, subtract the recording mean. Stored frames get
   no other filtering.
5. **Frame candidates.** Starts every 512 samples. The score is the mean power of the
   recording after a 4th-order zero-phase Butterworth high-pass at 250 Hz. The
   high-pass applies to scoring only, so wind and handling rumble do not win the
   ranking.
6. **Frame picks.** Greedy in descending score, no two frames overlapping, only frames
   within 15 dB of the recording's loudest frame, at most 20 per recording.
7. **Class balance.** Per species, round-robin over recordings by energy rank. All
   rank-1 frames go in, then all rank-2 frames, and so on until 750. The RNG
   subsamples the last partial rank. 1,205 of the 1,280 recordings end up with 9 or
   10 frames, 8 with 11, and 67 short or quiet ones with fewer.

Why 22,050 Hz and L = 2048: most bird song energy lies between 1 and 8 kHz, and a
Nyquist of 11 kHz keeps nearly all of it. At 16 kHz everything above 8 kHz would
go, and the wagtail and house-martin spectra still carry energy there. At 22,050 Hz a 2048-sample frame lasts 93 ms, which holds a
syllable or a few call pulses. Keeping 32 kHz would shrink that to 64 ms at the same
IRMF cost (about 0.4 s per frame).

## Counts

| y_species | Common name | Scientific name | Order | Recordings | Recordists | Frames |
|---|---|---|---|---:|---:|---:|
| barswa | Barn Swallow | Hirundo rustica | Passeriformes | 80 | 61 | 750 |
| blakit1 | Black Kite | Milvus migrans | Accipitriformes | 80 | 60 | 750 |
| cohmar1 | Common House-Martin | Delichon urbicum | Passeriformes | 80 | 62 | 750 |
| combul2 | Common Bulbul | Pycnonotus barbatus | Passeriformes | 80 | 55 | 750 |
| combuz1 | Common Buzzard | Buteo buteo | Accipitriformes | 80 | 61 | 750 |
| comsan | Common Sandpiper | Actitis hypoleucos | Charadriiformes | 80 | 66 | 750 |
| eaywag1 | Western Yellow Wagtail | Motacilla flava | Passeriformes | 80 | 63 | 750 |
| egygoo | Egyptian Goose | Alopochen aegyptiaca | Anseriformes | 80 | 59 | 750 |
| eubeat1 | European Bee-eater | Merops apiaster | Coraciiformes | 80 | 62 | 750 |
| grecor | Great Cormorant | Phalacrocorax carbo | Suliformes | 80 | 55 | 750 |
| greegr | Great Egret | Ardea alba | Pelecaniformes | 80 | 59 | 750 |
| hoopoe | Eurasian Hoopoe | Upupa epops | Bucerotiformes | 80 | 60 | 750 |
| litegr | Little Egret | Egretta garzetta | Pelecaniformes | 80 | 63 | 750 |
| thrnig1 | Thrush Nightingale | Luscinia luscinia | Passeriformes | 80 | 63 | 750 |
| wlwwar | Willow Warbler | Phylloscopus trochilus | Passeriformes | 80 | 65 | 750 |
| woosan | Wood Sandpiper | Tringa glareola | Charadriiformes | 80 | 62 | 750 |

Totals: 12,000 frames, 1,280 recordings, 451 recordists. A recordist contributes 2
to 258 frames, median 10. The largest group holds 2.2% of the frames. 12.7 h of
audio went through the frame scorer, and 446 MB of ogg files were downloaded.

## Grouping

`group` is the xeno-canto recordist, lower-cased with whitespace collapsed. That is
stricter than one group per recording. One recordist often uploads several
recordings of the same bird or site, made with the same microphone and the same
pre-filtering, and per-recording grouping would leak all of that across folds. A
recordist can appear under several species, so 175 groups carry more than one
label. `StratifiedGroupKFold` handles that. For the looser per-recording split, use
`recording_id`.

## Files

- `<DATA_ROOT>/birdclef2023/prepared/frames.npy`: float32 (12000, 1, 2048).
- `meta.parquet`: `frame_id`, `group`, `y_species`, `y_passerine`, `recording_id`,
  `source_file`, `start_sample` at 22,050 Hz, `start_seconds`, `energy_rank`,
  `score_db_rel`, `score_dbfs`, `frame_rms_dbfs`, taxonomy, `author`, `license`,
  `rating`, `call_type`, `latitude`, `longitude`, `xc_url`, `orig_sr`,
  `recording_seconds`.
- `dataset.json`: format keys, class and group counts, every preparation parameter,
  the seed, source, license and citation.
- `recordings.csv`: the 1,280 selected recordings with their metadata.
- `<DATA_ROOT>/birdclef2023/raw/`: `train_metadata.csv`, `eBird_Taxonomy_v2021.csv`
  and `train_audio/<species>/XC*.ogg` for the selected recordings only.
- Figures in `output/classification-emd-irmf/birdclef2023/data/`.
  `example_frames.png` shows one frame per species with a 1 s context spectrogram.
  `class_spectra.png` shows median frame spectra per species. `frame_energy.png`
  shows the selection score, frame level per species and frames per recording.

Reproduce from the worktree root:

```bash
.venv/bin/python -I experiments/classification-emd-irmf/datasets/birdclef2023/prepare.py
.venv/bin/python -I experiments/classification-emd-irmf/datasets/birdclef2023/figures.py
cd experiments/classification-emd-irmf && ../../.venv/bin/python -m emdirmf.validate birdclef2023
```

`prepare.py` reads the zip with HTTP range requests through `huggingface_hub.HfFileSystem`
at the pinned revision `b43091aff6033e548e57e71ef3b4a597ee55011a`. It fetches only
the two CSVs and the 1,280 selected members (about 5 minutes) and checks each
member's zip CRC. Output paths come from validated metadata names, never from the
archive. A second run with cached audio gave byte-identical frames and meta.
`--select-only` prints the selection without downloading audio.

## Source, license, citation

- Competition: <https://www.kaggle.com/competitions/birdclef-2023>. Mirror:
  <https://huggingface.co/datasets/sebasgar/birdclef-2023>. The mirror is a
  third-party re-upload, and its zip matches the Kaggle layout and the published
  count of 16,941 training recordings.
- License: CC BY-NC-SA 4.0 is the license Kaggle gives for BirdCLEF data. I could
  not load the Kaggle data page here (it needs JavaScript and a login), so that part
  is not re-checked. Per-recording licenses come from
  `train_metadata.csv` and are kept per frame. The selected frames carry CC BY-NC-SA
  4.0 (10,952), CC BY-NC-SA 3.0 (1,009), CC BY-SA 4.0 (29) and CC BY-SA 3.0 (10).
  Attribution is the recordist plus the XC URL. Use is non-commercial, and derived
  data must stay share-alike.
- Citation: Kahl, S., Denton, T., Klinck, H., Reers, H., Cherutich, F., Glotin, H.,
  Goeau, H., Vellinga, W.-P., Planque, R., Joly, A. (2023). Overview of BirdCLEF
  2023: Automated bird species identification in Eastern Africa. Working Notes of
  CLEF 2023, CEUR Workshop Proceedings 3497, 1934-1942. Credit xeno-canto and the
  individual recordists as well.

## Caveats

- Labels are weak. The label is the recording's primary species. A loud frame can
  still be another bird, an insect or noise. Excluding recordings with listed
  secondary species reduces this but does not remove it. One of the example hoopoe
  frames is a high chirp that is probably not the hoopoe, although the median
  hoopoe spectrum shows the expected peak near 600 Hz.
- These are not Kenyan soundscapes. Ranking by recording count favours widespread
  Palearctic migrants, and most recordings come from Europe (median latitude 47 N).
  Kaggle's soundscape test set is not used.
- Recording chains differ by recordist. Gain, microphones and any high-pass applied
  before upload all vary. Grouping by recordist keeps this out of the test folds,
  but frame level still differs by species (median frame RMS from -29.5 dBFS for
  comsan to -20.6 dBFS for thrnig1),
  so the protocol's with/without `log_scale` comparison matters here.
- Resampling clipped recordings overshoots slightly, and the largest |x| is 1.66.
  The protocol's median/MAD normalization makes this harmless.
- Recordings longer than 120 s contribute only their first 120 s.
