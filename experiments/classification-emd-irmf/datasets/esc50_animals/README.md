# esc50_animals: the 10 animal classes of ESC-50

A second, small dataset next to `birdclef2023`. It takes the ESC-50 "Animals" major
category: dog, rooster, pig, cow, frog, cat, hen, insects, sheep, crow. That gives
40 five-second clips per class, all cut from Freesound recordings. It is the cheap
literal reading of "animal classification", and it has a bird vs non-bird task.
The candidate search and the reasons for choosing BirdCLEF 2023 as the primary
dataset are in `../birdclef2023/README.md`.

5,100 mono frames of 2048 samples at 22,050 Hz (92.9 ms), 510 per class, from all
400 clips. Primary task `y_animal` (10 classes). Second task `y_bird`: rooster, hen
and crow (1,530 frames) vs the other 7 animals (3,570). `group` is the Freesound
uploader.

## Preparation

`prepare.py` reads `ESC-50-master.zip` in place and never extracts it. It imports
`select_frames` and `balance` from `../birdclef2023/prepare.py`, so the frame rule
is the same code. Mono, 44.1 kHz resampled to 22,050 Hz with `soxr_hq`, mean
removed, no filtering of the stored frames. Candidate starts every 512 samples.
The score is mean power after a 250 Hz zero-phase high-pass. Picks are greedy, do
not overlap, stay within 15 dB of the clip's loudest frame, and stop at 20 per clip.

One extra step comes first. ESC-50 pads short clips with digital silence, so
leading and trailing runs of exact zeros are trimmed before resampling. The mean
trimmed tail is 0.52 s. Without the trim, a frame that straddles the padding would
be mostly exact zeros, and its median/MAD normalization would be dominated by them.

Balance: the smallest class (dog, short barks) yields 510 candidate frames, so
every class gets 510 by the same round-robin over energy ranks, seed 0. Most clips
give 13 or 14 frames, and the range is 1 to 20.

## Grouping

`group` is the Freesound uploader. The archive's LICENSE file has an attribution
line for every clip, and the uploader comes from there. All 400 clips matched. This
is stricter than the official ESC-50 folds, which only keep clips from one source
file together. `src_file` and `esc50_fold` stay in `meta.parquet` for that looser
split.

Groups per class: pig 12, sheep 15, cow 16, frog 17, crow 18, insects 18, hen 19,
cat 20, rooster 28, dog 35. That is 157 groups, enough for 5-fold grouped CV, but
some are heavy. felix.blume holds 444 frames (8.7%) and dobroide holds 384. Within
crow, one uploader (inchadney) holds 187 of the 510 frames. Within pig, felix.blume
holds 158. Fold-to-fold class balance will therefore vary more than in
`birdclef2023`.

## Files

- `<DATA_ROOT>/esc50_animals/raw/ESC-50-master.zip`: GitHub archive of
  `karolpiczak/ESC-50` master at commit `33c8ce9eb2cf0b1c2f8bcf322eb349b6be34dbb6`,
  646 MB, sha256 `805afc618aff80eff0641d51311647871fead5807da68bb18e43a27db47a79ce`.
- `prepared/frames.npy`: float32 (5100, 1, 2048). `meta.parquet` has `frame_id`,
  `group`, `y_animal`, `y_bird`, `recording_id`, `source_file`, `start_sample` at
  22,050 Hz after the leading trim, `trim_lead_s`, `energy_rank`, scores,
  `frame_rms_dbfs`, `src_file`, `esc50_fold`, `esc10`, `uploader`, `license` and
  `freesound_url`. `dataset.json` and `recordings.csv` are next to them.
- Figures in `output/classification-emd-irmf/esc50_animals/data/`: the same three
  plots as `birdclef2023`.

Reproduce from the worktree root:

```bash
mkdir -p /Users/mikhail/Datasets/imf-classification/esc50_animals/raw
curl -L -o /Users/mikhail/Datasets/imf-classification/esc50_animals/raw/ESC-50-master.zip \
  https://codeload.github.com/karolpiczak/ESC-50/zip/refs/heads/master
.venv/bin/python -I experiments/classification-emd-irmf/datasets/esc50_animals/prepare.py
.venv/bin/python -I experiments/classification-emd-irmf/datasets/esc50_animals/figures.py
```

Check the sha256 after downloading, because master can move.

## Source, license, citation

- <https://github.com/karolpiczak/ESC-50>, doi:10.7910/DVN/YDEPUT.
- License: ESC-50 as a whole is CC BY-NC 3.0. Per-clip Freesound licenses are kept
  per frame: CC BY (2,645 frames), CC0 (1,976), CC BY-NC (372), Sampling+ (107).
- Citation: Piczak, K. J. (2015). ESC: Dataset for Environmental Sound
  Classification. Proceedings of the 23rd ACM International Conference on
  Multimedia, 1015-1018. doi:10.1145/2733373.2806390.

## Caveats

- It is small. There are 400 clips, and 12 to 35 uploader groups per class, so
  confidence intervals will be wide.
- Level differs a lot by class. Median frame RMS runs from -24.1 dBFS for insects
  to -12.1 dBFS for cow, so the protocol's with/without `log_scale` comparison
  matters even more here than for birds.
- Insects and some crow and sheep clips are dense or noisy textures, while dog and
  hen clips are short events. A frame of 93 ms catches part of one event, not the
  whole call.
