# Literature and existing solutions: decomposition features for classification

Scope: what is published about EMD-family features for micro-Doppler, RF drone
signals, and bird audio; what DroneRF results actually measure; which strong
baselines exist; and a ranked phase-3 plan for this study. Compiled
2026-10-20. Numbers marked "secondary" were not checked against the paper
itself.

## Summary

1. **There is no prior art for the core comparison.** No published work uses a
   robust or median-based multiscale smoother cascade as a classification
   feature extractor for radar, RF, or audio. The closest relatives are
   Bangham's median "sieves", which use increasing scales and have been used
   for lipreading and texture, and Iterative Filtering (Lin, Wang, Zhou 2009;
   Cicone's ALIF/FIF), which is a linear moving-average cascade and close to
   our `imf_linear`. The name "intrinsic multiscale filtering" is unused in the
   literature. Note that "IMF" already means *intrinsic mode function* in EMD
   papers [R1-R7].
2. **Head-to-head comparisons with strong spectral baselines usually go
   against EMD.**
   - CardRF + LightGBM: HHT features score 99.10/91.60% and plain DFT
     features 99.63/95.65% (3/10 classes) [E14].
   - UrbanSound8K: EMD-based mel energies score 54.6% and FFT mel energies
     63.4% [D2].
   - CWRU bearings: EMD 64-85% and EEMD 65-79%, against 93-97% with no
     decomposition at all. Only tuned VMD beat raw [E20].
   - Gains appear with weak classifiers, short windows, or fusion
     [D1, E12]. A decomposition result therefore means little without the
     raw/STFT control and the same classifier.
3. **Almost every published accuracy on DroneRF and the 60 GHz radar set
   comes from leaky splits.** BirdCLEF itself scores on held-out soundscapes,
   but frame-level bird studies often split windows of one recording.
   - DroneRF's reference protocol shuffles 22,700 spectra from 227
     recordings into 10 folds (verified in the authors' code).
   - The only grouped re-evaluation found (Shulman 2026, preprint) reports
     that AR-vs-Bebop macro-F1 falls from 0.74 to chance (0.46) under
     leave-one-recording-out CV [F8].
   - The 60 GHz radar authors' own code randomly permutes time stamps before
     splitting (see `datasets/uas_birds_60ghz/README.md`).
   - Our grouped protocol will give much lower numbers than the literature.
     That is expected, and it should be stated in every table.

## 1. EMD/HHT features in the three domains

### 1.1 Radar micro-Doppler (drones, birds, people)

- **EMD-feature pipelines for mini-UAVs (Oh, Guo, Lin and co-authors).**
  - EMD of CW radar returns, then 8 statistical/geometric features of the
    IMFs (blade-flash structure), then a nonlinear SVM [E1].
  - A follow-up compares six IMF entropies (Shannon, spectral, log-energy,
    approximate, fuzzy, permutation), keeps three, and fuses them over 8 IMF
    sets from downsampled signals before the SVM [E2].
  - An FMCW surveillance version uses about 13 IMF features [E3].
  - The abstracts give no accuracies. These are the closest templates for
    "IMF statistics as features" on radar.
- **Human activity:**
  - Fairchild & Narayanan: an EMD feature vector plus SVM on S- and W-band
    micro-Doppler, "very good" accuracy up to 90 m [E4].
  - Ma et al. (Sensors 2019): IMFs selected by eye for interference removal,
    then S-transform, then bagged trees. 97.3% vs 88.5% for STFT, 10-fold CV.
    No ablation isolates EMD [E5].
- **Drones vs birds without EMD (baselines and context):**
  - Molchanov et al. 2014: eigenpairs of the micro-Doppler correlation
    matrix, about 92% over aircraft/rotors/birds (X-band CW) [E6].
  - Rahman & Robertson: K/W-band signatures of drones and real birds; 100%
    drone vs bird at 94 GHz on a very small set [E7, E8].
  - Narayanan et al. 2023: 10 GHz spectrogram + SVM, about 96% drone vs bird and
    85% for 5 classes [E9].
  - Torvik et al. 2016: polarimetric features, 100% birds vs UAVs, but with
    random 5-fold CV [E10].
  - The bionic-bird 60 GHz set behind our radar data reports only CNN /
    ConvLSTM results on random splits [K1].
- **Other decompositions:**
  - Local mean decomposition for UAV-vs-bird [E11].
  - L-band EMD + K-band STFT fusion (Chen et al. 2025), +3-12 points at low SNR
    [E12].
  - No VMD or CEEMDAN drone-vs-bird classifier was found. Complex/bivariate
    EMD exists for I/Q signals (Rilling et al. 2007; Tanaka & Mandic 2007)
    [E13], but nobody has used it for classification. For complex radar data
    this study's channel-wise approach (I and Q, or amplitude and phase,
    decomposed separately) is the pragmatic choice.

### 1.2 RF drone signals

- **No EMD, VMD, or CEEMDAN study on DroneRF was found.**
- **Closest work on other datasets:**
  - Medaiyese et al. (CardRF): 2 IMFs x 13 Hilbert-energy statistics, then
    XGBoost. UAS-vs-other accuracy is 87.1%, rising to 91% with
    wavelet-packet features [E15].
  - Sung & Lee 2024 (CardRF, LightGBM): **HHT features lose to plain DFT
    features** [E14].
  - Xu et al. (ICASSP 2021): EMD/EEMD spectrograms of the top-5 IMFs with
    CNN/MLP/DT/SVM. EEMD helps for control links and EMD for video links (bar
    charts only) [E16].
  - Fu et al. 2026 (CardRF): VMD + ReliefF + SVM, >74.7% above -10 dB SNR
    [E18].
  - Tanveer et al. 2026 (LowSNR_DroneRF): EMD only as a denoiser before
    handcrafted features + RF/ANN/CNN. An independent reproduction exists
    [E19].
- **Wavelet transient features dominate RF fingerprinting:**
  - Ezuma et al.: Haar-DWT transients, higher-order statistics, kNN. 96.3% on
    14 controllers (2019) and 98.13% on 15 (2020) [E21].
  - Medaiyese et al.: wavelet scattering, up to 98.9% at 10 dB [E22].
  - These are the domain's handcrafted reference points.

### 1.3 Bioacoustics

- Thin, mostly in weak venues. No BirdCLEF-scale EMD study exists.
  - EEMD then Hilbert marginal spectrum then mel filterbank ("HHT-MFCC")
    improves crane-species clustering over MFCC [E23].
  - EMD/EEMD AM-FM features with an HMM beat MFCC/LPC for Bryde's whale calls
    [E24].
  - EMD stopping criteria for cicada detection give 96.7-98% [E25].
  - The clearest head-to-head is Ahmed et al. 2022 [D2]. EMD mel energies are
    worse than FFT mel energies on all four datasets; fusion adds 0.3-3
    points.

### 1.4 EMD-family comparisons and known failure modes

- **Variants.**
  - EEMD (noise-assisted ensembles against mode mixing) [R9].
  - CEEMDAN (exact reconstruction, fewer trials) [R10].
  - VMD (non-recursive, needs the mode count K and penalty alpha fixed in
    advance) [R11].
  - Ensemble methods cost 100-800x EMD.
  - In classification benchmarks, tuned VMD tends to beat EMD/EEMD [E20].
    EEG results are mixed (EEMD > EMD > DWT for seizures; wavelet packets >
    EMD for motor imagery) [E26-E28].
- **Failure modes relevant to features:**
  - On noise EMD behaves as a dyadic filter bank [R12, R13].
  - Two tones separate only below an amplitude-dependent frequency ratio
    [R14]. That is mode mixing.
  - Spline envelopes create edge artefacts, and a single spike spreads into
    every IMF [R15]. EMD under heavy-tailed noise is studied in [R16]. This
    is the case for a robust smoother.
  - The IMF count varies by frame, and IMF k is not a fixed band (Section 5).

### 1.5 Robust / median multiscale decompositions (prior art)

- **Bangham's sieves** are cascades of recursive median/morphological
  filters. Each scale removes extrema of a given extent, and the "granules"
  form a scale-space decomposition. It is robust to impulsive noise, adds no
  new extrema, and comes with a decomposition theorem [R1, R2, R3]. It has been
  used as a feature extractor (sieve-based lipreading with HMMs [R4]; texture).
  It differs from IRMF in three ways: scale increases where IRMF windows
  shrink, there is no M-estimator, and it is morphological rather than a
  local fit.
- **Iterative Filtering / ALIF / Fast IF** subtract local weighted moving
  averages iteratively [R5, R6, R7]. They are structurally close to
  `imf_linear` and used for EEG/bearing features (secondary).
- **Robust EMD variants:**
  - Order-statistics envelopes [R8].
  - Median ensemble EMD, which replaces the ensemble mean with a median [R17].
  - Median EEMD for EEG emotion classification [R18].
- **Robust smoothing and trend building blocks:**
  - Tukey/Velleman running medians 4253H [R19].
  - Local M-smoothers (Chu, Glad, Godtliebsen, Marron 1998) [R20].
  - l1 trend filtering [R21].
  - STL and RobustSTL [R22].
  - Median-interpolation pyramids (Donoho & Yu) [R23].
  - Cascaded Hampel filters [R24].
- **Positioning.** IRMF is in effect a coarse-to-fine cascade of local
  Huber-type M-smoothers. The nearest classification precedents are sieves
  (robust, multiscale) and IF (multiscale, linear). Citing both frames the
  contribution: "robust like sieves, local-fit and scale-ordered like IF/EMD".

## 2. DroneRF: what the published numbers measure

**Dataset** [F1]:

- 227 recorded segments: background 41, Bebop 84, AR 81, Phantom 21.
- Each segment has an L file (lower half of the 2.4 GHz band) and an H file
  (upper half). Each file holds 10^7 real-valued samples at 40 MS/s (0.25 s),
  stored as CSV (over 40 GB in total).
- Labels: 2-class (drone/none), 4-class (background + 3 drones), and 10-class
  (background, Bebop modes 1-4, AR modes 1-4, Phantom mode 1).
- Classes are imbalanced: Phantom has 21 segments and one mode only.

**Original protocol** (Al-Sa'd et al. 2019 [F2]; code verified in
`github.com/Al-Sad/DroneRF`):

- `Main_1_Data_aggregation.m` cuts each 10^7-sample file into 100 chunks of
  L = 10^5 samples. Each chunk is mean-removed and transformed with
  `fft(x, M=2048)`.
- In MATLAB `fft(x, 2048)` truncates the input, so each spectrum uses only
  the first 2048 samples (51 us) of its 2.5 ms chunk.
- The positive half (1024 bins per band) is kept. The H band is rescaled for
  spectral continuity using Q = 10 edge bins, the two are concatenated (2048
  features), and the result is squared.
- That gives 22,700 spectra (227 x 100).
- `Classification.py`: a Keras MLP (3 hidden layers of 128 ReLU units,
  sigmoid output, MSE loss, Adam, 200 epochs, batch 10) under
  `StratifiedKFold(n_splits=10, shuffle=True)` over all spectra.
- Spectra from the same 0.25 s recording therefore sit in train and test.
- Reported accuracy: **99.7% (2-class), 84.5% (4-class), 46.8% (10-class)**
  [F2].

**Later work** (all random frame/spectrum splits unless noted):

| work | input | model | 2 / 4 / 10-class | split |
|---|---|---|---|---|
| Al-Sa'd et al. 2019 [F2] | 2048-bin FFT power (L+H) | MLP | 99.7 / 84.5 / 46.8 | 10-fold shuffled spectra |
| Al-Emadi & Al-Senaid 2020 [F3] | FFT spectra | 1-D CNN | 99.7 / 88.4 (identification) | random |
| Huynh-The et al. 2022, RF-UAVNet [F4] | raw RF segments | grouped-conv CNN | 99.85 / 98.53 / 95.33 | random |
| Shulman 2026 (preprint) [F8] | several | several | AR vs Bebop macro-F1 0.74 random -> **0.46 leave-one-recording-out** | grouped |

<!-- DRONERF AGENT ROWS PENDING -->

**Lessons for this study.**

- Report our DroneRF results next to a *reproduction* of the Al-Sa'd
  pipeline under both splits. The literature's 10-class 46.8-95% figures are
  not comparable to grouped CV.
- With 227 recordings, grouped 5-fold CV is feasible for 2- and 4-class.
  For 10-class it is thin: Phantom has 21 segments in a single mode, and some
  AR/Bebop modes have about 20.
- Use `StratifiedGroupKFold` and check per-fold class presence.
- DroneRF is real-valued, so C = 2 means the two bands (L, H), not I/Q, as
  `datasets/dronerf/prepare.py` already does.

## 3. Strong standard approaches

### 3.1 Bird audio

- **Log-mel CNNs are the competition standard.** BirdCLEF entries resample to
  32 kHz and train ImageNet-pretrained EfficientNet/NFNet/ConvNeXt on
  log-mel images of 5 s windows.
  - Lasseck (BirdCLEF 2024): 128 mels, 20 Hz-16 kHz, eca_nfnet_l0 / EfficientNet-B0;
    test macro ROC-AUC about 0.69 on soundscapes [B1].
  - BirdCLEF+ 2025, 2nd place: EfficientNetV2-S + NFNet-L0 with pseudo-labels;
    private macro ROC-AUC 0.928 (winner 0.930) [B2].
  - The 2025 team validated with 5-fold CV stratified by species and
    **grouped by recordist**. Their local CV stopped tracking the leaderboard
    above 0.9.
  - BirdCLEF 2023, the source of our bird frames, scored padded cmAP
    (secondary).
- **Pretrained bird embeddings plus a linear probe are the strongest cheap
  baseline.**
  - Ghani et al. 2023 (Perch 1): bird-trained embeddings (Perch, BirdNET)
    clearly beat general-audio embeddings (AudioMAE, YAMNet, VGGish) on few-shot
    bird, bat, marine-mammal, and amphibian tasks [B3]. With 32 examples per
    class, Perch reached about 0.83 top-1 on RFCX birds and BirdNET about 0.78
    (from the paper's tables, secondary). The authors note that windows from
    one source recording can fall on both sides of their splits.
  - Perch 2.0 (2025) is an EfficientNet-B3 on 5 s at 32 kHz producing a 1536-d
    embedding. It reports BirdSet AUROC 0.908 and BEANS linear-probe accuracy
    of about 0.84 [B4].
  - BirdNET (Kahl et al. 2021) uses 3 s windows at 48 kHz [B5]. Its model
    weights are CC BY-NC-SA 4.0.
  - Benchmarks: BirdSet [B6] and BEANS [B7]. PANNs, AudioMAE, and BEATs trail
    the bird-specific models on bird tasks.
- **Consequences for this study.**
  - Our BirdCLEF 2023 frames (16 species, 22.05 kHz, 2048 samples = 93 ms)
    are 30-50x shorter than the windows these models expect. Neither
    pretrained embeddings nor competition-style CNNs are fair per-frame
    comparators.
  - Use them only as a recording-level upper bound: pool embeddings over each
    recording's 5 s windows.
  - For the paired method comparison, use a small log-mel or STFT CNN trained
    from scratch on the same 93 ms frames.
  - Expect low absolute frame-level accuracy. A 93 ms snippet often holds
    part of one syllable, or no bird at all, because of weak labels.

### 3.2 Time-series classification

- **The bake-off.** "Bake off redux" (Middlehurst, Schäfer, Bagnall, DMKD 2024)
  covers 112 UCR problems plus 30 new ones [T1].
  - HIVE-COTE 2 and MultiROCKET+Hydra are significantly better than the rest.
    Mean accuracy: HC2 0.891, MR-Hydra 0.884, RDST 0.876, InceptionTime 0.874,
    WEASEL 2.0 0.874, QUANT 0.867.
  - Train time over 142 problems: QUANT 0.22 h, MR-Hydra 1.33 h, HC2 264 h.
  - Univariate only, so it does not show how these behave with K+1 channels.
- **Primary papers:** ROCKET [T2], MiniRocket [T3], MultiRocket [T4],
  Hydra [T5], HC2 [T6], InceptionTime [T7], catch22 [T8].
- **aeon 1.6.0 (installed here, checked locally).**
  - Data layout: `(n_cases, n_channels, n_timepoints)`.
  - Classes in `aeon.classification.convolution_based`: `RocketClassifier`,
    `MiniRocketClassifier`, `MultiRocketClassifier`, `HydraClassifier`,
    `MultiRocketHydraClassifier` (multivariate, multithreaded, equal length
    only, `RidgeClassifierCV` head).
  - Also available: `aeon.classification.interval_based.QUANTClassifier` /
    `DrCIFClassifier`, `aeon.classification.hybrid.HIVECOTEV2` (contractable
    with `time_limit_in_minutes`), and
    `aeon.transformations.collection.feature_based.Catch22` /
    `aeon.classification.feature_based.Catch22Classifier`.
  - `aeon.classification.deep_learning.*` (InceptionTime, LITETime, ResNet) is
    Keras/TensorFlow. TF is not installed, and it would not use MPS anyway.
  - sktime 1.2.0 has a PyTorch `InceptionTimeClassifierTorch` (MPS support
    unverified). `tsai` (fastai/PyTorch) offers InceptionTimePlus and
    MiniRocket on GPU.
- **Feeding components to these models.** Every model above takes multichannel
  input natively.
  1. **Channel stacking:** (C x (K+1), L) per frame, which is the natural
     input for MiniRocket/MultiRocket/Hydra/QUANT and 1-D CNNs.
  2. **Per-component features:** catch22 or the protocol's list per
     component, then GBM/linear models.
  3. **Hilbert spectra as images:** instantaneous amplitude binned by
     instantaneous frequency, fed to a 2-D CNN.
- **Published evidence on these routes is thin and rarely ablated.**
  - Klaar et al. (Sensors 2024): CEEMDAN/EWT/VMD + MiniRocket + logistic
    regression on insulator ultrasound reached 0.992/0.995/0.980, against
    about 0.978 without decomposition at window 100. The gain grew at short
    windows [D1].
  - Ahmed et al. (Sensors 2022): EMD-TKEO log-mel band energies were *worse*
    than FFT log-mel on all four sound datasets (UrbanSound8K 54.6% vs 63.4%).
    Only the FFT+EMD combination helped, and only slightly [D2].
  - Yin et al. 2026 (infrasound, 6 classes): STFT 98.35% > CWT 94.0% > HHT
    93.6% images with the same CNN [D3].
  - The pattern: when the classifier is strong and the comparison fair,
    decomposition features at best match good spectral features. Gains appear
    mostly with weak classifiers, short windows, or noisy signals. That is the
    regime where a robust decomposition has its best chance, and it is why
    the raw-signal control with the *same* classifier matters.
  - I found no published use of catch22 per IMF. That makes it a cheap,
    novel, interpretable baseline here.

## 4. Recommended phase-3 ML plan

Cost assumptions: 16-core M4 Max with MPS, about 10k frames, L = 1024-2048,
C = 1-2 input channels, K+1 = 9-14 components per channel. Times were measured
locally on 2026-10-20 (aeon 1.6.0, torch 2.14.1, pimf 0.4.0, PyEMD).

| step | measured cost |
|---|---|
| PyEMD `EMD`, L = 1024, 1 core | 0.005 s/frame |
| pimf linear IMF, L = 1024, 1 core | 0.001 s/frame |
| pimf IRMF (`SmoothAbs`, H = 2 sigma_hat), L = 1024, 1 core | 0.11 s/frame, about 70 s per 10k frames on 16 processes |
| aeon `MiniRocket` fit + transform, 2000 series, C = 9, L = 1024, 16 jobs | 5 s (about 10k features) |
| aeon `MultiRocket`, same input | 14 s (about 50k features) |
| aeon `Catch22`, 500 series x 9 channels, L = 1024, 16 jobs | 1.3 s (float64 input only, see pitfalls) |
| InceptionTime-style 1-D CNN (6 modules, 32 filters), MPS, N = 8000, C = 2-18, L = 2048, batch 64 | 37 s/epoch (19 s at L = 1024); channel count barely matters |
| "lite" variant (3 modules, 16 filters), same input | 16 s/epoch (9 s at L = 1024) |

Decomposition is a one-off cache. Only IRMF at L = 4096 costs real time:
about 1.5 s/frame by PROTOCOL.md's figure, or roughly 16 min per channel per
10k frames.

Ranked plan. Each item runs on the same saved grouped splits for every method
(raw, emd, imf_linear, irmf), with and without `log_scale`.

1. **Per-component handcrafted features + gradient boosting** (LightGBM;
   `RandomForest` as a check). Features: the protocol's per-component list,
   plus catch22 per component (aeon `Catch22`, 22 x (K+1) x C features).
   Cost: minutes. This is the most interpretable comparison, and it links
   directly to phase 2. Tune only inside the training folds: about 20
   random configurations on an inner grouped split.
2. **MiniRocket / MultiRocket on the component stack.** Treat the K+1
   components (times C channels) as a multichannel series and pass it to aeon
   `MiniRocketClassifier` / `MultiRocketClassifier`. These fit a ridge
   classifier with built-in CV on the transform. Compare against the same
   model on the raw frame (C channels) and on random multichannel controls
   (the raw frame plus K copies with independent band-limited noise), which
   rules out "more channels" as the explanation. `imf_linear` already serves as
   the linear filter-bank control for "robust vs linear". Cost: about 1 min
   per fold for MiniRocket and 2-4 min for MultiRocket at 8k x (9-28) x 2048,
   extrapolated from the timings above. Also run `MultiRocketHydraClassifier`, the strongest single
   convolution model in the 2024 bake-off (Section 3).
3. **QUANT and DrCIF (interval-based) on the component stack.** QUANT is
   fast and strong. Its quantile features over intervals suit per-component
   amplitude distributions, which is where a robust decomposition should
   differ from a linear one. Cost: minutes.
4. **1-D CNN (InceptionTime-style, PyTorch on MPS) on the component stack.**
   aeon's `InceptionTimeClassifier` needs TensorFlow, which is not installed,
   so write a short PyTorch version or use `tsai`. MPS is the bottleneck. A
   full InceptionTime (6 modules, 32 filters) takes about 18 min per 30-epoch
   run at L = 2048. The "lite" variant (3 modules, 16 filters) takes about
   5 min per 20-epoch run. Plan on the lite net, 1 split repeat x 5 folds x 4
   inputs (raw, emd, imf_linear, irmf), which is about 2 h per dataset-task.
   Spend the other two split seeds only on the task where the decision is
   close. Augment only with circular shift and amplitude jitter. Amplitude
   jitter is harmless because inputs are robustly normalised.
5. **Time-frequency images.** Compare (a) the log-STFT spectrogram of the raw
   frame, (b) a Hilbert spectrum from the components (instantaneous amplitude
   binned by instantaneous frequency, for example with `emd.spectra.hilberthuang`),
   and (c) a "component scalogram" (K+1 rows of Hilbert amplitude envelopes).
   Use a small 2-D CNN (or ResNet-18 from scratch) on MPS. Cost: about 1-2 h
   per dataset-task (estimate, not measured).
   This tests whether a decomposition-derived TF image beats the STFT, the
   standard representation in micro-Doppler and bioacoustics.
6. **Reference points, not competitors:**
   - DroneRF: reproduce the Al-Sa'd DNN on 2048-bin FFT magnitudes (both
     bands), once with a random frame split and once with our grouped splits.
     The gap between them is the leakage effect.
   - Bird audio: a log-mel CNN on the same frames. Separately, report
     recording-level Perch/BirdNET embeddings plus logistic regression as an
     upper bound that needs longer context (Section 3).
   - 60 GHz radar: Conv1D/LSTM on amplitude and phase, mirroring Larrat & Sales.
7. **Optional, if 1-2 show a signal:** HIVE-COTE 2 on one task as the
   accuracy ceiling for classical TSC. It is very slow at L >= 1024 with many
   channels (hours to days), so run it on a 2k-frame subsample at most.

Statistics: keep the protocol's paired fold-level comparisons (Nadeau-Bengio
corrected t-test, Wilcoxon, Holm). Report every model family separately
rather than taking the best over families, which would bias towards the
method with the most tuning.

## 5. Pitfalls and how to handle them

- **Group leakage.** Frames from one recording share channel, gain, pose,
  recordist, and background. Random frame splits inflate accuracy, often to
  near 100% (DroneRF, the 60 GHz authors' own code, and frame-level
  bioacoustics). Use `StratifiedGroupKFold` by recording as the protocol
  requires. For the radar set, also try `group_pose` (a target+pose
  "session"). Report published random-split numbers only as context.
- **Amplitude and session cues.** Background level and gain differ by class
  in the 60 GHz data (P3P recordings have about twice the noise floor of bird
  recordings). DroneRF segments differ in received power by drone and
  distance. BirdCLEF recordists differ in gain. Robust per-frame normalisation
  plus a separately reported `log_scale` handles the first-order effect.
  Add a "noise-only" control (target-free cell or frame) where possible.
  Feature importance on relative-energy features can also reveal gain leakage.
- **Class imbalance.** DroneRF has many more background segments than
  per-mode drone segments. The 10-class modes are uneven, and P3P has only 9
  moving recordings. Balance frames per class at preparation, use balanced
  accuracy and macro-F1, use `class_weight="balanced"` for linear and tree
  models, and report per-class recall. With few groups per class, fold-level
  variance is large, so report CIs, not just means.
- **EMD-specific issues.**
  - Mode mixing: one IMF carries several scales, or one scale is spread over
    several IMFs. EEMD/CEEMDAN reduce it at roughly ensemble-size times the
    cost (100-800 trials).
  - End effects: the spline envelopes extrapolate at frame boundaries. Either
    trim about 5% at each end before computing features, or compute features
    on the central part only. Apply the same trimming to IMF/IRMF so that the
    comparison stays fair. pimf's default `boundary='wrap'` behaves
    differently at frame edges, which is itself a confound worth noting.
  - Variable IMF count: frames give different numbers of IMFs (roughly
    log2(L); a synthetic L = 1024 test frame gave 7). The protocol's fixed K_EMD = 8 with zero-padding and
    folding into the residue is the standard fix. Record the true count,
    because the count is itself a feature (it tracks the number of extrema),
    and check that it is not what drives accuracy.
  - Index alignment: "IMF k" does not mean the same frequency band across
    frames, whereas IMF/IRMF components have fixed bandwidths set by the
    window schedule. This favours IMF/IRMF in per-index features and in
    channel-stacked models. A fairer EMD variant groups IMFs by
    mean-frequency band (a frequency-binned EMD feature set). Report it as an
    ablation.
  - Noise and impulsive outliers: sifting reacts to spikes by generating extra
    extrema, and a single spike spreads into every IMF [R15]. That is exactly
    the regime where a robust smoother should win. Bursty DroneRF
    transmissions and phase jumps in the radar series are plausible
    instances. Check per dataset (for example, frame kurtosis) before
    claiming it.
- **Library gotchas found while testing:**
  - aeon 1.6.0 `Catch22` raises a numba typing error on float32 input. Cast to
    float64.
  - aeon deep-learning classifiers need TensorFlow, which is not installed.
  - pimf 0.4.0 takes `workers` in the `IMF(...)` constructor, not in
    `decompose`. `h_min` there is a fraction of the signal length in (0, h1]
    (for example 3/L), not a sample count. PROTOCOL.md's `decompose(...,
    workers=1)` should be read accordingly.
- **Short frames vs pretrained audio models.** The BirdCLEF prep uses
  2048-sample frames at 22.05 kHz (93 ms). BirdNET (3 s at 48 kHz) and Perch
  (5 s at 32 kHz) cannot be applied per frame. Use them only at recording or
  5 s window level, as an upper bound, never in the paired method comparison.
- **Nested tuning.** Any hyper-parameter search, including the IRMF `h_ratio`
  ablation, must be chosen on inner folds. Otherwise IRMF gets 3 tries and
  EMD 1.

## References

### Prior art: robust and linear multiscale cascades, EMD theory (R)

- [R1] J. A. Bangham, P. D. Ling, R. Young, "Multiscale recursive medians, scale-space, and transforms with applications to image processing," *IEEE Trans. Image Process.* 5(6):1043-1048, 1996. https://doi.org/10.1109/83.503918
- [R2] J. A. Bangham, P. Chardaire, C. J. Pye, P. D. Ling, "Multiscale nonlinear decomposition: the sieve decomposition theorem," *IEEE TPAMI* 18(5):529-539, 1996. https://doi.org/10.1109/34.494642
- [R3] J. A. Bangham, P. Ling, R. Harvey, "Scale-space from nonlinear filters," *IEEE TPAMI* 18(5):520-528, 1996. https://doi.org/10.1109/34.494641
- [R4] I. Matthews, T. F. Cootes, J. A. Bangham, S. Cox, R. Harvey, "Extraction of visual features for lipreading," *IEEE TPAMI* 24(2):198-213, 2002. https://doi.org/10.1109/34.982900
- [R5] L. Lin, Y. Wang, H. Zhou, "Iterative filtering as an alternative algorithm for empirical mode decomposition," *Adv. Adapt. Data Anal.* 1(4):543-560, 2009. https://doi.org/10.1142/S179353690900028X
- [R6] A. Cicone, J. Liu, H. Zhou, "Adaptive local iterative filtering for signal decomposition and instantaneous frequency analysis," arXiv:1411.6051. https://arxiv.org/abs/1411.6051
- [R7] A. Cicone, H. Zhou, "Numerical analysis for iterative filtering with new efficient implementations based on FFT" (Fast IF), *Numer. Math.* 147:1-28, 2021. https://doi.org/10.1007/s00211-020-01165-5
- [R8] S. M. A. Bhuiyan et al., order-statistics-filter-based EMD envelopes, *EURASIP J. Adv. Signal Process.* 2008:728356.
- [R9] Z. Wu, N. E. Huang, "Ensemble empirical mode decomposition: a noise-assisted data analysis method," *Adv. Adapt. Data Anal.* 1(1):1-41, 2009. https://doi.org/10.1142/S1793536909000047
- [R10] M. E. Torres, M. A. Colominas, G. Schlotthauer, P. Flandrin, "A complete ensemble empirical mode decomposition with adaptive noise," ICASSP 2011. https://doi.org/10.1109/ICASSP.2011.5947265
- [R11] K. Dragomiretskiy, D. Zosso, "Variational mode decomposition," *IEEE Trans. Signal Process.* 62(3):531-544, 2014. https://doi.org/10.1109/TSP.2013.2288675
- [R12] P. Flandrin, G. Rilling, P. Gonçalvès, "Empirical mode decomposition as a filter bank," *IEEE Signal Process. Lett.* 11(2):112-114, 2004. https://doi.org/10.1109/LSP.2003.821662
- [R13] Z. Wu, N. E. Huang, "A study of the characteristics of white noise using the empirical mode decomposition method," *Proc. R. Soc. A* 460:1597-1611, 2004. https://doi.org/10.1098/rspa.2003.1221
- [R14] G. Rilling, P. Flandrin, "One or two frequencies? The empirical mode decomposition answers," *IEEE Trans. Signal Process.* 56(1):85-95, 2008. https://doi.org/10.1109/TSP.2007.906771
- [R15] A. Stallone, A. Cicone, M. Materassi, "New insights and best practices for the successful use of Empirical Mode Decomposition, Iterative Filtering and derived algorithms," *Scientific Reports* 10:15161, 2020. https://doi.org/10.1038/s41598-020-72193-2
- [R16] A. Komaty et al., EMD under alpha-stable noise, *IEEE Signal Process. Lett.* 22:818, 2015.
- [R17] X. Lang et al., "Median ensemble empirical mode decomposition," *Signal Processing* 176:107686, 2020. https://doi.org/10.1016/j.sigpro.2020.107686
- [R18] P. Samal, M. F. Hashmi, median EEMD for EEG emotion classification, *IEEE Sensors Letters* 7(5), 2023.
- [R19] P. F. Velleman, "Definition and comparison of robust nonlinear data smoothing algorithms," *JASA* 75:609-615, 1980. https://doi.org/10.1080/01621459.1980.10477521
- [R20] C.-K. Chu, I. K. Glad, F. Godtliebsen, J. S. Marron, "Edge-preserving smoothers for image processing," *JASA* 93:526-541, 1998 (local M-smoother).
- [R21] S.-J. Kim, K. Koh, S. Boyd, D. Gorinevsky, "l1 trend filtering," *SIAM Review* 51(2):339-360, 2009. https://doi.org/10.1137/070690274
- [R22] R. B. Cleveland et al., "STL: a seasonal-trend decomposition procedure based on loess," *J. Official Statistics* 6:3-73, 1990; Q. Wen et al., "RobustSTL," AAAI 2019.
- [R23] D. L. Donoho, T. P.-Y. Yu, "Nonlinear pyramid transforms based on median-interpolation," *SIAM J. Math. Anal.* 31(5):1030-1061, 2000.
- [R24] R. K. Pearson et al., "Generalized Hampel filters," *EURASIP J. Adv. Signal Process.* 2016:87.
- [R25] N. E. Huang et al., "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis," *Proc. R. Soc. A* 454:903-995, 1998. https://doi.org/10.1098/rspa.1998.0193
- pimf (IMF/IRMF implementation used here): https://github.com/mkuziuk/pimf , docs https://mkuziuk.github.io/pimf/

### EMD/HHT features in applications (E)

- [E1] H. Oh, X. Guo, F. Wan, K.-A. Toh, Z. Lin, "Micro-Doppler mini-UAV classification using empirical-mode decomposition features," *IEEE GRSL* 15(2):227-231, 2018. https://doi.org/10.1109/LGRS.2017.2781711
- [E2] X. Ma, B.-S. Oh, L. Sun, K.-A. Toh, Z. Lin, EMD-based entropy features for micro-Doppler mini-UAV classification, ICPR 2018, pp. 1295-1300. https://doi.org/10.1109/ICPR.2018.8546180
- [E3] B.-S. Oh, X. Guo, Z. Lin, "A UAV classification system based on FMCW radar micro-Doppler signature analysis," *Expert Syst. Appl.* 132:239-255, 2019. https://doi.org/10.1016/j.eswa.2019.05.007
- [E4] D. P. Fairchild, R. M. Narayanan, "Classification of human motions using empirical mode decomposition of human micro-Doppler signatures," *IET Radar Sonar Navig.* 8(5):425-434, 2014. https://doi.org/10.1049/iet-rsn.2013.0165
- [E5] Ma et al., EMD + S-transform human-motion micro-Doppler classification, *Sensors* 19:2598, 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6603564/
- [E6] P. Molchanov, R. Harmanny, J. de Wit, K. Egiazarian, J. Astola, "Classification of small UAVs and birds by micro-Doppler signatures," *Int. J. Microw. Wirel. Technol.* 6(3-4):435-444, 2014. https://doi.org/10.1017/S1759078714000282
- [E7] S. Rahman, D. A. Robertson, "Radar micro-Doppler signatures of drones and birds at K-band and W-band," *Scientific Reports* 8:17396, 2018. https://doi.org/10.1038/s41598-018-35880-9
- [E8] S. Rahman, D. A. Robertson, 94 GHz drone/bird classification, SPIE 11003, 2019, https://doi.org/10.1117/12.2518846 ; CNN on spectrograms, *IET Radar Sonar Navig.* 14(5):653-661, 2020, https://doi.org/10.1049/iet-rsn.2019.0493
- [E9] R. M. Narayanan, B. Tsang, R. Bharadwaj, "Classification and discrimination of birds and small drones using radar micro-Doppler spectrogram images," *Signals* 4(2):337-358, 2023. https://doi.org/10.3390/signals4020018
- [E10] B. Torvik, K. E. Olsen, H. Griffiths, polarimetric classification of birds and UAVs, *IEEE GRSL*, 2016.
- [E11] Dai et al., local mean decomposition features for UAV vs bird, *Remote Sensing* 14:2196, 2022 (secondary).
- [E12] Chen et al., L-band EMD + K-band STFT fusion for drone/bird recognition, *Journal of Radars* 14(5), 2025. https://doi.org/10.12000/JR25004
- [E13] G. Rilling, P. Flandrin, P. Gonçalvès, J. M. Lilly, "Bivariate empirical mode decomposition," *IEEE SPL* 14:936-939, 2007, https://doi.org/10.1109/LSP.2007.904710 ; T. Tanaka, D. P. Mandic, "Complex empirical mode decomposition," *IEEE SPL* 14:101-104, 2007, https://doi.org/10.1109/LSP.2006.882107
- [E14] Sung, Lee, CardRF drone classification with DFT vs HHT features and LightGBM, *J. Korea Soc. Comput. Inf.* 29(10):59, 2024. https://doi.org/10.9708/jksci.2024.29.10.059
- [E15] O. O. Medaiyese, M. Ezuma, A. P. Lauf, A. A. Adeniran, EMD/HHT + wavelet-packet features for RF UAS detection on CardRF, *IEEE J. Radio Freq. Identif.* 6, 2022. https://arxiv.org/abs/2107.04908
- [E16] Xu, He, Chen, Jiang, Song, EMD/EEMD-based drone RF identification, ICASSP 2021. https://doi.org/10.1109/ICASSP39728.2021.9414985
- [E18] Fu, Zhang, He, VMD-feature RF drone recognition on CardRF, *Drones* 10(2):124, 2026. https://doi.org/10.3390/drones10020124
- [E19] Tanveer et al., "From Lab to Field Trials: Real-Time Multimodel Drone Detection in Low-SNR Environments," *IEEE Aerospace and Electronic Systems Magazine*, Feb 2026 (secondary); reproduction https://github.com/jainarein/LowSNR-DroneRF-Reproduction
- [E20] Wang et al., EMD/EEMD/CEEMDAN/VMD feature comparison on CWRU bearings, *Scientific Reports* 14, 2024. https://pmc.ncbi.nlm.nih.gov/articles/PMC11668891/
- [E21] M. Ezuma, F. Erden, C. K. Anjinappa, O. Ozdemir, I. Guvenc, "Micro-UAV detection and classification from RF fingerprints using machine learning techniques," IEEE Aerospace Conf. 2019, https://arxiv.org/abs/1901.07703 ; "Detection and classification of UAVs using RF fingerprints in the presence of Wi-Fi and Bluetooth interference," *IEEE OJCOMS* 1:60-76, 2020, https://doi.org/10.1109/OJCOMS.2019.2955889
- [E22] O. O. Medaiyese, M. Ezuma, A. P. Lauf, I. Guvenc, wavelet transform analytics for RF-based UAV detection and identification, *Pervasive Mob. Comput.* 82:101569, 2022. https://arxiv.org/abs/2102.11894
- [E23] Fu, Lv et al., HHT-MFCC for crane-species call clustering, *J. Phys.: Conf. Ser.* 1693:012134, 2020.
- [E24] O. O. Ogundile, A. M. Usman, D. J. J. Versfeld, EMD-based features for Bryde's whale call classification, *JASA Express Lett.* 147:EL125, 2020. https://doi.org/10.1121/10.0000717
- [E25] de Souza et al., EMD stopping criteria for cicada sound detection, *Comput. Electron. Agric.*, 2022. https://doi.org/10.1016/j.compag.2022.107181
- [E26] O. K. Cura et al., EMD vs EEMD vs DWT for seizure EEG, *BioMed. Eng. OnLine* 19:10, 2020.
- [E27] M. O. Mendez et al., wavelet vs EMD for ECG-based sleep apnea detection, *Physiol. Meas.* 31:273, 2010.
- [E28] J. Kevric, A. Subasi, EMD/DWT/WPD for motor-imagery EEG, *Biomed. Signal Process. Control* 31:398-406, 2017.

### DroneRF and 60 GHz radar (F, K)

- [F1] M. S. Allahham, M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, A. Erbad, "DroneRF dataset: A dataset of drones for RF-based detection, classification and identification," *Data in Brief* 26:104313, 2019. https://pmc.ncbi.nlm.nih.gov/articles/PMC6727013/ ; data https://doi.org/10.17632/f4c2b4n755.1
- [F2] M. F. Al-Sa'd, A. Al-Ali, A. Mohamed, T. Khattab, A. Erbad, "RF-based drone detection and identification using deep learning approaches: An initiative towards a large open source drone database," *Future Gener. Comput. Syst.* 100:86-97, 2019. https://doi.org/10.1016/j.future.2019.05.007 ; code https://github.com/Al-Sad/DroneRF (Apache-2.0)
- [F3] S. Al-Emadi, F. Al-Senaid, "Drone detection approach based on radio-frequency using convolutional neural network," IEEE ICIoT 2020. https://doi.org/10.1109/ICIoT48696.2020.9089489
- [F4] T. Huynh-The et al., "RF-UAVNet: High-performance convolutional network for RF-based drone surveillance systems," *IEEE Access* 10, 2022. https://doi.org/10.1109/ACCESS.2022.3172787
<!-- DRONERF AGENT REFS PENDING -->
- [F8] D. Shulman, "How Much Do RF Drone Benchmarks Overstate? A Controlled Study and Theory of Data Leakage in UAV Signal Identification," arXiv:2607.01025, July 2026 (preprint). https://arxiv.org/abs/2607.01025
- [K1] S. Kang et al., "Millimeter Wave Radar Measurements: Distinguishing UAS and Birds Based on 60 GHz micro-Doppler Signatures," IEEE VTC2024-Fall, https://doi.org/10.1109/VTC2024-Fall63153.2024.10757735 ; dataset https://doi.org/10.21227/c8r5-6h79 ; M. Larrat, C. Sales, *Sensors* 25(3):721, 2025, https://doi.org/10.3390/s25030721 . Details in `datasets/uas_birds_60ghz/README.md`.

### Bird audio (Section 3.1)

- [B1] M. Lasseck, BirdCLEF 2024 working note, CLEF 2024 Working Notes, CEUR-WS Vol-3740, paper 199. https://ceur-ws.org/Vol-3740/paper-199.pdf
- [B2] V. Sydorskyi, G. Gonçalves, BirdCLEF+ 2025 2nd-place working note, CEUR-WS Vol-4038, paper 256. https://ceur-ws.org/Vol-4038/paper_256.pdf ; code https://github.com/VSydorskyy/BirdCLEF_2025_2nd_place
- [B3] B. Ghani, T. Denton, S. Kahl, H. Klinck, "Global birdsong embeddings enable superior transfer learning for bioacoustic classification," *Scientific Reports* 13:22876, 2023. https://doi.org/10.1038/s41598-023-49989-z ; arXiv:2307.06292
- [B4] Perch 2.0 technical report, arXiv:2508.04665 (2025). https://arxiv.org/abs/2508.04665 ; weights: Kaggle `google/bird-vocalization-classifier`, tooling https://github.com/google-research/perch-hoplite
- [B5] S. Kahl, C. M. Wood, M. Eibl, H. Klinck, "BirdNET: A deep learning solution for avian diversity monitoring," *Ecological Informatics* 61:101236, 2021. https://doi.org/10.1016/j.ecoinf.2021.101236 ; https://github.com/birdnet-team/BirdNET-Analyzer
- [B6] L. Rauch et al., "BirdSet: A large-scale dataset for audio classification in avian bioacoustics," ICLR 2025. https://arxiv.org/abs/2403.10380
- [B7] M. Hagiwara et al., "BEANS: The Benchmark of Animal Sounds," arXiv:2210.12300. https://github.com/earthspecies/beans

### Time-series classification (Section 3.2)

- [T1] M. Middlehurst, P. Schäfer, A. Bagnall, "Bake off redux: a review and experimental evaluation of recent time series classification algorithms," *Data Mining and Knowledge Discovery* 38:1958-2031, 2024. https://doi.org/10.1007/s10618-024-01022-1 ; arXiv:2304.13029
- [T2] A. Dempster, F. Petitjean, G. I. Webb, "ROCKET," *DMKD* 34:1454-1495, 2020. https://doi.org/10.1007/s10618-020-00701-z
- [T3] A. Dempster, D. F. Schmidt, G. I. Webb, "MiniRocket," KDD 2021. https://doi.org/10.1145/3447548.3467231 ; arXiv:2012.08791
- [T4] C. W. Tan, A. Dempster, C. Bergmeir, G. I. Webb, "MultiRocket," *DMKD* 36:1623-1646, 2022. https://doi.org/10.1007/s10618-022-00844-1
- [T5] A. Dempster, D. F. Schmidt, G. I. Webb, "Hydra: competing convolutional kernels for fast and accurate time series classification," *DMKD* 37:1779-1805, 2023. https://doi.org/10.1007/s10618-023-00939-3
- [T6] M. Middlehurst et al., "HIVE-COTE 2.0," *Machine Learning* 110:3211-3243, 2021. https://doi.org/10.1007/s10994-021-06057-9
- [T7] H. Ismail Fawaz et al., "InceptionTime," *DMKD* 34:1936-1962, 2020. https://doi.org/10.1007/s10618-020-00710-y
- [T8] C. H. Lubba et al., "catch22: CAnonical Time-series CHaracteristics," *DMKD* 33:1821-1852, 2019. https://doi.org/10.1007/s10618-019-00647-x
- aeon docs: https://www.aeon-toolkit.org/ ; sktime: https://www.sktime.net/ ; tsai: https://github.com/timeseriesAI/tsai

### Decomposition components as classifier input (Section 3.2)

- [D1] A. C. R. Klaar, L. O. Seman, V. C. Mariani, L. d. S. Coelho, "Random Convolutional Kernel Transform with Empirical Mode Decomposition for Classification of Insulators from Power Grid," *Sensors* 24(4):1113, 2024. https://doi.org/10.3390/s24041113
- [D2] A. Ahmed, Y. Serrestou, K. Raoof, J.-F. Diouris, "Empirical Mode Decomposition-Based Feature Extraction for Environmental Sound Classification," *Sensors* 22(20):7717, 2022. https://doi.org/10.3390/s22207717
- [D3] Yin et al., infrasound classification with STFT/CWT/HHT images, *Frontiers in Signal Processing*, 2026. https://doi.org/10.3389/frsip.2026.1844194 (not checked beyond the agent summary)
