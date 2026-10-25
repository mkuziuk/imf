BirdCLEF2023 species balanced accuracy (%), without/with log_scale.

|Method|LGBM|XGB|ExtraTrees|MiniRocket|MultiRocket|CNN|STFT|
|---|---|---|---|---|---|---|---|
|Raw|46.8/46.0|46.2/45.5|45.4/45.1|50.5/50.5|48.1/48.0|52.6/51.9|51.7/52.2|
|EMD|50.5/50.1|50.3/50.3|47.3/47.1|50.7/50.6|47.5/47.5|51.2/50.6|n/a|
|Linear IMF|52.8/52.4|52.8/52.8|49.8/50.0|52.0/51.8|49.8/49.8|52.6/52.3|n/a|
|IRMF|52.9/52.8|52.9/52.8|50.3/50.3|52.1/52.1|50.4/50.6|52.6/52.6|n/a|
|Converged IRMF|53.1/53.0|52.9/52.9|50.1/50.4|52.1/52.1|49.9/50.1|52.2/52.2|n/a|

Passerine:

|Method|LGBM|XGB|ExtraTrees|MiniRocket|CNN|STFT|
|---|---|---|---|---|---|---|
|Raw|77.4/77.2|77.3/77.3|78.3/78.2|76.1/76.3|81.5/81.0|78.3/78.7|
|EMD|79.7/79.9|79.7/79.5|77.5/77.2|76.1/76.1|78.7/79.8|n/a|
|Linear IMF|81.8/81.7|81.7/81.7|80.2/79.9|77.0/77.0|81.7/80.0|n/a|
|IRMF|81.3/81.1|81.1/81.0|80.1/80.3|77.3/77.5|78.6/80.5|n/a|
|Converged IRMF|81.3/81.0|81.0/81.0|80.1/80.3|77.4/77.3|79.8/79.8|n/a|

Paired Δpp;Holm q, without/with scale. I=IRMF,E=EMD,L=linear IMF,R=raw,C=converged IRMF.

|species|I−E|I−L|I−R|L−R|E−R|
|---|---|---|---|---|---|
|LGBM|+2.44;0.5/+2.70;0.0049|+0.13;1/+0.35;1|+6.09;3.9e-05/+6.79;4.1e-06|+5.96;0.00013/+6.45;0.0013|+3.65;0.045/+4.10;0.0066|
|XGB|+2.60;0.033/+2.50;0.19|+0.10;1/+0.00;1|+6.67;9e-06/+7.28;8.6e-05|+6.57;1.5e-05/+7.27;2.1e-05|+4.07;0.00031/+4.78;0.0038|
|ExtraTrees|+3.01;0.69/+3.18;0.72|+0.50;1/+0.27;1|+4.94;0.016/+5.22;0.0072|+4.44;0.21/+4.95;0.026|+1.93;0.15/+2.05;0.089|
|MiniRocket|+1.31;1/+1.53;1|+0.10;1/+0.34;1|+1.55;1/+1.64;1|+1.45;1/+1.30;1|+0.24;1/+0.12;1|
|MultiRocket|+2.88;1/+3.09;0.69|+0.55;1/+0.79;1|+2.35;0.15/+2.57;0.76|+1.79;1/+1.78;0.54|-0.53;1/-0.52;1|
|CNN|+1.38;1/+2.00;1|-0.05;1/+0.30;1|-0.03;1/+0.71;1|+0.02;1/+0.42;1|-1.40;1/-1.28;1|

|passerine|I−E|I−L|I−R|L−R|E−R|
|---|---|---|---|---|---|
|LGBM|+1.61;0.015/+1.28;0.1|-0.54;1/-0.59;1|+3.87;0.00047/+3.89;0.00028|+4.41;0.002/+4.49;0.0017|+2.26;0.15/+2.61;0.0026|
|XGB|+1.43;0.085/+1.47;0.24|-0.63;1/-0.73;1|+3.76;0.015/+3.71;0.003|+4.38;0.027/+4.44;0.0052|+2.32;0.81/+2.24;0.28|
|ExtraTrees|+2.69;0.0019/+3.11;0.0056|-0.07;1/+0.35;1|+1.87;0.16/+2.10;0.17|+1.94;0.11/+1.75;0.94|-0.82;1/-1.00;1|
|MiniRocket|+1.17;1/+1.35;1|+0.29;1/+0.50;1|+1.16;1/+1.14;1|+0.87;1/+0.65;1|-0.01;1/-0.20;1|
|CNN|-0.12;1/+0.66;1|-3.05;1/+0.47;1|-2.83;1/-0.50;1|+0.22;1/-0.97;1|-2.71;1/-1.16;1|

Primary tests are Nadeau-Bengio; Holm families contain 120/100 comparisons per metric. [All species pairs/CIs, NB/Wilcoxon p/q](paired_all_metrics_species.md); [passerine](paired_all_metrics_passerine.md). [Macro-F1/accuracy](full_metrics_species.md), [passerine](full_metrics_passerine.md); [STFT species](stft_reference/results_folds_species.csv), [STFT passerine](stft_reference/results_folds_passerine.csv).

Phase-2's EMD gap shrinks; I−E means stay positive in 12/12 settings. Holm support: LGBM +log, XGB −log. IRMF does not establish a robust-specific advantage over linear IMF. Raw CNN remains competitive. Passerine trees also favor IRMF over EMD; CNN does not confirm that ordering.

[Recording BA](recording_comparisons_species.csv), LGBM, descriptive best shared family, pooled per seed: R=59.2/57.7,E=62.7/62.0,L=64.7/64.0,I=65.1/64.3,C=64.6/64.4. I−E q=1/1.

15-fold LightGBM 9-slot IRMF=52.7/52.3; fusion=53.1/52.9. Most mean EMD lead survives 180 features; fusion versus IRMF has q=1. [Exploratory checks](feature_checks/comparisons.csv). LGBM IRMF split importance: finest residual 11.4%, w7/w9 ≈8.4%; [all scales](scale_importance_species.csv).

Budgets: trees15 folds; species CNN15, passerine CNN5; MiniRocket5 both, MultiRocket5 species; STFT15. Boosting≤500, ExtraTrees300, CNN≤30 epochs, MiniRocket2016/MultiRocket1344 features, 10 head epochs. Methods share budgets/splits. No converged/default difference is significant after Holm.

Primary wall minutes under concurrent load: MultiRocket=28.3,STFT=21.5,XGB=95.9,LGBM=26.5,CNN=60.3,ExtraTrees=14.2,MiniRocket=66.3,checks=51.9; total elapsed=154.0. [Per-input costs](fit_costs_species.csv). Cached CPU s/frame: E=.01165,L=.00632,I=1.32049,C=.05792.

More compute: 15-fold/10k-feature ROCKET, matched-channel controls, frequency-binned EMD and longer clips. [Reproduction/files](REPRODUCE.md). Shared ml/ unchanged; local confidence-export/pooling fixes documented.
