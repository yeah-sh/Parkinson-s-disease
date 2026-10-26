# Methods

This document describes how every result in `reports/` was produced, and how the two-stage
research tool (`pdeeg train-final`, `pdeeg predict`, the Streamlit app) makes its predictions.
Every setting quoted here is in `configs/`, and each one has its reason written next to it there.
The results are in the reports this document links to.

## Contents

1. [Data](#1-data)
2. [What was pre-registered, and what was added](#2-what-was-pre-registered-and-what-was-added)
3. [Preprocessing](#3-preprocessing)
4. [Band power](#4-band-power)
5. [MFDFA](#5-mfdfa)
6. [Surrogates](#6-surrogates)
7. [Feature tables](#7-feature-tables)
8. [Group statistics (primary)](#8-group-statistics-primary)
9. [Classification (exploratory)](#9-classification-exploratory)
10. [The two-stage cascade (exploratory)](#10-the-two-stage-cascade-exploratory)
11. [The final models](#11-the-final-models)
12. [Prediction](#12-prediction)
13. [Software and reproducibility](#13-software-and-reproducibility)

## 1. Data

OpenNeuro ds002778, version 1.0.5 (doi:10.18112/openneuro.ds002778.v1.0.5, licence CC0):
resting-state
EEG of 16 healthy controls (one session, `hc`) and 15 people with Parkinson's disease (sessions
`off` and `on` medication), 46 recordings. The EEG was recorded with a 32-channel BioSemi system at
512 Hz. Recordings are read with MNE-BIDS, and only the 32 scalp channels of the biosemi32 montage
are kept. The citations the curators ask for are in the
[README](../README.md#citations-requested-by-the-dataset).

The ON sessions of `sub-pd6` and `sub-pd16` were exported from already preprocessed EEGLAB data
(`participants.tsv`), and the preprocessing QC found the same from the signals alone. They are
left out of the main OFF against ON analyses and of every model. A sensitivity analysis of the
group statistics keeps them.

The curators warn against presenting the classification of these patients against controls as a
finding, because a proper test split would hold fewer than 8 patients. The group statistics are
therefore the primary analysis, and everything that classifies is exploratory.

## 2. What was pre-registered, and what was added

| Analysis | Fixed in | When | What had been seen |
|---|---|---|---|
| Group statistics: features, contrasts, tests, correction | `configs/stats.yaml` | 2026-10-23 | No feature compared between groups or sessions |
| Classification tasks `pd_vs_hc` and `off_vs_on`, feature sets, models, grids, cross-validation, permutations | `configs/model.yaml` | 2026-10-23 | No classifier run |
| Tasks `pd_any_vs_hc` and `off_vs_on_single`, the cascade, its calibration rule | `configs/model.yaml` | 2026-10-26, after the pre-registration | No classifier run |
| The rule that picks each stage's final configuration (best balanced accuracy, ties by ROC AUC) | `configs/model.yaml` | 2026-10-26 | No classifier run |
| The rule for the uncertain bands | `configs/predict.yaml` | 2026-10-26 | No classifier run |
| The "better than chance" rule for the tool's stages | `configs/predict.yaml` | 2026-10-26 | The p- and q-values of `pd_vs_hc` and `off_vs_on` (so the q of the paired stage); nothing of the added tasks |
| The final configurations and the bands themselves | `configs/model.yaml` `final`, `configs/predict.yaml` `stages` | 2026-10-26, after classification and cascade | Filled in from what the rules above gave |

The two classification tasks in the pre-registration answer the study's questions. `pd_vs_hc`
asks PD OFF against HC with one recording per participant, and `off_vs_on` asks which of a
patient's two recordings is OFF. Neither fits a tool that labels a single recording. `pd_vs_hc`
never sees a patient ON medication, though a patient is usually ON in practice, and `off_vs_on`
needs both of a patient's recordings. Hence the added `pd_any_vs_hc` (stage 1) and
`off_vs_on_single` (stage 2), and the cascade that runs them in turn. `off_vs_on` stays as the
paired mode of stage 2.

## 3. Preprocessing

`pdeeg preprocess`, settings in `configs/preprocessing.yaml`. Each recording is kept continuous,
because MFDFA needs unbroken data, and nothing is cut out.

1. **Crop.** 180 s from trigger `1`, or from the file start where there is none. The trigger is
   undocumented, but most recordings carry it a few seconds in, with more than 180 s after it.
2. **Mains.** The mains frequency is estimated from each spectrum: 50 and 60 Hz are tested, and a
   peak counts at 3 times the neighbouring spectrum. It is logged next to the sidecar value. The
   pipeline uses the dataset's 60 Hz (`configs/data.yaml`).
3. **Filter.** A zero-phase FIR band-pass (firwin, Hamming) from 0.5 to 50 Hz, with transition
   bands of 0.5 and 5 Hz. The upper one puts the stop band at 55 Hz, below the mains.
4. **Reference.** The average of the 32 channels, since BioSemi recordings are reference-free.
5. **ICA.** Extended infomax (500 iterations, seed 97), with as many components as the data rank.
   It is fitted on a separate copy filtered 1 to 100 Hz and notched at the mains frequency, the
   way ICLabel expects, and then applied to the 0.5 to 50 Hz data. A component is removed when
   ICLabel's most probable class is eye blink, muscle artifact, heart beat or line noise with
   probability at least 0.8.
6. **Bad stretches.** A 1 s window in which any channel exceeds 150 µV peak-to-peak, or stays
   under 1 µV, is annotated BAD. It is marked, not removed.
7. **QC** ([`qc_preprocessing.md`](qc_preprocessing.md)). A recording is flagged when more than
   10 % of it is bad, when its longest clean stretch is under 60 s, or when a signal metric lies
   more than 3 robust standard deviations from the median of all recordings. A flag is a reason
   to look, not an exclusion.

## 4. Band power

`configs/features/psd.yaml`. Welch spectra with 2 s Hann segments overlapping by half, from 1 to
48 Hz. Each band's power is divided by the total power from 1 to 48 Hz, and its log10 is taken. The
bands are contiguous: delta 1 to 4, theta 4 to 8, alpha 8 to 13, beta 13 to 30 and gamma 30 to
48 Hz. Like every feature, band power is computed on all segments (`full`) and on the segments
that touch no bad stretch (`clean`).

## 5. MFDFA

`pdeeg features`, settings in `configs/features/mfdfa.yaml`, implementation in
`pdeeg.features.mfdfa`. The implementation is checked on synthetic signals with known exponents,
and against the PyPI `MFDFA` package, in
[`notebooks/01_mfdfa_validation.ipynb`](../notebooks/01_mfdfa_validation.ipynb) and the tests.

**Series.** Two kinds of series are analysed. One is the cleaned EEG itself ("broadband"). The
other is the amplitude envelope in theta (4 to 8 Hz), alpha (8 to 13 Hz) and beta (13 to 30 Hz).
Each envelope is the magnitude of the Hilbert transform after a zero-phase FIR band-pass with 2 Hz
transitions, and 1 s is trimmed from each end of it.

**Fluctuation function.** For moment orders q from -5 to 5 in steps of 0.5, each window's profile
is detrended with a polynomial of order 1. Windows whose residual variance is negligible against
the median of their scale are dropped, so that they cannot dominate negative q. No window is
longer than a tenth of the series. h(q) is the slope of log Fq(s) against log s over a fit range,
on 12 scales log-spaced within it. From h(q) follow the singularity spectrum f(alpha) and five
features:

| Feature | Definition |
|---|---|
| `h2` | h(2), the Hurst exponent of ordinary DFA |
| `delta_h` | h(q_min) - h(q_max) |
| `delta_alpha` | Width of the spectrum, max(alpha) - min(alpha) |
| `alpha0` | The alpha at which f(alpha) peaks |
| `asymmetry` | (left - right) / (left + right), with left = alpha0 - min(alpha) and right = max(alpha) - alpha0 |

**Fit ranges.** These were chosen before any feature was computed, from a label-blind scaling
inspection ([`mfdfa_scaling.md`](mfdfa_scaling.md), `pdeeg mfdfa-scaling`). Eight recordings were
picked at random (seed 20261009) and shown only as R1 to R8, with no subject, session or group
label read. Local slopes of log Fq(s) against log s were compared over 50 scales with noise put
through the same filters: 1/f^beta noise through the preprocessing band-pass, and white noise
through each band envelope, in 40 realisations. The chosen ranges are:

| Series | Range | Features used |
|---|---|---|
| Broadband, short | 0.045 to 0.09 s | `h2` only: on this range filtered monofractal noise is as wide as the EEG |
| Broadband, long | 0.3 to 1 s | all five |
| Envelopes, main | 2 to 17.8 s | all five |

The reason for every end of every range is written next to it in `configs/features/mfdfa.yaml`.

**Bad windows.** For `clean` values, every window that touches a bad stretch is left out where it
stands; data is never cut out and joined. For an envelope, the bad stretch is first widened by
1 s on both sides. A scale with fewer than 8 clean windows is left out of the fit, and no fit is
made on fewer than 8 scales. A fit is reported as poor when its worst R² over q is below 0.9
([`qc_mfdfa.md`](qc_mfdfa.md)).

## 6. Surrogates

Every channel and series is compared with 20 IAAFT surrogates and 20 shuffled surrogates (seed
20261009), made from the whole series and analysed on the same fit ranges. IAAFT keeps the
amplitude distribution and the power spectrum, so the width that survives it is what linear
correlations and the distribution explain. Shuffling keeps the distribution alone. The comparison
is in [`qc_mfdfa.md`](qc_mfdfa.md). No model feature uses the surrogates.

## 7. Feature tables

The feature tables are written as Parquet, one long table and one wide table per kind. The group
statistics use one value per channel. The classifiers use region averages: the mean over the
channels of each of 5 scalp regions (frontal, central, temporal, parietal, occipital; listed in
`configs/data.yaml`), from the `clean` segments. That gives 25 band-power features,
105 MFDFA features and 130 for both.

## 8. Group statistics (primary)

`pdeeg stats`, pre-registered in `configs/stats.yaml`, results in
[`stats_results.md`](stats_results.md).

- **Features.** The three primary features are the beta-envelope `h2` and `delta_alpha`, and the
  broadband `delta_alpha` on the 0.3 to 1 s range. Every other MFDFA and band-power feature is
  exploratory and corrected within its own family.
- **Contrasts.** PD OFF against HC, between participants, and OFF minus ON within patients without
  `sub-pd6` and `sub-pd16`. A sensitivity analysis keeps all 15 pairs, and another repeats PD OFF
  against HC on residuals after a least-squares fit of every channel on age and sex.
- **Test.** A cluster-based permutation test over channels with the biosemi32 neighbour template.
  Each channel gets Student's t, two-sided: two-sample with pooled variance, or one-sample on the
  differences. Channels with |t| above the p = 0.05 threshold form clusters, and a cluster's mass
  is the sum of its t-values. There are 10000 permutations, or every one of the 4096 sign flips
  for 13 pairs, with seed 20261023.
- **Correction.** Benjamini-Hochberg over the cluster p-values of the primary family in both
  contrasts. A test without a cluster counts as one p-value of 1, and a cluster is significant at
  q < 0.05.
- **Effect sizes.** Hedges' g per channel for PD against HC, and g_z for OFF against ON.
- **Confounds.** Age is compared with Welch's t-test and sex with Fisher's exact test.

## 9. Classification (exploratory)

`pdeeg classify`, settings in `configs/model.yaml`, results in
[`classification_results.md`](classification_results.md).

**Tasks.**

| Task | Status | Classes | Recordings and weights | Permutation |
|---|---|---|---|---|
| `pd_vs_hc` | Pre-registered | PD OFF against HC | One per participant | Across participants |
| `off_vs_on` | Pre-registered | OFF against ON, within patients | Two per patient, centred on the patient's mean; `sub-pd6` and `sub-pd16` left out | Within patients |
| `pd_any_vs_hc` | Added (stage 1) | PD OFF or ON against HC | Both of a patient's recordings at 0.5 each, so each participant counts once; ON sessions of `sub-pd6` and `sub-pd16` left out | Across participants |
| `off_vs_on_single` | Added (stage 2) | OFF against ON, one recording at a time | Two per patient, not centred; `sub-pd6` and `sub-pd16` left out | Within patients |

**Feature sets and models.** Band power, MFDFA, both, and age and sex alone as a confound
baseline. Age and sex are not run for the within-patient tasks. Each is paired with three models,
each with its own grid:

- logistic regression (saga): C 0.01, 0.1 and 1, l1_ratio 0, 0.5 and 1;
- a linear SVM: C 0.001, 0.01, 0.1 and 1;
- gradient boosting: 100 trees of depth 1 or 2, learning rate 0.1.

Every model sits behind a `StandardScaler` in one scikit-learn `Pipeline`.

**Nested cross-validation.** The outer loop is stratified 5-fold, grouped by participant. Inside
every outer training set, a grouped, stratified 3-fold grid search scored by balanced accuracy
tunes the whole pipeline. A wrapped splitter raises `LeakageError` when any outer or inner split
puts one participant on both sides. Leave-one-subject-out was timed first and would have taken
weeks with the permutation test (the timing is in `configs/model.yaml`). Weighted tasks route
their sample weights into the fit, into the inner-loop scores and into every metric.

**Metrics and uncertainty.** The metrics are balanced accuracy, ROC AUC, sensitivity and
specificity of the pooled out-of-fold predictions.

- **Intervals:** 2000 bootstrap resamples of participants, with all their recordings, at 95 %.
- **p-values:** 1000 relabellings, each through the whole nested procedure, with
  p = (1 + number at least as good) / (1 + relabellings).
- **q-values:** Benjamini-Hochberg over the balanced-accuracy p-values of a task's combinations.
- **Other splits:** the procedure is repeated on 10 other outer splits, to show what the split
  alone does.
- **Audit:** a pre-specified rule flags any balanced accuracy above 0.85 for a leakage
  and confound audit, and holds it back as a result until the audit is written in the report.

## 10. The two-stage cascade (exploratory)

`pdeeg cascade`, settings in `configs/model.yaml` (`cascade`), results in
[`cascade_results.md`](cascade_results.md).

- **Configuration.** Band power and MFDFA (130 features) with logistic
  regression, for both stages. This was fixed before any result, and nothing was picked on these
  data.
- **In every outer fold** (grouped by participant, stratified by patient or control), stage 1
  (`pd_any_vs_hc`) is tuned and fitted on the training participants. It is weighted so that each
  participant counts once. Stage 2 (`off_vs_on_single`) is tuned and fitted on the training
  patients' recordings only. Each has its own grouped 3-fold inner loop. A held-out recording is
  called HC when stage 1 says so, and PD_OFF or PD_ON by stage 2 otherwise. Stage 2 also calls
  the recordings that stage 1 sends to the controls, so that its errors on true patients can be
  measured apart from stage 1's. Every split of both stages is checked for shared participants.
- **Metrics.** A class's sensitivity is the share of its recordings called that class. Macro
  balanced accuracy is their mean, with chance at 1/3. Stage 1's balanced accuracy weighs each
  participant once, and stage 2's is computed on the true patients' recordings.
- **Intervals** come from 2000 bootstrap resamples of participants.
- **p-values** come from 1000 relabellings of participants, each through the whole cascade with
  its inner loops. Patient or control is shuffled among the participants, and every participant
  who becomes a patient gets OFF and ON by a coin flip.
- **"Beats chance"** in the report means p < 0.05 and a 95 % interval that excludes chance.
- **Calibration.** Out-of-fold probabilities are kept raw, and Platt-scaled (sigmoid) inside every
  outer training set on that stage's grouped inner-fold predictions. The rule, fixed before any
  classifier was run, calls a stage poorly calibrated if:
  - its Brier score is not below that of predicting the prevalence; or
  - the interval of its calibration slope leaves out 1; or
  - the interval of its calibration in the large leaves out 0.

  The slope is b in logit P(y) = a + b logit(p). Calibration in the large is a with b held at 1.
  The reliability diagrams use 5 bins of equal count, with Wilson intervals. The three-class
  Brier score uses P(HC) = 1 - p1, P(OFF) = p1 p2 and P(ON) = p1 (1 - p2).

## 11. The final models

`pdeeg train-final`, code in `pdeeg.predict.final`.

1. **Choice.** For every stage (stage 1: `pd_any_vs_hc`; stage 2 single: `off_vs_on_single`;
   stage 2 paired: `off_vs_on`), the feature set and model with the best cross-validated balanced
   accuracy of its task, ties broken by ROC AUC, age and sex alone left out. It is written into
   `final:` in `configs/model.yaml`, and `train-final` refuses any other. Being picked on the
   predictions it is scored on, its score is optimistic. The models' metadata carries the
   configuration fixed in advance, and the cascade's estimate, next to it.
2. **Out-of-fold check.** The stage's nested cross-validation is run again with the ablation's own
   seed. Its decisions must equal those the ablation stored, or `train-final` stops. It now also
   gives Platt-scaled probabilities.
3. **Uncertain band.** From those out-of-fold probabilities, participants are resampled 2000
   times, and each resample's logistic recalibration maps a probability p to what it turned out
   to mean, P(y = 1 | p). A probability keeps its label only when the whole 95 % interval of
   P(y = 1 | p) lies on the same side of the 0.5 threshold as p. Every other probability is in
   the band, tried on a grid of step 0.005. A stage whose probabilities cannot be told from a
   coin flip gets the band from 0 to 1, and never labels anything. For the paired stage, the two
   probabilities of a pair sum to 1, so the recalibration is symmetric about 0.5 and the rule only
   asks whether its slope is positive. `train-final` saves nothing until `configs/predict.yaml`
   holds the bands the rule gives.
4. **Fit.** The model is tuned by the grouped inner loop on all of the task's participants and
   refitted on all of them. Platt scaling is then fitted on its grouped inner-fold predictions,
   and the model is refitted on all rows, so the result is the same model with rescaled
   probabilities.
5. **Better than chance.** A stage counts as better than chance when the q-value of its
   configuration, which is adjusted for picking the best of the task's combinations, is below
   0.05, and the whole 95 % interval of its balanced accuracy is above 0.5. This rule was fixed
   after the results of the two pre-registered tasks had been seen (section 2). The raw p-value
   is shown next to it.
6. **Saved** to `models/<stage>/`:
   - `model.joblib`;
   - `feature_columns.json`;
   - `oof_predictions.parquet`;
   - `metadata.json`, which holds the digests of the feature settings and of `model.yaml` and
     `predict.yaml`, the dataset id and version, the git commit, the library versions, the
     cross-validated metrics with their intervals, the thresholds and band, and a model card.

   The training recordings are listed in `models/training_recordings.parquet`, with the SHA-256
   of each file and of its signal. Every run is also recorded in MLflow.

## 12. Prediction

`pdeeg predict` and the app, code in `pdeeg.predict`. The app calls the same function
(`pdeeg.predict.predictor.predict_files`).

1. **Input checks**, before anything else. These are the file, the channels, the sampling rate,
   the duration and the amplitude, with the limits in `configs/predict.yaml`; the
   [README](../README.md#what-an-input-must-be) lists them. Channels are matched to the montage by
   name. A recording at another rate is resampled to 512 Hz, with a note. A recording with too
   little EEG after the crop start, or with implausible amplitudes, is refused.
2. **Training recordings.** A recording is recognised as one the final models were trained on by
   its BIDS subject and session in ds002778, by the SHA-256 of its file, or by a hash of its
   signal, which survives renaming and header edits. It gets its stored out-of-fold predictions
   instead of the final models', with an explanation.
3. **Features.** The recording is cleaned (`clean_recording`, the code used for training, writing
   and reading the cleaned data the same way). Its MFDFA and band power are computed, and its wide
   tables built, by the training code with the training settings, in a working directory. The
   MFDFA surrogates are skipped, because no feature uses them and the runs of the original series
   do not depend on them. A digest of every setting that affects the features is compared with the
   one stored with the models, and a mismatch is refused. A test (marked `dataset`) checks on a
   ds002778 recording that this path gives the values of the training tables.
4. **Stages.** Stage 1 gives P(patient). Stage 2 on one recording gives P(OFF), and runs only when
   stage 1 calls a patient outside its uncertain band. With two recordings of one patient
   (`--paired`), the pair is centred as in `off_vs_on` training. The paired model's two
   probabilities are normalised to P(first is OFF) = p1 / (p1 + p2).
5. **Output.** For every stage it gives the probability, its label or "uncertain", the threshold
   and band, and the cross-validated metrics with the "better than chance" verdict. It also gives
   the features that moved the prediction: for a linear model, coefficient times standardised
   value, and for gradient boosting its global importances. The output is JSON plus a readable
   summary.

## 13. Software and reproducibility

- **Software.** Python 3.11 or 3.12, with exact pins in `pyproject.toml`: MNE 1.13.2, MNE-BIDS
  0.20.0, mne-icalabel 0.10.0, scikit-learn 1.9.1, numpy 2.4.6, scipy 1.17.1, pandas 3.0.6 and
  MLflow 3.17.0.
- **Caching.** Every step writes a stamp of its settings, inputs and outputs, and is skipped while
  they are unchanged.
- **Data.** Tables are Parquet and models are joblib.
- **Tracking.** Every classification and cascade evaluation, and every final model, is an MLflow
  run (`mlflow.db`, artifacts in `mlruns/`).
- **Tests.** `pytest -m "not dataset"` runs on synthetic data. It covers leakage across
  participants in every task and in both stages of the cascade, a planted three-class effect, the
  input checks, and the app's upload-to-prediction flow (Streamlit AppTest).
- **Docker.** The `Dockerfile` builds an image that runs every command on a mounted `data/`
  directory.
