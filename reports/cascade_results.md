# Two-stage cascade (exploratory)

Written by `pdeeg cascade`. Settings: `configs/model.yaml` (section `cascade`, tasks `pd_any_vs_hc` and `off_vs_on_single`), SHA-256 `7f2b1081ec2a0fc6ba242673973584da215a7ebfb4f7a9a7eae58e067dfee0d9`. The cascade and both of its tasks were added on 2026-10-26, after the pre-registration of 2026-10-23 and before any classifier had been run on the data. The evaluation is an MLflow run (`cascade/psd_mfdfa/logistic_regression`) in experiment `pdeeg`.

Like all classification here, this is secondary to the group statistics (`reports/stats_results.md`).

## Summary in plain language

- **The cascade** called every recording HC, PD_OFF or PD_ON. Macro balanced accuracy (the mean of the three sensitivities; chance 0.33) was 0.47 [0.32, 0.61]: not shown to beat chance: the interval [0.32, 0.61] includes chance (0.33).
- **Stage 1**, patient or control: balanced accuracy 0.58 [0.40, 0.74], not shown to beat chance: p = 0.163 and the interval [0.40, 0.74] includes chance (0.50).
- **Stage 2**, OFF or ON, on the true patients' recordings whatever stage 1 said: balanced accuracy 0.71 [0.58, 0.84], above chance (0.50): p = 0.031 and the interval excludes chance.
- **Where patients' recordings went:** 39% were called controls by stage 1, 18% were called patients in the wrong state by stage 2, and 43% were called right. 44% of the controls' recordings were called patients'.
- **Calibration:** by the rule fixed in advance, stage 1 gave poorly calibrated probabilities, so Platt-scaled probabilities are the ones to use there; after scaling, the rule finds stage 1 well calibrated.
- **What this is not.** Not a diagnostic test and not a medication monitor: the sample is small, the intervals are wide, there is no independent test set, and the dataset's curators warn against presenting classification of these patients as diagnostic.

## Design

- **Why it was added.** The tool is meant to answer two questions about one recording: is it a patient's or a control's, and if a patient's, OFF or ON medication? The pre-registered tasks do not fit. `pd_vs_hc` uses only OFF recordings, while a patient seen in practice is usually ON. `off_vs_on` centres each patient's features on their own two recordings, so it needs both and cannot label one recording on its own. Hence two tasks: `pd_any_vs_hc` (stage 1) and `off_vs_on_single` (stage 2, single-recording mode; `off_vs_on` stays as the paired mode).
- **Recordings:** 31 participants, 44 recordings: HC 16 (16 participants), PD_OFF 15 (15 participants), PD_ON 13 (13 participants). Left out: pd6 on, pd16 on (exported already preprocessed); those patients' other recordings stay in. Stage 2 never trains on pd6, pd16, for the same reason; their OFF recordings are still called by the cascade when held out.
- **Each outer fold** (stratified group kfold, 5 folds, grouped by participant, stratified by patient or control): stage 1 is tuned and fitted on all recordings of the training participants, weighted so that each participant counts once (0.5 per recording of a patient with two); stage 2 on the training patients' recordings only. Each has its own 3-fold inner loop, grouped by participant and scored by balanced accuracy. Every held-out recording is called HC when stage 1 says so, else PD_OFF or PD_ON by stage 2. Stage 2 also calls the recordings stage 1 sends to the controls, so its errors on true patients are measured apart from stage 1's. Every split is checked at run time to share no participant between training and test, for both stages.
- **Features and model:** band power + MFDFA, 130 features (clean segments, region level), and logistic regression for both stages, with the grid C [0.01, 0.1, 1.0], l1_ratio [0.0, 0.5, 1.0]. This one configuration was fixed before any result; nothing was picked on these data.
- **Metrics:** sensitivity of a class is the share of its recordings called that class; macro balanced accuracy is their mean (chance 1/3). Stage 1's balanced accuracy and ROC AUC weigh each participant once; stage 2's are on the true patients' recordings.
- **Intervals:** 2000 bootstrap resamples of participants (with all their recordings), 95%, on the out-of-fold calls.
- **p-values:** 1000 relabellings at the level of participants, each through the whole cascade, inner loops included. Patient or control is shuffled among the participants; every participant who is then a patient gets OFF and ON by a coin flip (one of each with two recordings, OFF or ON with one). p = (1 + number at least as good) / (1 + relabellings). Stage 1's and stage 2's p-values come from the same relabellings.
- **Other splits:** the cascade repeated on 10 other outer splits, to show how much the split alone moves the scores.

## Results

### Three classes

| True class \ called | HC | PD_OFF | PD_ON | Sensitivity [95 % CI] |
|---|---|---|---|---|
| HC | 9 | 3 | 4 | 0.56 [0.31, 0.80] |
| PD_OFF | 6 | 7 | 2 | 0.47 [0.21, 0.74] |
| PD_ON | 5 | 3 | 5 | 0.38 [0.12, 0.67] |

Macro balanced accuracy 0.47 [0.32, 0.61], p = 0.035 (95th percentile of the null: 0.46; chance 0.33).

![Confusion matrix of the cascade](figures/cascade/confusion.png)

### Each stage inside the cascade

| Stage | Balanced accuracy [95 % CI] | p | ROC AUC [95 % CI] | p (AUC) | Null 95th percentile (BA) |
|---|---|---|---|---|---|
| Stage 1: PD OFF / PD ON vs HC | 0.58 [0.40, 0.74] | 0.163 | 0.57 [0.37, 0.76] | 0.226 | 0.66 |
| Stage 2: PD OFF vs PD ON, single recordings (true patients) | 0.71 [0.58, 0.84] | 0.031 | 0.72 [0.52, 0.89] | 0.050 | 0.68 |

### Where the errors come from

| Recordings | Share [95 % CI] |
|---|---|
| Patients' called controls (stage 1 missed) | 0.39 [0.16, 0.63] |
| Patients' called patients in the wrong state | 0.18 [0.06, 0.30] |
| Patients' called right | 0.43 [0.25, 0.62] |
| Controls' called patients (stage 1 false alarm) | 0.44 [0.20, 0.69] |

### Other outer splits

| Metric | Minimum | Median | Maximum |
|---|---|---|---|
| Three classes, macro balanced accuracy | 0.44 | 0.47 | 0.50 |
| Stage 1, balanced accuracy | 0.61 | 0.67 | 0.69 |
| Stage 2 on true patients, balanced accuracy | 0.54 | 0.57 | 0.75 |

![Permutation nulls](figures/cascade/permutation_nulls.png)

## Calibration

Out-of-fold probabilities: P(patient) from stage 1 for every recording (weighted as stage 1), P(OFF) from stage 2 for the true patients' recordings. Platt scaling (sigmoid) was fitted inside every outer training set, on the stage's grouped inner-fold predictions, and computed for every run whatever the result. The rule, fixed before any classifier was run: a stage is poorly calibrated if its Brier score is not below that of predicting the prevalence, or the interval of its calibration slope leaves out 1, or that of calibration in the large leaves out 0.

| Stage | Probabilities | Brier | Brier of the prevalence | Brier skill | Calibration in the large | Calibration slope | Rule |
|---|---|---|---|---|---|---|---|
| Stage 1 | raw | 0.277 [0.203, 0.355] | 0.250 | -0.111 [-0.542, 0.158] | -0.207 [-1.184, 0.709] | 0.125 [-0.527, 1.032] | Brier score not below that of predicting the prevalence |
| Stage 1 | Platt | 0.240 [0.191, 0.292] | 0.250 | 0.040 [-0.272, 0.218] | -0.079 [-0.902, 0.720] | 0.683 [-0.424, 1.849] | well calibrated |
| Stage 2 | raw | 0.211 [0.156, 0.267] | 0.249 | 0.152 [-0.076, 0.370] | 0.007 [-0.319, 0.349] | 0.972 [0.048, 3.235] | well calibrated |
| Stage 2 | Platt | 0.260 [0.229, 0.288] | 0.249 | -0.044 [-0.162, 0.078] | -0.096 [-0.508, 0.335] | 0.256 [-0.420, 1.196] | Brier score not below that of predicting the prevalence |

Three-class Brier score of the cascade's probabilities (P(HC) = 1 - p1, P(OFF) = p1 p2, P(ON) = p1 (1 - p2); 0 is perfect, 2 the worst): raw 0.695, Platt-scaled 0.687; predicting the class shares for everyone gives 0.664.

![Reliability diagrams](figures/cascade/reliability.png)

<details><summary>Stage 1: PD OFF / PD ON vs HC, raw probabilities by bin</summary>

| Bin | Recordings | Mean probability | Observed [95 % CI] |
|---|---|---|---|
| 1 | 9 | 0.24 | 0.33 [0.11, 0.67] |
| 2 | 9 | 0.42 | 0.38 [0.13, 0.73] |
| 3 | 9 | 0.55 | 0.64 [0.27, 0.89] |
| 4 | 9 | 0.68 | 0.67 [0.30, 0.90] |
| 5 | 8 | 0.85 | 0.45 [0.15, 0.79] |

</details>

<details><summary>Stage 1: PD OFF / PD ON vs HC, Platt-scaled probabilities by bin</summary>

| Bin | Recordings | Mean probability | Observed [95 % CI] |
|---|---|---|---|
| 1 | 9 | 0.28 | 0.38 [0.13, 0.73] |
| 2 | 9 | 0.42 | 0.33 [0.11, 0.67] |
| 3 | 9 | 0.50 | 0.64 [0.27, 0.89] |
| 4 | 9 | 0.61 | 0.38 [0.13, 0.73] |
| 5 | 8 | 0.77 | 0.80 [0.38, 0.96] |

</details>

<details><summary>Stage 2: PD OFF vs PD ON, single recordings (true patients), raw probabilities by bin</summary>

| Bin | Recordings | Mean probability | Observed [95 % CI] |
|---|---|---|---|
| 1 | 6 | 0.29 | 0.17 [0.03, 0.56] |
| 2 | 6 | 0.44 | 0.50 [0.19, 0.81] |
| 3 | 6 | 0.55 | 0.83 [0.44, 0.97] |
| 4 | 5 | 0.62 | 0.40 [0.12, 0.77] |
| 5 | 5 | 0.84 | 0.80 [0.38, 0.96] |

</details>

<details><summary>Stage 2: PD OFF vs PD ON, single recordings (true patients), Platt-scaled probabilities by bin</summary>

| Bin | Recordings | Mean probability | Observed [95 % CI] |
|---|---|---|---|
| 1 | 6 | 0.37 | 0.33 [0.10, 0.70] |
| 2 | 6 | 0.49 | 0.67 [0.30, 0.90] |
| 3 | 6 | 0.54 | 0.67 [0.30, 0.90] |
| 4 | 5 | 0.66 | 0.40 [0.12, 0.77] |
| 5 | 5 | 0.79 | 0.60 [0.23, 0.88] |

</details>

**Secondary: the cascade called from the Platt-scaled probabilities** (the rule flags stage 1). The same models, rescaled, so this changes where the 0.5 threshold falls, not how the recordings are ranked. It was not permutation-tested.

| True class \ called | HC | PD_OFF | PD_ON | Sensitivity [95 % CI] |
|---|---|---|---|---|
| HC | 10 | 5 | 1 | 0.62 [0.38, 0.86] |
| PD_OFF | 8 | 3 | 4 | 0.20 [0.00, 0.43] |
| PD_ON | 5 | 6 | 2 | 0.15 [0.00, 0.36] |

Macro balanced accuracy 0.33 [0.21, 0.44]; stage 1 0.58 [0.40, 0.74]; stage 2 on patients 0.53 [0.40, 0.67].

## The stages on their own

The same configuration (band power + MFDFA, logistic regression) in the classification ablation (`reports/classification_results.md`, which has every feature set and model), each task with its own outer split and permutation test. The pre-registered `pd_vs_hc` and the paired `off_vs_on` are shown for comparison.

| Task | Name | Participants | Recordings | Balanced accuracy [95 % CI] | p | q | ROC AUC [95 % CI] |
|---|---|---|---|---|---|---|---|
| PD OFF / PD ON vs HC | `pd_any_vs_hc` | 31 | 44 | 0.68 [0.52, 0.83] | 0.044 | 0.176 | 0.74 [0.55, 0.90] |
| PD OFF vs HC | `pd_vs_hc` | 31 | 31 | 0.61 [0.43, 0.78] | 0.123 | 0.295 | 0.68 [0.48, 0.86] |
| PD OFF vs PD ON, single recordings | `off_vs_on_single` | 13 | 26 | 0.62 [0.46, 0.77] | 0.148 | 0.266 | 0.64 [0.42, 0.83] |
| PD OFF vs PD ON, within patients | `off_vs_on` | 13 | 26 | 0.62 [0.38, 0.85] | 0.234 | 0.263 | 0.66 [0.35, 0.95] |

## Limitations

- **Sample size.** 31 participants; stage 2 was trained on 10 to 11 patients per outer fold. Each misclassified recording moves a sensitivity by several points, and the intervals are wide.
- **The curators' caution.** The dataset's README warns against presenting machine-learning classification of these patients against controls as a diagnostic result, because 15 patients are too few for reliable test statistics.
- **No external test set.** Every number comes from cross-validation within one dataset from one site and one EEG system.
- **Medication and everything else.** Stage 2 learns whatever differs between a patient's two sessions. Anything other than the medication that differs systematically between OFF and ON sessions would be learnt as well.
- **The null.** Shuffling patient and control among participants also moves the number of recordings: under the null a control can become a 'patient' with one recording, whose OFF or ON label is a coin flip. The p-values are about features carrying no information on the labels, not about this design alone.
- **Calibration on few recordings.** The reliability bins hold about 8 (stage 1) and 5 (stage 2) recordings each, and the intercept and slope have wide intervals.
- **Fixed predictions in the bootstrap.** The intervals are about which participants were sampled, not about how the models would change with other training data; the other splits show part of that.
