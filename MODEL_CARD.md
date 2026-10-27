# Model card: the pdeeg two-stage research tool

> **Research demo, not a diagnostic tool.** Trained on 31 people from one dataset
> (OpenNeuro ds002778) and not validated on any other recording. These models must not be used to
> decide whether a person has Parkinson's disease, or whether a patient is on medication.

This card covers the three models that `pdeeg train-final` fits and that `pdeeg predict` and the
Streamlit app use. Every number in it is read from the reports in `reports/`, and every setting
from `configs/`. Each saved model also carries its own model card, in its `metadata.json`.

## Model details

- **What.** Three binary classifiers on resting-state EEG features. Each is a scikit-learn pipeline
  of standardisation, the classifier and Platt scaling:

| Stage | Question | Task | Features | Model |
|---|---|---|---|---|
| Stage 1 | Is this a patient's recording or a control's? | `pd_any_vs_hc` | band power + MFDFA (130 features) | linear SVM |
| Stage 2, one recording | OFF or ON medication? | `off_vs_on_single` | band power (25 features) | logistic regression |
| Stage 2, paired | Which of a patient's two recordings is OFF? | `off_vs_on` | band power (25 features) | logistic regression |

- **How they are used.** Stage 1 runs on every recording. Stage 2 (one recording) runs only when
  stage 1 calls a patient's recording outside its uncertain band. The paired model runs when two
  recordings of the same patient are given.
- **How the configuration was chosen.** For each stage, the feature set and model with the best
  cross-validated balanced accuracy of its task among 9 combinations, ties broken by ROC AUC. The
  rule was written before any classifier was run. The scores below are optimistic for that
  reason.
- **Training.** Each model was tuned by a grouped inner loop on all of its task's participants,
  refitted on all of them, and given Platt scaling fitted on grouped inner folds.
- **Software.** `pdeeg` 0.1.0, scikit-learn 1.9.1 (pinned in `pyproject.toml`). A model refuses
  to load under another scikit-learn version, or with feature settings other than those it was
  trained with.
- **Licence.** This repository has no licence file. The data are CC0. Acknowledgements are in the
  [README](README.md#acknowledgements-and-references).

## Intended use

- **For:** research and teaching. The tool shows what band power and MFDFA features of resting-state
  EEG can and cannot tell apart in this dataset, with each answer next to how well that stage did
  in cross-validation.
- **Users:** researchers and students who can read a cross-validated estimate with its interval.
- **Not for:** diagnosis, screening, monitoring a patient's medication, or any decision about a
  person's care. It is also not for recordings from other sites or EEG systems, which no model
  here was tested on. And it is not for presenting the classification of these patients as a
  finding, which the dataset's curators ask people not to do.

How to run it, and what an input must be, is in the
[README](README.md#using-the-two-stage-tool). In short, the input is a BDF or EDF file with all
32 biosemi32 channels and at least 180 s of EEG, at 512 Hz (other rates are resampled, with a
note). A file that fails a check is refused before anything else is done with it.

## Training data

OpenNeuro ds002778 version 1.0.5 (doi:10.18112/openneuro.ds002778.v1.0.5), recorded at one site
with one 32-channel BioSemi system at 512 Hz. The ON sessions of `sub-pd6` and `sub-pd16` were
exported already preprocessed, and are left out of every model.

| Stage | Participants | Recordings | Notes |
|---|---|---|---|
| Stage 1 | 31 | 44 | HC 16, PD_OFF 15, PD_ON 13; both of a patient's recordings at half weight, so each participant counts once |
| Stage 2, one recording | 13 | 26 | One OFF and one ON recording per patient, each classified on its own; `sub-pd6` and `sub-pd16` left out |
| Stage 2, paired | 13 | 26 | The same patients' pairs, each centred on the patient's own mean; `sub-pd6` and `sub-pd16` left out |

Age and sex of the patients and controls (from `reports/stats_results.md`): PD OFF: age 63.3 ± 8.2,
7 m / 8 f; HC: age 63.5 ± 9.7, 7 m / 9 f.

## Evaluation

There is no external test set. Every number below comes from cross-validation within ds002778
([`reports/methods.md`](reports/methods.md)):

- **Splits.** Nested cross-validation, every split grouped by participant: an outer stratified
  5-fold loop, and an inner grouped 3-fold grid search. A guard stops any split that puts one
  participant on both sides.
- **Intervals.** 95 % bootstrap intervals over participants (2000 resamples).
- **p-values.** 1000 relabellings of participants, each through the whole nested procedure.
- **q-values.** Benjamini-Hochberg over every evaluation of a task: 12 p-values for the tasks
  between people (9 combinations and the 3 age-and-sex baselines), 9 for the tasks within
  patients.
- **Other splits.** The spread over 10 other outer splits.
- **Metrics.** Balanced accuracy, the main one, with chance at 0.5. Also ROC AUC, sensitivity
  (the patients, or the OFF recordings, that were found), specificity, and, in the cascade, Brier
  scores and calibration.

### The final configurations in cross-validation

These are the scores of the selected configurations in the classification ablation
([`reports/classification_results.md`](reports/classification_results.md)). They are optimistic,
because each was picked as the best of 9 on these same predictions.

| Stage | Configuration | Balanced accuracy [95 % CI] | ROC AUC [95 % CI] | Sensitivity [95 % CI] | Specificity [95 % CI] | p | q | BA over 10 other splits |
|---|---|---|---|---|---|---|---|---|
| Stage 1 | band power + MFDFA, linear SVM | 0.71 [0.55, 0.85] | 0.79 [0.61, 0.93] | 0.67 [0.46, 0.87] | 0.75 [0.50, 0.94] | 0.020 | 0.176 | 0.55-0.72 |
| Stage 2, one recording | band power, logistic regression | 0.77 [0.65, 0.92] | 0.76 [0.62, 0.93] | 0.77 [0.54, 1.00] | 0.77 [0.54, 1.00] | < 0.001 | 0.009 | 0.58-0.77 |
| Stage 2, paired | band power, logistic regression | held back for the audit (above 0.85) | - | - | - | - | - | - |

Stage 2, paired: its configuration scored a balanced accuracy above 0.85. By a rule fixed in
advance, such a result is held back until a leakage and confound audit is written in
`reports/classification_results.md`, and that audit has not been written. Its numbers are therefore
left out here.

### Better than chance, and when a stage abstains

A stage counts as better than chance when its q-value is below 0.05 and the whole 95 % interval
of its balanced accuracy is above 0.5. This rule is in `configs/predict.yaml`. It was fixed
after the results of the two pre-registered tasks had been seen, and before any result of the two
added tasks. A stage gives no label to a probability inside its uncertain band. The band is worked
out from the stage's out-of-fold probabilities by a rule fixed before any classifier was run
([`reports/methods.md`](reports/methods.md#11-the-final-models)).

| Stage | Better than chance? | Why | Threshold | Uncertain band |
|---|---|---|---|---|
| Stage 1 | **no**: shown greyed out | q = 0.176 is not below 0.05 | 0.5 | no label from 0.305 to 0.68 |
| Stage 2, one recording | yes | q = 0.009 is below 0.05, and [0.65, 0.92] is above 0.5 | 0.5 | 0 to 1: never gives a label |
| Stage 2, paired | **no**: shown greyed out | q is not below 0.05; its numbers are held back for the audit | 0.5 | none (labels every input; 0.5 itself is uncertain) |

### The configuration fixed in advance

The cascade's configuration was fixed before any result: band power and MFDFA with logistic
regression. Here it is on all four tasks
([`reports/cascade_results.md`](reports/cascade_results.md)).
Nothing was picked on these numbers.

| Task | Name | Participants | Recordings | Balanced accuracy [95 % CI] | p | q | ROC AUC [95 % CI] |
|---|---|---|---|---|---|---|---|
| PD OFF / PD ON vs HC | `pd_any_vs_hc` | 31 | 44 | 0.68 [0.52, 0.83] | 0.044 | 0.176 | 0.74 [0.55, 0.90] |
| PD OFF vs HC | `pd_vs_hc` | 31 | 31 | 0.61 [0.43, 0.78] | 0.123 | 0.295 | 0.68 [0.48, 0.86] |
| PD OFF vs PD ON, single recordings | `off_vs_on_single` | 13 | 26 | 0.62 [0.46, 0.77] | 0.148 | 0.266 | 0.64 [0.42, 0.83] |
| PD OFF vs PD ON, within patients | `off_vs_on` | 13 | 26 | 0.62 [0.38, 0.85] | 0.234 | 0.263 | 0.66 [0.35, 0.95] |

### The two stages together: the cascade

In every outer fold, both stages were fitted on the training participants. Every held-out
recording was then called HC, PD_OFF or PD_ON, with the configuration fixed in advance. Rows are
the true class:

| True class \ called | HC | PD_OFF | PD_ON | Sensitivity [95 % CI] |
|---|---|---|---|---|
| HC | 9 | 3 | 4 | 0.56 [0.31, 0.80] |
| PD_OFF | 6 | 7 | 2 | 0.47 [0.21, 0.74] |
| PD_ON | 5 | 3 | 5 | 0.38 [0.12, 0.67] |

Macro balanced accuracy was 0.47 [0.32, 0.61] (chance 0.33,
p = 0.035). The interval includes chance, so the cascade is not shown to beat chance.

| Stage | Balanced accuracy [95 % CI] | p | ROC AUC [95 % CI] | p (AUC) | Null 95th percentile (BA) |
|---|---|---|---|---|---|
| Stage 1: PD OFF / PD ON vs HC | 0.58 [0.40, 0.74] | 0.163 | 0.57 [0.37, 0.76] | 0.226 | 0.66 |
| Stage 2: PD OFF vs PD ON, single recordings (true patients) | 0.71 [0.58, 0.84] | 0.031 | 0.72 [0.52, 0.89] | 0.050 | 0.68 |

**Calibration.** Out-of-fold probabilities, raw and Platt-scaled, were checked against a rule
fixed before any classifier was run: a stage is poorly calibrated if its Brier score is not below
that of predicting the prevalence, or the interval of its calibration slope leaves out 1, or that of
calibration in the large leaves out 0. By that rule, stage 1's raw probabilities were poorly
calibrated (Brier score not below that of predicting the prevalence) and its Platt-scaled ones well
calibrated. Stage 2's raw probabilities were well calibrated and its Platt-scaled ones poorly
calibrated (Brier score not below that of predicting the prevalence). The
three-class Brier score was 0.695 (raw) and
0.687 (Platt-scaled), against 0.664 for predicting the class shares, so
the cascade's three-class probabilities are no better than the class shares.

| Stage | Probabilities | Brier | Brier of the prevalence | Brier skill | Calibration in the large | Calibration slope | Rule |
|---|---|---|---|---|---|---|---|
| Stage 1 | raw | 0.277 [0.203, 0.355] | 0.250 | -0.111 [-0.542, 0.158] | -0.207 [-1.184, 0.709] | 0.125 [-0.527, 1.032] | Brier score not below that of predicting the prevalence |
| Stage 1 | Platt | 0.240 [0.191, 0.292] | 0.250 | 0.040 [-0.272, 0.218] | -0.079 [-0.902, 0.720] | 0.683 [-0.424, 1.849] | well calibrated |
| Stage 2 | raw | 0.211 [0.156, 0.267] | 0.249 | 0.152 [-0.076, 0.370] | 0.007 [-0.319, 0.349] | 0.972 [0.048, 3.235] | well calibrated |
| Stage 2 | Platt | 0.260 [0.229, 0.288] | 0.249 | -0.044 [-0.162, 0.078] | -0.096 [-0.508, 0.335] | 0.256 [-0.420, 1.196] | Brier score not below that of predicting the prevalence |

## Pre-registered and added

| Task or analysis | Status |
|---|---|
| Group statistics (`configs/stats.yaml`) | Pre-registered on 2026-10-23; the primary analysis |
| `pd_vs_hc`, `off_vs_on` | Pre-registered on 2026-10-23; `off_vs_on` is the paired stage |
| `pd_any_vs_hc` (stage 1), `off_vs_on_single` (stage 2), the cascade | Added on 2026-10-26 for the tool, after the pre-registration and before any classifier was run |
| Final configurations, uncertain bands | Filled in from rules after the classification and the cascade had run |
| The "better than chance" rule | Fixed after the results of `pd_vs_hc` and `off_vs_on` had been seen |

The pre-registered group statistics found no group difference that survived correction (smallest
q = 0.471; [`reports/stats_results.md`](reports/stats_results.md)).

## Ethical considerations

- **Misleading people.** The dataset's curators "strongly advise against using any such approach
  because it would mislead patients and people who are interested in knowing if they have
  Parkinson's disease". A probability from this tool says nothing reliable about a person. Every
  output carries the disclaimer and shows the stage's cross-validated performance next to it. The
  app says in words when a stage is not better than chance, and its research view greys that
  stage out.
- **Health data.** EEG is health data. The app is served on the local machine only, and handles
  uploads in a temporary directory that is deleted afterwards; the cleaned data stay in memory
  while the app runs. `pdeeg predict` writes the cleaned data and work files under its output
  directory (`predictions/` by default), which git ignores.
- **Training recordings.** A recording from the training data gets its out-of-fold prediction, made
  by models that never saw that participant, rather than a prediction from models trained on it.
- **Subgroups.** The sample is too small to evaluate any subgroup, by age, sex, disease stage or
  medication. Performance for any group of people is unknown.

## Caveats and recommendations

- **31 participants, one site, one EEG system.** There are 15 patients and 16 controls.
  Every interval is wide, and a single misclassified person moves balanced accuracy by several
  points.
- **No external test set.** Nothing here has been checked on people or equipment from anywhere
  else. Recordings at another sampling rate are resampled, which the models never saw.
- **The curators' warning.** With a proper split there would be "fewer than 8 patients in the
  testing set". The classification of these patients against controls is not a finding.
- **Single-recording OFF against ON.** The best single-recording combination scored 0.77 [0.65,
  0.92] (q 0.009), but it was picked as the best of 9. The configuration fixed in advance scored
  0.62 [0.46, 0.77] (p 0.148), and stage 2 inside the cascade scored 0.71 [0.58, 0.84] (p 0.031).
  Over 10 other outer splits the best combination ranged from 0.58 to 0.77. By the band rule, none
  of its out-of-fold probabilities backed a label, so the tool never labels a single recording OFF
  or ON: it always says uncertain.
- **Training on ON-medication data.** Stage 1 learned what a patient's recording looks like from
  OFF and ON sessions together (15 and 13 recordings). The ON sessions carry the
  effects of these patients' medication, at doses and times not modelled here, so stage 1 may
  partly have learned the medication rather than the disease. Stage 2's "ON" means "as these
  patients were in their ON session". Anything else that differed between the two sessions would
  have been learned as well.
- **Platt scaling for every stage.** The final models use Platt-scaled probabilities for every
  stage. In the cascade's evaluation, with the configuration fixed in advance, the rule found the
  Platt-scaled probabilities well calibrated for stage 1, and poorly calibrated for stage 2. A
  probability from a stage flagged there should be read as a ranking more than as a frequency.
- **Selection.** Each configuration is the best of 9 on the same cross-validation, so its score is
  optimistic. The q-value and the configuration fixed in advance give the fairer picture.
- **Not diagnostic.** Nothing here is validated for any clinical use.
